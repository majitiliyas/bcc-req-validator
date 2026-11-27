"""Оркестратор пайплайна.

demo:  инжест → нарезка → правила → grounding → метрики → отчёт (без сети и ключей).
llm:   то же + Pass 0 (реестр) → Pass 1 (требования) → Pass 2 (набор) → Pass 3 (grounding gate + Критик).
Любой сбой LLM-слоя деградирует до детерминированного результата с пометками в паспорте.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from datetime import datetime
from pathlib import Path

from . import __version__
from .auditor import AuditContext, LLMAuditOutcome, run_llm_audit
from .cache import DiskStepCache
from .config import Settings, load_settings
from .grounding import deduplicate_by_fragment, deduplicate_defects, ground_defects
from .guards import AgentBudgetGuard, TraceLogger
from .ingestion import ingest
from .models import (
    AuditReport,
    Defect,
    Hint,
    IngestionResult,
    BlockType,
    FunnelStage,
    FindingSource,
    LLMRunInfo,
    PassStatus,
    RunPassport,
    SetAssessment,
)
from .pii import PiiMasker, mask_pii  # noqa: F401 — mask_pii реэкспортируется для совместимости
from .prompts import prompt_versions
from .router import LLMRouter, Transport
from .rules import check_requirement, check_set
from .scoring import (
    assess_requirements,
    assess_set,
    build_remediation,
    compute_metrics,
    compute_roi,
    decide_verdict,
)
from .segmenter import segment

DEMO_MODEL_LABEL = "нет (демо-режим: детерминированные правила, без LLM и сети)"
_REPO_DIR = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=_REPO_DIR, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def git_info() -> tuple[str, str | None]:
    """(commit, tag) репозитория агента. Без коммитов — понятная подпись вместо ошибки."""
    commit = _git("rev-parse", "--short=12", "HEAD")
    if commit is None:
        return "нет коммитов (рабочая копия)", None
    if _git("status", "--porcelain", "--untracked-files=no"):
        commit += "-dirty"
    return commit, _git("describe", "--tags", "--exact-match")


def _sort_and_number(defects: list[Defect], order: dict[str | None, int]) -> list[Defect]:
    ranked = sorted(defects, key=lambda d: (d.severity.rank, order.get(d.req_id, -1), d.quote_start or 0, d.rule_id))
    return [d.model_copy(update={"defect_id": f"DEF-{i:03d}"}) for i, d in enumerate(ranked, start=1)]


def build_router(settings: Settings, guard: AgentBudgetGuard, tracer: TraceLogger,
                 transport: Transport | None = None) -> LLMRouter:
    return LLMRouter.from_settings(settings, guard=guard, tracer=tracer, transport=transport)


def run_audit(input_path: str | Path, *, demo: bool = True, settings: Settings | None = None,
              transport: Transport | None = None, router: LLMRouter | None = None) -> AuditReport:
    """Провести аудит и вернуть валидированный AuditReport. Не бросает исключений на плохом входе и сбоях API."""
    settings = settings or load_settings()
    t0 = time.perf_counter()
    started = datetime.now().astimezone()
    tracer = TraceLogger(settings.trace_dir, f"{started:%Y%m%dT%H%M%S}")
    doc = ingest(input_path, enable_ocr=settings.enable_ocr)
    run_id = f"{started:%Y%m%dT%H%M%S}-{(doc.sha256 or '0' * 8)[:8]}"
    tracer.event("ingest", run_id=run_id, status=doc.status, format=doc.source_format,
                 encoding=doc.detected_encoding, blocks=len(doc.blocks), warnings=doc.warnings)

    commit, tag = git_info()
    notes: list[str] = []

    if not doc.status.can_proceed:
        model = DEMO_MODEL_LABEL if demo else "не вызывалась (документ не прочитан)"
        report = _empty_report(doc, run_id, started, t0, demo, model, commit, tag, notes)
        tracer.event("verdict", status=report.verdict.status, reason="ingestion_failed")
        return report

    seg = segment(doc)
    reqs = seg.requirements
    notes += seg.warnings
    tracer.event("segment", requirements=len(reqs), informative_blocks=len(seg.informative_block_ids))

    drafts: list[Defect] = []
    hints: list[Hint] = []
    for req in reqs:
        d, h = check_requirement(req)
        drafts += d
        hints += h
    set_findings = check_set(doc, reqs)
    drafts += set_findings.defects
    rule_defects, rejected = ground_defects(drafts, reqs, doc.raw_markdown, settings.citation_threshold)
    rule_defects = deduplicate_defects(rule_defects)
    if rejected:
        notes.append(f"Отклонено замечаний правил без подтверждённой цитаты: {len(rejected)}.")
    tracer.event("rules", defects=len(rule_defects), hints=len(hints))

    llm_info: LLMRunInfo | None = None
    outcome: LLMAuditOutcome | None = None
    if not demo and reqs:
        outcome, llm_info = _run_llm(settings, doc, reqs, rule_defects, hints, tracer, transport, router, notes)

    combined = rule_defects + (outcome.defects if outcome else [])
    all_defects = deduplicate_by_fragment(combined)
    if len(all_defects) < len(combined):
        notes.append(f"Снято повторов одного и того же фрагмента: {len(combined) - len(all_defects)}.")
    order = {r.req_id: i for i, r in enumerate(reqs)}
    order[None] = len(reqs)
    defects = _sort_and_number(all_defects, order)

    llm_full = outcome is not None and all(p.status in ("ok", "skipped") for p in outcome.passes)
    preliminary = demo or not llm_full
    metrics = compute_metrics(reqs, defects, len(hints))
    assessments = assess_requirements(reqs, defects, hints, llm_checked_ids=outcome.checked_req_ids if outcome else set())
    set_assessment = assess_set(defects, set_findings.notes, llm_checked=bool(outcome and outcome.set_checked))
    verdict = decide_verdict(metrics, can_proceed=True, preliminary=preliminary, defects=defects)
    tracer.event("verdict", status=verdict.status, gate=verdict.gate, score=metrics.readiness_score,
                 fatal=metrics.fatal_count, major=metrics.major_count, minor=metrics.minor_count)

    if demo:
        model = DEMO_MODEL_LABEL
    elif llm_info and llm_info.models_used:
        model = ", ".join(f"{m} ×{n}" for m, n in sorted(llm_info.models_used.items(), key=lambda kv: -kv[1]))
        extra = [m for m in llm_info.cached_models if m not in llm_info.models_used]
        if extra:
            model += "; из кэша: " + ", ".join(extra)
    elif llm_info and llm_info.cached_models:
        model = ", ".join(llm_info.cached_models) + " (все LLM-шаги взяты из кэша)"
    else:
        model = "LLM недоступна — отчёт построен детерминированным слоем"
    if llm_info is not None and outcome is not None:
        final_llm = sum(1 for d in defects if d.source == FindingSource.LLM)
        llm_info.funnel = [FunnelStage(stage=s, count=c, note=n) for s, c, n in outcome.funnel]
        llm_info.funnel.append(FunnelStage(stage="Итог: находок LLM в отчёте", count=final_llm,
                                           note="после Критика и снятия повторов одного фрагмента"))
    passport = _passport(doc, run_id, started, t0, demo, model, commit, tag, notes)
    passport.llm = llm_info
    tokens = (llm_info.prompt_tokens, llm_info.completion_tokens) if llm_info else None
    roi = compute_roi(metrics, settings, llm_tokens=tokens)
    # Единый источник чисел: «сэкономленные часы» в сводке = базовый сценарий ROI.
    metrics = metrics.model_copy(update={"estimated_saved_hours": roi.base.total_hours})
    return AuditReport(passport=passport, requirements=reqs, assessments=assessments, set_assessment=set_assessment,
                       defects=defects, metrics=metrics, verdict=verdict, remediation_plan=build_remediation(defects),
                       roi=roi)


def _run_llm(settings: Settings, doc: IngestionResult, reqs: list, rule_defects: list[Defect], hints: list[Hint],
             tracer: TraceLogger, transport: Transport | None, router: LLMRouter | None,
             notes: list[str]) -> tuple[LLMAuditOutcome | None, LLMRunInfo]:
    guard = router.guard if router is not None else AgentBudgetGuard(
        max_steps=settings.budget_max_steps, max_tokens=settings.budget_max_tokens,
        max_wall_seconds=settings.budget_max_wall_seconds, max_identical_calls=settings.budget_max_identical_calls,
        circuit_breaker_threshold=settings.circuit_breaker_threshold)
    router = router or build_router(settings, guard, tracer, transport)
    cache = DiskStepCache(settings.cache_db, enabled=settings.use_cache)
    ctx = AuditContext(settings=settings, router=router, cache=cache, tracer=tracer,
                       masker=PiiMasker(use_presidio=settings.use_presidio))
    info = LLMRunInfo(groups={g: [d.name for d in deps] for g, deps in router.groups.items()},
                      prompt_versions=prompt_versions())
    outcome: LLMAuditOutcome | None = None
    try:
        if not router.has_deployments():
            notes.append("LLM-проходы пропущены: нет ни одного доступного провайдера (ключи не заданы).")
            info.passes = [PassStatus(name="LLM-проходы", status="skipped", detail="нет провайдеров")]
            return None, info
        headings = [b for b in doc.blocks if b.block_type == BlockType.HEADING]
        outcome = asyncio.run(run_llm_audit(ctx, title=doc.title or doc.file_name, raw=doc.raw_markdown, reqs=reqs,
                                            rule_defects=rule_defects, hints=hints, headings=headings,
                                            blocks=list(doc.blocks)))
    except Exception as exc:  # noqa: BLE001 — LLM-слой не должен ронять прогон
        tracer.event("llm_pipeline_error", error=f"{type(exc).__name__}: {exc}"[:400])
        notes.append(f"LLM-аудит прерван ({type(exc).__name__}: {str(exc)[:160]}); отчёт построен детерминированным слоем.")
        info.passes.append(PassStatus(name="LLM-аудит", status="failed", detail=str(exc)[:240]))
    finally:
        cache.close()
    stats = router.stats
    info.models_used = dict(stats.models_used)
    info.llm_calls = ctx.llm_calls
    info.cache_hits = ctx.cache_hits
    info.prompt_tokens = stats.prompt_tokens
    info.completion_tokens = stats.completion_tokens
    info.rate_limited = stats.rate_limited
    info.reroutes = stats.reroutes
    info.pii_masked = ctx.masker.stats
    info.budget = router.guard.snapshot()
    if outcome is not None:
        info.passes = outcome.passes
        info.cached_models = sorted({m for p in outcome.passes for m in p.models})
        info.unverified_rejected = len(outcome.rejected)
        info.rejected_examples = [f"{label} — {reason}" for label, reason in outcome.rejected[:10]]
        info.critic = outcome.critic_stats
        info.critic_pairs = outcome.critic_pairs
        info.full_context = bool(ctx.full_context)
        info.registry_size = {"terms": len(outcome.registry.terms), "roles": len(outcome.registry.roles),
                              "state_models": len(outcome.registry.state_models),
                              "numeric_facts": len(outcome.registry.numeric_facts)}
        if ctx.cache_hits and not ctx.llm_calls:
            notes.append("Все LLM-шаги взяты из дискового кэша (0 обращений к API).")
    return outcome, info


def _passport(doc: IngestionResult, run_id: str, started: datetime, t0: float, demo: bool, model: str,
              commit: str, tag: str | None, notes: list[str]) -> RunPassport:
    return RunPassport(
        document_title=doc.title or doc.file_name, file_name=doc.file_name, file_path=doc.file_path,
        file_size_bytes=doc.file_size_bytes, sha256=doc.sha256 or "—", source_format=doc.source_format,
        detected_encoding=doc.detected_encoding or "—", page_count=doc.page_count, run_id=run_id,
        run_started_at=started, duration_s=round(time.perf_counter() - t0, 3), mode="demo" if demo else "llm",
        model=model, agent_version=__version__, git_commit=commit, git_tag=tag, ingestion_status=doc.status,
        ingestion_warnings=doc.warnings, pipeline_notes=notes,
    )


def _empty_report(doc: IngestionResult, run_id: str, started: datetime, t0: float, demo: bool, model: str,
                  commit: str, tag: str | None, notes: list[str]) -> AuditReport:
    metrics = compute_metrics([], [], 0)
    reason = f"Статус чтения документа: {doc.status}. " + " ".join(doc.warnings)
    verdict = decide_verdict(metrics, can_proceed=False, preliminary=demo, reason=reason.strip())
    return AuditReport(
        passport=_passport(doc, run_id, started, t0, demo, model, commit, tag,
                           notes + ["Вызовы LLM и проверки пропущены: документ не прочитан."]),
        requirements=[], assessments=[], set_assessment=SetAssessment(checks=[]), defects=[],
        metrics=metrics, verdict=verdict, remediation_plan=[],
    )
