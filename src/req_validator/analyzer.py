"""Оркестратор пайплайна: инжест → нарезка → правила → grounding → метрики → AuditReport.

Спринт 1: полностью детерминированный слой. LLM-проходы (Pass 0 Global Context, Pass 1 Items,
Pass 2 Holistic + Critic) подключаются в Спринте 2 между шагами «правила» и «grounding».
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import Settings, load_settings
from .grounding import deduplicate_defects, ground_defects
from .ingestion import ingest
from .models import (
    AuditReport,
    Defect,
    Hint,
    IngestionResult,
    RunPassport,
    SetAssessment,
)
from .rules import check_requirement, check_set
from .scoring import assess_requirements, assess_set, build_remediation, compute_metrics, decide_verdict
from .segmenter import segment

DEMO_MODEL_LABEL = "нет (демо-режим: детерминированные правила, без LLM и сети)"
_REPO_DIR = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Версия агента и трассировка
# --------------------------------------------------------------------------- #


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


_PII = [
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[PAN]"),
    (re.compile(r"\b\d{12}\b"), "[IIN]"),
    (re.compile(r"\+?\d[\d\s()-]{9,}\d"), "[PHONE]"),
    (re.compile(r"\b(?:AIza[\w-]{20,}|gsk_\w{20,}|sk-[\w-]{20,})"), "[SECRET]"),
]


def mask_pii(text: str) -> str:
    for rx, repl in _PII:
        text = rx.sub(repl, text)
    return text


class Tracer:
    """JSONL-трасса шагов пайплайна в .cache/traces/<run_id>.jsonl. Сбой записи не ломает прогон."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.path = None  # type: ignore[assignment]

    def event(self, step: str, **data: Any) -> None:
        if self.path is None:
            return
        record = {"ts": datetime.now().astimezone().isoformat(timespec="milliseconds"), "step": step, **data}
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(mask_pii(json.dumps(record, ensure_ascii=False, default=str)) + "\n")
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Пайплайн
# --------------------------------------------------------------------------- #


def _sort_and_number(defects: list[Defect], order: dict[str | None, int]) -> list[Defect]:
    ranked = sorted(defects, key=lambda d: (d.severity.rank, order.get(d.req_id, -1), d.quote_start or 0, d.rule_id))
    return [d.model_copy(update={"defect_id": f"DEF-{i:03d}"}) for i, d in enumerate(ranked, start=1)]


def run_audit(input_path: str | Path, *, demo: bool = True, settings: Settings | None = None) -> AuditReport:
    """Провести аудит документа и вернуть валидированный AuditReport. Не бросает исключений на плохом входе."""
    settings = settings or load_settings()
    t0 = time.perf_counter()
    started = datetime.now().astimezone()
    doc = ingest(input_path, enable_ocr=settings.enable_ocr)
    run_id = f"{started:%Y%m%dT%H%M%S}-{(doc.sha256 or '0' * 8)[:8]}"
    tracer = Tracer(settings.cache_dir / "traces" / f"{run_id}.jsonl")
    tracer.event("ingest", status=doc.status, format=doc.source_format, encoding=doc.detected_encoding,
                 blocks=len(doc.blocks), warnings=doc.warnings)

    commit, tag = git_info()
    notes: list[str] = []
    model = DEMO_MODEL_LABEL if demo else settings.primary_model
    if not demo:
        notes.append("LLM-проходы (семантический аудит, Critic) будут подключены в Спринте 2; "
                     "отчёт построен детерминированным слоем.")

    if not doc.status.can_proceed:
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
    tracer.event("rules", defects=len(drafts), hints=len(hints))

    accepted, rejected = ground_defects(drafts, reqs, doc.raw_markdown, settings.citation_threshold)
    accepted = deduplicate_defects(accepted)
    if rejected:
        notes.append(f"Отклонено замечаний без подтверждённой цитаты: {len(rejected)}.")
    tracer.event("grounding", accepted=len(accepted), rejected=[r for _, r in rejected])

    order = {r.req_id: i for i, r in enumerate(reqs)}
    order[None] = len(reqs)
    defects = _sort_and_number(accepted, order)

    metrics = compute_metrics(reqs, defects, len(hints))
    assessments = assess_requirements(reqs, defects, hints, llm_checked=False)
    set_assessment = assess_set(defects, set_findings.notes, llm_checked=False)
    verdict = decide_verdict(metrics, can_proceed=True, preliminary=True)
    tracer.event("verdict", status=verdict.status, score=metrics.readiness_score, fatal=metrics.fatal_count,
                 major=metrics.major_count, minor=metrics.minor_count)

    return AuditReport(
        passport=_passport(doc, run_id, started, t0, demo, model, commit, tag, notes),
        requirements=reqs, assessments=assessments, set_assessment=set_assessment, defects=defects,
        metrics=metrics, verdict=verdict, remediation_plan=build_remediation(defects),
    )


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
