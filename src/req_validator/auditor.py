"""Трёхпроходный LLM-аудит + независимый Критик.

Pass 0  Global Outline & Entity Registry (deep)     — реестр терминов, ролей, статусов, чисел, ссылок.
Pass 1  Per-Requirement Audit, п. 5.2.5 (fast→deep) — батчи по 4–5 требований.
Pass 2  Holistic & Cross-Section, п. 5.2.6 / 4.3    — противоречия (2 цитаты), пропуски, TBD, проектные решения.
Pass 3  Grounding Gate (verify_citation + 1 целевой ремонт цитаты) → Skeptical Critic для FATAL/MAJOR
        в чистом контексте и, по возможности, на другой модели.

Всё, что уходит наружу, маскируется (pii.PiiMasker) и оборачивается в <untrusted_document>.
Каждый структурированный шаг кэшируется в SQLite; ответы валидируются Pydantic с ретраем по ошибке.
Сбой прохода не роняет прогон: проход помечается failed/partial, отчёт строится из того, что есть.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .cache import DiskStepCache, canonical_json, sha256_hex
from .config import Settings
from .grounding import normalize_with_map, verify_citation
from .guards import GuardError, TraceLogger, estimate_tokens
from .llm_schemas import (
    CriticOutput,
    GlobalRegistry,
    Pass1Output,
    Pass2Output,
    QuoteRepair,
    api_schema,
    extract_json_text,
    schema_hash,
)
from .models import (
    Defect,
    DefectCategory,
    FindingSource,
    Hint,
    PassStatus,
    Requirement,
    ReviewStatus,
    Severity,
)
from .pii import PiiMasker
from .prompts import CRITIC, PASS0_REGISTRY, PASS1_AUDIT, PASS2_HOLISTIC, QUOTE_REPAIR, PromptTemplate, spotlight
from .router import AllDeploymentsFailedError, ChatRequest, LLMRouter
from .standards import FINTECH_CHECKLIST, SET_CRITERIA, SINGLE_CRITERIA, catalog_text, criterion_of, norm_text

T = TypeVar("T", bound=BaseModel)
PASS0_CHUNK_TOKENS = 60_000
RISK_BY_SEVERITY = {
    Severity.FATAL: "Риск остановки разработки, финансовых или регуляторных потерь.",
    Severity.MAJOR: "Риск разночтений между бизнесом, разработкой и QA; переделки на тестировании.",
    Severity.MINOR: "Усложняет сопровождение требований и тест-дизайн.",
}
_LLM_ERRORS = (AllDeploymentsFailedError, GuardError)


class StructuredCallError(RuntimeError):
    pass


@dataclass
class AuditContext:
    settings: Settings
    router: LLMRouter
    cache: DiskStepCache
    tracer: TraceLogger | None = None
    masker: PiiMasker = field(default_factory=PiiMasker)
    llm_calls: int = 0
    cache_hits: int = 0

    def trace(self, event: str, **data: Any) -> None:
        if self.tracer is not None:
            self.tracer.event(event, **data)


@dataclass
class LLMAuditOutcome:
    defects: list[Defect]
    rejected: list[tuple[str, str]]  # (описание находки, причина)
    passes: list[PassStatus]
    checked_req_ids: set[str]
    set_checked: bool
    registry: GlobalRegistry
    critic_stats: dict[str, int]


# --------------------------------------------------------------------------- #
# Структурированный вызов: кэш → LLM → JSON → Pydantic (ретрай с текстом ошибки)
# --------------------------------------------------------------------------- #


async def structured_call(
    ctx: AuditContext,
    *,
    step: str,
    group: str,
    prompt: PromptTemplate,
    values: dict[str, str],
    schema: type[T],
    canonical_input: Any,
    prefer_not: set[str] | None = None,
) -> tuple[T, str, bool]:
    """Вернуть (валидный объект, модель, из_кэша). Ключ кэша = sha256(step + input + prompt_hash + schema_hash)."""
    s_hash = schema_hash(schema)
    key = DiskStepCache.make_key(step.split(":")[0], canonical_json(canonical_input), prompt.hash, s_hash)
    if ctx.settings.use_cache:
        hit = ctx.cache.get(key)
        if hit is not None:
            try:
                obj = schema.model_validate(hit.payload)
                ctx.cache_hits += 1
                ctx.trace("cache_hit", step=step, model=hit.model, key=key[:16])
                return obj, hit.model, True
            except ValidationError:
                ctx.trace("cache_stale", step=step, key=key[:16])

    system, user = prompt.render(**values)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_error = ""
    for attempt in range(1 + ctx.settings.max_validation_retries):
        req = ChatRequest(messages=list(messages), json_schema=api_schema(schema), schema_name=schema.__name__,
                          max_tokens=ctx.settings.max_output_tokens, step=step)
        resp = await ctx.router.complete(group, req, prefer_not=prefer_not)
        ctx.llm_calls += 1
        try:
            obj = schema.model_validate(json.loads(extract_json_text(resp.text)))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)[:1500]
            ctx.trace("validation_error", step=step, attempt=attempt, model=resp.model, error=last_error[:300])
            messages += [
                {"role": "assistant", "content": resp.text[:6000]},
                {"role": "user", "content": "Ответ не прошёл валидацию JSON-схемы:\n" + last_error
                 + "\nВерни исправленный JSON-объект строго по схеме, без пояснений."},
            ]
            continue
        ctx.cache.put(key, step_name=step, prompt_hash=prompt.hash, schema_hash=s_hash, model=resp.model,
                      payload=obj.model_dump(mode="json"), prompt_tokens=resp.prompt_tokens,
                      completion_tokens=resp.completion_tokens)
        return obj, resp.model, False
    raise StructuredCallError(f"{step}: ответ модели не прошёл валидацию после ретраев: {last_error[:200]}")


# --------------------------------------------------------------------------- #
# Подготовка входа
# --------------------------------------------------------------------------- #


def render_requirement(r: Requirement, masker: PiiMasker) -> str:
    head = f"[{r.req_id}]" + (f" (ID в документе: {r.source_id})" if r.source_id else "")
    meta = f"Раздел: {r.heading_path or '—'}"
    if r.context:
        meta += f" | Контекст: {r.context}"
    return f"{head} {masker.mask(meta)}\n{masker.mask(r.text)}"


def render_requirements(reqs: list[Requirement], masker: PiiMasker) -> str:
    return "\n\n".join(render_requirement(r, masker) for r in reqs)


def _known_for(req_ids: set[str] | None, rule_defects: list[Defect], hints: list[Hint], masker: PiiMasker) -> str:
    lines = [f"- {d.req_id or 'документ'}: {d.rule_id} — {masker.mask(d.exact_quote[:120])}"
             for d in rule_defects if req_ids is None or d.req_id in req_ids]
    lines += [f"- {h.req_id}: сигнал {h.rule_id} — {h.message}" for h in hints if req_ids is None or h.req_id in req_ids]
    return "\n".join(lines[:60]) or "—"


def _chunks(reqs: list[Requirement], size: int) -> list[list[Requirement]]:
    return [reqs[i:i + size] for i in range(0, len(reqs), size)]


def _token_chunks(reqs: list[Requirement], limit: int) -> list[list[Requirement]]:
    out: list[list[Requirement]] = [[]]
    used = 0
    for r in reqs:
        t = estimate_tokens(r.text) + 30
        if out[-1] and used + t > limit:
            out.append([])
            used = 0
        out[-1].append(r)
        used += t
    return [c for c in out if c]


# --------------------------------------------------------------------------- #
# Pass 0
# --------------------------------------------------------------------------- #


async def pass0_registry(ctx: AuditContext, title: str, reqs: list[Requirement], raw: str) -> tuple[GlobalRegistry, PassStatus]:
    registry = GlobalRegistry()
    calls0, hits0 = ctx.llm_calls, ctx.cache_hits
    failed: list[str] = []
    chunks = _token_chunks(reqs, PASS0_CHUNK_TOKENS)
    for i, chunk in enumerate(chunks):
        doc = render_requirements(chunk, ctx.masker)
        try:
            part, _, _ = await structured_call(
                ctx, step=f"pass0:{i}", group="deep", prompt=PASS0_REGISTRY,
                values={"title": ctx.masker.mask(title), "document": spotlight(doc)},
                schema=GlobalRegistry, canonical_input={"title": title, "doc": doc})
        except (StructuredCallError, *_LLM_ERRORS) as exc:
            failed.append(str(exc)[:200])
            ctx.trace("pass_failed", name="pass0", chunk=i, error=str(exc)[:300])
            continue
        registry = registry.merge(GlobalRegistry.model_validate(ctx.masker.unmask_obj(part.model_dump())))

    # Grounding реестра: числовой факт без подтверждённой цитаты не используется.
    by_id = {r.req_id: r for r in reqs}
    kept = []
    for fact in registry.numeric_facts:
        req = by_id.get(fact.req_id)
        span = (req.start_char, req.end_char) if req else None
        check = verify_citation(fact.quote, raw, ctx.settings.citation_threshold, handle_span=span)
        if check.verified and check.handle_ok is not False:
            kept.append(fact.model_copy(update={"quote": check.matched_text}))
    dropped = len(registry.numeric_facts) - len(kept)
    registry = registry.model_copy(update={"numeric_facts": kept})
    status = "failed" if len(failed) == len(chunks) else ("partial" if failed else "ok")
    detail = f"терминов {len(registry.terms)}, ролей {len(registry.roles)}, числовых фактов {len(kept)}"
    if dropped:
        detail += f" (отброшено без подтверждённой цитаты: {dropped})"
    if failed:
        detail += "; ошибки: " + "; ".join(failed[:2])
    return registry, PassStatus(name="Pass 0 — Global Registry", status=status, detail=detail,
                                llm_calls=ctx.llm_calls - calls0, cache_hits=ctx.cache_hits - hits0, items=len(chunks))


# --------------------------------------------------------------------------- #
# Pass 1
# --------------------------------------------------------------------------- #


def _to_defect(finding: Any, *, rule_suffix: str, model: str, pass_name: str, by_id: dict[str, Requirement]) -> Defect:
    crit = criterion_of(finding.rule_id)
    req = by_id.get(finding.req_id) if finding.req_id else None
    sev = Severity(finding.severity)
    return Defect(
        defect_id="L-000", severity=sev, rule_id=f"{finding.rule_id}/{rule_suffix}",
        criterion=crit.name_ru if crit else finding.rule_id, category=DefectCategory(finding.category),
        req_id=finding.req_id if req else None, section_path=req.heading_path if req else "",
        exact_quote=finding.verbatim_quote, explanation=finding.explanation, suggestion=finding.suggestion,
        risk=RISK_BY_SEVERITY[sev], source=FindingSource.LLM, review_status=ReviewStatus.NOT_REVIEWED,
        model=model, pass_name=pass_name,
        related_req_ids=[finding.related_req_id] if getattr(finding, "related_req_id", None) else [],
        related_quote=getattr(finding, "related_quote", None),
    )


async def pass1_items(ctx: AuditContext, reqs: list[Requirement], registry: GlobalRegistry, rule_defects: list[Defect],
                      hints: list[Hint]) -> tuple[list[Defect], set[str], PassStatus]:
    by_id = {r.req_id: r for r in reqs}
    sem = asyncio.Semaphore(ctx.settings.llm_concurrency)
    calls0, hits0 = ctx.llm_calls, ctx.cache_hits
    criteria = catalog_text(SINGLE_CRITERIA)

    async def run_batch(batch: list[Requirement]) -> tuple[list[Defect], set[str], str | None]:
        ids = {r.req_id for r in batch}
        doc = render_requirements(batch, ctx.masker)
        reg = canonical_json(registry.compact(ids))
        known = _known_for(ids, rule_defects, hints, ctx.masker)
        step = f"pass1:{batch[0].req_id}-{batch[-1].req_id}"
        async with sem:
            try:
                out, model, _ = await structured_call(
                    ctx, step=step, group=ctx.settings.pass1_group, prompt=PASS1_AUDIT,
                    values={"criteria": criteria, "registry": ctx.masker.mask(reg), "known": known,
                            "requirements": spotlight(doc)},
                    schema=Pass1Output, canonical_input={"doc": doc, "reg": reg, "known": known})
            except (StructuredCallError, *_LLM_ERRORS) as exc:
                ctx.trace("pass_failed", name="pass1", batch=step, error=str(exc)[:300])
                return [], set(), f"{step}: {str(exc)[:160]}"
        out = Pass1Output.model_validate(ctx.masker.unmask_obj(out.model_dump()))
        defects = []
        for f in out.findings:
            if f.req_id not in ids:
                ctx.trace("finding_outside_batch", step=step, req_id=f.req_id)
                continue
            defects.append(_to_defect(f, rule_suffix="LLM", model=model, pass_name="pass1", by_id=by_id))
        return defects, ids, None

    results = await asyncio.gather(*(run_batch(b) for b in _chunks(reqs, ctx.settings.pass1_batch_size)))
    defects = [d for ds, _, _ in results for d in ds]
    checked = set().union(*(ids for _, ids, _ in results)) if results else set()
    errors = [e for _, _, e in results if e]
    status = "failed" if results and len(errors) == len(results) else ("partial" if errors else "ok")
    detail = f"батчей {len(results)}, проверено требований {len(checked)} из {len(reqs)}, находок {len(defects)}"
    if errors:
        detail += f"; неуспешных батчей {len(errors)}: " + "; ".join(errors[:2])
    return defects, checked, PassStatus(name="Pass 1 — проверка требований (п. 5.2.5)", status=status, detail=detail,
                                        llm_calls=ctx.llm_calls - calls0, cache_hits=ctx.cache_hits - hits0,
                                        items=len(results))


# --------------------------------------------------------------------------- #
# Pass 2
# --------------------------------------------------------------------------- #


async def pass2_holistic(ctx: AuditContext, reqs: list[Requirement], registry: GlobalRegistry,
                         known_defects: list[Defect]) -> tuple[list[Defect], bool, PassStatus]:
    by_id = {r.req_id: r for r in reqs}
    calls0, hits0 = ctx.llm_calls, ctx.cache_hits
    doc = render_requirements(reqs, ctx.masker)
    reg = canonical_json(registry.compact())
    known = _known_for(None, known_defects, [], ctx.masker)
    try:
        out, model, _ = await structured_call(
            ctx, step="pass2:all", group="deep", prompt=PASS2_HOLISTIC,
            values={"criteria": catalog_text(SET_CRITERIA), "checklist": FINTECH_CHECKLIST,
                    "registry": ctx.masker.mask(reg), "known": known, "requirements": spotlight(doc)},
            schema=Pass2Output, canonical_input={"doc": doc, "reg": reg, "known": known})
    except (StructuredCallError, *_LLM_ERRORS) as exc:
        ctx.trace("pass_failed", name="pass2", error=str(exc)[:300])
        return [], False, PassStatus(name="Pass 2 — проверка набора (п. 5.2.6, IEEE 830 п. 4.3)", status="failed",
                                     detail=str(exc)[:240], llm_calls=ctx.llm_calls - calls0,
                                     cache_hits=ctx.cache_hits - hits0)
    out = Pass2Output.model_validate(ctx.masker.unmask_obj(out.model_dump()))
    defects = [_to_defect(f, rule_suffix=f.kind, model=model, pass_name="pass2", by_id=by_id) for f in out.findings]
    return defects, True, PassStatus(name="Pass 2 — проверка набора (п. 5.2.6, IEEE 830 п. 4.3)", status="ok",
                                     detail=f"находок {len(defects)}", llm_calls=ctx.llm_calls - calls0,
                                     cache_hits=ctx.cache_hits - hits0, items=1)


# --------------------------------------------------------------------------- #
# Pass 3a — Grounding Gate
# --------------------------------------------------------------------------- #


def _owner(reqs: list[Requirement], start: int, end: int) -> Requirement | None:
    return next((r for r in reqs if r.start_char <= start and end <= r.end_char), None)


async def _verify_quote(ctx: AuditContext, quote: str, req_id: str | None, reqs: list[Requirement],
                        by_id: dict[str, Requirement], raw: str, explanation: str, tag: str,
                        repairs: list[int]) -> tuple[str, int, int, str | None] | str:
    """Проверить цитату в рамках хендла; при провале — 1 целевой ремонт. Вернуть (цитата, s, e, req_id) или причину."""
    req = by_id.get(req_id) if req_id else None
    if req_id and req is None:
        reason = f"неизвестный хендл {req_id}"
    else:
        span = (req.start_char, req.end_char) if req else None
        check = verify_citation(quote, raw, ctx.settings.citation_threshold, handle_span=span)
        if check.verified and check.handle_ok is not False:
            owner = req or _owner(reqs, check.start_char or 0, check.end_char or 0)
            return check.matched_text, check.start_char or 0, check.end_char or 0, owner.req_id if owner else None
        reason = check.reason or "цитата не подтверждена"
    if req is None:
        return reason  # ремонтировать не к чему: нет текста-хендла
    repairs[0] += 1
    try:
        fixed, _, _ = await structured_call(
            ctx, step=f"repair:{tag}", group="fast", prompt=QUOTE_REPAIR,
            values={"explanation": ctx.masker.mask(explanation[:600]), "quote": ctx.masker.mask(quote[:400]),
                    "document": spotlight(render_requirement(req, ctx.masker))},
            schema=QuoteRepair, canonical_input={"req": req.text, "quote": quote, "expl": explanation})
    except (StructuredCallError, *_LLM_ERRORS) as exc:
        return f"{reason}; ремонт цитаты не удался ({str(exc)[:80]})"
    new_quote = ctx.masker.unmask(fixed.verbatim_quote or "")
    if not new_quote:
        return f"{reason}; при ремонте модель отозвала цитату"
    check = verify_citation(new_quote, raw, ctx.settings.citation_threshold, handle_span=(req.start_char, req.end_char))
    if check.verified and check.handle_ok is not False:
        return check.matched_text, check.start_char or 0, check.end_char or 0, req.req_id
    return f"{reason}; повторная цитата тоже не подтверждена ({check.reason})"


async def grounding_gate(ctx: AuditContext, drafts: list[Defect], reqs: list[Requirement],
                         raw: str) -> tuple[list[Defect], list[tuple[str, str]], int]:
    by_id = {r.req_id: r for r in reqs}
    accepted: list[Defect] = []
    rejected: list[tuple[str, str]] = []
    repairs = [0]
    for i, d in enumerate(drafts):
        label = f"{d.pass_name}/{d.rule_id} {d.req_id or 'набор'}: «{d.exact_quote[:60]}»"
        tag = sha256_hex(f"{i}|{d.rule_id}|{d.req_id}|{d.exact_quote}")[:12]
        res = await _verify_quote(ctx, d.exact_quote, d.req_id, reqs, by_id, raw, d.explanation, tag, repairs)
        if isinstance(res, str):
            rejected.append((label, res))
            ctx.trace("grounding_reject", finding=label, reason=res)
            continue
        quote, s, e, owner = res
        update: dict[str, Any] = {"exact_quote": quote, "quote_verified": True, "quote_start": s, "quote_end": e,
                                  "req_id": owner, "section_path": by_id[owner].heading_path if owner else d.section_path}
        if d.related_quote:
            rel_id = d.related_req_ids[0] if d.related_req_ids else None
            rel = await _verify_quote(ctx, d.related_quote, rel_id, reqs, by_id, raw, d.explanation, tag + "r", repairs)
            if isinstance(rel, str):
                rejected.append((label, "вторая цитата противоречия: " + rel))
                ctx.trace("grounding_reject", finding=label, reason="related: " + rel)
                continue
            update.update(related_quote=rel[0], related_quote_verified=True, related_req_ids=[rel[3]] if rel[3] else [])
        accepted.append(d.model_copy(update=update))
    return accepted, rejected, repairs[0]


def _overlap(a: Defect, b: Defect) -> bool:
    if a.quote_start is None or b.quote_start is None:
        return False
    return a.quote_start < (b.quote_end or 0) and b.quote_start < (a.quote_end or 0)


def merge_with_rules(rule_defects: list[Defect], llm_defects: list[Defect]) -> list[Defect]:
    """LLM-находка, совпадающая с правилом (то же требование, критерий, пересечение цитат), отбрасывается;
    дубли между LLM-находками схлопываются с сохранением максимальной критичности."""
    out: list[Defect] = []
    for d in llm_defects:
        base = d.rule_id.split("/")[0]
        if any(r.req_id == d.req_id and r.rule_id.split("/")[0] == base and _overlap(r, d) for r in rule_defects):
            continue
        dup = next((x for x in out if x.req_id == d.req_id and x.rule_id.split("/")[0] == base
                    and (_overlap(x, d) or normalize_with_map(x.exact_quote)[0] == normalize_with_map(d.exact_quote)[0])),
                   None)
        if dup is not None:
            if d.severity.rank < dup.severity.rank:
                out[out.index(dup)] = d
            continue
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# Pass 3b — Independent Skeptical Critic
# --------------------------------------------------------------------------- #


async def critic_review(ctx: AuditContext, defects: list[Defect], reqs: list[Requirement]) -> tuple[list[Defect], dict[str, int], PassStatus]:
    by_id = {r.req_id: r for r in reqs}
    stats = {"reviewed": 0, "confirmed": 0, "downgraded": 0, "rejected": 0, "not_reviewed": 0, "same_model": 0}
    calls0, hits0 = ctx.llm_calls, ctx.cache_hits
    targets = [(f"F{i + 1}", d) for i, d in enumerate(defects)
               if d.source == FindingSource.LLM and d.severity in (Severity.FATAL, Severity.MAJOR)]
    decisions: dict[str, Any] = {}
    critic_models: dict[str, str] = {}
    errors: list[str] = []
    for batch_start in range(0, len(targets), ctx.settings.critic_batch_size):
        batch = targets[batch_start:batch_start + ctx.settings.critic_batch_size]
        items = []
        for item_id, d in batch:
            req = by_id.get(d.req_id or "")
            block = [f"### {item_id}", f"Норма: {norm_text(d.rule_id)}", f"Заявленная критичность: {d.severity.value}",
                     f"Описание дефекта: {d.explanation}", f"Проверенная цитата: «{d.exact_quote}»"]
            if d.related_quote:
                block.append(f"Вторая проверенная цитата ({', '.join(d.related_req_ids) or '—'}): «{d.related_quote}»")
            if req is not None:
                block.append("Текст требования:\n" + spotlight(req.text))
            items.append(ctx.masker.mask("\n".join(block)))
        items_text = "\n\n".join(items)
        gen_models = {d.model for _, d in batch if d.model}
        try:
            out, model, _ = await structured_call(
                ctx, step=f"critic:{sha256_hex(items_text)[:12]}", group="deep", prompt=CRITIC,
                values={"items": items_text}, schema=CriticOutput, canonical_input={"items": items_text},
                prefer_not=gen_models)
        except (StructuredCallError, *_LLM_ERRORS) as exc:
            errors.append(str(exc)[:160])
            ctx.trace("pass_failed", name="critic", error=str(exc)[:300])
            continue
        for dec in out.decisions:
            decisions[dec.item_id] = dec
            critic_models[dec.item_id] = model

    result: list[Defect] = []
    target_ids = {id(d): item_id for item_id, d in targets}
    for d in defects:
        item_id = target_ids.get(id(d))
        if item_id is None:
            result.append(d)
            continue
        dec = decisions.get(item_id)
        if dec is None:
            stats["not_reviewed"] += 1
            result.append(d.model_copy(update={"review_note": "Критик недоступен — находка не подтверждена."}))
            continue
        stats["reviewed"] += 1
        cmodel = critic_models[item_id]
        same = cmodel == d.model
        stats["same_model"] += int(same)
        note = f"Критик ({cmodel}{', та же модель' if same else ''}): {ctx.masker.unmask(dec.reason)}"
        if dec.decision == "REJECTED":
            stats["rejected"] += 1
            ctx.trace("critic_reject", rule=d.rule_id, req=d.req_id, reason=dec.reason[:200])
            continue
        if dec.decision == "DOWNGRADED":
            new = Severity(dec.new_severity) if dec.new_severity else None
            if new is None or new.rank <= d.severity.rank:
                new = Severity.MAJOR if d.severity == Severity.FATAL else Severity.MINOR
            stats["downgraded"] += 1
            result.append(d.model_copy(update={"severity": new, "original_severity": d.severity, "risk": RISK_BY_SEVERITY[new],
                                               "review_status": ReviewStatus.DOWNGRADED, "review_note": note}))
            continue
        stats["confirmed"] += 1
        result.append(d.model_copy(update={"review_status": ReviewStatus.CONFIRMED, "review_note": note}))

    status = "skipped" if not targets else ("failed" if errors and not decisions else ("partial" if errors else "ok"))
    detail = (f"на проверке {len(targets)} (FATAL/MAJOR): подтверждено {stats['confirmed']}, понижено "
              f"{stats['downgraded']}, отклонено {stats['rejected']}, без проверки {stats['not_reviewed']}")
    if stats["same_model"]:
        detail += f"; та же модель, что у генератора: {stats['same_model']}"
    return result, stats, PassStatus(name="Pass 3b — независимый Критик", status=status, detail=detail,
                                     llm_calls=ctx.llm_calls - calls0, cache_hits=ctx.cache_hits - hits0,
                                     items=len(targets))


# --------------------------------------------------------------------------- #
# Оркестрация
# --------------------------------------------------------------------------- #


async def run_llm_audit(ctx: AuditContext, *, title: str, raw: str, reqs: list[Requirement],
                        rule_defects: list[Defect], hints: list[Hint]) -> LLMAuditOutcome:
    passes: list[PassStatus] = []
    registry, st0 = await pass0_registry(ctx, title, reqs, raw)
    passes.append(st0)
    p1, checked, st1 = await pass1_items(ctx, reqs, registry, rule_defects, hints)
    passes.append(st1)
    p2, set_checked, st2 = await pass2_holistic(ctx, reqs, registry, rule_defects + p1)
    passes.append(st2)

    calls0, hits0 = ctx.llm_calls, ctx.cache_hits
    grounded, rejected, repairs = await grounding_gate(ctx, p1 + p2, reqs, raw)
    merged = merge_with_rules(rule_defects, grounded)
    passes.append(PassStatus(
        name="Pass 3a — Grounding Gate", status="ok",
        detail=(f"находок LLM {len(p1) + len(p2)}: подтверждено цитатой {len(grounded)}, отклонено {len(rejected)}, "
                f"ремонтов цитат {repairs}, дублей с правилами снято {len(grounded) - len(merged)}"),
        llm_calls=ctx.llm_calls - calls0, cache_hits=ctx.cache_hits - hits0, items=len(p1) + len(p2)))

    reviewed, critic_stats, st3 = await critic_review(ctx, merged, reqs)
    passes.append(st3)
    return LLMAuditOutcome(defects=reviewed, rejected=rejected, passes=passes, checked_req_ids=checked,
                           set_checked=set_checked, registry=registry, critic_stats=critic_stats)
