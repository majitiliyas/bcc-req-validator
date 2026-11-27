"""Рендер отчёта из валидированного AuditReport: Markdown / HTML (Jinja2), PDF (Typst), JSON.

Шаблоны лежат в пакете (src/req_validator/templates), чтобы попадать в wheel и не зависеть от cwd.
Все форматы строятся из одной view-модели `build_view()` — содержимое отчётов идентично.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from .models import (
    AuditReport,
    BlockType,
    CheckStatus,
    Defect,
    FindingSource,
    RequirementKind,
    ReviewStatus,
    Severity,
    VerdictStatus,
)
from .standards import SET_CRITERIA, SINGLE_CRITERIA

TEMPLATES_DIR = Path(__file__).parent / "templates"
FORMATS = ("md", "html", "pdf", "json")

STATUS_ICON = {
    CheckStatus.PASS: "✅", CheckStatus.WARN: "⚠️", CheckStatus.FAIL: "❌",
    CheckStatus.NOT_CHECKED: "◻️", CheckStatus.NOT_APPLICABLE: "—",
}
STATUS_TEXT = {
    CheckStatus.PASS: "соответствует", CheckStatus.WARN: "требует проверки", CheckStatus.FAIL: "нарушение",
    CheckStatus.NOT_CHECKED: "не проверялось", CheckStatus.NOT_APPLICABLE: "неприменимо",
}
STATUS_SHORT = {  # для PDF: шрифты могут не содержать эмодзи
    CheckStatus.PASS: "OK", CheckStatus.WARN: "?", CheckStatus.FAIL: "X",
    CheckStatus.NOT_CHECKED: "·", CheckStatus.NOT_APPLICABLE: "—",
}
VERDICT_TEXT = {
    VerdictStatus.APPROVED: "ГОТОВО К РАЗРАБОТКЕ (APPROVED)",
    VerdictStatus.CONDITIONAL_PASS: "УСЛОВНО ГОТОВО (CONDITIONAL PASS)",
    VerdictStatus.REJECTED: "НЕ ГОТОВО — ВОЗВРАТ АНАЛИТИКУ (REJECTED)",
    VerdictStatus.CANNOT_PROCEED: "ПРОВЕРКА НЕВОЗМОЖНА (CANNOT PROCEED)",
}
VERDICT_ICON = {
    VerdictStatus.APPROVED: "🟢", VerdictStatus.CONDITIONAL_PASS: "🟡",
    VerdictStatus.REJECTED: "🔴", VerdictStatus.CANNOT_PROCEED: "⚫",
}
KIND_TEXT = {
    RequirementKind.FUNCTIONAL: "функциональное", RequirementKind.NON_FUNCTIONAL: "нефункциональное",
    RequirementKind.DATA: "данные", RequirementKind.INTEGRATION: "интеграция", RequirementKind.INTERFACE: "API/интерфейс",
    RequirementKind.ACCEPTANCE: "критерий приёмки", RequirementKind.BUSINESS: "бизнес", RequirementKind.DESIGN: "архитектура/UI",
    RequirementKind.TESTING: "тестирование", RequirementKind.OTHER: "прочее",
}
UNIT_TEXT = {
    BlockType.PARAGRAPH: "абзац", BlockType.LIST_ITEM: "пункт списка", BlockType.TABLE: "строка таблицы",
    BlockType.CODE: "фрагмент кода", BlockType.HEADING: "заголовок",
}
CRITERION_SHORT = {
    "ISO-525-01": "Необх.", "ISO-525-02": "Уместн.", "ISO-525-03": "Однозн.", "ISO-525-04": "Полн.",
    "ISO-525-05": "Атом.", "ISO-525-06": "Реализ.", "ISO-525-07": "Провер.", "ISO-525-08": "Коррект.",
    "ISO-525-09": "Шаблон",
}
DEFECT_TYPE = {
    "ISO-525-07": "непроверяемая", "ISO-526-02": "противоречивая", "ISO-525-04": "неполная", "ISO-526-01": "неполная",
    "ISO-525-03": "неоднозначная", "ISO-525-05": "неатомарная", "ISO-526-06": "нетрассируемая",
    "ISO-526-05": "немодифицируемая", "ISO-526-07": "неранжированная", "ISO-525-09": "не по шаблону",
}


def _section(path: str, depth: int = 2) -> str:
    parts = [p for p in path.split(" > ") if p]
    return " > ".join(parts[-depth:]) if parts else "—"


REVIEW_TEXT = {
    ReviewStatus.DETERMINISTIC: "детерминированное правило",
    ReviewStatus.CONFIRMED: "подтверждено Критиком",
    ReviewStatus.DOWNGRADED: "понижено Критиком",
    ReviewStatus.NOT_REVIEWED: "не проверялось Критиком",
}
GATE_TEXT = {
    "FATAL_GATE": "некомпенсаторный FATAL-гейт", "SCORE": "индекс готовности", "UNREVIEWED_FATAL":
    "неподтверждённый FATAL (только ручная проверка)", "INGESTION": "документ не прочитан", "—": "—", "": "—",
}
PASS_STATUS_TEXT = {"ok": "✅ выполнен", "partial": "⚠️ частично", "failed": "❌ сбой", "skipped": "— пропущен"}


def _source_label(d: Defect) -> str:
    if d.source == FindingSource.RULES:
        return "правила"
    where = {"pass1": "LLM Pass 1", "pass2": "LLM Pass 2"}.get(d.pass_name or "", "LLM")
    return f"{where} ({d.model})" if d.model else where


def _defect_view(d: Defect) -> dict[str, Any]:
    return {
        "source_label": _source_label(d), "review": REVIEW_TEXT[d.review_status], "review_note": d.review_note,
        "binding_note": d.binding_note,
        "related_quote": d.related_quote or "", "related_req": ", ".join(d.related_req_ids) or "",
        "original_severity": d.original_severity.value if d.original_severity else "",
        "id": d.defect_id, "severity": d.severity.value, "priority": d.priority, "rule_id": d.rule_id,
        "criterion": d.criterion, "type": DEFECT_TYPE.get(d.rule_id.split("/")[0], "прочее"),
        "req_id": d.req_id or "весь документ", "section": _section(d.section_path, 3), "quote": d.exact_quote,
        "verified": d.quote_verified, "position": f"{d.quote_start}–{d.quote_end}" if d.quote_start is not None else "—",
        "explanation": d.explanation, "suggestion": d.suggestion, "risk": d.risk, "source": d.source.value,
    }


def build_view(report: AuditReport) -> dict[str, Any]:
    p = report.passport
    m = report.metrics
    v = report.verdict
    defects = [_defect_view(d) for d in report.defects]
    by_id = {d["id"]: d for d in defects}
    assessments = {a.req_id: a for a in report.assessments}

    requirements, matrix, details = [], [], []
    for r in report.requirements:
        a = assessments.get(r.req_id)
        src = r.source_id or "—"
        if r.source_id and not r.has_explicit_id:
            src = f"{r.source_id} (нумерация списка)"
        row = {
            "req_id": r.req_id, "source_id": src, "explicit": r.has_explicit_id, "section": _section(r.heading_path),
            "kind": KIND_TEXT[r.kind], "unit": UNIT_TEXT[r.unit_type], "short": r.short, "quote": r.text,
            "context": r.context or "", "page": r.page_num, "position": f"{r.start_char}–{r.end_char}",
        }
        requirements.append(row)
        if a is None:
            continue
        matrix.append({
            "req_id": r.req_id, "source_id": r.source_id or "—",
            "icons": [STATUS_ICON[c.status] for c in a.checks], "short": [STATUS_SHORT[c.status] for c in a.checks],
            "n_defects": len(a.defect_ids), "n_hints": len(a.hints),
        })
        if a.defect_ids or a.hints:
            details.append({
                **row,
                "defects": [by_id[i] for i in a.defect_ids if i in by_id],
                "hints": [{"rule_id": h.rule_id, "message": h.message, "quote": h.exact_quote} for h in a.hints],
                "checks": [{"criterion": c.criterion, "rule_id": c.rule_id, "icon": STATUS_ICON[c.status],
                            "status": STATUS_TEXT[c.status], "short": STATUS_SHORT[c.status], "comment": c.comment}
                           for c in a.checks if c.status in (CheckStatus.FAIL, CheckStatus.WARN)],
            })

    set_checks = [{
        "rule_id": c.rule_id, "criterion": c.criterion, "icon": STATUS_ICON[c.status], "status": STATUS_TEXT[c.status],
        "short": STATUS_SHORT[c.status], "comment": c.comment, "defect_ids": c.defect_ids,
        "reference": next((x.reference for x in SET_CRITERIA if x.rule_id == c.rule_id), ""),
    } for c in report.set_assessment.checks]

    critical = [d for d in defects if d["severity"] in (Severity.FATAL.value, Severity.MAJOR.value)]
    return {
        "title": p.document_title,
        "passport": {
            "document_title": p.document_title, "file_name": p.file_name, "file_size": f"{p.file_size_bytes:,}".replace(",", " "),
            "sha256": p.sha256, "format": p.source_format, "encoding": p.detected_encoding,
            "pages": p.page_count if p.page_count is not None else "—",
            "run_date": f"{p.run_started_at:%d.%m.%Y %H:%M:%S %Z}".strip(), "run_id": p.run_id,
            "duration": f"{p.duration_s:.2f} с", "mode": "демо (без LLM и сети)" if p.mode == "demo" else "LLM",
            "model": p.model, "agent_version": p.agent_version, "git_commit": p.git_commit, "git_tag": p.git_tag or "—",
            "ingestion_status": p.ingestion_status.value, "warnings": p.ingestion_warnings, "notes": p.pipeline_notes,
        },
        "metrics": m.model_dump(),
        "verdict": {
            "status": v.status.value, "text": VERDICT_TEXT[v.status], "icon": VERDICT_ICON[v.status],
            "rationale": v.rationale, "dev": "да" if v.can_start_development else "нет",
            "test": "да" if v.can_start_testing else "нет", "dev_note": v.development_note, "test_note": v.testing_note,
            "preliminary": v.is_preliminary, "gate": GATE_TEXT.get(v.gate, v.gate),
            "blocking": ", ".join(v.blocking_defect_ids),
        },
        "llm": _llm_view(report),
        "roi": _roi_view(report),
        "criteria": [{"rule_id": c.rule_id, "name": c.name_ru, "short": CRITERION_SHORT[c.rule_id], "reference": c.reference,
                      "definition": c.definition, "coverage": c.rules_coverage or "только LLM (Спринт 2)"} for c in SINGLE_CRITERIA],
        "set_criteria": [{"rule_id": c.rule_id, "name": c.name_ru, "reference": c.reference} for c in SET_CRITERIA],
        "requirements": requirements, "matrix": matrix, "details": details, "set_checks": set_checks,
        "set_defects": [d for d in defects if d["req_id"] == "весь документ"],
        "defects": defects, "critical": critical,
        "remediation": [r.model_dump() for r in report.remediation_plan],
        "top_fixes": [r.model_dump() for r in report.remediation_plan[:5]],
    }


def _llm_view(report: AuditReport) -> dict[str, Any] | None:
    info = report.passport.llm
    if info is None:
        return None
    return {
        "models_used": (", ".join(f"{m} ×{n}" for m, n in info.models_used.items())
                        or (f"в этом прогоне вызовов нет; из кэша: {', '.join(info.cached_models)}"
                            if info.cached_models else "—")),
        "routes": [{"group": g, "chain": " → ".join(ds) if ds else "нет доступных моделей"}
                   for g, ds in info.groups.items()],
        "prompts": ", ".join(f"{k}@{v}" for k, v in info.prompt_versions.items()),
        "passes": [{"name": p.name, "status": PASS_STATUS_TEXT.get(p.status, p.status), "raw_status": p.status,
                    "detail": p.detail, "calls": p.llm_calls, "cache": p.cache_hits,
                    "models": ", ".join(p.models) or "—"} for p in info.passes],
        "calls": info.llm_calls, "cache_hits": info.cache_hits,
        "tokens": f"{info.prompt_tokens} / {info.completion_tokens}",
        "rate_limited": info.rate_limited, "reroutes": info.reroutes,
        "unverified": info.unverified_rejected, "rejected_examples": info.rejected_examples,
        "critic": ", ".join(f"{k}: {v}" for k, v in info.critic.items()) or "—",
        "pii": ", ".join(f"{k}: {v}" for k, v in info.pii_masked.items()) or "не обнаружено",
        "budget": ", ".join(f"{k}: {v}" for k, v in info.budget.items()),
        "registry": ", ".join(f"{k}: {v}" for k, v in info.registry_size.items()) or "—",
        "funnel": _funnel_view(info.funnel),
    }


def _funnel_view(stages: list[Any]) -> list[dict[str, Any]]:
    top = max((st.count for st in stages), default=0) or 1
    return [{"stage": st.stage, "count": st.count, "note": st.note, "pct": round(st.count / top, 3),
             "bar": "█" * max(1 if st.count else 0, round(20 * st.count / top))} for st in stages]


def _num(value: float, digits: int = 0) -> str:
    text = f"{value:,.{digits}f}".replace(",", " ")
    return text


def _roi_view(report: AuditReport) -> dict[str, Any] | None:
    roi = report.roi
    if roi is None:
        return None
    return {
        "lines": [{"item": ln.item, "formula": ln.formula, "hours": _num(ln.hours, 1)} for ln in roi.lines],
        "total_hours": _num(roi.total_hours, 1), "rate": _num(roi.hourly_rate), "currency": roi.currency,
        "money": _num(roi.total_money), "assumptions": roi.assumptions, "llm_cost": roi.llm_cost,
    }


# --------------------------------------------------------------------------- #
# Рендеринг
# --------------------------------------------------------------------------- #


def _md_cell(value: Any) -> str:
    text = " ".join(str(value if value is not None else "—").split())
    if text.count("`") % 2:
        text = text.replace("`", "")
    return text.replace("|", "\\|") or "—"


def _md_finalize(value: Any) -> Any:
    """Экранирование для Markdown: `$` (иначе просмотрщики с KaTeX/MathJax превращают текст между двумя `$`
    — например, curl-команду с $BASE … $TOKEN — в формулу)."""
    if isinstance(value, str):
        return value.replace("\\$", "$").replace("$", "\\$")
    return "" if value is None else value


def _md_quote(value: Any) -> str:
    lines = str(value).strip().split("\n")
    return "\n".join("> " + line.strip() for line in lines)


def _env(autoescape: bool) -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html"]) if autoescape else False,
        undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
        finalize=None if autoescape else _md_finalize,
    )
    env.filters["cell"] = _md_cell
    env.filters["mdquote"] = _md_quote
    return env


def render_markdown(report: AuditReport) -> str:
    return _env(False).get_template("report.md.j2").render(**build_view(report))


def render_html(report: AuditReport) -> str:
    return _env(True).get_template("report.html.j2").render(**build_view(report))


def render_json(report: AuditReport) -> str:
    return report.model_dump_json(indent=2)


def render_pdf(report: AuditReport) -> bytes:
    import typst

    template = (TEMPLATES_DIR / "report.typ").read_bytes()
    data = json.dumps(build_view(report), ensure_ascii=False, default=str)
    result = typst.compile(template, sys_inputs={"report": data})
    if not isinstance(result, bytes):
        raise RuntimeError("typst не вернул PDF")
    return result


def resolve_formats(out_path: Path, fmt: str) -> list[str]:
    fmt = fmt.lower()
    if fmt == "auto":
        ext = out_path.suffix.lower().lstrip(".")
        return [ext if ext in FORMATS else "md"]
    if fmt == "all":
        return list(FORMATS)
    if fmt == "both":
        return ["md", "pdf"]
    chosen = [f.strip() for f in fmt.split(",") if f.strip()]
    unknown = [f for f in chosen if f not in FORMATS]
    if unknown:
        raise ValueError(f"Неизвестный формат: {', '.join(unknown)}. Допустимо: {', '.join(FORMATS)}, all, both.")
    return chosen


def write_reports(report: AuditReport, out_path: str | Path, fmt: str = "auto") -> tuple[list[Path], list[str]]:
    """Записать отчёт в выбранных форматах. Возвращает (записанные файлы, ошибки). Не бросает исключений рендера."""
    out = Path(out_path)
    formats = resolve_formats(out, fmt)
    out.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    errors: list[str] = []
    renderers = {"md": render_markdown, "html": render_html, "json": render_json, "pdf": render_pdf}
    for f in formats:
        target = out if out.suffix.lower() == f".{f}" else out.with_suffix(f".{f}")
        try:
            content = renderers[f](report)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content, encoding="utf-8")
            written.append(target)
        except Exception as exc:  # noqa: BLE001 — сбой одного формата не должен терять остальные
            errors.append(f"Не удалось сформировать {f.upper()}: {type(exc).__name__}: {exc}")
    return written, errors
