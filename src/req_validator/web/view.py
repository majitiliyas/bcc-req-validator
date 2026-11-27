"""Представление отчёта для веб-дашборда: компактный JSON из валидированного AuditReport.

Только преобразование готовых чисел — метрики, вердикт и ROI уже посчитаны детерминированно в ядре.
"""

from __future__ import annotations

from typing import Any

from ..models import AuditReport, VerdictStatus
from ..reporting import GATE_TEXT, VERDICT_TEXT, _defect_view

DOWNLOAD_FORMATS = ("pdf", "md", "html")

_CRITIC_KEYS = ("reviewed", "confirmed", "downgraded", "rejected", "not_reviewed", "same_model")


def defect_card(d: dict[str, Any]) -> dict[str, Any]:
    """Поля дефекта, нужные карточке «Дефекты и правки»."""
    keys = ("id", "severity", "priority", "rule_id", "criterion", "req_id", "related_req", "section", "quote_view",
            "related_view", "explanation", "risk", "suggestion", "source_label", "review", "review_note",
            "original_severity", "source")
    return {k: d.get(k, "") for k in keys}


def report_to_web(report: AuditReport, *, downloads: dict[str, str] | None = None) -> dict[str, Any]:
    p, m, v = report.passport, report.metrics, report.verdict
    info = p.llm
    llm: dict[str, Any] | None = None
    if info is not None:
        llm = {
            "calls": info.llm_calls, "cache_hits": info.cache_hits,
            "tokens_in": info.prompt_tokens, "tokens_out": info.completion_tokens,
            "models_used": [f"{k} ×{n}" for k, n in info.models_used.items()] or list(info.cached_models),
            "critic": {k: int(info.critic.get(k, 0)) for k in _CRITIC_KEYS},
            "funnel": [{"stage": s.stage, "count": s.count, "note": s.note} for s in info.funnel],
            "passes": [{"name": x.name, "status": x.status, "detail": x.detail, "models": x.models} for x in info.passes],
        }
    roi: dict[str, Any] | None = None
    if report.roi is not None:
        roi = {
            "rate": report.roi.hourly_rate, "currency": report.roi.currency, "docs": report.roi.docs_per_month,
            "llm_cost": report.roi.llm_cost, "assumptions": report.roi.assumptions,
            "scenarios": [{"key": s.key, "name": s.name, "total_hours": s.total_hours, "money": s.money,
                           "lines": [{"item": ln.item, "hours": ln.hours, "formula": ln.formula} for ln in s.lines]}
                          for s in report.roi.scenarios],
        }
    return {
        "source": "live",
        "title": p.document_title,
        "passport": {
            "file_name": p.file_name, "file_size_bytes": p.file_size_bytes, "sha256": p.sha256,
            "format": p.source_format, "encoding": p.detected_encoding,
            "run_id": p.run_id, "run_date": f"{p.run_started_at:%d.%m.%Y %H:%M:%S}",
            "duration_s": round(p.duration_s, 2), "mode": p.mode, "model": p.model,
            "agent_version": p.agent_version, "git_commit": p.git_commit, "git_tag": p.git_tag or "",
            "ingestion_status": p.ingestion_status.value, "warnings": p.ingestion_warnings, "notes": p.pipeline_notes,
        },
        "verdict": {
            "status": v.status.value, "text": VERDICT_TEXT[v.status], "rationale": v.rationale,
            "dev": v.can_start_development, "test": v.can_start_testing, "dev_note": v.development_note,
            "test_note": v.testing_note, "preliminary": v.is_preliminary, "gate": GATE_TEXT.get(v.gate, v.gate),
            "blocking": list(v.blocking_defect_ids),
        },
        "metrics": m.model_dump(),
        "defects": [defect_card(_defect_view(d)) for d in report.defects],
        "llm": llm,
        "roi": roi,
        "remediation": [r.model_dump() for r in report.remediation_plan],
        "downloads": downloads or {},
    }


def is_cannot_proceed(view: dict[str, Any]) -> bool:
    return view.get("verdict", {}).get("status") == VerdictStatus.CANNOT_PROCEED.value
