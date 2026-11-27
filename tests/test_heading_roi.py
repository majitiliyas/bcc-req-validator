"""Привязка цитат-заголовков раздела, детерминированный ROI и воронка качества в отчёте."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from req_validator.config import Settings
from req_validator.grounding import HeadingAnchor, heading_anchor
from req_validator.ingestion import ingest
from req_validator.models import BlockType, SummaryMetrics
from req_validator.scoring import compute_roi
from req_validator.segmenter import segment
from test_auditor_pipeline import FakeLLM, audit

DOC = """# API уведомлений

## Ответ

* **API-01.** Если результат 200 или 201 — успешный.

### 200. Successful — Created

```json
{"result":"OK","amount":"50000"}
```

```json
{"result":"ERROR","message":"Проверьте сумму"}
```

* **API-02.** Ответ содержит поле result со значением OK или ERROR.

### Пустой раздел

### Ошибки

* **ERR-01.** Ответ 400 возвращается при неверных параметрах.
"""


@pytest.fixture
def parsed(tmp_path: Path) -> tuple[Any, Any, list[Any]]:
    p = tmp_path / "api.md"
    p.write_text(DOC, encoding="utf-8")
    doc = ingest(p)
    seg = segment(doc)
    heads = [b for b in doc.blocks if b.block_type == BlockType.HEADING]
    return doc, seg, heads


def test_requirements_layout(parsed: Any) -> None:
    _, seg, _ = parsed
    assert [r.source_id for r in seg.requirements] == ["API-01", "API-02", "ERR-01"]  # JSON-примеры — контекст
    assert any("примеров JSON — 2" in w for w in seg.warnings)


def test_exact_heading_binds_to_first_requirement_of_section(parsed: Any) -> None:
    doc, seg, heads = parsed
    a = heading_anchor("200. Successful — Created", heads, seg.requirements, doc.raw_markdown)
    assert isinstance(a, HeadingAnchor)
    assert a.req_id == seg.requirements[1].req_id  # первое требование раздела — API-02
    assert doc.raw_markdown[a.start_char:a.end_char] == "200. Successful — Created"
    assert "через заголовок раздела «200. Successful — Created»" in a.note


@pytest.mark.parametrize("quote", ["#### 200. Successful — Created", "  200.  Successful — Created ", "**200. Successful — Created**"])
def test_heading_markup_and_spaces_ignored(parsed: Any, quote: str) -> None:
    doc, seg, heads = parsed
    assert isinstance(heading_anchor(quote, heads, seg.requirements, doc.raw_markdown), HeadingAnchor)


@pytest.mark.parametrize("quote", ["200. Successful", "200. successful — created", "200. Successful — Createdd",
                                   "Successful — Created", ""])
def test_no_fuzzy_or_partial_heading_matches(parsed: Any, quote: str) -> None:
    doc, seg, heads = parsed
    assert isinstance(heading_anchor(quote, heads, seg.requirements, doc.raw_markdown), str)


def test_heading_without_requirements_and_duplicates_rejected(parsed: Any, tmp_path: Path) -> None:
    doc, seg, heads = parsed
    assert "нет выделенных требований" in heading_anchor("Пустой раздел", heads, seg.requirements, doc.raw_markdown)
    dup = tmp_path / "dup.md"
    dup.write_text("# A\n\n## Ответ\n\n* **R-1.** Система должна x.\n\n## Ответ\n\n* **R-2.** Система должна y.\n",
                   encoding="utf-8")
    d = ingest(dup)
    hs = [b for b in d.blocks if b.block_type == BlockType.HEADING]
    assert "неоднозначна" in heading_anchor("Ответ", hs, segment(d).requirements, d.raw_markdown)


def test_heading_anchor_restores_contradiction_end_to_end(tmp_path: Path) -> None:
    p = tmp_path / "api.md"
    p.write_text(DOC, encoding="utf-8")
    s = Settings(_env_file=None, cache_dir=tmp_path / ".cache", gemini_api_key=SecretStr("g"),
                 nvidia_api_key=SecretStr("n"), groq_api_key=SecretStr("q"), deep_models="gemini/model-a,nvidia/model-b",
                 fast_models="groq/model-fast", enable_ollama=False, use_presidio=False)

    def pass2(req: Any) -> dict[str, Any]:
        return {"findings": [{
            "kind": "CONTRADICTION", "rule_id": "ISO-526-02", "severity": "MAJOR", "category": "SET_CONSISTENCY",
            "req_id": "REQ-001", "verbatim_quote": "Если результат 200 или 201 — успешный",
            "related_req_id": "REQ-002", "related_quote": "200. Successful — Created",
            "explanation": "Раздел успешного ответа 200 содержит пример с result=ERROR.", "suggestion": "Вернуть 400."}]}

    fake = FakeLLM(pass1=lambda r: {"reviewed_req_ids": [], "findings": []}, pass2=pass2, pass0=lambda r: {"terms": []})
    report = audit(s, p, fake)
    d = next(x for x in report.defects if x.rule_id == "ISO-526-02/CONTRADICTION")
    assert d.related_quote == "200. Successful — Created" and d.related_quote_verified
    assert d.related_req_ids == ["REQ-002"]
    assert "через заголовок раздела" in d.binding_note
    funnel = {f.stage: f.count for f in report.passport.llm.funnel}
    assert funnel["Кандидаты LLM (Pass 1 + Pass 2)"] == 1 and funnel["Подтверждено цитатой"] == 1
    assert funnel["Итог: находок LLM в отчёте"] == 1
    assert [f.note for f in report.passport.llm.funnel if f.stage == "Подтверждено цитатой"][0] == "точное совпадение: 1"


# --------------------------------------------------------------------------- #
# ROI
# --------------------------------------------------------------------------- #


def _m(n: int, f: int, ma: int, mi: int) -> SummaryMetrics:
    return SummaryMetrics(total_requirements=n, requirements_with_explicit_id=0, requirements_with_defects=0,
                          fatal_count=f, major_count=ma, minor_count=mi, total_defects=f + ma + mi, hints_count=0,
                          defect_density=0, readiness_score=50, estimated_saved_hours=0)


def test_roi_is_exact_decimal_arithmetic() -> None:
    s = Settings(_env_file=None)
    roi = compute_roi(_m(67, 1, 14, 25), s, llm_tokens=(1000, 200))
    cons, base = roi.scenarios
    assert (cons.key, base.key) == ("conservative", "base")
    # консервативный: 67×1.5/60=1.7; (16+84+0)×0.4=40.0; −40×3/60=2.0 → 39.7
    assert [ln.hours for ln in cons.lines] == [1.7, 40.0, -2.0] and cons.total_hours == 39.7
    # базовый: 67×2/60=2.2; (16+84+25×0.25)×0.5=53.1 (53.125→53.1); −2.0 → 53.3
    assert [ln.hours for ln in base.lines] == [2.2, 53.1, -2.0] and base.total_hours == 53.3
    assert base.money == 533_000.0 and base.monthly_hours == 1599.0 and roi.docs_per_month == 30
    assert roi.base is base and "1000 на вход" in roi.llm_cost
    text = " ".join(roi.assumptions)
    assert "NIST Planning Report 02-3" in text and "IBM Systems Sciences Institute" in text and "справочник" not in text
    assert "40–50 %" in text and "10 000 ₸" in text


def test_roi_never_negative_and_minor_free_in_conservative() -> None:
    s = Settings(_env_file=None)
    roi = compute_roi(_m(2, 0, 0, 40), s)  # только MINOR: консервативно rework = 0, разбор отчёта съедает экономию
    assert roi.scenarios[0].total_hours == 0.0 and roi.scenarios[0].money == 0.0


def test_roi_custom_coefficients() -> None:
    s = Settings(_env_file=None, roi_review_minutes_per_req_base=10, roi_hourly_rate=2.5, roi_rework_hours_major=3,
                 roi_manual_miss_rate_base=1, roi_triage_minutes_per_defect=0, roi_docs_per_month=2)
    base = compute_roi(_m(3, 0, 1, 0), s).base
    assert base.total_hours == 3.5 and base.money == 9.0 and base.monthly_hours == 7.0  # (0.5 + 3) × 2.5 = 8.75 → 9


def test_roi_and_funnel_rendered(tmp_path: Path, fixtures: Path, settings: Settings) -> None:
    from req_validator.analyzer import run_audit
    from req_validator.reporting import write_reports

    report = run_audit(fixtures / "flawed_fintech_srs.md", demo=True, settings=settings)
    written, errors = write_reports(report, tmp_path / "out" / "r.md", "md,pdf,html")
    assert errors == []
    md = (tmp_path / "out" / "r.md").read_text(encoding="utf-8")
    assert "### Бизнес-ценность (ROI)" in md and "**Итого на этот документ**" in md and "ROI_HOURLY_RATE" in md
    hours = report.roi.base.total_hours
    assert f"| Сэкономленные трудозатраты (базовый сценарий ROI) | {hours:.1f} ч |" in md  # одно число в сводке и ROI
    import pymupdf

    assert "Бизнес-ценность" in "".join(p.get_text() for p in pymupdf.open(tmp_path / "out" / "r.pdf"))


def test_funnel_rendered_in_llm_report(tmp_path: Path) -> None:
    from req_validator.reporting import write_reports

    p = tmp_path / "api.md"
    p.write_text(DOC, encoding="utf-8")
    s = Settings(_env_file=None, cache_dir=tmp_path / ".cache", gemini_api_key=SecretStr("g"),
                 nvidia_api_key=SecretStr("n"), groq_api_key=SecretStr("q"), deep_models="gemini/model-a,nvidia/model-b",
                 fast_models="groq/model-fast", enable_ollama=False, use_presidio=False)
    report = audit(s, p, FakeLLM())
    written, errors = write_reports(report, tmp_path / "out" / "r.md", "md,pdf,html")
    assert errors == []
    md = (tmp_path / "out" / "r.md").read_text(encoding="utf-8")
    assert "**Воронка качества и отсева галлюцинаций**" in md and "Итог: находок LLM в отчёте" in md
    html = (tmp_path / "out" / "r.html").read_text(encoding="utf-8")
    assert "Воронка качества" in html
    import pymupdf

    assert "Воронка качества" in "".join(pg.get_text() for pg in pymupdf.open(tmp_path / "out" / "r.pdf"))
