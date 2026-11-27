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
    texts = [r.text[:20] for r in seg.requirements]
    assert texts[0].startswith("**API-01.**") and texts[1].startswith('{"result":"OK"')
    assert texts[2].startswith('{"result":"ERROR"')


def test_exact_heading_binds_to_first_requirement_of_section(parsed: Any) -> None:
    doc, seg, heads = parsed
    a = heading_anchor("200. Successful — Created", heads, seg.requirements, doc.raw_markdown)
    assert isinstance(a, HeadingAnchor)
    assert a.req_id == seg.requirements[1].req_id  # первый блок под заголовком — пример OK
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
    hours = {ln.item.split(": ")[-1] if ":" in ln.item else "review": ln.hours for ln in roi.lines}
    assert hours == {"review": 6.7, "FATAL": 16.0, "MAJOR": 84.0, "MINOR": 25.0}
    assert roi.total_hours == 131.7 and roi.total_money == 1_317_000.0 and roi.currency == "₸"
    assert any("16.0 ч" in a for a in roi.assumptions) and "1000" in roi.llm_cost


def test_roi_custom_coefficients() -> None:
    s = Settings(_env_file=None, roi_review_minutes_per_req=10, roi_hourly_rate=2.5, roi_rework_hours_major=3)
    roi = compute_roi(_m(3, 0, 1, 0), s)
    assert roi.total_hours == 3.5 and roi.total_money == 9.0  # (0.5 + 3) × 2.5 = 8.75 → 9 (HALF_UP)


def test_roi_and_funnel_rendered(tmp_path: Path, fixtures: Path, settings: Settings) -> None:
    from req_validator.analyzer import run_audit
    from req_validator.reporting import write_reports

    report = run_audit(fixtures / "flawed_fintech_srs.md", demo=True, settings=settings)
    written, errors = write_reports(report, tmp_path / "out" / "r.md", "md,pdf,html")
    assert errors == []
    md = (tmp_path / "out" / "r.md").read_text(encoding="utf-8")
    assert "### Бизнес-ценность (ROI)" in md and "**Итого сэкономлено**" in md and "ROI_HOURLY_RATE" in md
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
