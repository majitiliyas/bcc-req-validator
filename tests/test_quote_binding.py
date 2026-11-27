"""Короткие точные цитаты (< 5 символов) и перепривязка находок к требованию, где цитата найдена дословно."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from req_validator.config import Settings
from req_validator.grounding import verify_citation
from req_validator.router import ChatRequest
from test_auditor_pipeline import FakeLLM, audit

ROW = "| **REST type** | GET |"
SRC = f"| Поле | Значение |\n| :--- | :--- |\n| **Description** | Сохранение порога |\n{ROW}\n\nTARGET get GET-запрос."


def _span(fragment: str, text: str = SRC) -> tuple[int, int]:
    s = text.index(fragment)
    return s, s + len(fragment)


# --------------------------------------------------------------------------- #
# verify_citation: короткие цитаты
# --------------------------------------------------------------------------- #


def test_short_exact_quote_inside_handle_accepted() -> None:
    c = verify_citation("GET", SRC, handle_span=_span(ROW))
    assert c.verified and c.short and c.method == "exact" and c.handle_ok is True
    assert SRC[c.start_char:c.end_char] == "GET" and c.start_char >= SRC.index(ROW)


@pytest.mark.parametrize("quote", ["get", "GTE", "GE"])
def test_short_quote_requires_exact_case_and_no_fuzzy(quote: str) -> None:
    c = verify_citation(quote, SRC, handle_span=_span(ROW))
    assert not c.verified or c.handle_ok is False  # «get» есть только вне строки; «GTE»/«GE» — нигде целым токеном
    assert not (c.verified and c.handle_ok)


def test_short_quote_must_be_whole_token() -> None:
    text = "| REST type | TARGET |"
    assert not verify_citation("GET", text, handle_span=(0, len(text))).verified  # «GET» внутри «TARGET» — не цитата


@pytest.mark.parametrize("quote", ["", " ", "—", "G", "|"])
def test_meaningless_short_quotes_rejected(quote: str) -> None:
    assert not verify_citation(quote, SRC, handle_span=_span(ROW)).verified


def test_short_quote_outside_handle_reports_all_occurrences() -> None:
    c = verify_citation("GET", SRC, handle_span=_span("| **Description** | Сохранение порога |"))
    assert c.verified and c.handle_ok is False and len(c.spans) == 2  # строка таблицы и «GET-запрос»
    assert "вне указанного требования" in c.reason


def test_long_quotes_keep_five_char_minimum_and_fuzzy() -> None:
    c = verify_citation("Сохранение порогa", SRC)  # латинская «a» — нечёткое совпадение длинной цитаты
    assert c.verified and c.method == "fuzzy" and not c.short


# --------------------------------------------------------------------------- #
# Сквозной сценарий: Pass 2 → grounding gate
# --------------------------------------------------------------------------- #

API_DOC = """# API уведомлений

## Общие параметры

| Поле | Значение |
| :--- | :--- |
| **Description** | Сохранение порога |
| **REST type** | GET |

## Ответ

* **API-01.** Если результат 200 или 201 — успешный.
* **API-02.** Ответ 200 содержит {"result":"ERROR"} при отклонении суммы.
"""


def _c(**kw: Any) -> dict[str, Any]:
    base = {"kind": "CONTRADICTION", "rule_id": "ISO-526-02", "severity": "MAJOR", "category": "SET_CONSISTENCY",
            "related_req_id": None, "related_quote": None, "explanation": "e", "suggestion": "s"}
    return {**base, **kw}


def pass2(req: ChatRequest) -> dict[str, Any]:
    return {"findings": [
        # 1. короткая точная цитата в своём требовании
        _c(req_id="REQ-002", verbatim_quote="GET", related_req_id="REQ-001", related_quote="Сохранение порога",
           explanation="Метод GET для операции сохранения."),
        # 2. вторая цитата указана с неверным ID → перепривязка к REQ-004
        _c(req_id="REQ-003", verbatim_quote="Если результат 200 или 201 — успешный", related_req_id="REQ-003",
           related_quote='содержит {"result":"ERROR"}', explanation="Код 200 с телом ошибки."),
        # 3. основная цитата указана с неверным ID → перепривязка к REQ-003
        _c(kind="MISSING_SCENARIO", rule_id="ISO-526-01", req_id="REQ-001", verbatim_quote="Если результат 200 или 201",
           explanation="Не описаны коды 4xx/5xx."),
        # 4. короткая цитата без хендла, встречается в двух требованиях → неоднозначно → отклонение
        _c(kind="OTHER", rule_id="ISO-526-01", req_id=None, verbatim_quote="200", explanation="Неясный код."),
        # 5. нечёткое совпадение вне заявленного требования → не перепривязывается
        _c(kind="OTHER", rule_id="ISO-526-01", req_id="REQ-001", verbatim_quote="Если результат 200 или 201 — успешнй",
           explanation="Опечатка в цитате."),
    ]}


@pytest.fixture
def report(tmp_path: Path):
    p = tmp_path / "api.md"
    p.write_text(API_DOC, encoding="utf-8")
    s = Settings(_env_file=None, cache_dir=tmp_path / ".cache", gemini_api_key=SecretStr("g"),
                 nvidia_api_key=SecretStr("n"), groq_api_key=SecretStr("q"), deep_models="gemini/model-a,nvidia/model-b",
                 fast_models="groq/model-fast", enable_ollama=False, use_presidio=False)
    fake = FakeLLM(pass1=lambda r: {"reviewed_req_ids": [], "findings": []}, pass2=pass2,
                   pass0=lambda r: {"terms": []}, repair=lambda r: {"verbatim_quote": None})
    rep = audit(s, p, fake)
    rep.__dict__["_fake"] = fake
    return rep


def _llm(report: Any) -> list[Any]:
    return [d for d in report.defects if d.pass_name == "pass2"]


def test_requirement_ids_in_fixture(report: Any) -> None:
    assert [(r.req_id, r.unit_type.value) for r in report.requirements[:2]] == [("REQ-001", "table"), ("REQ-002", "table")]
    assert [r.source_id for r in report.requirements[2:]] == ["API-01", "API-02"]


def test_short_get_contradiction_restored(report: Any) -> None:
    d = next(x for x in _llm(report) if x.exact_quote == "GET")
    assert d.req_id == "REQ-002" and d.quote_verified and d.binding_note == ""
    assert d.related_req_ids == ["REQ-001"] and d.related_quote == "Сохранение порога"


def test_related_quote_rebound_to_owner(report: Any) -> None:
    d = next(x for x in _llm(report) if x.exact_quote.startswith("Если результат 200 или 201 — успешный"))
    assert d.related_req_ids == ["REQ-004"] and d.related_quote_verified
    assert d.related_quote == 'содержит {"result":"ERROR"}'
    assert "REQ-003 → REQ-004" in d.binding_note


def test_primary_quote_rebound_with_note(report: Any) -> None:
    d = next(x for x in _llm(report) if x.rule_id == "ISO-526-01/MISSING_SCENARIO")
    assert d.req_id == "REQ-003" and "ID скорректирован: REQ-001 → REQ-003" in d.binding_note
    raw = API_DOC
    assert raw[d.quote_start:d.quote_end] == d.exact_quote


def test_ambiguous_short_and_fuzzy_outside_rejected(report: Any) -> None:
    quotes = [d.exact_quote for d in _llm(report)]
    assert "200" not in quotes
    assert not any("успешнй" in q for q in quotes)
    examples = " | ".join(report.passport.llm.rejected_examples)
    assert "неоднозначна" in examples and "нечётко" in examples
    assert report.passport.llm.unverified_rejected == 2
    repairs = [c for c in report.__dict__["_fake"].calls if c[1].startswith("repair")]
    assert len(repairs) == 1  # ремонт только для нечёткой цитаты с хендлом; неоднозначная без хендла — сразу отклонена


def test_binding_note_rendered(report: Any, tmp_path: Path) -> None:
    from req_validator.reporting import write_reports

    written, errors = write_reports(report, tmp_path / "out" / "r.md", "md,pdf,html")
    assert errors == []
    md = (tmp_path / "out" / "r.md").read_text(encoding="utf-8")
    assert "**Привязка:** ID скорректирован: REQ-001 → REQ-003" in md
    assert "Привязка:" in (tmp_path / "out" / "r.html").read_text(encoding="utf-8")
    import pymupdf

    assert "Привязка" in "".join(p.get_text() for p in pymupdf.open(tmp_path / "out" / "r.pdf"))
