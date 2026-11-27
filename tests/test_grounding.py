"""Grounding: проверка цитат, хендлов, чисел; безопасная Decimal-арифметика."""

from __future__ import annotations

from decimal import Decimal

import pytest

from req_validator.grounding import (
    NumericError,
    deduplicate_defects,
    from_minor_units,
    ground_defects,
    numbers_in_text,
    parse_decimal,
    quantize,
    safe_decimal_eval,
    to_minor_units,
    verify_citation,
)
from req_validator.models import BlockType, Defect, DefectCategory, Requirement, RequirementKind, Severity

SOURCE = (
    "## Функциональные требования\n"
    "* **Ф-1.** Держатель активной карты KZT задаёт целый порог от 1 000 до 1 000 000 ₸ включительно.\n"
    "* **Ф-2.** Месяц определяется по дате совершения покупки в Алматы.\n"
)


def _req(req_id: str, fragment: str) -> Requirement:
    s = SOURCE.index(fragment)
    return Requirement(req_id=req_id, source_id=None, has_explicit_id=False, kind=RequirementKind.FUNCTIONAL,
                       unit_type=BlockType.LIST_ITEM, heading_path="ФТ", text=fragment, plain_text=fragment,
                       start_char=s, end_char=s + len(fragment), block_id="B-1", fingerprint="x")


def _defect(quote: str, req_id: str | None, rule: str = "ISO-525-03/X", sev: Severity = Severity.MAJOR) -> Defect:
    return Defect(defect_id="DEF-000", severity=sev, rule_id=rule, criterion="Однозначность",
                  category=DefectCategory.SINGLE_REQ_QUALITY, req_id=req_id, exact_quote=quote,
                  explanation="e", suggestion="s")


def test_exact_match_returns_offsets() -> None:
    c = verify_citation("задаёт целый порог", SOURCE)
    assert c.verified and c.method == "exact"
    assert SOURCE[c.start_char:c.end_char] == "задаёт целый порог"


def test_normalized_match_ignores_markdown_whitespace_case_yo() -> None:
    c = verify_citation("Ф-1. держатель   активной карты KZT задает", SOURCE)
    assert c.verified and c.method == "normalized"
    assert c.matched_text.endswith("задаёт")  # возвращается дословный текст документа
    assert SOURCE[c.start_char:c.end_char] == c.matched_text


def test_fuzzy_match_with_typo() -> None:
    c = verify_citation("Месяц определяеться по дате совершения покупки в Алмате", SOURCE)
    assert c.verified and c.method == "fuzzy" and c.score >= 85


def test_number_mismatch_is_rejected_not_substituted() -> None:
    c = verify_citation("задаёт целый порог от 1 000 до 100 000 ₸ включительно", SOURCE)
    assert not c.verified and not c.numbers_ok
    assert "Числа" in c.reason


def test_hallucinated_and_short_quotes_rejected() -> None:
    assert not verify_citation("Система должна шифровать данные алгоритмом ГОСТ", SOURCE).verified
    assert not verify_citation("abc", SOURCE).verified
    assert not verify_citation("", SOURCE).verified


def test_handle_check() -> None:
    r1 = _req("REQ-001", "Держатель активной карты KZT задаёт целый порог от 1 000 до 1 000 000 ₸ включительно.")
    inside = verify_citation("задаёт целый порог", SOURCE, handle_span=(r1.start_char, r1.end_char))
    assert inside.handle_ok is True
    outside = verify_citation("Месяц определяется по дате", SOURCE, handle_span=(r1.start_char, r1.end_char))
    assert outside.verified and outside.handle_ok is False


def test_ground_defects_rejects_unknown_handle_and_reassigns() -> None:
    r1 = _req("REQ-001", "Держатель активной карты KZT задаёт целый порог от 1 000 до 1 000 000 ₸ включительно.")
    r2 = _req("REQ-002", "Месяц определяется по дате совершения покупки в Алматы.")
    defects = [
        _defect("задаёт целый порог", "REQ-001"),
        _defect("задаёт целый порог", "REQ-999"),  # несуществующий хендл
        _defect("Месяц определяется по дате", "REQ-001"),  # цитата из другого требования → перепривязка
        _defect("Выдумка, которой нет в документе вовсе", "REQ-002"),
    ]
    accepted, rejected = ground_defects(defects, [r1, r2], SOURCE)
    assert [d.req_id for d in accepted] == ["REQ-001", "REQ-002"]
    assert all(d.quote_verified and d.quote_start is not None for d in accepted)
    assert accepted[0].binding_note == ""
    assert accepted[1].binding_note.startswith("ID скорректирован: REQ-001 → REQ-002")
    assert len(rejected) == 2


def test_deduplicate_keeps_most_severe() -> None:
    a = _defect("задаёт целый порог", "REQ-001", sev=Severity.MINOR)
    b = _defect("задаёт  целый порог", "REQ-001", sev=Severity.FATAL)
    c = _defect("задаёт целый порог", "REQ-001", rule="ISO-525-07/Y")
    out = deduplicate_defects([a, b, c])
    assert len(out) == 2 and out[0].severity == Severity.FATAL


def test_numbers_in_text() -> None:
    assert numbers_in_text("от 1 000 до 1 000 000 ₸, 23:58, 1,5 %") == [
        Decimal(1000), Decimal(1000000), Decimal(23), Decimal(58), Decimal("1.5")]


# --- Decimal / simpleeval ---------------------------------------------------- #


def test_safe_eval_decimal_precision() -> None:
    assert safe_decimal_eval("0.1 + 0.2") == Decimal("0.3")
    assert safe_decimal_eval("48500 + 1500") == Decimal(50000)
    assert safe_decimal_eval("limit - spent", {"limit": "50 000", "spent": "48 500 ₸"}) == Decimal(1500)
    assert safe_decimal_eval("round_half_even(2.675, 2)") == Decimal("2.68")
    assert safe_decimal_eval("round_half_even(2.665, 2)") == Decimal("2.66")
    assert safe_decimal_eval("max(1, 2) * -3") == Decimal(-6)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hacked')", "2 ** 100000", "(1).real", "'a' * 3",
    "[1, 2]", "lambda: 1", "x", "open('f')", "1 if 1 else 2", "",
])
def test_safe_eval_blocks_dangerous_input(expr: str) -> None:
    with pytest.raises(NumericError):
        safe_decimal_eval(expr)


def test_safe_eval_division_by_zero() -> None:
    with pytest.raises(NumericError):
        safe_decimal_eval("1 / 0")


def test_float_is_forbidden_for_money() -> None:
    with pytest.raises(NumericError):
        parse_decimal(0.1)  # type: ignore[arg-type]
    with pytest.raises(NumericError):
        safe_decimal_eval("a + 1", {"a": 0.1})  # type: ignore[dict-item]


def test_money_helpers() -> None:
    assert parse_decimal("1 000 000 ₸") == Decimal(1000000)
    assert parse_decimal("1 000,50") == Decimal("1000.50")
    assert quantize(Decimal("2.345"), 2, "HALF_EVEN") == Decimal("2.34")
    assert quantize(Decimal("2.345"), 2, "HALF_UP") == Decimal("2.35")
    assert quantize(Decimal("2.341"), 2, "CEILING") == Decimal("2.35")
    assert to_minor_units("48 500 ₸") == 4_850_000
    assert from_minor_units(4_850_050) == Decimal("48500.50")
    with pytest.raises(NumericError):
        to_minor_units("1.005")
