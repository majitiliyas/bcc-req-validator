"""Нарезка: явные ID, синтетические REQ-xxx, List Unrolling, строки таблиц, стабильность и смещения."""

from __future__ import annotations

from pathlib import Path

from req_validator.ingestion import ingest
from req_validator.models import BlockType, RequirementKind
from req_validator.segmenter import find_explicit_id, segment, split_sentences


def _seg(path: Path):
    doc = ingest(path)
    return doc, segment(doc)


def test_explicit_ids_and_sequential_report_ids(fixtures: Path) -> None:
    doc, seg = _seg(fixtures / "clean_srs.md")
    reqs = seg.requirements
    assert [r.req_id for r in reqs] == [f"REQ-{i:03d}" for i in range(1, len(reqs) + 1)]
    src = [r.source_id for r in reqs if r.has_explicit_id]
    assert src == ["FR-01", "FR-02", "FR-03", "FR-04", "NFR-01", "AC-01", "AC-02"]
    nfr = next(r for r in reqs if r.source_id == "NFR-01")
    assert nfr.kind == RequirementKind.NON_FUNCTIONAL
    ac = next(r for r in reqs if r.source_id == "AC-01")
    assert ac.kind == RequirementKind.ACCEPTANCE and ac.unit_type == BlockType.TABLE


def test_informative_sections_skipped(fixtures: Path) -> None:
    _, seg = _seg(fixtures / "clean_srs.md")
    assert not any("Назначение документа" in r.heading_path for r in seg.requirements)


def test_quotes_are_exact_slices(fixtures: Path) -> None:
    for name in ("clean_srs.md", "flawed_fintech_srs.md", "tables_lists.md"):
        doc, seg = _seg(fixtures / name)
        for r in seg.requirements:
            assert doc.raw_markdown[r.start_char:r.end_char] == r.text, (name, r.req_id)


def test_segmentation_is_deterministic(fixtures: Path) -> None:
    _, a = _seg(fixtures / "flawed_fintech_srs.md")
    _, b = _seg(fixtures / "flawed_fintech_srs.md")
    assert [(r.req_id, r.fingerprint, r.start_char) for r in a.requirements] == \
           [(r.req_id, r.fingerprint, r.start_char) for r in b.requirements]


def test_document_without_ids_gets_synthetic_ids(tmp_path: Path) -> None:
    p = tmp_path / "noids.md"
    p.write_text(
        "# ТЗ\n\n## Функциональные требования\n\n"
        "Система должна блокировать карту после трёх неверных вводов ПИН. "
        "Система должна уведомлять клиента о блокировке push-сообщением.\n",
        encoding="utf-8",
    )
    _, seg = _seg(p)
    assert len(seg.requirements) == 2  # абзац разбит на предложения
    assert all(not r.has_explicit_id and r.source_id is None for r in seg.requirements)
    assert seg.requirements[1].text.startswith("Система должна уведомлять")


def test_list_unrolling_and_tables(fixtures: Path) -> None:
    _, seg = _seg(fixtures / "tables_lists.md")
    items = [r for r in seg.requirements if r.unit_type == BlockType.LIST_ITEM]
    assert items[0].context == "Система должна отклонять заявку в следующих случаях:"
    assert len(items) == 4
    rows = [r for r in seg.requirements if r.unit_type == BlockType.TABLE]
    assert [r.source_id for r in rows] == ["RBAC-1", "RBAC-2"]
    assert rows[0].plain_text.startswith("ID: RBAC-1; Роль: Оператор")


def test_ordered_list_numbers_are_not_explicit_ids(fixtures: Path) -> None:
    _, seg = _seg(fixtures / "flawed_fintech_srs.md")
    items = [r for r in seg.requirements if r.unit_type == BlockType.LIST_ITEM]
    assert items[0].source_id == "п. 1" and not items[0].has_explicit_id


def test_find_explicit_id_patterns() -> None:
    assert find_explicit_id("Ф-1. Держатель карты задаёт порог") == "Ф-1"
    assert find_explicit_id("REQ-001 Система должна") == "REQ-001"
    assert find_explicit_id("[FR-2.1] Система должна") == "FR-2.1"
    assert find_explicit_id("3.2.1 Система должна") == "3.2.1"
    assert find_explicit_id("200. Ответ в случае успеха") is None
    assert find_explicit_id("API используется для") is None


def test_split_sentences_respects_abbreviations() -> None:
    text = "Проверить ИНН, БИК и т.д. в справочнике. Сумма 1.5 млн руб. допускается. Итог."
    parts = [text[s:e] for s, e in split_sentences(text)]
    assert parts[0] == "Проверить ИНН, БИК и т.д. в справочнике."
    assert parts[1].startswith("Сумма 1.5 млн руб. допускается.")
