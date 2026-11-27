"""Инжест: пустые/битые/бинарные файлы, кодировки, PDF, таблицы, списки, точные смещения."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_text_pdf
from req_validator.ingestion import ingest
from req_validator.models import BlockType, IngestionStatus

RU_TEXT = (
    "# Требования к переводам\n\n"
    "## Функциональные требования\n\n"
    "* **FR-01.** Система должна проверять остаток счёта перед списанием.\n"
    "* **FR-02.** Система должна отклонять перевод при недостаточном остатке.\n"
)


@pytest.mark.parametrize("content", [b"", b"   \n\t\n  ", b"\xef\xbb\xbf  \n"])
def test_empty_file(tmp_path: Path, content: bytes) -> None:
    p = tmp_path / "empty.md"
    p.write_bytes(content)
    r = ingest(p)
    assert r.status == IngestionStatus.EMPTY_DOCUMENT
    assert r.blocks == []
    assert r.warnings


def test_missing_file(tmp_path: Path) -> None:
    r = ingest(tmp_path / "nope.md")
    assert r.status == IngestionStatus.FILE_NOT_FOUND
    assert not r.status.can_proceed


def test_broken_pdf_fixture(fixtures: Path) -> None:
    r = ingest(fixtures / "broken_edge_case.pdf")
    assert r.status == IngestionStatus.CORRUPTED_FILE
    assert r.sha256 and r.warnings


def test_truncated_pdf(tmp_path: Path) -> None:
    good = make_text_pdf(tmp_path / "good.pdf", ["<p>Система должна работать.</p>"])
    data = good.read_bytes()
    bad = tmp_path / "truncated.pdf"
    bad.write_bytes(data[: len(data) // 3])
    r = ingest(bad)  # не должно быть исключения
    assert r.status in (IngestionStatus.CORRUPTED_FILE, IngestionStatus.EMPTY_DOCUMENT, IngestionStatus.NO_TEXT_LAYER,
                        IngestionStatus.OK)


def test_pdf_extension_without_signature(tmp_path: Path) -> None:
    p = tmp_path / "fake.pdf"
    p.write_text("это не pdf", encoding="utf-8")
    assert ingest(p).status == IngestionStatus.CORRUPTED_FILE


def test_binary_garbage_with_md_extension(tmp_path: Path) -> None:
    p = tmp_path / "garbage.md"
    p.write_bytes(bytes(range(256)) * 20)
    r = ingest(p)
    assert r.status == IngestionStatus.CORRUPTED_FILE


def test_unsupported_docx(tmp_path: Path) -> None:
    p = tmp_path / "spec.docx"
    p.write_bytes(b"PK\x03\x04" + b"\x00" * 50)
    assert ingest(p).status == IngestionStatus.UNSUPPORTED


@pytest.mark.parametrize("encoding", ["cp1251", "koi8_r", "utf-16"])
def test_cyrillic_encodings(tmp_path: Path, encoding: str) -> None:
    p = tmp_path / f"spec_{encoding}.md"
    p.write_bytes(RU_TEXT.encode(encoding))
    r = ingest(p)
    assert r.status == IngestionStatus.OK
    assert r.raw_markdown == RU_TEXT
    assert r.detected_encoding.replace("-", "_").lower().startswith(encoding.replace("-", "_")[:5])
    assert r.title == "Требования к переводам"
    items = [b for b in r.blocks if b.block_type == BlockType.LIST_ITEM]
    assert len(items) == 2 and "остаток счёта" in items[0].text


def test_cp1251_txt_reports_warning(tmp_path: Path) -> None:
    p = tmp_path / "spec.txt"
    p.write_bytes(RU_TEXT.encode("cp1251"))
    r = ingest(p)
    assert r.detected_encoding == "cp1251"
    assert any("cp1251" in w for w in r.warnings)


def test_tables_lists_structure_and_offsets(fixtures: Path) -> None:
    r = ingest(fixtures / "tables_lists.md")
    assert r.status == IngestionStatus.OK
    for b in r.blocks:  # каждый блок — дословный срез документа
        assert r.raw_markdown[b.start_char:b.end_char] == b.text
        for row in b.table_rows:
            assert r.raw_markdown[row.start_char:row.end_char] == row.raw

    items = [b for b in r.blocks if b.block_type == BlockType.LIST_ITEM]
    assert [b.level for b in items] == [1, 1, 2, 1]
    assert items[0].list_intro == "Система должна отклонять заявку в следующих случаях:"
    assert items[1].text == "скоринговый балл ниже 600;"  # без вложенного пункта
    assert items[0].heading_path == "Заявка на кредит > Функциональные требования"

    table = next(b for b in r.blocks if b.block_type == BlockType.TABLE)
    assert table.table_header == ["ID", "Роль", "Создание", "Утверждение"]
    assert [row.cells[0] for row in table.table_rows] == ["RBAC-1", "RBAC-2"]
    assert table.heading_titles[-1] == "Матрица ролей"

    code = next(b for b in r.blocks if b.block_type == BlockType.CODE)
    assert code.code_lang == "json" and code.text.startswith('{"status"')


def test_text_pdf(tmp_path: Path) -> None:
    pdf = make_text_pdf(tmp_path / "spec.pdf", [
        "<h1>Требования</h1><p>Ф-1. Система должна отправлять уведомление в течение 60 секунд.</p>",
        "<p>Ф-2. Система должна хранить историю уведомлений не менее 5 лет.</p>",
    ])
    r = ingest(pdf)
    assert r.status == IngestionStatus.OK
    assert r.page_count == 2
    assert "в течение 60 секунд" in r.raw_markdown
    pages = {b.page_num for b in r.blocks}
    assert pages == {1, 2}
    for b in r.blocks:
        assert r.raw_markdown[b.start_char:b.end_char] == b.text


def test_scanned_pdf_without_text_layer(tmp_path: Path) -> None:
    pdf = make_text_pdf(tmp_path / "scan.pdf", [None, None])
    r = ingest(pdf)
    assert r.status in (IngestionStatus.NO_TEXT_LAYER, IngestionStatus.PARTIAL_OCR)
    assert any("текстового слоя" in w or "OCR" in w for w in r.warnings)


def test_mixed_pdf_warns_about_scan_page(tmp_path: Path) -> None:
    pdf = make_text_pdf(tmp_path / "mixed.pdf", ["<p>Система должна вести журнал всех операций клиента.</p>", None])
    r = ingest(pdf, enable_ocr=False)
    assert r.status == IngestionStatus.OK
    assert any("2" in w and "текстового слоя" in w for w in r.warnings)
