"""Детерминированный инжест документов: .md / .txt / .pdf → IngestionResult.

Контракт: функция `ingest()` никогда не бросает исключение наружу. Любая проблема
входного файла превращается в типизированный статус и человекочитаемое предупреждение.

Все смещения (`start_char`/`end_char`) указывают в `IngestionResult.raw_markdown` —
это декодированный текст документа с нормализованными переводами строк (`\\n`).
Для PDF raw_markdown — Markdown, построенный из текстового слоя (pymupdf4llm) или OCR.
"""

from __future__ import annotations

import bisect
import contextlib
import hashlib
import io
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token

from .models import BlockType, DocumentBlock, IngestionResult, IngestionStatus, TableRow

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
MIN_PAGE_TEXT_CHARS = 30  # порог «пустой» страницы (скан) по CLAUDE.md §2.1
TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".text"}
PDF_EXTENSIONS = {".pdf"}
KNOWN_UNSUPPORTED = {
    ".docx": "DOCX", ".doc": "DOC", ".rtf": "RTF", ".odt": "ODT", ".xlsx": "XLSX",
    ".pptx": "PPTX", ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".zip": "ZIP",
}

# Частые строчные буквы русского языка — для выбора между однобайтовыми кириллическими кодировками.
_RU_FREQUENT = set("оеаинтсрвлкмдпу")
_CYRILLIC_FALLBACKS = ("cp1251", "koi8_r", "cp866", "mac_cyrillic")


# --------------------------------------------------------------------------- #
# Публичный API
# --------------------------------------------------------------------------- #


def ingest(path: str | Path, *, enable_ocr: bool = True) -> IngestionResult:
    """Прочитать файл и вернуть IngestionResult. Никогда не бросает исключений."""
    p = Path(path)
    base = {"file_name": p.name, "file_path": str(p)}
    try:
        if not p.exists():
            return IngestionResult(status=IngestionStatus.FILE_NOT_FOUND, warnings=[f"Файл не найден: {p}"], **base)
        if not p.is_file():
            return IngestionResult(status=IngestionStatus.UNSUPPORTED, warnings=[f"Путь не является файлом: {p}"], **base)
        size = p.stat().st_size
        base["file_size_bytes"] = size
        if size > MAX_FILE_SIZE_BYTES:
            return IngestionResult(
                status=IngestionStatus.UNSUPPORTED,
                warnings=[f"Файл слишком большой ({size} байт > {MAX_FILE_SIZE_BYTES})."],
                **base,
            )
        data = p.read_bytes()
        base["sha256"] = hashlib.sha256(data).hexdigest()
        if not data.strip():
            return IngestionResult(
                status=IngestionStatus.EMPTY_DOCUMENT, source_format=_guess_format(p, data),
                warnings=["Файл пуст или содержит только пробельные символы."], **base,
            )
        fmt = _guess_format(p, data)
        if fmt == "pdf":
            return _ingest_pdf(data, base, enable_ocr=enable_ocr, ext_mismatch=p.suffix.lower() not in PDF_EXTENSIONS)
        if fmt in ("md", "txt", "text?"):
            return _ingest_text(data, base, fmt)
        label = KNOWN_UNSUPPORTED.get(p.suffix.lower(), p.suffix or "без расширения")
        return IngestionResult(
            status=IngestionStatus.UNSUPPORTED, source_format="unknown",
            warnings=[f"Формат {label} не поддерживается. Поддерживаются: .md, .txt, .pdf."], **base,
        )
    except Exception as exc:  # noqa: BLE001 — граница graceful degradation
        return IngestionResult(
            status=IngestionStatus.CORRUPTED_FILE,
            warnings=[f"Непредвиденная ошибка чтения файла: {type(exc).__name__}: {exc}"],
            **base,
        )


def parse_markdown(md: str, page_starts: list[int] | None = None) -> tuple[list[DocumentBlock], str]:
    """Разобрать Markdown в плоский список блоков с heading_path и точными смещениями.

    Возвращает (blocks, title). `page_starts` — смещения начала страниц (для PDF).
    """
    return _MarkdownBlockParser(md, page_starts).parse()


# --------------------------------------------------------------------------- #
# Формат и кодировка
# --------------------------------------------------------------------------- #


def _guess_format(p: Path, data: bytes) -> str:
    head = data.lstrip()[:8]
    if head.startswith(b"%PDF"):
        return "pdf"
    ext = p.suffix.lower()
    if ext in PDF_EXTENSIONS:
        return "pdf"  # битый PDF без сигнатуры — пусть разберётся PDF-ветка
    if ext in (".md", ".markdown"):
        return "md"
    if ext in (".txt", ".text"):
        return "txt"
    if ext in KNOWN_UNSUPPORTED or data[:2] == b"PK":
        return "unknown"
    return "text?"  # неизвестное расширение: попробуем как текст


def decode_text(data: bytes) -> tuple[str | None, str, list[str]]:
    """Определить кодировку и декодировать. Возвращает (text|None, encoding, warnings)."""
    warnings: list[str] = []
    boms = (
        (b"\xef\xbb\xbf", "utf-8-sig"),
        (b"\xff\xfe\x00\x00", "utf-32"),
        (b"\x00\x00\xfe\xff", "utf-32"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
    )
    for bom, enc in boms:
        if data.startswith(bom):
            try:
                return data.decode(enc), enc, warnings
            except UnicodeDecodeError:
                break
    try:
        return data.decode("utf-8"), "utf-8", warnings
    except UnicodeDecodeError:
        pass

    from charset_normalizer import from_bytes

    best = from_bytes(data).best()
    candidates: list[tuple[str, str]] = []
    if best is not None:
        candidates.append((best.encoding, str(best)))
    for enc in _CYRILLIC_FALLBACKS:
        if best is not None and enc == best.encoding:
            continue
        try:
            candidates.append((enc, data.decode(enc)))
        except UnicodeDecodeError:
            continue
    if not candidates:
        return None, "unknown", ["Не удалось определить кодировку текста."]

    def ru_score(text: str) -> int:
        return sum(1 for ch in text if ch in _RU_FREQUENT)

    enc, text = max(candidates, key=lambda c: (ru_score(c[1]), c is candidates[0]))
    if best is None or ru_score(text) == 0:
        enc, text = candidates[0]
    warnings.append(f"Кодировка файла не UTF-8: определена как {enc}.")
    return text, enc, warnings


def _looks_binary(text: str) -> bool:
    if "\x00" in text:
        return True
    ctrl = sum(1 for ch in text if ord(ch) < 32 and ch not in "\t\n\r\f\v")
    return ctrl > max(3, len(text) * 0.02)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")


# --------------------------------------------------------------------------- #
# Текстовые форматы
# --------------------------------------------------------------------------- #


def _ingest_text(data: bytes, base: dict[str, Any], fmt: str) -> IngestionResult:
    text, encoding, warnings = decode_text(data)
    source_format = "txt" if fmt == "txt" else "md"
    if fmt == "text?":
        warnings.append("Неизвестное расширение файла — содержимое обработано как Markdown/текст.")
    if text is None or _looks_binary(text):
        return IngestionResult(
            status=IngestionStatus.CORRUPTED_FILE, source_format=source_format, detected_encoding=encoding,
            warnings=warnings + ["Файл содержит двоичные данные и не является текстовым документом."], **base,
        )
    text = _normalize_newlines(text)
    if not text.strip():
        return IngestionResult(
            status=IngestionStatus.EMPTY_DOCUMENT, source_format=source_format, detected_encoding=encoding,
            warnings=warnings + ["Документ не содержит текста."], **base,
        )
    blocks, title = parse_markdown(text)
    if not any(b.block_type != BlockType.HEADING for b in blocks):
        warnings.append("В документе нет содержательных блоков (только заголовки).")
    return IngestionResult(
        status=IngestionStatus.OK, source_format=source_format, detected_encoding=encoding,
        warnings=warnings, title=title or Path(base["file_name"]).stem, raw_markdown=text, blocks=blocks, **base,
    )


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


def _ingest_pdf(data: bytes, base: dict[str, Any], *, enable_ocr: bool, ext_mismatch: bool) -> IngestionResult:
    import pymupdf

    warnings: list[str] = []
    if ext_mismatch:
        warnings.append("Расширение файла не .pdf, но содержимое распознано как PDF.")
    common = {"source_format": "pdf", "detected_encoding": "pdf-text-layer", **base}
    if not data.lstrip().startswith(b"%PDF"):
        return IngestionResult(
            status=IngestionStatus.CORRUPTED_FILE, warnings=warnings + ["Файл не содержит сигнатуры PDF (%PDF)."], **common
        )
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        return IngestionResult(
            status=IngestionStatus.CORRUPTED_FILE, warnings=warnings + [f"PDF повреждён и не открывается: {exc}"], **common
        )
    with doc:
        if doc.needs_pass:
            return IngestionResult(
                status=IngestionStatus.UNSUPPORTED, warnings=warnings + ["PDF защищён паролем."], **common
            )
        page_count = doc.page_count
        if page_count == 0:
            return IngestionResult(
                status=IngestionStatus.CORRUPTED_FILE, page_count=0,
                warnings=warnings + ["PDF не содержит ни одной страницы."], **common,
            )

        layer_texts: list[str] = []
        for i in range(page_count):
            try:
                layer_texts.append(doc[i].get_text("text") or "")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Страница {i + 1}: не удалось извлечь текстовый слой ({exc}).")
                layer_texts.append("")

        page_md = _pdf_to_markdown_pages(doc, layer_texts, warnings)

        low_pages = [i for i, t in enumerate(layer_texts) if len(t.strip()) < MIN_PAGE_TEXT_CHARS]
        ocr_pages: list[int] = []
        ocr_missing = False
        for i in low_pages:
            ocr_text = _ocr_page(doc[i]) if enable_ocr else None
            if ocr_text is None:
                ocr_missing = True
                if layer_texts[i].strip():
                    warnings.append(f"Страница {i + 1}: мало текста ({len(layer_texts[i].strip())} симв.) — возможно, скан.")
                continue
            if len(ocr_text.strip()) > len(page_md[i].strip()):
                page_md[i] = ocr_text
                ocr_pages.append(i + 1)

        if low_pages and ocr_missing:
            pages = ", ".join(str(i + 1) for i in low_pages)
            reason = "OCR отключён" if not enable_ocr else "OCR недоступен (установите extra `ocr`: uv sync --extra ocr)"
            warnings.append(f"Страницы без текстового слоя: {pages}; {reason}.")

    md_parts: list[str] = []
    page_starts: list[int] = []
    cursor = 0
    for text in page_md:
        text = _normalize_newlines(text).strip("\n")
        page_starts.append(cursor)
        md_parts.append(text)
        cursor += len(text) + 2  # разделитель "\n\n"
    md = "\n\n".join(md_parts)

    if not md.strip():
        status = IngestionStatus.NO_TEXT_LAYER if low_pages else IngestionStatus.EMPTY_DOCUMENT
        msg = "PDF не содержит извлекаемого текста (скан без текстового слоя)." if low_pages else "PDF не содержит текста."
        return IngestionResult(
            status=status, page_count=page_count, warnings=warnings + [msg], ocr_pages=ocr_pages, **common
        )

    blocks, title = parse_markdown(md, page_starts)
    status = IngestionStatus.PARTIAL_OCR if ocr_pages else IngestionStatus.OK
    if ocr_pages:
        warnings.append(f"Страницы распознаны через OCR: {', '.join(map(str, ocr_pages))} — возможны ошибки распознавания.")
    return IngestionResult(
        status=status, page_count=page_count, ocr_pages=ocr_pages, warnings=warnings,
        title=title or Path(base["file_name"]).stem, raw_markdown=md, blocks=blocks, **common,
    )


def _pdf_to_markdown_pages(doc: Any, layer_texts: list[str], warnings: list[str]) -> list[str]:
    """pymupdf4llm → Markdown по страницам; при сбое — простой текстовый слой."""
    try:
        import pymupdf4llm

        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            chunks = pymupdf4llm.to_markdown(doc, page_chunks=True, show_progress=False)
        pages = [""] * len(layer_texts)
        for idx, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {}) or {}
            num = meta.get("page_number") or meta.get("page") or idx + 1
            if 1 <= int(num) <= len(pages):
                pages[int(num) - 1] = chunk.get("text", "") or ""
        return pages
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"pymupdf4llm не смог построить Markdown ({type(exc).__name__}); использован простой текстовый слой.")
        return list(layer_texts)


@lru_cache(maxsize=1)
def _ocr_engine() -> Any | None:
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        try:
            from rapidocr import RapidOCR  # type: ignore[import-not-found,no-redef]
        except Exception:  # noqa: BLE001
            return None
    try:
        return RapidOCR()
    except Exception:  # noqa: BLE001
        return None


def ocr_available() -> bool:
    return _ocr_engine() is not None


def _ocr_page(page: Any) -> str | None:
    """Распознать страницу через RapidOCR (CPU). None — если OCR недоступен или упал."""
    engine = _ocr_engine()
    if engine is None:
        return None
    try:
        png = page.get_pixmap(dpi=200).tobytes("png")
        result = engine(png)
        items = result[0] if isinstance(result, tuple) else getattr(result, "txts", None)
        if not items:
            return ""
        if isinstance(items[0], (list, tuple)):  # rapidocr_onnxruntime: [box, text, score]
            items = sorted(items, key=lambda it: (round(it[0][0][1] / 12), it[0][0][0]))
            return "\n".join(str(it[1]) for it in items)
        return "\n".join(str(t) for t in items)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Markdown → блоки
# --------------------------------------------------------------------------- #

_LIST_MARKER_RE = re.compile(r"^\s*(?:[*+\-]|\d{1,9}[.)])\s+")
_LIST_OPEN = {"bullet_list_open", "ordered_list_open"}
_LIST_CLOSE = {"bullet_list_close", "ordered_list_close"}


def inline_to_plain(token: Token | None) -> str:
    """Текст inline-токена без markdown-разметки."""
    if token is None:
        return ""
    if not token.children:
        return token.content.strip()
    parts: list[str] = []
    for child in token.children:
        if child.type in ("text", "code_inline", "html_inline"):
            parts.append(child.content)
        elif child.type in ("softbreak", "hardbreak"):
            parts.append(" ")
        elif child.type == "image":
            parts.append(child.content or "")
    return " ".join("".join(parts).split())


class _MarkdownBlockParser:
    def __init__(self, md: str, page_starts: list[int] | None) -> None:
        self.md = md
        self.lines = md.split("\n")
        self.line_off: list[int] = []
        off = 0
        for line in self.lines:
            self.line_off.append(off)
            off += len(line) + 1
        self.page_starts = page_starts
        self.tokens = MarkdownIt("commonmark").enable("table").parse(md)
        self.blocks: list[DocumentBlock] = []
        self.heads: list[tuple[int, str]] = []
        self.title = ""

    # --- помощники -------------------------------------------------------- #

    def _page(self, start: int) -> int | None:
        if not self.page_starts:
            return None
        return bisect.bisect_right(self.page_starts, start)

    def _span(self, s: int, e: int, *, skip_marker: bool = False) -> tuple[int, int]:
        while e > s + 1 and not self.lines[e - 1].strip():
            e -= 1
        first = self.lines[s]
        lead = len(first) - len(first.lstrip())
        if skip_marker:
            m = _LIST_MARKER_RE.match(first)
            if m:
                lead = m.end()
        start = self.line_off[s] + lead
        last = self.lines[e - 1]
        end = self.line_off[e - 1] + len(last.rstrip())
        return start, max(start, end)

    def _add(self, btype: BlockType, start: int, end: int, **extra: Any) -> DocumentBlock:
        titles = [t for _, t in self.heads]
        block = DocumentBlock(
            block_id=f"B-{len(self.blocks) + 1:04d}",
            block_type=btype,
            heading_path=" > ".join(titles),
            heading_titles=titles,
            text=self.md[start:end],
            start_char=start,
            end_char=end,
            page_num=self._page(start),
            **extra,
        )
        self.blocks.append(block)
        return block

    def _close_index(self, i: int, close_type: str) -> int:
        level = self.tokens[i].level
        for j in range(i + 1, len(self.tokens)):
            if self.tokens[j].type == close_type and self.tokens[j].level == level:
                return j
        return len(self.tokens) - 1

    # --- основной проход -------------------------------------------------- #

    def parse(self) -> tuple[list[DocumentBlock], str]:
        toks = self.tokens
        list_depth = 0
        list_intro_stack: list[str | None] = []
        last_paragraph: DocumentBlock | None = None
        i = 0
        while i < len(toks):
            tok = toks[i]
            t = tok.type
            if t == "heading_open" and tok.map:
                level = int(tok.tag[1])
                text = inline_to_plain(toks[i + 1] if i + 1 < len(toks) else None)
                while self.heads and self.heads[-1][0] >= level:
                    self.heads.pop()
                self.heads.append((level, text))
                if not self.title and (level == 1 or not self.blocks):
                    self.title = text
                s, e = self._span(*tok.map)
                self._add(BlockType.HEADING, s, e, level=level)
                last_paragraph = None
                i = self._close_index(i, "heading_close") + 1
                continue
            if t in _LIST_OPEN:
                intro = None
                if list_depth == 0 and last_paragraph is not None and last_paragraph.text.rstrip().endswith(":"):
                    intro = " ".join(last_paragraph.text.split())
                elif list_depth > 0 and self.blocks and self.blocks[-1].block_type == BlockType.LIST_ITEM:
                    parent = self.blocks[-1].text.rstrip()
                    intro = " ".join(parent.split()) if parent.endswith(":") else list_intro_stack[-1] if list_intro_stack else None
                list_intro_stack.append(intro)
                list_depth += 1
            elif t in _LIST_CLOSE:
                list_depth = max(0, list_depth - 1)
                if list_intro_stack:
                    list_intro_stack.pop()
                last_paragraph = None
            elif t == "list_item_open" and tok.map:
                close = self._close_index(i, "list_item_close")
                s, e = tok.map
                for j in range(i + 1, close):
                    if toks[j].type in _LIST_OPEN and toks[j].map and toks[j].level == tok.level + 1:
                        e = toks[j].map[0]
                        break
                start, end = self._span(s, max(e, s + 1), skip_marker=True)
                marker = f"{tok.info}{tok.markup}" if tok.info else tok.markup
                self._add(
                    BlockType.LIST_ITEM, start, end, level=list_depth, list_marker=marker,
                    list_intro=list_intro_stack[-1] if list_intro_stack else None,
                )
            elif t == "paragraph_open" and tok.map and list_depth == 0:
                s, e = self._span(*tok.map)
                last_paragraph = self._add(BlockType.PARAGRAPH, s, e)
                i = self._close_index(i, "paragraph_close") + 1
                continue
            elif t == "table_open" and tok.map:
                close = self._close_index(i, "table_close")
                self._add_table(i, close)
                last_paragraph = None
                i = close + 1
                continue
            elif t in ("fence", "code_block") and tok.map:
                self._add_code(tok)
                last_paragraph = None
            i += 1
        return self.blocks, self.title

    def _add_code(self, tok: Token) -> None:
        s, e = tok.map  # type: ignore[misc]
        content = tok.content.rstrip("\n")
        start, end = self._span(s, e)
        if tok.type == "fence" and content:
            idx = self.md.find(content, self.line_off[s], self.line_off[e - 1] + len(self.lines[e - 1]) + 1)
            if idx >= 0:
                start, end = idx, idx + len(content)
        self._add(BlockType.CODE, start, end, code_lang=(tok.info or "").strip() or None)

    def _add_table(self, i: int, close: int) -> None:
        toks = self.tokens
        rows: list[list[str]] = []
        current: list[str] | None = None
        for j in range(i, close):
            tt = toks[j].type
            if tt == "tr_open":
                current = []
            elif tt in ("th_open", "td_open") and current is not None:
                current.append(inline_to_plain(toks[j + 1] if toks[j + 1].type == "inline" else None))
            elif tt == "tr_close" and current is not None:
                rows.append(current)
                current = None
        s, e = toks[i].map  # type: ignore[misc]
        start, end = self._span(s, e)
        header = rows[0] if rows else []
        body = rows[1:]
        line_idxs = [k for k in range(s + 2, e) if self.lines[k].strip()]
        table_rows: list[TableRow] = []
        if len(line_idxs) == len(body):
            for cells, k in zip(body, line_idxs, strict=True):
                rs, re_ = self._span(k, k + 1)
                table_rows.append(TableRow(cells=cells, raw=self.md[rs:re_], start_char=rs, end_char=re_))
        else:  # нестандартная разметка: строки без точных смещений строки — привязка ко всей таблице
            table_rows = [TableRow(cells=c, raw=self.md[start:end], start_char=start, end_char=end) for c in body]
        self._add(BlockType.TABLE, start, end, table_header=header, table_rows=table_rows)
