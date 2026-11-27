"""Детерминированная нарезка документа на атомарные требования (ARU).

Метод — Hierarchical Contextual Chunking (docs/iso_29148_ieee_830.md §6.1):
1. Каждый блок получает путь заголовков (heading_path) и тип раздела по ключевым словам.
2. Блоки нормативных разделов (или содержащие модальные глаголы / явный ID) нарезаются:
   пункт списка → 1 ARU (с вводной фразой списка как контекст), строка таблицы → 1 ARU,
   абзац → по строкам с жёстким переносом и далее по предложениям, блок кода → 1 ARU.
3. Явный ID из документа (Ф-1, REQ-001, FR-2.1, 3.2.1) сохраняется как source_id; ID в отчёте
   всегда присваивается заново: REQ-001, REQ-002… в порядке следования в документе.

Смещения каждого ARU указывают в IngestionResult.raw_markdown, `text` — дословная цитата.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

from .models import (
    BlockType,
    DocumentBlock,
    IngestionResult,
    Requirement,
    RequirementKind,
    SegmentationResult,
)

# --------------------------------------------------------------------------- #
# Классификация разделов
# --------------------------------------------------------------------------- #

_INFORMATIVE_HEADINGS = re.compile(
    r"общие сведения|ревью требований|блок согласующ|лист согласован|история изменений|"
    r"введение|назначение документа|глоссари|термины и определения|содержание|оглавление|"
    r"список сокращений|контакты",
    re.IGNORECASE,
)

_SECTION_KINDS: list[tuple[re.Pattern[str], RequirementKind]] = [
    (re.compile(r"нефункциональ|\bnfr\b|производительност|надёжност|надежност|\bsla\b", re.I), RequirementKind.NON_FUNCTIONAL),
    (re.compile(r"критери\w* при[её]мк|при[её]мк|acceptance", re.I), RequirementKind.ACCEPTANCE),
    (re.compile(r"тестирован", re.I), RequirementKind.TESTING),
    (re.compile(r"интеграц", re.I), RequirementKind.INTEGRATION),
    (re.compile(r"функциональн|функци[ия]\b|use case|сценари|user stor|бизнес-правил|особые правила|"
                r"логика работы|алгоритм|правила валидац|тариф|лимит|комисси|ставк|ролев|матриц\w* (?:ролей|доступа|прав)|"
                r"права доступа|статусн\w* модел|жизненн\w* цикл", re.I), RequirementKind.FUNCTIONAL),
    (re.compile(r"требования к данным|модель данных|\bданны[ех]\b|\bdata\b|хранени", re.I), RequirementKind.DATA),
    (re.compile(r"\bapi\b|параметр|request|response|\bответ\b|статус|контракт|интерфейс|endpoint|метод", re.I),
     RequirementKind.INTERFACE),
    (re.compile(r"\bцел[иь]\b|бизнес-процесс|бизнес-требован|\bbpmn\b", re.I), RequirementKind.BUSINESS),
    (re.compile(r"архитектур|дизайн|\bui\b|\bux\b|sequence|схем|диаграмм|макет", re.I), RequirementKind.DESIGN),
    (re.compile(r"требован", re.I), RequirementKind.OTHER),
]

MODAL_MARKERS = re.compile(
    r"\b(должн\w*|обязан\w*|необходимо|требуется|следует|запрещ\w*|не допускается|"
    r"shall|must|should)\b",
    re.IGNORECASE,
)

# Явные идентификаторы требований в начале фрагмента.
_EXPLICIT_ID = re.compile(
    r"^\s*\[?(?P<id>"
    r"[A-ZА-ЯЁ][A-ZА-ЯЁ0-9]{0,7}(?:[-_.][A-ZА-ЯЁ0-9]{1,8})*[-_.]?\d+(?:\.\d+)*"  # Ф-1, REQ-001, FR-2.1
    r"|\d+(?:\.\d+){1,4}(?=\.?\s+[A-ZА-ЯЁ«\"])"  # 3.2.1 Система …
    r")\]?[.):]?(?=\s|$)"
)
_ID_COLUMN = re.compile(r"^(id|№|#|ид\.?|идентификатор|код требования|номер)$", re.IGNORECASE)

_ABBREVIATIONS = (
    "т.д.", "т.п.", "т.е.", "т.к.", "т.ч.", "т. д.", "т. п.", "т. е.", "др.", "пр.", "п.", "пп.", "ст.", "г.",
    "гг.", "руб.", "коп.", "тыс.", "млн.", "млрд.", "мин.", "сек.", "ч.", "см.", "рис.", "табл.", "стр.",
    "напр.", "e.g.", "i.e.", "etc.", "vs.", "ориг.",
)
_SENT_END = re.compile(r"[.!?…]+(?=\s|$)")
_MIN_UNIT_WORDS = 2
_MERGE_BELOW_WORDS = 4


@dataclass(frozen=True)
class SectionInfo:
    kind: RequirementKind | None  # None — информационный раздел
    informative_heading: bool


def classify_section(heading_titles: list[str]) -> SectionInfo:
    """Определить тип раздела по пути заголовков (от самого глубокого к корню)."""
    for title in reversed(heading_titles):
        if _INFORMATIVE_HEADINGS.search(title):
            return SectionInfo(kind=None, informative_heading=True)
        for pattern, kind in _SECTION_KINDS:
            if pattern.search(title):
                return SectionInfo(kind=kind, informative_heading=False)
    return SectionInfo(kind=None, informative_heading=False)


# --------------------------------------------------------------------------- #
# Текстовые помощники
# --------------------------------------------------------------------------- #

_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


def strip_markdown(text: str) -> str:
    """Убрать inline-разметку Markdown (жирный, курсив, код, ссылки) и схлопнуть пробелы."""
    text = _MD_LINK.sub(r"\1", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"(?<![\w*])\*(?=\S)|(?<=\S)\*(?![\w*])", "", text)
    return " ".join(text.split())


def word_count(text: str) -> int:
    """Число слов (буквенно-цифровых токенов); стрелки и знаки препинания не считаются."""
    return len(re.findall(r"\w+", text))


def find_explicit_id(plain: str) -> str | None:
    m = _EXPLICIT_ID.match(plain)
    return m.group("id") if m else None


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Разбить текст на предложения; возвращает относительные спаны (start, end) без краевых пробелов."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.end()
        tail = text[max(0, m.start() - 6): end].lower()
        if any(tail.endswith(a) for a in _ABBREVIATIONS):
            continue
        rest = text[end:].lstrip()
        if rest and not (rest[0].isupper() or rest[0].isdigit() or rest[0] in "«\"(*[`"):
            continue
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    trimmed: list[tuple[int, int]] = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            trimmed.append((s, e))
    return trimmed


def _merge_short(spans: list[tuple[int, int]], text: str) -> list[tuple[int, int]]:
    """Слить слишком короткие предложения со следующими («Карты → Расходы.» + следующее)."""
    out: list[tuple[int, int]] = []
    pending: tuple[int, int] | None = None
    for s, e in spans:
        if pending is not None:
            s = pending[0]
            pending = None
        if word_count(text[s:e]) < _MERGE_BELOW_WORDS and (s, e) != spans[-1]:
            pending = (s, e)
            continue
        out.append((s, e))
    if pending is not None:
        out.append(pending)
    return out


def _line_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for line in text.split("\n"):
        s = pos + (len(line) - len(line.lstrip()))
        e = pos + len(line.rstrip())
        if e > s:
            spans.append((s, e))
        pos += len(line) + 1
    return spans


def _fingerprint(plain: str) -> str:
    return hashlib.sha1(" ".join(plain.lower().split()).encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Сегментация
# --------------------------------------------------------------------------- #


@dataclass
class _Unit:
    block: DocumentBlock
    start: int
    end: int
    plain: str
    source_id: str | None
    explicit: bool
    context: str | None
    unit_type: BlockType


def segment(doc: IngestionResult) -> SegmentationResult:
    """Нарезать документ на требования. Детерминированно: один вход → одни и те же ID."""
    raw = doc.raw_markdown
    units: list[_Unit] = []
    informative: list[str] = []
    kinds: dict[int, RequirementKind] = {}

    for block in doc.blocks:
        if block.block_type == BlockType.HEADING:
            continue
        info = classify_section(block.heading_titles)
        block_units = _units_from_block(block, raw)
        kept = []
        for u in block_units:
            normative = info.kind is not None or (not info.informative_heading and bool(MODAL_MARKERS.search(u.plain)))
            if u.explicit and not info.informative_heading:
                normative = True
            if normative:
                kept.append(u)
                kinds[id(u)] = info.kind or RequirementKind.OTHER
        if kept:
            units.extend(kept)
        else:
            informative.append(block.block_id)

    requirements: list[Requirement] = []
    for n, u in enumerate(units, start=1):
        requirements.append(
            Requirement(
                req_id=f"REQ-{n:03d}",
                source_id=u.source_id,
                has_explicit_id=u.explicit,
                kind=kinds[id(u)],
                unit_type=u.unit_type,
                heading_path=u.block.heading_path,
                text=raw[u.start:u.end],
                plain_text=u.plain,
                context=u.context,
                start_char=u.start,
                end_char=u.end,
                page_num=u.block.page_num,
                block_id=u.block.block_id,
                fingerprint=_fingerprint(u.plain),
            )
        )

    warnings: list[str] = []
    if not requirements:
        warnings.append("Не выделено ни одного требования: в документе нет нормативных разделов или формулировок.")
    else:
        explicit = sum(r.has_explicit_id for r in requirements)
        warnings.append(f"Выделено требований: {len(requirements)}; с явным ID документа: {explicit}.")
        dup = [i for i, c in Counter(r.source_id for r in requirements if r.has_explicit_id).items() if c > 1]
        if dup:
            warnings.append(f"Повторяющиеся идентификаторы в документе: {', '.join(sorted(dup))}.")
    return SegmentationResult(requirements=requirements, informative_block_ids=informative, warnings=warnings)


def _units_from_block(block: DocumentBlock, raw: str) -> list[_Unit]:
    if block.block_type == BlockType.LIST_ITEM:
        plain = strip_markdown(block.text)
        if word_count(plain) < 1:
            return []
        sid = find_explicit_id(plain)
        explicit = sid is not None
        if not explicit and block.list_marker and block.list_marker[0].isdigit():
            sid = f"п. {block.list_marker.rstrip('.)')}"
        return [_Unit(block, block.start_char, block.end_char, plain, sid, explicit, block.list_intro, BlockType.LIST_ITEM)]

    if block.block_type == BlockType.TABLE:
        out: list[_Unit] = []
        header = block.table_header
        id_col = next((i for i, h in enumerate(header) if _ID_COLUMN.match(h.strip())), None)
        context = " | ".join(header) if header else None
        for row in block.table_rows:
            if not any(c.strip() for c in row.cells):
                continue
            pairs = [f"{h}: {c}" if h else c for h, c in zip(header, row.cells, strict=False) if c.strip()]
            plain = "; ".join(pairs) if pairs else " ".join(row.cells)
            sid = row.cells[id_col].strip() if id_col is not None and id_col < len(row.cells) else None
            sid = sid or find_explicit_id(row.cells[0].strip() if row.cells else "")
            out.append(_Unit(block, row.start_char, row.end_char, plain, sid or None, bool(sid), context, BlockType.TABLE))
        return out

    if block.block_type == BlockType.CODE:
        plain = " ".join(block.text.split())
        if not plain:
            return []
        return [_Unit(block, block.start_char, block.end_char, plain, None, False, block.code_lang, BlockType.CODE)]

    # PARAGRAPH: строки с жёстким переносом / ID → отдельно; далее — по предложениям.
    text = block.text
    lines = _line_spans(text)
    hard = len(lines) > 1 and (
        any(text.split("\n")[k].endswith(("  ", "\\")) for k in range(len(lines) - 1))
        or sum(1 for s, _ in lines if find_explicit_id(strip_markdown(text[s:]).strip()) or text[s:s + 2] == "**") > 1
    )
    chunks = lines if hard else [(0, len(text))]
    out = []
    for cs, ce in chunks:
        chunk = text[cs:ce].rstrip(" \\")
        ce = cs + len(chunk)
        sid = find_explicit_id(strip_markdown(chunk))
        if sid:
            spans = [(0, len(chunk))]
        else:
            spans = _merge_short(split_sentences(chunk), chunk)
        for s, e in spans:
            plain = strip_markdown(chunk[s:e])
            if word_count(plain) < _MIN_UNIT_WORDS:
                continue
            start = block.start_char + cs + s
            end = block.start_char + cs + e
            out.append(_Unit(block, start, end, plain, sid, sid is not None, None, BlockType.PARAGRAPH))
    return out
