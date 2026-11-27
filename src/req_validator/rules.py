"""Детерминированные правила (пре-фильтр) по regex-словарю стоп-паттернов из docs/iso_29148_ieee_830.md §2.

Два уровня выхода:
* Defect — высокоточные нарушения (TBD, «и т.д.», «по возможности», «быстро» без метрики…);
  учитываются в метриках.
* Hint — низкоточные сигналы (пассивный залог, местоимения, «может», нет ветки ELSE);
  в метриках не учитываются, передаются LLM-проходу (Спринт 2) и показываются в отчёте.

Каждая находка цитирует дословный фрагмент требования (предложение, где сработал паттерн).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .models import (
    BlockType,
    Defect,
    DefectCategory,
    Hint,
    IngestionResult,
    Requirement,
    RequirementKind,
    Severity,
)
from .segmenter import MODAL_MARKERS, find_explicit_id, split_sentences, strip_markdown, word_count
from .standards import CRITERIA_BY_ID

_HAS_NUMBER = re.compile(r"\d")
_INCLUSIVE = re.compile(r"включительно|не включая|исключительно|не более|не менее|строго|обе границы|[<>≤≥]", re.I)
_PROSE_KINDS = {
    RequirementKind.FUNCTIONAL, RequirementKind.NON_FUNCTIONAL, RequirementKind.DATA,
    RequirementKind.INTEGRATION, RequirementKind.OTHER,
}
_INFINITIVE_END = ("ть", "ти", "чь", "ться", "тись")
_ELSE_MARKERS = re.compile(r"иначе|в противном случае|если не|при отсутствии|при ошибке|при отклонении|при неуспех|в случае отказ", re.I)


@dataclass(frozen=True)
class Pattern:
    code: str  # ISO-525-03/WEAK_MODALITY
    regex: re.Pattern[str]
    severity: Severity | None  # None → Hint
    category: DefectCategory
    explanation: str
    suggestion: str
    risk: str
    downgrade_if_number: bool = False  # с числовой метрикой в предложении → Hint


def _rx(*alts: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{a})" for a in alts), re.IGNORECASE)


_R_AMBIG = "Риск разночтений между бизнесом, разработкой и QA; переделки на этапе тестирования."
_R_BLOCK = "Риск остановки разработки: требование невозможно реализовать без уточнений."

PATTERNS: tuple[Pattern, ...] = (
    Pattern("ISO-526-01/TBD", _rx(r"\bTBD\b", r"\bTBR\b", r"\bTBA\b", r"\bTODO\b", r"\bFIXME\b", r"\bуточнить\b",
                                  r"будет определено позже", r"на этапе разработки", r"по согласованию", r"\?{2,}"),
            Severity.FATAL, DefectCategory.SET_CONSISTENCY,
            "Требование содержит заглушку вместо значения/логики (TBD).",
            "Заменить заглушку конкретным значением или правилом; если решение не принято — вынести пункт в Out of Scope с владельцем и сроком.",
            _R_BLOCK),
    Pattern("ISO-525-04/PLACEHOLDER", _rx(r"\bзаполня(?:ется|ются)\b", r"будет (?:заполнен|добавлен|описан)\w*", r"\bне заполнен\w*"),
            Severity.MAJOR, DefectCategory.SINGLE_REQ_QUALITY,
            "Содержание раздела отложено («заполняется позже») — требование фактически отсутствует.",
            "Заполнить раздел до передачи в разработку или явно указать, кто и к какому сроку его заполнит.",
            "Риск неполного покрытия: QA не получит критериев проверки."),
    Pattern("ISO-525-04/MISSING_VALUE", _rx(r"(?<=:)\s*[—–]\s*(?=$|[.;,|]|\s*\|)"),
            Severity.MAJOR, DefectCategory.SINGLE_REQ_QUALITY,
            "Значение параметра не указано (прочерк после метки).",
            "Указать значение или явно написать «не применяется» с причиной.",
            _R_BLOCK),
    Pattern("ISO-525-03/WEAK_MODALITY", _rx(r"\bпо возможности\b", r"\bжелательно\b", r"\bрекомендуется\b", r"\bкак правило\b",
                                            r"\bдопускается\b", r"\bпри необходимости\b", r"\bв случае целесообразности\b",
                                            r"\bпо усмотрению\b"),
            Severity.MAJOR, DefectCategory.SINGLE_REQ_QUALITY,
            "Слабая модальность делает требование необязательным и неоднозначным.",
            "Заменить на «Система должна…» с явным условием применения (EARS: «Если <условие>, система должна…»).",
            _R_AMBIG),
    Pattern("ISO-525-03/OPEN_LIST", _rx(r"\bи т\.\s?д\.?", r"\bи т\.\s?п\.?", r"\bи др\.", r"\bи проч\.?", r"включая,? но не ограничиваясь",
                                        r"\bи далее\b", r"\bи так далее\b", r"\betc\.?"),
            Severity.MAJOR, DefectCategory.SINGLE_REQ_QUALITY,
            "Открытый список: множество значений не определено, его нельзя реализовать и протестировать полностью.",
            "Перечислить все значения закрытым списком или сослаться на справочник/таблицу.",
            _R_AMBIG),
    Pattern("ISO-525-07/UNMEASURABLE", _rx(r"\bбыстр\w*", r"\bмгновенн\w*", r"\bв реальном времени\b", r"\bудобн\w*", r"\bинтуитивн\w*",
                                           r"\bнадежн\w*", r"\bнадёжн\w*", r"\bбезопасн\w*", r"\bкорректн\w*", r"\bадекватн\w*",
                                           r"\bоптимальн\w*", r"\bгибк\w*", r"\bпрозрачн\w*", r"\bмаксимальн\w*", r"\bминимальн\w*",
                                           r"\bв кратчайшие сроки\b", r"\bсвоевременн\w*", r"\bвысок\w* нагрузк\w*", r"\bэффективн\w*"),
            Severity.MAJOR, DefectCategory.SINGLE_REQ_QUALITY,
            "Неизмеримая характеристика без метрики приёмки — требование непроверяемо.",
            "Заменить прилагательное измеримым порогом (например, «не более 2 с для 95-го перцентиля»).",
            "Риск спора при приёмке: нет объективного критерия «выполнено».",
            downgrade_if_number=True),
    Pattern("ISO-525-07/GLOBAL_QUANTIFIER", _rx(r"\bникогда\b", r"\bисключить люб\w+", r"\bабсолютно\b", r"\b100\s?% (?:случаев|времени)"),
            Severity.MAJOR, DefectCategory.SINGLE_REQ_QUALITY,
            "Глобальный квантор без границ невозможно проверить конечным тестом.",
            "Сформулировать измеримую цель (доступность 99,9 % в месяц, допустимая доля ошибок и т. п.).",
            "Риск невыполнимого критерия приёмки."),
    # --- подсказки ---
    Pattern("ISO-525-03/MAY", _rx(r"\bможет\b", r"\bмогут\b"), None, DefectCategory.SINGLE_REQ_QUALITY,
            "«может» — возможность или обязанность? Проверьте модальность.", "", ""),
    Pattern("ISO-525-03/EXAMPLE_LIST", _rx(r"\bнапример\b", r"\bв том числе\b(?!:)"), None, DefectCategory.SINGLE_REQ_QUALITY,
            "Пример вместо исчерпывающего правила — проверьте, закрыт ли список.", "", ""),
    Pattern("ISO-525-03/PRONOUN", _rx(r"\b(?:он|она|оно|они|данн(?:ый|ая|ое|ые|ого|ой)|указанн\w+ выше|вышеуказанн\w+)\b"), None,
            DefectCategory.SINGLE_REQ_QUALITY, "Местоимение/отсылка — убедитесь, что антецедент однозначен.", "", ""),
    Pattern("ISO-525-04/PASSIVE", _rx(r"\b(?:производится|осуществляется|выполняется|формируется|отправляется|блокируется|списывается|"
                                      r"зачисляется|определяется|переводится|сохраняется|исключаются|исключается|передаётся|передается|"
                                      r"рассчитывается|проверяется|обрабатывается|создаётся|создается|начисляется|возвращается|"
                                      r"устанавливается|удаляется)\b"), None,
            DefectCategory.SINGLE_REQ_QUALITY, "Пассивный залог: проверьте, указан ли актор (система/оператор/внешний сервис).", "", ""),
)
PATTERNS_BY_CODE = {p.code: p for p in PATTERNS}


def _criterion_name(code: str) -> str:
    c = CRITERIA_BY_ID.get(code.split("/", 1)[0])
    return c.name_ru if c else code


def _draft(code: str, severity: Severity, category: DefectCategory, req: Requirement | None, quote: str,
           explanation: str, suggestion: str, risk: str, section: str = "") -> Defect:
    return Defect(
        defect_id="DEF-000", severity=severity, rule_id=code, criterion=_criterion_name(code), category=category,
        req_id=req.req_id if req else None, section_path=req.heading_path if req else section,
        exact_quote=quote, explanation=explanation, suggestion=suggestion, risk=risk,
    )


def _sentence_at(text: str, pos: int, limit: int = 320) -> str:
    for s, e in split_sentences(text):
        if s <= pos < e:
            sent = text[s:e]
            break
    else:
        sent = text
    if len(sent) > limit:
        start = max(0, min(pos - limit // 2, len(sent) - limit))
        sent = sent[start:start + limit]
    return sent.strip()


# --------------------------------------------------------------------------- #
# Правила уровня требования
# --------------------------------------------------------------------------- #


def check_requirement(req: Requirement) -> tuple[list[Defect], list[Hint]]:
    defects: list[Defect] = []
    hints: list[Hint] = []
    text = req.text
    seen: set[str] = set()
    for p in PATTERNS:
        if p.code == "ISO-525-04/MISSING_VALUE" and req.unit_type == BlockType.CODE:
            continue
        m = p.regex.search(text)
        if not m:
            continue
        if p.code == "ISO-525-04/MISSING_VALUE" and req.unit_type == BlockType.TABLE:
            continue  # прочерки в таблицах проверяются отдельно (_check_table_cells)
        quote = _sentence_at(text, m.start())
        severity = p.severity
        if severity is not None and p.downgrade_if_number and _HAS_NUMBER.search(quote):
            severity = None
        if severity is None:
            hints.append(Hint(rule_id=p.code, req_id=req.req_id, exact_quote=quote,
                              message=p.explanation + (" (рядом есть число — проверьте, задана ли метрика)" if p.severity else "")))
            continue
        if p.code in seen:
            continue
        seen.add(p.code)
        found = m.group().strip() or "—"
        defects.append(_draft(p.code, severity, p.category, req, quote,
                              f"{p.explanation} Маркер: «{found}».", p.suggestion, p.risk))

    defects += _check_boundaries(req)
    defects += _check_table_cells(req)
    d, h = _check_conforming(req)
    defects += d
    hints += h
    hints += _structure_hints(req)
    return defects, hints


_BOUNDARY = re.compile(r"\b(?:до|свыше|более|менее|от)\s+\d[\d  ]*", re.IGNORECASE)


def _check_boundaries(req: Requirement) -> list[Defect]:
    if req.unit_type == BlockType.CODE:
        return []
    for m in _BOUNDARY.finditer(req.text):
        if not m.group().lower().startswith("до"):
            continue
        quote = _sentence_at(req.text, m.start())
        if _INCLUSIVE.search(quote) or re.search(r"\d{1,2}:\d{2}", m.group() + req.text[m.end():m.end() + 3]):
            continue
        return [_draft("ISO-525-03/BOUNDARY", Severity.MAJOR, DefectCategory.FINTECH_MATH_LIMITS, req, quote,
                       f"Граница «{m.group().strip()}» не уточняет, входит ли значение в диапазон (< или ≤).",
                       "Указать «включительно» / «не включая» или записать неравенство явно (x ≤ N).",
                       "Риск ошибок на граничных значениях (BVA), спорные кейсы при приёмке.")]
    return []


_EMPTY_CELL = {"—", "–", "-", "--"}


def _check_table_cells(req: Requirement) -> list[Defect]:
    if req.unit_type != BlockType.TABLE:
        return []
    raw_cells = [c.strip() for c in req.text.strip().strip("|").split("|")]
    if any(c in _EMPTY_CELL for c in raw_cells):
        return [_draft("ISO-526-01/EMPTY_CELL", Severity.MINOR, DefectCategory.SET_CONSISTENCY, req, req.text.strip(),
                       "Прочерк в ячейке без пояснения в легенде: неясно «не применяется», «запрещено» или «забыли заполнить».",
                       "Заменить прочерк явным значением или добавить легенду («— = не применяется»).",
                       "Риск неверной интерпретации контракта/таблицы.")]
    m = re.search(r"(?<=:)\s*[—–]\s*(?=$|[.;,|]|\s*\|)", req.text)
    if m:
        return [_draft("ISO-525-04/MISSING_VALUE", Severity.MAJOR, DefectCategory.SINGLE_REQ_QUALITY, req, req.text.strip(),
                       "Значение параметра не указано (прочерк после метки).",
                       "Указать значение или явно написать «не применяется» с причиной.", _R_BLOCK)]
    return []


def _first_word(plain: str) -> str:
    sid = find_explicit_id(plain)
    rest = plain[plain.find(sid) + len(sid):] if sid else plain
    words = re.findall(r"[A-Za-zА-Яа-яЁё-]+", rest)
    return words[0].lower() if words else ""


def _check_conforming(req: Requirement) -> tuple[list[Defect], list[Hint]]:
    """Нет модального глагола → ПОДСКАЗКА (вне метрик и ROI): стилистика EARS, а не дефект смысла.
    Для целей/бизнес-процесса и непрозаических блоков не применяется."""
    if req.kind not in _PROSE_KINDS or req.unit_type not in (BlockType.PARAGRAPH, BlockType.LIST_ITEM):
        return [], []
    if any(re.search(r"\bцел[иь]\b|\bgoals?\b", t, re.I) for t in req.heading_path.split(" > ")):
        return [], []
    plain = req.plain_text
    if "→" in plain or word_count(plain) < 4 or MODAL_MARKERS.search(plain):
        return [], []
    if _first_word(plain).endswith(_INFINITIVE_END):
        return [], []  # императив-инфинитив («Хранить…», «Проверить…») — допустимая форма
    first = split_sentences(req.text)
    quote = req.text[first[0][0]:first[0][1]] if first else req.text
    return [], [Hint(rule_id="ISO-525-09/NO_MODAL", req_id=req.req_id, exact_quote=quote.strip(),
                     message="Нет модального глагола («должна») — по EARS: «<Система> должна <действие> [при <условии>]».")]


def _structure_hints(req: Requirement) -> list[Hint]:
    hints: list[Hint] = []
    if req.unit_type in (BlockType.PARAGRAPH, BlockType.LIST_ITEM):
        sentences = split_sentences(req.text)
        if len(sentences) >= 3 or req.text.count(";") >= 2:
            hints.append(Hint(rule_id="ISO-525-05/MULTI_STATEMENT", req_id=req.req_id, exact_quote=req.text.strip()[:320],
                              message=f"В одном требовании {len(sentences)} утверждения — проверьте атомарность."))
        for s, e in sentences:
            sent = req.text[s:e]
            if re.match(r"^\W*(?:если|при|когда)\b", strip_markdown(sent), re.I) and not _ELSE_MARKERS.search(req.plain_text):
                hints.append(Hint(rule_id="ISO-525-04/NO_ELSE", req_id=req.req_id, exact_quote=sent.strip(),
                                  message="Условие без альтернативной ветки: что происходит, если условие не выполнено?"))
                break
    return hints


# --------------------------------------------------------------------------- #
# Правила уровня набора
# --------------------------------------------------------------------------- #

_PRIORITY = re.compile(r"\b(must|should|could|won't|moscow|приоритет\w*|mvp|p[0-3])\b", re.IGNORECASE)
_VERSION = re.compile(r"верси[яи]\s*(?:документа)?\s*[:№]?\s*\d|\bv\d+(?:\.\d+)+|редакци[яи]\s*№?\s*\d", re.IGNORECASE)
_SYNONYMS = {
    "клиент": r"\bклиент\w*", "пользователь": r"\bпользовател\w*", "держатель карты": r"\bдержател\w*",
    "владелец карты": r"\bвладел\w* карты", "заявитель": r"\bзаявител\w*", "физлицо": r"\bфизлиц\w*|\bфизическ\w+ лиц\w*",
}


@dataclass
class SetFindings:
    defects: list[Defect]
    notes: dict[str, str]  # rule_id критерия → комментарий для WARN без дефекта


def check_set(doc: IngestionResult, reqs: list[Requirement]) -> SetFindings:
    raw = doc.raw_markdown
    defects: list[Defect] = []
    notes: dict[str, str] = {}
    title_quote = doc.title or (reqs[0].text[:80] if reqs else "")
    n = len(reqs)
    if n == 0:
        return SetFindings(defects, notes)

    with_id = [r for r in reqs if r.has_explicit_id]
    if len(with_id) < n / 2:
        first = next(r for r in reqs if not r.has_explicit_id)
        defects.append(_draft(
            "ISO-526-06/NO_IDS", Severity.MAJOR, DefectCategory.SET_CONSISTENCY, None, _cut(first.text, 120),
            f"Уникальные идентификаторы есть только у {len(with_id)} из {n} требований; остальные пронумерованы агентом (REQ-xxx). "
            "Без ID невозможно трассировать требования к тестам и изменениям.",
            "Присвоить каждому нормативному пункту неизменяемый ID (например, FR-01, NFR-01, API-01) и использовать его в тест-кейсах.",
            "Риск потери трассируемости: изменения и дефекты нельзя однозначно привязать к требованию.",
            section=first.heading_path))
    dup = [sid for sid, c in Counter(r.source_id for r in with_id).items() if c > 1]
    for sid in dup:
        r = next(x for x in with_id if x.source_id == sid)
        defects.append(_draft("ISO-526-06/DUP_ID", Severity.MAJOR, DefectCategory.SET_CONSISTENCY, r, _cut(r.text, 120),
                              f"Идентификатор «{sid}» встречается в документе несколько раз.",
                              "Сделать идентификаторы уникальными.", "Риск неверной трассировки."))

    if not _VERSION.search(raw):
        m = re.search(r"[^\n.]*верси[яи][^\n.]*\.", raw, re.IGNORECASE)
        defects.append(_draft(
            "ISO-526-05/NO_VERSION", Severity.MINOR, DefectCategory.SET_CONSISTENCY, None,
            m.group().strip() if m else title_quote,
            "Версия и дата документа не зафиксированы — невозможно понять, какая редакция требований согласована и проверена.",
            "Добавить в шапку версию, дату и историю изменений; обновлять при каждом изменении требований.",
            "Риск разработки по устаревшей редакции.", section=doc.title))

    if n > 50 and not _PRIORITY.search(raw):
        defects.append(_draft(
            "ISO-526-07/NO_PRIORITY", Severity.MINOR, DefectCategory.SET_CONSISTENCY, None, title_quote,
            f"В документе {n} требований и нет приоритизации (MoSCoW / MVP / P0–P3).",
            "Указать приоритет и стабильность для каждого требования или группы.",
            "Риск неверного планирования релиза.", section=doc.title))

    defects += _empty_tables(doc)
    defects += _unfilled_checklists(doc)

    used = [name for name, rx in _SYNONYMS.items() if re.search(rx, raw, re.IGNORECASE)]
    if len(used) >= 3:
        notes["ISO-526-02"] = ("Одна сущность названа по-разному: " + ", ".join(f"«{u}»" for u in used)
                               + ". Проверьте, один ли это актор, и закрепите термин в глоссарии.")
    return SetFindings(defects, notes)


def _empty_tables(doc: IngestionResult) -> list[Defect]:
    out: list[Defect] = []
    for b in doc.blocks:
        if b.block_type != BlockType.TABLE or not b.table_rows:
            continue
        if all(not c.strip() for row in b.table_rows for c in row.cells):
            header_line = b.text.split("\n", 1)[0].strip()
            out.append(_draft(
                "ISO-526-01/EMPTY_TABLE", Severity.MINOR, DefectCategory.SET_CONSISTENCY, None, header_line,
                f"Таблица в разделе «{b.heading_titles[-1] if b.heading_titles else '—'}» не заполнена.",
                "Заполнить таблицу (ответственные, согласования) или удалить её из документа.",
                "Риск передачи в разработку несогласованного документа.", section=b.heading_path))
    return out


_UNSELECTED = re.compile(r"^\s*да\s*/\s*нет(\s*/\s*не\s+требуется)?\s*$", re.IGNORECASE)


def _unfilled_checklists(doc: IngestionResult) -> list[Defect]:
    """Чек-лист, где в ячейках остались варианты «Да / Нет» без выбора, — признак незавершённого ревью документа."""
    out: list[Defect] = []
    for b in doc.blocks:
        if b.block_type != BlockType.TABLE:
            continue
        rows = [r for r in b.table_rows if any(_UNSELECTED.match(c) for c in r.cells)]
        if not rows:
            continue
        section = b.heading_titles[-1] if b.heading_titles else "—"
        out.append(_draft(
            "ISO-526-01/UNFILLED_CHECKLIST", Severity.MINOR, DefectCategory.SET_CONSISTENCY, None, rows[0].raw.strip(),
            f"Чек-лист в разделе «{section}» не заполнен: в {len(rows)} из {len(b.table_rows)} строк остались варианты "
            "«Да / Нет» без выбора — ревью требований не проведено или не зафиксировано.",
            "Заполнить чек-лист (выбрать значение в каждой строке, указать примечания) до передачи в разработку.",
            "Риск передачи в разработку документа без внутреннего ревью.", section=b.heading_path))
    return out


def _cut(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut[: cut.rfind(" ")] if " " in cut else cut
