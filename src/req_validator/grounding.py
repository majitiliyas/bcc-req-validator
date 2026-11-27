"""Grounding: верификация цитат (RapidFuzz), привязка к хендлу требования, дедупликация,
а также безопасная десятичная арифметика (Decimal + simpleeval, без eval/exec).

Принципы:
* Источник истины — документ. Возвращаемая цитата всегда дословный срез исходного текста.
* Числа не «подтягиваются» нечётким поиском: если в цитате 5 000, а в найденном
  фрагменте 50 000 — цитата отклоняется, а не исправляется.
* Хендл (req_id) обязан существовать; цитата ищется сначала внутри этого требования.
"""

from __future__ import annotations

import ast
import operator
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_UP, Decimal
from decimal import InvalidOperation, localcontext
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from simpleeval import SimpleEval

from .models import Defect, Requirement

DEFAULT_THRESHOLD = 85.0
MIN_QUOTE_CHARS = 5

# --------------------------------------------------------------------------- #
# Нормализация с картой индексов
# --------------------------------------------------------------------------- #

_CHAR_MAP = str.maketrans({
    "«": '"', "»": '"', "„": '"', "“": '"', "”": '"', "‘": "'", "’": "'",
    "–": "-", "—": "-", "‑": "-", "−": "-",
    "ё": "е", "Ё": "Е",
    " ": " ", " ": " ", " ": " ", " ": " ",
})
_DROP_CHARS = set("*`")


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Нормализовать текст для сравнения и вернуть карту: индекс в norm → индекс в исходнике.

    Убираются markdown-маркеры (* и `), типографские кавычки/тире приводятся к ASCII,
    ё→е, регистр понижается, пробельные последовательности схлопываются в один пробел.
    """
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text):
        if ch in _DROP_CHARS:
            continue
        ch = ch.translate(_CHAR_MAP)
        if ch.isspace():
            if not out or out[-1] == " ":
                continue
            ch = " "
        low = ch.lower()
        out.append(low if len(low) == 1 else ch)
        idx.append(i)
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


# --------------------------------------------------------------------------- #
# Числа
# --------------------------------------------------------------------------- #

_NUMBER_RE = re.compile(r"(?<![\w.,])\d{1,3}(?:[    ]\d{3})+(?:[.,]\d+)?(?![\d])|(?<![\w.,])\d+(?:[.,]\d+)?(?![\d])")


def numbers_in_text(text: str) -> list[Decimal]:
    """Извлечь числа с учётом разделителей разрядов («1 000 000» → 1000000, «1,5» → 1.5)."""
    result: list[Decimal] = []
    for m in _NUMBER_RE.finditer(text):
        token = re.sub(r"[    ]", "", m.group()).replace(",", ".")
        try:
            value = Decimal(token).normalize()
        except InvalidOperation:
            continue
        result.append(value.quantize(Decimal(1)) if value == value.to_integral_value() else value)
    return result


# --------------------------------------------------------------------------- #
# Верификация цитат
# --------------------------------------------------------------------------- #


class CitationCheck(BaseModel):
    verified: bool = Field(description="Цитата подтверждена документом")
    method: Literal["exact", "normalized", "fuzzy", "none"]
    score: float = Field(description="Сходство 0..100")
    matched_text: str = Field(default="", description="Дословный срез документа, найденный для цитаты")
    start_char: int | None = None
    end_char: int | None = None
    numbers_ok: bool = True
    handle_ok: bool | None = Field(default=None, description="Найдена ли цитата внутри указанного требования")
    reason: str = ""


def _search(quote: str, source: str, offset: int, threshold: float) -> CitationCheck | None:
    if not source:
        return None
    idx = source.find(quote)
    if idx >= 0:
        return CitationCheck(verified=True, method="exact", score=100.0, matched_text=quote,
                             start_char=offset + idx, end_char=offset + idx + len(quote))
    nq, _ = normalize_with_map(quote)
    ns, smap = normalize_with_map(source)
    if len(nq) < MIN_QUOTE_CHARS or not ns:
        return None
    pos = ns.find(nq)
    if pos >= 0:
        s, e = smap[pos], smap[pos + len(nq) - 1] + 1
        return CitationCheck(verified=True, method="normalized", score=100.0, matched_text=source[s:e],
                             start_char=offset + s, end_char=offset + e)
    al = fuzz.partial_ratio_alignment(nq, ns, score_cutoff=threshold)
    if al is None or al.dest_end <= al.dest_start:
        return None
    s, e = smap[al.dest_start], smap[al.dest_end - 1] + 1
    matched = source[s:e]
    ratio = len(normalize_with_map(matched)[0]) / max(1, len(nq))
    if not 0.7 <= ratio <= 1.3:
        return None
    return CitationCheck(verified=True, method="fuzzy", score=round(al.score, 1), matched_text=matched,
                         start_char=offset + s, end_char=offset + e)


def verify_citation(
    quote: str,
    source_text: str,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    handle_span: tuple[int, int] | None = None,
) -> CitationCheck:
    """Проверить, что цитата дословно (или почти дословно) присутствует в документе.

    1. Точное совпадение → нормализованное (пробелы, кавычки, тире, регистр, markdown) →
       нечёткое (RapidFuzz partial_ratio_alignment, порог `threshold`).
    2. Если задан `handle_span` (границы требования-хендла), поиск идёт сначала в нём;
       при находке вне хендла `handle_ok=False`.
    3. Числа в цитате и в найденном фрагменте должны совпасть как мультимножества,
       иначе цитата отклоняется (никакой подмены значений).
    """
    if not quote or len(" ".join(quote.split())) < MIN_QUOTE_CHARS:
        return CitationCheck(verified=False, method="none", score=0.0, reason="Цитата пустая или короче 5 символов.")

    check: CitationCheck | None = None
    handle_ok: bool | None = None
    if handle_span is not None:
        hs, he = max(0, handle_span[0]), min(len(source_text), handle_span[1])
        check = _search(quote, source_text[hs:he], hs, threshold)
        handle_ok = check is not None
    if check is None:
        check = _search(quote, source_text, 0, threshold)
        if check is not None and handle_span is not None:
            s, e = check.start_char or 0, check.end_char or 0
            handle_ok = handle_span[0] <= s and e <= handle_span[1]
    if check is None:
        return CitationCheck(verified=False, method="none", score=0.0, handle_ok=handle_ok,
                             reason="Цитата не найдена в документе (возможная галлюцинация).")

    check.handle_ok = handle_ok
    if check.method == "fuzzy" and Counter(numbers_in_text(quote)) != Counter(numbers_in_text(check.matched_text)):
        check.verified = False
        check.numbers_ok = False
        check.reason = "Числа в цитате не совпадают с документом — цитата отклонена без подмены значений."
        return check
    if handle_ok is False:
        check.reason = "Цитата найдена в документе, но вне указанного требования."
    return check


def ground_defects(
    defects: Iterable[Defect],
    requirements: Iterable[Requirement],
    source_text: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[Defect], list[tuple[Defect, str]]]:
    """Проверить цитату и хендл каждого дефекта. Возвращает (принятые, [(отклонённый, причина)]).

    * Неизвестный req_id → отклонение.
    * Цитата не найдена / числа расходятся → отклонение.
    * Цитата найдена в другом требовании → дефект перепривязывается к нему (доказательство — цитата).
    * Принятому дефекту проставляются дословная цитата из документа и её смещения.
    """
    reqs = list(requirements)
    by_id = {r.req_id: r for r in reqs}
    accepted: list[Defect] = []
    rejected: list[tuple[Defect, str]] = []
    for d in defects:
        span = None
        if d.req_id is not None:
            req = by_id.get(d.req_id)
            if req is None:
                rejected.append((d, f"Неизвестный идентификатор требования {d.req_id}."))
                continue
            span = (req.start_char, req.end_char)
        check = verify_citation(d.exact_quote, source_text, threshold, handle_span=span)
        if not check.verified:
            rejected.append((d, check.reason))
            continue
        update: dict[str, object] = {
            "exact_quote": check.matched_text, "quote_verified": True,
            "quote_start": check.start_char, "quote_end": check.end_char,
        }
        if check.handle_ok is False:
            owner = next((r for r in reqs if r.start_char <= (check.start_char or 0) and (check.end_char or 0) <= r.end_char), None)
            if owner is None:
                rejected.append((d, "Цитата не относится ни к одному выделенному требованию."))
                continue
            update.update(req_id=owner.req_id, section_path=owner.heading_path)
        accepted.append(d.model_copy(update=update))
    return accepted, rejected


def deduplicate_defects(defects: Iterable[Defect]) -> list[Defect]:
    """Убрать дубли: одно правило + одно требование + одна цитата → остаётся самый серьёзный."""
    best: dict[tuple[str, str | None, str], Defect] = {}
    order: list[tuple[str, str | None, str]] = []
    for d in defects:
        key = (d.rule_id, d.req_id, normalize_with_map(d.exact_quote)[0])
        if key not in best:
            best[key] = d
            order.append(key)
        elif d.severity.rank < best[key].severity.rank:
            best[key] = d
    return [best[k] for k in order]


# --------------------------------------------------------------------------- #
# Безопасная десятичная арифметика
# --------------------------------------------------------------------------- #


class NumericError(ValueError):
    """Ошибка разбора или вычисления числового выражения."""


_ROUNDING = {
    "HALF_EVEN": ROUND_HALF_EVEN, "HALF_UP": ROUND_HALF_UP, "CEILING": ROUND_CEILING,
    "FLOOR": ROUND_FLOOR, "DOWN": ROUND_DOWN, "UP": ROUND_UP,
}
_AMOUNT_CLEAN = re.compile(r"[\s   ]|₸|тг\.?|тенге|руб\.?|₽|kzt|rub|usd|\$|€", re.IGNORECASE)


def parse_decimal(text: str | int | Decimal) -> Decimal:
    """Разобрать сумму/число в Decimal: «1 000 000 ₸» → 1000000, «1 000,50» → 1000.50. float запрещён."""
    if isinstance(text, bool) or isinstance(text, float):
        raise NumericError("Тип float запрещён для денежных значений — передайте строку или Decimal.")
    if isinstance(text, Decimal):
        return text
    if isinstance(text, int):
        return Decimal(text)
    cleaned = _AMOUNT_CLEAN.sub("", text.strip()).replace(",", ".").replace("−", "-")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", cleaned):
        raise NumericError(f"Не число: {text!r}")
    return Decimal(cleaned)


def quantize(value: Decimal, places: int = 2, rounding: str = "HALF_EVEN") -> Decimal:
    """Округлить до `places` знаков по правилу (HALF_EVEN — банковское, HALF_UP, CEILING, FLOOR…)."""
    if rounding not in _ROUNDING:
        raise NumericError(f"Неизвестное правило округления: {rounding}")
    return value.quantize(Decimal(1).scaleb(-places), rounding=_ROUNDING[rounding])


def to_minor_units(amount: Decimal | str | int, exponent: int = 2) -> int:
    """Сумма → минорные единицы (тиыны/копейки). Дробный остаток недопустим."""
    value = parse_decimal(amount) * (Decimal(10) ** exponent)
    if value != value.to_integral_value():
        raise NumericError(f"Сумма {amount} не представима целым числом минорных единиц (10^-{exponent}).")
    return int(value)


def from_minor_units(minor: int, exponent: int = 2) -> Decimal:
    return (Decimal(minor) / (Decimal(10) ** exponent)).quantize(Decimal(1).scaleb(-exponent))


_LITERAL_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd,
)
_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _round_fn(rounding: str):  # noqa: ANN202
    def fn(value: Decimal, places: Decimal = Decimal(2)) -> Decimal:
        return quantize(Decimal(value), int(places), rounding)

    return fn


_FUNCTIONS = {
    "min": min, "max": max, "abs": abs,
    "round_half_even": _round_fn("HALF_EVEN"), "round_half_up": _round_fn("HALF_UP"),
    "round_ceiling": _round_fn("CEILING"), "round_floor": _round_fn("FLOOR"),
}


def safe_decimal_eval(expr: str, variables: Mapping[str, Decimal | int | str] | None = None) -> Decimal:
    """Вычислить арифметическое выражение в Decimal без eval/exec.

    Допускаются: + - * / // %, унарный минус, скобки, переменные из `variables`,
    функции min/max/abs/round_half_even/round_half_up/round_ceiling/round_floor.
    Возведение в степень, атрибуты, индексы, строки, лямбды и т.п. запрещены.
    Десятичный разделитель в выражении — точка. float на входе запрещён.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise NumericError("Пустое выражение.")
    if len(expr) > 500:
        raise NumericError("Выражение слишком длинное.")
    names: dict[str, Decimal] = {}
    for key, val in (variables or {}).items():
        if not key.isidentifier() or key.startswith("_"):
            raise NumericError(f"Недопустимое имя переменной: {key!r}")
        names[key] = parse_decimal(val)

    def repl(m: re.Match[str]) -> str:
        key = f"_n{len(names)}"
        names[key] = Decimal(m.group())
        return key

    prepared = _LITERAL_RE.sub(repl, expr)
    try:
        tree = ast.parse(prepared, mode="eval")
    except SyntaxError as exc:
        raise NumericError(f"Синтаксическая ошибка в выражении: {expr!r}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise NumericError(f"Недопустимая конструкция в выражении: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in names and node.id not in _FUNCTIONS:
            raise NumericError(f"Неизвестное имя: {node.id}")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS or node.keywords):
            raise NumericError("Разрешены только вызовы встроенных числовых функций без именованных аргументов.")

    evaluator = SimpleEval(names=names, functions=_FUNCTIONS, operators=_OPERATORS)
    try:
        with localcontext() as ctx:
            ctx.prec = 34
            result = evaluator.eval(prepared)
    except (ArithmeticError, InvalidOperation) as exc:
        raise NumericError(f"Арифметическая ошибка: {exc.__class__.__name__}") from exc
    except Exception as exc:  # noqa: BLE001 — ошибки simpleeval
        raise NumericError(f"Выражение не вычислено: {exc}") from exc
    if isinstance(result, bool) or not isinstance(result, (Decimal, int)):
        raise NumericError(f"Результат не является числом Decimal: {type(result).__name__}")
    return Decimal(result)


def fragment_key(d: Defect) -> tuple[str | None, str, str]:
    """Нормализованный ключ фрагмента: требование + цитата (+ вторая цитата противоречия)."""
    return (d.req_id, normalize_with_map(d.exact_quote)[0], normalize_with_map(d.related_quote or "")[0])


def deduplicate_by_fragment(defects: Iterable[Defect]) -> list[Defect]:
    """Один и тот же фрагмент засчитывается один раз, даже если его нашли разные правила/проходы.

    Остаётся самая серьёзная находка; при равной серьёзности — детерминированное правило, затем
    подтверждённая Критиком. Коды поглощённых правил дописываются в review_note.
    """
    from .models import FindingSource, ReviewStatus

    def rank(d: Defect) -> tuple[int, int, int]:
        review = {ReviewStatus.DETERMINISTIC: 0, ReviewStatus.CONFIRMED: 1, ReviewStatus.DOWNGRADED: 2,
                  ReviewStatus.NOT_REVIEWED: 3}[d.review_status]
        return d.severity.rank, 0 if d.source == FindingSource.RULES else 1, review

    groups: dict[tuple[str | None, str, str], list[Defect]] = {}
    for d in defects:
        groups.setdefault(fragment_key(d), []).append(d)
    out: list[Defect] = []
    for items in groups.values():
        best = min(items, key=rank)
        others = sorted({x.rule_id for x in items if x is not best and x.rule_id != best.rule_id})
        if others:
            note = (best.review_note + " " if best.review_note else "") + f"Тот же фрагмент также: {', '.join(others)}."
            best = best.model_copy(update={"review_note": note})
        out.append(best)
    return out
