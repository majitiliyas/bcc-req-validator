"""Локальное маскирование ПДн и чувствительных реквизитов перед отправкой во внешние LLM API.

Бесплатные тарифы LLM могут логировать промпты и обучаться на них, поэтому всё, что уходит
наружу, проходит через `PiiMasker.mask()`. Псевдонимы обратимы (`unmask()`) и детерминированы:
одинаковый вход в одинаковом порядке → одинаковые псевдонимы, поэтому ключи кэша стабильны.

Детекторы детерминированные (regex + контрольные суммы: Luhn для PAN, ИИН/БИН РК, mod-97 для IBAN),
чтобы не маскировать суммы и лимиты. Если установлен `presidio-analyzer`, дополнительно маскируются
PERSON/LOCATION; без него модуль работает только на правилах.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

PLACEHOLDER_RE = re.compile(r"<<([A-Z_]+)_(\d+)>>")


def luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def kz_iin_ok(digits: str) -> bool:
    """Контрольная сумма ИИН/БИН Республики Казахстан (12 цифр)."""
    if len(digits) != 12 or not digits.isdigit():
        return False
    d = [int(c) for c in digits]
    s = sum(d[i] * (i + 1) for i in range(11)) % 11
    if s == 10:
        w = [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]
        s = sum(d[i] * w[i] for i in range(11)) % 11
        if s == 10:
            return False
    return s == d[11]


def iban_ok(value: str) -> bool:
    v = value.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", v):
        return False
    rearranged = v[4:] + v[:4]
    num = "".join(str(int(c, 36)) for c in rearranged)
    return int(num) % 97 == 1


@dataclass(frozen=True)
class Detector:
    label: str
    regex: re.Pattern[str]
    validate: Callable[[str], bool] | None = None


DETECTORS: tuple[Detector, ...] = (
    Detector("SECRET", re.compile(r"\b(?:AIza[\w-]{20,}|gsk_\w{20,}|sk-(?:or-)?[\w-]{20,}|nvapi-[\w-]{20,}|"
                                  r"eyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,})")),
    Detector("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    Detector("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"), iban_ok),
    Detector("PAN", re.compile(r"(?<!\d)(?<!\d[.,])\d{4}(?:[ -]?\d{4}){2}[ -]?\d{1,7}(?!\d|[.,]\d)"),
             lambda s: 13 <= len(re.sub(r"\D", "", s)) <= 19 and luhn_ok(re.sub(r"\D", "", s))),
    Detector("IIN", re.compile(r"(?<!\d)(?<!\d[.,])\d{12}(?!\d|[.,]\d)"), kz_iin_ok),
    Detector("PHONE", re.compile(r"(?<![\w+])(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)")),
)


def _presidio_spans(text: str) -> list[tuple[int, int, str]]:
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 — presidio опционален
        return []
    try:
        engine = _presidio_engine(AnalyzerEngine)
        results = engine.analyze(text=text, language="ru", entities=["PERSON", "LOCATION"])
        return [(r.start, r.end, r.entity_type) for r in results if r.score >= 0.6]
    except Exception:  # noqa: BLE001
        return []


_ENGINE: Any = None


def _presidio_engine(cls: Any) -> Any:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = cls()
    return _ENGINE


def presidio_available() -> bool:
    try:
        import presidio_analyzer  # type: ignore[import-not-found]  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@dataclass
class PiiMasker:
    """Обратимое детерминированное маскирование: «4276 1234 5678 9012» → «<<PAN_1>>»."""

    use_presidio: bool = True
    forward: dict[str, str] = field(default_factory=dict)  # оригинал → псевдоним
    backward: dict[str, str] = field(default_factory=dict)  # псевдоним → оригинал
    counters: dict[str, int] = field(default_factory=dict)

    def _alias(self, label: str, original: str) -> str:
        if original in self.forward:
            return self.forward[original]
        self.counters[label] = self.counters.get(label, 0) + 1
        alias = f"<<{label}_{self.counters[label]}>>"
        self.forward[original] = alias
        self.backward[alias] = original
        return alias

    def spans(self, text: str) -> list[tuple[int, int, str]]:
        found: list[tuple[int, int, str]] = []
        for det in DETECTORS:
            for m in det.regex.finditer(text):
                if det.validate is None or det.validate(m.group()):
                    found.append((m.start(), m.end(), det.label))
        if self.use_presidio:
            found += _presidio_spans(text)
        found.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        merged: list[tuple[int, int, str]] = []
        for s in found:
            if merged and s[0] < merged[-1][1]:
                continue  # пересечение — оставляем более ранний/длинный
            merged.append(s)
        return merged

    def mask(self, text: str) -> str:
        if not text:
            return text
        out, pos = [], 0
        for s, e, label in self.spans(text):
            out.append(text[pos:s])
            out.append(self._alias(label, text[s:e]))
            pos = e
        out.append(text[pos:])
        return "".join(out)

    def unmask(self, text: str) -> str:
        if not text or "<<" not in text:
            return text
        return PLACEHOLDER_RE.sub(lambda m: self.backward.get(m.group(0), m.group(0)), text)

    def unmask_obj(self, obj: Any) -> Any:
        """Рекурсивно вернуть оригиналы во всех строках dict/list."""
        if isinstance(obj, str):
            return self.unmask(obj)
        if isinstance(obj, list):
            return [self.unmask_obj(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.unmask_obj(v) for k, v in obj.items()}
        return obj

    @property
    def stats(self) -> dict[str, int]:
        return dict(self.counters)


_LONG_DIGITS = re.compile(r"(?<!\d)(?<!\d[.,])\d(?:[ -]?\d){11,18}(?!\d|[.,]\d)")


def mask_pii(text: str) -> str:
    """Необратимое маскирование для логов и трасс (агрессивнее: любые 12–19-значные номера)."""
    return _LONG_DIGITS.sub("[NUM]", PiiMasker(use_presidio=False).mask(text))
