"""Защита агента: бюджет вызовов/токенов/времени, детектор зацикливания, circuit breaker, JSONL-трасса."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .pii import mask_pii


class GuardError(RuntimeError):
    """Базовая ошибка защитного слоя: пайплайн должен деградировать, а не падать."""


class BudgetExceededError(GuardError):
    pass


class LoopDetectedError(GuardError):
    pass


class CircuitOpenError(GuardError):
    pass


def estimate_tokens(text: str) -> int:
    """Консервативная оценка токенов без токенизатора: латиница ≈ 4 символа/токен, кириллица ≈ 2.2."""
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    other = len(text) - ascii_chars
    return math.ceil(ascii_chars / 4 + other / 2.2) + 4


@dataclass
class AgentBudgetGuard:
    """Жёсткие лимиты на прогон. Проверка — до каждого логического вызова LLM (кэш-попадания бесплатны)."""

    max_steps: int = 80
    max_tokens: int = 600_000
    max_wall_seconds: float = 1200.0
    max_identical_calls: int = 2
    circuit_breaker_threshold: int = 4
    clock: Callable[[], float] = time.monotonic

    steps: int = 0
    tokens_used: int = 0
    consecutive_failures: int = 0
    circuit_open: bool = False
    _started: float | None = None
    _call_hashes: Counter[str] = field(default_factory=Counter)

    def start(self) -> None:
        if self._started is None:
            self._started = self.clock()

    @property
    def elapsed(self) -> float:
        return 0.0 if self._started is None else self.clock() - self._started

    def before_call(self, request_hash: str, estimated_tokens: int = 0) -> None:
        self.start()
        if self.circuit_open:
            raise CircuitOpenError(
                f"Circuit breaker разомкнут: {self.consecutive_failures} неудачных вызовов подряд — LLM-проходы остановлены.")
        if self.steps + 1 > self.max_steps:
            raise BudgetExceededError(f"Превышен лимит шагов LLM: {self.steps + 1} > {self.max_steps}.")
        if self.tokens_used + estimated_tokens > self.max_tokens:
            raise BudgetExceededError(
                f"Превышен лимит токенов: {self.tokens_used} + ~{estimated_tokens} > {self.max_tokens}.")
        if self.elapsed > self.max_wall_seconds:
            raise BudgetExceededError(f"Превышен лимит времени: {self.elapsed:.0f} с > {self.max_wall_seconds:.0f} с.")
        self._call_hashes[request_hash] += 1
        if self._call_hashes[request_hash] > self.max_identical_calls:
            raise LoopDetectedError(
                f"Обнаружено зацикливание: одинаковый запрос отправлен {self._call_hashes[request_hash]} раз.")
        self.steps += 1

    def record_success(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.tokens_used += prompt_tokens + completion_tokens
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.circuit_breaker_threshold:
            self.circuit_open = True

    def snapshot(self) -> dict[str, Any]:
        return {"steps": self.steps, "tokens": self.tokens_used, "elapsed_s": round(self.elapsed, 2),
                "circuit_open": self.circuit_open}


class TraceLogger:
    """Пошаговый JSONL-лог `<dir>/run_<timestamp>.jsonl`. ПДн маскируются; сбой записи не ломает прогон."""

    def __init__(self, directory: Path, run_stamp: str | None = None) -> None:
        stamp = run_stamp or datetime.now().strftime("%Y%m%dT%H%M%S")
        self.path: Path | None = directory / f"run_{stamp}.jsonl"
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.path = None

    def event(self, event: str, /, **data: Any) -> None:
        if self.path is None:
            return
        record = {"ts": datetime.now().astimezone().isoformat(timespec="milliseconds"), "event": event, **data}
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(mask_pii(json.dumps(record, ensure_ascii=False, default=str)) + "\n")
        except OSError:
            pass
