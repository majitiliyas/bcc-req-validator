"""Контентно-адресуемый дисковый кэш шагов LLM в SQLite (`.cache/agent_checkpoints.db`).

Ключ = sha256(step_name + canonical_input + prompt_hash + schema_hash). Модель в ключ намеренно
не входит: если результат шага уже валидирован, повторный прогон (даже на другом провайдере после
fallback) берёт его из кэша без расхода квот. В кэш попадают только ответы, прошедшие Pydantic-валидацию.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """Каноническая сериализация: сортировка ключей, без пробелов, UTF-8 как есть."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedStep:
    key: str
    step_name: str
    payload: Any
    model: str
    created_at: str


class DiskStepCache:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS steps (
        key TEXT PRIMARY KEY,
        step_name TEXT NOT NULL,
        prompt_hash TEXT NOT NULL,
        schema_hash TEXT NOT NULL,
        model TEXT NOT NULL,
        payload TEXT NOT NULL,
        prompt_tokens INTEGER DEFAULT 0,
        completion_tokens INTEGER DEFAULT 0,
        created_at TEXT NOT NULL
    )"""

    def __init__(self, path: Path | str, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(self._SCHEMA)
            self._conn.commit()

    @staticmethod
    def make_key(step_name: str, canonical_input: str, prompt_hash: str, schema_hash: str) -> str:
        return sha256_hex("\x1f".join((step_name, canonical_input, prompt_hash, schema_hash)))

    def get(self, key: str) -> CachedStep | None:
        if self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT key, step_name, payload, model, created_at FROM steps WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return CachedStep(key=row[0], step_name=row[1], payload=json.loads(row[2]), model=row[3], created_at=row[4])

    def put(self, key: str, *, step_name: str, prompt_hash: str, schema_hash: str, model: str, payload: Any,
            prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        if self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO steps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (key, step_name, prompt_hash, schema_hash, model, canonical_json(payload), prompt_tokens,
                 completion_tokens, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            self._conn.commit()

    def count(self) -> int:
        if self._conn is None:
            return 0
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0])

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
