"""Строгие схемы ответов LLM (Pydantic v2). Из них же генерируется JSON Schema для `response_format`."""

from __future__ import annotations

import copy
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .cache import canonical_json, sha256_hex

SeverityLit = Literal["FATAL", "MAJOR", "MINOR"]
SingleRule = Literal["ISO-525-01", "ISO-525-02", "ISO-525-03", "ISO-525-04", "ISO-525-05", "ISO-525-06",
                     "ISO-525-07", "ISO-525-08", "ISO-525-09"]
SetRule = Literal["ISO-526-01", "ISO-526-02", "ISO-526-03", "ISO-526-04", "ISO-526-05", "ISO-526-06", "ISO-526-07"]
CategoryLit = Literal["SINGLE_REQ_QUALITY", "SET_CONSISTENCY", "FINTECH_TRANSACTIONS", "FINTECH_MATH_LIMITS",
                      "FINTECH_SECURITY_PII", "STATE_MACHINE_GRAPH"]
MAX_QUOTE_WORDS = 25


def _clip_words(value: str | None, limit: int = MAX_QUOTE_WORDS) -> str | None:
    """Обрезать цитату до N слов. Префикс дословной цитаты остаётся дословным — grounding не страдает."""
    if value is None:
        return None
    words = value.strip().split()
    return " ".join(words[:limit]) if len(words) > limit else value.strip()


# --------------------------------------------------------------------------- #
# Pass 0 — Global Outline & Entity Registry
# --------------------------------------------------------------------------- #


class RegistryTerm(BaseModel):
    term: str = Field(description="Термин или сущность, как в документе")
    meaning: str = Field(description="Смысл по документу (1 фраза); пусто, если определения нет")
    synonyms: list[str] = Field(default_factory=list, description="Другие названия той же сущности в документе")
    req_ids: list[str] = Field(default_factory=list, description="REQ-xxx, где встречается")


class RegistryRole(BaseModel):
    name: str = Field(description="Актор/роль/внешняя система")
    synonyms: list[str] = Field(default_factory=list)
    req_ids: list[str] = Field(default_factory=list)


class RegistryStateModel(BaseModel):
    entity: str = Field(description="Сущность со статусами (заявка, платёж, настройка…)")
    states: list[str] = Field(default_factory=list)
    transitions: list[str] = Field(default_factory=list, description="Переходы в виде 'A -> B: триггер'")
    req_ids: list[str] = Field(default_factory=list)


class NumericFact(BaseModel):
    entity: str = Field(description="К чему относится число (порог, таймаут, лимит, код ответа…)")
    attribute: str = Field(description="min, max, timeout, unit, format, http_status…")
    value: str = Field(description="Значение как в документе")
    unit: str = Field(default="", description="Единица: ₸, тиын, мин, мс, %, символы…")
    condition: str = Field(default="", description="Условие применимости, если есть")
    req_id: str = Field(description="REQ-xxx, откуда взят факт")
    quote: str = Field(description="Дословная цитата (до 25 слов), содержащая значение")

    @field_validator("quote")
    @classmethod
    def clip_quote(cls, v: str | None) -> str | None:
        return _clip_words(v)


class CrossRef(BaseModel):
    req_id: str
    target: str = Field(description="На что ссылается требование (документ, раздел, таблица)")


class GlobalRegistry(BaseModel):
    """Компактная карта документа для борьбы с Cross-Section Cliff."""

    terms: list[RegistryTerm] = Field(default_factory=list)
    roles: list[RegistryRole] = Field(default_factory=list)
    state_models: list[RegistryStateModel] = Field(default_factory=list)
    numeric_facts: list[NumericFact] = Field(default_factory=list)
    cross_refs: list[CrossRef] = Field(default_factory=list)
    external_systems: list[str] = Field(default_factory=list)

    def merge(self, other: GlobalRegistry) -> GlobalRegistry:
        return GlobalRegistry(
            terms=self.terms + other.terms, roles=self.roles + other.roles,
            state_models=self.state_models + other.state_models, numeric_facts=self.numeric_facts + other.numeric_facts,
            cross_refs=self.cross_refs + other.cross_refs,
            external_systems=sorted(set(self.external_systems) | set(other.external_systems)),
        )

    def compact(self, req_ids: set[str] | None = None) -> dict[str, Any]:
        """Сжатое представление для промптов; с req_ids — только относящиеся к батчу числовые факты/статусы."""
        def rel(ids: list[str]) -> bool:
            return req_ids is None or not ids or bool(set(ids) & req_ids)

        return {
            "terms": [f"{t.term}" + (f" (= {', '.join(t.synonyms)})" if t.synonyms else "") for t in self.terms],
            "roles": [r.name + (f" (= {', '.join(r.synonyms)})" if r.synonyms else "") for r in self.roles],
            "state_models": [f"{s.entity}: {'; '.join(s.transitions) or ', '.join(s.states)}"
                             for s in self.state_models if rel(s.req_ids)],
            "numeric_facts": [f"{f.req_id}: {f.entity}.{f.attribute} = {f.value} {f.unit}".strip()
                              + (f" [{f.condition}]" if f.condition else "")
                              for f in self.numeric_facts if req_ids is None or f.req_id in req_ids],
            "cross_refs": [f"{c.req_id} → {c.target}" for c in self.cross_refs if req_ids is None or c.req_id in req_ids],
            "external_systems": self.external_systems,
        }


# --------------------------------------------------------------------------- #
# Pass 1 — Per-Requirement Audit
# --------------------------------------------------------------------------- #


class ReqFinding(BaseModel):
    req_id: str = Field(description="ID проверяемого требования из входа (REQ-xxx)")
    rule_id: SingleRule = Field(description="Нарушенный критерий ISO 29148 п. 5.2.5")
    severity: SeverityLit
    category: CategoryLit = "SINGLE_REQ_QUALITY"
    verbatim_quote: str = Field(description="Дословная цитата из текста этого требования, до 25 слов")
    explanation: str = Field(description="Конкретно: в чём дефект и чем он опасен для разработки/QA")
    suggestion: str = Field(description="Готовая исправленная формулировка или точный вопрос аналитику")

    @field_validator("verbatim_quote")
    @classmethod
    def clip_quote(cls, v: str | None) -> str | None:
        return _clip_words(v)


class Pass1Output(BaseModel):
    reviewed_req_ids: list[str] = Field(description="Все REQ-xxx батча, которые были проверены")
    findings: list[ReqFinding] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Pass 2 — Holistic & Cross-Section Audit
# --------------------------------------------------------------------------- #


class SetFinding(BaseModel):
    kind: Literal["CONTRADICTION", "MISSING_SCENARIO", "TBD", "DESIGN_DECISION", "BOUNDARY", "TERMINOLOGY",
                  "SCOPE", "OTHER"]
    rule_id: SetRule
    severity: SeverityLit
    category: CategoryLit = "SET_CONSISTENCY"
    req_id: str | None = Field(default=None, description="Основное требование-якорь (REQ-xxx) или null")
    verbatim_quote: str = Field(description="Дословная цитата якоря, до 25 слов")
    related_req_id: str | None = Field(default=None, description="Второе требование для CONTRADICTION")
    related_quote: str | None = Field(default=None, description="Дословная цитата второго требования, до 25 слов")
    explanation: str
    suggestion: str

    @field_validator("verbatim_quote", "related_quote")
    @classmethod
    def clip_quote(cls, v: str | None) -> str | None:
        return _clip_words(v)


class Pass2Output(BaseModel):
    findings: list[SetFinding] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Pass 3 — Critic и ремонт цитат
# --------------------------------------------------------------------------- #


class CriticDecision(BaseModel):
    item_id: str = Field(description="ID находки из входа")
    decision: Literal["CONFIRMED", "DOWNGRADED", "REJECTED"]
    new_severity: SeverityLit | None = Field(default=None, description="Только для DOWNGRADED: ниже исходной")
    reason: str = Field(description="Краткое обоснование со ссылкой на норму и цитату")


class CriticOutput(BaseModel):
    decisions: list[CriticDecision] = Field(default_factory=list)


class QuoteRepair(BaseModel):
    verbatim_quote: str | None = Field(description="Точная подстрока текста требования или null, если дефект не подтверждается")

    @field_validator("verbatim_quote")
    @classmethod
    def clip_quote(cls, v: str | None) -> str | None:
        return _clip_words(v)


# --------------------------------------------------------------------------- #
# JSON Schema для API
# --------------------------------------------------------------------------- #


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].split("/")[-1]
            return _inline_refs(copy.deepcopy(defs[name]), defs)
        return {k: _inline_refs(v, defs) for k, v in node.items() if k not in ("title", "default", "$defs")}
    if isinstance(node, list):
        return [_inline_refs(x, defs) for x in node]
    return node


def api_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema без $ref/$defs/title/default — совместимо с Gemini/OpenAI-compatible structured output."""
    raw = model.model_json_schema()
    return _inline_refs(raw, raw.get("$defs", {}))


def schema_hash(model: type[BaseModel]) -> str:
    return sha256_hex(canonical_json(api_schema(model)))[:16]


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json_text(text: str) -> str:
    """Убрать <think>-блоки reasoning-моделей и markdown-ограждения; взять внешний JSON-объект."""
    t = _THINK.sub("", text or "").strip()
    t = _FENCE.sub("", t).strip()
    start, end = t.find("{"), t.rfind("}")
    return t[start:end + 1] if start >= 0 and end > start else t
