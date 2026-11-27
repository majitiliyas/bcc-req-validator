"""Pydantic-схемы данных пайплайна: инжест, требования, дефекты, отчёт.

Все границы данных (вход от парсеров, будущие ответы LLM, отчёт) валидируются здесь.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


class IngestionStatus(StrEnum):
    OK = "OK"
    PARTIAL_OCR = "PARTIAL_OCR"  # часть страниц распознана через OCR
    NO_TEXT_LAYER = "NO_TEXT_LAYER"  # скан без текстового слоя, OCR недоступен/не дал текста
    EMPTY_DOCUMENT = "EMPTY_DOCUMENT"  # файл пуст или содержит только пробелы
    CORRUPTED_FILE = "CORRUPTED_FILE"  # битый PDF / бинарный мусор
    UNSUPPORTED = "UNSUPPORTED"  # формат не поддерживается
    FILE_NOT_FOUND = "FILE_NOT_FOUND"

    @property
    def can_proceed(self) -> bool:
        return self in (IngestionStatus.OK, IngestionStatus.PARTIAL_OCR)


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"


class TableRow(BaseModel):
    cells: list[str] = Field(description="Текст ячеек строки без markdown-разметки")
    raw: str = Field(description="Исходная строка таблицы как в документе")
    start_char: int = Field(description="Смещение начала строки в raw_markdown")
    end_char: int = Field(description="Смещение конца строки (не включительно)")


class DocumentBlock(BaseModel):
    block_id: str = Field(description="Стабильный идентификатор блока B-0001…")
    block_type: BlockType
    heading_path: str = Field(description="Путь заголовков через ' > '")
    heading_titles: list[str] = Field(default_factory=list, description="Заголовки пути списком")
    text: str = Field(description="Дословный фрагмент raw_markdown[start_char:end_char]")
    start_char: int
    end_char: int
    page_num: int | None = Field(default=None, description="Номер страницы (1-based) для PDF")
    level: int | None = Field(default=None, description="Уровень заголовка или глубина вложенности списка")
    list_marker: str | None = Field(default=None, description="Маркер пункта: '*', '-', '1.' …")
    list_intro: str | None = Field(default=None, description="Вводная фраза перед списком (для List Unrolling)")
    code_lang: str | None = None
    table_header: list[str] = Field(default_factory=list)
    table_rows: list[TableRow] = Field(default_factory=list)


class IngestionResult(BaseModel):
    status: IngestionStatus
    file_name: str
    file_path: str
    file_size_bytes: int = 0
    sha256: str = ""
    source_format: str = Field(default="unknown", description="md | txt | pdf | unknown")
    detected_encoding: str = ""
    page_count: int | None = None
    ocr_pages: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list, description="Человекочитаемая диагностика для паспорта")
    title: str = ""
    raw_markdown: str = ""
    blocks: list[DocumentBlock] = Field(default_factory=list)

    @property
    def diagnostics(self) -> list[str]:  # совместимость с CLAUDE.md
        return self.warnings


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


class RequirementKind(StrEnum):
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    DATA = "data"
    INTEGRATION = "integration"
    INTERFACE = "interface"  # API-контракт, параметры, статусы
    ACCEPTANCE = "acceptance"
    BUSINESS = "business"  # цели, бизнес-процесс
    DESIGN = "design"  # архитектура, UI, схемы
    TESTING = "testing"
    OTHER = "other"


class Requirement(BaseModel):
    req_id: str = Field(description="ID в отчёте: REQ-001…")
    source_id: str | None = Field(default=None, description="ID в документе (Ф-1, REQ-7, п. 3), если есть")
    has_explicit_id: bool = Field(description="True, если ID взят из документа, а не присвоен агентом")
    kind: RequirementKind
    unit_type: BlockType = Field(description="Из какого блока получено: абзац, пункт списка, строка таблицы, код")
    heading_path: str
    text: str = Field(description="Дословная цитата = raw_markdown[start_char:end_char]")
    plain_text: str = Field(description="Текст без markdown-разметки (для правил и LLM)")
    context: str | None = Field(default=None, description="Вводная фраза списка или заголовки таблицы")
    start_char: int
    end_char: int
    page_num: int | None = None
    block_id: str
    fingerprint: str = Field(description="SHA-1 нормализованного текста (трассировка между версиями)")

    @property
    def short(self) -> str:
        t = " ".join(self.plain_text.split())
        return t if len(t) <= 160 else t[:157].rstrip() + "…"


class SegmentationResult(BaseModel):
    requirements: list[Requirement] = Field(default_factory=list)
    informative_block_ids: list[str] = Field(default_factory=list, description="Блоки-контекст, не требования")
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Findings / Defects
# --------------------------------------------------------------------------- #


class Severity(StrEnum):
    FATAL = "FATAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"

    @property
    def priority(self) -> str:
        return {"FATAL": "P0", "MAJOR": "P1", "MINOR": "P2"}[self.value]

    @property
    def rank(self) -> int:
        return {"FATAL": 0, "MAJOR": 1, "MINOR": 2}[self.value]


class DefectCategory(StrEnum):
    SINGLE_REQ_QUALITY = "SINGLE_REQ_QUALITY"
    SET_CONSISTENCY = "SET_CONSISTENCY"
    FINTECH_TRANSACTIONS = "FINTECH_TRANSACTIONS"
    FINTECH_MATH_LIMITS = "FINTECH_MATH_LIMITS"
    FINTECH_SECURITY_PII = "FINTECH_SECURITY_PII"
    STATE_MACHINE_GRAPH = "STATE_MACHINE_GRAPH"


class FindingSource(StrEnum):
    RULES = "rules"  # детерминированные правила
    LLM = "llm"


class Defect(BaseModel):
    defect_id: str = Field(description="DEF-001…, присваивается детерминированно")
    severity: Severity
    rule_id: str = Field(description="ID правила, напр. ISO-525-03/WEAK_MODALITY")
    criterion: str = Field(description="Нарушенное свойство по стандарту, напр. 'Однозначность'")
    category: DefectCategory
    req_id: str | None = Field(default=None, description="REQ-xxx; None для дефектов уровня набора")
    related_req_ids: list[str] = Field(default_factory=list)
    section_path: str = ""
    exact_quote: str = Field(description="Дословная цитата из документа")
    quote_verified: bool = False
    quote_start: int | None = None
    quote_end: int | None = None
    explanation: str
    suggestion: str
    risk: str = Field(default="", description="Тип риска для бизнеса/разработки/QA")
    source: FindingSource = FindingSource.RULES

    @property
    def priority(self) -> str:
        return self.severity.priority


class Hint(BaseModel):
    """Низкоточная подсказка правил: не дефект, не влияет на метрики, требует экспертной/LLM-проверки."""

    rule_id: str
    req_id: str
    exact_quote: str
    message: str


class CheckStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    NOT_CHECKED = "NOT_CHECKED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CriterionCheck(BaseModel):
    rule_id: str
    criterion: str
    status: CheckStatus
    comment: str = ""
    defect_ids: list[str] = Field(default_factory=list)


class RequirementAssessment(BaseModel):
    req_id: str
    checks: list[CriterionCheck]
    defect_ids: list[str] = Field(default_factory=list)
    hints: list[Hint] = Field(default_factory=list)


class SetAssessment(BaseModel):
    checks: list[CriterionCheck]
    defect_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


class VerdictStatus(StrEnum):
    APPROVED = "APPROVED"
    CONDITIONAL_PASS = "CONDITIONAL_PASS"
    REJECTED = "REJECTED"
    CANNOT_PROCEED = "CANNOT_PROCEED"


class RunPassport(BaseModel):
    document_title: str
    file_name: str
    file_path: str
    file_size_bytes: int
    sha256: str
    source_format: str
    detected_encoding: str
    page_count: int | None
    run_id: str
    run_started_at: datetime
    duration_s: float
    mode: str = Field(description="demo | llm")
    model: str
    agent_version: str
    git_commit: str
    git_tag: str | None
    ingestion_status: IngestionStatus
    ingestion_warnings: list[str] = Field(default_factory=list)
    pipeline_notes: list[str] = Field(default_factory=list)


class SummaryMetrics(BaseModel):
    total_requirements: int
    requirements_with_explicit_id: int
    requirements_with_defects: int
    fatal_count: int
    major_count: int
    minor_count: int
    total_defects: int
    hints_count: int
    defect_density: float
    readiness_score: float = Field(ge=0, le=100)
    estimated_saved_hours: float


class RemediationItem(BaseModel):
    priority: str = Field(description="P0 | P1 | P2")
    action_item: str
    target_defect_ids: list[str]
    business_impact: str


class Verdict(BaseModel):
    status: VerdictStatus
    can_start_development: bool
    can_start_testing: bool
    rationale: str
    development_note: str
    testing_note: str
    is_preliminary: bool = Field(default=False, description="True в демо-режиме: семантика LLM не проверялась")


class AuditReport(BaseModel):
    passport: RunPassport
    requirements: list[Requirement]
    assessments: list[RequirementAssessment]
    set_assessment: SetAssessment
    defects: list[Defect]
    metrics: SummaryMetrics
    verdict: Verdict
    remediation_plan: list[RemediationItem]

    def defect(self, defect_id: str) -> Defect | None:
        return next((d for d in self.defects if d.defect_id == defect_id), None)
