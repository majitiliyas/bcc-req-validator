"""Метрики качества и вердикт Definition of Ready (docs/iso_29148_ieee_830.md §7). Только детерминированная математика."""

from __future__ import annotations

from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal

from .models import (
    CheckStatus,
    CriterionCheck,
    Defect,
    Hint,
    RemediationItem,
    Requirement,
    RequirementAssessment,
    SetAssessment,
    Severity,
    SummaryMetrics,
    Verdict,
    VerdictStatus,
)
from .rules import PATTERNS_BY_CODE
from .standards import SET_CRITERIA, SINGLE_CRITERIA
from .models import BlockType


def _r(value: Decimal, places: int = 2) -> float:
    return float(value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def compute_metrics(requirements: list[Requirement], defects: list[Defect], hints_count: int) -> SummaryMetrics:
    n = len(requirements)
    f = sum(d.severity == Severity.FATAL for d in defects)
    ma = sum(d.severity == Severity.MAJOR for d in defects)
    mi = sum(d.severity == Severity.MINOR for d in defects)
    if n:
        nd = Decimal(n)
        dd = (Decimal(f * 10 + ma * 3 + mi)) / nd
        score = Decimal(100) - (Decimal(f * 15) + Decimal(ma) / nd * 100 + Decimal(mi) / nd * 20)
        score = max(Decimal(0), min(Decimal(100), score))
    else:
        dd, score = Decimal(0), Decimal(0)
    return SummaryMetrics(
        total_requirements=n,
        requirements_with_explicit_id=sum(r.has_explicit_id for r in requirements),
        requirements_with_defects=len({d.req_id for d in defects if d.req_id}),
        fatal_count=f, major_count=ma, minor_count=mi, total_defects=len(defects), hints_count=hints_count,
        defect_density=_r(dd), readiness_score=_r(score, 1),
        estimated_saved_hours=float(f * 16 + ma * 6 + mi * 1),
    )


def decide_verdict(metrics: SummaryMetrics, *, can_proceed: bool, preliminary: bool, reason: str = "") -> Verdict:
    if not can_proceed or metrics.total_requirements == 0:
        return Verdict(
            status=VerdictStatus.CANNOT_PROCEED, can_start_development=False, can_start_testing=False,
            rationale=reason or "Документ не удалось прочитать или в нём нет требований — проверка невозможна.",
            development_note="Разработка невозможна: нет проверяемого документа.",
            testing_note="Тестирование невозможно: нет требований для тест-дизайна.",
            is_preliminary=preliminary,
        )
    s, f = metrics.readiness_score, metrics.fatal_count
    if f >= 1 or s < 65:
        status = VerdictStatus.REJECTED
        why = []
        if f:
            why.append(f"найдено критических дефектов (FATAL): {f}")
        if s < 65:
            why.append(f"индекс готовности {s} < 65")
        rationale = "Документ не готов: " + "; ".join(why) + ". Требуется исправление и повторный аудит."
        dev, test = False, False
        dev_note = "Начинать разработку нельзя: сначала устранить все FATAL-дефекты и пройти повторную проверку."
        test_note = "Тест-дизайн по документу преждевременен: критерии приёмки изменятся после исправлений."
    elif s < 85:
        status = VerdictStatus.CONDITIONAL_PASS
        rationale = f"FATAL-дефектов нет, индекс готовности {s} (65–85): допуск с оговорками."
        dev, test = True, False
        dev_note = "Можно начинать проектирование и каркас сервисов; MAJOR-дефекты закрыть в текущем спринте."
        test_note = "Тест-дизайн можно начинать; интеграционное тестирование — после закрытия MAJOR-дефектов."
    else:
        status = VerdictStatus.APPROVED
        rationale = f"FATAL-дефектов нет, индекс готовности {s} ≥ 85: Definition of Ready пройден."
        dev, test = True, True
        dev_note = "Документ можно передавать в разработку."
        test_note = "Документ можно передавать в тест-дизайн и тестирование."
    if preliminary and status == VerdictStatus.APPROVED:
        # Без семантических проверок (необходимость, корректность, противоречия) DoR не подтверждается.
        status = VerdictStatus.CONDITIONAL_PASS
        rationale = (f"По детерминированным правилам FATAL-дефектов нет, индекс готовности {s} ≥ 85, однако без "
                     "семантической проверки вердикт APPROVED не выдаётся.")
        dev, test = True, False
        dev_note = "Можно начинать проектирование; финальный допуск — после полного (LLM) аудита."
        test_note = "Тест-дизайн можно начинать; приёмочные критерии подтвердить после полного аудита."
    if preliminary:
        rationale += (" Вердикт предварительный: в демо-режиме выполнены только детерминированные правила; "
                      "семантические проверки (противоречия, пропуски, корректность) требуют LLM-прохода.")
    return Verdict(status=status, can_start_development=dev, can_start_testing=test, rationale=rationale,
                   development_note=dev_note, testing_note=test_note, is_preliminary=preliminary)


def assess_requirements(
    requirements: list[Requirement], defects: list[Defect], hints: list[Hint], *, llm_checked: bool
) -> list[RequirementAssessment]:
    by_req: dict[str, list[Defect]] = defaultdict(list)
    for d in defects:
        if d.req_id:
            by_req[d.req_id].append(d)
    hints_by_req: dict[str, list[Hint]] = defaultdict(list)
    for h in hints:
        hints_by_req[h.req_id].append(h)

    out: list[RequirementAssessment] = []
    for req in requirements:
        rdefs = by_req.get(req.req_id, [])
        rhints = hints_by_req.get(req.req_id, [])
        checks: list[CriterionCheck] = []
        for c in SINGLE_CRITERIA:
            cdefs = [d for d in rdefs if d.rule_id.split("/")[0] == c.rule_id]
            chints = [h for h in rhints if h.rule_id.split("/")[0] == c.rule_id]
            if c.rule_id == "ISO-525-09" and req.unit_type in (BlockType.TABLE, BlockType.CODE):
                checks.append(CriterionCheck(rule_id=c.rule_id, criterion=c.name_ru, status=CheckStatus.NOT_APPLICABLE,
                                             comment="Строка таблицы/фрагмент кода — синтаксис EARS не применяется."))
                continue
            if cdefs:
                status, comment = CheckStatus.FAIL, "; ".join(f"[{d.severity}] {d.rule_id}" for d in cdefs)
            elif chints:
                status, comment = CheckStatus.WARN, "; ".join(sorted({h.message for h in chints}))
            elif c.rules_coverage or llm_checked:
                status, comment = CheckStatus.PASS, "Нарушений не выявлено" + ("" if llm_checked else " (правила)")
            else:
                status, comment = CheckStatus.NOT_CHECKED, "Требует семантической проверки (LLM-проход)."
            checks.append(CriterionCheck(rule_id=c.rule_id, criterion=c.name_ru, status=status, comment=comment,
                                         defect_ids=[d.defect_id for d in cdefs]))
        out.append(RequirementAssessment(req_id=req.req_id, checks=checks, defect_ids=[d.defect_id for d in rdefs], hints=rhints))
    return out


def assess_set(defects: list[Defect], notes: dict[str, str], *, llm_checked: bool) -> SetAssessment:
    checks: list[CriterionCheck] = []
    for c in SET_CRITERIA:
        related = [d for d in defects if d.rule_id.split("/")[0] == c.rule_id]
        if c.rule_id == "ISO-526-01":  # полнота набора учитывает и локальные заглушки/пропуски значений
            related += [d for d in defects if d.rule_id in ("ISO-525-04/PLACEHOLDER", "ISO-525-04/MISSING_VALUE") and d not in related]
        if related:
            status = CheckStatus.FAIL
            comment = f"Нарушений: {len(related)} — " + ", ".join(sorted({d.rule_id for d in related}))
        elif c.rule_id in notes:
            status, comment = CheckStatus.WARN, notes[c.rule_id]
        elif c.rules_coverage or llm_checked:
            status, comment = CheckStatus.PASS, "Нарушений не выявлено" + ("" if llm_checked else " (правила)")
        else:
            status, comment = CheckStatus.NOT_CHECKED, "Требует семантической проверки (LLM-проход)."
        checks.append(CriterionCheck(rule_id=c.rule_id, criterion=c.name_ru, status=status, comment=comment,
                                     defect_ids=[d.defect_id for d in related]))
    return SetAssessment(checks=checks, defect_ids=[d.defect_id for d in defects if d.req_id is None])


_IMPACT = {
    Severity.FATAL: "P0 Blocker: без исправления разработка невозможна; высокий риск финансовых и регуляторных потерь.",
    Severity.MAJOR: "P1 High: риск разночтений и переделок на этапе тестирования.",
    Severity.MINOR: "P2 Medium: усложняет сопровождение ТЗ и написание тест-кейсов.",
}
_SET_ACTIONS = {
    "ISO-526-06/NO_IDS": "Присвоить всем нормативным пунктам уникальные неизменяемые ID.",
    "ISO-526-06/DUP_ID": "Устранить дублирующиеся идентификаторы требований.",
    "ISO-526-05/NO_VERSION": "Зафиксировать версию, дату и историю изменений документа.",
    "ISO-526-07/NO_PRIORITY": "Добавить приоритизацию требований (MoSCoW / MVP).",
    "ISO-526-01/EMPTY_TABLE": "Заполнить пустые таблицы (согласующие) или удалить их.",
    "ISO-526-01/EMPTY_CELL": "Убрать прочерки в таблицах или добавить легенду значений.",
    "ISO-525-03/BOUNDARY": "Уточнить включительность всех числовых границ.",
}


def build_remediation(defects: list[Defect]) -> list[RemediationItem]:
    groups: dict[str, list[Defect]] = defaultdict(list)
    for d in defects:
        groups[d.rule_id].append(d)
    ordered = sorted(groups.items(), key=lambda kv: (min(d.severity.rank for d in kv[1]), -len(kv[1]), kv[0]))
    items: list[RemediationItem] = []
    for rule_id, ds in ordered:
        sev = min((d.severity for d in ds), key=lambda s: s.rank)
        pattern = PATTERNS_BY_CODE.get(rule_id)
        action = _SET_ACTIONS.get(rule_id) or (pattern.suggestion if pattern else ds[0].suggestion)
        reqs = sorted({d.req_id for d in ds if d.req_id})
        where = f" Затронуто требований: {len(reqs)} ({', '.join(reqs[:8])}{'…' if len(reqs) > 8 else ''})." if reqs else ""
        items.append(RemediationItem(priority=sev.priority, action_item=f"[{rule_id}] {action}{where}",
                                     target_defect_ids=[d.defect_id for d in ds], business_impact=_IMPACT[sev]))
    return items
