"""Метрики качества и вердикт Definition of Ready (docs/iso_29148_ieee_830.md §7). Только детерминированная математика."""

from __future__ import annotations

from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal

from .models import RoiEstimate, RoiLine
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


def decide_verdict(
    metrics: SummaryMetrics,
    *,
    can_proceed: bool,
    preliminary: bool,
    reason: str = "",
    defects: list[Defect] | None = None,
) -> Verdict:
    """Вердикт DoR. Некомпенсаторный гейт: хотя бы один ПОДТВЕРЖДЁННЫЙ FATAL → REJECTED при любом индексе.

    Подтверждённый FATAL — найденный детерминированным правилом или подтверждённый независимым Критиком.
    Неподтверждённый FATAL (Критик недоступен) не блокирует безусловно, но исключает APPROVED.
    """
    if not can_proceed or metrics.total_requirements == 0:
        return Verdict(
            status=VerdictStatus.CANNOT_PROCEED, can_start_development=False, can_start_testing=False,
            rationale=reason or "Документ не удалось прочитать или в нём нет требований — проверка невозможна.",
            development_note="Разработка невозможна: нет проверяемого документа.",
            testing_note="Тестирование невозможно: нет требований для тест-дизайна.",
            is_preliminary=preliminary, gate="INGESTION",
        )
    fatal = [d for d in (defects or []) if d.severity == Severity.FATAL]
    if defects is None:  # совместимость: без списка дефектов считаем все FATAL подтверждёнными
        confirmed_ids = ["?"] * metrics.fatal_count
        unreviewed_ids: list[str] = []
    else:
        confirmed_ids = [d.defect_id for d in fatal if d.review_status.confirmed]
        unreviewed_ids = [d.defect_id for d in fatal if not d.review_status.confirmed]
    s = metrics.readiness_score
    gate = "—"
    if confirmed_ids:
        status, gate = VerdictStatus.REJECTED, "FATAL_GATE"
        ids = ", ".join(i for i in confirmed_ids if i != "?")
        rationale = (f"Сработал некомпенсаторный FATAL-гейт: подтверждённых критических дефектов — {len(confirmed_ids)}"
                     + (f" ({ids})" if ids else "") + f". Индекс готовности ({s}) этого не компенсирует. "
                     "Документ возвращается аналитику; после исправления нужен повторный аудит.")
        dev, test = False, False
        dev_note = "Начинать разработку нельзя: сначала устранить все FATAL-дефекты и пройти повторную проверку."
        test_note = "Тест-дизайн преждевременен: критерии приёмки изменятся после исправления FATAL-дефектов."
    elif s < 65:
        status, gate = VerdictStatus.REJECTED, "SCORE"
        rationale = f"Индекс готовности {s} < 65: слишком много дефектов MAJOR/MINOR. Требуется исправление и повторный аудит."
        dev, test = False, False
        dev_note = "Начинать разработку нельзя: сначала устранить дефекты P1 и пройти повторную проверку."
        test_note = "Тест-дизайн преждевременен: требования будут существенно изменены."
    elif s < 85 or unreviewed_ids:
        status = VerdictStatus.CONDITIONAL_PASS
        gate = "UNREVIEWED_FATAL" if unreviewed_ids else "SCORE"
        rationale = f"Подтверждённых FATAL-дефектов нет, индекс готовности {s}: допуск с оговорками."
        if unreviewed_ids:
            rationale += (f" Есть FATAL-находки без подтверждения Критиком ({', '.join(unreviewed_ids)}) — "
                          "их нужно проверить вручную до начала разработки затронутых функций.")
        dev, test = True, False
        dev_note = "Можно начинать проектирование и каркас сервисов; MAJOR-дефекты закрыть в текущем спринте."
        test_note = "Тест-дизайн можно начинать; интеграционное тестирование — после закрытия MAJOR-дефектов."
    else:
        status = VerdictStatus.APPROVED
        rationale = f"Подтверждённых FATAL-дефектов нет, индекс готовности {s} ≥ 85: Definition of Ready пройден."
        dev, test = True, True
        dev_note = "Документ можно передавать в разработку."
        test_note = "Документ можно передавать в тест-дизайн и тестирование."
    if preliminary and status == VerdictStatus.APPROVED:
        # Без семантических проверок (необходимость, корректность, противоречия) DoR не подтверждается.
        status = VerdictStatus.CONDITIONAL_PASS
        rationale = (f"По выполненным проверкам FATAL-дефектов нет, индекс готовности {s} ≥ 85, однако семантическая "
                     "проверка выполнена не полностью, поэтому вердикт APPROVED не выдаётся.")
        dev, test = True, False
        dev_note = "Можно начинать проектирование; финальный допуск — после полного (LLM) аудита."
        test_note = "Тест-дизайн можно начинать; приёмочные критерии подтвердить после полного аудита."
    if preliminary:
        rationale += (" Вердикт предварительный: семантические LLM-проверки выполнены не полностью "
                      "(демо-режим или сбой проходов) — непроверенные критерии отмечены в отчёте.")
    return Verdict(status=status, can_start_development=dev, can_start_testing=test, rationale=rationale,
                   development_note=dev_note, testing_note=test_note, is_preliminary=preliminary,
                   blocking_defect_ids=[i for i in confirmed_ids if i != "?"], gate=gate)


def assess_requirements(
    requirements: list[Requirement], defects: list[Defect], hints: list[Hint], *,
    llm_checked_ids: set[str] | None = None,
) -> list[RequirementAssessment]:
    llm_checked_ids = llm_checked_ids or set()
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
        llm_checked = req.req_id in llm_checked_ids
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
            elif chints and not llm_checked:
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
    """P0: каждый FATAL — отдельный пункт; P1/P2: группы по правилу, самые массовые первыми."""
    items: list[RemediationItem] = []
    for d in defects:
        if d.severity == Severity.FATAL:
            where = d.req_id or "весь документ"
            items.append(RemediationItem(priority="P0", action_item=f"[{d.rule_id}] {where}: {d.suggestion}",
                                         target_defect_ids=[d.defect_id], business_impact=_IMPACT[Severity.FATAL]))
    groups: dict[str, list[Defect]] = defaultdict(list)
    for d in defects:
        if d.severity != Severity.FATAL:
            groups[d.rule_id].append(d)
    ordered = sorted(groups.items(), key=lambda kv: (min(d.severity.rank for d in kv[1]), -len(kv[1]), kv[0]))
    for rule_id, ds in ordered:
        sev = min((d.severity for d in ds), key=lambda s: s.rank)
        pattern = PATTERNS_BY_CODE.get(rule_id)
        action = _SET_ACTIONS.get(rule_id) or (pattern.suggestion if pattern else "")
        if not action:  # LLM-находки: конкретные рекомендации из самих дефектов
            uniq = list(dict.fromkeys(d.suggestion.strip() for d in ds if d.suggestion.strip()))
            action = f"Критерий «{ds[0].criterion}»: " + " | ".join(u[:160] for u in uniq[:3])
            if len(uniq) > 3:
                action += f" | … и ещё {len(uniq) - 3}"
        reqs = sorted({d.req_id for d in ds if d.req_id})
        where = f" Затронуто требований: {len(reqs)} ({', '.join(reqs[:8])}{'…' if len(reqs) > 8 else ''})." if reqs else ""
        items.append(RemediationItem(priority=sev.priority, action_item=f"[{rule_id}] {action}{where}",
                                     target_defect_ids=[d.defect_id for d in ds], business_impact=_IMPACT[sev]))
    return items


def _dec(value: float | int) -> Decimal:
    return Decimal(str(value))


def _fmt(value: Decimal) -> str:
    """Decimal без лишних нулей и с пробелами-разделителями: 10000.0 → «10 000», 6.0 → «6», 0.5 → «0.5»."""
    v = value.normalize()
    if v == v.to_integral_value():
        return f"{int(v):,}".replace(",", " ")
    return format(v, "f")


def compute_roi(metrics: SummaryMetrics, settings: object, *, llm_tokens: tuple[int, int] | None = None) -> RoiEstimate:
    """Прозрачная оценка ROI на Decimal (без LLM).

    Часы = N_req × минуты_ревью / 60 + Σ дефекты × часы_rework(уровень); деньги = часы × ставка.
    Коэффициенты rework 16/6/1 ч — docs/iso_29148_ieee_830.md §7.2 (IBM SSI / NIST); ставка и минуты ревью —
    настраиваемые допущения (ROI_* в .env). Учитываются все дефекты отчёта (прошедшие проверку цитат).
    """
    g = lambda name: _dec(getattr(settings, name))  # noqa: E731
    f = lambda name: _fmt(g(name))  # noqa: E731
    hour = Decimal("0.1")
    review = (_dec(metrics.total_requirements) * g("roi_review_minutes_per_req") / Decimal(60)).quantize(hour)
    fatal = (_dec(metrics.fatal_count) * g("roi_rework_hours_fatal")).quantize(hour)
    major = (_dec(metrics.major_count) * g("roi_rework_hours_major")).quantize(hour)
    minor = (_dec(metrics.minor_count) * g("roi_rework_hours_minor")).quantize(hour)
    total = review + fatal + major + minor
    rate = g("roi_hourly_rate")
    money = (total * rate).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    cur = getattr(settings, "roi_currency", "₸")
    lines = [
        RoiLine(item="Автоматизированное ревью вместо ручного",
                formula=f"{metrics.total_requirements} треб. × {f('roi_review_minutes_per_req')} мин / 60",
                hours=float(review)),
        RoiLine(item="Предотвращённый rework: FATAL",
                formula=f"{metrics.fatal_count} × {f('roi_rework_hours_fatal')} ч", hours=float(fatal)),
        RoiLine(item="Предотвращённый rework: MAJOR",
                formula=f"{metrics.major_count} × {f('roi_rework_hours_major')} ч", hours=float(major)),
        RoiLine(item="Предотвращённый rework: MINOR",
                formula=f"{metrics.minor_count} × {f('roi_rework_hours_minor')} ч", hours=float(minor)),
    ]
    llm_cost = "0 (бесплатные тарифы API и локальная модель)"
    if llm_tokens is not None:
        llm_cost += (f"; токенов в этом прогоне: {llm_tokens[0]} на вход / {llm_tokens[1]} на выход "
                     "(шаги из кэша не тратят квоту)")
    return RoiEstimate(
        lines=lines, total_hours=float(total), hourly_rate=float(rate), currency=cur, total_money=float(money),
        assumptions=[
            f"Ручное ревью одного требования — {f('roi_review_minutes_per_req')} мин (ROI_REVIEW_MINUTES_PER_REQ).",
            f"Rework при позднем обнаружении: FATAL {f('roi_rework_hours_fatal')} ч, MAJOR {f('roi_rework_hours_major')} ч, "
            f"MINOR {f('roi_rework_hours_minor')} ч — по §7.2 справочника (IBM SSI / NIST: исправление на тестировании "
            "в 5–15 раз дороже, чем в требованиях).",
            f"Стоимость часа команды — {_fmt(rate)} {cur} (ROI_HOURLY_RATE, допущение).",
            "Учитываются все дефекты отчёта: каждый подтверждён цитатой, FATAL/MAJOR от LLM — ещё и Критиком.",
            "Оценка верхняя: часть дефектов могла бы быть найдена и при ручном ревью.",
        ],
        llm_cost=llm_cost,
    )
