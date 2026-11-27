"""Переключение моделей при пустом ответе / битом JSON, блочные режимы Pass 0 и Pass 2,
negative examples и страховка Pass 1, дедупликация по фрагменту, экранирование `$` в Markdown."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from req_validator.config import Settings
from req_validator.grounding import deduplicate_by_fragment
from req_validator.models import Defect, DefectCategory, FindingSource, ReviewStatus, Severity
from req_validator.router import ChatRequest, ChatResponse, Deployment, is_empty_response
from test_auditor_pipeline import DOC, FakeLLM, _ids, _user, audit, default_pass0, default_pass2

BROKEN = '{"terms": [{"term": "лимит", "meaning": "значение "в кавычках" без экранирования"}]}'


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None, cache_dir=tmp_path / ".cache", gemini_api_key=SecretStr("g"),
        nvidia_api_key=SecretStr("n"), groq_api_key=SecretStr("q"), openrouter_api_key=None,
        deep_models="gemini/model-a,nvidia/model-b", fast_models="groq/model-fast", enable_ollama=False,
        use_presidio=False, llm_concurrency=2,
    )


@pytest.fixture
def doc_path(tmp_path: Path) -> Path:
    p = tmp_path / "limits.md"
    p.write_text(DOC, encoding="utf-8")
    return p


class PerModel(FakeLLM):
    """FakeLLM, где обработчик получает и деплоймент: handlers[kind](dep, req)."""

    def __init__(self, **by_model: Any) -> None:
        super().__init__()
        self.by_model = by_model

    async def __call__(self, dep: Deployment, req: ChatRequest) -> ChatResponse:
        kind = req.step.split(":")[0]
        if kind in self.by_model:
            self.calls.append((dep.name, req.step, req))
            payload = self.by_model[kind](dep, req)
            text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
            tokens = 3 if not text.strip() else 80
            return ChatResponse(text=text, prompt_tokens=100, completion_tokens=tokens, model=dep.name)
        return await super().__call__(dep, req)


def _pass(report: Any, prefix: str) -> Any:
    return next(p for p in report.passport.llm.passes if p.name.startswith(prefix))


# --------------------------------------------------------------------------- #
# Пустой ответ и битый JSON → следующая модель цепочки
# --------------------------------------------------------------------------- #


def test_is_empty_response() -> None:
    def r(text: str, tokens: int) -> ChatResponse:
        return ChatResponse(text=text, prompt_tokens=1, completion_tokens=tokens, model="m")

    assert is_empty_response(r("", 0)) and is_empty_response(r("   ", 3)) and is_empty_response(r("ok", 1))
    assert is_empty_response(r("<think>…</think>", 4))
    assert not is_empty_response(r('{"findings": []}', 5))  # короткий, но валидный пустой результат
    assert not is_empty_response(r('{"a": "очень длинный ответ без закрывающей скобки"', 40))  # это уже «битый JSON»


def test_empty_response_switches_model_without_retrying_same(settings: Settings, doc_path: Path) -> None:
    fake = PerModel(pass2=lambda dep, req: "" if dep.name == "gemini/model-a" else default_pass2(req))
    report = audit(settings, doc_path, fake)
    p2 = _pass(report, "Pass 2")
    assert p2.status == "ok" and p2.models == ["nvidia/model-b"]
    pass2_calls = [dep for dep, step, _ in fake.calls if step.startswith("pass2")]
    assert pass2_calls == ["gemini/model-a", "nvidia/model-b"]  # без повторов пустой модели
    assert any(d.rule_id == "ISO-526-02/CONTRADICTION" for d in report.defects)


def test_broken_json_exhausts_retries_then_next_model(settings: Settings, doc_path: Path) -> None:
    fake = PerModel(pass0=lambda dep, req: BROKEN if dep.name == "gemini/model-a" else default_pass0(req))
    report = audit(settings, doc_path, fake)
    p0 = _pass(report, "Pass 0")
    assert p0.status == "ok" and p0.models == ["nvidia/model-b"]
    calls = [dep for dep, step, _ in fake.calls if step.startswith("pass0")]
    assert calls == ["gemini/model-a"] * 3 + ["nvidia/model-b"]  # 1 + 2 ретрая валидации, затем смена модели
    retry = [req for dep, step, req in fake.calls if step.startswith("pass0")][1]
    assert "не прошёл валидацию" in retry.messages[-1]["content"] and "экранируй" in retry.messages[-1]["content"]
    fresh = [req for dep, step, req in fake.calls if step.startswith("pass0")][3]
    assert len(fresh.messages) == 2  # новая модель начинает с чистого диалога


def test_pass0_block_mode_when_all_models_break_on_full_input(settings: Settings, doc_path: Path) -> None:
    def pass0(dep: Deployment, req: ChatRequest) -> Any:
        return BROKEN if len(_ids(_user(req))) > 2 else default_pass0(req)

    fake = PerModel(pass0=pass0)
    report = audit(settings, doc_path, fake)
    p0 = _pass(report, "Pass 0")
    assert p0.status == "ok" and "блочный режим" in p0.detail
    block_steps = {step for _, step, _ in fake.calls if step.startswith("pass0:0.")}
    assert block_steps == {"pass0:0.0", "pass0:0.1"}
    assert report.passport.llm.registry_size["numeric_facts"] >= 1  # цитаты и ID требований сохранены


def test_pass2_block_mode_keeps_cross_section_contradiction(settings: Settings, doc_path: Path) -> None:
    def pass2(dep: Deployment, req: ChatRequest) -> Any:
        if "весь документ целиком" in _user(req):
            return ""  # полный проход: пустой ответ у всех моделей
        ids = _ids(_user(req))
        if "REQ-001" in ids:  # блок «Функциональные требования», противоречие с другим разделом через реестр
            return default_pass2(req)
        return {"findings": []}

    fake = PerModel(pass2=pass2)
    report = audit(settings, doc_path, fake)
    p2 = _pass(report, "Pass 2")
    assert p2.status == "ok" and "блочный режим" in p2.detail and p2.items == 2
    blocks = [req for _, step, req in fake.calls if step.startswith("pass2:block")]
    assert blocks and all("Глобальный реестр" in _user(r) and "блок" in _user(r) for r in blocks)
    c = next(d for d in report.defects if d.rule_id == "ISO-526-02/CONTRADICTION")
    assert c.related_req_ids == ["REQ-004"] and c.related_quote_verified
    assert not report.verdict.is_preliminary  # блочный режим — полноценный проход


def test_pass2_failed_when_every_block_fails(settings: Settings, doc_path: Path) -> None:
    fake = PerModel(pass2=lambda dep, req: "")
    report = audit(settings, doc_path, fake)
    assert _pass(report, "Pass 2").status == "failed" and report.verdict.is_preliminary


def test_pass2_prompt_has_generic_contract_checks(settings: Settings, doc_path: Path) -> None:
    fake = FakeLLM()
    audit(settings, doc_path, fake)
    text = _user(fake.steps("pass2")[0][2])
    assert "семантика HTTP-метода" in text and "коды ответов против тела" in text
    assert "ОБЛАСТЬ ПРОВЕРКИ: весь документ целиком" in text


# --------------------------------------------------------------------------- #
# Pass 1: negative examples + страховка от придирок к схемам-стрелкам
# --------------------------------------------------------------------------- #

ARROW_DOC = DOC + """
## Архитектура решения

Мобильное приложение → сервис настроек → хранилище лимитов.
"""


def test_pass1_negative_examples_and_arrow_guard(settings: Settings, tmp_path: Path) -> None:
    p = tmp_path / "arrows.md"
    p.write_text(ARROW_DOC, encoding="utf-8")

    def pass1(req: ChatRequest) -> dict[str, Any]:
        ids = _ids(_user(req))
        findings = []
        if "REQ-005" in ids:
            findings.append({"req_id": "REQ-005", "rule_id": "ISO-525-04", "severity": "MAJOR",
                             "verbatim_quote": "Мобильное приложение → сервис настроек",
                             "explanation": "Нет триггера и ветки ошибки.", "suggestion": "Описать."})
        return {"reviewed_req_ids": ids, "findings": findings}

    fake = FakeLLM(pass1=pass1, pass2=lambda r: {"findings": []})
    report = audit(settings, p, fake)
    prompt = _user(fake.steps("pass1")[0][2])
    assert "NEGATIVE EXAMPLES" in prompt and "схемы со стрелками" in prompt
    assert not any(d.req_id == "REQ-005" and d.source == FindingSource.LLM for d in report.defects)
    assert "снято придирок к схемам-стрелкам: 1" in _pass(report, "Pass 1").detail


# --------------------------------------------------------------------------- #
# Дедупликация по фрагменту
# --------------------------------------------------------------------------- #


def _d(rule: str, sev: Severity, src: FindingSource, quote: str = "заполняются тестировщиками",
       review: ReviewStatus = ReviewStatus.DETERMINISTIC) -> Defect:
    return Defect(defect_id="X", severity=sev, rule_id=rule, criterion="c", category=DefectCategory.SINGLE_REQ_QUALITY,
                  req_id="REQ-018", exact_quote=quote, explanation="e", suggestion="s", source=src, review_status=review)


def test_same_fragment_counted_once() -> None:
    rule = _d("ISO-525-04/PLACEHOLDER", Severity.MAJOR, FindingSource.RULES)
    llm = _d("ISO-525-03/LLM", Severity.MAJOR, FindingSource.LLM, quote="Заполняются  тестировщиками",
             review=ReviewStatus.CONFIRMED)
    other = _d("ISO-525-07/LLM", Severity.MINOR, FindingSource.LLM, quote="другой фрагмент требования")
    out = deduplicate_by_fragment([llm, rule, other])
    assert len(out) == 2
    kept = next(d for d in out if d.exact_quote.lower().startswith("заполняются"))
    assert kept.rule_id == "ISO-525-04/PLACEHOLDER"  # при равной серьёзности — детерминированное правило
    assert "ISO-525-03/LLM" in kept.review_note


def test_fragment_dedup_keeps_most_severe() -> None:
    a = _d("ISO-525-04/PLACEHOLDER", Severity.MAJOR, FindingSource.RULES)
    b = _d("ISO-526-01/LLM", Severity.FATAL, FindingSource.LLM, review=ReviewStatus.CONFIRMED)
    out = deduplicate_by_fragment([a, b])
    assert len(out) == 1 and out[0].severity == Severity.FATAL


def test_contradictions_with_different_second_quote_not_merged() -> None:
    a = _d("ISO-526-02/CONTRADICTION", Severity.MAJOR, FindingSource.LLM).model_copy(update={"related_quote": "x 1"})
    b = _d("ISO-526-02/CONTRADICTION", Severity.MAJOR, FindingSource.LLM).model_copy(update={"related_quote": "y 2"})
    assert len(deduplicate_by_fragment([a, b])) == 2


# --------------------------------------------------------------------------- #
# Вёрстка: `$` и маршрутизация
# --------------------------------------------------------------------------- #


def test_markdown_escapes_dollar_and_splits_routes(settings: Settings, tmp_path: Path) -> None:
    from req_validator.reporting import render_html, write_reports

    p = tmp_path / "curl.md"
    p.write_text(DOC + '\n## Интеграция\n\n* **API-01.** Система должна вызывать `curl -G "$BASE/limits" '
                 '-H "Authorization: Bearer $TOKEN"` при запросе.\n', encoding="utf-8")
    report = audit(settings, p, FakeLLM())
    written, errors = write_reports(report, tmp_path / "out" / "r.md", "md,pdf,html")
    assert errors == []
    md = (tmp_path / "out" / "r.md").read_text(encoding="utf-8")
    assert "\\$BASE" in md and "\\$TOKEN" in md
    assert not re.search(r"(?<!\\)\$", md)  # ни одного неэкранированного $
    assert "Маршрут группы `deep`" in md and "Маршрут группы `fast`" in md
    assert "| Проход | Статус | Модели |" in md
    assert "$BASE" in render_html(report) and "\\$" not in render_html(report)
    import pymupdf

    pdf_text = "".join(page.get_text() for page in pymupdf.open(tmp_path / "out" / "r.pdf"))
    assert "$BASE" in pdf_text and "\\$" not in pdf_text  # Typst выводит строку как текст
