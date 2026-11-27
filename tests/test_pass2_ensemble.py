"""Ансамбль Pass 2: две разные модели, объединение находок до Grounding Gate, graceful fallback,
независимость Критика от обеих моделей, раздельный кэш участников."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from req_validator.auditor import merge_pass2
from req_validator.config import Settings
from req_validator.models import Defect, DefectCategory, FindingSource, ReviewStatus, Severity
from req_validator.router import ChatRequest, ChatResponse, Deployment, ProviderError
from test_auditor_pipeline import DOC, FakeLLM, audit, default_pass2

# --------------------------------------------------------------------------- #
# merge_pass2 (юнит)
# --------------------------------------------------------------------------- #


def _d(model: str, rule: str, req: str, quote: str, sev: Severity = Severity.MAJOR, *, rel: str | None = None,
       rel_q: str | None = None) -> Defect:
    return Defect(defect_id="L", severity=sev, rule_id=rule, criterion="c", category=DefectCategory.SET_CONSISTENCY,
                  req_id=req, exact_quote=quote, explanation="e", suggestion="s", source=FindingSource.LLM,
                  review_status=ReviewStatus.NOT_REVIEWED, model=model, pass_name="pass2",
                  related_req_ids=[rel] if rel else [], related_quote=rel_q)


def test_same_contradiction_from_both_models_merged_by_req_pair() -> None:
    a = [_d("m/a", "ISO-526-02/CONTRADICTION", "REQ-001", "лимит 100 000", rel="REQ-004", rel_q="150 000")]
    b = [_d("m/b", "ISO-526-02/CONTRADICTION", "REQ-004", "составляет 150 000 ₸", Severity.FATAL,
            rel="REQ-001", rel_q="суточный лимит 100 000 ₸")]  # другая формулировка, обратный порядок пары
    merged, agreed = merge_pass2(a, b)
    assert len(merged) == 1 and agreed == 1
    assert merged[0].severity == Severity.FATAL  # остаётся более серьёзная оценка
    assert merged[0].agreement == ["m/a", "m/b"]


def test_unique_findings_of_each_model_are_kept() -> None:
    a = [_d("m/a", "ISO-526-01/MISSING_SCENARIO", "REQ-002", "отправлять push")]
    b = [_d("m/b", "ISO-526-02/CONTRADICTION", "REQ-003", "GET", rel="REQ-002", rel_q="Сохранение"),
         _d("m/b", "ISO-526-01/MISSING_SCENARIO", "REQ-002", "отправлять push-уведомление")]  # вложенная цитата
    merged, agreed = merge_pass2(a, b)
    assert len(merged) == 2 and agreed == 1
    assert {tuple(d.agreement) for d in merged} == {("m/a", "m/b"), ("m/b",)}


def test_duplicates_inside_one_model_are_not_merged() -> None:
    a = [_d("m/a", "ISO-526-01/OTHER", "REQ-001", "Если результат 200"),
         _d("m/a", "ISO-526-01/OTHER", "REQ-001", "Если результат 200 или 201")]
    merged, agreed = merge_pass2(a, [])
    assert len(merged) == 2 and agreed == 0


def test_different_criteria_on_same_pair_not_merged() -> None:
    a = [_d("m/a", "ISO-526-02/CONTRADICTION", "REQ-001", "x", rel="REQ-002", rel_q="y")]
    b = [_d("m/b", "ISO-526-04/SCOPE", "REQ-001", "x", rel="REQ-002", rel_q="y")]
    assert len(merge_pass2(a, b)[0]) == 2


# --------------------------------------------------------------------------- #
# Сквозные сценарии
# --------------------------------------------------------------------------- #


@pytest.fixture
def doc_path(tmp_path: Path) -> Path:
    p = tmp_path / "limits.md"
    p.write_text(DOC, encoding="utf-8")
    return p


def _settings(tmp_path: Path, **kw: Any) -> Settings:
    base = dict(_env_file=None, cache_dir=tmp_path / ".cache", gemini_api_key=SecretStr("g"),
                nvidia_api_key=SecretStr("n"), groq_api_key=None, openrouter_api_key=None,
                deep_models="gemini/model-a,gemini/model-c,nvidia/model-b", fast_models="gemini/model-a",
                critic_models="gemini/model-a,nvidia/model-b,gemini/model-c", enable_ollama=False, use_presidio=False)
    base.update(kw)
    return Settings(**base)


def _unique_b(req: ChatRequest) -> dict[str, Any]:
    out = default_pass2(req)
    out["findings"].append({
        "kind": "MISSING_SCENARIO", "rule_id": "ISO-526-01", "severity": "MAJOR", "category": "SET_CONSISTENCY",
        "req_id": "REQ-003", "verbatim_quote": "записывать в журнал операцию по карте",
        "explanation": "Не описано маскирование номера карты в журнале.", "suggestion": "Маскировать PAN (6+4)."})
    return out


class TwoModels(FakeLLM):
    """Pass 2: первая модель (gemini/*) — default_pass2, вторая (nvidia/*) — default + уникальная находка."""

    def __init__(self, *, second_fails: bool = False) -> None:
        super().__init__()
        self.second_fails = second_fails

    async def __call__(self, dep: Deployment, req: ChatRequest) -> ChatResponse:
        if req.step.startswith("pass2") and dep.provider == "nvidia":
            self.calls.append((dep.name, req.step, req))
            if self.second_fails:
                raise ProviderError("server", "503", status=503)
            import json

            return ChatResponse(text=json.dumps(_unique_b(req), ensure_ascii=False), prompt_tokens=100,
                                completion_tokens=50, model=dep.name)
        if req.step.startswith("pass2") and dep.name != "gemini/model-a" and self.second_fails:
            self.calls.append((dep.name, req.step, req))
            raise ProviderError("server", "503", status=503)
        return await super().__call__(dep, req)


def _pass2_status(report: Any) -> Any:
    return next(p for p in report.passport.llm.passes if p.name.startswith("Pass 2"))


def test_ensemble_uses_other_family_and_merges(tmp_path: Path, doc_path: Path) -> None:
    fake = TwoModels()
    report = audit(_settings(tmp_path), doc_path, fake)
    p2 = [dep for dep, step, _ in fake.calls if step.startswith("pass2")]
    assert p2 == ["gemini/model-a", "nvidia/model-b"]  # вторая — другого семейства, а не gemini/model-c
    st = _pass2_status(report)
    assert st.status == "ok" and st.models == ["gemini/model-a", "nvidia/model-b"]
    assert "ансамбль 2 моделей" in st.detail and "совпало у обеих: 1" in st.detail
    contra = next(d for d in report.defects if d.rule_id == "ISO-526-02/CONTRADICTION")
    assert contra.agreement == ["gemini/model-a", "nvidia/model-b"]
    uniq = next(d for d in report.defects if d.rule_id == "ISO-526-01/MISSING_SCENARIO" and d.req_id == "REQ-003")
    assert uniq.agreement == ["nvidia/model-b"]  # уникальная находка второй модели сохранена
    assert not report.verdict.is_preliminary


def test_critic_independent_of_both_ensemble_models(tmp_path: Path, doc_path: Path) -> None:
    fake = TwoModels()
    report = audit(_settings(tmp_path), doc_path, fake)
    critic = {dep for dep, step, _ in fake.calls if step.startswith("critic")}
    assert critic == {"gemini/model-c"}  # ни model-a, ни model-b (обе — генераторы Pass 2)
    assert report.passport.llm.critic["same_model"] == 0


def test_graceful_fallback_when_second_model_fails(tmp_path: Path, doc_path: Path) -> None:
    fake = TwoModels(second_fails=True)
    report = audit(_settings(tmp_path), doc_path, fake)
    st = _pass2_status(report)
    assert st.status == "ok" and st.models == ["gemini/model-a"]
    assert "вторая модель недоступна" in st.detail and "использован результат gemini/model-a" in st.detail
    assert any(d.rule_id == "ISO-526-02/CONTRADICTION" for d in report.defects)
    assert not report.verdict.is_preliminary  # проход выполнен, отказ второй модели не делает вердикт предварительным


def test_ensemble_can_be_disabled(tmp_path: Path, doc_path: Path) -> None:
    fake = TwoModels()
    report = audit(_settings(tmp_path, pass2_ensemble=False), doc_path, fake)
    assert [dep for dep, step, _ in fake.calls if step.startswith("pass2")] == ["gemini/model-a"]
    assert "ансамбль" not in _pass2_status(report).detail


def test_ensemble_members_cached_separately(tmp_path: Path, doc_path: Path) -> None:
    s = _settings(tmp_path)
    audit(s, doc_path, TwoModels())
    con = sqlite3.connect(s.cache_db)
    models = sorted(m for (m,) in con.execute("SELECT model FROM steps WHERE step_name LIKE 'pass2%'"))
    assert models == ["gemini/model-a", "nvidia/model-b"]
    again = TwoModels()
    r2 = audit(s, doc_path, again)
    assert again.calls == [] and _pass2_status(r2).models == ["gemini/model-a", "nvidia/model-b"]
