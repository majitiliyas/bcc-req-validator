"""Red-team правки: сегментатор (метаданные, JSON-примеры), NO_MODAL как подсказка, незаполненные чек-листы,
полный контекст документа, отдельная группа Критика и телеметрия пар, привязка цитат из контекста раздела,
план исправлений без смешения P1/P2, чистая вёрстка PDF/HTML, --no-cache."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from req_validator.config import Settings
from req_validator.ingestion import ingest
from req_validator.models import Defect, DefectCategory, FindingSource, ReviewStatus, Severity
from req_validator.reporting import build_view, clean_md, write_reports
from req_validator.router import ChatRequest
from req_validator.rules import check_requirement, check_set
from req_validator.scoring import build_remediation, shorten
from req_validator.segmenter import segment
from test_auditor_pipeline import FakeLLM, _ids, _user, audit

DOC = """# Сервис уведомлений

## Цели

Дать клиенту возможность узнать о превышении бюджета.

## Функциональные требования

* **Ф-1.** Сервис отправляет push при достижении порога.
* **Ф-2.** Система должна хранить порог в тенге.

## REST API

| Поле | Значение |
| :--- | :--- |
| **REST type** | GET |
| **Author** | Школа тестирования |
| **Developer** | Команда практикума |
| **Gitlab project** | Учебный контур |
| **Framework** | Учебный контур |

### 200. Successful — Created

```json
{"result":"ERROR","message":"Проверьте сумму"}
```

* **API-1.** Ответ содержит поле result.

## Ревью требований

| Проверка | Создано или доступно | Проверено |
| :--- | :--- | :--- |
| Требование может быть протестировано | Да / Нет | Да / Нет |
| Поддерживается версионность | Да | Да |
"""


@pytest.fixture
def doc_path(tmp_path: Path) -> Path:
    p = tmp_path / "rt.md"
    p.write_text(DOC, encoding="utf-8")
    return p


@pytest.fixture
def s(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, cache_dir=tmp_path / ".cache", gemini_api_key=SecretStr("g"),
                    nvidia_api_key=SecretStr("n"), groq_api_key=SecretStr("q"), openrouter_api_key=SecretStr("o"),
                    deep_models="gemini/model-a,nvidia/model-b", fast_models="gemini/model-a",
                    critic_models="openrouter/critic-x,nvidia/model-b", enable_ollama=False, use_presidio=False)


# --------------------------------------------------------------------------- #
# Сегментатор и правила
# --------------------------------------------------------------------------- #


def test_admin_rows_and_json_examples_are_context_not_requirements(doc_path: Path) -> None:
    seg = segment(ingest(doc_path))
    texts = " ".join(r.plain_text for r in seg.requirements)
    for admin in ("Author", "Developer", "Gitlab project", "Framework"):
        assert admin not in texts
    assert "REST type" in texts  # содержательные строки контракта остаются требованиями
    assert not any(r.text.lstrip().startswith("{") for r in seg.requirements)
    assert any("строк метаданных — 4, примеров JSON — 1" in w for w in seg.warnings)


def test_no_modal_is_hint_and_skipped_for_goals(doc_path: Path) -> None:
    doc = ingest(doc_path)
    seg = segment(doc)
    f1 = next(r for r in seg.requirements if r.source_id == "Ф-1")
    defects, hints = check_requirement(f1)
    assert not any(d.rule_id == "ISO-525-09/NO_MODAL" for d in defects)
    assert any(h.rule_id == "ISO-525-09/NO_MODAL" for h in hints)
    goal = next(r for r in seg.requirements if "Цели" in r.heading_path)
    assert not any(h.rule_id == "ISO-525-09/NO_MODAL" for h in check_requirement(goal)[1])


def test_unfilled_checklist_detected(doc_path: Path, fixtures: Path) -> None:
    doc = ingest(doc_path)
    found = [d for d in check_set(doc, segment(doc).requirements).defects if d.rule_id == "ISO-526-01/UNFILLED_CHECKLIST"]
    assert len(found) == 1 and "в 1 из 2 строк" in found[0].explanation
    assert found[0].exact_quote.startswith("| Требование может быть протестировано")
    clean = ingest(fixtures / "clean_srs.md")
    assert not [d for d in check_set(clean, segment(clean).requirements).defects if "UNFILLED" in d.rule_id]


# --------------------------------------------------------------------------- #
# LLM: полный контекст, Критик, привязка через контекст раздела
# --------------------------------------------------------------------------- #


def _pass2(req: ChatRequest) -> dict[str, Any]:
    return {"findings": [{
        "kind": "CONTRADICTION", "rule_id": "ISO-526-02", "severity": "MAJOR", "category": "SET_CONSISTENCY",
        "req_id": "REQ-005", "verbatim_quote": "Ответ содержит поле result",
        "related_req_id": None, "related_quote": '{"result":"ERROR","message":"Проверьте сумму"}',
        "explanation": "Успешный раздел 200 содержит пример с ошибкой.", "suggestion": "Вернуть 400."}]}


def _pass1(req: ChatRequest) -> dict[str, Any]:
    ids = _ids(_user(req))
    findings = [{"req_id": i, "rule_id": "ISO-525-07", "severity": "MAJOR", "verbatim_quote": q,
                 "explanation": "x", "suggestion": "y"}
                for i, q in (("REQ-002", "Сервис отправляет push"), ("REQ-003", "хранить порог в тенге")) if i in ids]
    return {"reviewed_req_ids": ids, "findings": findings}


def test_full_context_reaches_pass1_pass2_and_critic(s: Settings, doc_path: Path) -> None:
    fake = FakeLLM(pass1=_pass1, pass2=_pass2)
    audit(s, doc_path, fake)
    for step in ("pass1", "pass2", "critic"):
        text = _user(fake.steps(step)[0][2])
        assert "<full_document_context>" in text and "Учебный контур" in text, step  # метаданные видны в контексте
    assert "уже заданы в другом разделе или соседнем пункте" in _user(fake.steps("pass1")[0][2])


def test_full_context_omitted_for_large_documents(s: Settings, doc_path: Path) -> None:
    fake = FakeLLM(pass1=_pass1, pass2=_pass2)
    report = audit(s.model_copy(update={"full_context_max_bytes": 100}), doc_path, fake)
    assert "<full_document_context>" not in _user(fake.steps("pass1")[0][2])
    assert report.passport.llm.full_context is False


def test_quote_from_json_example_binds_to_section_requirement(s: Settings, doc_path: Path) -> None:
    report = audit(s, doc_path, FakeLLM(pass1=lambda r: {"reviewed_req_ids": [], "findings": []}, pass2=_pass2))
    d = next(x for x in report.defects if x.rule_id == "ISO-526-02/CONTRADICTION")
    assert d.related_quote == '{"result":"ERROR","message":"Проверьте сумму"}' and d.related_quote_verified
    assert d.related_req_ids == ["REQ-005"]
    assert "через контекст раздела «200. Successful — Created» (пример/фрагмент кода" in d.binding_note


def test_critic_uses_separate_chain_and_pairs_telemetry(s: Settings, doc_path: Path) -> None:
    fake = FakeLLM(pass1=_pass1, pass2=_pass2)
    report = audit(s, doc_path, fake)
    critic_deps = {dep for dep, step, _ in fake.calls if step.startswith("critic")}
    assert critic_deps == {"openrouter/critic-x"}  # первая модель группы critic, не генератор
    pairs = report.passport.llm.critic_pairs
    assert pairs == {"gemini/model-a → openrouter/critic-x": report.passport.llm.critic["reviewed"]}
    assert report.passport.llm.critic["same_model"] == 0


def test_critic_avoids_generator_even_if_first_in_critic_chain(s: Settings, doc_path: Path) -> None:
    s2 = s.model_copy(update={"critic_models": "gemini/model-a,nvidia/model-b", "pass2_ensemble": False})
    fake = FakeLLM(pass1=_pass1, pass2=_pass2)
    audit(s2, doc_path, fake)
    assert {dep for dep, step, _ in fake.calls if step.startswith("critic")} == {"nvidia/model-b"}


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #


def _d(i: int, sev: Severity, rule: str, sugg: str) -> Defect:
    return Defect(defect_id=f"DEF-{i:03d}", severity=sev, rule_id=rule, criterion="Полнота",
                  category=DefectCategory.SINGLE_REQ_QUALITY, req_id=f"REQ-{i:03d}", exact_quote="q", explanation="e",
                  suggestion=sugg, source=FindingSource.LLM, review_status=ReviewStatus.CONFIRMED)


def test_remediation_never_mixes_p1_and_p2_and_lists_actions() -> None:
    long = "Описать " + "очень подробно ветку отказа и коды ответа " * 20
    items = build_remediation([_d(1, Severity.MAJOR, "ISO-525-04/LLM", "Добавить ветку отказа."),
                               _d(2, Severity.MINOR, "ISO-525-04/LLM", long),
                               _d(3, Severity.MAJOR, "ISO-525-04/LLM", "Указать коды 4xx.")])
    p1 = [i for i in items if i.priority == "P1"]
    p2 = [i for i in items if i.priority == "P2"]
    assert len(p1) == 1 and p1[0].target_defect_ids == ["DEF-001", "DEF-003"]
    assert len(p2) == 1 and p2[0].target_defect_ids == ["DEF-002"]
    assert p1[0].actions == ["Добавить ветку отказа.", "Указать коды 4xx."] and "|" not in p1[0].action_item
    assert p2[0].actions[0].endswith("…") and len(p2[0].actions[0]) <= 241


def test_shorten_cuts_on_word_boundary() -> None:
    out = shorten("слово " * 100, 50)
    assert out.endswith("…") and out[:-1].split()[-1] == "слово"


def test_clean_md() -> None:
    assert clean_md("| **REST type** | GET |") == "REST type · GET"
    assert clean_md("**Ф-1.** Держатель `card_id` карты") == "Ф-1. Держатель card_id карты"
    assert clean_md("* пункт списка") == "пункт списка"


def test_report_layout_pdf_html_md(s: Settings, doc_path: Path, tmp_path: Path) -> None:
    report = audit(s, doc_path, FakeLLM(pass1=_pass1, pass2=_pass2))
    written, errors = write_reports(report, tmp_path / "out" / "r.md", "md,pdf,html")
    assert errors == []
    md = (tmp_path / "out" / "r.md").read_text(encoding="utf-8")
    html = (tmp_path / "out" / "r.html").read_text(encoding="utf-8")
    import pymupdf

    pdf = "".join(pg.get_text() for pg in pymupdf.open(tmp_path / "out" / "r.pdf"))
    assert "₸" not in pdf and "KZT" in pdf and "〒" not in pdf  # тенге в PDF — текстом
    assert "**" not in re.sub(r"<[^>]+>", "", html.split("<h2>2.")[1])  # цитаты в HTML без сырой разметки
    assert "страниц:" not in md  # для .md поле «страниц» не выводится
    tag = report.passport.git_tag
    assert (f", тег `{tag}`" in md) if tag else (", тег" not in md)  # пустой тег не выводится
    assert "| Деф. | Набор |" in md
    view = build_view(report)
    row = next(r for r in view["matrix"] if r["req_id"] == "REQ-005")
    assert row["n_set"] == 1  # противоречие уровня набора — в отдельной колонке
    assert "справочник" not in md and "CLAUDE" not in md and "§" not in md


def test_cli_no_cache_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, doc_path: Path) -> None:
    from req_validator import cli

    monkeypatch.chdir(tmp_path)
    seen: dict[str, Any] = {}

    def fake_run(path: Path, *, demo: bool, settings: Settings):  # noqa: ANN202
        from req_validator.analyzer import run_audit

        seen["use_cache"] = settings.use_cache
        return run_audit(path, demo=True, settings=settings)

    monkeypatch.setattr(cli, "run_audit", fake_run)
    res = CliRunner().invoke(cli.app, ["analyze", str(doc_path), "-o", str(tmp_path / "r.md"), "--demo", "--no-cache"])
    assert res.exit_code == 0 and seen["use_cache"] is False
