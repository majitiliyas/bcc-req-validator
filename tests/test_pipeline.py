"""Пайплайн целиком: правила (recall/precision на golden-фикстурах), отчёт во всех форматах, CLI, роутер."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from req_validator.analyzer import mask_pii, run_audit
from req_validator.cli import app
from req_validator.models import Severity, VerdictStatus
from req_validator.reporting import render_html, render_markdown, write_reports

REQUIRED_SECTIONS = ("Паспорт прогона", "1. Нарезка требований", "2. Проверка каждого требования",
                     "3. Проверка набора требований целиком", "4. Сводка", "5. Итог")


def test_flawed_fixture_recall(fixtures: Path, settings) -> None:
    report = run_audit(fixtures / "flawed_fintech_srs.md", demo=True, settings=settings)
    rules = {d.rule_id for d in report.defects}
    # Дефекты, которые обязан ловить детерминированный слой.
    expected = {"ISO-526-01/TBD", "ISO-525-03/OPEN_LIST", "ISO-525-07/UNMEASURABLE", "ISO-525-03/WEAK_MODALITY",
                "ISO-525-03/BOUNDARY", "ISO-526-01/EMPTY_CELL", "ISO-526-06/NO_IDS"}
    recall = len(expected & rules) / len(expected)
    assert recall >= 0.9, f"пропущены: {expected - rules}"
    assert report.verdict.status == VerdictStatus.REJECTED  # есть FATAL (TBD)
    assert all(d.quote_verified for d in report.defects)
    raw = Path(fixtures / "flawed_fintech_srs.md").read_text(encoding="utf-8")
    for d in report.defects:
        assert raw[d.quote_start:d.quote_end] == d.exact_quote


def test_clean_fixture_has_no_serious_defects(fixtures: Path, settings) -> None:
    report = run_audit(fixtures / "clean_srs.md", demo=True, settings=settings)
    serious = [d for d in report.defects if d.severity in (Severity.FATAL, Severity.MAJOR)]
    assert serious == [], [(d.rule_id, d.exact_quote) for d in serious]
    assert report.metrics.readiness_score >= 85
    # в демо-режиме APPROVED не выдаётся: семантика не проверялась
    assert report.verdict.status == VerdictStatus.CONDITIONAL_PASS and report.verdict.is_preliminary


def test_defect_ids_sorted_by_severity(fixtures: Path, settings) -> None:
    report = run_audit(fixtures / "flawed_fintech_srs.md", demo=True, settings=settings)
    ranks = [d.severity.rank for d in report.defects]
    assert ranks == sorted(ranks)
    assert [d.defect_id for d in report.defects] == [f"DEF-{i:03d}" for i in range(1, len(ranks) + 1)]


def test_metrics_formulas(fixtures: Path, settings) -> None:
    r = run_audit(fixtures / "flawed_fintech_srs.md", demo=True, settings=settings)
    m = r.metrics
    n = m.total_requirements
    expected = max(0.0, 100 - (m.fatal_count * 15 + m.major_count / n * 100 + m.minor_count / n * 20))
    assert abs(m.readiness_score - round(expected, 1)) < 0.06
    assert m.estimated_saved_hours == m.fatal_count * 16 + m.major_count * 6 + m.minor_count


def test_report_contains_all_sections(fixtures: Path, settings) -> None:
    report = run_audit(fixtures / "flawed_fintech_srs.md", demo=True, settings=settings)
    md = render_markdown(report)
    for section in REQUIRED_SECTIONS:
        assert section in md
    assert "REQ-001" in md and "ID в документе" in md and "SHA-256" in md
    assert "[FATAL]" in md
    html = render_html(report)
    assert "<html" in html and "Паспорт прогона" in html


def test_write_all_formats(fixtures: Path, settings, tmp_path: Path) -> None:
    report = run_audit(fixtures / "tables_lists.md", demo=True, settings=settings)
    written, errors = write_reports(report, tmp_path / "out" / "report.md", "all")
    assert errors == []
    assert {p.suffix for p in written} == {".md", ".html", ".pdf", ".json"}
    assert (tmp_path / "out" / "report.pdf").read_bytes().startswith(b"%PDF")
    data = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert data["passport"]["mode"] == "demo"


def test_cannot_proceed_report_for_broken_input(fixtures: Path, settings, tmp_path: Path) -> None:
    report = run_audit(fixtures / "broken_edge_case.pdf", demo=True, settings=settings)
    assert report.verdict.status == VerdictStatus.CANNOT_PROCEED
    assert report.metrics.total_requirements == 0
    written, errors = write_reports(report, tmp_path / "broken.md", "both")
    assert errors == [] and len(written) == 2
    assert "CANNOT PROCEED" in (tmp_path / "broken.md").read_text(encoding="utf-8")


def test_cli_demo_run(fixtures: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "report.md"
    result = CliRunner().invoke(app, ["analyze", str(fixtures / "flawed_fintech_srs.md"), "--out", str(out), "--demo"])
    assert result.exit_code == 0, result.output
    assert out.exists() and "Отчёт о проверке требований" in out.read_text(encoding="utf-8")


def test_cli_empty_file_still_writes_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    result = CliRunner().invoke(app, ["analyze", str(empty), "--out", str(tmp_path / "r.md"), "--demo"])
    assert result.exit_code == 0
    assert "CANNOT_PROCEED" in result.output


def test_cli_requires_key_without_demo(fixtures: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    result = CliRunner().invoke(app, ["analyze", str(fixtures / "clean_srs.md"), "--out", str(tmp_path / "r.md")])
    assert result.exit_code == 2
    assert "--demo" in result.output


def test_cli_rejects_unknown_format(fixtures: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["analyze", str(fixtures / "clean_srs.md"), "-o", str(tmp_path / "r.md"),
                                      "-f", "docx", "--demo"])
    assert result.exit_code == 2


def test_sdk_retries_disabled(settings) -> None:
    from req_validator.router import make_gemini_client, make_openai_compatible_client

    client = make_openai_compatible_client("groq", "test-key", settings)
    assert client.max_retries == 0
    gem = make_gemini_client("test-key", settings)
    assert gem is not None


def test_mask_pii() -> None:
    masked = mask_pii("карта 4276 1234 5678 9012, ключ AIzaSyA1234567890abcdefghijkl")
    assert "4276" not in masked and "AIza" not in masked
