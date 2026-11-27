"""Веб-интерфейс DAIEK: главная страница, кейс v1.0.2, загрузка, плохие файлы → CANNOT_PROCEED, скачивание отчётов."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from req_validator.config import Settings
from req_validator.web.app import SAMPLE_FLAWED, create_app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = Settings(_env_file=None, cache_dir=tmp_path / ".cache", gemini_api_key=None, nvidia_api_key=None,
                        groq_api_key=None, openrouter_api_key=None, enable_ollama=False)
    return TestClient(create_app(settings, runs_dir=tmp_path / "runs"))


def test_index_has_team_branding(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Команда «ВУПИПАЙ»" in r.text and "Выпушивай" not in r.text
    assert "Кейс БЦК (отчёт v1.0.2)" in r.text and "Прогнать тестовое ТЗ с багами" in r.text
    for asset in ("/static/app.js", "/static/app.css", "/static/brand/daiek_logo_main.jpg",
                  "/static/bob/02_slide1_bob_intro.jpg", "/static/bob/04_slide3_bob_overwhelmed_tz.jpg",
                  "/static/bob/03_slide2_bob_found_bug.jpg", "/static/bob/06_slide5_bob_selected_daiek.jpg",
                  "/static/bob/07_slide6_bob_happy_savings.jpg"):
        assert client.get(asset).status_code == 200, asset


def test_case_snapshot_matches_release_v102(client: TestClient) -> None:
    body = client.get("/api/case").json()
    assert body["ok"]
    v = body["view"]
    m = v["metrics"]
    assert (m["total_requirements"], m["total_defects"], m["hints_count"], m["readiness_score"]) == (57, 20, 28, 68.5)
    assert (m["fatal_count"], m["major_count"], m["minor_count"]) == (1, 7, 12)
    assert v["verdict"]["status"] == "REJECTED" and v["verdict"]["blocking"] == ["DEF-001"]
    ids = {d["id"]: d for d in v["defects"]}
    assert ids["DEF-001"]["severity"] == "FATAL" and "GET" in ids["DEF-001"]["quote_view"]
    assert "integer" in ids["DEF-006"]["quote_view"] and "string" in ids["DEF-006"]["related_view"]
    assert v["llm"]["critic"] == {"reviewed": 17, "confirmed": 5, "downgraded": 3, "rejected": 9,
                                  "not_reviewed": 0, "same_model": 0}
    funnel = {f["stage"]: f["count"] for f in v["llm"]["funnel"]}
    assert funnel["Кандидаты LLM (Pass 1 + Pass 2)"] == 22 and funnel["Подтверждено цитатой"] == 22
    assert funnel["Отклонено проверкой цитат"] == 0
    roi = {s["key"]: (s["total_hours"], s["money"]) for s in v["roi"]["scenarios"]}
    assert roi == {"conservative": (23.6, 236000.0), "base": (31.4, 314000.0)}
    assert v["passport"]["git_tag"] == "v1.0.2" and v["passport"]["git_commit"].startswith("66340f6")
    for url in v["downloads"].values():
        assert client.get(url).status_code == 200


def test_upload_markdown_demo_and_download_reports(client: TestClient) -> None:
    data = (FIXTURES / "flawed_fintech_srs.md").read_bytes()
    r = client.post("/api/analyze", files={"file": ("flawed.md", data, "text/markdown")}, data={"mode": "demo"})
    body = r.json()
    assert r.status_code == 200 and body["ok"]
    v = body["view"]
    assert v["source"] == "live" and v["passport"]["mode"] == "demo" and v["verdict"]["preliminary"]
    assert v["metrics"]["total_requirements"] > 0 and v["defects"]
    assert all(d["quote_view"] for d in v["defects"])
    assert set(v["downloads"]) == {"md", "html", "pdf"}
    pdf = client.get(v["downloads"]["pdf"])
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    md = client.get(v["downloads"]["md"])
    assert md.status_code == 200 and "Отчёт о проверке требований" in md.text


def test_sample_flawed_button_runs_real_demo(client: TestClient) -> None:
    assert SAMPLE_FLAWED.read_bytes() == (FIXTURES / "flawed_fintech_srs.md").read_bytes()
    body = client.post("/api/sample/flawed").json()
    assert body["ok"] and body["view"]["passport"]["file_name"] == "flawed_fintech_srs.md"
    assert body["view"]["metrics"]["total_defects"] > 0


@pytest.mark.parametrize(("name", "payload", "status"), [
    ("empty.md", b"", "EMPTY_DOCUMENT"),
    ("spaces.txt", b"   \n\t  \n", "EMPTY_DOCUMENT"),
    ("broken.pdf", (FIXTURES / "broken_edge_case.pdf").read_bytes(), None),
    ("garbage.pdf", b"%PDF-1.7\n\x00\x01\x02 not really a pdf", "CORRUPTED_FILE"),
    ("photo.png", b"\x89PNG\r\n\x1a\n\x00\x00", None),
])
def test_bad_files_give_cannot_proceed_not_500(client: TestClient, name: str, payload: bytes, status: str | None) -> None:
    r = client.post("/api/analyze", files={"file": (name, payload, "application/octet-stream")}, data={"mode": "demo"})
    assert r.status_code == 200
    v = r.json()["view"]
    assert v["verdict"]["status"] == "CANNOT_PROCEED"
    assert v["defects"] == [] and v["roi"] is None
    if status:
        assert v["passport"]["ingestion_status"] == status
    assert client.get(v["downloads"]["md"]).status_code == 200  # диагностический отчёт тоже скачивается


def test_llm_mode_without_keys_is_friendly_error(client: TestClient) -> None:
    r = client.post("/api/analyze", files={"file": ("a.md", b"# T\n\n1. The system shall work.", "text/markdown")},
                    data={"mode": "llm"})
    body = r.json()
    assert r.status_code == 200 and body["ok"] is False and "ключ" in body["error"]


def test_missing_file_and_bad_download_paths(client: TestClient) -> None:
    body = client.post("/api/analyze", data={"mode": "demo"}).json()
    assert body["ok"] is False and "Файл не выбран" in body["error"]
    assert client.get("/api/runs/../../etc/report.md").status_code == 404
    assert client.get("/api/runs/" + "0" * 32 + "/report.exe").status_code == 404
    assert client.get("/api/runs/" + "0" * 32 + "/report.pdf").status_code == 404


def test_health_reports_version_and_team(client: TestClient) -> None:
    h = client.get("/api/health").json()
    assert h["ok"] and h["team"] == "Команда «ВУПИПАЙ»" and h["llm_available"] is False
