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


# --------------------------------------------------------------------------- #
# Полный LLM-аудит: фоновая задача + опрос прогресса (регрессия «этапы зелёные, а результатов нет»)
# --------------------------------------------------------------------------- #

import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from pydantic import SecretStr  # noqa: E402

import req_validator.web.app as web_app  # noqa: E402
from req_validator.analyzer import run_audit as real_run_audit  # noqa: E402


@pytest.fixture
def llm_client(tmp_path: Path) -> TestClient:
    settings = Settings(_env_file=None, cache_dir=tmp_path / ".cache", gemini_api_key=SecretStr("test-key"),
                        nvidia_api_key=None, groq_api_key=None, openrouter_api_key=None, enable_ollama=False)
    return TestClient(create_app(settings, runs_dir=tmp_path / "runs"))


def _poll(client: TestClient, key: str, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{key}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise AssertionError("задача не завершилась")


def _upload_llm(client: TestClient, name: str = "flawed.md", data: bytes | None = None) -> dict:
    data = (FIXTURES / "flawed_fintech_srs.md").read_bytes() if data is None else data
    return client.post("/api/analyze", files={"file": (name, data, "text/markdown")}, data={"mode": "llm"}).json()


def test_llm_mode_returns_job_immediately_and_result_via_polling(llm_client: TestClient, monkeypatch) -> None:
    gate = threading.Event()

    def slow_audit(path, *, demo, settings):  # реальный пайплайн без сети, но «долгий», как бесплатные API
        assert demo is False
        gate.wait(5)
        return real_run_audit(path, demo=True, settings=settings)

    monkeypatch.setattr(web_app, "run_audit", slow_audit)
    t0 = time.time()
    body = _upload_llm(llm_client)
    assert time.time() - t0 < 3, "POST не должен ждать окончания LLM-аудита"
    assert body["ok"] and "view" not in body and len(body["job"]) == 32
    running = llm_client.get(f"/api/jobs/{body['job']}").json()
    assert running["ok"] and running["status"] in ("queued", "running") and "view" not in running
    assert {"stage", "calls", "note", "elapsed_s", "running_s"} <= running.keys()
    gate.set()
    done = _poll(llm_client, body["job"])
    assert done["ok"] and done["status"] == "done"
    v = done["view"]
    assert v["verdict"]["status"] in ("REJECTED", "CONDITIONAL_PASS", "APPROVED") and v["defects"]
    assert llm_client.get(v["downloads"]["pdf"]).content.startswith(b"%PDF")


def test_second_llm_job_waits_in_visible_queue(llm_client: TestClient, monkeypatch) -> None:
    gate = threading.Event()

    def held_audit(path, *, demo, settings):
        gate.wait(5)
        return real_run_audit(path, demo=True, settings=settings)

    monkeypatch.setattr(web_app, "run_audit", held_audit)
    first = _upload_llm(llm_client)["job"]
    second = _upload_llm(llm_client)["job"]
    time.sleep(0.2)
    s2 = llm_client.get(f"/api/jobs/{second}").json()
    assert s2["status"] == "queued" and s2["ahead"] == 1
    gate.set()
    assert _poll(llm_client, first)["status"] == "done"
    assert _poll(llm_client, second)["status"] == "done"


def test_unreadable_file_in_llm_mode_skips_queue(llm_client: TestClient, monkeypatch) -> None:
    gate = threading.Event()
    calls: list[bool] = []

    def held_audit(path, *, demo, settings):
        calls.append(demo)
        if len(calls) == 1:
            gate.wait(5)
        return real_run_audit(path, demo=True, settings=settings)

    monkeypatch.setattr(web_app, "run_audit", held_audit)
    busy = _upload_llm(llm_client)["job"]  # занимает очередь LLM
    time.sleep(0.1)
    t0 = time.time()
    body = _upload_llm(llm_client, "empty.md", b"")
    assert time.time() - t0 < 3 and body["ok"] and "job" not in body
    assert body["view"]["verdict"]["status"] == "CANNOT_PROCEED"
    gate.set()
    _poll(llm_client, busy)


def test_llm_job_failure_is_reported_not_silent(llm_client: TestClient, monkeypatch) -> None:
    def boom(path, *, demo, settings):
        raise RuntimeError("квота исчерпана")

    monkeypatch.setattr(web_app, "run_audit", boom)
    body = _poll(llm_client, _upload_llm(llm_client)["job"])
    assert body["ok"] is False and body["status"] == "error" and "квота исчерпана" in body["error"]


def test_unknown_job_is_friendly(llm_client: TestClient) -> None:
    for key in ("0" * 32, "not-a-key"):
        body = llm_client.get(f"/api/jobs/{key}").json()
        assert body["ok"] is False and "Задача не найдена" in body["error"]


def test_progress_stage_mapping_from_core_trace() -> None:
    from req_validator.web.app import _stage_of

    assert _stage_of({"event": "ingest"}) == 0
    assert _stage_of({"event": "llm_call", "step": "pass0:0"}) == 1
    assert _stage_of({"event": "llm_call", "step": "pass2:all"}) == 2
    assert _stage_of({"event": "pass2_ensemble"}) == 3
    assert _stage_of({"event": "empty_response", "step": "critic:04b2"}) == 4


def test_frontend_polls_jobs_and_keeps_last_stage_running() -> None:
    js = (Path(web_app.STATIC) / "app.js").read_text(encoding="utf-8")
    assert "/api/jobs/" in js and "pollJob" in js
    assert "if (i < steps.length) anim = setTimeout(tick, stepMs);" in js  # анимация не «закрывает» последний этап


def test_job_progress_sticks_to_its_own_trace(tmp_path: Path) -> None:
    """Регрессия: параллельный CLI или второй сервер с тем же файлом не должен «перехватывать» прогресс задачи."""
    from req_validator.web.app import find_trace

    def write(stamp: str, sha8: str) -> Path:
        f = tmp_path / f"run_{stamp}.jsonl"
        f.write_text(json.dumps({"event": "ingest", "run_id": f"{stamp}-{sha8}"}) + "\n", encoding="utf-8")
        return f

    started = time.mktime(time.strptime("20261127T195353", "%Y%m%dT%H%M%S"))
    write("20261127T193532", "8274dd4e")  # чужой прогон того же файла, начатый раньше задачи
    own = write("20261127T195353", "8274dd4e")
    write("20261127T195400", "aaaaaaaa")  # другой документ
    write("20261127T195736", "8274dd4e")  # чужой прогон того же файла, начатый позже
    assert find_trace(tmp_path, started, "8274dd4e") == own
    assert find_trace(tmp_path, started, "deadbeef") is None


def test_frontend_regressions_from_crash_test() -> None:
    static = Path(web_app.STATIC)
    css = (static / "app.css").read_text(encoding="utf-8")
    js = (static / "app.js").read_text(encoding="utf-8")
    # .results {display:flex} перебивал атрибут hidden: во время нового прогона висел старый отчёт
    assert "[hidden] { display: none !important; }" in css
    # счётчики KPI и индекс не должны застревать на нуле, если ответ пришёл во фоновой вкладке
    assert "document.hidden" in js and "settle(" in js
    # чистое демо-ТЗ без дефектов не должно получать реплику про «пару замечаний»
    assert "правила не нашли ни одного дефекта" in js
