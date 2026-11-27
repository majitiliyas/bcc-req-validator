"""Веб-интерфейс DAIEK поверх того же CLI-ядра: `req-validator web`.

Тонкая обёртка: загрузка файла → `run_audit` (тот же пайплайн, что у `analyze`) → `write_reports` → JSON для дашборда.
Никаких HTTP 500 и трейсбеков: плохой вход превращается в отчёт CANNOT_PROCEED, сбой — в понятное сообщение.

Быстрый демо-аудит отвечает синхронно (~1 с). Полный LLM-аудит на бесплатных API идёт минутами, поэтому он
запускается фоновой задачей: страница опрашивает `/api/jobs/{key}` и показывает реальный этап по трейсу прогона,
таймер, число вызовов LLM и место в очереди (LLM-прогоны выполняются по одному, чтобы не сжечь бесплатные квоты).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .. import __version__
from ..analyzer import run_audit
from ..config import Settings, load_settings
from ..ingestion import ingest
from ..models import AuditReport
from ..reporting import write_reports
from .view import DOWNLOAD_FORMATS, report_to_web

HERE = Path(__file__).parent
STATIC = HERE / "static"
SAMPLE_FLAWED = HERE / "samples" / "flawed_fintech_srs.md"
CASE_JSON = STATIC / "case" / "bcc_requirements_v1.0.2.json"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
RUN_KEY = re.compile(r"^[0-9a-f]{32}$")
TEAM = "Команда «ВУПИПАЙ»"


def _safe_name(name: str | None) -> str:
    base = Path(name or "").name.strip() or "document.md"
    base = re.sub(r"[^\w.\-]+", "_", base)[:120].lstrip(".")
    return base or "document.md"


def _fail(message: str, *, hint: str = "") -> JSONResponse:
    return JSONResponse({"ok": False, "error": message, "hint": hint})


@dataclass
class Job:
    """Фоновый полный LLM-аудит одного файла."""

    key: str
    path: Path
    work: Path
    sha8: str
    created: float
    status: str = "queued"  # queued | running | done | error
    started: float | None = None
    finished: float | None = None
    result: dict[str, Any] | None = None
    error: str = ""
    trace: Path | None = None  # трейс этого прогона, найденный один раз


def _stage_of(event: dict[str, Any]) -> int:
    """Этап из пяти на странице по событию трейса ядра."""
    name, step = event.get("event", ""), str(event.get("step", ""))
    if name.startswith("critic") or step.startswith("critic") or name == "verdict":
        return 4
    if name == "pass2_ensemble" or name.startswith("quote"):
        return 3
    if step.startswith(("pass1", "pass2")):
        return 2
    if name in ("segment", "rules") or step.startswith("pass0"):
        return 1
    return 0


def _event_note(e: dict[str, Any]) -> str:
    name, dep = e.get("event", ""), e.get("deployment") or e.get("model") or ""
    step = str(e.get("step", "")).split(":")[0]
    if name == "llm_call":
        return f"{step}: {dep} ответила за {float(e.get('latency_s', 0)):.0f} с"
    if name == "empty_response":
        return f"{step}: {dep} вернула пустой ответ — переключаемся на резервную модель"
    if name == "llm_error":
        kind = {"rate_limit": "лимит 429", "timeout": "таймаут", "server": "ошибка сервера"}.get(e.get("kind"), e.get("kind"))
        return f"{step}: {dep} — {kind}, ждём и переключаемся"
    if name == "validation_error":
        return f"{step}: ответ не по схеме — просим модель исправить JSON"
    if name == "pass2_ensemble":
        return f"ансамбль Pass 2: {e.get('first')} + {e.get('second')}, совпало у обеих: {e.get('agreed')}"
    if name == "critic_reject":
        return "Критик отклонил придирку"
    if name == "cooldown":
        return f"{dep} на паузе {float(e.get('seconds', 0)):.0f} с после ошибок — работают резервные модели"
    if name == "model_fallback_ok":
        return f"{step}: ответила резервная модель {dep}"
    if name == "cache_hit":
        return f"{step}: шаг взят из кэша SQLite — 0 токенов"
    if name in ("segment", "rules"):
        return "сегментация и детерминированные правила готовы"
    return name


def find_trace(trace_dir: Path, started: float, sha8: str) -> Path | None:
    """Самый ранний трейс документа `sha8`, начатый не раньше `started`.

    Трейсы пишутся в общий каталог (CLI, другие экземпляры сервера), поэтому прогоны того же файла, начатые раньше
    или позже, игнорируются: задача «прилипает» к своему трейсу и не перескакивает на чужой.
    """
    floor = time.strftime("run_%Y%m%dT%H%M%S", time.localtime(int(started)))
    for f in sorted(trace_dir.glob("run_*.jsonl")):
        if f.stem < floor:
            continue
        try:
            with f.open(encoding="utf-8") as fh:
                first = json.loads(fh.readline() or "{}")
        except (OSError, ValueError):
            continue
        if str(first.get("run_id", "")).endswith(sha8):
            return f
    return None


def _render(report: AuditReport, work: Path, key: str) -> dict[str, Any]:
    written, errors = write_reports(report, work / "report.md", "md,html,pdf")
    downloads = {p.suffix.lstrip("."): f"/api/runs/{key}/report.{p.suffix.lstrip('.')}" for p in written}
    view = report_to_web(report, downloads=downloads)
    view["render_errors"] = errors
    return view


def create_app(settings: Settings | None = None, runs_dir: Path | None = None) -> FastAPI:
    settings = settings or load_settings()
    runs = runs_dir or settings.cache_dir / "web_runs"
    jobs: dict[str, Job] = {}
    jobs_lock = threading.Lock()
    llm_mutex = threading.Lock()  # бесплатные квоты: один полный LLM-аудит за раз, остальные ждут в очереди

    app = FastAPI(title="DAIEK — AI-аудит требований", version=__version__, docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(Exception)
    async def _any_error(_: Request, exc: Exception) -> JSONResponse:  # последний рубеж: без трейсбеков в браузере
        return JSONResponse({"ok": False, "error": f"Внутренняя ошибка: {type(exc).__name__}. Попробуйте ещё раз."},
                            status_code=200)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html = (STATIC / "index.html").read_text(encoding="utf-8").replace("{{VERSION}}", __version__)
        return HTMLResponse(html)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "team": TEAM, "llm_available": settings.has_any_llm_key()}

    @app.get("/api/case")
    async def case() -> JSONResponse:
        try:
            view = json.loads(CASE_JSON.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return _fail("Снимок отчёта v1.0.2 не найден в пакете.")
        return JSONResponse({"ok": True, "view": view})

    def _workdir() -> tuple[str, Path]:
        key = uuid.uuid4().hex
        work = runs / key
        work.mkdir(parents=True, exist_ok=True)
        return key, work

    def _sync_audit(path: Path, demo: bool, work: Path, key: str) -> dict[str, Any]:
        report = run_audit(path, demo=demo, settings=settings)
        return _render(report, work, key)

    def _run_job(job: Job) -> None:
        with llm_mutex:
            job.status, job.started = "running", time.time()
            try:
                report = run_audit(job.path, demo=False, settings=settings)
                job.result = _render(report, job.work, job.key)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 — сбой фоновой задачи показываем на странице, а не в логе
                job.error, job.status = f"{type(exc).__name__}: {exc}", "error"
            finally:
                job.finished = time.time()

    def _progress(job: Job) -> dict[str, Any]:
        """Этап, число вызовов и последнее событие — из JSONL-трейса этого прогона."""
        out: dict[str, Any] = {"stage": 0, "calls": 0, "note": ""}
        if job.started is None:
            return out
        trace = job.trace or find_trace(settings.trace_dir, job.started, job.sha8)
        job.trace = trace
        if trace is None:
            return out
        try:
            events = [json.loads(ln) for ln in trace.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except (OSError, ValueError):
            return out
        out["stage"] = max((_stage_of(e) for e in events), default=0)
        out["calls"] = sum(e.get("event") == "llm_call" for e in events)
        out["note"] = _event_note(events[-1]) if events else ""
        return out

    @app.post("/api/analyze")
    async def analyze(file: UploadFile | None = File(None), mode: str = Form("demo")) -> JSONResponse:
        if file is None or not file.filename:
            return _fail("Файл не выбран.", hint="Перетащите .md, .txt или .pdf в область загрузки.")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            return _fail("Файл больше 20 МБ.", hint="Разделите документ или загрузите текстовую версию.")
        key, work = _workdir()
        path = work / _safe_name(file.filename)
        path.write_bytes(data)
        if mode != "llm":
            return JSONResponse({"ok": True, "view": await run_in_threadpool(_sync_audit, path, True, work, key)})
        if not settings.has_any_llm_key():
            return _fail("Полный LLM-аудит недоступен: не задан ни один ключ LLM.",
                         hint="Включите «Быстрый демо-аудит» или впишите GEMINI_API_KEY в .env и перезапустите сервер.")
        # Нечитаемый файл не ждёт очереди LLM: ядро всё равно вернёт CANNOT_PROCEED без вызовов модели.
        pre = await run_in_threadpool(ingest, path, enable_ocr=settings.enable_ocr)
        if not pre.status.can_proceed:
            return JSONResponse({"ok": True, "view": await run_in_threadpool(_sync_audit, path, False, work, key)})
        job = Job(key=key, path=path, work=work, sha8=hashlib.sha256(data).hexdigest()[:8], created=time.time())
        with jobs_lock:
            jobs[key] = job
        threading.Thread(target=_run_job, args=(job,), name=f"daiek-llm-{key[:8]}", daemon=True).start()
        return JSONResponse({"ok": True, "job": key, "status": job.status})

    @app.get("/api/jobs/{key}")
    async def job_status(key: str) -> JSONResponse:
        job = jobs.get(key) if RUN_KEY.match(key) else None
        if job is None:
            return JSONResponse({"ok": False, "error": "Задача не найдена: сервер перезапускался или ссылка устарела.",
                                 "hint": "Запустите проверку заново."})
        with jobs_lock:
            ahead = sum(1 for j in jobs.values() if j.status in ("queued", "running") and j.created < job.created)
        now = job.finished or time.time()
        body: dict[str, Any] = {
            "ok": job.status != "error", "status": job.status, "ahead": ahead if job.status == "queued" else 0,
            "elapsed_s": round(now - job.created, 1),
            "running_s": round(now - job.started, 1) if job.started else 0.0,
            **_progress(job),
        }
        if job.status == "done":
            body["view"] = job.result
        if job.status == "error":
            body["error"] = f"Полный LLM-аудит завершился ошибкой ({job.error})."
            body["hint"] = "Попробуйте ещё раз или используйте быстрый демо-аудит."
        return JSONResponse(body)

    @app.post("/api/sample/flawed")
    async def sample_flawed() -> JSONResponse:
        key, work = _workdir()
        path = work / SAMPLE_FLAWED.name
        shutil.copyfile(SAMPLE_FLAWED, path)
        return JSONResponse({"ok": True, "view": await run_in_threadpool(_sync_audit, path, True, work, key)})

    @app.get("/api/runs/{key}/report.{fmt}")
    async def download(key: str, fmt: str) -> Any:
        target = runs / key / f"report.{fmt}"
        if not RUN_KEY.match(key) or fmt not in DOWNLOAD_FORMATS or not target.is_file():
            return JSONResponse({"ok": False, "error": "Отчёт не найден: запустите проверку заново."}, status_code=404)
        media = {"pdf": "application/pdf", "md": "text/markdown; charset=utf-8", "html": "text/html; charset=utf-8"}
        return FileResponse(target, media_type=media[fmt], filename=f"daiek_report.{fmt}")

    return app
