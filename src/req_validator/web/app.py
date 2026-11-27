"""Веб-интерфейс DAIEK поверх того же CLI-ядра: `req-validator web`.

Тонкая обёртка: загрузка файла → `run_audit` (тот же пайплайн, что у `analyze`) → `write_reports` → JSON для дашборда.
Никаких HTTP 500 и трейсбеков: плохой вход превращается в отчёт CANNOT_PROCEED, сбой — в понятное сообщение.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .. import __version__
from ..analyzer import run_audit
from ..config import Settings, load_settings
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


def create_app(settings: Settings | None = None, runs_dir: Path | None = None) -> FastAPI:
    settings = settings or load_settings()
    runs = runs_dir or settings.cache_dir / "web_runs"
    llm_lock = asyncio.Semaphore(1)  # бесплатные квоты: один полный LLM-аудит за раз

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

    async def _audit(path: Path, demo: bool, work: Path, key: str) -> JSONResponse:
        if not demo and not settings.has_any_llm_key():
            return _fail("Полный LLM-аудит недоступен: не задан ни один ключ LLM.",
                         hint="Включите «Быстрый демо-аудит» или впишите GEMINI_API_KEY в .env и перезапустите сервер.")
        if demo:
            report = await run_in_threadpool(run_audit, path, demo=True, settings=settings)
        else:
            async with llm_lock:
                report = await run_in_threadpool(run_audit, path, demo=False, settings=settings)
        written, errors = await run_in_threadpool(write_reports, report, work / "report.md", "md,html,pdf")
        downloads = {p.suffix.lstrip("."): f"/api/runs/{key}/report.{p.suffix.lstrip('.')}" for p in written}
        view = report_to_web(report, downloads=downloads)
        view["render_errors"] = errors
        return JSONResponse({"ok": True, "view": view})

    def _workdir() -> tuple[str, Path]:
        key = uuid.uuid4().hex
        work = runs / key
        work.mkdir(parents=True, exist_ok=True)
        return key, work

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
        return await _audit(path, mode != "llm", work, key)

    @app.post("/api/sample/flawed")
    async def sample_flawed() -> JSONResponse:
        key, work = _workdir()
        path = work / SAMPLE_FLAWED.name
        shutil.copyfile(SAMPLE_FLAWED, path)
        return await _audit(path, True, work, key)

    @app.get("/api/runs/{key}/report.{fmt}")
    async def download(key: str, fmt: str) -> Any:
        target = runs / key / f"report.{fmt}"
        if not RUN_KEY.match(key) or fmt not in DOWNLOAD_FORMATS or not target.is_file():
            return JSONResponse({"ok": False, "error": "Отчёт не найден: запустите проверку заново."}, status_code=404)
        media = {"pdf": "application/pdf", "md": "text/markdown; charset=utf-8", "html": "text/html; charset=utf-8"}
        return FileResponse(target, media_type=media[fmt], filename=f"daiek_report.{fmt}")

    return app
