"""CLI агента: `req-validator analyze <файл> --out <отчёт>` и `req-validator doctor`."""

from __future__ import annotations

import asyncio
import importlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .analyzer import git_info, run_audit
from .cache import DiskStepCache
from .config import Settings, load_settings
from .models import VerdictStatus
from .reporting import resolve_formats, write_reports
from .router import ollama_reachable

app = typer.Typer(
    add_completion=False, no_args_is_help=True, rich_markup_mode=None,
    help="Агент статической проверки требований по ISO/IEC/IEEE 29148:2018 и IEEE Std 830-1998.",
)
console = Console(stderr=True)

_VERDICT_STYLE = {
    VerdictStatus.APPROVED: "bold green", VerdictStatus.CONDITIONAL_PASS: "bold yellow",
    VerdictStatus.REJECTED: "bold red", VerdictStatus.CANNOT_PROCEED: "bold white on red",
}


@app.command()
def analyze(
    input_path: Annotated[Path, typer.Argument(help="Путь к документу требований (.md, .txt, .pdf)")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Путь к файлу отчёта")] = Path("report.md"),
    fmt: Annotated[str, typer.Option("--format", "-f", help="auto (по расширению --out) | md | html | pdf | json | both (md+pdf) | all | список через запятую")] = "auto",
    demo: Annotated[bool, typer.Option("--demo", help="Без LLM и сети: только детерминированные правила")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Не использовать дисковый кэш шагов LLM")] = False,
    max_steps: Annotated[int | None, typer.Option("--max-steps", help="Лимит вызовов LLM на прогон")] = None,
) -> None:
    """Проверить документ требований и записать отчёт."""
    settings = load_settings()
    updates: dict[str, object] = {}
    if no_cache:
        updates["use_cache"] = False
    if max_steps is not None:
        updates["budget_max_steps"] = max_steps
    if updates:
        settings = settings.model_copy(update=updates)
    try:
        formats = resolve_formats(out, fmt)
    except ValueError as exc:
        console.print(f"[red]Ошибка:[/red] {exc}")
        raise typer.Exit(2) from exc

    if not demo and not settings.has_any_llm_key():
        if settings.enable_ollama and ollama_reachable(settings):
            console.print("[yellow]Ключи облачных LLM не заданы — используется только локальный Ollama.[/yellow]")
        else:
            console.print(
                "[red]Не найден ни один ключ LLM[/red] (GEMINI_API_KEY, NVIDIA_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY) "
                "и локальный Ollama недоступен.\nСкопируйте .env.example в .env и впишите бесплатный ключ "
                "(aistudio.google.com) или запустите без модели: [bold]--demo[/bold]. Диагностика: "
                "[bold]req-validator doctor[/bold]."
            )
            raise typer.Exit(2)

    console.print(f"Проверка документа: [bold]{input_path}[/bold] (режим: {'демо' if demo else 'LLM'})")
    report = run_audit(input_path, demo=demo, settings=settings)
    written, errors = write_reports(report, out, ",".join(formats))

    m, v, info = report.metrics, report.verdict, report.passport.llm
    console.print(f"Статус чтения: {report.passport.ingestion_status.value}")
    for w in report.passport.ingestion_warnings:
        console.print(f"  [yellow]•[/yellow] {w}")
    if info is not None:
        console.print(f"LLM: вызовов {info.llm_calls}, из кэша {info.cache_hits}, модели: "
                      f"{', '.join(info.models_used) or '—'}")
        for p in info.passes:
            color = {"ok": "green", "partial": "yellow", "failed": "red"}.get(p.status, "white")
            console.print(f"  [{color}]{p.status:>7}[/] {p.name}: {p.detail}")
        if any(p.status == "failed" for p in info.passes):
            console.print("[yellow]Часть LLM-проходов не выполнена (квоты/ключи/сеть). Отчёт содержит то, что "
                          "удалось проверить; вердикт помечен как предварительный. Подробности: "
                          "req-validator doctor --online и .cache/traces/.[/yellow]")
    for n in report.passport.pipeline_notes:
        if "прерван" in n or "пропущены" in n:
            console.print(f"[yellow]{n}[/yellow]")
    console.print(
        f"Требований: {m.total_requirements} · нарушений: {m.total_defects} "
        f"(FATAL {m.fatal_count} / MAJOR {m.major_count} / MINOR {m.minor_count}) · индекс готовности: {m.readiness_score}"
    )
    console.print(f"Вердикт: [{_VERDICT_STYLE[v.status]}]{v.status.value}[/]"
                  + (f" — гейт {v.gate}, блокирующие: {', '.join(v.blocking_defect_ids)}" if v.blocking_defect_ids else ""))
    for err in errors:
        console.print(f"[red]{err}[/red]")
    for path in written:
        console.print(f"Отчёт записан: [bold]{path}[/bold]")
    if not written:
        raise typer.Exit(1)


async def _probe_models(settings: Settings) -> dict[str, tuple[bool, str]]:
    """Онлайн-проверка: /models каждого провайдера с ключом; совпадают ли сконфигурированные slug'и."""
    from openai import AsyncOpenAI

    from .router import build_groups

    results: dict[str, tuple[bool, str]] = {}
    seen: dict[str, set[str] | str] = {}
    for deps in build_groups(settings).values():
        for dep in deps:
            if dep.provider not in seen:
                t0 = time.perf_counter()
                try:
                    client = AsyncOpenAI(api_key=dep.api_key or "ollama", base_url=dep.base_url, max_retries=0, timeout=15)
                    page = await client.models.list()
                    seen[dep.provider] = {m.id.removeprefix("models/") for m in page.data}
                    results[f"{dep.provider} (/models)"] = (True, f"{len(seen[dep.provider])} моделей, "
                                                                  f"{time.perf_counter() - t0:.1f} с")
                except Exception as exc:  # noqa: BLE001
                    seen[dep.provider] = f"{type(exc).__name__}: {str(exc)[:120]}"
                    results[f"{dep.provider} (/models)"] = (False, str(seen[dep.provider]))
            ids = seen[dep.provider]
            if isinstance(ids, set):
                ok = dep.model in ids or dep.model.removeprefix("models/") in ids
                results[dep.name] = (ok, "slug найден" if ok else "slug НЕ найден в /models — проверьте имя в .env")
    return results


@app.command()
def doctor(
    online: Annotated[bool, typer.Option("--online", help="Проверить ключи и slug'и моделей через /models (сеть, без расхода токенов)")] = False,
) -> None:
    """Проверить окружение: Python, зависимости, ключи и маршрутизацию LLM, кэш, OCR, рендер PDF."""
    from .ingestion import ocr_available
    from .pii import presidio_available
    from .prompts import prompt_versions
    from .router import build_groups

    settings = load_settings()
    table = Table(title="Диагностика окружения req-validator", show_lines=False)
    table.add_column("Проверка")
    table.add_column("Статус")
    table.add_column("Детали")
    ok = True

    def row(name: str, good: bool | None, details: str) -> None:
        nonlocal ok
        mark = "[green]OK[/green]" if good else ("[yellow]—[/yellow]" if good is None else "[red]ОШИБКА[/red]")
        if good is False:
            ok = False
        table.add_row(name, mark, details)

    row("Python", sys.version_info >= (3, 12), f"{platform.python_version()} (нужен ≥ 3.12)")
    row("Версия агента", True, f"{__version__}, коммит {git_info()[0]}")
    for mod in ("pydantic", "markdown_it", "charset_normalizer", "pymupdf", "pymupdf4llm", "rapidfuzz", "simpleeval",
                "jinja2", "typst", "openai"):
        try:
            m = importlib.import_module(mod)
            row(f"Пакет {mod}", True, str(getattr(m, "__version__", getattr(m, "VERSION", "установлен"))))
        except Exception as exc:  # noqa: BLE001
            row(f"Пакет {mod}", False, f"{type(exc).__name__}: {exc}")

    keys = settings.configured_keys()
    for provider, present in keys.items():
        row(f"Ключ {provider}", True if present else None, "задан" if present else "не задан")
    local = settings.enable_ollama and ollama_reachable(settings)
    row("Ollama (локально)", True if local else None,
        f"доступен, модель {settings.ollama_model}" if local else "не запущен (необязательный резерв группы deep)")
    groups = build_groups(settings, probe_local=True)
    for g, deps in groups.items():
        row(f"Группа {g}", True if deps else None,
            " → ".join(d.name for d in deps) or "нет доступных провайдеров (задайте ключи или запустите Ollama)")
    if not any(keys.values()) and not local:
        row("Режим LLM", None, "недоступен — работает только --demo")
    row("Ретраи SDK", settings.sdk_max_retries == 0,
        f"max_retries={settings.sdk_max_retries}; роутер: allowed_fails={settings.allowed_fails}, "
        f"cooldown={settings.cooldown_time:.0f} с, backoff full jitter, Retry-After")
    row("Бюджет прогона", True, f"шагов ≤ {settings.budget_max_steps}, токенов ≤ {settings.budget_max_tokens}, "
                                f"время ≤ {settings.budget_max_wall_seconds:.0f} с")
    row("Маскирование ПДн", True, "правила + Presidio" if presidio_available() else "детерминированные правила "
                                  "(Presidio не установлен)")
    row("Промпты", True, json.dumps(prompt_versions(), ensure_ascii=False))
    try:
        cache = DiskStepCache(settings.cache_db, enabled=True)
        row("Кэш шагов (SQLite)", True, f"{settings.cache_db} — записей: {cache.count()}")
        cache.close()
    except Exception as exc:  # noqa: BLE001
        row("Кэш шагов (SQLite)", False, str(exc))
    row("OCR (RapidOCR)", True if ocr_available() else None,
        "доступен" if ocr_available() else "не установлен: сканы PDF без текста не распознаются (uv sync --extra ocr)")
    try:
        import typst

        pdf = typst.compile("#set text(lang: \"ru\")\nПроверка".encode())
        row("Рендер PDF (typst)", isinstance(pdf, bytes) and pdf.startswith(b"%PDF"), "кириллица компилируется")
    except Exception as exc:  # noqa: BLE001
        row("Рендер PDF (typst)", False, f"{type(exc).__name__}: {exc}")

    if online:
        if not any(groups.values()):
            row("Онлайн-проверка", None, "нет провайдеров для проверки")
        else:
            for name, (good, detail) in asyncio.run(_probe_models(settings)).items():
                row(f"Онлайн: {name}", good, detail)

    Console().print(table)
    if not ok:
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
