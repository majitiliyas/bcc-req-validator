"""CLI агента: `req-validator analyze <файл> --out <отчёт>` и `req-validator doctor`."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .analyzer import git_info, run_audit
from .config import load_settings
from .models import VerdictStatus
from .reporting import resolve_formats, write_reports

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
) -> None:
    """Проверить документ требований и записать отчёт."""
    settings = load_settings()
    try:
        formats = resolve_formats(out, fmt)
    except ValueError as exc:
        console.print(f"[red]Ошибка:[/red] {exc}")
        raise typer.Exit(2) from exc
    if not demo and not settings.has_any_llm_key():
        console.print(
            "[red]Не найден ключ LLM.[/red] Укажите GEMINI_API_KEY в файле .env (см. .env.example) "
            "или запустите без модели: [bold]--demo[/bold]."
        )
        raise typer.Exit(2)

    console.print(f"Проверка документа: [bold]{input_path}[/bold] (режим: {'демо' if demo else 'LLM'})")
    report = run_audit(input_path, demo=demo, settings=settings)
    written, errors = write_reports(report, out, ",".join(formats))

    m, v = report.metrics, report.verdict
    console.print(f"Статус чтения: {report.passport.ingestion_status.value}")
    for w in report.passport.ingestion_warnings:
        console.print(f"  [yellow]•[/yellow] {w}")
    console.print(
        f"Требований: {m.total_requirements} · нарушений: {m.total_defects} "
        f"(FATAL {m.fatal_count} / MAJOR {m.major_count} / MINOR {m.minor_count}) · индекс готовности: {m.readiness_score}"
    )
    console.print(f"Вердикт: [{_VERDICT_STYLE[v.status]}]{v.status.value}[/]")
    for err in errors:
        console.print(f"[red]{err}[/red]")
    for path in written:
        console.print(f"Отчёт записан: [bold]{path}[/bold]")
    if not written:
        raise typer.Exit(1)


@app.command()
def doctor() -> None:
    """Проверить окружение: Python, зависимости, OCR, ключи LLM, git, рендер PDF."""
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
                "jinja2", "typst", "openai", "google.genai"):
        try:
            m = importlib.import_module(mod)
            row(f"Пакет {mod}", True, str(getattr(m, "__version__", getattr(m, "VERSION", "установлен"))))
        except Exception as exc:  # noqa: BLE001
            row(f"Пакет {mod}", False, f"{type(exc).__name__}: {exc}")

    from .ingestion import ocr_available

    row("OCR (RapidOCR)", True if ocr_available() else None,
        "доступен" if ocr_available() else "не установлен: сканы PDF без текста не распознаются (uv sync --extra ocr)")
    row("Ключ LLM", True if settings.has_any_llm_key() else None,
        f"задан, модель {settings.primary_model}" if settings.has_any_llm_key() else "не задан — доступен режим --demo")
    row("Ретраи SDK", settings.sdk_max_retries == 0, f"max_retries={settings.sdk_max_retries} (повторы — только в роутере)")
    try:
        import typst

        pdf = typst.compile(b"#set text(lang: \"ru\")\n\xd0\x9f\xd1\x80\xd0\xbe\xd0\xb2\xd0\xb5\xd1\x80\xd0\xba\xd0\xb0")
        row("Рендер PDF (typst)", isinstance(pdf, bytes) and pdf.startswith(b"%PDF"), "кириллица компилируется")
    except Exception as exc:  # noqa: BLE001
        row("Рендер PDF (typst)", False, f"{type(exc).__name__}: {exc}")
    try:
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        row("Каталог кэша/трасс", True, str(settings.cache_dir.resolve()))
    except OSError as exc:
        row("Каталог кэша/трасс", False, str(exc))

    Console().print(table)
    if not ok:
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
