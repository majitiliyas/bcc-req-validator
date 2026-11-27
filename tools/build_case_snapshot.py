"""Снимок реального LLM-прогона v1.0.2 (кейс БЦК) для веб-дашборда.

Читает Markdown-отчёт `reports/requirements_llm_report.md`, собирает JSON в формате веб-представления
(`req_validator.web.view.report_to_web`) и копирует оригинальные отчёты md/html/pdf рядом.
Числа не пересчитываются — только переносятся из отчёта; итоговые счётчики сверяются с таблицей «Сводка».

    uv run python tools/build_case_snapshot.py [reports/requirements_llm_report.md]
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from req_validator.reporting import clean_md

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "src/req_validator/web/static/case"
STEM = "bcc_requirements_v1.0.2"

DEF_RE = re.compile(r"^- \*\*\[(FATAL|MAJOR|MINOR)\]\*\* `(DEF-\d+)` · (.+?) \(`([^`]+)`\) · (.+)$")
REQ_HEAD = re.compile(r"^#### (REQ-\d+)(?: · [^—]+)? — (.+)$")
QUOTE_RE = re.compile(r"«(.*?)»(?: \(проверена, позиция [^)]*\))?(?:\s+- \*\*Противоречит \(([^)]+)\):\*\* «(.*)»)?\s*$")


def _unescape(s: str) -> str:
    return s.replace("\\|", "|").replace("\\$", "$")


def _table(lines: list[str], header_prefix: str) -> list[list[str]]:
    start = next(i for i, ln in enumerate(lines) if ln.startswith(header_prefix))
    rows = []
    for ln in lines[start + 2:]:
        if not ln.startswith("|"):
            break
        rows.append([_unescape(c.strip()) for c in ln.strip().strip("|").split("|")])
    return rows


def _num(s: str) -> float:
    return float(re.sub(r"[^\d.,-]", "", s.replace("−", "-")).replace(",", "."))


def parse_defects(lines: list[str]) -> tuple[list[dict], int]:
    defects: dict[str, dict] = {}
    hints = 0
    req, section = "весь документ", ""
    cur: dict | None = None
    for ln in lines:
        if ln.startswith("- ⚠️ *Подсказка*"):
            hints += 1
            cur = None
            continue
        if h := REQ_HEAD.match(ln):
            req, section, cur = h.group(1), h.group(2), None
            continue
        if ln.startswith("## 3."):
            req, section, cur = "весь документ", "", None
        if ln.startswith("## 4."):
            break
        if m := DEF_RE.match(ln):
            sev, did, criterion, rule, rest = m.groups()
            parts = [x.strip() for x in rest.split(" · ")]
            prio = next((x.split()[-1] for x in parts if x.startswith("приоритет")), {"FATAL": "P0", "MAJOR": "P1"}.get(sev, "P2"))
            src = next((x.removeprefix("источник: ") for x in parts if x.startswith("источник:")), "")
            review = parts[-1]
            orig = re.search(r"было (FATAL|MAJOR|MINOR)", review)
            sec = section or next((x for x in parts if not x.startswith(("приоритет", "источник"))), "")
            cur = {"id": did, "severity": sev, "priority": prio, "rule_id": rule, "criterion": criterion,
                   "req_id": req, "related_req": "", "section": sec, "quote_view": "", "related_view": "",
                   "explanation": "", "risk": "", "suggestion": "", "source_label": src,
                   "review": re.sub(r" \(было \w+\)", "", review).strip(), "review_note": "",
                   "original_severity": orig.group(1) if orig else "",
                   "source": "rules" if src == "правила" else "llm"}
            if did in defects:  # повтор той же находки под связанным требованием
                cur = None
                continue
            defects[did] = cur
            continue
        if cur is None:
            continue
        s = ln.strip()
        if s.startswith("- **Цитата:**"):
            q = QUOTE_RE.search(s.removeprefix("- **Цитата:**").strip())
            if q:
                cur["quote_view"] = clean_md(_unescape(q.group(1)))
                if q.group(2):
                    cur["related_req"], cur["related_view"] = q.group(2), clean_md(_unescape(q.group(3)))
        elif s.startswith("- **Проблема:**"):
            cur["explanation"] = _unescape(s.removeprefix("- **Проблема:**").strip())
        elif s.startswith("- **Риск:**"):
            cur["risk"] = _unescape(s.removeprefix("- **Риск:**").strip())
        elif s.startswith("- **Как исправить:**"):
            cur["suggestion"] = _unescape(s.removeprefix("- **Как исправить:**").strip())
        elif s.startswith("- *Критик"):
            cur["review_note"] = _unescape(s.removeprefix("- *").rstrip("*"))
    return sorted(defects.values(), key=lambda d: d["id"]), hints


def parse_remediation(lines: list[str]) -> list[dict]:
    start = next(i for i, ln in enumerate(lines) if ln.startswith("### Что исправить в первую очередь"))
    items: list[dict] = []
    for ln in lines[start + 1:]:
        if ln.startswith("---"):
            break
        if m := re.match(r"^\d+\. \*\*(P\d)\*\* — (.+)$", ln):
            items.append({"priority": m.group(1), "action_item": _unescape(m.group(2)), "actions": [],
                          "target_defect_ids": [], "business_impact": ""})
        elif items and (m := re.match(r"^\s+- Дефекты: ([^.]+)\. \*(.+)\*$", ln)):
            items[-1]["target_defect_ids"] = [x.strip() for x in m.group(1).split(",")]
            items[-1]["business_impact"] = m.group(2)
        elif items and ln.startswith("   - "):
            items[-1]["actions"].append(_unescape(ln.strip()[2:]))
    return items


def build(md_path: Path) -> dict:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    kv = {r[0]: r[1] for r in _table(lines, "| Параметр | Значение |")}
    llm_kv = {}
    second = [i for i, ln in enumerate(lines) if ln.startswith("| Параметр | Значение |")][1]
    for r in _table(lines[second:], "| Параметр | Значение |"):
        llm_kv[r[0]] = r[1]
    summary = {r[0]: r[1] for r in _table(lines, "| Показатель | Значение |")}
    funnel = [{"stage": r[0], "count": int(r[1]), "note": r[3]} for r in _table(lines, "| Этап | Кол-во |")]
    passes = [{"name": r[0], "status": "ok" if "выполнен" in r[1] else r[1], "detail": r[4],
               "models": [] if r[2] == "—" else [x.strip() for x in r[2].split(",")]}
              for r in _table(lines, "| Проход | Статус |")]
    critic = {k: int(v) for k, v in (x.split(": ") for x in llm_kv["Критик"].split(", "))}
    calls, cache = (int(x) for x in llm_kv["Вызовов LLM / из кэша SQLite"].split(" / "))
    tin, tout = (int(x) for x in llm_kv["Токены (вход / выход)"].split(" / "))

    roi_rows = _table(lines, "| Статья | Консервативный | Базовый |")
    roi_lines = [r for r in roi_rows if not r[0].startswith("**") and not r[0].startswith(("Поток", "Стоимость"))]
    totals = next(r for r in roi_rows if r[0].startswith("**Итого"))
    money = next(r for r in roi_rows if r[0].startswith("**В деньгах"))
    llm_cost = next(r for r in roi_rows if r[0].startswith("Стоимость"))[1]
    a_start = next(i for i, ln in enumerate(lines) if ln.startswith("Допущения:"))
    assumptions = []
    for ln in lines[a_start + 1:]:
        if not ln.startswith("- "):
            break
        assumptions.append(ln[2:])
    scenarios = []
    for idx, (key, name) in enumerate((("conservative", "Консервативный"), ("base", "Базовый")), start=1):
        scenarios.append({"key": key, "name": name, "total_hours": _num(totals[idx]), "money": _num(money[idx]),
                          "lines": [{"item": r[0], "hours": _num(r[idx]), "formula": r[3] if idx == 2 else ""}
                                    for r in roi_lines]})

    defects, hints = parse_defects(lines)
    sev = [int(x) for x in summary["FATAL / MAJOR / MINOR"].split(" / ")]
    metrics = {
        "total_requirements": int(summary["Требований выделено"]),
        "requirements_with_explicit_id": int(summary["С явным ID в документе"]),
        "requirements_with_defects": int(summary["Требований с нарушениями"]),
        "fatal_count": sev[0], "major_count": sev[1], "minor_count": sev[2],
        "total_defects": int(summary["Нарушений всего"]), "hints_count": int(summary["Подсказок (не входят в метрики)"]),
        "defect_density": _num(summary["Плотность дефектов (DD)"]),
        "readiness_score": _num(summary["Индекс готовности (S_DoR)"].split("/")[0]),
        "estimated_saved_hours": _num(summary["Сэкономленные трудозатраты (базовый сценарий ROI)"]),
    }
    # Сверка: перенесённые карточки совпадают со сводкой отчёта.
    counts = [sum(d["severity"] == s for d in defects) for s in ("FATAL", "MAJOR", "MINOR")]
    assert len(defects) == metrics["total_defects"] and counts == sev, (len(defects), counts, sev)
    assert hints == metrics["hints_count"], (hints, metrics["hints_count"])

    i_v = next(i for i, ln in enumerate(lines) if ln.startswith("## 5. Итог"))
    rationale = lines[i_v + 4].strip()
    title = lines[0].removeprefix("# Отчёт о проверке требований: ").strip()
    tag = re.search(r"тег `([^`]+)`", kv["Версия агента"])
    commit = re.search(r"коммит `([^`]+)`", kv["Версия агента"])
    return {
        "source": "snapshot",
        "title": title,
        "passport": {
            "file_name": "requirements.md", "file_size_bytes": int(re.search(r"\(([\d\s ]+) байт", kv["Файл"]).group(1)
                                                                    .replace(" ", "").replace(" ", "")),
            "sha256": kv["SHA-256"].strip("`"), "format": "md", "encoding": "utf-8",
            "run_id": kv["Идентификатор прогона"].strip("`"), "run_date": kv["Дата и время прогона"],
            "duration_s": _num(kv["Длительность"]), "mode": "llm", "model": kv["Модель"],
            "agent_version": kv["Версия агента"].split(",")[0], "git_commit": commit.group(1) if commit else "",
            "git_tag": tag.group(1) if tag else "", "ingestion_status": kv["Статус чтения документа"].strip("`"),
            "warnings": [], "notes": [],
        },
        "verdict": {
            "status": "REJECTED", "text": "НЕ ГОТОВО — ВОЗВРАТ АНАЛИТИКУ (REJECTED)", "rationale": rationale,
            "dev": False, "test": False, "dev_note": "", "test_note": "", "preliminary": False,
            "gate": "некомпенсаторный FATAL-гейт", "blocking": ["DEF-001"],
        },
        "metrics": metrics,
        "defects": defects,
        "llm": {"calls": calls, "cache_hits": cache, "tokens_in": tin, "tokens_out": tout,
                "models_used": [x.strip() for x in llm_kv["Использованные модели"].split(",")],
                "critic": critic, "funnel": funnel, "passes": passes},
        "roi": {"rate": 10000.0, "currency": "KZT", "docs": 30, "llm_cost": llm_cost, "assumptions": assumptions,
                "scenarios": scenarios},
        "remediation": parse_remediation(lines),
        "downloads": {f: f"/static/case/{STEM}.{f}" for f in ("pdf", "md", "html")},
    }


def main() -> None:
    md = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports/requirements_llm_report.md"
    data = build(md)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{STEM}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for ext in ("md", "html", "pdf"):
        shutil.copyfile(md.with_suffix(f".{ext}"), OUT_DIR / f"{STEM}.{ext}")
    m = data["metrics"]
    print(f"OK: {m['total_requirements']} треб., {m['total_defects']} дефектов "
          f"({m['fatal_count']}/{m['major_count']}/{m['minor_count']}), подсказок {m['hints_count']}, "
          f"индекс {m['readiness_score']}, ROI {[s['total_hours'] for s in data['roi']['scenarios']]}")


if __name__ == "__main__":
    main()
