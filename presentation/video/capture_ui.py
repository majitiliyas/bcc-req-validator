"""Скриншоты реального веб-интерфейса DAIEK для видео-демо (Playwright + установленный Google Chrome).

    uv run req-validator web --port 8020            # в отдельном терминале
    uv run --with playwright python presentation/video/capture_ui.py [http://127.0.0.1:8020]

Кадры 1920×1080 при deviceScaleFactor=2 (чёткие крупные планы) → presentation/video/shots/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8020"
OUT = Path(__file__).parent / "shots"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=2)
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(800)
        page.screenshot(path=OUT / "ui_home.png")
        page.locator("#pipeline").screenshot(path=OUT / "ui_pipeline_idle.png")

        page.click("#btn-case")
        page.wait_for_timeout(1600)  # этапы в процессе
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=OUT / "ui_processing.png")
        page.wait_for_selector("#results:not([hidden])", timeout=15000)
        page.wait_for_timeout(2600)  # счётчики и индикатор доиграли

        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(400)
        page.locator("#bob").screenshot(path=OUT / "ui_bob_rejected.png")

        page.evaluate("document.querySelector('#results').scrollIntoView({block: 'start'})")
        page.wait_for_timeout(500)
        page.screenshot(path=OUT / "ui_verdict.png")
        page.locator("#verdict").screenshot(path=OUT / "ui_verdict_card.png")
        page.locator("#kpis").screenshot(path=OUT / "ui_kpis.png")

        for did in ("DEF-001", "DEF-004", "DEF-006"):
            card = page.locator("article.def", has_text=did).first
            card.scroll_into_view_if_needed()
            page.wait_for_timeout(200)
            card.screenshot(path=OUT / f"ui_{did.lower()}.png")

        for tab in ("funnel", "roi", "passport"):
            page.click(f".tab[data-tab='{tab}']")
            page.evaluate("document.querySelector('.tabs-card').scrollIntoView({block: 'start'})")
            page.wait_for_timeout(1400)
            if tab == "roi":
                page.evaluate("const r=document.querySelector('#docs'); r.value=30; r.dispatchEvent(new Event('input'))")
                page.wait_for_timeout(300)
            page.screenshot(path=OUT / f"ui_{tab}.png")
            page.locator(f"#pane-{tab}").screenshot(path=OUT / f"ui_{tab}_pane.png")
        browser.close()
    print(sorted(f.name for f in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
