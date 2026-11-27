from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def settings(tmp_path: Path):
    from req_validator.config import Settings

    return Settings(_env_file=None, cache_dir=tmp_path / ".cache", gemini_api_key=None)


def make_text_pdf(path: Path, pages_html: list[str | None]) -> Path:
    """PDF с текстовым слоем; None → пустая страница (имитация скана без текста)."""
    import pymupdf

    doc = pymupdf.open()
    for html in pages_html:
        page = doc.new_page()
        if html is None:
            page.draw_rect(pymupdf.Rect(80, 80, 400, 300), color=(0, 0, 0), fill=(0.6, 0.6, 0.6))
        else:
            page.insert_htmlbox(pymupdf.Rect(50, 50, 550, 800), html)
    doc.save(path)
    return path
