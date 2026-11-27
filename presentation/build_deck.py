"""Сборка презентации DAIEK (Команда «ВУПИПАЙ») для финала BCC-HUB × MNU FinTech Hackathon.

    uv run --with python-pptx python presentation/build_deck.py

Пишет presentation/DAIEK_WoopiePie_BCC_Hackathon.pptx и presentation/speaker_notes.md.
Все цифры — из финального отчёта релиза v1.0.2 (коммит 66340f6), см. src/req_validator/web/static/case/.
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.parts.image import Image
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation"
STATIC = ROOT / "src/req_validator/web/static"
BOB, BRAND, SCENE = STATIC / "bob", STATIC / "brand", OUT / "assets"

TEAM = "Команда «ВУПИПАЙ»"
MEMBERS = ("Елубай Илияс", "Мубарак Мансур", "Мылтыкбай Улпан")
TOTAL = 10
H_FONT, B_FONT, M_FONT = "Arial Black", "Arial", "Consolas"

NAVY, NAVY2, CARD, CARD2 = "070D22", "151049", "0F1736", "18224F"
BLUE, VIOLET, GOLD, GOLD2 = "3B82F6", "8B5CF6", "D4AF37", "F1D27A"
FATAL, MAJOR, PASS, MINOR = "EF4444", "F59E0B", "10B981", "60A5FA"
WHITE, MUTED, DIM, INK = "FFFFFF", "AEB6DA", "7A83AE", "1A1840"


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


def alpha(shape, pct: int) -> None:
    """Прозрачность сплошной заливки (python-pptx не даёт API)."""
    for clr in shape._element.spPr.iter(qn("a:srgbClr")):
        a = etree.SubElement(clr, qn("a:alpha"))
        a.set("val", str(pct * 1000))
        return


def no_line(shape) -> None:
    shape.line.fill.background()


def rect(slide, x, y, w, h, fill: str | None = CARD, *, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12,
         line: str | None = None, line_w=1.0, a: int | None = None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
        if a is not None:
            alpha(s, a)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(line_w)
    else:
        no_line(s)
    s.shadow.inherit = False
    return s


def grad(shape, c1: str, c2: str, angle: float = 0) -> None:
    shape.fill.gradient()
    shape.fill.gradient_angle = angle
    st = shape.fill.gradient_stops
    st[0].color.rgb, st[1].color.rgb = rgb(c1), rgb(c2)


def text(slide, x, y, w, h, paras, *, size=16, color=WHITE, font=B_FONT, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.08, space_after=0):
    """paras: str | list[str | list[(text, {style})]]."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(space_after)
        runs = [(para, {})] if isinstance(para, str) else para
        for t, st in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = st.get("font", font)
            f.size = Pt(st.get("size", size))
            f.bold = st.get("bold", bold)
            f.color.rgb = rgb(st.get("color", color))
    return tb


def pill(slide, x, y, w, h, label, *, fill=VIOLET, fill2: str | None = BLUE, color=WHITE, size=12, font=H_FONT,
         line: str | None = None):
    s = rect(slide, x, y, w, h, fill, radius=0.5, line=line)
    if fill2:
        grad(s, fill, fill2, 0)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = font, Pt(size), font != H_FONT, rgb(color)
    return s


def picture(slide, path: Path, x, y, w, h, *, rounded=True, line: str | None = GOLD, line_w=2.0, focus_y=0.5):
    """Вставка с обрезкой «cover» под рамку и скруглёнными углами."""
    iw, ih = Image.from_file(str(path)).size
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    frame, img = w / h, iw / ih
    if img > frame:  # шире рамки — режем бока
        cut = 1 - frame / img
        pic.crop_left = pic.crop_right = cut / 2
    else:  # выше рамки — режем по вертикали вокруг focus_y
        cut = 1 - img / frame
        pic.crop_top, pic.crop_bottom = cut * focus_y, cut * (1 - focus_y)
    if rounded:
        pic.auto_shape_type = MSO_SHAPE.ROUNDED_RECTANGLE
        geom = pic._element.spPr.find(qn("a:prstGeom"))
        av = geom.find(qn("a:avLst"))
        gd = etree.SubElement(av, qn("a:gd"))
        gd.set("name", "adj")
        gd.set("fmla", "val 8000")
    if line:
        pic.line.color.rgb = rgb(line)
        pic.line.width = Pt(line_w)
    return pic


def star(slide, x, y, s=0.16, color=WHITE, a=85):
    st = rect(slide, x, y, s, s, color, shape=MSO_SHAPE.STAR_4_POINT, a=a)
    st.adjustments[0] = 0.12
    return st


def background(slide, n: int) -> None:
    bg = rect(slide, 0, 0, 13.333, 7.5, NAVY, shape=MSO_SHAPE.RECTANGLE)
    grad(bg, NAVY, NAVY2, 315)
    g1 = rect(slide, 9.2, -2.6, 6.2, 6.2, VIOLET, shape=MSO_SHAPE.OVAL, a=16)
    g2 = rect(slide, -2.4, 4.6, 5.4, 5.4, BLUE, shape=MSO_SHAPE.OVAL, a=12)
    del g1, g2
    for (sx, sy, ss) in ((0.35, 1.55, 0.16), (12.75, 1.15, 0.2), (6.45, 7.0, 0.12), (12.2, 6.55, 0.14), (0.9, 6.75, 0.1)):
        star(slide, sx, sy, ss, a=70)
    # шапка: событие слева, команда справа, номер слайда снизу
    text(slide, 0.6, 0.3, 7.5, 0.3, [[("BCC-HUB × MNU FinTech Hackathon", {"bold": True, "color": MUTED}),
                                      ("  ·  DAIEK", {"color": GOLD2, "bold": True})]], size=11)
    pill(slide, 10.13, 0.22, 2.6, 0.42, TEAM, size=12)
    text(slide, 11.6, 7.05, 1.13, 0.3, f"{n} / {TOTAL}", size=10, color=DIM, align=PP_ALIGN.RIGHT)
    text(slide, 0.6, 7.05, 8, 0.3, "DAIEK · данные релиза v1.0.2 (66340f6) · ISO/IEC/IEEE 29148:2018 · IEEE Std 830-1998", size=10, color=DIM)


def title(slide, eyebrow: str, head, *, y=0.85, w=12.1, size=34):
    """Заголовок в одну строку: кегль подбирается по длине (Arial Black ≈ 0.7 em на символ)."""
    n = sum(len(t) for t, _ in head[0]) if isinstance(head, list) else len(head)
    size = min(size, int(w * 72 / (n * 0.7)))
    text(slide, 0.6, y, w, 0.3, eyebrow.upper(), size=12, color=GOLD2, bold=True)
    text(slide, 0.6, y + 0.32, w, 0.8, head, size=size, font=H_FONT, spacing=0.95, anchor=MSO_ANCHOR.MIDDLE)


def card(slide, x, y, w, h, head: str, body, *, accent=VIOLET, head_size=15, body_size=12.5, icon: str | None = None):
    rect(slide, x, y, w, h, CARD, line=accent, line_w=1.25)
    rect(slide, x, y + 0.18, 0.07, h - 0.36, accent, radius=0.5)
    tx = x + 0.28
    if icon:
        b = rect(slide, tx, y + 0.2, 0.46, 0.46, accent, shape=MSO_SHAPE.OVAL)
        b.text_frame.text = icon
        p = b.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.runs[0].font.size, p.runs[0].font.bold, p.runs[0].font.name = Pt(14), True, H_FONT
        p.runs[0].font.color.rgb = rgb(WHITE)
        b.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        b.text_frame.margin_left = b.text_frame.margin_right = 0
        text(slide, tx + 0.6, y + 0.2, w - 0.95, 0.5, head, size=head_size, font=H_FONT, anchor=MSO_ANCHOR.MIDDLE)
        body_y = y + 0.78
    else:
        text(slide, tx, y + 0.18, w - 0.45, 0.45, head, size=head_size, font=H_FONT)
        body_y = y + 0.62
    text(slide, tx, body_y, w - 0.45, h - (body_y - y) - 0.12, body, size=body_size, color=MUTED, spacing=1.1)


def quote(slide, x, y, w, h, label: str, q: str, *, size=12):
    rect(slide, x, y, w, h, GOLD, radius=0.08, a=10)
    rect(slide, x, y, 0.06, h, GOLD, shape=MSO_SHAPE.RECTANGLE)
    text(slide, x + 0.16, y + 0.06, w - 0.25, 0.25, label, size=9.5, color=MUTED)
    text(slide, x + 0.16, y + 0.28, w - 0.25, h - 0.3, q, size=size, color="F6E7B4", font=M_FONT)


def kpi(slide, x, y, w, h, value: str, label: str, *, color=WHITE, vsize=30, border=VIOLET):
    rect(slide, x, y, w, h, CARD, line=border, line_w=1.25)
    text(slide, x + 0.2, y + 0.12, w - 0.3, h * 0.55, value, size=vsize, font=H_FONT, color=color, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, x + 0.2, y + h * 0.6, w - 0.3, h * 0.38, label, size=11.5, color=MUTED)


def bubble(slide, x, y, w, h, who: str, line: str, *, size=13):
    rect(slide, x, y, w, h, "F7F4FF", radius=0.18)
    text(slide, x + 0.2, y + 0.12, w - 0.4, 0.25, who.upper(), size=9.5, color=VIOLET, bold=True)
    text(slide, x + 0.2, y + 0.36, w - 0.4, h - 0.45, line, size=size, color=INK, bold=True, spacing=1.1)


NOTES: dict[int, tuple[str, str]] = {}


def notes(slide, n: int, timing: str, body: str) -> None:
    NOTES[n] = (timing, body)
    slide.notes_slide.notes_text_frame.text = f"[{timing}] {body}"


# --------------------------------------------------------------------------- #
# Слайды
# --------------------------------------------------------------------------- #


def s1(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); background(s, 1)
    logo = rect(s, 0.6, 1.25, 1.35, 1.35, WHITE, radius=0.22, line=GOLD, line_w=2)
    del logo
    picture(s, BRAND / "daiek_logo_main.jpg", 0.66, 1.31, 1.23, 1.23, line=None)
    text(s, 2.2, 1.28, 6.8, 1.3, "DAIEK", size=80, font=H_FONT, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.6, 2.95, 7.9, 1.7, [[("AI-аудит требований ", {}), ("до старта разработки", {"color": GOLD2})]],
         size=36, font=H_FONT, spacing=0.95)
    text(s, 0.6, 4.55, 7.6, 1.0,
         "Статическая проверка ТЗ по стандартам ISO/IEC/IEEE 29148 и IEEE 830. Каждое замечание — "
         "с дословной цитатой, REQ-ID и пунктом стандарта.", size=16, color=MUTED, spacing=1.15)
    pill(s, 0.6, 5.75, 5.9, 0.56, f"{TEAM}  ·  Версия v1.0.2 (184 теста)", size=14)
    text(s, 0.6, 6.45, 7.5, 0.35, "«Дәйек» по-казахски — доказательство. Без цитаты замечание не принимается.",
         size=12, color=DIM)
    picture(s, BOB / "01_bob_mascot_base.jpg", 8.75, 0.95, 3.95, 5.9, focus_y=0.3)
    star(s, 8.5, 1.0, 0.3, GOLD2, a=95)
    star(s, 12.55, 5.95, 0.24, GOLD2, a=95)
    notes(s, 1, "0:00–0:20", "Добрый день! Мы — Команда «ВУПИПАЙ», проект DAIEK: AI-аудит требований, который ловит "
          "дефекты в ТЗ до того, как они станут багами в коде. «Дәйек» по-казахски — доказательство: у каждого "
          "нашего замечания есть дословная цитата. Все цифры дальше — из реального прогона релиза v1.0.2.")


def s2(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); background(s, 2)
    title(s, "Знакомьтесь", [[("Это Боб — ", {}), ("QA-инженер в FinTech", {"color": GOLD2})]], w=8)
    picture(s, SCENE / "slide_01_bob_intro_crop.jpg", 8.75, 0.95, 3.95, 5.9, focus_y=0.2)
    text(s, 0.6, 2.25, 7.7, 1.1, "Боб отвечает за то, чтобы новая фича карты не превратилась в инцидент. "
         "Сегодня у тебя, Боб, свежее ТЗ из банка:", size=17, color=MUTED, spacing=1.15)
    rect(s, 0.6, 3.25, 7.75, 1.25, CARD, line=GOLD, line_w=1.5)
    text(s, 0.85, 3.37, 7.3, 0.4, "Кейс БЦК", size=11, color=GOLD2, bold=True)
    text(s, 0.85, 3.65, 7.3, 0.5, "«Уведомление о расходах по карте»", size=22, font=H_FONT)
    text(s, 0.85, 4.1, 7.3, 0.35, "месячный порог расходов · push-уведомление · REST API настройки порога",
         size=12.5, color=MUTED)
    card(s, 0.6, 4.75, 3.75, 1.95, "Ожидание", "Прочитать за кофе, кивнуть и отдать в спринт.", accent=PASS, body_size=15)
    card(s, 4.6, 4.75, 3.75, 1.95, "Реальность",
         [[("57", {"color": WHITE, "bold": True}), (" требований в двух частях — бизнес и REST API. Явный ID — только у ", {}),
           ("4 из 57", {"color": WHITE, "bold": True}), (".", {})]], accent=FATAL, body_size=15)
    notes(s, 2, "0:20–0:50", "Знакомьтесь, это Боб. Боб — QA-инженер в финтехе. Боб, сегодня у тебя новое ТЗ: "
          "«Уведомление о расходах по карте» — месячный порог, push и REST API. Ожидание: прочитать за кофе и "
          "отдать в спринт. Реальность: 57 требований в двух частях, и явный идентификатор есть только у 4 из 57.")


def s3(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); background(s, 3)
    title(s, "Боль", [[("30 ТЗ в месяц ", {}), ("и скрытые мины до кода", {"color": GOLD2})]])
    picture(s, SCENE / "slide_03_hundreds_of_tz_crop.jpg", 0.6, 2.15, 3.55, 4.55, focus_y=0.3)
    picture(s, SCENE / "slide_02_found_bugs_crop.jpg", 4.35, 2.15, 2.9, 4.55, focus_y=0.25)
    text(s, 7.55, 2.15, 5.2, 0.45, "У ТЗ раздвоение личности, Боб:", size=17, font=H_FONT)
    mines = [
        ("GET сохраняет настройки", "REST type: GET у метода /alerts/saveLimit, который «заменяет настройку»"),
        ("amount меняет тип", "integer в запросе ↔ string в ответе"),
        ("«В момент покупки» ↔ клиринг", "цель обещает push сразу, а в расходы входят проведённые покупки"),
    ]
    for i, (h, b) in enumerate(mines):
        y = 2.72 + i * 0.95
        rect(s, 7.55, y, 5.2, 0.82, FATAL, radius=0.15, a=12)
        rect(s, 7.55, y, 0.07, 0.82, FATAL, shape=MSO_SHAPE.RECTANGLE)
        text(s, 7.8, y + 0.08, 4.85, 0.32, h, size=13.5, bold=True)
        text(s, 7.8, y + 0.4, 4.85, 0.4, b, size=11, color=MUTED)
    rect(s, 7.55, 5.65, 5.2, 1.05, CARD, line=GOLD, line_w=1.5)
    text(s, 7.8, 5.72, 1.7, 0.9, "5–15×", size=32, font=H_FONT, color=GOLD2, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 9.55, 5.75, 3.1, 0.9, "дороже исправить дефект требований на тестировании или в проде, чем на ревью "
         "(NIST 02-3, IBM SSI)", size=10.5, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, 3, "0:50–1:25", "Боб, у тебя 30 таких ТЗ в месяц, а ты один. И у этого ТЗ раздвоение личности: в одном "
          "месте GET сохраняет настройки, в другом amount в запросе integer, а в ответе string, а «в момент покупки» "
          "спорит с проведёнными покупками, то есть с клирингом. Найдёшь это на тестировании — заплатишь в 5–15 раз "
          "дороже, чем на ревью: это порядок величин из NIST и IBM.")


def s4(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); background(s, 4)
    title(s, "Поиск решения", [[("Почему обычный ChatGPT ", {}), ("не подходит для банка", {"color": GOLD2})]])
    picture(s, SCENE / "slide_04_many_ai_agents_crop.jpg", 9.1, 2.15, 3.6, 4.55, focus_y=0.3)
    text(s, 0.6, 2.1, 8.2, 0.5, [[("Галлюцинации и лишние голоса: ", {"color": WHITE, "bold": True}),
                                 ("модель уверенно говорит то, чего в ТЗ нет.", {})]], size=15, color=MUTED)
    cards = [
        ("1", "Галлюцинации цитат", "Пересказывает «своими словами» и цитирует фразы, которых в ТЗ нет. "
         "Замечание нельзя проверить.", FATAL),
        ("2", "Потеря кросс-секций", "Не видит, что REST type в части B спорит с «Логикой работы метода» и целями ТЗ.", MAJOR),
        ("3", "Риск утечки ПДн", "Номера карт, телефоны и токены из ТЗ уходят во внешний API как есть.", VIOLET),
        ("4", "Ошибки в расчётах", "Считает дефекты, индекс готовности и ROI «на глаз» — каждый раз по-разному.", BLUE),
    ]
    for i, (ic, h, b, c) in enumerate(cards):
        x, y = 0.6 + (i % 2) * 4.2, 2.75 + (i // 2) * 2.0
        card(s, x, y, 4.0, 1.8, h, b, accent=c, icon=ic, head_size=14, body_size=12)
    notes(s, 4, "1:25–1:55", "Боб пробует обычный чат-бот и слышит лишние голоса: модель уверенно цитирует то, чего в "
          "документе нет, теряет связь между частью A и REST API, отправляет номера карт во внешний API как есть и "
          "считает метрики на глаз. Для банка каждый из этих пунктов — стоп-фактор.")


def s5(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); background(s, 5)
    title(s, "Решение", [[("Боб выбирает ", {}), ("DAIEK", {"color": GOLD2})]], w=8.2)
    text(s, 0.6, 1.95, 8.2, 0.4, "каз. «дәйек» — доказательство: замечание без цитаты не принимается",
         size=14, color=MUTED)
    picture(s, SCENE / "slide_05_selected_daiek_crop.jpg", 9.1, 0.95, 3.6, 5.9, focus_y=0.3)
    # два интерфейса над одним ядром
    for i, (h, cmd, items, c) in enumerate([
        ("CLI для CI/CD", "uv run req-validator analyze ТЗ.md\n  --format md,pdf,html",
         ["одна команда — отчёт MD / HTML / PDF / JSON", "встраивается в пайплайн перед спринтом",
          "демо без сети: --demo, диагностика: doctor"], BLUE),
        ("Web UI с Бобом", "uv run req-validator web --open",
         ["drag & drop .md / .txt / .pdf", "дашборд: дефекты, воронка, ROI, паспорт",
          "Боб комментирует каждый этап проверки"], VIOLET),
    ]):
        x = 0.6 + i * 4.25
        rect(s, x, 2.6, 4.05, 3.35, CARD, line=c, line_w=1.5)
        text(s, x + 0.25, 2.75, 3.6, 0.45, h, size=18, font=H_FONT)
        rect(s, x + 0.25, 3.3, 3.55, 0.85, "0A1028", radius=0.1, line=c)
        text(s, x + 0.4, 3.4, 3.3, 0.7, cmd, size=10.5, font=M_FONT, color="C7D2FE", anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.25, 4.3, 3.6, 1.6, [[("✓  ", {"color": PASS, "bold": True}), (t, {})] for t in items],
             size=12, color=MUTED, space_after=5)
    rect(s, 0.6, 6.15, 8.3, 0.6, GOLD, radius=0.5, a=12, line=GOLD)
    text(s, 0.85, 6.15, 7.9, 0.6, "Одно ядро: парсинг, ID, цитаты, метрики и ROI — детерминированный код; "
         "LLM — только смысл.", size=12.5, color=GOLD2, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, 5, "1:55–2:25", "Поэтому Боб выбирает DAIEK. Два интерфейса над одним ядром: CLI — одна команда, "
          "встраивается в CI/CD; и веб-интерфейс, где Боб сам комментирует ход проверки. Правило одно: замечание без "
          "цитаты — не замечание. Всё, что можно посчитать кодом, считает код; модели остаётся только смысл.")


def s6(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); background(s, 6)
    title(s, "Архитектура", [[("5 уровней защиты ", {}), ("от галлюцинаций и падений", {"color": GOLD2})]])
    rect(s, 0.6, 2.05, 12.13, 2.0, WHITE, radius=0.08, line=GOLD, line_w=1.5)
    picture(s, BRAND / "daiek_pipeline_architecture.jpg", 2.6, 2.1, 8.13, 1.9, rounded=False, line=None)
    steps = [
        ("Pass 0", "PII-маска + Registry", "ПДн маскируются до LLM; реестр: 3 термина, 6 ролей, 15 числовых фактов", BLUE),
        ("Pass 1", "ISO 29148", "проверено 57 из 57 требований, 12 батчей", VIOLET),
        ("Pass 2", "2-модельный ансамбль", "две модели независимо; совпало у обеих: 3", "A855F7"),
        ("Pass 3a", "RapidFuzz 22/22", "каждая цитата найдена в тексте; отклонено 0", GOLD),
        ("Pass 3b", "Критик + FATAL Gate", "same_model: 0 · подтвердил 5, понизил 3, отклонил 9", FATAL),
    ]
    w, gap = 2.28, 0.18
    for i, (p, h, b, c) in enumerate(steps):
        x = 0.6 + i * (w + gap)
        rect(s, x, 4.3, w, 2.45, CARD, line=c, line_w=1.5)
        pill(s, x + 0.18, 4.45, 1.05, 0.34, p, fill=c, fill2=None, size=10.5,
             color=NAVY if c in (GOLD,) else WHITE)
        text(s, x + 0.18, 4.9, w - 0.32, 0.65, h, size=13.5, font=H_FONT, spacing=0.95)
        text(s, x + 0.18, 5.6, w - 0.32, 1.1, b, size=11, color=MUTED, spacing=1.08)
        if i < len(steps) - 1:
            arr = rect(s, x + w + 0.01, 5.4, 0.16, 0.24, c, shape=MSO_SHAPE.CHEVRON)
            del arr
    notes(s, 6, "2:25–3:10", "Внутри — пять уровней защиты. Pass 0 маскирует персональные данные и строит "
          "глобальный реестр документа: термины, роли, числовые факты. Pass 1 проверяет каждое из 57 требований по "
          "ISO 29148. Pass 2 — две разные модели независимо ищут противоречия во всём наборе. Pass 3a — Grounding "
          "Gate: RapidFuzz сверяет каждую цитату с текстом, 22 из 22 подтверждены, галлюцинаций ноль. Pass 3b — "
          "Критик другой модели в чистом контексте, same_model ноль: из 17 находок он подтвердил 5, понизил 3 и "
          "отклонил 9 придирок. И FATAL-гейт: критический дефект не компенсируется никакими баллами.")


def s7(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); background(s, 7)
    title(s, "Живой кейс БЦК · requirements.md", [[("Вердикт: ", {}), ("REJECTED", {"color": FATAL}),
                                                  ("  ·  68.5 / 100", {"color": GOLD2})]])
    # KPI-ряд
    vals = [("57", "требований", WHITE, VIOLET), ("20", "дефектов в метриках", GOLD2, GOLD),
            ("1 / 7 / 12", "FATAL/MAJOR/MINOR", WHITE, FATAL), ("28", "подсказок вне метрик", MINOR, BLUE)]
    for i, (v, lbl, col, brd) in enumerate(vals):
        kpi(s, 0.6 + i * 2.2, 2.1, 2.05, 1.1, v, lbl, color=col, vsize=24 if "/" in v else 28, border=brd)
    picture(s, BOB / "03_slide2_bob_found_bug.jpg", 9.55, 2.1, 3.18, 4.65, focus_y=0.2, line=FATAL)
    st = rect(s, 9.75, 5.55, 2.8, 0.55, WHITE, radius=0.15, line=FATAL, line_w=3)
    st.rotation = -10
    tf = st.text_frame
    tf.text = "FATAL GATE · REJECTED"
    r = tf.paragraphs[0].runs[0]
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = M_FONT, Pt(14), True, rgb(FATAL)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    defs = [
        ("FATAL", "DEF-001", "GET для сохранения", "REST type: GET  ·  /alerts/saveLimit", "→ POST или PUT", FATAL),
        ("MAJOR", "DEF-004", "Нет метода чтения", "«Сохранение порога. Функциональные требования: чтение порога»",
         "→ описать метод чтения настроек и расходов", MAJOR),
        ("MAJOR", "DEF-006", "amount: integer ↔ string", "запрос: integer, 32 bit  ↔  ответ: string",
         "→ один тип в запросе и ответе", MAJOR),
    ]
    for i, (sev, did, h, q, fix, c) in enumerate(defs):
        y = 3.4 + i * 1.13
        rect(s, 0.6, y, 8.75, 1.0, CARD, line=c, line_w=1.25)
        pill(s, 0.78, y + 0.14, 0.9, 0.3, sev, fill=c, fill2=None, size=10, color=NAVY)
        text(s, 1.78, y + 0.12, 3.0, 0.35, did, size=13, font=H_FONT)
        text(s, 0.78, y + 0.52, 2.95, 0.4, h, size=12.5, bold=True)
        text(s, 3.85, y + 0.12, 5.35, 0.45, q, size=10.5, font=M_FONT, color="F6E7B4")
        text(s, 3.85, y + 0.58, 5.35, 0.35, fix, size=11.5, color="6EE7B7", bold=True)
    notes(s, 7, "3:10–3:50", "Живой кейс. Вердикт — REJECTED, индекс готовности 68,5 из 100. 57 требований, "
          "20 дефектов: 1 FATAL, 7 MAJOR, 12 MINOR и ещё 28 подсказок вне метрик. Главный — DEF-001: GET "
          "/alerts/saveLimit для сохранения порога. DEF-004: в описании заявлено чтение порога, а метода чтения нет. "
          "DEF-006: amount — integer в запросе и string в ответе. По каждому — цитата из документа и готовая правка.")


def s8(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); background(s, 8)
    title(s, "Сравнение", [[("Ручное ревью vs ChatGPT vs ", {}), ("DAIEK", {"color": GOLD2})]])
    rows = [
        ("", "Ручное ревью", "Обычный ChatGPT", "DAIEK («ВУПИПАЙ»)"),
        ("Время на ТЗ (57 треб.)", "1,4–1,9 ч только первичное ревью", "минуты, но без гарантии полноты",
         "демо ~1 с; полный LLM-аудит 704 с, 23 вызова"),
        ("Проверка цитат", "глазами", "нет — цитаты могут быть выдуманы", "RapidFuzz: 22/22, галлюцинаций 0"),
        ("Ложные придирки", "зависят от ревьюера", "не фильтруются", "Критик (same_model: 0) отклонил 9 из 17"),
        ("Защита ПДн", "внутри банка", "текст уходит во внешний API как есть", "маскирование до отправки в LLM"),
        ("Стандарты ISO / IEEE", "зависит от опыта", "«в общем»", "ISO 29148 п. 5.2.5, 5.2.6; IEEE 830 п. 4.3"),
        ("Отчёт", "заметки и переписка", "ответ в чате", "PDF (Typst) + MD + HTML + паспорт прогона"),
    ]
    shp = s.shapes.add_table(len(rows), 4, Inches(0.6), Inches(2.1), Inches(12.13), Inches(4.6))
    tbl = shp.table
    tblpr = shp._element.graphic.graphicData.tbl.tblPr
    tblpr.set("bandRow", "0")
    tblpr.set("firstRow", "0")
    style = tblpr.find(qn("a:tableStyleId"))
    if style is not None:
        style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # No Style, Table Grid
    widths = [2.45, 2.9, 3.0, 3.78]
    for i, w in enumerate(widths):
        tbl.columns[i].width = Inches(w)
    for r, row in enumerate(rows):
        tbl.rows[r].height = Inches(0.6 if r else 0.55)
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            head = r == 0
            daiek = c == 3
            cell.fill.fore_color.rgb = rgb("2A1F6B" if head and daiek else "1B2452" if head else
                                           "1F2A1E" if False else "17204A" if daiek else CARD if r % 2 else "121B40")
            cell.margin_left = cell.margin_right = Inches(0.12)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = ""
            run = p.add_run()
            run.text = (("✓ " if daiek and not head else "") + val)
            f = run.font
            f.name = H_FONT if head or c == 0 else B_FONT
            f.size = Pt(13 if head else 12 if c == 0 else 11.5)
            f.bold = daiek and not head
            f.color.rgb = rgb(GOLD2 if head and daiek else WHITE if head or c == 0 else "B7F7DD" if daiek else MUTED)
    table_h = 0.55 + 0.6 * (len(rows) - 1)
    frame = rect(s, 0.6 + sum(widths[:3]), 2.08, widths[3], table_h + 0.04, None, line=GOLD, line_w=2.25, radius=0.03)
    del frame
    notes(s, 8, "3:50–4:20", "Сравнение. Ручное ревью надёжно, но медленно и зависит от опыта ревьюера. Чат-бот "
          "быстрый, но без проверки цитат и с риском для персональных данных. DAIEK: 22 из 22 цитат подтверждены, "
          "Критик отсеял 9 из 17 придирок, ПДн маскируются до LLM, правила ISO 29148 и IEEE 830 разложены на "
          "атомарные проверки, отчёт — в PDF.")


def s9(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); background(s, 9)
    title(s, "Бизнес-ценность", [[("Боб продуктивен. ", {}), ("БЦК экономит бюджет", {"color": GOLD2})]], w=8.4)
    picture(s, SCENE / "slide_06_happy_bob_savings_crop.jpg", 9.1, 0.95, 3.6, 5.9, focus_y=0.3)
    big = [
        ("236–314 тыс. KZT", "на 1 ТЗ  ·  23,6–31,4 ч", "консервативный и базовый сценарий"),
        ("7,1–9,4 млн KZT/мес", "на поток 30 ТЗ/мес  ·  708–942 ч", "линейная экстраполяция"),
    ]
    for i, (v, l1, l2) in enumerate(big):
        y = 2.1 + i * 1.62
        r = rect(s, 0.6, y, 8.25, 1.45, GOLD, radius=0.12, line=GOLD, line_w=1.5)
        grad(r, "3A2F12", "201A4E", 0)
        text(s, 0.9, y + 0.12, 7.8, 0.75, v, size=36, font=H_FONT, color=GOLD2, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 0.9, y + 0.88, 4.6, 0.4, l1, size=14, bold=True)
        text(s, 5.4, y + 0.9, 3.3, 0.4, l2, size=11, color=MUTED, align=PP_ALIGN.RIGHT)
    kpi(s, 0.6, 5.4, 2.2, 1.3, "$0", "бюджет на LLM API", color=PASS, vsize=34, border=PASS)
    rect(s, 3.0, 5.4, 5.85, 1.3, CARD, line=VIOLET)
    text(s, 3.2, 5.5, 5.5, 1.15, [
        [("Как считаем: ", {"color": WHITE, "bold": True}),
         ("Decimal, без LLM. Rework (1×16 + 7×6 + 12×0,25) ч × 50 % пропуска ручным ревью + первичное ревью "
          "57 × 2 мин − разбор отчёта; 10 000 ₸/ч. Все коэффициенты — явные допущения в .env.", {})],
        [("Боб, это не галлюцинация — это Decimal.", {"color": GOLD2, "bold": True, "size": 13})],
    ], size=11, color=MUTED, spacing=1.1, space_after=8)
    notes(s, 9, "4:20–4:50", "Деньги. Боб, это не галлюцинация — это Decimal. На одном ТЗ экономия 23,6–31,4 часа, "
          "то есть 236–314 тысяч тенге. На потоке 30 ТЗ в месяц — 708–942 часа, 7,1–9,4 миллиона тенге в месяц. "
          "Бюджет на API — ноль долларов: бесплатные тарифы и локальная модель. Все коэффициенты вынесены в .env "
          "как явные допущения.")


def s10(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); background(s, 10)
    title(s, "Безопасность · Roadmap · Команда", [[("Готово к банку. ", {}), ("Дальше — Jira и Confluence", {"color": GOLD2})]])
    rect(s, 0.6, 2.1, 4.1, 2.35, WHITE, radius=0.08, line=GOLD, line_w=1.5)
    picture(s, BRAND / "daiek_finops_security_shield.jpg", 0.65, 2.15, 4.0, 2.25, line=None)
    risks = [("429 / 503", "каскад провайдеров, backoff с jitter, кэш шагов SQLite"),
             ("184 автотеста", "ядро v1.0.2 без сети: фейковый транспорт LLM"),
             ("Битый / пустой PDF", "CANNOT_PROCEED с диагностикой, без трейсбека"),
             ("ПДн", "маскирование карт, телефонов, токенов до LLM")]
    rect(s, 4.9, 2.1, 4.0, 4.65, CARD, line=PASS, line_w=1.25)
    text(s, 5.12, 2.22, 3.6, 0.4, "Закрытые риски", size=16, font=H_FONT)
    for i, (h, b) in enumerate(risks):
        y = 2.75 + i * 0.98
        text(s, 5.12, y, 3.6, 0.32, [[("✓  ", {"color": PASS}), (h, {})]], size=13, bold=True)
        text(s, 5.47, y + 0.32, 3.3, 0.6, b, size=11, color=MUTED)
    road = [("Jira", "дефекты → задачи с REQ-ID и приоритетом P0–P2"),
            ("Confluence", "проверка страницы ТЗ по кнопке или webhook"),
            ("CI/CD", "гейт качества ТЗ перед стартом спринта")]
    rect(s, 9.1, 2.1, 3.63, 4.65, CARD, line=BLUE, line_w=1.25)
    text(s, 9.32, 2.22, 3.3, 0.4, "Roadmap", size=16, font=H_FONT)
    for i, (h, b) in enumerate(road):
        y = 2.75 + i * 1.3
        pill(s, 9.32, y, 1.55, 0.34, h, fill=BLUE, fill2=VIOLET, size=11)
        text(s, 9.32, y + 0.45, 3.25, 0.75, b, size=11.5, color=MUTED)
    team = rect(s, 0.6, 4.65, 4.1, 2.1, VIOLET, radius=0.1)
    grad(team, VIOLET, BLUE, 0)
    text(s, 0.85, 4.75, 3.7, 0.25, "Мы —", size=11, color="E0E7FF", bold=True)
    text(s, 0.85, 4.97, 3.7, 0.4, TEAM, size=18, font=H_FONT)
    text(s, 0.85, 5.42, 3.7, 0.85, [[("•  ", {"color": GOLD2}), (n, {})] for n in MEMBERS],
         size=13, bold=True, spacing=1.05)
    text(s, 0.85, 6.35, 3.7, 0.3, "github.com/majitiliyas/bcc-req-validator", size=10.5, color="E0E7FF")
    notes(s, 10, "4:50–5:20", "Надёжность: при 429 срабатывает каскад провайдеров с backoff и кэшем, ядро покрыто "
          "184 автотестами, битые и пустые файлы дают понятный CANNOT_PROCEED вместо трейсбека, ПДн маскируются до "
          "LLM. Дальше — интеграция с Jira и Confluence: дефекты становятся задачами с REQ-ID и приоритетом. Мы — "
          "Команда «ВУПИПАЙ»: Елубай Илияс, Мубарак Мансур и Мылтыкбай Улпан. Спасибо! Боб готов к вопросам.")


def write_notes_md() -> None:
    titles = {1: "Титул", 2: "Знакомство", 3: "Боль", 4: "Поиск решения", 5: "Решение", 6: "Архитектура",
              7: "Живой кейс БЦК", 8: "Сравнение", 9: "Бизнес-ценность (ROI)", 10: "Безопасность, Roadmap и Команда"}
    lines = [
        "# DAIEK — текст спикера (Команда «ВУПИПАЙ»)",
        "",
        "Финал BCC-HUB × MNU FinTech Hackathon · 10 слайдов · ≈ 5 мин 20 с.",
        "",
        "Тон — как в «Знакомьтесь, Боб»: закадровый голос обращается к Бобу на «ты», лёгкая ирония "
        "«ожидание vs реальность», но цифры строго из релиза v1.0.2 (коммит `66340f6`).",
        "",
        "Проверенные цифры: 57 требований · 20 дефектов (1 FATAL / 7 MAJOR / 12 MINOR) · 28 подсказок вне "
        "метрик · индекс 68.5 / 100 · REJECTED (DEF-001) · цитаты 22 / 22 · Критик same_model: 0 "
        "(подтвердил 5, понизил 3, отклонил 9) · ROI 23,6–31,4 ч (236–314 тыс. KZT) на 1 ТЗ · 708–942 ч "
        "(7,1–9,4 млн KZT/мес) на 30 ТЗ/мес · $0.",
        "",
    ]
    for n in range(1, TOTAL + 1):
        timing, body = NOTES[n]
        lines += [f"## Слайд {n}. {titles[n]} · {timing}", "", body, ""]
    lines += [
        "## Если спросят",
        "",
        "- **Откуда 5–15×?** Порядок величин из NIST Planning Report 02-3 (2002) и IBM Systems Sciences "
        "Institute; в ROI это явные коэффициенты в `.env`, а не вывод модели.",
        "- **Почему 68.5, а всё равно REJECTED?** FATAL-гейт некомпенсаторный: подтверждённый критический "
        "дефект (DEF-001) блокирует старт при любом индексе.",
        "- **Чем Критик отличается от генератора?** Другая модель (`same_model: 0`), чистый контекст, "
        "проверяет только FATAL/MAJOR; из 17 находок отклонил 9.",
        "- **Что будет с пустым или битым файлом?** Статус `EMPTY_DOCUMENT` / `CORRUPTED_FILE`, вердикт "
        "`CANNOT_PROCEED`, LLM не вызывается — покажем это в веб-интерфейсе.",
        "- **Демо вживую:** `uv run req-validator web --open` → «⚡ Кейс БЦК (отчёт v1.0.2)» и "
        "«🧪 Прогнать тестовое ТЗ с багами».",
        "",
    ]
    (OUT / "speaker_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(int(13.333 * 914400)), Emu(int(7.5 * 914400))
    for build in (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10):
        build(prs)
    path = OUT / "DAIEK_WoopiePie_BCC_Hackathon.pptx"
    prs.save(path)
    write_notes_md()
    print(path)


if __name__ == "__main__":
    main()
