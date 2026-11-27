"""Сборка 2-минутного видео-демо DAIEK (Команда «ВУПИПАЙ») в стиле «Знакомьтесь, Боб».

    uv run --with pillow --with imageio-ffmpeg python presentation/video/build_video.py [--no-music] [--skip-tts]

Шаги:
1. Озвучка: edge-tts (ru-RU-SvetlanaNeural) по сегментам раскадровки; темп подгоняется под слот сегмента.
   Резерв — системный голос macOS `say -v Milena`. Текст озвучки фонетически выровнен (DAIEK → «Дайек» и т. п.).
2. Субтитры .srt по сегментам (время фраз — пропорционально длине произносимого текста внутри сегмента).
3. Кадры 1920×1080 / 30 fps на Pillow (плавные zoom/pan, crossfade, анимированные плашки, бейдж команды),
   кодирование H.264 + AAC через ffmpeg из пакета imageio-ffmpeg.

Цифры — строго из релиза v1.0.2 (коммит 66340f6). Секреты не используются: сеть нужна только edge-tts.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SHOTS = HERE / "shots"
BOB = ROOT / "src/req_validator/web/static/bob"
BRAND = ROOT / "src/req_validator/web/static/brand"
WORK = HERE / "build"
OUT_MP4 = HERE / "DAIEK_WoopiePie_Demo.mp4"
FF = imageio_ffmpeg.get_ffmpeg_exe()

W, H, FPS, DUR = 1920, 1080, 30, 120.0
TEAM_BADGE = "Команда «ВУПИПАЙ» · BCC-HUB × MNU"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
F_BLACK, F_BOLD, F_REG = FONT_DIR / "Arial Black.ttf", FONT_DIR / "Arial Bold.ttf", FONT_DIR / "Arial.ttf"

NAVY, NAVY2, CARD = (7, 13, 34), (21, 16, 73), (15, 23, 54)
VIOLET, BLUE, GOLD, GOLD2 = (139, 92, 246), (59, 130, 246), (212, 175, 55), (241, 210, 122)
FATAL, MAJOR, PASS, WHITE, MUTED = (239, 68, 68), (245, 158, 11), (16, 185, 129), (255, 255, 255), (174, 182, 218)

# --------------------------------------------------------------------------- #
# Текст: сегменты раскадровки (слоты строго по ТЗ на ролик)
# --------------------------------------------------------------------------- #

SPOKEN = [  # отображаемое → произносимое (фонетическое выравнивание для TTS)
    ("QA-инженер", "кью-эй инженер"), ("FinTech", "финтех"), ("GET-запросе", "гет-запросе"),
    ("integer", "интеджер"), ("string", "стринг"), ("DAIEK", "Дайек"), ("БЦК", "бэ-цэ-ка"),
    ("REJECTED", "риджектед"), ("API", "эй-пи-ай"), ("«Вупипай»", "Вупипай"),
]


def spoken(text: str) -> str:
    for a, b in SPOKEN:
        text = text.replace(a, b)
    return re.sub(r"\bТЗ\b", "тэ-зэ", text)


@dataclass
class Segment:
    start: float
    end: float
    lines: list[tuple[str, str]]  # (субтитр, произносимое; пусто → spoken(субтитр))
    rate: int = -4
    lead: float = 0.35  # старт реплики относительно начала слота (может быть < 0)
    audio: Path | None = None
    speech: float = 0.0

    def say(self) -> str:
        return " ".join(s or spoken(d) for d, s in self.lines)


SEGMENTS = [
    Segment(0, 22, [
        ("Знакомьтесь, это Боб.", ""),
        ("Сегодня Боб — QA-инженер в FinTech-команде, и он свято верит,", ""),
        ("что новое ТЗ на уведомления по карте пройдёт как по маслу.", ""),
        ("57 требований, таблицы, критерии приёмки — звучит мирно.",
         "Пятьдесят семь требований, таблицы, критерии приёмки — звучит мирно."),
        ("Но в банковских ТЗ спокойствие обычно длится ровно до первой строчки кода.", ""),
    ]),
    Segment(22, 48, [
        ("Приглядись повнимательнее: сохранение порога висит на GET-запросе,", ""),
        ("метод чтения настроек вообще забыли описать,", ""),
        ("а сумма в одном месте integer, а в другом уже string.", ""),
        ("Если пропустить это сейчас и поймать на тесте или в проде,", ""),
        ("переделки обойдутся в разы дороже.", ""),
    ], rate=-8),
    Segment(48, 72, [
        ("Боб мог бы просто кинуть документ в обычный чат.", ""),
        ("Но там легко получить выдуманную цитату, потерять связь между разделами", ""),
        ("или отправить в облако лишние данные.", ""),
        ("Поэтому Боб запускает DAIEK — нашего агента проверки требований.", ""),
    ], rate=-8),
    Segment(72, 102, [
        ("Внутри DAIEK код и модели работают в связке:", ""),
        ("детерминированные правила и маскирование данных держат базу,", ""),
        ("цитаты сверяются по тексту, а независимый Критик отсеивает лишний шум.", ""),
        ("На кейсе БЦК система нашла 20 дефектов и 28 подсказок,",
         "На кейсе бэ-цэ-ка система нашла двадцать дефектов и двадцать восемь подсказок,"),
        ("проверила цитаты и поставила честный REJECTED до старта разработки.", ""),
    ], rate=-8),
    Segment(102, 120, [
        ("Итог: Боб не тушит пожары ночью, а команда экономит", ""),
        ("от 236 до 314 тысяч тенге на одном ТЗ —",
         "от двухсот тридцати шести до трёхсот четырнадцати тысяч тенге на одном тэ-зэ —"),
        ("или от 7,1 до 9,4 млн тенге в месяц на потоке 30 документов",
         "или от семи с лишним до девяти с лишним миллионов в месяц на потоке тридцати документов"),
        ("при нулевом бюджете на API.", ""),
        ("DAIEK от команды «Вупипай» — проверяем ТЗ до того, как оно станет болью.", ""),
    ], rate=6, lead=-2.2),  # итог начинается под конец паспорта, чтобы не тараторить
]
LEAD_IN = 0.35  # пауза перед репликой в начале слота


# --------------------------------------------------------------------------- #
# Озвучка
# --------------------------------------------------------------------------- #


def media_duration(path: Path) -> float:
    out = subprocess.run([FF, "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0


def tts(text: str, rate: int, out: Path) -> str:
    """edge-tts (Светлана) с повторами; резерв — macOS say (Милена). Возвращает имя голоса."""
    uvx = shutil.which("uvx")
    for attempt in range(10):  # сервис иногда отвечает пустым потоком — повторяем с паузой
        if uvx is None:
            break
        if attempt:
            time.sleep(2 + attempt)
        out.unlink(missing_ok=True)
        subprocess.run([uvx, "edge-tts", "--voice", "ru-RU-SvetlanaNeural", f"--rate={rate:+d}%", "--text", text,
                        "--write-media", str(out)], capture_output=True)
        if out.exists() and media_duration(out) > 1.0:
            return "edge-tts ru-RU-SvetlanaNeural"
    aiff = out.with_suffix(".aiff")
    words_per_min = int(175 * (1 + rate / 100))
    subprocess.run(["say", "-v", "Milena", "-r", str(words_per_min), "-o", str(aiff), text], check=True)
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(aiff), str(out)], check=True)
    return "macOS say Milena"


def build_voice(skip: bool) -> list[str]:
    WORK.mkdir(parents=True, exist_ok=True)
    voices = []
    for i, seg in enumerate(SEGMENTS, 1):
        seg.audio = WORK / f"seg{i}.mp3"
        slot = seg.end - seg.start - seg.lead - 0.7
        if not (skip and seg.audio.exists()):
            for _ in range(4):
                voices.append(tts(seg.say(), seg.rate, seg.audio))
                d = media_duration(seg.audio)
                if d <= slot or seg.rate >= 18:
                    break
                seg.rate = min(18, seg.rate + max(2, math.ceil((d / slot - 1) * 100)))  # чуть быстрее — и в слот
        seg.speech = media_duration(seg.audio)
        print(f"сегмент {i}: {seg.speech:.1f} с из слота {seg.end - seg.start:.0f} с (темп {seg.rate:+d}%)")
    return voices


def build_audio(music: bool) -> Path:
    """Озвучка по слотам + выравнивание громкости (+ очень тихий фоновый пэд)."""
    inputs, filters = [], []
    for i, seg in enumerate(SEGMENTS):
        inputs += ["-i", str(seg.audio)]
        ms = int((seg.start + seg.lead) * 1000)
        filters.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={ms}|{ms}[v{i}]")
    n = len(SEGMENTS)
    mix = "".join(f"[v{i}]" for i in range(n))
    filters.append(f"{mix}amix=inputs={n}:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7,apad,atrim=0:{DUR}[voice]")
    if music:
        pad = ("aevalsrc='0.5*sin(2*PI*110*t)+0.35*sin(2*PI*164.81*t)+0.3*sin(2*PI*220*t)+0.2*sin(2*PI*277.18*t)"
               f"*(0.6+0.4*sin(2*PI*0.05*t))':s=48000:d={DUR}")
        inputs += ["-f", "lavfi", "-i", pad]
        filters.append(f"[{n}:a]lowpass=f=900,volume=0.035,aformat=channel_layouts=stereo,"
                       f"afade=t=in:d=3,afade=t=out:st={DUR - 4}:d=4[bg]")
        filters.append("[voice][bg]amix=inputs=2:normalize=0,afade=t=out:st=119.0:d=1.0[out]")
    else:
        filters.append("[voice]afade=t=out:st=119.0:d=1.0[out]")
    out = WORK / "soundtrack.m4a"
    subprocess.run([FF, "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]",
                    "-ar", "48000", "-c:a", "aac", "-b:a", "192k", "-t", str(DUR), str(out)], check=True)
    return out


# --------------------------------------------------------------------------- #
# Субтитры и текст озвучки
# --------------------------------------------------------------------------- #


@dataclass
class Cue:
    start: float
    end: float
    text: str


def build_cues() -> list[Cue]:
    cues = []
    for seg in SEGMENTS:
        t0 = seg.start + seg.lead
        weights = [len(s or spoken(d)) for d, s in seg.lines]
        total = sum(weights)
        t = t0
        for (disp, _), w in zip(seg.lines, weights):
            dt = seg.speech * w / total
            cues.append(Cue(t, t + dt, disp))
            t += dt
    return cues


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_texts(cues: list[Cue], voices: list[str]) -> None:
    (HERE / "subtitles.srt").write_text(
        "\n".join(f"{i}\n{srt_time(c.start)} --> {srt_time(min(c.end, DUR))}\n{c.text}\n" for i, c in enumerate(cues, 1)),
        encoding="utf-8")
    titles = ["Знакомство с Бобом и ловушка спринта", "«Раздвоение» ТЗ и скрытые баги",
              "Почему «просто закинуть в чат» не спасает", "Работа DAIEK и живой кейс БЦК",
              "Экономия в тенге и финал команды «ВУПИПАЙ»"]
    lines = ["DAIEK — видео-демо (Команда «ВУПИПАЙ»), 120 с, 1920×1080, 30 fps", "",
             f"Голос: {', '.join(sorted(set(voices))) or 'из кэша build/'}; тон — «Знакомьтесь, Боб».", ""]
    for seg, title in zip(SEGMENTS, titles):
        m0, s0, m1, s1 = int(seg.start // 60), int(seg.start % 60), int(seg.end // 60), int(seg.end % 60)
        lines += [f"[{m0:02d}:{s0:02d}–{m1:02d}:{s1:02d}] {title}",
                  "Текст: " + " ".join(d for d, _ in seg.lines),
                  "Для TTS: " + seg.say(), ""]
    (HERE / "voiceover_script.txt").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Графика
# --------------------------------------------------------------------------- #


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def ease(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def rounded_mask(size: tuple[int, int], r: int) -> Image.Image:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], r, fill=255)
    return m


def framed(img: Image.Image, size: tuple[int, int], *, radius=28, border=GOLD, bw=5, focus_y=0.3) -> Image.Image:
    """Картинка «cover» в рамке со скруглением и мягкой тенью → RGBA с полями под тень."""
    iw, ih = img.size
    sc = max(size[0] / iw, size[1] / ih)
    r = img.convert("RGB").resize((math.ceil(iw * sc), math.ceil(ih * sc)), Image.LANCZOS)
    left = (r.width - size[0]) // 2
    top = int((r.height - size[1]) * focus_y)
    r = r.crop((left, top, left + size[0], top + size[1]))
    pad = 40
    out = Image.new("RGBA", (size[0] + 2 * pad, size[1] + 2 * pad), (0, 0, 0, 0))
    shadow = Image.new("RGBA", out.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle([pad, pad + 14, pad + size[0], pad + size[1] + 14], radius, fill=(0, 0, 0, 150))
    out = Image.alpha_composite(out, shadow.filter(ImageFilter.GaussianBlur(18)))
    out.paste(r, (pad, pad), rounded_mask(size, radius))
    if border:
        ImageDraw.Draw(out).rounded_rectangle([pad, pad, pad + size[0] - 1, pad + size[1] - 1], radius, outline=border, width=bw)
    return out


def make_background() -> Image.Image:
    bg = Image.new("RGB", (W, H), NAVY)
    px = bg.load()
    for y in range(0, H, 2):
        for x in range(0, W, 2):
            k = (x / W * 0.55 + (1 - y / H) * 0.45)
            c = tuple(int(NAVY[i] + (NAVY2[i] - NAVY[i]) * k) for i in range(3))
            px[x, y] = c
            if x + 1 < W:
                px[x + 1, y] = c
            if y + 1 < H:
                px[x, y + 1] = c
                if x + 1 < W:
                    px[x + 1, y + 1] = c
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(glow)
    d.ellipse([1300, -520, 2300, 480], fill=(*VIOLET, 46))
    d.ellipse([-420, 640, 520, 1500], fill=(*BLUE, 34))
    bg = Image.alpha_composite(bg.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(120)))
    d = ImageDraw.Draw(bg)
    for (sx, sy, s) in ((90, 260, 9), (1830, 170, 11), (980, 1000, 7), (1760, 940, 8), (160, 960, 6), (1180, 120, 6)):
        d.line([sx - s, sy, sx + s, sy], fill=(255, 255, 255, 170), width=2)
        d.line([sx, sy - s, sx, sy + s], fill=(255, 255, 255, 170), width=2)
    return bg


def pill_image(text: str, *, size=24, c1=VIOLET, c2=BLUE, fg=WHITE, pad=(26, 12), fnt=F_BOLD) -> Image.Image:
    f = font(fnt, size)
    tw = int(f.getlength(text))
    w, h = tw + 2 * pad[0], size + 2 * pad[1] + 4
    grad = Image.new("RGBA", (w, h))
    gd = ImageDraw.Draw(grad)
    for x in range(w):
        k = x / max(1, w - 1)
        gd.line([x, 0, x, h], fill=tuple(int(c1[i] + (c2[i] - c1[i]) * k) for i in range(3)) + (255,))
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(grad, (0, 0), rounded_mask((w, h), h // 2))
    ImageDraw.Draw(out).text((pad[0], pad[1] - 1), text, font=f, fill=fg)
    return out


def plate(title: str, sub: str = "", *, accent=GOLD, width=None, tsize=46, ssize=26, tcolor=GOLD2) -> Image.Image:
    ft, fs = font(F_BLACK, tsize), font(F_BOLD, ssize)
    tw = max(ft.getlength(title), fs.getlength(sub) if sub else 0)
    w = int(width or tw + 70)
    h = 34 + tsize + (ssize + 20 if sub else 0) + 26
    out = Image.new("RGBA", (w + 40, h + 40), (0, 0, 0, 0))
    sh = Image.new("RGBA", out.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([20, 28, 20 + w, 28 + h], 22, fill=(0, 0, 0, 140))
    out = Image.alpha_composite(out, sh.filter(ImageFilter.GaussianBlur(14)))
    d = ImageDraw.Draw(out)
    d.rounded_rectangle([20, 20, 20 + w, 20 + h], 22, fill=(*CARD, 238), outline=accent, width=3)
    d.rounded_rectangle([20, 38, 28, 2 + h], 4, fill=accent)
    d.text((54, 40), title, font=ft, fill=tcolor)
    if sub:
        d.text((54, 40 + tsize + 18), sub, font=fs, fill=MUTED)
    return out


def text_img(text: str, size: int, color=WHITE, fnt=F_BLACK, max_w: int | None = None, spacing=10) -> Image.Image:
    f = font(fnt, size)
    lines = [text] if max_w is None else wrap(text, f, max_w)
    lh = int(size * 1.18)
    w = int(max(f.getlength(ln) for ln in lines)) + 8
    out = Image.new("RGBA", (w, lh * len(lines) + spacing), (0, 0, 0, 0))
    d = ImageDraw.Draw(out)
    for i, ln in enumerate(lines):
        d.text((0, i * lh), ln, font=f, fill=color)
    return out


def wrap(text: str, f: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        nxt = f"{cur} {w}".strip()
        if f.getlength(nxt) <= max_w or not cur:
            cur = nxt
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def paste(frame: Image.Image, img: Image.Image, xy: tuple[float, float], alpha: float = 1.0, scale: float = 1.0) -> None:
    if alpha <= 0.01:
        return
    if abs(scale - 1) > 1e-3:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
    if alpha < 0.999:
        img = img.copy()
        img.putalpha(img.getchannel("A").point(lambda a: int(a * alpha)))
    frame.alpha_composite(img, (int(xy[0]), int(xy[1])))


def appear(t: float, t0: float, dur: float = 0.6) -> float:
    return ease((t - t0) / dur)


def pop(frame, img, x, y, t, t0, *, dy=30, until: float | None = None):
    """Плашка: въезжает снизу с проявлением; опционально гаснет к `until`."""
    a = appear(t, t0)
    if until is not None:
        a *= 1 - appear(t, until - 0.5, 0.5)
    paste(frame, img, (x, y + dy * (1 - a)), a)


# --------------------------------------------------------------------------- #
# Сцены
# --------------------------------------------------------------------------- #


class Assets:
    def __init__(self) -> None:
        self.bg = make_background()
        self.badge = pill_image(TEAM_BADGE, size=25)
        logo = Image.open(BRAND / "daiek_logo_main.jpg").convert("RGB").resize((62, 62), Image.LANCZOS)
        mark = Image.new("RGBA", (360, 76), (0, 0, 0, 0))
        tile = Image.new("RGBA", (70, 70), (255, 255, 255, 255))
        tile.paste(logo, (4, 4))
        mark.paste(tile, (0, 3), rounded_mask((70, 70), 16))
        ImageDraw.Draw(mark).text((86, 12), "DAIEK", font=font(F_BLACK, 38), fill=WHITE)
        self.mark = mark
        self.bob = {k: Image.open(BOB / f"{k}.jpg") for k in (
            "02_slide1_bob_intro", "04_slide3_bob_overwhelmed_tz", "03_slide2_bob_found_bug",
            "05_slide4_bob_choosing_agents", "07_slide6_bob_happy_savings", "01_bob_mascot_base")}
        self.shot = {p.stem: Image.open(p).convert("RGB") for p in SHOTS.glob("*.png")}
        self.cache: dict[str, Image.Image] = {}
        band = Image.new("RGBA", (W, 190), (0, 0, 0, 0))
        bd = ImageDraw.Draw(band)
        for y in range(190):
            bd.line([0, y, W, y], fill=(*NAVY, int(235 * (1 - y / 190) ** 1.6)))
        self.top_band = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.top_band.paste(band, (0, 0))

    def card(self, key: str, size: tuple[int, int], border=GOLD, focus_y=0.3) -> Image.Image:
        ck = f"card:{key}:{size}:{border}"
        if ck not in self.cache:
            src = self.bob.get(key) or self.shot[key]
            self.cache[ck] = framed(src, size, border=border, focus_y=focus_y)
        return self.cache[ck]

    def fit(self, key: str, max_w: int, max_h: int, border=VIOLET) -> Image.Image:
        ck = f"fit:{key}:{max_w}:{max_h}"
        if ck not in self.cache:
            src = self.shot[key]
            sc = min(max_w / src.width, max_h / src.height)
            size = (int(src.width * sc), int(src.height * sc))
            self.cache[ck] = framed(src, size, radius=18, border=border, bw=3, focus_y=0.0)
        return self.cache[ck]

    def cached(self, key: str, make) -> Image.Image:
        if key not in self.cache:
            self.cache[key] = make()
        return self.cache[key]


def kenburns(frame: Image.Image, img: Image.Image, t: float, t0: float, t1: float, *, z0=1.0, z1=1.08,
             c0=(0.5, 0.5), c1=(0.5, 0.5)) -> None:
    """Полноэкранный скриншот с плавным zoom/pan."""
    k = ease((t - t0) / max(0.01, t1 - t0))
    z = z0 + (z1 - z0) * k
    cx = c0[0] + (c1[0] - c0[0]) * k
    cy = c0[1] + (c1[1] - c0[1]) * k
    cw, ch = img.width / z, img.width / z * H / W
    x0 = min(max(0, cx * img.width - cw / 2), img.width - cw)
    y0 = min(max(0, cy * img.height - ch / 2), img.height - ch)
    frame.paste(img.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch)), (0, 0))


def float_card(frame, img, cx, cy, t, t0, t1, z0=1.0, z1=1.04, alpha=1.0):
    k = ease((t - t0) / max(0.01, t1 - t0))
    s = z0 + (z1 - z0) * k
    w, h = img.width * s, img.height * s
    paste(frame, img, (cx - w / 2, cy - h / 2), alpha, s)


def eyebrow(frame, A, text, x, y, t, t0, color=GOLD2):
    pop(frame, A.cached(f"eb:{text}", lambda: text_img(text.upper(), 26, color, F_BOLD)), x, y, t, t0, dy=16)


def headline(frame, A, text, x, y, t, t0, size=72, color=WHITE, max_w=900):
    pop(frame, A.cached(f"hl:{text}:{size}:{color}", lambda: text_img(text, size, color, F_BLACK, max_w)), x, y, t, t0)


SAFE_BOTTOM = 880  # ниже — зона субтитров


def dim(frame: Image.Image, a: int = 90) -> None:
    frame.alpha_composite(Image.new("RGBA", (W, H), (*NAVY, a)))


def row(frame, imgs, y, t, starts, gap=18):
    """Плашки в ряд по центру кадра, каждая со своим временем появления."""
    total = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    x = (W - total) / 2
    for img, st in zip(imgs, starts):
        pop(frame, img, x, y, t, st)
        x += img.width + gap


# Каждая сцена: (начало, конец, функция(frame, A, t, t0, t1))


def sc_intro(f, A, t, t0, t1):
    float_card(f, A.card("02_slide1_bob_intro", (640, 860)), 1430, 560, t, t0, t1, 1.0, 1.05)
    eyebrow(f, A, "Знакомьтесь", 120, 300, t, t0 + 0.4)
    headline(f, A, "Это Боб.", 120, 345, t, t0 + 0.7, size=110)
    headline(f, A, "QA-инженер в FinTech-команде", 120, 500, t, t0 + 1.6, size=44, color=GOLD2, max_w=900)
    pop(f, A.cached("p:intro", lambda: plate("Ожидание:", "новое ТЗ пройдёт как по маслу", accent=PASS, tsize=34,
                                             tcolor=(110, 231, 183))), 105, 640, t, t0 + 4.0)


def sc_overwhelmed(f, A, t, t0, t1):
    float_card(f, A.card("04_slide3_bob_overwhelmed_tz", (600, 800)), 470, 560, t, t0, t1, 1.0, 1.05)
    eyebrow(f, A, "Кейс БЦК", 900, 190, t, t0 + 0.3)
    headline(f, A, "«Уведомление о расходах по карте»", 900, 235, t, t0 + 0.5, size=54, max_w=920)
    a = appear(t, t0 + 1.4, 0.4)
    n = int(round(57 * ease((t - t0 - 1.4) / 2.2)))
    num = text_img(str(n), 210, GOLD2, F_BLACK)
    paste(f, num, (900, 410), a)
    pop(f, A.cached("hl:req", lambda: text_img("требований", 56, WHITE, F_BLACK)), 1260, 560, t, t0 + 1.8)
    chips = ["таблицы", "критерии приёмки", "REST API"]
    x = 900
    for i, c in enumerate(chips):
        img = A.cached(f"chip:{c}", lambda c=c: pill_image(c, size=26, c1=(40, 50, 100), c2=(60, 50, 130)))
        pop(f, img, x, 665, t, t0 + 2.6 + i * 0.25, dy=14)
        x += img.width + 16
    pop(f, A.cached("p:real", lambda: plate("Реальность:", "спокойствие — ровно до первой строчки кода", accent=FATAL,
                                            tsize=30, ssize=24, tcolor=(252, 165, 165))), 885, 745, t, t0 + 7.2)


def sc_bugs(f, A, t, t0, t1):
    float_card(f, A.card("03_slide2_bob_found_bug", (470, 630), border=FATAL), 300, 560, t, t0, t1, 1.0, 1.04)
    eyebrow(f, A, "Приглядись повнимательнее", 600, 140, t, t0 + 0.2)
    headline(f, A, "У ТЗ раздвоение личности", 600, 180, t, t0 + 0.4, size=58, max_w=1250)
    shots = [  # (скриншот карточки, подпись, цвет, начало, конец) — по одному дефекту крупным планом
        ("ui_def-001", "DEF-001 · FATAL — GET для сохранения порога", FATAL, 1.0, 7.0),
        ("ui_def-004", "DEF-004 · MAJOR — нет метода чтения настроек", MAJOR, 7.0, 12.5),
        ("ui_def-006", "DEF-006 · MAJOR — amount: integer ↔ string", MAJOR, 12.5, 18.0),
    ]
    for key, title, col, a, b in shots:
        if not (t0 + a <= t < t0 + b + 0.6):
            continue
        k = appear(t, t0 + a) * (1 - appear(t, t0 + b - 0.4, 0.5))
        cap = A.cached(f"cap:{key}", lambda title=title, col=col: plate(title, accent=col, tsize=40, tcolor=WHITE))
        paste(f, cap, (560, 290 + 20 * (1 - k)), k)
        card = A.fit(key, 1300, 480, border=col)
        paste(f, card, (540, 410 + 30 * (1 - k)), k)
    if t >= t0 + 18.0:
        for i, (lbl, col) in enumerate([("DEF-001  GET вместо POST/PUT", FATAL), ("DEF-004  нет метода чтения", MAJOR),
                                        ("DEF-006  integer ↔ string", MAJOR)]):
            img = A.cached(f"sum:{lbl}", lambda lbl=lbl, col=col: plate(lbl, accent=col, tsize=36, tcolor=WHITE, width=760))
            pop(f, img, 600, 300 + i * 120, t, t0 + 18.0 + i * 0.3, dy=20)
        pop(f, A.cached("p:cost", lambda: plate("На тесте или в проде — в разы дороже",
                                                "порядок величин NIST / IBM: 5–15× к цене исправления на ревью",
                                                accent=GOLD, tsize=36)), 600, 680, t, t0 + 19.2)

def sc_chat(f, A, t, t0, t1):
    float_card(f, A.card("05_slide4_bob_choosing_agents", (560, 760)), 420, 580, t, t0, t1, 1.0, 1.05)
    eyebrow(f, A, "«Просто закинуть в чат»?", 840, 220, t, t0 + 0.3)
    headline(f, A, "Обычный чат-бот не спасает", 840, 262, t, t0 + 0.5, size=58, max_w=1000)
    items = ["выдуманная цитата", "потеря связи между разделами", "лишние данные в облаке"]
    for i, s in enumerate(items):
        img = A.cached(f"x:{s}", lambda s=s: plate("×  " + s, accent=FATAL, tsize=38, tcolor=WHITE, width=900))
        pop(f, img, 820, 420 + i * 150, t, t0 + 2.6 + i * 1.6, dy=26)


def sc_daiek(f, A, t, t0, t1):
    kenburns(f, A.shot["ui_home"], t, t0, t1, z0=1.16, z1=1.26, c0=(0.48, 0.57), c1=(0.44, 0.6))
    dim(f, 70)
    pop(f, A.cached("p:daiek", lambda: plate("Боб запускает DAIEK", "веб-интерфейс поверх того же CLI-ядра",
                                             accent=VIOLET, tsize=46)), 80, 640, t, t0 + 0.6)
    pop(f, A.cached("p:pii", lambda: plate("ПДн маскируются до LLM", "номера карт, телефоны, токены", accent=PASS, tsize=36,
                                           tcolor=(110, 231, 183))), 1150, 520, t, t0 + 4.0)
    pop(f, A.cached("p:5", lambda: plate("5 этапов проверки", "парсинг → REQ-ID → ISO 29148 → цитаты → Критик",
                                         accent=GOLD, tsize=36)), 1150, 690, t, t0 + 6.0)

def sc_verdict(f, A, t, t0, t1):
    float_card(f, A.fit("ui_verdict_card", 1720, 300, border=FATAL), W / 2, 280, t, t0, t1, 1.0, 1.02)
    float_card(f, A.fit("ui_kpis", 1720, 200), W / 2, 545, t, t0 + 0.4, t1, 1.0, 1.02, appear(t, t0 + 0.4))
    row(f, [A.cached("p:rej", lambda: plate("REJECTED · 68,5 / 100", "DEF-001 блокирует старт", accent=FATAL, tsize=38,
                                            tcolor=(252, 165, 165))),
            A.cached("p:20", lambda: plate("20 дефектов · 1 / 7 / 12", "FATAL / MAJOR / MINOR", accent=GOLD, tsize=38)),
            A.cached("p:28", lambda: plate("28 подсказок", "вне метрик", accent=BLUE, tsize=38, tcolor=(191, 219, 254)))],
        690, t, [t0 + 1.2, t0 + 3.5, t0 + 5.5])

def sc_funnel(f, A, t, t0, t1):
    float_card(f, A.fit("ui_funnel_pane", 1560, 560), W / 2, 380, t, t0, t1, 1.0, 1.03)
    row(f, [A.cached("p:22", lambda: plate("Цитаты RapidFuzz: 22 / 22", "0 галлюцинаций", accent=PASS, tsize=40,
                                           tcolor=(110, 231, 183))),
            A.cached("p:crit", lambda: plate("Критик: same_model 0", "подтвердил 5 · понизил 3 · отклонил 9", accent=VIOLET,
                                             tsize=40, tcolor=(196, 181, 253)))], 690, t, [t0 + 1.0, t0 + 3.0])

def sc_passport(f, A, t, t0, t1):
    float_card(f, A.fit("ui_passport_pane", 1560, 560), W / 2, 380, t, t0, t1, 1.0, 1.03)
    row(f, [A.cached("p:tests", lambda: plate("184 базовых теста · 205 с вебом", "ядро v1.0.2 проверено без сети",
                                              accent=GOLD, tsize=38)),
            A.cached("p:cannot", lambda: plate("Битый файл → CANNOT_PROCEED", "диагностика, без HTTP 500", accent=MAJOR,
                                               tsize=38, tcolor=(253, 211, 77)))], 690, t, [t0 + 1.0, t0 + 3.0])

def sc_roi(f, A, t, t0, t1):
    float_card(f, A.fit("ui_roi_pane", 1500, 520, border=GOLD), W / 2, 360, t, t0, t1, 1.0, 1.03)
    row(f, [A.cached("p:roi1", lambda: plate("236 000–314 000 KZT на 1 ТЗ", "23,6–31,4 ч · Decimal, без LLM", accent=GOLD,
                                             tsize=38)),
            A.cached("p:roi2", lambda: plate("7,1–9,4 млн KZT/мес", "708–942 ч на 30 ТЗ/мес", accent=GOLD, tsize=38)),
            A.cached("p:zero", lambda: plate("$0", "на инференс", accent=PASS, tsize=38, tcolor=(110, 231, 183)))],
        680, t, [t0 + 0.8, t0 + 2.4, t0 + 4.0])

def sc_happy(f, A, t, t0, t1):
    float_card(f, A.card("07_slide6_bob_happy_savings", (600, 800), border=PASS), 560, 560, t, t0, t1, 1.0, 1.05)
    eyebrow(f, A, "Итог", 1000, 330, t, t0 + 0.3)
    headline(f, A, "Боб не тушит пожары ночью", 1000, 372, t, t0 + 0.5, size=60, max_w=800)
    pop(f, A.cached("p:sprint", lambda: plate("ТЗ проверено до спринта", "с цитатами и готовыми правками", accent=PASS,
                                              tsize=38, tcolor=(110, 231, 183))), 985, 600, t, t0 + 1.6)


def sc_final(f, A, t, t0, t1):
    logo = A.cached("bigLogo", lambda: framed(Image.open(BRAND / "daiek_logo_main.jpg"), (300, 300), radius=48,
                                              border=GOLD, bw=6))
    float_card(f, logo, W / 2, 330, t, t0, t1, 0.92, 1.0, appear(t, t0, 0.8))
    pop(f, A.cached("fin1", lambda: text_img("DAIEK", 120, WHITE, F_BLACK)), (W - 470) / 2, 500, t, t0 + 0.5)
    sub = A.cached("fin2", lambda: text_img("Проверяем ТЗ до того, как оно станет болью", 44, GOLD2, F_BLACK))
    pop(f, sub, (W - sub.width) / 2, 660, t, t0 + 1.1)
    team = A.cached("fin3", lambda: pill_image("Команда «ВУПИПАЙ» · BCC-HUB × MNU FinTech Hackathon", size=34))
    pop(f, team, (W - team.width) / 2, 760, t, t0 + 1.6)


SCENES = [
    (0.0, 9.5, sc_intro), (9.5, 22.0, sc_overwhelmed),
    (22.0, 48.0, sc_bugs),
    (48.0, 60.0, sc_chat), (60.0, 72.0, sc_daiek),
    (72.0, 82.0, sc_verdict), (82.0, 92.0, sc_funnel), (92.0, 102.0, sc_passport),
    (102.0, 111.0, sc_roi), (111.0, 115.5, sc_happy), (115.5, 120.0, sc_final),
]
XFADE = 0.6


def draw_scene(A: Assets, idx: int, t: float) -> Image.Image:
    t0, t1, fn = SCENES[idx]
    frame = A.bg.copy()
    fn(frame, A, t, t0, t1)
    return frame


def subtitle_img(A: Assets, text: str) -> Image.Image:
    def make() -> Image.Image:
        f = font(F_BOLD, 38)
        lines = wrap(text, f, 1500)
        lh = 50
        w = int(max(f.getlength(ln) for ln in lines)) + 64
        h = lh * len(lines) + 30
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, w - 1, h - 1], 16, fill=(7, 13, 34, 205))
        for i, ln in enumerate(lines):
            lw = f.getlength(ln)
            d.text(((w - lw) / 2, 14 + i * lh), ln, font=f, fill=WHITE)
        return img
    return A.cached(f"sub:{text}", make)


def render_frame(A: Assets, t: float, cues: list[Cue]) -> Image.Image:
    idx = next(i for i, (a, b, _) in enumerate(SCENES) if a <= t < b or i == len(SCENES) - 1)
    frame = draw_scene(A, idx, t)
    t0 = SCENES[idx][0]
    if idx > 0 and t < t0 + XFADE:  # мягкий crossfade с предыдущей сценой
        prev = draw_scene(A, idx - 1, t)
        frame = Image.blend(prev, frame, ease((t - t0) / XFADE))
    # постоянные элементы: мягкое затемнение под шапкой, знак DAIEK и бейдж команды
    frame.alpha_composite(A.top_band)
    paste(frame, A.mark, (48, 34), 0.95)
    paste(frame, A.badge, (W - A.badge.width - 48, 40), 1.0)
    cue = next((c for c in cues if c.start <= t < c.end), None)
    if cue:
        img = subtitle_img(A, cue.text)
        a = min(appear(t, cue.start, 0.18), 1 - appear(t, cue.end - 0.18, 0.18))
        paste(frame, img, ((W - img.width) / 2, H - img.height - 40), max(a, 0.0))
    if t < 0.8:  # мягкое появление и затухание кадра
        frame = Image.blend(Image.new("RGBA", (W, H), (*NAVY, 255)), frame, ease(t / 0.8))
    if t > DUR - 1.2:
        frame = Image.blend(frame, Image.new("RGBA", (W, H), (*NAVY, 255)), ease((t - DUR + 1.2) / 1.2))
    return frame


def render_video(audio: Path, cues: list[Cue], preview: list[float]) -> None:
    A = Assets()
    WORK.mkdir(parents=True, exist_ok=True)
    for t in preview:  # контрольные кадры для проверки глазами
        render_frame(A, t, cues).convert("RGB").save(WORK / f"preview_{t:06.2f}.jpg", quality=88)
    if "--preview-only" in sys.argv:
        return
    cmd = [FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium",
           "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-t", str(DUR),
           "-movflags", "+faststart", str(OUT_MP4)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = int(DUR * FPS)
    for i in range(total):
        proc.stdin.write(render_frame(A, i / FPS, cues).convert("RGB").tobytes())
        if i % 300 == 0:
            print(f"кадр {i}/{total}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg завершился с ошибкой")


def main() -> None:
    voices = build_voice(skip="--skip-tts" in sys.argv)
    audio = build_audio(music="--no-music" not in sys.argv)
    cues = build_cues()
    write_texts(cues, voices)
    preview = [2.5, 8.0, 15.0, 21.0, 26.0, 37.0, 45.0, 53.0, 66.0, 76.0, 87.0, 97.0, 106.0, 113.0, 118.0]
    render_video(audio, cues, preview)
    meta = {"duration": DUR, "fps": FPS, "size": [W, H],
            "speech": {f"seg{i}": round(s.speech, 2) for i, s in enumerate(SEGMENTS, 1)}}
    (WORK / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT_MP4)


if __name__ == "__main__":
    main()
