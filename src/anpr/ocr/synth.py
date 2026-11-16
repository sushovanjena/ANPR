"""Synthetic Indian number plates with realistic degradations for OCR training."""

from __future__ import annotations

import random
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"
STATE_CODES = (
    "MH KA KL TN DL UP GJ RJ MP HR PB AP TS WB BR OD AS JH CG UK HP JK CH GA PY "
    "AN AR DD DN LA LD ML MN MZ NL SK TR TG"
).split()
HEAVY_STATES = ["MH", "KA", "KL", "TN", "DL", "UP", "GJ", "RJ", "HR", "AP", "TS", "WB"]
_PARTS = re.compile(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{1,4})$")

# (background, text colour, probability): private white, commercial yellow,
# rental black, EV green
STYLES = [
    ((250, 250, 250), (15, 15, 15), 0.58),
    ((245, 200, 20), (15, 15, 15), 0.26),
    ((20, 20, 20), (240, 240, 240), 0.06),
    ((20, 20, 20), (245, 205, 30), 0.04),
    ((40, 140, 70), (250, 250, 250), 0.03),
    ((235, 235, 225), (30, 30, 60), 0.03),
]


def _font_paths() -> list[str]:
    root = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    names = ["DejaVuSans-Bold.ttf", "DejaVuSansMono-Bold.ttf", "DejaVuSerif-Bold.ttf"]
    paths = [str(root / n) for n in names if (root / n).exists()]
    for p in Path("/usr/share/fonts").rglob("*.ttf"):
        if any(
            k in p.name
            for k in (
                "LiberationSansNarrow-Bold",
                "LiberationSans-Bold",
                "FreeSansBold",
            )
        ):
            paths.append(str(p))
    return paths


FONTS = _font_paths()


def random_plate_text(rng: random.Random) -> str:
    """Draw a plate string following Indian registration patterns."""
    if rng.random() < 0.03:
        year, num = rng.randint(21, 26), rng.randint(0, 9999)
        return f"{year}BH{num:04d}{rng.choice(LETTERS)}{rng.choice(LETTERS)}"
    state = rng.choice(HEAVY_STATES) if rng.random() < 0.6 else rng.choice(STATE_CODES)
    district = (
        f"{rng.randint(1, 99):02d}" if rng.random() < 0.93 else str(rng.randint(1, 9))
    )
    n_series = rng.choices([1, 2, 3], [0.15, 0.7, 0.15])[0]
    series = "".join(rng.choice(LETTERS) for _ in range(n_series))
    number = (
        f"{rng.randint(1, 9999):04d}"
        if rng.random() < 0.9
        else str(rng.randint(1, 9999))
    )
    return state + district + series + number


def random_generic_text(rng: random.Random) -> str:
    """Draw a non-Indian-looking plate string so the model keeps no format prior."""
    kind = rng.random()
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits = "0123456789"
    if kind < 0.25:
        return "".join(rng.choice(digits) for _ in range(rng.randint(5, 8)))
    if kind < 0.5:
        a = "".join(rng.choice(letters) for _ in range(3))
        return a + "".join(rng.choice(digits) for _ in range(rng.randint(3, 4)))
    if kind < 0.75:
        return (
            rng.choice(digits)
            + "".join(rng.choice(letters) for _ in range(3))
            + "".join(rng.choice(digits) for _ in range(3))
        )
    pool = letters + digits
    return "".join(rng.choice(pool) for _ in range(rng.randint(5, 9)))


def _boundaries(text: str) -> set[int]:
    m = _PARTS.match(text)
    if not m:
        return set()
    a, b, c = (len(g) for g in m.groups()[:3])
    return {a, a + b, a + b + c}


def _draw_line(draw, text, box, fg, rng, gaps):
    x0, y0, x1, y1 = box
    path = rng.choice(FONTS) if FONTS else None
    size = int((y1 - y0) * rng.uniform(0.72, 0.9))
    spacing = rng.uniform(0.02, 0.12)
    while True:
        font = ImageFont.truetype(path, size) if path else ImageFont.load_default()
        widths = [font.getlength(ch) for ch in text]
        extra = [
            size
            * spacing
            * (rng.uniform(1.0, 2.2) if i in gaps and rng.random() < 0.6 else 1.0)
            for i in range(len(text))
        ]
        total = sum(widths) + sum(extra[:-1])
        if total <= (x1 - x0) * 0.98 or size < 12:
            break
        size = int(size * 0.92)
    x = x0 + ((x1 - x0) - total) / 2
    stroke = rng.choice([0, 0, 1])
    for ch, w, e in zip(text, widths, extra):
        draw.text(
            (x, y0 + (y1 - y0 - size) / 2 - size * 0.1),
            ch,
            fill=fg,
            font=font,
            stroke_width=stroke,
            stroke_fill=fg,
        )
        x += w + e


def render_plate(text: str, rng: random.Random) -> np.ndarray:
    """Render a clean plate image (BGR) for the given text."""
    bg, fg, _ = rng.choices(STYLES, [s[2] for s in STYLES])[0]
    two_line = rng.random() < 0.3
    h, w = (190, 320) if two_line else (100, 430)
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    m = rng.randint(3, 8)
    d.rounded_rectangle(
        [m, m, w - m, h - m],
        radius=rng.randint(4, 12),
        outline=fg,
        width=rng.randint(2, 4),
    )
    left = m + 10
    if not two_line and rng.random() < 0.35:
        d.rectangle([m + 3, m + 3, m + 40, h - m - 3], fill=(20, 60, 160))
        d.text(
            (m + 6, h // 2 - 8),
            "IND",
            fill=(255, 255, 255),
            font=ImageFont.load_default(),
        )
        left = m + 48
    if two_line:
        top, bottom = text[:4], text[4:]
        _draw_line(d, top, (left, m + 8, w - m - 10, h * 0.48), fg, rng, set())
        _draw_line(d, bottom, (left, h * 0.5, w - m - 10, h - m - 8), fg, rng, set())
    else:
        _draw_line(
            d, text, (left, m + 10, w - m - 10, h - m - 10), fg, rng, _boundaries(text)
        )
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _occlude(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    kind = rng.choice(["lemons", "garland", "dirt", "sticker", "scratch"])
    out = img.copy()
    if kind == "lemons":
        y = rng.randint(int(h * 0.2), int(h * 0.8))
        pts = []
        for k in range(rng.randint(3, 7)):
            x = int(w * (k + rng.uniform(0.2, 0.8)) / 7)
            yy = y + int(8 * np.sin(k * 1.3))
            pts.append((x, yy))
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            cv2.line(out, (x1, y1), (x2, y2), (30, 90, 40), 2)
        for x, yy in pts:
            col = rng.choice([(40, 210, 235), (40, 190, 225), (60, 170, 70)])
            cv2.ellipse(
                out,
                (x, yy),
                (rng.randint(9, 16), rng.randint(11, 19)),
                rng.randint(0, 90),
                0,
                360,
                col,
                -1,
            )
    elif kind == "garland":
        for _ in range(rng.randint(5, 12)):
            col = rng.choice(
                [
                    (0, 140, 255),
                    (0, 200, 255),
                    (40, 40, 220),
                    (60, 60, 200),
                    (0, 170, 240),
                ]
            )
            cx = rng.randint(0, w)
            cy = rng.choice(
                [rng.randint(0, int(h * 0.3)), rng.randint(int(h * 0.7), h)]
            )
            cv2.circle(out, (cx, cy), rng.randint(8, 18), col, -1)
    elif kind == "dirt":
        layer = out.copy()
        for _ in range(rng.randint(2, 6)):
            cv2.ellipse(
                layer,
                (rng.randint(0, w), rng.randint(0, h)),
                (rng.randint(8, 40), rng.randint(5, 25)),
                rng.randint(0, 180),
                0,
                360,
                (30, 40, 60),
                -1,
            )
        out = cv2.addWeighted(
            layer, rng.uniform(0.25, 0.6), out, 1 - rng.uniform(0.25, 0.6), 0
        )
    elif kind == "sticker":
        x, y = rng.randint(0, w - 40), rng.randint(0, h - 30)
        cv2.rectangle(
            out,
            (x, y),
            (x + rng.randint(20, 50), y + rng.randint(10, 30)),
            tuple(
                int(c)
                for c in np.random.default_rng(rng.randint(0, 9999)).integers(0, 255, 3)
            ),
            -1,
        )
    else:
        for _ in range(rng.randint(2, 6)):
            cv2.line(
                out,
                (rng.randint(0, w), rng.randint(0, h)),
                (rng.randint(0, w), rng.randint(0, h)),
                (120, 120, 120),
                1,
            )
    return out


def degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply occlusion, perspective, blur, resolution loss, noise and JPEG artefacts."""
    h, w = img.shape[:2]
    if rng.random() < 0.3:
        img = _occlude(img, rng)
    pad = rng.uniform(0.0, 0.12)
    px, py = int(w * pad), int(h * pad * 1.5)
    canvas_col = rng.choice(
        [(60, 60, 60), (200, 200, 200), (30, 30, 90), (120, 140, 150), (20, 20, 20)]
    )
    img = cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_CONSTANT, value=canvas_col)
    h, w = img.shape[:2]
    j = rng.uniform(0.0, 0.1)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32(
        [[rng.uniform(-j, j) * w, rng.uniform(-j, j) * h] for _ in range(4)]
    )
    mat = cv2.getPerspectiveTransform(src, dst)
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-6, 6), 1.0)
    img = cv2.warpPerspective(img, mat, (w, h), borderValue=canvas_col)
    img = cv2.warpAffine(img, rot, (w, h), borderValue=canvas_col)
    if rng.random() < 0.5:
        s = rng.uniform(0.3, 2.2)
        img = cv2.GaussianBlur(img, (0, 0), s)
    elif rng.random() < 0.5:
        k = rng.randint(3, 11)
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        ang = rng.uniform(0, 180)
        kernel = cv2.warpAffine(
            kernel,
            cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1.0),
            (k, k),
        )
        img = cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))
    tw = rng.randint(50, 300)
    th = max(12, int(img.shape[0] * tw / img.shape[1]))
    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        f = rng.uniform(1.0, 2.5)
        img = cv2.resize(
            img,
            None,
            fx=f,
            fy=f,
            interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]),
        )
    img = np.clip(
        img.astype(np.float32) * rng.uniform(0.55, 1.25) + rng.uniform(-35, 35), 0, 255
    )
    if rng.random() < 0.5:
        img = 255 * (img / 255) ** rng.uniform(0.6, 1.6)
    if rng.random() < 0.6:
        img = img + np.random.default_rng(rng.randint(0, 99999)).normal(
            0, rng.uniform(1, 12), img.shape
        )
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        ok, enc = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 90)]
        )
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img
    return img


def generate_dataset(
    n: int, out_dir: Path, seed: int = 0, generic_frac: float = 0.0
) -> Path:
    """Write n synthetic plates and a tab-separated label file; return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    lines = []
    for i in range(n):
        text = (
            random_generic_text(rng)
            if rng.random() < generic_frac
            else random_plate_text(rng)
        )
        img = degrade(render_plate(text, rng), rng)
        path = out_dir / ("s" + str(i).zfill(6) + ".png")
        cv2.imwrite(str(path), img)
        lines.append(str(path) + chr(9) + text)
    gt = out_dir / "gt.txt"
    gt.write_text(chr(10).join(lines) + chr(10))
    return gt
