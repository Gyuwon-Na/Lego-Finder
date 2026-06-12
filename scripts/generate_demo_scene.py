#!/usr/bin/env python3
"""Generate a deterministic LEGO-like pile image for local smoke testing."""

from __future__ import annotations

import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "demo"


COLORS = {
    "red": (229, 57, 53),
    "blue": (30, 136, 229),
    "yellow": (253, 216, 53),
    "green": (67, 160, 71),
    "orange": (251, 140, 0),
    "purple": (142, 36, 170),
    "black": (24, 24, 24),
}


def main() -> None:
    random.seed(7)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (960, 640), (34, 38, 48))
    draw = ImageDraw.Draw(image)
    for _ in range(500):
        x = random.randint(0, 960)
        y = random.randint(0, 640)
        r = random.randint(2, 16)
        shade = random.randint(38, 90)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(shade, shade + random.randint(-8, 12), shade + random.randint(-8, 16)))

    labels = []
    labels.append(draw_brick(draw, 110, 120, 240, 118, "red", "target red 2x4 brick"))
    labels.append(draw_brick(draw, 470, 105, 245, 118, "yellow", "yellow 2x4 brick with red patch"))
    draw.rounded_rectangle((470, 105, 548, 223), radius=18, fill=COLORS["red"])
    labels.append({"x": 470, "y": 105, "w": 78, "h": 118, "color": "red", "name": "red patch on yellow brick"})

    for spec in [
        (190, 345, 185, 88, "blue", "blue 2x3 brick"),
        (610, 330, 122, 122, "green", "green 2x2 brick"),
        (705, 155, 170, 82, "orange", "orange 1x4 plate"),
        (80, 470, 132, 64, "purple", "purple 1x3 plate"),
        (390, 420, 160, 76, "black", "black 2x4 tile"),
    ]:
        labels.append(draw_brick(draw, *spec))

    image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=3))
    image_path = OUT_DIR / "lego_pile.png"
    image.save(image_path)
    (OUT_DIR / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {image_path}")


def draw_brick(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, color_name: str, name: str) -> dict:
    color = COLORS[color_name]
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=color)
    stud_count = max(2, round(max(w, h) / 58))
    for i in range(stud_count):
        cx = x + int((i + 0.5) * w / stud_count)
        cy = y + int(h * 0.35)
        draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=tuple(min(255, c + 28) for c in color))
        draw.ellipse((cx - 11, cy - 11, cx + 11, cy + 11), outline=tuple(max(0, c - 35) for c in color), width=2)
    return {"x": x, "y": y, "w": w, "h": h, "color": color_name, "name": name}


if __name__ == "__main__":
    main()
