#!/usr/bin/env python3
"""Generate a tiny synthetic segmentation dataset for ROI model experiments.

This mirrors the PDF's Copy-Paste synthetic data idea. It creates colored LEGO-like
rectangles over noisy backgrounds and binary masks. Use it only as a starter; real
training needs photographed/3D-rendered parts and diverse backgrounds.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install Pillow first: pip install pillow") from exc

COLORS = {
    "red": (229, 57, 53),
    "blue": (30, 136, 229),
    "yellow": (253, 216, 53),
    "green": (67, 160, 71),
    "black": (24, 24, 24),
}


def make_background(size: tuple[int, int]) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, tuple(random.randint(30, 80) for _ in range(3)))
    draw = ImageDraw.Draw(img)
    for _ in range(260):
        x = random.randint(0, w)
        y = random.randint(0, h)
        radius = random.randint(3, 18)
        color = tuple(random.randint(40, 160) for _ in range(3))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return img.filter(ImageFilter.GaussianBlur(radius=0.6))


def draw_brick(img: Image.Image, mask: Image.Image, color_name: str) -> dict[str, int | str]:
    draw = ImageDraw.Draw(img)
    mask_draw = ImageDraw.Draw(mask)
    w, h = img.size
    bw = random.randint(55, 120)
    bh = random.randint(32, 75)
    x = random.randint(10, w - bw - 10)
    y = random.randint(10, h - bh - 10)
    color = COLORS[color_name]
    draw.rounded_rectangle((x, y, x + bw, y + bh), radius=12, fill=color)
    for sx in range(x + 16, x + bw - 8, 28):
        draw.ellipse((sx - 8, y - 10, sx + 8, y + 6), fill=tuple(min(255, c + 25) for c in color))
    mask_draw.rectangle((x, y, x + bw, y + bh), fill=255)
    return {"x": x, "y": y, "w": bw, "h": bh, "color": color_name}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic", help="output directory")
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    labels = []
    for i in range(args.count):
        color = random.choice(list(COLORS))
        img = make_background((640, 480))
        mask = Image.new("L", img.size, 0)
        label = draw_brick(img, mask, color)
        labels.append({"file": f"{i:04d}.png", **label, "prompt": f"{color} 2x3 brick"})
        img.save(out / "images" / f"{i:04d}.png")
        mask.save(out / "masks" / f"{i:04d}.png")
    (out / "labels.json").write_text(__import__("json").dumps(labels, indent=2), encoding="utf-8")
    print(f"wrote {args.count} synthetic samples to {out}")


if __name__ == "__main__":
    main()
