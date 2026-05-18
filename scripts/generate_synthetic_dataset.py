#!/usr/bin/env python3
"""Generate synthetic LEGO pile data for image-search evaluation and YOLO.

The generated dataset is intentionally self-contained:

- images/train/*.png: synthetic LEGO pile scenes
- labels/train/*.txt: YOLO segmentation labels, class 0 = lego_piece
- masks/train/*.png: target-piece binary masks for quick visual checks
- annotations.json: all generated instance metadata
- eval_queries.json: one prompt + target bbox/mask per image for accuracy checks
- dataset.yaml: Ultralytics YOLO segmentation config

This is a starter dataset. It is useful for bootstrapping the proposal model and
for testing the evaluation loop, but real phone photos are still needed for final
accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install Pillow first: pip install pillow") from exc


ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "red": (229, 57, 53),
    "blue": (30, 136, 229),
    "yellow": (253, 216, 53),
    "green": (67, 160, 71),
    "orange": (251, 140, 0),
    "purple": (142, 36, 170),
    "black": (24, 24, 24),
    "white": (238, 238, 232),
    "gray": (138, 143, 152),
}

PART_SPECS = [
    {"category": "brick", "width": 2, "length": 4, "weight": 9},
    {"category": "brick", "width": 2, "length": 3, "weight": 7},
    {"category": "brick", "width": 2, "length": 2, "weight": 6},
    {"category": "brick", "width": 1, "length": 4, "weight": 5},
    {"category": "brick", "width": 1, "length": 2, "weight": 5},
    {"category": "plate", "width": 2, "length": 4, "weight": 4},
    {"category": "plate", "width": 1, "length": 4, "weight": 4},
    {"category": "plate", "width": 2, "length": 2, "weight": 4},
    {"category": "tile", "width": 2, "length": 2, "weight": 3},
    {"category": "tile", "width": 1, "length": 2, "weight": 3},
]

CATEGORY_LABELS = {"brick": "brick", "plate": "plate", "tile": "tile"}


def weighted_part() -> Dict[str, int]:
    total = sum(int(spec["weight"]) for spec in PART_SPECS)
    roll = random.randint(1, total)
    cursor = 0
    for spec in PART_SPECS:
        cursor += int(spec["weight"])
        if roll <= cursor:
            return dict(spec)
    return dict(PART_SPECS[0])


def make_background(size: Tuple[int, int]) -> Image.Image:
    width, height = size
    base = Image.new("RGB", size, tuple(random.randint(22, 62) for _ in range(3)))
    draw = ImageDraw.Draw(base)

    for _ in range(random.randint(220, 420)):
        x = random.randint(-20, width + 20)
        y = random.randint(-20, height + 20)
        radius = random.randint(3, 26)
        shade = random.randint(35, 145)
        tint = (
            clamp_int(shade + random.randint(-18, 24)),
            clamp_int(shade + random.randint(-14, 30)),
            clamp_int(shade + random.randint(-20, 26)),
        )
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=tint)

    # Add faint table/lighting bands so the model sees non-uniform backgrounds.
    for _ in range(random.randint(3, 7)):
        x0 = random.randint(-width // 3, width)
        color = tuple(clamp_int(c + random.randint(-12, 32)) for c in base.getpixel((width // 2, height // 2)))
        draw.rectangle((x0, 0, x0 + random.randint(45, 160), height), fill=color)

    blur = random.uniform(0.3, 1.4)
    base = base.filter(ImageFilter.GaussianBlur(radius=blur))
    enhancer = ImageEnhance.Color(base)
    base = enhancer.enhance(random.uniform(0.75, 1.3))
    return base


def render_piece(spec: Dict[str, int], color_key: str, stud_px: int) -> Tuple[Image.Image, Image.Image]:
    width_studs = int(spec["width"])
    length_studs = int(spec["length"])
    category = str(spec["category"])
    px_w = max(36, length_studs * stud_px)
    px_h = max(32, width_studs * stud_px)
    pad = max(18, stud_px // 2)
    layer = Image.new("RGBA", (px_w + pad * 2, px_h + pad * 2), (0, 0, 0, 0))
    mask = Image.new("L", layer.size, 0)
    draw = ImageDraw.Draw(layer)
    mask_draw = ImageDraw.Draw(mask)
    color = COLORS[color_key]
    x0, y0 = pad, pad
    x1, y1 = pad + px_w, pad + px_h
    radius = max(8, stud_px // 3)

    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle((x0 + 5, y0 + 7, x1 + 5, y1 + 7), radius=radius, fill=(0, 0, 0, 70))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=5))
    layer.alpha_composite(shadow)

    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=color + (255,))
    draw.rounded_rectangle((x0, y0, x1, y0 + max(5, px_h // 5)), radius=radius, fill=lighten(color, 26) + (115,))
    draw.line((x0 + 3, y1 - 3, x1 - 3, y1 - 3), fill=darken(color, 34) + (145,), width=3)
    mask_draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=255)

    if category != "tile":
        stud_radius = max(7, stud_px // 5)
        stud_fill = lighten(color, 34)
        stud_outline = darken(color, 42)
        for row in range(width_studs):
            for col in range(length_studs):
                cx = x0 + int((col + 0.5) * px_w / length_studs)
                cy = y0 + int((row + 0.5) * px_h / width_studs)
                if category == "plate":
                    cy -= max(1, stud_px // 12)
                draw.ellipse(
                    (cx - stud_radius, cy - stud_radius, cx + stud_radius, cy + stud_radius),
                    fill=stud_fill + (255,),
                )
                draw.ellipse(
                    (cx - stud_radius + 4, cy - stud_radius + 4, cx + stud_radius - 4, cy + stud_radius - 4),
                    outline=stud_outline + (130,),
                    width=2,
                )

    if category == "plate":
        draw.rectangle((x0, y1 - max(6, px_h // 8), x1, y1), fill=darken(color, 28) + (115,))

    layer = apply_piece_augmentation(layer)
    return layer, mask


def apply_piece_augmentation(layer: Image.Image) -> Image.Image:
    if random.random() < 0.55:
        layer = ImageEnhance.Brightness(layer).enhance(random.uniform(0.82, 1.16))
    if random.random() < 0.55:
        layer = ImageEnhance.Color(layer).enhance(random.uniform(0.82, 1.18))
    if random.random() < 0.25:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    return layer


def place_piece(
    image: Image.Image,
    instance_mask: Image.Image,
    spec: Dict[str, int],
    color_key: str,
    instance_id: int,
    target: bool,
) -> Dict[str, object]:
    width, height = image.size
    stud_px = random.randint(34, 52)
    piece, raw_mask = render_piece(spec, color_key, stud_px)
    angle = random.uniform(-42, 42)
    piece = piece.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    raw_mask = raw_mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)

    piece_w, piece_h = piece.size
    if piece_w >= width - 20 or piece_h >= height - 20:
        scale = min((width - 30) / piece_w, (height - 30) / piece_h, 0.9)
        piece = piece.resize((int(piece_w * scale), int(piece_h * scale)), Image.Resampling.BICUBIC)
        raw_mask = raw_mask.resize(piece.size, Image.Resampling.NEAREST)
        piece_w, piece_h = piece.size

    x = random.randint(-piece_w // 5, width - piece_w + piece_w // 5)
    y = random.randint(-piece_h // 5, height - piece_h + piece_h // 5)
    image.alpha_composite(piece, (x, y))

    mask_value = 255 if target else max(25, min(230, instance_id + 1))
    colored_mask = Image.new("L", raw_mask.size, 0)
    colored_mask.paste(mask_value, mask=raw_mask)
    instance_mask.paste(colored_mask, (x, y), raw_mask)

    bbox = mask_bbox(raw_mask, x, y, width, height)
    polygon = bbox_polygon(bbox)
    annotation = {
        "id": instance_id,
        "color": color_key,
        "category": spec["category"],
        "width": spec["width"],
        "length": spec["length"],
        "prompt": prompt_for(color_key, spec),
        "bbox": {"x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]},
        "polygon": polygon,
        "target": target,
    }
    return annotation


def mask_bbox(mask: Image.Image, offset_x: int, offset_y: int, image_w: int, image_h: int) -> Tuple[int, int, int, int]:
    bbox = mask.getbbox()
    if bbox is None:
        return (0, 0, 0, 0)
    x0 = max(0, offset_x + bbox[0])
    y0 = max(0, offset_y + bbox[1])
    x1 = min(image_w, offset_x + bbox[2])
    y1 = min(image_h, offset_y + bbox[3])
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def bbox_polygon(bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    x, y, w, h = bbox
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def yolo_seg_line(annotation: Dict[str, object], image_size: Tuple[int, int]) -> str:
    image_w, image_h = image_size
    points = annotation["polygon"]
    values = ["0"]
    for x, y in points:
        values.append(format_float(float(x) / image_w))
        values.append(format_float(float(y) / image_h))
    return " ".join(values)


def format_float(value: float) -> str:
    return f"{max(0.0, min(1.0, value)):.6f}".rstrip("0").rstrip(".")


def prompt_for(color_key: str, spec: Dict[str, int]) -> str:
    return "{} {}x{} {}".format(color_key, spec["width"], spec["length"], CATEGORY_LABELS[str(spec["category"])])


def split_name(index: int, train_ratio: float) -> str:
    return "train" if random.random() < train_ratio or index == 0 else "val"


def generate_dataset(out: Path, count: int, size: Tuple[int, int], seed: int, train_ratio: float) -> None:
    random.seed(seed)
    for folder in ["images/train", "images/val", "labels/train", "labels/val", "masks/train", "masks/val"]:
        (out / folder).mkdir(parents=True, exist_ok=True)

    annotations = []
    eval_queries = []
    for index in range(count):
        split = split_name(index, train_ratio)
        file_stem = f"{index:05d}"
        image = make_background(size).convert("RGBA")
        instance_mask = Image.new("L", size, 0)
        instance_count = random.randint(7, 14)
        scene_annotations = []

        for instance_id in range(instance_count - 1):
            spec = weighted_part()
            color_key = random.choice(list(COLORS.keys()))
            annotation = place_piece(image, instance_mask, spec, color_key, instance_id, False)
            if annotation["bbox"]["width"] >= 8 and annotation["bbox"]["height"] >= 8:
                scene_annotations.append(annotation)

        target_spec = weighted_part()
        target_color = random.choice(list(COLORS.keys()))
        if random.random() < 0.62:
            target_spec = {"category": "brick", "width": 2, "length": 4, "weight": 1}
            target_color = "red"
        target_annotation = place_piece(image, instance_mask, target_spec, target_color, instance_count - 1, True)
        if target_annotation["bbox"]["width"] >= 8 and target_annotation["bbox"]["height"] >= 8:
            scene_annotations.append(target_annotation)

        image = apply_scene_augmentation(image.convert("RGB"))
        image_file = f"{file_stem}.png"
        image.save(out / "images" / split / image_file)

        yolo_lines = [yolo_seg_line(annotation, size) for annotation in scene_annotations]
        (out / "labels" / split / f"{file_stem}.txt").write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        target_mask = Image.new("L", size, 0)
        target_annotation = next((item for item in scene_annotations if item.get("target")), scene_annotations[-1])
        target_box = target_annotation["bbox"]
        target_mask_draw = ImageDraw.Draw(target_mask)
        target_mask_draw.rectangle(
            (
                target_box["x"],
                target_box["y"],
                target_box["x"] + target_box["width"],
                target_box["y"] + target_box["height"],
            ),
            fill=255,
        )
        target_mask.save(out / "masks" / split / image_file)

        scene_record = {
            "file": f"images/{split}/{image_file}",
            "label_file": f"labels/{split}/{file_stem}.txt",
            "mask_file": f"masks/{split}/{image_file}",
            "split": split,
            "width": size[0],
            "height": size[1],
            "instances": scene_annotations,
        }
        annotations.append(scene_record)
        eval_queries.append(
            {
                "file": scene_record["file"],
                "split": split,
                "prompt": target_annotation["prompt"],
                "target": target_annotation,
                "instances": scene_annotations,
            }
        )

    (out / "annotations.json").write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "eval_queries.json").write_text(json.dumps(eval_queries, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "dataset.yaml").write_text(dataset_yaml(out), encoding="utf-8")


def apply_scene_augmentation(image: Image.Image) -> Image.Image:
    if random.random() < 0.75:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.78, 1.22))
    if random.random() < 0.65:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.82, 1.28))
    if random.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.9)))
    return image


def dataset_yaml(out: Path) -> str:
    rel = out.resolve()
    return "\n".join(
        [
            f"path: {rel}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: lego_piece",
            "",
        ]
    )


def lighten(color: Tuple[int, int, int], amount: int) -> Tuple[int, int, int]:
    return tuple(clamp_int(value + amount) for value in color)


def darken(color: Tuple[int, int, int], amount: int) -> Tuple[int, int, int]:
    return tuple(clamp_int(value - amount) for value in color)


def clamp_int(value: int) -> int:
    return max(0, min(255, int(value)))


def parse_size(value: str) -> Tuple[int, int]:
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT")
    left, right = value.lower().split("x", 1)
    return int(left), int(right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "data" / "synthetic_lego"), help="output directory")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--size", type=parse_size, default=(960, 640), help="image size, e.g. 960x640")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--train-ratio", type=float, default=0.82)
    args = parser.parse_args()

    out = Path(args.out)
    generate_dataset(out, args.count, args.size, args.seed, args.train_ratio)
    print(f"wrote {args.count} synthetic LEGO scenes to {out}")
    print(f"YOLO config: {out / 'dataset.yaml'}")
    print(f"Evaluation prompts: {out / 'eval_queries.json'}")


if __name__ == "__main__":
    main()
