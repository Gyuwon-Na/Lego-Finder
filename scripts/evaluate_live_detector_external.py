#!/usr/bin/env python3
"""Download small external image samples and tune the live camera detector.

The browser AR prototype intentionally uses a lightweight detector, so this
script mirrors that heuristic instead of the heavier backend image-search path.
It builds a small positive/negative set from Wikimedia Commons search results:

- positive: red LEGO brick images
- negative: red non-LEGO objects likely to trigger color-only false positives

The goal is not to train a model. It is a fast regression/tuning loop for the
frontend thresholds so red cups, boxes, books, and similar objects are less
likely to appear as AR hits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install Pillow first: .venv/bin/python -m pip install Pillow") from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "external_eval" / "live_detector"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "BrickFinderPrototype/0.1 external detector evaluation"

POSITIVE_QUERIES = [
    "red LEGO brick",
    "red Lego bricks",
    "LEGO bricks red",
]

NEGATIVE_QUERIES = [
    "red cup",
    "red box",
    "red book",
    "red stapler",
    "red toy car",
    "red flower",
]


@dataclass(frozen=True)
class Params:
    score_min: float = 0.52
    fill_min: float = 0.32
    shape_min: float = 0.55
    texture_min: float = 0.16
    separation_min: float = 0.24
    area_min: float = 0.012
    area_max: float = 0.22
    short_side_min: int = 8
    long_side_min: int = 16


@dataclass
class Detection:
    score: float
    fill: float
    shape_score: float
    texture_score: float
    background_separation_score: float
    area_ratio: float
    bbox: dict[str, float]
    source: str = "color"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--per-query", type=int, default=6)
    parser.add_argument("--synthetic-positives", type=int, default=12)
    parser.add_argument("--synthetic-negatives", type=int, default=24)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.dataset)
    manifest_path = root / "manifest.json"
    if args.refresh or not manifest_path.exists():
        manifest = build_dataset(root, args.per_query, args.synthetic_positives, args.synthetic_negatives)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = ensure_synthetic_samples(root, manifest, args.synthetic_positives, args.synthetic_negatives)

    if args.tune:
        report = tune(manifest, root)
    else:
        report = evaluate(manifest, root, Params())

    output_path = Path(args.out) if args.out else root / "live_detector_report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(report, output_path.with_suffix(".csv"))

    summary = {key: value for key, value in report.items() if key != "rows"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")


def build_dataset(root: Path, per_query: int, synthetic_positives: int, synthetic_negatives: int) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []
    for label, queries in (("positive", POSITIVE_QUERIES), ("negative", NEGATIVE_QUERIES)):
        for query in queries:
            for item in commons_search_images(query, per_query):
                saved = download_item(root / label, item)
                if saved:
                    samples.append(
                        {
                            "label": label,
                            "query": query,
                            "file": str(saved.relative_to(root)),
                            "source": item,
                        }
                    )
                time.sleep(1.0)

    for path in generate_synthetic_lego_positives(root / "positive_synthetic", synthetic_positives):
        samples.append(
            {
                "label": "positive",
                "query": "synthetic red LEGO brick positive",
                "file": str(path.relative_to(root)),
                "source": {
                    "title": "synthetic red LEGO brick",
                    "descriptionurl": "local synthetic positive",
                },
            }
        )

    for path in generate_synthetic_hard_negatives(root / "negative_synthetic", synthetic_negatives):
        samples.append(
            {
                "label": "negative",
                "query": "synthetic red non-lego hard negative",
                "file": str(path.relative_to(root)),
                "source": {
                    "title": "synthetic red non-LEGO object",
                    "descriptionurl": "local synthetic hard negative",
                },
            }
        )

    manifest = {
        "source": "Wikimedia Commons API + local synthetic hard negatives",
        "target_color": "red",
        "positive_queries": POSITIVE_QUERIES,
        "negative_queries": NEGATIVE_QUERIES,
        "synthetic_positives": synthetic_positives,
        "synthetic_negatives": synthetic_negatives,
        "samples": samples,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def ensure_synthetic_samples(
    root: Path,
    manifest: dict[str, object],
    synthetic_positives: int,
    synthetic_negatives: int,
) -> dict[str, object]:
    samples = list(manifest.get("samples", []))
    seen = {str(sample.get("file")) for sample in samples if isinstance(sample, dict)}

    for path in generate_synthetic_lego_positives(root / "positive_synthetic", synthetic_positives):
        rel = str(path.relative_to(root))
        if rel not in seen:
            samples.append(
                {
                    "label": "positive",
                    "query": "synthetic red LEGO brick positive",
                    "file": rel,
                    "source": {"title": "synthetic red LEGO brick", "descriptionurl": "local synthetic positive"},
                }
            )
            seen.add(rel)

    for path in generate_synthetic_hard_negatives(root / "negative_synthetic", synthetic_negatives):
        rel = str(path.relative_to(root))
        if rel not in seen:
            samples.append(
                {
                    "label": "negative",
                    "query": "synthetic red non-lego hard negative",
                    "file": rel,
                    "source": {"title": "synthetic red non-LEGO object", "descriptionurl": "local synthetic hard negative"},
                }
            )
            seen.add(rel)

    manifest["samples"] = samples
    manifest["synthetic_positives"] = synthetic_positives
    manifest["synthetic_negatives"] = synthetic_negatives
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def commons_search_images(query: str, limit: int) -> Iterable[dict[str, object]]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrlimit": str(limit * 2),
        "gsrsearch": query,
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": "960",
    }
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(make_request(url), timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    pages = list((payload.get("query", {}).get("pages", {}) or {}).values())
    yielded = 0
    for page in pages:
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        mime = info.get("mime", "")
        if mime not in {"image/jpeg", "image/png"}:
            continue
        if int(info.get("width", 0)) < 180 or int(info.get("height", 0)) < 180:
            continue
        image_url = info.get("thumburl") or info.get("url")
        if not image_url:
            continue
        yielded += 1
        yield {
            "title": page.get("title", ""),
            "url": image_url,
            "descriptionurl": info.get("descriptionurl", ""),
            "mime": mime,
            "width": info.get("width"),
            "height": info.get("height"),
        }
        if yielded >= limit:
            break


def download_item(out_dir: Path, item: dict[str, object]) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = str(item["url"])
    suffix = ".jpg" if str(item.get("mime")) == "image/jpeg" else ".png"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:14]
    path = out_dir / f"{digest}{suffix}"
    if path.exists():
        return path
    try:
        with urllib.request.urlopen(make_request(url), timeout=30) as response:
            data = response.read()
        if len(data) < 4096:
            return None
        path.write_bytes(data)
        with Image.open(path) as image:
            image.verify()
        return path
    except Exception as exc:  # pragma: no cover - network/image quality dependent
        print(f"skip {url}: {exc}", file=sys.stderr)
        path.unlink(missing_ok=True)
        return None


def generate_synthetic_hard_negatives(out_dir: Path, count: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(count):
        path = out_dir / f"red_non_lego_{index:03d}.png"
        if not path.exists():
            image = synthetic_red_scene(index)
            image.save(path)
        paths.append(path)
    return paths


def generate_synthetic_lego_positives(out_dir: Path, count: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(count):
        path = out_dir / f"red_lego_{index:03d}.png"
        if not path.exists():
            image = synthetic_lego_scene(index)
            image.save(path)
        paths.append(path)
    return paths


def synthetic_lego_scene(seed: int) -> Image.Image:
    import random

    random.seed(1000 + seed)
    width, height = 640, 420
    image = Image.new("RGB", (width, height), tuple(random.randint(36, 78) for _ in range(3)))
    draw = ImageDraw.Draw(image)

    for _ in range(130):
        x = random.randint(-30, width + 30)
        y = random.randint(-30, height + 30)
        radius = random.randint(5, 30)
        shade = random.randint(50, 145)
        color = (
            max(0, min(255, shade + random.randint(-20, 20))),
            max(0, min(255, shade + random.randint(-20, 24))),
            max(0, min(255, shade + random.randint(-20, 24))),
        )
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    add_simple_lego_brick(draw, random.randint(190, 260), random.randint(130, 210), scale=random.uniform(0.95, 1.25))
    if seed % 3 == 0:
        add_simple_lego_brick(draw, random.randint(350, 450), random.randint(210, 300), scale=random.uniform(0.7, 0.9), color=(30, 120, 220))

    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.35)))


def add_simple_lego_brick(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    scale: float,
    color: tuple[int, int, int] = (226, 48, 45),
) -> None:
    body_w = int(165 * scale)
    body_h = int(82 * scale)
    radius = int(12 * scale)
    x0, y0 = cx - body_w // 2, cy - body_h // 2
    x1, y1 = cx + body_w // 2, cy + body_h // 2
    shadow = (18, 18, 20)
    draw.rounded_rectangle((x0 + 8, y0 + 10, x1 + 8, y1 + 10), radius=radius, fill=shadow)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=color)
    draw.rectangle((x0 + 8, y0 + 9, x1 - 8, y0 + int(20 * scale)), fill=lighten_rgb(color, 42))
    draw.line((x0 + 5, y1 - 6, x1 - 5, y1 - 6), fill=darken_rgb(color, 42), width=max(2, int(4 * scale)))

    stud_r = max(7, int(12 * scale))
    for row in range(2):
        for col in range(4):
            sx = x0 + int((col + 0.5) * body_w / 4)
            sy = y0 + int((row + 0.5) * body_h / 2)
            draw.ellipse((sx - stud_r, sy - stud_r, sx + stud_r, sy + stud_r), fill=lighten_rgb(color, 36))
            draw.ellipse(
                (sx - stud_r + 4, sy - stud_r + 4, sx + stud_r - 4, sy + stud_r - 4),
                outline=darken_rgb(color, 48),
                width=max(1, int(2 * scale)),
            )


def synthetic_red_scene(seed: int) -> Image.Image:
    import random

    random.seed(seed)
    width, height = 640, 420
    base_color = tuple(random.randint(40, 82) for _ in range(3))
    image = Image.new("RGB", (width, height), base_color)
    draw = ImageDraw.Draw(image)

    for _ in range(90):
        x = random.randint(-40, width)
        y = random.randint(-40, height)
        radius = random.randint(8, 42)
        shade = random.randint(55, 130)
        color = (
            max(0, min(255, shade + random.randint(-20, 24))),
            max(0, min(255, shade + random.randint(-18, 22))),
            max(0, min(255, shade + random.randint(-20, 24))),
        )
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    red = (
        random.randint(190, 240),
        random.randint(22, 68),
        random.randint(28, 72),
    )
    kind = seed % 6
    cx = random.randint(180, 460)
    cy = random.randint(120, 300)

    if kind == 0:  # flat book or card
        box = (cx - 130, cy - 70, cx + 130, cy + 70)
        draw.rounded_rectangle(box, radius=10, fill=red)
        draw.line((box[0] + 18, box[1] + 22, box[2] - 16, box[1] + 22), fill=(245, 190, 190), width=4)
    elif kind == 1:  # cup-like cylinder
        draw.ellipse((cx - 72, cy - 96, cx + 72, cy - 34), fill=lighten_rgb(red, 28))
        draw.rectangle((cx - 72, cy - 65, cx + 72, cy + 82), fill=red)
        draw.ellipse((cx - 72, cy + 48, cx + 72, cy + 112), fill=darken_rgb(red, 28))
        draw.arc((cx + 48, cy - 30, cx + 140, cy + 70), -70, 75, fill=lighten_rgb(red, 20), width=18)
    elif kind == 2:  # box face
        draw.polygon([(cx - 120, cy - 60), (cx + 80, cy - 85), (cx + 145, cy + 55), (cx - 55, cy + 92)], fill=red)
        draw.line((cx - 118, cy - 58, cx + 80, cy - 83), fill=lighten_rgb(red, 38), width=5)
    elif kind == 3:  # long tool/marker
        draw.rounded_rectangle((cx - 170, cy - 26, cx + 170, cy + 26), radius=18, fill=red)
        draw.rectangle((cx + 100, cy - 20, cx + 160, cy + 20), fill=darken_rgb(red, 35))
    elif kind == 4:  # flower-like organic red object
        for angle in range(0, 360, 45):
            ox = int(math.cos(math.radians(angle)) * 46)
            oy = int(math.sin(math.radians(angle)) * 34)
            draw.ellipse((cx + ox - 42, cy + oy - 28, cx + ox + 42, cy + oy + 28), fill=red)
        draw.ellipse((cx - 30, cy - 24, cx + 30, cy + 24), fill=darken_rgb(red, 40))
    else:  # toy car-ish blob
        draw.rounded_rectangle((cx - 135, cy - 50, cx + 135, cy + 54), radius=24, fill=red)
        draw.polygon([(cx - 58, cy - 50), (cx - 8, cy - 100), (cx + 72, cy - 92), (cx + 112, cy - 50)], fill=lighten_rgb(red, 20))
        draw.ellipse((cx - 92, cy + 32, cx - 38, cy + 86), fill=(28, 28, 30))
        draw.ellipse((cx + 52, cy + 32, cx + 106, cy + 86), fill=(28, 28, 30))

    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.7)))
    return image


def lighten_rgb(color: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    return tuple(min(255, channel + amount) for channel in color)


def darken_rgb(color: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    return tuple(max(0, channel - amount) for channel in color)


def make_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def tune(manifest: dict[str, object], root: Path) -> dict[str, object]:
    best: dict[str, object] | None = None
    cache = build_candidate_cache(manifest, root)
    for score_min in [0.46, 0.52, 0.58]:
        for fill_min in [0.32, 0.40]:
            for shape_min in [0.45, 0.55]:
                for texture_min in [0.16, 0.18, 0.22]:
                    for separation_min in [0.12, 0.18, 0.24, 0.30]:
                        params = Params(
                            score_min=score_min,
                            fill_min=fill_min,
                            shape_min=shape_min,
                            texture_min=texture_min,
                            separation_min=separation_min,
                        )
                        report = evaluate(manifest, root, params, cache=cache)
                        positive_rate = float(report["positive_detection_rate"])
                        false_positive_rate = float(report["negative_false_positive_rate"])
                        objective = positive_rate - (false_positive_rate * 1.7)
                        report["objective"] = round(objective, 4)
                        if best is None or objective > float(best["objective"]):
                            best = report
    assert best is not None
    return best


def build_candidate_cache(manifest: dict[str, object], root: Path) -> dict[str, list[Detection]]:
    cache: dict[str, list[Detection]] = {}
    for sample in manifest.get("samples", []):
        path = root / str(dict(sample)["file"])
        cache[str(path)] = extract_image_candidates(path)
    return cache


def evaluate(
    manifest: dict[str, object],
    root: Path,
    params: Params,
    cache: dict[str, list[Detection]] | None = None,
) -> dict[str, object]:
    rows = []
    positive_total = positive_hits = 0
    negative_total = negative_hits = 0
    for sample in manifest.get("samples", []):
        sample = dict(sample)
        path = root / str(sample["file"])
        candidates = cache[str(path)] if cache is not None else extract_image_candidates(path)
        detection = select_detection(candidates, params)
        detected = detection is not None
        label = sample["label"]
        if label == "positive":
            positive_total += 1
            positive_hits += int(detected)
        else:
            negative_total += 1
            negative_hits += int(detected)
        rows.append(
            {
                "label": label,
                "query": sample["query"],
                "file": sample["file"],
                "detected": detected,
                "score": round(detection.score, 4) if detection else 0,
                "fill": round(detection.fill, 4) if detection else 0,
                "shape_score": round(detection.shape_score, 4) if detection else 0,
                "texture_score": round(detection.texture_score, 4) if detection else 0,
                "background_separation_score": round(detection.background_separation_score, 4) if detection else 0,
                "area_ratio": round(detection.area_ratio, 4) if detection else 0,
                "bbox": detection.bbox if detection else None,
                "candidate_source": detection.source if detection else "",
                "source_url": sample.get("source", {}).get("descriptionurl", ""),
            }
        )

    positive_rate = positive_hits / max(1, positive_total)
    false_positive_rate = negative_hits / max(1, negative_total)
    return {
        "source": manifest.get("source"),
        "target_color": manifest.get("target_color", "red"),
        "params": asdict(params),
        "positive_total": positive_total,
        "negative_total": negative_total,
        "positive_detection_rate": round(positive_rate, 4),
        "negative_false_positive_rate": round(false_positive_rate, 4),
        "balanced_accuracy_proxy": round((positive_rate + (1 - false_positive_rate)) / 2, 4),
        "rows": rows,
    }


def write_csv(report: dict[str, object], output_path: Path) -> None:
    rows = report.get("rows", [])
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def detect_image(path: Path, params: Params) -> Detection | None:
    return select_detection(extract_image_candidates(path), params)


def extract_image_candidates(path: Path) -> list[Detection]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((320, 240))
        width, height = image.size
        pixels = list(image.getdata())
    mask = build_red_mask(pixels, width, height)
    base_components = [{**component, "source": "color"} for component in connected_components(mask, width, height)]
    components = base_components + shade_components_from_large_regions(pixels, mask, width, height, base_components)
    candidates: list[Detection] = []
    image_area = max(1, width * height)
    expected_ratio = 2.0

    for component in components:
        if component["area"] <= 55:
            continue
        box_w = component["max_x"] - component["min_x"] + 1
        box_h = component["max_y"] - component["min_y"] + 1
        fill = component["area"] / max(1, box_w * box_h)
        short_side = min(box_w, box_h)
        long_side = max(box_w, box_h)
        box_ratio = long_side / max(1, short_side)
        shape_score = clamp(1 - abs(math.log((box_ratio or 1) / expected_ratio)) * 0.5, 0, 1)
        size_score = clamp((component["area"] - 80) / 1500, 0, 1)
        compactness_score = clamp((fill - 0.28) / 0.5, 0, 1)
        lego_like_score = clamp((shape_score * 0.5) + (compactness_score * 0.3) + (size_score * 0.2), 0, 1)
        center_x = component["min_x"] + box_w / 2
        center_y = component["min_y"] + box_h / 2
        dx = (center_x - width / 2) / max(1, width / 2)
        dy = (center_y - height / 2) / max(1, height / 2)
        center_score = clamp(1 - math.hypot(dx, dy) / 1.42, 0, 1)
        area_ratio = component["area"] / image_area
        texture_score = internal_texture_score(pixels, width, height, component)
        separation_score, _, _ = foreground_separation_score(pixels, mask, width, height, component)
        score = clamp(
            (lego_like_score * 0.48)
            + (min(1, component["area"] / 1800) * 0.09)
            + (fill * 0.10)
            + (center_score * 0.07)
            + (texture_score * 0.12)
            + (separation_score * 0.14)
            + (0.04 if component.get("source") == "shade" else 0.0),
            0,
            0.99,
        )
        candidates.append(
            Detection(
                score=score,
                fill=fill,
                shape_score=shape_score,
                texture_score=texture_score,
                background_separation_score=separation_score,
                area_ratio=area_ratio,
                bbox={
                    "x": component["min_x"],
                    "y": component["min_y"],
                    "width": box_w,
                    "height": box_h,
                },
                source=str(component.get("source", "color")),
            )
        )
    return candidates


def select_detection(candidates: list[Detection], params: Params) -> Detection | None:
    filtered = []
    for candidate in candidates:
        width = float(candidate.bbox["width"])
        height = float(candidate.bbox["height"])
        short_side = min(width, height)
        long_side = max(width, height)
        if (
            candidate.score < params.score_min
            or candidate.fill < params.fill_min
            or candidate.shape_score < params.shape_min
            or candidate.texture_score < params.texture_min
            or candidate.background_separation_score < params.separation_min
            or candidate.area_ratio < params.area_min
            or candidate.area_ratio > params.area_max
            or short_side < params.short_side_min
            or long_side < params.long_side_min
        ):
            continue
        filtered.append(candidate)
    return max(filtered, key=lambda item: item.score, default=None)


def build_red_mask(pixels: list[tuple[int, int, int]], width: int, height: int) -> bytearray:
    mask = bytearray(width * height)
    for index, (r, g, b) in enumerate(pixels):
        h, s, light, value = rgb_to_hsl_hsv(r, g, b)
        if s >= 0.28 and 0.12 <= light <= 0.92 and (h >= 342 or h <= 16):
            mask[index] = 1
    dilate(mask, width, height)
    return mask


def rgb_to_hsl_hsv(r: int, g: int, b: int) -> tuple[float, float, float, float]:
    rn, gn, bn = r / 255, g / 255, b / 255
    max_c = max(rn, gn, bn)
    min_c = min(rn, gn, bn)
    delta = max_c - min_c
    hue = 0.0
    if delta:
        if max_c == rn:
            hue = ((gn - bn) / delta) % 6
        elif max_c == gn:
            hue = (bn - rn) / delta + 2
        else:
            hue = (rn - gn) / delta + 4
        hue *= 60
        if hue < 0:
            hue += 360
    light = (max_c + min_c) / 2
    saturation = 0.0 if delta == 0 else delta / (1 - abs(2 * light - 1))
    return hue, saturation, light, max_c


def dilate(mask: bytearray, width: int, height: int) -> None:
    source = mask[:]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            index = y * width + x
            if source[index]:
                continue
            if source[index - 1] or source[index + 1] or source[index - width] or source[index + width]:
                mask[index] = 1


def shade_components_from_large_regions(
    pixels: list[tuple[int, int, int]],
    base_mask: bytearray,
    width: int,
    height: int,
    base_components: list[dict[str, object]],
) -> list[dict[str, object]]:
    image_area = max(1, width * height)
    large_area = max(1600, image_area * 0.055)
    candidates: list[dict[str, object]] = []
    for region in base_components:
        min_x = int(region["min_x"])
        max_x = int(region["max_x"])
        min_y = int(region["min_y"])
        max_y = int(region["max_y"])
        region_width = max_x - min_x + 1
        region_height = max_y - min_y + 1
        if int(region["area"]) < large_area and region_width * region_height < image_area * 0.10:
            continue

        values: list[float] = []
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                index = y * width + x
                if base_mask[index]:
                    values.append(luma(pixels[index]))
        if len(values) < 200:
            continue
        values.sort()

        splits = [
            ("dark", quantile_sorted(values, 0.34)),
            ("dark", quantile_sorted(values, 0.46)),
            ("bright", quantile_sorted(values, 0.68)),
            ("bright", quantile_sorted(values, 0.78)),
        ]
        for mode, threshold in splits:
            shade_mask = bytearray(width * height)
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    index = y * width + x
                    if not base_mask[index]:
                        continue
                    value = luma(pixels[index])
                    if (mode == "dark" and value <= threshold) or (mode == "bright" and value >= threshold):
                        shade_mask[index] = 1

            for component in connected_components(shade_mask, width, height):
                component_width = component["max_x"] - component["min_x"] + 1
                component_height = component["max_y"] - component["min_y"] + 1
                component_box_area = component_width * component_height
                if component["area"] <= 55:
                    continue
                if (
                    component["min_x"] < min_x
                    or component["max_x"] > max_x
                    or component["min_y"] < min_y
                    or component["max_y"] > max_y
                ):
                    continue
                if component_box_area < image_area * 0.006 or component_box_area > image_area * 0.22:
                    continue
                if min(component_width, component_height) < 8 or max(component_width, component_height) < 16:
                    continue
                candidates.append({**component, "source": "shade"})
    return candidates[:24]


def quantile_sorted(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, round((len(values) - 1) * ratio)))
    return values[index]


def connected_components(mask: bytearray, width: int, height: int) -> list[dict[str, int]]:
    visited = bytearray(width * height)
    components: list[dict[str, int]] = []
    for y in range(height):
        for x in range(width):
            index = y * width + x
            if not mask[index] or visited[index]:
                continue
            stack = [(x, y)]
            visited[index] = 1
            area = 0
            min_x = max_x = x
            min_y = max_y = y
            while stack:
                cx, cy = stack.pop()
                area += 1
                min_x, max_x = min(min_x, cx), max(max_x, cx)
                min_y, max_y = min(min_y, cy), max(max_y, cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    ni = ny * width + nx
                    if mask[ni] and not visited[ni]:
                        visited[ni] = 1
                        stack.append((nx, ny))
            components.append({"area": area, "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y})
    return components


def foreground_separation_score(
    pixels: list[tuple[int, int, int]],
    base_mask: bytearray,
    width: int,
    height: int,
    component: dict[str, object],
) -> tuple[float, float, float]:
    min_x = int(component["min_x"])
    max_x = int(component["max_x"])
    min_y = int(component["min_y"])
    max_y = int(component["max_y"])
    box_width = max_x - min_x + 1
    box_height = max_y - min_y + 1
    margin = max(4, min(box_width, box_height) // 6)
    x0 = max(0, min_x - margin)
    y0 = max(0, min_y - margin)
    x1 = min(width - 1, max_x + margin)
    y1 = min(height - 1, max_y + margin)

    outer_count = 0
    outer_mask_count = 0
    outer_luma = 0.0
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            inside = min_x <= x <= max_x and min_y <= y <= max_y
            if inside:
                continue
            index = y * width + x
            outer_count += 1
            outer_mask_count += int(bool(base_mask[index]))
            outer_luma += luma(pixels[index])

    border_width = max(2, min(box_width, box_height) // 10)
    inner_count = 0
    inner_luma = 0.0
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            on_border = (
                x - min_x < border_width
                or max_x - x < border_width
                or y - min_y < border_width
                or max_y - y < border_width
            )
            if not on_border:
                continue
            inner_count += 1
            inner_luma += luma(pixels[y * width + x])

    gradients: list[float] = []
    step = max(1, max(box_width, box_height) // 52)
    for x in range(min_x, max_x + 1, step):
        gradients.append(luma_gradient_at(pixels, width, height, x, min_y))
        gradients.append(luma_gradient_at(pixels, width, height, x, max_y))
    for y in range(min_y, max_y + 1, step):
        gradients.append(luma_gradient_at(pixels, width, height, min_x, y))
        gradients.append(luma_gradient_at(pixels, width, height, max_x, y))
    gradients.sort()

    outer_target_share = outer_mask_count / max(1, outer_count)
    luma_delta = abs((inner_luma / max(1, inner_count)) - (outer_luma / max(1, outer_count)))
    border_gradient = quantile_sorted(gradients, 0.75)
    edge_contrast = clamp((border_gradient - 18) / 42, 0, 1)
    luma_score = clamp(luma_delta / 28, 0, 1)
    outer_penalty = clamp((outer_target_share - 0.42) / 0.50, 0, 1)
    score = clamp(0.16 + edge_contrast * 0.66 + luma_score * 0.28 - outer_penalty * 0.34, 0, 1)
    return score, outer_target_share, edge_contrast


def internal_texture_score(
    pixels: list[tuple[int, int, int]],
    width: int,
    height: int,
    component: dict[str, object],
) -> float:
    min_x = max(1, int(component["min_x"]))
    max_x = min(width - 2, int(component["max_x"]))
    min_y = max(1, int(component["min_y"]))
    max_y = min(height - 2, int(component["max_y"]))
    if max_x <= min_x or max_y <= min_y:
        return 0.0
    step = max(1, int(max(max_x - min_x, max_y - min_y) / 36))
    strong_edges = 0
    total = 0
    for y in range(min_y, max_y + 1, step):
        for x in range(min_x, max_x + 1, step):
            center = luma(pixels[y * width + x])
            gx = abs(luma(pixels[y * width + x + 1]) - luma(pixels[y * width + x - 1]))
            gy = abs(luma(pixels[(y + 1) * width + x]) - luma(pixels[(y - 1) * width + x]))
            local_contrast = max(gx, gy, abs(center - luma(pixels[(y - 1) * width + x - 1])))
            strong_edges += int(local_contrast > 18)
            total += 1
    return strong_edges / max(1, total)


def luma(pixel: tuple[int, int, int]) -> float:
    r, g, b = pixel
    return 0.299 * r + 0.587 * g + 0.114 * b


def luma_gradient_at(pixels: list[tuple[int, int, int]], width: int, height: int, x: int, y: int) -> float:
    left = luma(pixels[y * width + max(0, x - 1)])
    right = luma(pixels[y * width + min(width - 1, x + 1)])
    top = luma(pixels[max(0, y - 1) * width + x])
    bottom = luma(pixels[min(height - 1, y + 1) * width + x])
    return max(abs(right - left), abs(bottom - top))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


if __name__ == "__main__":
    main()
