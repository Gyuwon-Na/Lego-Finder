#!/usr/bin/env python3
"""Fetch Rebrickable parts and build a local metadata/vector artifact.

This script needs a Rebrickable API key:
    $env:REBRICKABLE_API_KEY="..."
    python scripts/build_rebrickable_index.py --pages 3

The generated JSON is intentionally simple so it can feed either hnswlib,
Qdrant, or an on-device bundle later.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backend"))

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from app.catalog import PART_NUMBERS, make_target  # noqa: E402
from app.main import infer_category, infer_color_key, infer_dimensions  # noqa: E402
from app.vector_index import text_part_embedding  # noqa: E402


DEFAULT_SET_NUMS = ["75257-1"]
DEFAULT_PART_NUMS = sorted(set(PART_NUMBERS.values()))
PART_NUM_ALIASES = {
    "3068": ["3068b"],
    "3069": ["3069b"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/rebrickable/parts_index.json")
    parser.add_argument("--pages", type=int, default=0, help="number of generic Rebrickable parts pages to fetch")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--set-nums", default=",".join(DEFAULT_SET_NUMS), help="comma-separated set numbers")
    parser.add_argument("--common-parts", default=",".join(DEFAULT_PART_NUMS), help="comma-separated part numbers")
    parser.add_argument("--colors-per-part", type=int, default=10)
    parser.add_argument("--download-images", action="store_true", help="cache Rebrickable part images for template matching")
    parser.add_argument("--download-limit", type=int, default=120)
    args = parser.parse_args()

    key = os.getenv("REBRICKABLE_API_KEY")
    if not key:
        raise SystemExit("Set REBRICKABLE_API_KEY before running this script.")

    records = []
    common_parts = split_csv(args.common_parts)
    set_nums = split_csv(args.set_nums)
    if common_parts:
        records.extend(fetch_part_color_variants(key, common_parts, args.colors_per_part))
    if set_nums:
        records.extend(fetch_set_parts(key, set_nums, page_size=args.page_size))
    if args.pages > 0:
        records.extend(fetch_parts(key, pages=args.pages, page_size=args.page_size))
    records = dedupe(records)
    if args.download_images:
        cache_images(key, records, ROOT / "data" / "rebrickable" / "images", limit=args.download_limit)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    maybe_write_hnsw(records, out_path.with_suffix(".hnsw.bin"))
    print(f"wrote {len(records)} records to {out_path}")


def fetch_parts(key: str, pages: int, page_size: int) -> list[dict]:
    headers = {"Authorization": f"key {key}"}
    url = "https://rebrickable.com/api/v3/lego/parts/"
    records = []
    with httpx.Client(timeout=20.0) as client:
        for page in range(1, pages + 1):
            res = client.get(url, headers=headers, params={"page": page, "page_size": page_size})
            res.raise_for_status()
            for item in res.json().get("results", []):
                record = normalize_part(item)
                if record:
                    records.append(record)
    return records


def fetch_part_color_variants(key: str, part_nums: list[str], colors_per_part: int) -> list[dict]:
    headers = {"Authorization": f"key {key}"}
    records = []
    with httpx.Client(timeout=20.0) as client:
        for part_num in part_nums:
            part_info = None
            colors_payload = None
            resolved_part_num = None
            for candidate in [part_num, *PART_NUM_ALIASES.get(part_num, [])]:
                part_info = fetch_part_detail(client, headers, candidate)
                if not part_info:
                    continue
                url = f"https://rebrickable.com/api/v3/lego/parts/{candidate}/colors/"
                res = client.get(url, headers=headers, params={"page_size": max(1, colors_per_part)})
                if res.status_code == 404:
                    continue
                res.raise_for_status()
                colors_payload = res.json()
                resolved_part_num = candidate
                break
            if not part_info or not colors_payload or not resolved_part_num:
                print(f"skipped missing part {part_num}")
                continue
            for item in colors_payload.get("results", []):
                record = normalize_part_color_variant(part_info, item)
                if record:
                    records.append(record)
    return records


def fetch_part_detail(client: httpx.Client, headers: dict[str, str], part_num: str) -> dict:
    res = client.get(f"https://rebrickable.com/api/v3/lego/parts/{part_num}/", headers=headers)
    if res.status_code == 404:
        return {}
    res.raise_for_status()
    return res.json()


def fetch_set_parts(key: str, set_nums: list[str], page_size: int) -> list[dict]:
    headers = {"Authorization": f"key {key}"}
    records = []
    with httpx.Client(timeout=25.0) as client:
        for set_num in set_nums:
            url = f"https://rebrickable.com/api/v3/lego/sets/{set_num}/parts/"
            page = 1
            while True:
                res = client.get(
                    url,
                    headers=headers,
                    params={
                        "page": page,
                        "page_size": page_size,
                        "inc_color_details": 1,
                        "inc_part_details": 1,
                    },
                )
                res.raise_for_status()
                payload = res.json()
                for item in payload.get("results", []):
                    record = normalize_set_part(set_num, item)
                    if record:
                        records.append(record)
                if not payload.get("next"):
                    break
                page += 1
    return records


def normalize_part(item: dict) -> dict | None:
    name = item.get("name") or ""
    parsed = parse_supported_part(name)
    if not parsed:
        return None
    color_key = infer_color_key(name)
    width, length, category = parsed
    target = make_target(color_key, width, length, category, source="rebrickable-index")
    payload = target.to_dict()
    payload["partNum"] = item.get("part_num") or payload["partNum"]
    payload["externalName"] = name
    payload["partImgUrl"] = item.get("part_img_url")
    return make_record(payload, target, source_id=payload["partNum"])


def normalize_part_color_variant(part_info: dict, color_item: dict) -> dict | None:
    name = part_info.get("name") or ""
    parsed = parse_supported_part(name)
    if not parsed:
        return None
    color_name = color_item.get("color_name") or ""
    color_key = infer_color_key(color_name)
    width, length, category = parsed
    target = make_target(color_key, width, length, category, source="rebrickable-color-index")
    payload = target.to_dict()
    payload["partNum"] = part_info.get("part_num") or payload["partNum"]
    payload["externalName"] = name
    payload["rebrickableColorName"] = color_name
    payload["rebrickableColorId"] = color_item.get("color_id")
    payload["numSets"] = color_item.get("num_sets")
    payload["numSetParts"] = color_item.get("num_set_parts")
    payload["partImgUrl"] = color_item.get("part_img_url")
    payload["elements"] = color_item.get("elements") or []
    return make_record(payload, target, source_id=f"{payload['partNum']}-{payload['rebrickableColorId']}")


def normalize_set_part(set_num: str, item: dict) -> dict | None:
    part_info = item.get("part") or {}
    color_info = item.get("color") or {}
    name = part_info.get("name") or ""
    parsed = parse_supported_part(name)
    if not parsed:
        return None
    color_name = color_info.get("name") or ""
    color_key = infer_color_key(color_name)
    width, length, category = parsed
    target = make_target(color_key, width, length, category, item.get("quantity", 1), "rebrickable-set-index")
    payload = target.to_dict()
    payload["partNum"] = part_info.get("part_num") or payload["partNum"]
    payload["externalName"] = name
    payload["setNum"] = set_num
    payload["rebrickableColorName"] = color_name
    payload["rebrickableColorId"] = color_info.get("id")
    payload["partImgUrl"] = item.get("part_img_url") or part_info.get("part_img_url")
    return make_record(payload, target, source_id=f"{set_num}-{payload['partNum']}-{payload.get('rebrickableColorId')}")


def make_record(payload: dict, target: object, source_id: str) -> dict:
    return {
        "id": source_id,
        "payload": payload,
        "vector": text_part_embedding(target).round(6).tolist(),
    }


def parse_supported_part(name: str) -> tuple[int, int, str] | None:
    lowered = name.lower()
    if not any(token in lowered for token in ["brick", "plate", "tile"]):
        return None
    match = re.search(r"(\d+)\s*x\s*(\d+)", name, re.I)
    if not match:
        return None
    width, length = int(match.group(1)), int(match.group(2))
    return width, length, infer_category(name)


def dedupe(records: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for record in records:
        payload = record["payload"]
        key = f"{payload.get('partNum')}:{payload.get('rebrickableColorId', payload.get('colorKey'))}"
        existing = best.get(key)
        if existing is None:
            best[key] = record
            continue
        existing_score = existing["payload"].get("numSetParts") or 0
        score = payload.get("numSetParts") or 0
        if score > existing_score:
            best[key] = record
    return sorted(best.values(), key=lambda record: str(record["id"]))


def cache_images(key: str, records: list[dict], out_dir: Path, limit: int) -> None:
    headers = {"Authorization": f"key {key}"}
    out_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for record in records[: max(0, limit)]:
            url = record["payload"].get("partImgUrl")
            if not url:
                continue
            filename = image_filename(record["payload"])
            path = out_dir / filename
            if path.exists():
                record["payload"]["localImagePath"] = str(path.relative_to(ROOT)).replace("\\", "/")
                continue
            try:
                res = client.get(url, headers=headers)
                res.raise_for_status()
            except httpx.HTTPError:
                continue
            path.write_bytes(res.content)
            record["payload"]["localImagePath"] = str(path.relative_to(ROOT)).replace("\\", "/")


def image_filename(payload: dict) -> str:
    part_num = sanitize(str(payload.get("partNum") or "part"))
    color = sanitize(str(payload.get("rebrickableColorName") or payload.get("colorKey") or "color"))
    color_id = sanitize(str(payload.get("rebrickableColorId") or "x"))
    return f"{part_num}-{color_id}-{color}.jpg"


def sanitize(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()[:80]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def maybe_write_hnsw(records: list[dict], path: Path) -> None:
    try:
        import hnswlib  # type: ignore
    except Exception:
        return
    vectors = np.array([record["vector"] for record in records], dtype=np.float32)
    if len(vectors) == 0:
        return
    index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
    index.init_index(max_elements=len(vectors), ef_construction=120, M=16)
    index.add_items(vectors, list(range(len(vectors))))
    index.set_ef(32)
    index.save_index(str(path))


if __name__ == "__main__":
    main()
