#!/usr/bin/env python3
"""Search a local LEGO pile image with a text prompt.

Example:
    python scripts/search_image.py --image data/demo/lego_pile.png --query "2x4 red brick"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backend"))

from app.ar_render import save_annotation, save_ar_overlay  # noqa: E402
from app.vision import search_image_bytes  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="photo containing many LEGO blocks")
    parser.add_argument("--query", required=True, help='target prompt, e.g. "2x4 red brick"')
    parser.add_argument("--out", default="data/search_results/annotated.png", help="annotated image path")
    parser.add_argument("--ar-out", default="data/search_results/ar_overlay.png", help="AR-style overlay image path")
    parser.add_argument("--max-results", type=int, default=8)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"image not found: {image_path}")

    result = search_image_bytes(image_path.read_bytes(), args.query, max_results=args.max_results)
    out_path = ROOT / args.out
    if Path(args.out).is_absolute():
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ar_out_path = ROOT / args.ar_out
    if Path(args.ar_out).is_absolute():
        ar_out_path = Path(args.ar_out)
    save_annotation(image_path, out_path, result)
    save_ar_overlay(image_path, ar_out_path, result)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"annotated={out_path}")
    print(f"ar_overlay={ar_out_path}")


if __name__ == "__main__":
    main()
