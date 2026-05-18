#!/usr/bin/env python3
"""Evaluate BrickFinder image search against synthetic eval_queries.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.vision import search_image_bytes  # noqa: E402


def box_iou(left: Dict[str, int], right: Dict[str, int]) -> float:
    lx0, ly0 = left["x"], left["y"]
    lx1, ly1 = lx0 + left["width"], ly0 + left["height"]
    rx0, ry0 = right["x"], right["y"]
    rx1, ry1 = rx0 + right["width"], ry0 + right["height"]
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    intersection = iw * ih
    union = left["width"] * left["height"] + right["width"] * right["height"] - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def evaluate(root: Path, limit: int, iou_threshold: float) -> Dict[str, Any]:
    queries = json.loads((root / "eval_queries.json").read_text(encoding="utf-8"))
    annotations_path = root / "annotations.json"
    if annotations_path.exists():
        annotation_map = {
            item["file"]: item.get("instances", [])
            for item in json.loads(annotations_path.read_text(encoding="utf-8"))
        }
        for query in queries:
            query["instances"] = annotation_map.get(query["file"], [])
    if limit:
        queries = queries[:limit]

    rows: List[Dict[str, Any]] = []
    top1_hits = 0
    top3_hits = 0
    any_hits = 0
    iou_sum = 0.0

    for index, query in enumerate(queries, start=1):
        image_path = root / query["file"]
        image_bytes = image_path.read_bytes()
        result = search_image_bytes(image_bytes, query["prompt"], max_results=8)
        detections = result.get("detections", [])
        target = query["target"]
        match_boxes = matching_target_boxes(query, target)
        target_box = target["bbox"]
        ious = [best_iou_for_detection(match_boxes, detection["bbox"]) for detection in detections]
        best_iou = max(ious or [0.0])
        top1_iou = ious[0] if ious else 0.0
        top3_iou = max(ious[:3] or [0.0])

        top1_ok = top1_iou >= iou_threshold
        top3_ok = top3_iou >= iou_threshold
        any_ok = best_iou >= iou_threshold
        top1_hits += int(top1_ok)
        top3_hits += int(top3_ok)
        any_hits += int(any_ok)
        iou_sum += top1_iou
        rows.append(
            {
                "file": query["file"],
                "prompt": query["prompt"],
                "target_bbox": target_box,
                "accepted_boxes": match_boxes,
                "detections": len(detections),
                "top1_iou": round(top1_iou, 4),
                "top3_iou": round(top3_iou, 4),
                "best_iou": round(best_iou, 4),
                "top1_ok": top1_ok,
                "top3_ok": top3_ok,
                "any_ok": any_ok,
                "top1_bbox": detections[0]["bbox"] if detections else None,
            }
        )
        print(
            "{:03d}/{:03d} {} top1_iou={:.3f} best_iou={:.3f} detections={}".format(
                index, len(queries), query["prompt"], top1_iou, best_iou, len(detections)
            )
        )

    total = max(1, len(queries))
    return {
        "total": len(queries),
        "iou_threshold": iou_threshold,
        "top1_accuracy": round(top1_hits / total, 4),
        "top3_accuracy": round(top3_hits / total, 4),
        "any_accuracy": round(any_hits / total, 4),
        "mean_top1_iou": round(iou_sum / total, 4),
        "rows": rows,
    }


def matching_target_boxes(query: Dict[str, Any], target: Dict[str, Any]) -> List[Dict[str, int]]:
    instances = query.get("instances")
    if not instances:
        return [target["bbox"]]
    boxes = []
    for instance in instances:
        if (
            instance.get("color") == target.get("color")
            and instance.get("category") == target.get("category")
            and sorted((instance.get("width"), instance.get("length")))
            == sorted((target.get("width"), target.get("length")))
        ):
            boxes.append(instance["bbox"])
    return boxes or [target["bbox"]]


def best_iou_for_detection(target_boxes: List[Dict[str, int]], detection_box: Dict[str, int]) -> float:
    return max((box_iou(target_box, detection_box) for target_box in target_boxes), default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "data" / "synthetic_lego"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.dataset)
    report = evaluate(root, args.limit, args.iou)
    output = Path(args.out) if args.out else root / "evaluation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
