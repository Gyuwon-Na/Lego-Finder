"""Render annotated and AR-style result images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_annotation(image_path: Path, out_path: Path, result: dict) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image: {image_path}")
    target = result["target"]
    color = hex_to_bgr(target["colorCss"])
    for detection in result.get("detections", []):
        draw_detection(image, detection, color, glow=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def save_ar_overlay(image_path: Path, out_path: Path, result: dict) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image: {image_path}")
    target = result["target"]
    color = hex_to_bgr(target["colorCss"])
    overlay = image.copy()
    for detection in result.get("detections", [])[:5]:
        draw_detection(overlay, detection, color, glow=True)
    image = cv2.addWeighted(overlay, 0.76, image, 0.24, 0)
    add_status_bar(image, result)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def draw_detection(image: np.ndarray, detection: dict, color: tuple[int, int, int], glow: bool) -> None:
    box = detection["bbox"]
    x = box["x"]
    y = box["y"]
    w = box["width"]
    h = box["height"]
    polygon = np.array(detection.get("maskPolygon") or [], dtype=np.int32)
    if glow:
        for thickness, alpha in [(22, 0.10), (14, 0.15), (8, 0.22)]:
            layer = image.copy()
            if len(polygon) >= 3:
                cv2.polylines(layer, [polygon], True, color, thickness, cv2.LINE_AA)
            else:
                cv2.rectangle(layer, (x, y), (x + w, y + h), color, thickness)
            cv2.addWeighted(layer, alpha, image, 1 - alpha, 0, image)
    if len(polygon) >= 3:
        cv2.polylines(image, [polygon], True, color, 4, cv2.LINE_AA)
    else:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 4)
    label = f"#{detection['rank']} {detection['score']:.2f}"
    if "ransac" in detection:
        label += f" R{detection['ransac'].get('inliers', 0)}"
    cv2.putText(image, label, (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)


def add_status_bar(image: np.ndarray, result: dict) -> None:
    height, width = image.shape[:2]
    bar_h = 76
    cv2.rectangle(image, (0, height - bar_h), (width, height), (8, 12, 26), -1)
    target = result["target"]
    text = f"Searching: {target['colorLabel']} {target['width']}x{target['length']} {target['categoryLabel']}"
    found = f"found: {len(result.get('detections', []))}"
    cv2.putText(image, text, (24, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (235, 235, 235), 2)
    cv2.putText(image, found, (24, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (155, 170, 205), 2)


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    clean = hex_color.lstrip("#")
    red = int(clean[0:2], 16)
    green = int(clean[2:4], 16)
    blue = int(clean[4:6], 16)
    return blue, green, red
