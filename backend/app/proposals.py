"""Candidate proposal stage for LEGO pile images."""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np

from .catalog import COLORS


HSV_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 90, 45), (12, 255, 255)), ((170, 90, 45), (180, 255, 255))],
    "orange": [((8, 90, 55), (24, 255, 255))],
    "yellow": [((22, 82, 70), (38, 255, 255))],
    "green": [((40, 78, 40), (88, 255, 255))],
    "blue": [((92, 78, 35), (132, 255, 255))],
    "purple": [((132, 78, 35), (164, 255, 255))],
}


@dataclass(frozen=True)
class Proposal:
    id: int
    bbox: dict[str, int]
    area: int
    aspectRatio: float
    fillRatio: float
    solidity: float
    dominantColorKey: str
    colorShares: dict[str, float]
    proposalScore: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenCVCandidateProposalModel:
    """Fast local proposal model used until YOLO/SAM/MobileNet weights exist."""

    name = "opencv-color-connected-components"

    def __init__(self, max_side: int = 1280):
        self.max_side = max_side

    def propose(self, image: np.ndarray, max_proposals: int = 60) -> list[Proposal]:
        resized, scale = resize_for_search(image, max_side=self.max_side)
        hsv = cv2.cvtColor(cv2.GaussianBlur(resized, (5, 5), 0), cv2.COLOR_BGR2HSV)
        masks = {key: color_mask(hsv, key) for key in COLORS}
        combined = all_color_mask(hsv)
        kernel = np.ones((5, 5), dtype=np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        proposals: list[Proposal] = []
        for index, contour in enumerate(contours):
            contour_area = int(cv2.contourArea(contour))
            if contour_area < 100:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 10 or h < 10:
                continue
            bbox_area = max(1, w * h)
            fill_ratio = int(np.count_nonzero(combined[y : y + h, x : x + w])) / bbox_area
            if fill_ratio < 0.16:
                continue
            aspect_ratio = max(w, h) / max(1, min(w, h))
            solidity = min(1.0, contour_area / bbox_area)
            shares = color_shares(masks, x, y, w, h)
            dominant = max(shares.items(), key=lambda item: item[1])[0] if shares else "red"
            proposal_score = min(0.999, fill_ratio * 0.42 + solidity * 0.34 + min(1.0, contour_area / 2200) * 0.24)
            proposals.append(
                Proposal(
                    id=index,
                    bbox={
                        "x": int(round(x / scale)),
                        "y": int(round(y / scale)),
                        "width": int(round(w / scale)),
                        "height": int(round(h / scale)),
                    },
                    area=int(round(contour_area / (scale * scale))),
                    aspectRatio=round(float(aspect_ratio), 3),
                    fillRatio=round(float(fill_ratio), 3),
                    solidity=round(float(solidity), 3),
                    dominantColorKey=dominant,
                    colorShares={key: round(float(value), 3) for key, value in shares.items()},
                    proposalScore=round(float(proposal_score), 4),
                )
            )

        return sorted(proposals, key=lambda proposal: proposal.proposalScore, reverse=True)[:max_proposals]


class YoloSegCandidateProposalModel:
    """Optional YOLO segmentation proposal model with OpenCV fallback semantics.

    Set BRICKFINDER_YOLO_MODEL to a local ultralytics segmentation checkpoint
    such as a LEGO-finetuned YOLOv8/YOLO11 seg model. The model is loaded lazily
    so the default MVP still runs without heavyweight dependencies.
    """

    name = "yolo-segmentation"

    def __init__(self, model_path: str | None = None, max_side: int = 1280, fallback: OpenCVCandidateProposalModel | None = None):
        self.model_path = model_path or os.getenv("BRICKFINDER_YOLO_MODEL")
        self.max_side = max_side
        self.fallback = fallback or OpenCVCandidateProposalModel(max_side=max_side)
        self._model = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.model_path and self._load_model() is not None)

    @property
    def status(self) -> str:
        if self.available:
            return f"{self.name}:{self.model_path}"
        if self._load_error:
            return f"{self.name}-unavailable:{self._load_error}; fallback={self.fallback.name}"
        return f"{self.name}-not-configured; fallback={self.fallback.name}"

    def propose(self, image: np.ndarray, max_proposals: int = 60) -> list[Proposal]:
        model = self._load_model()
        if model is None:
            return self.fallback.propose(image, max_proposals=max_proposals)

        resized, scale = resize_for_search(image, max_side=self.max_side)
        results = model.predict(resized, verbose=False, imgsz=max(resized.shape[:2]), conf=0.18)
        proposals: list[Proposal] = []
        if not results:
            return self.fallback.propose(image, max_proposals=max_proposals)

        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None:
            return self.fallback.propose(image, max_proposals=max_proposals)

        xyxy = boxes.xyxy.cpu().numpy() if getattr(boxes, "xyxy", None) is not None else []
        confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.ones(len(xyxy))
        mask_data = masks.data.cpu().numpy() if masks is not None and getattr(masks, "data", None) is not None else []
        hsv = cv2.cvtColor(cv2.GaussianBlur(resized, (5, 5), 0), cv2.COLOR_BGR2HSV)
        color_masks = {key: color_mask(hsv, key) for key in COLORS}

        for index, box in enumerate(xyxy):
            x0, y0, x1, y1 = [int(round(value)) for value in box]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(resized.shape[1], x1), min(resized.shape[0], y1)
            w, h = max(0, x1 - x0), max(0, y1 - y0)
            if w < 8 or h < 8:
                continue
            if index < len(mask_data):
                mask = cv2.resize(mask_data[index].astype(np.float32), (resized.shape[1], resized.shape[0])) > 0.5
                mask_crop = mask[y0:y1, x0:x1]
                area = int(np.count_nonzero(mask_crop))
            else:
                mask_crop = np.ones((h, w), dtype=bool)
                area = w * h
            if area < 80:
                continue
            bbox_area = max(1, w * h)
            fill_ratio = area / bbox_area
            if fill_ratio < 0.12:
                continue
            shares = masked_color_shares(color_masks, x0, y0, w, h, mask_crop)
            dominant = max(shares.items(), key=lambda item: item[1])[0] if shares else "red"
            aspect_ratio = max(w, h) / max(1, min(w, h))
            confidence = float(confs[index]) if index < len(confs) else 0.5
            proposal_score = min(0.999, confidence * 0.46 + fill_ratio * 0.26 + min(1.0, area / 2200) * 0.28)
            proposals.append(
                Proposal(
                    id=index,
                    bbox={
                        "x": int(round(x0 / scale)),
                        "y": int(round(y0 / scale)),
                        "width": int(round(w / scale)),
                        "height": int(round(h / scale)),
                    },
                    area=int(round(area / (scale * scale))),
                    aspectRatio=round(float(aspect_ratio), 3),
                    fillRatio=round(float(fill_ratio), 3),
                    solidity=round(float(fill_ratio), 3),
                    dominantColorKey=dominant,
                    colorShares={key: round(float(value), 3) for key, value in shares.items()},
                    proposalScore=round(float(proposal_score), 4),
                )
            )

        if not proposals:
            return self.fallback.propose(image, max_proposals=max_proposals)
        return sorted(proposals, key=lambda proposal: proposal.proposalScore, reverse=True)[:max_proposals]

    def _load_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        if not self.model_path:
            return None
        if self._load_error:
            return None
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.model_path)
            return self._model
        except Exception as exc:  # pragma: no cover - optional dependency
            self._load_error = exc.__class__.__name__
            return None


def resize_for_search(image: np.ndarray, max_side: int = 960) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(width, height)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    resized = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def color_mask(hsv: np.ndarray, color_key: str) -> np.ndarray:
    if color_key == "black":
        return cv2.inRange(hsv, np.array((0, 0, 0)), np.array((180, 255, 72)))
    if color_key == "white":
        return cv2.inRange(hsv, np.array((0, 0, 185)), np.array((180, 50, 255)))
    if color_key == "gray":
        return cv2.inRange(hsv, np.array((0, 0, 70)), np.array((180, 55, 210)))
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_RANGES.get(color_key, HSV_RANGES["red"]):
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))
    return mask


def all_color_mask(hsv: np.ndarray, include_neutral: bool = False) -> np.ndarray:
    keys = list(HSV_RANGES.keys())
    if include_neutral:
        keys.extend(["black", "white", "gray"])
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for key in keys:
        mask = cv2.bitwise_or(mask, color_mask(hsv, key))
    return mask


def color_shares(masks: dict[str, np.ndarray], x: int, y: int, w: int, h: int) -> dict[str, float]:
    counts = {}
    total = 0
    for key, mask in masks.items():
        count = int(np.count_nonzero(mask[y : y + h, x : x + w]))
        if key in {"black", "white", "gray"}:
            count = int(count * 0.45)
        counts[key] = count
        total += count
    if total <= 0:
        return {key: 0.0 for key in counts}
    return {key: count / total for key, count in counts.items()}


def masked_color_shares(
    masks: dict[str, np.ndarray],
    x: int,
    y: int,
    w: int,
    h: int,
    local_mask: np.ndarray,
) -> dict[str, float]:
    counts = {}
    total = 0
    mask_bool = local_mask.astype(bool)
    for key, mask in masks.items():
        crop = mask[y : y + h, x : x + w]
        if crop.shape[:2] != mask_bool.shape[:2]:
            crop = cv2.resize(crop, (mask_bool.shape[1], mask_bool.shape[0]), interpolation=cv2.INTER_NEAREST)
        count = int(np.count_nonzero((crop > 0) & mask_bool))
        if key in {"black", "white", "gray"}:
            count = int(count * 0.45)
        counts[key] = count
        total += count
    if total <= 0:
        return {key: 0.0 for key in counts}
    return {key: count / total for key, count in counts.items()}
