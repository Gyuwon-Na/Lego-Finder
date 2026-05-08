"""Candidate proposal stage for LEGO pile images."""

from __future__ import annotations

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
