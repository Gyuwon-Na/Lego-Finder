"""End-to-end image search pipeline for the BrickFinder MVP.

The concrete implementation is local and lightweight:
query parser -> vector index -> candidate proposal model -> verifier/reranker.
Each boundary mirrors the production design, so CLIP/MobileCLIP, YOLO/SAM, and
Qdrant/hnswlib can be swapped in without changing API callers.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np

from .catalog import COLORS
from .proposals import YoloSegCandidateProposalModel, Proposal, color_mask, contour_polygon
from .query_parser import parse_query
from .vector_index import PartVectorIndex, cosine, proposal_embedding, text_part_embedding


@dataclass(frozen=True)
class Detection:
    rank: int
    score: float
    bbox: dict[str, int]
    area: int
    aspectRatio: float
    fillRatio: float
    colorDominance: float
    shapeScore: float
    solidity: float
    proposalScore: float
    embeddingScore: float
    vectorScore: float
    edgeContrast: float
    outerTargetShare: float
    backgroundSeparation: float
    dominantColorKey: str
    colorShares: dict[str, float]
    ransac: dict[str, Any]
    maskPolygon: list[list[int]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MOBILE_ACCURACY_PROFILE = {
    "name": "mobile-accuracy",
    "proposal_max_side": 1280,
    "max_proposals": 96,
    "index_top_k": 16,
    "default_max_results": 8,
}

PART_INDEX = PartVectorIndex()
PROPOSAL_MODEL = YoloSegCandidateProposalModel(max_side=MOBILE_ACCURACY_PROFILE["proposal_max_side"])


def search_image_bytes(image_bytes: bytes, query: str, max_results: int = 8) -> dict[str, Any]:
    target = parse_query(query, source="image-search")
    image = decode_image(image_bytes)
    index_hits = PART_INDEX.search(target, top_k=MOBILE_ACCURACY_PROFILE["index_top_k"])
    proposals = PROPOSAL_MODEL.propose(image, max_proposals=MOBILE_ACCURACY_PROFILE["max_proposals"])
    detections = rank_proposals(image, proposals, target, max_results=max_results)
    height, width = image.shape[:2]
    return {
        "target": target.to_dict(),
        "image": {"width": width, "height": height},
        "index": {
            "backend": PART_INDEX.backend,
            "hits": [hit.to_dict() for hit in index_hits],
        },
        "proposals": {
            "model": PROPOSAL_MODEL.name,
            "count": len(proposals),
            "items": [proposal.to_dict() for proposal in proposals[:20]],
        },
        "detections": [detection.to_dict() for detection in detections],
        "pipeline": {
            "profile": MOBILE_ACCURACY_PROFILE["name"],
            "query_encoder": "deterministic parser + CLIP-compatible metadata embedding fallback",
            "vector_index": f"{PART_INDEX.backend}; hnswlib is used automatically when installed",
            "index_records": len(PART_INDEX.payloads),
            "candidate_proposal": PROPOSAL_MODEL.name,
            "segmentation": PROPOSAL_MODEL.status,
            "verification": "color dominance + dimension geometry + RANSAC corner homography fallback",
            "prompt_understanding": "structured target spec with color/dimensions/category/height/shape/negative terms; CLIP-ready",
            "ar_rendering": "scripts/search_image.py can save annotation and AR-style overlay images",
        },
    }


def decode_image(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image")
    return image


def find_target_detections(
    image: np.ndarray,
    color_key: str,
    target_width: int,
    target_length: int,
    max_results: int = 8,
) -> list[Detection]:
    query = f"{target_width}x{target_length} {color_key} brick"
    target = parse_query(query, source="image-search")
    proposals = PROPOSAL_MODEL.propose(image)
    return rank_proposals(image, proposals, target, max_results=max_results)


def rank_proposals(image: np.ndarray, proposals: list[Proposal], target: Any, max_results: int = 8) -> list[Detection]:
    expected_ratio = max(target.width, target.length) / max(1, min(target.width, target.length))
    target_vector = text_part_embedding(target)
    image_area = max(1, image.shape[0] * image.shape[1])
    min_target_area = max(240, int(image_area * 0.0005))
    expected_studs = max(1, target.width * target.length)
    expected_bbox_area = image_area * 0.024 * (expected_studs / 8) ** 0.72
    detections: list[Detection] = []

    candidate_features = target_color_image_components(image, target.colorKey)
    candidate_features.extend(target_color_shade_components(image, target.colorKey, expected_ratio, expected_bbox_area))
    candidate_features.extend(target_color_edge_rectangles(image, target.colorKey, expected_ratio, expected_bbox_area))
    for proposal in proposals:
        candidate_features.extend(target_color_subproposals(image, proposal, target.colorKey))
    if needs_local_window_search(image, proposals):
        candidate_features.extend(target_color_search_windows(image, target.colorKey, expected_ratio, expected_bbox_area))

    for box_features in candidate_features:
        target_share = box_features["colorShares"].get(target.colorKey, 0.0)
        if target_share < 0.08:
            continue
        if box_features["area"] < min_target_area:
            continue
        aspect_ratio = box_features["aspectRatio"]
        fill_ratio = box_features["fillRatio"]
        dominant_color = box_features["dominantColorKey"]
        shape_score = ratio_score(aspect_ratio, expected_ratio)
        bbox_area = max(1, box_features["bbox"]["width"] * box_features["bbox"]["height"])
        color_area_score = min(1.0, box_features["area"] / max(1.0, expected_bbox_area * 0.55))
        bbox_size_score = size_floor_score(bbox_area, expected_bbox_area)
        embedding_score = cosine(
            proposal_embedding(dominant_color, aspect_ratio, fill_ratio, target),
            target_vector,
        )
        ransac = verify_geometry(image, box_features, expected_ratio)
        ransac_score = ransac["score"]
        keypoint_score = min(1.0, ransac["inliers"] / max(8.0, expected_studs * 8.0 + 8.0))
        vector_score = embedding_score
        border_score = border_containment_score(image, box_features["bbox"])
        separation_score = float(box_features.get("backgroundSeparationScore", 0.5))
        edge_contrast = float(box_features.get("edgeContrast", 0.0))
        outer_target_share = float(box_features.get("outerTargetShare", 0.0))

        score = (
            target_share * 0.16
            + bbox_size_score * 0.20
            + keypoint_score * 0.14
            + shape_score * 0.09
            + embedding_score * 0.09
            + box_features["proposalScore"] * 0.08
            + color_area_score * 0.06
            + ransac_score * 0.05
            + separation_score * 0.11
            + border_score * 0.02
        )
        if bbox_area < expected_bbox_area * 0.32:
            score *= 0.58
        elif bbox_area < expected_bbox_area * 0.48:
            score *= 0.78
        if bbox_area > image_area * 0.42:
            score *= 0.62
        if dominant_color != target.colorKey:
            score *= 0.72
        if target_share < 0.35:
            score *= 0.45
        elif target_share < 0.55:
            score *= 0.68
        elif target_share < 0.72:
            score *= 0.82
        if box_features.get("backgroundWindow"):
            score *= 0.58 + separation_score * 0.42
            if outer_target_share > 0.86 and separation_score < 0.22:
                score *= 0.42
            elif outer_target_share > 0.72 and separation_score < 0.34:
                score *= 0.66
        elif outer_target_share > 0.88 and separation_score < 0.18:
            score *= 0.48
        elif outer_target_share > 0.72 and separation_score < 0.28:
            score *= 0.70
        if box_features.get("shadeComponent") and separation_score > 0.34 and shape_score > 0.55:
            score *= 1.08
        if box_features.get("edgeRectangle") and separation_score > 0.28 and shape_score > 0.55:
            score *= 1.12
        if edge_contrast < 0.12 and outer_target_share > 0.70:
            score *= 0.72
        score *= border_score

        detections.append(
            Detection(
                rank=0,
                score=round(float(min(score, 0.999)), 4),
                bbox=box_features["bbox"],
                area=box_features["area"],
                aspectRatio=round(float(aspect_ratio), 3),
                fillRatio=round(float(fill_ratio), 3),
                colorDominance=round(float(target_share), 3),
                shapeScore=round(float(shape_score), 3),
                solidity=round(float(box_features["solidity"]), 3),
                proposalScore=round(float(box_features["proposalScore"]), 4),
                embeddingScore=round(float(embedding_score), 4),
                vectorScore=round(float(vector_score), 4),
                edgeContrast=round(float(edge_contrast), 3),
                outerTargetShare=round(float(outer_target_share), 3),
                backgroundSeparation=round(float(separation_score), 3),
                dominantColorKey=dominant_color,
                colorShares=box_features["colorShares"],
                ransac=ransac,
                maskPolygon=box_features.get("maskPolygon") or ransac.get("polygon"),
            )
        )

    ranked = non_max_suppression(
        sorted(detections, key=lambda item: item.score, reverse=True),
        iou_threshold=0.30,
        overlap_threshold=0.55,
    )
    ranked = ranked[:max_results]
    return [
        Detection(**{**detection.to_dict(), "rank": index + 1})
        for index, detection in enumerate(ranked)
    ]


def target_color_subproposals(image: np.ndarray, proposal: Proposal, color_key: str) -> list[dict[str, Any]]:
    """Split a broad proposal into target-color components.

    The first proposal stage intentionally over-generates. Before verification we
    constrain boxes to connected components of the requested color so adjacent
    blocks or background color bridges do not become one screen-sized ROI.
    """

    box = proposal.bbox
    x, y, w, h = clip_box(image, box)
    if w <= 0 or h <= 0:
        return []
    crop = image[y : y + h, x : x + w]
    hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (3, 3), 0), cv2.COLOR_BGR2HSV)
    mask = target_component_mask(hsv, color_key)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features: list[dict[str, Any]] = []
    min_area = max(80, int(image.shape[0] * image.shape[1] * 0.00008))
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area:
            continue
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if cw < 8 or ch < 8:
            continue
        global_box = {"x": x + cx, "y": y + cy, "width": cw, "height": ch}
        item = box_features(image, global_box, proposal.proposalScore, mask_polygon=contour_polygon(contour, offset=(x, y)))
        if item["fillRatio"] >= 0.16:
            features.append(item)

    if features:
        return sorted(features, key=lambda item: item["proposalScore"], reverse=True)
    return [box_features(image, proposal.bbox, proposal.proposalScore, mask_polygon=proposal.maskPolygon)] if proposal.colorShares.get(color_key, 0.0) >= 0.08 else []


def target_color_image_components(image: np.ndarray, color_key: str) -> list[dict[str, Any]]:
    """Add target-color components from the whole image.

    The generic proposal stage may miss neutral pieces or split saturated pieces
    oddly. This target-conditioned pass gives the verifier a direct chance to
    rank color components that were not covered by the first proposal set.
    """

    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2HSV)
    mask = target_component_mask(hsv, color_key)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = max(1, image.shape[0] * image.shape[1])
    min_area = max(90, int(image_area * 0.00012))
    max_area = int(image_area * 0.42)
    features: list[dict[str, Any]] = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 8 or h < 8:
            continue
        bbox_area = max(1, w * h)
        fill_ratio = int(np.count_nonzero(mask[y : y + h, x : x + w])) / bbox_area
        if fill_ratio < 0.14:
            continue
        base_score = min(0.999, fill_ratio * 0.36 + min(1.0, area / 2200) * 0.34 + 0.18)
        features.append(
            box_features(
                image,
                {"x": x, "y": y, "width": w, "height": h},
                base_score,
                mask_polygon=contour_polygon(contour),
            )
        )
    return features


def target_color_shade_components(
    image: np.ndarray,
    color_key: str,
    expected_ratio: float,
    expected_bbox_area: float,
) -> list[dict[str, Any]]:
    """Find foreground pieces whose color is merged with a same-color baseplate.

    A large green baseplate and a green 2x4 brick can be one connected color
    component. In that case a color mask alone cannot provide an object contour,
    so we split broad regions by local brightness changes and keep brick-shaped
    subcomponents with real edge separation.
    """

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = target_component_mask(hsv, color_key)
    value = hsv[:, :, 2]
    image_area = max(1, image.shape[0] * image.shape[1])
    min_area = max(120, int(image_area * 0.00025))
    max_bbox_area = min(image_area * 0.22, expected_bbox_area * 6.0)
    min_bbox_area = max(320.0, expected_bbox_area * 0.28)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for region in contours:
        region_area = int(cv2.contourArea(region))
        if region_area < min_area * 5:
            continue
        x, y, w, h = cv2.boundingRect(region)
        if w * h < expected_bbox_area * 1.35:
            continue

        region_mask = mask[y : y + h, x : x + w]
        values = value[y : y + h, x : x + w][region_mask > 0]
        if values.size < min_area:
            continue

        thresholds: list[tuple[str, float]] = []
        for percentile in (32, 42, 52):
            thresholds.append(("dark", float(np.percentile(values, percentile))))
        for percentile in (68, 78):
            thresholds.append(("bright", float(np.percentile(values, percentile))))

        for mode, threshold in thresholds:
            if mode == "dark":
                local = ((region_mask > 0) & (value[y : y + h, x : x + w] <= threshold)).astype(np.uint8) * 255
            else:
                local = ((region_mask > 0) & (value[y : y + h, x : x + w] >= threshold)).astype(np.uint8) * 255
            local = cv2.morphologyEx(local, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
            local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=2)
            sub_contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in sub_contours:
                area = int(cv2.contourArea(contour))
                if area < min_area:
                    continue
                cx, cy, cw, ch = cv2.boundingRect(contour)
                if cw < 12 or ch < 12:
                    continue
                bbox_area = cw * ch
                if bbox_area < min_bbox_area or bbox_area > max_bbox_area:
                    continue
                aspect_ratio = max(cw, ch) / max(1, min(cw, ch))
                shape_score = ratio_score(aspect_ratio, expected_ratio)
                if shape_score < 0.42:
                    continue
                global_box = {"x": x + cx, "y": y + cy, "width": cw, "height": ch}
                target_density = np.count_nonzero(mask[y + cy : y + cy + ch, x + cx : x + cx + cw]) / max(1, bbox_area)
                if target_density < 0.34:
                    continue
                key = (
                    int(round(global_box["x"] / 4) * 4),
                    int(round(global_box["y"] / 4) * 4),
                    int(round(global_box["width"] / 4) * 4),
                    int(round(global_box["height"] / 4) * 4),
                )
                if key in seen:
                    continue
                seen.add(key)
                item = box_features(
                    image,
                    global_box,
                    min(0.92, 0.60 + shape_score * 0.20 + min(1.0, area / 2600) * 0.12),
                    mask_polygon=contour_polygon(contour, offset=(x, y)),
                    metadata={"shadeComponent": True},
                )
                if item["backgroundSeparationScore"] >= 0.16 or shape_score >= 0.74:
                    features.append(item)

    return sorted(
        features,
        key=lambda item: (
            item["backgroundSeparationScore"] * 0.46
            + ratio_score(item["aspectRatio"], expected_ratio) * 0.34
            + item["proposalScore"] * 0.20
        ),
        reverse=True,
    )[:24]


def target_color_edge_rectangles(
    image: np.ndarray,
    color_key: str,
    expected_ratio: float,
    expected_bbox_area: float,
) -> list[dict[str, Any]]:
    """Propose brick boxes from parallel edge pairs inside same-color regions."""

    height, width = image.shape[:2]
    image_area = max(1, height * width)
    short_expected = max(12.0, np.sqrt(max(1.0, expected_bbox_area) / max(1.0, expected_ratio)))
    long_expected = short_expected * expected_ratio
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred_gray, 35, 95)
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2HSV)
    target_mask = target_component_mask(hsv, color_key)
    target_mask = cv2.dilate(target_mask, np.ones((7, 7), dtype=np.uint8), iterations=1)
    edges = cv2.bitwise_and(edges, target_mask)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=42,
        minLineLength=max(40, int(long_expected * 0.34)),
        maxLineGap=14,
    )
    if lines is None:
        return []

    verticals: list[dict[str, float]] = []
    for raw_line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(value) for value in raw_line]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < max(38.0, long_expected * 0.34):
            continue
        angle = abs(float(np.degrees(np.arctan2(dy, dx))))
        angle_from_vertical = abs(90.0 - angle)
        if angle_from_vertical > 14.0:
            continue
        verticals.append(
            {
                "x": (x1 + x2) / 2.0,
                "y0": min(y1, y2),
                "y1": max(y1, y2),
                "length": length,
                "tilt": angle_from_vertical,
            }
        )

    features: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for index, left in enumerate(verticals):
        for right in verticals[index + 1 :]:
            x_left = min(left["x"], right["x"])
            x_right = max(left["x"], right["x"])
            box_width = x_right - x_left
            if box_width < short_expected * 0.45 or box_width > short_expected * 2.45:
                continue
            overlap = min(left["y1"], right["y1"]) - max(left["y0"], right["y0"])
            if overlap / max(1.0, min(left["length"], right["length"])) < 0.34:
                continue
            y_top = min(left["y0"], right["y0"])
            y_bottom = max(left["y1"], right["y1"])
            box_height = y_bottom - y_top
            if box_height < long_expected * 0.48 or box_height > long_expected * 2.35:
                continue
            aspect_ratio = max(box_width, box_height) / max(1.0, min(box_width, box_height))
            shape_score = ratio_score(aspect_ratio, expected_ratio)
            if shape_score < 0.52:
                continue

            pad = 0
            box = {
                "x": int(round(x_left - pad)),
                "y": int(round(y_top - pad)),
                "width": int(round(box_width + pad * 2)),
                "height": int(round(box_height + pad * 2)),
            }
            x, y, w, h = clip_box(image, box)
            if w * h < expected_bbox_area * 0.34 or w * h > min(image_area * 0.18, expected_bbox_area * 6.5):
                continue
            target_density = np.count_nonzero(target_mask[y : y + h, x : x + w]) / max(1, w * h)
            if target_density < 0.38:
                continue
            key = (int(round(x / 6) * 6), int(round(y / 6) * 6), int(round(w / 6) * 6), int(round(h / 6) * 6))
            if key in seen:
                continue
            seen.add(key)
            item = box_features(
                image,
                {"x": x, "y": y, "width": w, "height": h},
                min(0.94, 0.58 + shape_score * 0.24 + min(1.0, (left["length"] + right["length"]) / (long_expected * 2.0)) * 0.12),
                mask_polygon=[[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                metadata={"edgeRectangle": True},
            )
            if item["backgroundSeparationScore"] >= 0.20 and item["outerTargetShare"] >= 0.55:
                features.append(item)

    return sorted(
        features,
        key=lambda item: item["backgroundSeparationScore"] * 0.42
        + ratio_score(item["aspectRatio"], expected_ratio) * 0.38
        + item["proposalScore"] * 0.20,
        reverse=True,
    )[:16]


def target_color_search_windows(
    image: np.ndarray,
    color_key: str,
    expected_ratio: float,
    expected_bbox_area: float,
) -> list[dict[str, Any]]:
    """Generate local target-color windows inside broad connected regions."""

    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2HSV)
    mask = target_component_mask(hsv, color_key)
    image_area = max(1, image.shape[0] * image.shape[1])
    min_pixels = max(240, int(image_area * 0.0005))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_pixels:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < expected_bbox_area * 1.8:
            continue
        for box in local_windows_for_component(image, mask, x, y, w, h, expected_ratio, expected_bbox_area):
            key = (box["x"], box["y"], box["width"], box["height"])
            if key in seen:
                continue
            seen.add(key)
            item = box_features(image, box, 0.66, metadata={"backgroundWindow": True})
            if (
                item["colorShares"].get(color_key, 0.0) >= 0.22
                and item["area"] >= min_pixels
                and (item["backgroundSeparationScore"] >= 0.08 or item["outerTargetShare"] < 0.72)
            ):
                features.append(item)

    return sorted(features, key=lambda item: item["proposalScore"], reverse=True)[:32]


def needs_local_window_search(image: np.ndarray, proposals: list[Proposal]) -> bool:
    if len(proposals) <= 4:
        return True
    image_area = max(1, image.shape[0] * image.shape[1])
    largest = max((proposal.bbox["width"] * proposal.bbox["height"] for proposal in proposals), default=0)
    return largest / image_area >= 0.34


def local_windows_for_component(
    image: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    expected_ratio: float,
    expected_bbox_area: float,
) -> list[dict[str, int]]:
    height, width = image.shape[:2]
    windows: list[dict[str, int]] = []
    for multiplier in (1.5, 2.2, 3.1, 4.0):
        area = max(400.0, expected_bbox_area * multiplier)
        for ratio in (expected_ratio, max(1.0, expected_ratio * 0.72), min(3.2, expected_ratio * 1.28)):
            long_side = int(round(np.sqrt(area * ratio)))
            short_side = int(round(max(12.0, area / max(1, long_side))))
            for ww, hh in ((long_side, short_side), (short_side, long_side)):
                if ww > width or hh > height:
                    continue
                step_x = max(16, ww // 3)
                step_y = max(16, hh // 3)
                x_start = max(0, x - ww // 4)
                x_end = min(width - ww, x + w - (ww * 3) // 4)
                y_start = max(0, y - hh // 4)
                y_end = min(height - hh, y + h - (hh * 3) // 4)
                if x_end < x_start or y_end < y_start:
                    continue
                for yy in range(y_start, y_end + 1, step_y):
                    for xx in range(x_start, x_end + 1, step_x):
                        crop = mask[yy : yy + hh, xx : xx + ww]
                        color_density = np.count_nonzero(crop) / max(1, ww * hh)
                        if color_density >= 0.18:
                            windows.append({"x": xx, "y": yy, "width": ww, "height": hh})
    return windows


def box_features(
    image: np.ndarray,
    box: dict[str, int],
    base_score: float,
    mask_polygon: list[list[int]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    x, y, w, h = clip_box(image, box)
    if w <= 0 or h <= 0:
        return {
            "bbox": {"x": x, "y": y, "width": w, "height": h},
            "area": 0,
            "aspectRatio": 1.0,
            "fillRatio": 0.0,
            "solidity": 0.0,
            "dominantColorKey": "red",
            "colorShares": {key: 0.0 for key in COLORS},
            "proposalScore": 0.0,
            "edgeContrast": 0.0,
            "outerTargetShare": 0.0,
            "backgroundSeparationScore": 0.0,
            "maskPolygon": mask_polygon or [],
            **(metadata or {}),
        }
    hsv = cv2.cvtColor(cv2.GaussianBlur(image[y : y + h, x : x + w], (3, 3), 0), cv2.COLOR_BGR2HSV)
    masks = {key: color_mask(hsv, key) for key in COLORS}
    shares = local_color_shares(masks)
    dominant = max(shares.items(), key=lambda item: item[1])[0] if shares else "red"
    target_pixels = int(np.count_nonzero(masks[dominant])) if dominant in masks else 0
    bbox_area = max(1, w * h)
    fill_ratio = target_pixels / bbox_area
    aspect_ratio = max(w, h) / max(1, min(w, h))
    solidity = min(1.0, target_pixels / bbox_area)
    area_score = min(1.0, target_pixels / 1800)
    separation = foreground_separation_features(image, {"x": x, "y": y, "width": w, "height": h}, dominant)
    proposal_score = min(
        0.999,
        base_score * 0.29
        + fill_ratio * 0.24
        + solidity * 0.17
        + area_score * 0.14
        + separation["backgroundSeparationScore"] * 0.16,
    )
    return {
        "bbox": {"x": x, "y": y, "width": w, "height": h},
        "area": target_pixels,
        "aspectRatio": aspect_ratio,
        "fillRatio": fill_ratio,
        "solidity": solidity,
        "dominantColorKey": dominant,
        "colorShares": {key: round(float(value), 3) for key, value in shares.items()},
        "proposalScore": proposal_score,
        "edgeContrast": separation["edgeContrast"],
        "outerTargetShare": separation["outerTargetShare"],
        "backgroundSeparationScore": separation["backgroundSeparationScore"],
        "maskPolygon": mask_polygon or [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        **(metadata or {}),
    }


def foreground_separation_features(image: np.ndarray, box: dict[str, int], color_key: str) -> dict[str, float]:
    """Measure whether a candidate has a real boundary against its surround."""

    x, y, w, h = clip_box(image, box)
    if w <= 0 or h <= 0:
        return {"edgeContrast": 0.0, "outerTargetShare": 0.0, "backgroundSeparationScore": 0.0}

    height, width = image.shape[:2]
    margin = max(6, min(w, h) // 6)
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(width, x + w + margin)
    y1 = min(height, y + h + margin)
    expanded_w = x1 - x0
    expanded_h = y1 - y0
    if expanded_w <= w and expanded_h <= h:
        return {"edgeContrast": 0.0, "outerTargetShare": 0.0, "backgroundSeparationScore": 0.0}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2HSV)
    target_mask = target_component_mask(hsv, color_key) > 0

    ring = np.ones((expanded_h, expanded_w), dtype=bool)
    local_x = x - x0
    local_y = y - y0
    ring[local_y : local_y + h, local_x : local_x + w] = False
    ring_pixels = int(np.count_nonzero(ring))
    if ring_pixels <= 0:
        outer_target_share = 0.0
        outer_values = np.array([], dtype=np.float32)
    else:
        outer_target_share = float(np.count_nonzero(target_mask[y0:y1, x0:x1][ring]) / ring_pixels)
        outer_values = gray[y0:y1, x0:x1][ring]

    border_width = max(2, min(w, h) // 10)
    inner_border = np.zeros((h, w), dtype=bool)
    inner_border[:border_width, :] = True
    inner_border[-border_width:, :] = True
    inner_border[:, :border_width] = True
    inner_border[:, -border_width:] = True
    inner_values = gray[y : y + h, x : x + w][inner_border]
    luma_delta = abs(float(inner_values.mean()) - float(outer_values.mean())) if inner_values.size and outer_values.size else 0.0

    crop = gray[y0:y1, x0:x1]
    sobel_x = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    border_band = np.zeros_like(gradient, dtype=bool)
    band = 3
    border_band[max(0, local_y - band) : min(expanded_h, local_y + band + 1), local_x : local_x + w] = True
    border_band[
        max(0, local_y + h - band) : min(expanded_h, local_y + h + band + 1),
        local_x : local_x + w,
    ] = True
    border_band[local_y : local_y + h, max(0, local_x - band) : min(expanded_w, local_x + band + 1)] = True
    border_band[
        local_y : local_y + h,
        max(0, local_x + w - band) : min(expanded_w, local_x + w + band + 1),
    ] = True
    border_gradient = float(np.percentile(gradient[border_band], 75)) if np.count_nonzero(border_band) else 0.0

    edge_score = clamp_float((border_gradient - 18.0) / 42.0)
    luma_score = clamp_float(luma_delta / 28.0)
    outer_penalty = clamp_float((outer_target_share - 0.42) / 0.50)
    separation_score = clamp_float(0.16 + edge_score * 0.66 + luma_score * 0.28 - outer_penalty * 0.34)
    return {
        "edgeContrast": round(float(edge_score), 4),
        "outerTargetShare": round(float(outer_target_share), 4),
        "backgroundSeparationScore": round(float(separation_score), 4),
    }


def clamp_float(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return float(max(minimum, min(maximum, value)))


def local_color_shares(masks: dict[str, np.ndarray]) -> dict[str, float]:
    counts = {}
    total = 0
    for key, mask in masks.items():
        count = int(np.count_nonzero(mask))
        if key in {"black", "white", "gray"}:
            count = int(count * 0.45)
        counts[key] = count
        total += count
    if total <= 0:
        return {key: 0.0 for key in counts}
    return {key: count / total for key, count in counts.items()}


def target_component_mask(hsv: np.ndarray, color_key: str) -> np.ndarray:
    """Use a stricter mask for target-conditioned box extraction.

    The broad catalog color ranges are useful for scoring, but for proposals a
    slightly stricter saturated mask keeps red bricks from merging with orange
    shadows and yellow highlights in cluttered pile photos.
    """

    if color_key == "red":
        lower_red = cv2.inRange(hsv, np.array((0, 115, 55)), np.array((8, 255, 255)))
        upper_red = cv2.inRange(hsv, np.array((174, 115, 55)), np.array((180, 255, 255)))
        return cv2.bitwise_or(lower_red, upper_red)
    if color_key == "orange":
        return cv2.inRange(hsv, np.array((10, 105, 60)), np.array((22, 255, 255)))
    if color_key == "yellow":
        return cv2.inRange(hsv, np.array((24, 95, 80)), np.array((37, 255, 255)))
    return color_mask(hsv, color_key)


def clip_box(image: np.ndarray, box: dict[str, int]) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    x = max(0, int(box["x"]))
    y = max(0, int(box["y"]))
    x1 = min(width, x + max(0, int(box["width"])))
    y1 = min(height, y + max(0, int(box["height"])))
    return x, y, max(0, x1 - x), max(0, y1 - y)


def verify_geometry(image: np.ndarray, proposal: Any, expected_ratio: float) -> dict[str, Any]:
    box = proposal["bbox"] if isinstance(proposal, dict) else proposal.bbox
    x, y, w, h = box["x"], box["y"], box["width"], box["height"]
    height, width = image.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    crop = image[y0:y1, x0:x1]
    method = "corner-ransac-fallback"
    inliers = 0

    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=160)
        keypoints, _ = orb.detectAndCompute(gray, None)
        inliers = len(keypoints or [])
        method = "orb-keypoints+corner-ransac"

    src = np.float32([[0, 0], [expected_ratio, 0], [expected_ratio, 1], [0, 1]])
    dst = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    corner_inliers = int(mask.sum()) if mask is not None else 0
    observed_ratio = proposal["aspectRatio"] if isinstance(proposal, dict) else proposal.aspectRatio
    aspect_score = ratio_score(observed_ratio, expected_ratio)
    keypoint_score = min(1.0, inliers / 24.0)
    score = min(1.0, aspect_score * 0.72 + (corner_inliers / 4.0) * 0.18 + keypoint_score * 0.10)

    return {
        "method": method,
        "score": round(float(score), 4),
        "inliers": int(max(inliers, corner_inliers)),
        "cornerInliers": corner_inliers,
        "homography": homography.round(4).tolist() if homography is not None else None,
        "polygon": [[int(x0), int(y0)], [int(x1), int(y0)], [int(x1), int(y1)], [int(x0), int(y1)]],
    }


def ratio_score(observed: float, expected: float) -> float:
    expected = max(1.0, expected)
    observed = max(1.0, observed)
    score = np.exp(-abs(np.log(observed / expected)) * 1.35)
    return float(max(0.0, min(1.0, score)))


def size_floor_score(observed_area: float, expected_area: float) -> float:
    expected_area = max(1.0, expected_area)
    observed_area = max(1.0, observed_area)
    score = observed_area / (expected_area * 0.82)
    return float(max(0.0, min(1.0, score)))


def border_containment_score(image: np.ndarray, box: dict[str, int]) -> float:
    """Prefer fully visible bricks over clipped edge fragments."""

    height, width = image.shape[:2]
    x, y, w, h = clip_box(image, box)
    if w <= 0 or h <= 0:
        return 0.0
    touches_left = x <= 1
    touches_top = y <= 1
    touches_right = x + w >= width - 1
    touches_bottom = y + h >= height - 1
    touches = sum([touches_left, touches_top, touches_right, touches_bottom])
    if touches >= 2:
        return 0.58
    if touches == 1:
        edge_span = 0.0
        if touches_left or touches_right:
            edge_span = max(edge_span, h / max(1, height))
        if touches_top or touches_bottom:
            edge_span = max(edge_span, w / max(1, width))
        return 0.78 if edge_span > 0.22 else 0.88
    return 1.0


def non_max_suppression(
    detections: list[Detection],
    iou_threshold: float,
    overlap_threshold: float,
) -> list[Detection]:
    kept: list[Detection] = []
    for detection in detections:
        if all(
            box_iou(detection.bbox, existing.bbox) < iou_threshold
            and box_overlap_ratio(detection.bbox, existing.bbox) < overlap_threshold
            and not same_object_cluster(detection.bbox, existing.bbox)
            for existing in kept
        ):
            kept.append(detection)
    return kept


def box_iou(first: dict[str, int], second: dict[str, int]) -> float:
    ax0, ay0 = first["x"], first["y"]
    ax1, ay1 = ax0 + first["width"], ay0 + first["height"]
    bx0, by0 = second["x"], second["y"]
    bx1, by1 = bx0 + second["width"], by0 + second["height"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    intersection = iw * ih
    union = first["width"] * first["height"] + second["width"] * second["height"] - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def box_overlap_ratio(first: dict[str, int], second: dict[str, int]) -> float:
    ax0, ay0 = first["x"], first["y"]
    ax1, ay1 = ax0 + first["width"], ay0 + first["height"]
    bx0, by0 = second["x"], second["y"]
    bx1, by1 = bx0 + second["width"], by0 + second["height"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    intersection = iw * ih
    smaller = min(first["width"] * first["height"], second["width"] * second["height"])
    if smaller <= 0:
        return 0.0
    return intersection / smaller


def same_object_cluster(first: dict[str, int], second: dict[str, int]) -> bool:
    overlap = box_overlap_ratio(first, second)
    if overlap < 0.25:
        return False
    first_cx = first["x"] + first["width"] / 2
    first_cy = first["y"] + first["height"] / 2
    second_cx = second["x"] + second["width"] / 2
    second_cy = second["y"] + second["height"] / 2
    distance = float(np.hypot(first_cx - second_cx, first_cy - second_cy))
    first_diag = float(np.hypot(first["width"], first["height"]))
    second_diag = float(np.hypot(second["width"], second["height"]))
    reference = max(1.0, (first_diag + second_diag) / 2)
    return distance / reference < 0.38


__all__ = [
    "COLORS",
    "Detection",
    "color_mask",
    "decode_image",
    "find_target_detections",
    "ratio_score",
    "search_image_bytes",
]
