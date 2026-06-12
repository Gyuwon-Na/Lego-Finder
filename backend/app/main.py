from __future__ import annotations

import os
import json
import math
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - optional in unit-test environments
    httpx = None  # type: ignore[assignment]
try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover - lets pure helpers import without API deps
    StaticFiles = None  # type: ignore[assignment]

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class JSONResponse(dict):
        pass

    class UploadFile:
        content_type: Optional[str] = None

    class BaseModel:
        pass

    class CORSMiddleware:
        pass

    def Field(default: Any, **_: Any) -> Any:
        return default

    def File(default: Any = None, **_: Any) -> Any:
        return default

    def Form(default: Any = None, **_: Any) -> Any:
        return default

    class FastAPI:
        def __init__(self, *_: Any, **__: Any):
            pass

        def add_middleware(self, *_: Any, **__: Any) -> None:
            return None

        def get(self, *_: Any, **__: Any) -> Any:
            return lambda func: func

        def post(self, *_: Any, **__: Any) -> Any:
            return lambda func: func

from .catalog import COMMON_PARTS, COLORS, SAMPLE_SET, CATEGORY_LABELS, make_target
from .query_parser import parse_query
from .vision import search_image_bytes

app = FastAPI(
    title="BrickFinder Prototype API",
    description="Backend facade for the LEGO real-time search web prototype.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextQuery(BaseModel):
    text: str = Field(..., examples=["빨간색 2x3 기본 브릭"])


class MatchResponse(BaseModel):
    target: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    pipeline: Dict[str, Any]


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/query/text", response_model=MatchResponse)
def query_text(payload: TextQuery) -> Dict[str, Any]:
    target = parse_query(payload.text)
    candidates = rank_candidates(target.colorKey, target.width, target.length, target.category, target=target, query_text=payload.text)
    return {
        "target": target.to_dict(),
        "candidates": candidates,
        "pipeline": {
            "query_encoder": "expanded deterministic TargetSpec parser with Korean/English aliases",
            "part_index": f"COMMON_PARTS + Rebrickable JSON payloads ({len(all_part_payloads())} candidates)",
            "candidate_ranker": "color/category/dimension/shape/attribute/text/popularity weighted reranker",
            "vision": "browser HSV ROI + optional backend YOLO segmentation; replace/add CLIP re-ranker for open-vocabulary recognition",
        },
    }


@app.post("/api/query/image")
async def query_image(file: UploadFile = File(...)) -> JSONResponse:
    # Placeholder for manual-image query. The browser performs dominant color
    # estimation already; this endpoint is where a CLIP image encoder would run.
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image file required")
    return JSONResponse(
        {
            "target": make_target("red", 2, 3, "brick", source="server-image-placeholder").to_dict(),
            "note": "Image endpoint placeholder. Add CLIP/MobileCLIP image embedding here.",
        }
    )


@app.post("/api/search/image")
async def search_image(
    text: str = Form(..., examples=["2x4 red brick"]),
    file: UploadFile = File(...),
    max_results: int = Form(8),
) -> Dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image file required")
    try:
        return search_image_bytes(await file.read(), text, max_results=max(1, min(max_results, 20)))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/search/live-frame")
async def search_live_frame(
    text: str = Form(..., examples=["2x4 red brick"]),
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image file required")
    try:
        result = search_image_bytes(await file.read(), text, max_results=1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["mode"] = "live-frame"
    return result


@app.get("/api/parts/search")
def search_parts(q: str = "") -> Dict[str, Any]:
    q_lower = q.lower().strip()
    results = []
    for part in all_part_payloads():
        text = searchable_text(part)
        if not q_lower or q_lower in text:
            results.append(part)
    return {"results": results[:50]}


@app.get("/api/rebrickable/set/{set_num}")
async def rebrickable_set(set_num: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    key = api_key or os.getenv("REBRICKABLE_API_KEY")
    if not key:
        return {**SAMPLE_SET, "set_num": set_num, "name": f"{set_num} 샘플/오프라인 모드"}
    if httpx is None:
        raise HTTPException(status_code=503, detail="httpx is required for Rebrickable API proxy")

    headers = {"Authorization": f"key {key}"}
    base = f"https://rebrickable.com/api/v3/lego/sets/{set_num}/parts/"
    params = {"page_size": "1000", "inc_color_details": "1", "inc_part_details": "1"}
    async with httpx.AsyncClient(timeout=12.0) as client:
        try:
            res = await client.get(base, headers=headers, params=params)
            res.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Rebrickable request failed: {exc}") from exc
    payload = res.json()
    parts = [normalize_rebrickable_part(item) for item in payload.get("results", [])]
    return {
        "set_num": set_num,
        "name": f"Rebrickable {set_num}",
        "total_parts": sum(p.get("quantity", 1) for p in parts),
        "parts": parts[:200],
        "source": "rebrickable",
    }


def rank_candidates(
    color_key: str,
    width: int,
    length: int,
    category: str,
    target: Optional[Any] = None,
    query_text: str = "",
) -> List[Dict[str, Any]]:
    requested_dims = sorted((width, length))
    negative_terms = set(getattr(target, "negativeTerms", []) or [])
    target_height = getattr(target, "height", "standard")
    target_shape = getattr(target, "shape", "rectangular")
    target_attributes = set(getattr(target, "attributes", []) or [])
    dimension_known = bool(getattr(target, "dimensionKnown", True))
    query = (query_text or "").lower()
    exact_part_num = find_part_number(query)

    def score(part: dict[str, Any]) -> float:
        part_dims = sorted((int(part.get("width", 1)), int(part.get("length", 1))))
        dimension_score = dimension_match_score(part_dims, requested_dims, dimension_known)
        category_score = category_match_score(category, str(part.get("category", "")), str(part.get("externalName") or part.get("name") or ""))
        text_score = lexical_match_score(query, part)
        attribute_score = attribute_match_score(target_attributes, part)
        shape_score = shape_match_score(target_shape, part)
        color_score = 1.0 if part.get("colorKey") == color_key else 0.15 if not color_key else 0.0
        popularity_score = popularity_match_score(part)
        value = (
            color_score * 0.24
            + category_score * 0.25
            + dimension_score * (0.22 if dimension_known else 0.08)
            + shape_score * 0.08
            + attribute_score * 0.09
            + text_score * 0.10
            + popularity_score * 0.04
        )
        if exact_part_num and str(part.get("partNum", "")).lower() == exact_part_num:
            value += 0.55
        if part.get("colorKey") in negative_terms or part.get("category") in negative_terms:
            value *= 0.18
        return value

    ranked = sorted(dedupe_payloads(all_part_payloads()), key=score, reverse=True)
    return [{**part, "score": round(score(part), 3)} for part in ranked[:12]]


@lru_cache(maxsize=1)
def all_part_payloads() -> tuple[dict[str, Any], ...]:
    payloads: list[dict[str, Any]] = [part.to_dict() for part in COMMON_PARTS]
    path = Path(__file__).resolve().parents[2] / "data" / "rebrickable" / "parts_index.json"
    if path.exists():
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(records, dict):
                records = records.get("records", [])
            for record in records:
                payload = record.get("payload") if isinstance(record, dict) else None
                if isinstance(payload, dict):
                    payloads.append(normalize_payload_shape(payload))
        except Exception:
            pass
    return tuple(dedupe_payloads(payloads))


def normalize_payload_shape(payload: dict[str, Any]) -> dict[str, Any]:
    category = payload.get("category") if payload.get("category") in CATEGORY_LABELS else infer_category(payload.get("externalName") or payload.get("name") or "")
    category_label = CATEGORY_LABELS.get(category, CATEGORY_LABELS["special"])
    return {
        **payload,
        "category": category,
        "categoryLabel": payload.get("categoryLabel") or category_label,
        "width": int(payload.get("width") or 1),
        "length": int(payload.get("length") or 1),
        "shape": payload.get("shape") or infer_shape(payload.get("externalName") or payload.get("name") or ""),
        "attributes": payload.get("attributes") or infer_attributes(payload.get("externalName") or payload.get("name") or ""),
        "dimensionKnown": bool(payload.get("dimensionKnown", True)),
    }


def dedupe_payloads(payloads: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        key = f"{payload.get('partNum')}:{payload.get('colorKey')}:{payload.get('category')}:{payload.get('rebrickableColorId', '')}"
        existing = best.get(key)
        if existing is None:
            best[key] = dict(payload)
            continue
        if popularity_match_score(payload) > popularity_match_score(existing):
            best[key] = dict(payload)
    return list(best.values())


def searchable_text(part: dict[str, Any]) -> str:
    fields = [
        part.get("name"),
        part.get("externalName"),
        part.get("partNum"),
        part.get("colorLabel"),
        part.get("colorKey"),
        part.get("categoryLabel"),
        part.get("category"),
        part.get("rebrickableColorName"),
        " ".join(part.get("attributes") or []),
    ]
    return " ".join(str(field or "") for field in fields).lower()


def find_part_number(query: str) -> str:
    import re

    match = re.search(r"\b([a-z]?\d{3,6}[a-z]?)\b", query, re.I)
    return match.group(1).lower() if match else ""


def dimension_match_score(part_dims: list[int], requested_dims: list[int], dimension_known: bool) -> float:
    if part_dims == requested_dims:
        return 1.0
    if not dimension_known:
        return 0.55
    part_area = max(1, part_dims[0] * part_dims[1])
    requested_area = max(1, requested_dims[0] * requested_dims[1])
    area_score = math.exp(-abs(math.log(part_area / requested_area)) * 1.2)
    part_ratio = max(part_dims) / max(1, min(part_dims))
    requested_ratio = max(requested_dims) / max(1, min(requested_dims))
    ratio_score = math.exp(-abs(math.log(part_ratio / requested_ratio)) * 1.4)
    return float(max(0.0, min(1.0, area_score * 0.42 + ratio_score * 0.58)))


def category_match_score(target_category: str, part_category: str, external_name: str) -> float:
    if target_category == part_category:
        return 1.0
    text = external_name.lower()
    aliases = {
        "technic_pin": ["technic pin", "pin", "peg"],
        "axle": ["axle"],
        "wheel": ["wheel"],
        "tire": ["tire", "tyre"],
        "hinge": ["hinge"],
        "clip": ["clip", "claw"],
        "bar": ["bar", "rod", "wand"],
        "slope": ["slope"],
        "wedge": ["wedge"],
        "cone": ["cone"],
        "window": ["window", "windscreen", "windshield"],
        "door": ["door"],
        "bracket": ["bracket"],
        "panel": ["panel"],
        "head": ["head", "minifig"],
        "minifigure": ["minifig", "torso", "leg", "body"],
    }
    if any(alias in text for alias in aliases.get(target_category, [])):
        return 0.9
    if target_category in {"brick", "plate", "tile"} and target_category in text:
        return 0.8
    if part_category == "special":
        return 0.35
    return 0.08


def lexical_match_score(query: str, part: dict[str, Any]) -> float:
    if not query:
        return 0.0
    text = searchable_text(part)
    tokens = [token for token in query.replace("×", "x").replace("*", "x").split() if len(token) >= 2]
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in text)
    return min(1.0, hits / max(2, len(tokens)))


def attribute_match_score(target_attributes: set[str], part: dict[str, Any]) -> float:
    if not target_attributes:
        return 0.45
    part_attributes = set(part.get("attributes") or [])
    text = searchable_text(part)
    hits = 0
    for attribute in target_attributes:
        if attribute in part_attributes or attribute.replace("_", " ") in text:
            hits += 1
        elif attribute == "face_print" and any(word in text for word in ["face", "head", "smile"]):
            hits += 1
        elif attribute == "technic" and "technic" in text:
            hits += 1
    return hits / max(1, len(target_attributes))


def shape_match_score(target_shape: str, part: dict[str, Any]) -> float:
    part_shape = str(part.get("shape") or infer_shape(part.get("externalName") or part.get("name") or ""))
    if target_shape == part_shape:
        return 1.0
    if target_shape == "rectangular":
        return 0.45
    text = searchable_text(part)
    if target_shape in text:
        return 0.75
    return 0.18


def popularity_match_score(part: dict[str, Any]) -> float:
    count = float(part.get("numSetParts") or part.get("numSets") or 0)
    return min(1.0, math.log1p(count) / 9.0)


def normalize_rebrickable_part(item: Dict[str, Any]) -> Dict[str, Any]:
    part_info = item.get("part") or {}
    color_info = item.get("color") or {}
    name = f"{part_info.get('name', '')} {color_info.get('name', '')}"
    color_key = infer_color_key(color_info.get("name") or name)
    width, length = infer_dimensions(name)
    category = infer_category(name)
    target = make_target(
        color_key,
        width,
        length,
        category,
        item.get("quantity", 1),
        "rebrickable",
        shape=infer_shape(name),
        attributes=infer_attributes(name),
        dimension_known=bool(has_explicit_dimensions(name)),
    )
    data = target.to_dict()
    data["partNum"] = part_info.get("part_num") or data["partNum"]
    data["externalName"] = part_info.get("name") or data["name"]
    data["partImgUrl"] = item.get("part_img_url") or part_info.get("part_img_url")
    return data


def infer_color_key(value: str) -> str:
    text = (value or "").lower()
    if any(token in text for token in ["brown", "tan", "sand", "nougat", "copper", "bronze"]):
        return "gray"
    if any(token in text for token in ["silver", "metal", "chrome", "pearl", "trans-clear", "no color", "any color"]):
        return "gray"
    if "pink" in text or "magenta" in text:
        return "purple"
    if "turquoise" in text or "aqua" in text:
        return "blue"
    if "red" in text or "빨" in text:
        return "red"
    if "blue" in text or "파" in text:
        return "blue"
    if "yellow" in text or "노" in text:
        return "yellow"
    if "green" in text or "초" in text or "녹" in text:
        return "green"
    if "orange" in text or "주황" in text:
        return "orange"
    if "purple" in text or "보라" in text:
        return "purple"
    if "black" in text or "검" in text:
        return "black"
    if "white" in text or "흰" in text:
        return "white"
    if "gray" in text or "grey" in text or "회" in text:
        return "gray"
    return "gray"


def infer_category(value: str) -> str:
    text = (value or "").lower()
    if any(token in text for token in ["minifig head", "minifigure head", "head", "머리", "헤드"]):
        return "head"
    if any(token in text for token in ["technic pin", "pin connector", "connector peg", "friction pin"]):
        return "technic_pin"
    if "axle" in text:
        return "axle"
    if "tire" in text or "tyre" in text:
        return "tire"
    if "wheel" in text:
        return "wheel"
    if "hinge" in text:
        return "hinge"
    if "clip" in text or "claw" in text:
        return "clip"
    if any(token in text for token in ["bar ", " bar", "rod", "wand"]):
        return "bar"
    if "window" in text or "windscreen" in text or "windshield" in text:
        return "window"
    if "door" in text:
        return "door"
    if "bracket" in text:
        return "bracket"
    if "panel" in text:
        return "panel"
    if "cone" in text:
        return "cone"
    if "wedge" in text:
        return "wedge"
    if "slope" in text:
        return "slope"
    if "plate" in text or "플레이트" in text:
        return "plate"
    if "tile" in text or "타일" in text:
        return "tile"
    if "brick" in text or "브릭" in text or "block" in text:
        return "brick"
    if any(token in text for token in ["minifig", "torso", "leg", "arm", "body"]):
        return "minifigure"
    return "brick"


def infer_dimensions(value: str) -> Tuple[int, int]:
    import re

    match = re.search(r"(\d+)\s*x\s*(\d+)", value or "", re.I)
    if not match:
        return (2, 3)
    return int(match.group(1)), int(match.group(2))


def has_explicit_dimensions(value: str) -> bool:
    import re

    return bool(re.search(r"\d+\s*x\s*\d+", value or "", re.I))


def infer_shape(value: str) -> str:
    text = (value or "").lower()
    if any(token in text for token in ["slope", "sloped", "wedge"]):
        return "slope"
    if any(token in text for token in ["round", "cylinder", "cone", "wheel", "tire", "tyre", "head", "pin"]):
        return "round"
    if "corner" in text:
        return "corner"
    return "rectangular"


def infer_attributes(value: str) -> list[str]:
    text = (value or "").lower()
    attributes: list[str] = []
    checks = [
        ("face_print", ["face", "smile", "head decorated"]),
        ("printed", ["print", "pattern", "decorated"]),
        ("transparent", ["trans-", "transparent", "clear"]),
        ("technic", ["technic"]),
        ("hinge", ["hinge"]),
        ("wheel", ["wheel", "tire", "tyre"]),
        ("studs", ["stud", "jumper"]),
    ]
    for key, aliases in checks:
        if any(alias in text for alias in aliases):
            attributes.append(key)
    return attributes


def mount_frontend() -> None:
    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    if StaticFiles is not None and frontend_dir.exists() and hasattr(app, "mount"):
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


mount_frontend()
