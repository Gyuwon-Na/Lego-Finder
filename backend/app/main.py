from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - optional in unit-test environments
    httpx = None  # type: ignore[assignment]
try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover - lets pure helpers import without API deps
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

from .catalog import COMMON_PARTS, COLORS, SAMPLE_SET, make_target
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
    candidates = rank_candidates(target.colorKey, target.width, target.length, target.category, target=target)
    return {
        "target": target.to_dict(),
        "candidates": candidates,
        "pipeline": {
            "query_encoder": "structured deterministic parser with CLIP-ready target spec",
            "vector_index": "offline mock; replace with hnswlib or Qdrant",
            "vision": "browser HSV ROI now; replace with ONNX/MediaPipe detector",
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


@app.get("/api/parts/search")
def search_parts(q: str = "") -> Dict[str, Any]:
    q_lower = q.lower().strip()
    results = []
    for part in COMMON_PARTS:
        text = f"{part.name} {part.partNum} {part.colorLabel} {part.categoryLabel}".lower()
        if not q_lower or q_lower in text:
            results.append(part.to_dict())
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
) -> List[Dict[str, Any]]:
    requested_dims = sorted((width, length))
    negative_terms = set(getattr(target, "negativeTerms", []) or [])
    target_height = getattr(target, "height", "standard")
    target_shape = getattr(target, "shape", "rectangular")

    def score(part: Any) -> float:
        part_dims = sorted((part.width, part.length))
        dimension_score = 1.0 if part_dims == requested_dims else 0.0
        value = 0.0
        value += 0.42 if part.colorKey == color_key else 0.0
        value += 0.24 if part.category == category else 0.0
        value += 0.34 * dimension_score
        value += 0.06 if getattr(part, "height", "standard") == target_height else 0.0
        value += 0.04 if getattr(part, "shape", "rectangular") == target_shape else 0.0
        if part.colorKey in negative_terms or part.category in negative_terms:
            value *= 0.18
        return value

    ranked = sorted(COMMON_PARTS, key=score, reverse=True)
    return [{**part.to_dict(), "score": round(score(part), 3)} for part in ranked[:8]]


def normalize_rebrickable_part(item: Dict[str, Any]) -> Dict[str, Any]:
    part_info = item.get("part") or {}
    color_info = item.get("color") or {}
    name = f"{part_info.get('name', '')} {color_info.get('name', '')}"
    color_key = infer_color_key(color_info.get("name") or name)
    width, length = infer_dimensions(name)
    category = infer_category(name)
    target = make_target(color_key, width, length, category, item.get("quantity", 1), "rebrickable")
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
    if "plate" in text or "플레이트" in text:
        return "plate"
    if "tile" in text or "타일" in text:
        return "tile"
    return "brick"


def infer_dimensions(value: str) -> Tuple[int, int]:
    import re

    match = re.search(r"(\d+)\s*x\s*(\d+)", value or "", re.I)
    if not match:
        return (2, 3)
    return int(match.group(1)), int(match.group(2))
