"""Small offline LEGO part catalog used by the web prototype.

The production system should replace this with Rebrickable CSV/API ingestion plus
an HNSW/Qdrant vector index. Keeping this catalog dependency-free makes the MVP
usable without an API key.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


COLORS: dict[str, dict[str, str]] = {
    "red": {"label": "빨간색", "css": "#e53935", "emoji": "🔴"},
    "blue": {"label": "파란색", "css": "#1e88e5", "emoji": "🔵"},
    "yellow": {"label": "노란색", "css": "#fdd835", "emoji": "🟡"},
    "green": {"label": "초록색", "css": "#43a047", "emoji": "🟢"},
    "orange": {"label": "주황색", "css": "#fb8c00", "emoji": "🟠"},
    "purple": {"label": "보라색", "css": "#8e24aa", "emoji": "🟣"},
    "black": {"label": "검은색", "css": "#151515", "emoji": "⚫"},
    "white": {"label": "흰색", "css": "#f4f4f4", "emoji": "⚪"},
    "gray": {"label": "회색", "css": "#8a8f98", "emoji": "⚙️"},
}

PART_NUMBERS = {
    "brick-2x4": "3001",
    "brick-2x3": "3002",
    "brick-2x2": "3003",
    "brick-1x2": "3004",
    "brick-1x4": "3010",
    "plate-1x2": "3023",
    "plate-2x2": "3022",
    "plate-1x4": "3710",
    "tile-2x2": "3068",
    "tile-1x2": "3069",
    "head-1x1": "3626",
}

CATEGORY_LABELS = {"brick": "기본 브릭", "plate": "플레이트", "tile": "타일", "head": "미니피겨 헤드"}


@dataclass(frozen=True)
class PartTarget:
    id: str
    name: str
    colorKey: str
    colorLabel: str
    colorCss: str
    colorEmoji: str
    width: int
    length: int
    category: str
    categoryLabel: str
    partNum: str
    quantity: int = 1
    source: str = "server"
    height: str = "standard"
    shape: str = "rectangular"
    attributes: list[str] = field(default_factory=list)
    negativeTerms: list[str] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_target(
    color_key: str = "red",
    width: int = 2,
    length: int = 3,
    category: str = "brick",
    quantity: int = 1,
    source: str = "server",
    height: str = "standard",
    shape: str = "rectangular",
    attributes: list[str] | None = None,
    negative_terms: list[str] | None = None,
    confidence: dict[str, float] | None = None,
) -> PartTarget:
    color = COLORS.get(color_key, COLORS["red"])
    category = category if category in CATEGORY_LABELS else "brick"
    key = f"{category}-{width}x{length}"
    alt_key = f"{category}-{length}x{width}"
    part_num = PART_NUMBERS.get(key) or PART_NUMBERS.get(alt_key) or "unknown"
    category_label = CATEGORY_LABELS[category]
    if category == "head":
        name = f"{color['label']} 사람 얼굴 {category_label}" if "face_print" in (attributes or []) else f"{color['label']} {category_label}"
    else:
        name = f"{color['label']} {width}x{length} {category_label}"
    return PartTarget(
        id=f"{color_key}-{category}-{width}x{length}",
        name=name,
        colorKey=color_key,
        colorLabel=color["label"],
        colorCss=color["css"],
        colorEmoji=color["emoji"],
        width=width,
        length=length,
        category=category,
        categoryLabel=category_label,
        partNum=part_num,
        quantity=quantity,
        source=source,
        height=height,
        shape=shape,
        attributes=attributes or [],
        negativeTerms=negative_terms or [],
        confidence=confidence or {},
    )


COMMON_PARTS: list[PartTarget] = [
    make_target(color, width, length, category, source="catalog")
    for color in COLORS
    for category, dims in {
        "brick": [(2, 4), (2, 3), (2, 2), (1, 4), (1, 2)],
        "plate": [(1, 2), (2, 2), (1, 4)],
        "tile": [(2, 2), (1, 2)],
        "head": [(1, 1)],
    }.items()
    for width, length in dims
]

SAMPLE_SET = {
    "set_num": "75257-1",
    "name": "샘플 세트: 밀레니엄 팔콘 모드",
    "total_parts": 42,
    "parts": [
        make_target("red", 2, 4, "brick", 12, "sample-set").to_dict(),
        make_target("blue", 1, 4, "brick", 6, "sample-set").to_dict(),
        make_target("yellow", 2, 2, "plate", 8, "sample-set").to_dict(),
        make_target("green", 1, 2, "plate", 10, "sample-set").to_dict(),
        make_target("black", 2, 2, "tile", 6, "sample-set").to_dict(),
    ],
}
