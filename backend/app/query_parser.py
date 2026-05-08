"""Natural-language query parser for BrickFinder MVP.

This is intentionally deterministic. In production, keep this parser as a safe
fallback and add an LLM/CLIP text encoder for ambiguous instructions.
"""

from __future__ import annotations

import re

from .catalog import make_target, PartTarget

COLOR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("red", re.compile(r"빨간|빨강|레드|red", re.I)),
    ("blue", re.compile(r"파란|파랑|블루|blue", re.I)),
    ("yellow", re.compile(r"노란|노랑|옐로|yellow", re.I)),
    ("green", re.compile(r"초록|녹색|그린|green", re.I)),
    ("orange", re.compile(r"주황|오렌지|orange", re.I)),
    ("purple", re.compile(r"보라|퍼플|purple", re.I)),
    ("black", re.compile(r"검은|검정|블랙|black", re.I)),
    ("white", re.compile(r"흰|하얀|화이트|white", re.I)),
    ("gray", re.compile(r"회색|그레이|gray|grey", re.I)),
]


def parse_query(text: str, source: str = "server-text") -> PartTarget:
    raw = text or ""
    color_key = "red"
    for key, pattern in COLOR_PATTERNS:
        if pattern.search(raw):
            color_key = key
            break

    dim_match = re.search(r"(\d+)\s*[x×*]\s*(\d+)", raw, re.I) or re.search(
        r"(\d+)\s*by\s*(\d+)", raw, re.I
    )
    width, length = (2, 3)
    if dim_match:
        width, length = int(dim_match.group(1)), int(dim_match.group(2))

    category = "brick"
    if re.search(r"플레이트|plate", raw, re.I):
        category = "plate"
    if re.search(r"타일|tile", raw, re.I):
        category = "tile"
    if re.search(r"브릭|brick", raw, re.I):
        category = "brick"

    return make_target(color_key, width, length, category, source=source)
