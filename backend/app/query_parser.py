"""Natural-language query parser for BrickFinder MVP.

This is intentionally deterministic. In production, keep this parser as a safe
fallback and add an LLM/CLIP text encoder for ambiguous instructions.
"""

from __future__ import annotations

import re

from .catalog import make_target, PartTarget

COLOR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("red", re.compile(r"빨간|빨강|빨갛|레드|red|dark red|reddish", re.I)),
    ("blue", re.compile(r"파란|파랑|파랗|블루|blue|navy|azure", re.I)),
    ("yellow", re.compile(r"노란|노랑|노랗|옐로|yellow", re.I)),
    ("green", re.compile(r"초록|녹색|그린|green", re.I)),
    ("orange", re.compile(r"주황|오렌지|orange", re.I)),
    ("purple", re.compile(r"보라|퍼플|purple|violet|lavender", re.I)),
    ("black", re.compile(r"검은|검정|까만|블랙|black", re.I)),
    ("white", re.compile(r"흰|하얀|화이트|white", re.I)),
    ("gray", re.compile(r"회색|회색빛|그레이|gray|grey|silver|metallic", re.I)),
]

NUMBER_WORDS = {
    "한": 1,
    "하나": 1,
    "일": 1,
    "두": 2,
    "둘": 2,
    "이": 2,
    "세": 3,
    "셋": 3,
    "삼": 3,
    "네": 4,
    "넷": 4,
    "사": 4,
    "다섯": 5,
    "오": 5,
    "여섯": 6,
    "육": 6,
    "일곱": 7,
    "칠": 7,
    "여덟": 8,
    "팔": 8,
}

CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("plate", re.compile(r"플레이트|납작한?|얇은|plate|flat|thin", re.I)),
    ("tile", re.compile(r"타일|스터드\s*없|민자|매끈|tile|smooth|studless", re.I)),
    ("brick", re.compile(r"브릭|블럭|블록|기본\s*블록|brick|block", re.I)),
]

SHAPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("slope", re.compile(r"경사|사선|비스듬|슬로프|slope|sloped|wedge", re.I)),
    ("round", re.compile(r"둥근|원형|라운드|round|cylinder|cylindrical", re.I)),
    ("corner", re.compile(r"코너|모서리|corner", re.I)),
    ("rectangular", re.compile(r"직사각|사각|rectangular|rectangle", re.I)),
]

ATTRIBUTE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("transparent", re.compile(r"투명|클리어|transparent|trans[- ]?clear", re.I)),
    ("printed", re.compile(r"프린트|무늬|패턴|printed|pattern", re.I)),
    ("hinge", re.compile(r"힌지|관절|hinge", re.I)),
    ("wheel", re.compile(r"바퀴|휠|wheel|tire|tyre", re.I)),
    ("studs", re.compile(r"스터드|studs?", re.I)),
]


def parse_query(text: str, source: str = "server-text") -> PartTarget:
    raw = text or ""
    color_key = "red"
    matched_color = False
    for key, pattern in COLOR_PATTERNS:
        if pattern.search(raw):
            color_key = key
            matched_color = True
            break

    width, length, matched_dims = parse_dimensions(raw)

    category = "brick"
    matched_category = False
    for key, pattern in CATEGORY_PATTERNS:
        if pattern.search(raw):
            category = key
            matched_category = True
            break

    height = infer_height(raw, category)
    shape = "rectangular"
    for key, pattern in SHAPE_PATTERNS:
        if pattern.search(raw):
            shape = key
            break

    attributes = [key for key, pattern in ATTRIBUTE_PATTERNS if pattern.search(raw)]
    negative_terms = parse_negative_terms(raw)
    confidence = {
        "color": 0.96 if matched_color else 0.28,
        "dimensions": 0.92 if matched_dims else 0.24,
        "category": 0.88 if matched_category else 0.42,
        "shape": 0.82 if shape != "rectangular" or re.search(r"직사각|사각|rectangular", raw, re.I) else 0.36,
        "height": 0.8 if height != "standard" or matched_category else 0.4,
    }

    return make_target(
        color_key,
        width,
        length,
        category,
        source=source,
        height=height,
        shape=shape,
        attributes=attributes,
        negative_terms=negative_terms,
        confidence=confidence,
    )


def parse_dimensions(raw: str) -> tuple[int, int, bool]:
    patterns = [
        r"(\d+)\s*[x×*]\s*(\d+)",
        r"(\d+)\s*by\s*(\d+)",
        r"(\d+)\s*(?:칸|스터드|studs?)\s*(?:짜리|x|×|by|에|,|\s)\s*(\d+)\s*(?:칸|스터드|studs?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, re.I)
        if match:
            return int(match.group(1)), int(match.group(2)), True

    word = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    match = re.search(rf"({word})\s*(?:칸|스터드)\s*({word})\s*(?:칸|스터드)", raw)
    if match:
        return NUMBER_WORDS[match.group(1)], NUMBER_WORDS[match.group(2)], True

    return 2, 3, False


def infer_height(raw: str, category: str) -> str:
    if re.search(r"납작|얇|플레이트|plate|flat|thin", raw, re.I):
        return "thin"
    if re.search(r"타일|tile|스터드\s*없|studless|smooth", raw, re.I):
        return "tile"
    if re.search(r"높은|두꺼운|tall|thick", raw, re.I):
        return "tall"
    if category == "plate":
        return "thin"
    if category == "tile":
        return "tile"
    return "standard"


def parse_negative_terms(raw: str) -> list[str]:
    terms: list[str] = []
    contexts = []
    for pattern in [
        r"(.{0,18})(?:아닌|말고|빼고|제외)",
        r"(?:not|without|except)\s+(.{0,18})",
    ]:
        contexts.extend(match.group(1) for match in re.finditer(pattern, raw, re.I))
    if not contexts:
        return terms
    context = " ".join(contexts)
    for key, pattern in [*COLOR_PATTERNS, *CATEGORY_PATTERNS, *SHAPE_PATTERNS, *ATTRIBUTE_PATTERNS]:
        if pattern.search(context) and key not in terms:
            terms.append(key)
    return terms
