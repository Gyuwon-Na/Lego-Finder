from backend.app.query_parser import parse_query


def test_parse_korean_brick():
    target = parse_query("빨간색 2x3 기본 브릭")
    assert target.colorKey == "red"
    assert target.width == 2
    assert target.length == 3
    assert target.category == "brick"
    assert target.partNum == "3002"


def test_parse_english_plate():
    target = parse_query("blue 1x4 plate")
    assert target.colorKey == "blue"
    assert target.width == 1
    assert target.length == 4
    assert target.category == "plate"
    assert target.partNum == "3710"


def test_parse_structured_prompt_details():
    target = parse_query("타일 말고 납작한 파란 두 칸 네 칸 블록")
    assert target.colorKey == "blue"
    assert target.width == 2
    assert target.length == 4
    assert target.category == "plate"
    assert target.height == "thin"
    assert "tile" in target.negativeTerms
    assert target.confidence["dimensions"] > 0.8


def test_parse_shape_and_attribute_terms():
    target = parse_query("transparent gray 1 by 2 slope")
    assert target.colorKey == "gray"
    assert target.width == 1
    assert target.length == 2
    assert target.shape == "slope"
    assert "transparent" in target.attributes


def test_parse_korean_face_head_does_not_default_to_2x3_brick():
    target = parse_query("노란색 사람 얼굴 모양 브릭")
    assert target.colorKey == "yellow"
    assert target.width == 1
    assert target.length == 1
    assert target.category == "head"
    assert target.categoryLabel == "미니피겨 헤드"
    assert target.shape == "round"
    assert "face_print" in target.attributes
    assert "2x3" not in target.name
