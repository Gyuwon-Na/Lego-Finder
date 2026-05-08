import cv2
import numpy as np

from backend.app.vision import search_image_bytes


def test_red_2x4_prefers_dominant_red_block_over_red_patch():
    image = np.full((260, 520, 3), (42, 42, 42), dtype=np.uint8)
    cv2.rectangle(image, (40, 70), (220, 160), (53, 57, 229), -1)
    cv2.rectangle(image, (300, 60), (480, 165), (53, 216, 253), -1)
    cv2.rectangle(image, (300, 60), (360, 165), (53, 57, 229), -1)

    ok, encoded = cv2.imencode(".png", image)
    assert ok

    result = search_image_bytes(encoded.tobytes(), "2x4 red brick")
    assert result["detections"]
    top = result["detections"][0]
    assert top["bbox"]["x"] < 260
    assert top["colorDominance"] > 0.75


def test_target_color_refinement_does_not_keep_multi_color_blob_box():
    image = np.full((260, 520, 3), (42, 42, 42), dtype=np.uint8)
    cv2.rectangle(image, (300, 60), (480, 165), (53, 216, 253), -1)
    cv2.rectangle(image, (300, 60), (360, 165), (53, 57, 229), -1)

    ok, encoded = cv2.imencode(".png", image)
    assert ok

    result = search_image_bytes(encoded.tobytes(), "2x4 red brick")
    assert result["detections"]
    top = result["detections"][0]
    assert top["bbox"]["width"] < 100
    assert top["bbox"]["height"] < 130


def test_dimension_parser_accepts_reversed_orientation_in_catalog_rank():
    from backend.app.main import rank_candidates

    ranked = rank_candidates("red", 4, 2, "brick")
    assert ranked[0]["partNum"] == "3001"
