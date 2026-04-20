import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 전역: CLIP 모델 (최초 1회만 로드) ──────────────────────────────
print("CLIP 모델 로딩 중... (첫 실행 시 약 600MB 다운로드)")
_clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_clip_model.eval()
print("CLIP 로드 완료!")

ALL_COLORS = [
    "red", "blue", "green", "black", "white",
    "gray", "grey", "orange", "yellow", "brown",
    "purple", "pink", "tan", "dark"
]

def search_rebrickable(query: str, api_key: str, top_k: int = 20) -> list:
    url = "https://rebrickable.com/api/v3/lego/parts/"
    params = {
        "key":          api_key,
        "search":       query,
        "page_size":    top_k,
        "inc_part_img": 1,
        # color_id 제거 — API 필터 대신 후처리로 색상 걸러냄
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    # 쿼리에서 색상 키워드 추출
    queried_colors = [c for c in ALL_COLORS if c in query.lower()]
    other_colors   = [c for c in ALL_COLORS if c not in queried_colors]

    results = []
    for part in data.get("results", []):
        img_url = part.get("part_img_url") or ""
        if not img_url:
            continue

        name_lower = part["name"].lower()

        # 다른 색상 키워드가 이름에 있으면 제외
        # 예: "yellow 2x4" 검색 시 "Brick 2x4 with Red Stripe" 제외
        if queried_colors and any(c in name_lower for c in other_colors):
            continue

        results.append({
            "part_num": part["part_num"],
            "name":     part["name"],
            "img_url":  img_url,
        })

    print(f"  → Rebrickable에서 {len(results)}개 후보 확보")
    return results


def _load_image_from_url(url: str):
    """URL에서 PIL 이미지를 다운로드. 실패 시 None 반환."""
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def rank_with_clip(
    query: str,
    candidates: list,
    top_k: int = 5,
) -> list:
    """
    CLIP으로 텍스트-이미지 유사도를 계산해 후보를 재정렬한다.
    """
    # 1) 텍스트 임베딩
    text_inputs  = _clip_processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_feat = _clip_model.get_text_features(**text_inputs)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)   # L2 정규화

    # 2) 이미지 임베딩 + 유사도 계산
    scored = []
    for i, part in enumerate(candidates):
        print(f"  이미지 처리 중 {i+1}/{len(candidates)}: {part['name'][:40]}")
        img = _load_image_from_url(part["img_url"])
        if img is None:
            continue

        img_inputs = _clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            img_feat = _clip_model.get_image_features(**img_inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        similarity = (text_feat @ img_feat.T).item()   # 코사인 유사도
        scored.append({**part, "score": similarity, "pil_image": img})

    # 3) 유사도 내림차순 정렬
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def display_results(results: list, query: str) -> None:
    """검색 결과를 matplotlib 그리드로 시각화한다."""
    n = len(results)
    if n == 0:
        print("표시할 결과가 없습니다.")
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(f'검색어: "{query}"', fontsize=14, fontweight="bold", y=1.02)

    for ax, part in zip(axes, results):
        ax.imshow(part["pil_image"])
        ax.axis("off")

        score_pct = part["score"] * 100
        # 유사도에 따라 테두리 색 변경 (녹 → 황 → 적)
        if score_pct >= 25:
            color = "green"
        elif score_pct >= 15:
            color = "orange"
        else:
            color = "red"

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)

        ax.set_title(
            f"{part['name'][:28]}\n"
            f"#{part['part_num']}  |  유사도 {score_pct:.1f}%",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("result.png", dpi=150, bbox_inches="tight")
    print("\n결과를 result.png로 저장했습니다.")
    plt.show()