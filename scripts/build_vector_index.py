#!/usr/bin/env python3
"""Build a tiny offline vector index for the BrickFinder prototype.

Production version:
- ingest Rebrickable CSV/API catalog
- render or download official part images
- encode image/text with MobileCLIP, SigLIP, or CLIP
- write hnswlib index for mobile bundle or Qdrant collection for backend

This MVP creates deterministic pseudo-embeddings from part metadata so the rest
of the service pipeline can be exercised without model weights.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backend"))

from app.catalog import COMMON_PARTS  # noqa: E402

OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)
DIM = 64


def pseudo_embedding(text: str, dim: int = DIM) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    seed = digest
    while len(values) < dim:
        for byte in seed:
            values.append((byte / 255.0) * 2 - 1)
            if len(values) == dim:
                break
        seed = hashlib.sha256(seed).digest()
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [round(v / norm, 6) for v in values]


def main() -> None:
    records = []
    for part in COMMON_PARTS:
        text = f"{part.name} {part.colorLabel} {part.width}x{part.length} {part.categoryLabel} {part.partNum}"
        records.append({"id": part.id, "payload": part.to_dict(), "vector": pseudo_embedding(text)})

    json_path = OUT_DIR / "mock_part_vectors.json"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import hnswlib  # type: ignore
        import numpy as np  # type: ignore

        vectors = np.array([record["vector"] for record in records], dtype=np.float32)
        index = hnswlib.Index(space="cosine", dim=DIM)
        index.init_index(max_elements=len(records), ef_construction=120, M=16)
        index.add_items(vectors, list(range(len(records))))
        index.set_ef(32)
        hnsw_path = OUT_DIR / "mock_parts_hnsw.bin"
        index.save_index(str(hnsw_path))
        print(f"wrote {json_path} and {hnsw_path}")
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"wrote {json_path}; skipped hnswlib binary ({exc})")


if __name__ == "__main__":
    main()
