"""Small vector index abstraction for LEGO part candidates.

The production path should replace the fallback embedder with CLIP/MobileCLIP
and use hnswlib/Qdrant for ANN search. The local fallback keeps the same shape:
query vector -> nearest part metadata -> downstream visual verifier.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .catalog import COLORS, COMMON_PARTS, PartTarget


CATEGORY_ORDER = ["brick", "plate", "tile"]
HEIGHT_ORDER = ["standard", "thin", "tile", "tall"]
SHAPE_ORDER = ["rectangular", "slope", "round", "corner"]
EMBEDDING_DIM = len(COLORS) + len(CATEGORY_ORDER) + len(HEIGHT_ORDER) + len(SHAPE_ORDER) + 8


@dataclass(frozen=True)
class IndexHit:
    rank: int
    score: float
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PartVectorIndex:
    def __init__(self, parts: Iterable[PartTarget] = COMMON_PARTS, artifact_path: Path | None = None):
        self.parts = list(parts)
        self.payloads: list[dict[str, Any]]
        self.payloads = [part.to_dict() for part in self.parts]
        self.artifact_path = artifact_path or default_artifact_path()
        self.loaded_artifact = False
        loaded_vectors = self._load_artifact(self.artifact_path)
        if loaded_vectors is None:
            vectors = [text_part_embedding(part) for part in self.parts]
        else:
            vectors = loaded_vectors
            self.loaded_artifact = True
        self.vectors = np.vstack(vectors).astype(np.float32)
        self.backend = "numpy-cosine+json-artifact" if self.loaded_artifact else "numpy-cosine"
        self._hnsw = None
        try:
            import hnswlib  # type: ignore

            index = hnswlib.Index(space="cosine", dim=self.vectors.shape[1])
            index.init_index(max_elements=len(self.parts), ef_construction=120, M=16)
            index.add_items(self.vectors, np.arange(len(self.parts)))
            index.set_ef(32)
            self._hnsw = index
            self.backend = "hnswlib+json-artifact" if self.loaded_artifact else "hnswlib"
        except Exception:
            self._hnsw = None

    def search(self, target: PartTarget, top_k: int = 8) -> list[IndexHit]:
        query = text_part_embedding(target)
        if self._hnsw is not None:
            labels, distances = self._hnsw.knn_query(query.reshape(1, -1), k=min(top_k, len(self.parts)))
            rows = [(int(label), 1.0 - float(distance)) for label, distance in zip(labels[0], distances[0])]
        else:
            scores = self.vectors @ query
            order = np.argsort(-scores)[:top_k]
            rows = [(int(index), float(scores[index])) for index in order]
        return [
            IndexHit(rank=rank, score=round(score, 4), payload=self.payloads[index])
            for rank, (index, score) in enumerate(rows, start=1)
        ]

    def _load_artifact(self, path: Path) -> list[np.ndarray] | None:
        if not path.exists():
            return None
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(records, dict):
                records = records.get("records", [])
            payloads = []
            vectors = []
            for record in records:
                payload = record.get("payload")
                vector = record.get("vector")
                if not payload or not vector:
                    continue
                if len(vector) != EMBEDDING_DIM:
                    return None
                payloads.append(payload)
                vectors.append(np.array(vector, dtype=np.float32))
            if not vectors:
                return None
            self.payloads = payloads
            self.parts = []
            return vectors
        except Exception:
            return None


def text_part_embedding(part: PartTarget) -> np.ndarray:
    color_keys = list(COLORS.keys())
    color = one_hot(color_keys.index(part.colorKey), len(color_keys))
    category_index = CATEGORY_ORDER.index(part.category) if part.category in CATEGORY_ORDER else 0
    category = one_hot(category_index, len(CATEGORY_ORDER))
    height_index = HEIGHT_ORDER.index(part.height) if part.height in HEIGHT_ORDER else 0
    height = one_hot(height_index, len(HEIGHT_ORDER))
    shape_index = SHAPE_ORDER.index(part.shape) if part.shape in SHAPE_ORDER else 0
    shape = one_hot(shape_index, len(SHAPE_ORDER))
    attribute_count = min(len(part.attributes), 4) / 4.0
    negative_count = min(len(part.negativeTerms), 4) / 4.0
    dims = np.array(
        [
            part.width / 8.0,
            part.length / 8.0,
            min(part.width, part.length) / 8.0,
            max(part.width, part.length) / 8.0,
            max(part.width, part.length) / max(1, min(part.width, part.length)) / 4.0,
            1.0,
            attribute_count,
            negative_count,
        ],
        dtype=np.float32,
    )
    vector = np.concatenate([color * 1.8, category * 1.2, height * 0.75, shape * 0.55, dims])
    return normalize(vector)


def proposal_embedding(
    dominant_color_key: str,
    aspect_ratio: float,
    fill_ratio: float,
    target: PartTarget,
) -> np.ndarray:
    color_keys = list(COLORS.keys())
    color_index = color_keys.index(dominant_color_key) if dominant_color_key in color_keys else 0
    color = one_hot(color_index, len(color_keys))
    category = one_hot(CATEGORY_ORDER.index(target.category), len(CATEGORY_ORDER))
    height_index = HEIGHT_ORDER.index(target.height) if target.height in HEIGHT_ORDER else 0
    height = one_hot(height_index, len(HEIGHT_ORDER))
    shape_index = SHAPE_ORDER.index(target.shape) if target.shape in SHAPE_ORDER else 0
    shape = one_hot(shape_index, len(SHAPE_ORDER))
    expected_long = max(target.width, target.length)
    expected_short = min(target.width, target.length)
    dims = np.array(
        [
            target.width / 8.0,
            target.length / 8.0,
            expected_short / 8.0,
            expected_long / 8.0,
            aspect_ratio / 4.0,
            fill_ratio,
            min(len(target.attributes), 4) / 4.0,
            0.0,
        ],
        dtype=np.float32,
    )
    vector = np.concatenate([color * 1.8, category * 0.75, height * 0.55, shape * 0.4, dims])
    return normalize(vector)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(normalize(a) @ normalize(b))


def one_hot(index: int, size: int) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def default_artifact_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "rebrickable" / "parts_index.json"
