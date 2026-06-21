"""Multimodal retrieval over text/table/figure-text units."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from src.embeddings import embed_text, embed_texts
from src.extensions.multimodal import MultimodalUnit
from src.retrievers import RetrievedChunk


def _tokenize(text: str) -> list[str]:
    return [tok for tok in text.lower().replace("\n", " ").split(" ") if tok]


@dataclass
class MultimodalIndex:
    units: list[MultimodalUnit]
    embeddings: np.ndarray
    bm25: BM25Okapi


class MultimodalRetriever:
    """Retrieve across modalities with dense+sparse fusion and modality weighting."""

    name = "multimodal_hybrid"

    def __init__(
        self,
        units: list[MultimodalUnit],
        modality_weights: dict[str, float] | None = None,
    ):
        self.units = units
        self.modality_weights = modality_weights or {
            "text": 1.0,
            "table": 1.15,
            "figure_text": 1.05,
            "ocr_text": 1.2,
            "vision_text": 1.1,
        }

        texts = [u.text for u in units]
        self.embeddings = embed_texts(texts) if texts else np.zeros((0, 2560), dtype=np.float32)
        self.bm25 = BM25Okapi([_tokenize(t) for t in texts]) if texts else None

    def _dense_scores(self, query: str) -> np.ndarray:
        if not self.units:
            return np.zeros((0,), dtype=np.float32)
        qvec = embed_text(query)
        return np.dot(self.embeddings, qvec).astype(np.float32)

    def _sparse_scores(self, query: str) -> np.ndarray:
        if not self.units or self.bm25 is None:
            return np.zeros((0,), dtype=np.float32)
        arr = np.array(self.bm25.get_scores(_tokenize(query)), dtype=np.float32)
        return arr

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        max_val = float(scores.max())
        min_val = float(scores.min())
        if max_val - min_val <= 1e-9:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> list[RetrievedChunk]:
        if not self.units:
            return []

        dense = self._normalize(self._dense_scores(query))
        sparse = self._normalize(self._sparse_scores(query))
        fused = 0.6 * dense + 0.4 * sparse

        weighted = []
        for idx, unit in enumerate(self.units):
            modality_w = self.modality_weights.get(unit.modality, 1.0)
            score = float(fused[idx] * modality_w)
            weighted.append((idx, score))

        weighted.sort(key=lambda row: row[1], reverse=True)
        top = weighted[:k]

        out: list[RetrievedChunk] = []
        for idx, score in top:
            unit = self.units[idx]
            out.append(
                RetrievedChunk(
                    chunk_id=unit.unit_id,
                    filing_id=unit.filing_id,
                    ticker=unit.ticker,
                    company_name=unit.company_name,
                    section=unit.section,
                    text=unit.text,
                    sentence_ids=[],
                    score=score,
                    source=self.name,
                    vector_score=float(dense[idx]),
                    graph_score=0.0,
                    via=f"multimodal_{unit.modality}",
                    metadata={
                        "modality": unit.modality,
                        "dense_score": float(dense[idx]),
                        "sparse_score": float(sparse[idx]),
                        **unit.metadata,
                    },
                )
            )

        return out
