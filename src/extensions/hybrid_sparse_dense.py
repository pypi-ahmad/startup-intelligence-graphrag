"""Sparse+dense hybrid retrieval extensions.

Adds a domain-aware hybrid retriever that combines:
- existing dense vector retrieval
- additive BM25 sparse retrieval

This module is intentionally separate from existing retrievers.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.extensions.sparse import SparseBM25Retriever
from src.retrievers import DenseVectorRetriever, RetrievedChunk
from src.vectorstore import FaissVectorStore


@dataclass
class HybridConfig:
    dense_weight: float = 0.55
    sparse_weight: float = 0.45
    rrf_k: int = 60


class HybridSparseDenseRetriever:
    """Hybrid retriever combining dense and sparse channels.

    Supports:
    - weighted score fusion
    - reciprocal rank fusion (RRF)
    """

    name = "hybrid_sparse_dense"

    def __init__(
        self,
        store: FaissVectorStore,
        dense_weight: float = 0.55,
        sparse_weight: float = 0.45,
        rrf_k: int = 60,
    ):
        self.store = store
        self.dense = DenseVectorRetriever(store)
        self.sparse = SparseBM25Retriever(store)
        self.config = HybridConfig(
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            rrf_k=rrf_k,
        )

    def _weighted_fusion(
        self,
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
        k: int,
    ) -> list[RetrievedChunk]:
        dense_map = {row.chunk_id: row for row in dense_results}
        sparse_map = {row.chunk_id: row for row in sparse_results}

        all_ids = set(dense_map.keys()) | set(sparse_map.keys())
        merged: list[RetrievedChunk] = []

        for chunk_id in all_ids:
            dense_row = dense_map.get(chunk_id)
            sparse_row = sparse_map.get(chunk_id)

            base = dense_row or sparse_row
            assert base is not None

            dscore = dense_row.score if dense_row else 0.0
            sscore = sparse_row.score if sparse_row else 0.0

            fused_score = (
                self.config.dense_weight * dscore
                + self.config.sparse_weight * sscore
            )

            merged.append(
                RetrievedChunk(
                    chunk_id=base.chunk_id,
                    filing_id=base.filing_id,
                    ticker=base.ticker,
                    company_name=base.company_name,
                    section=base.section,
                    text=base.text,
                    sentence_ids=base.sentence_ids,
                    score=float(fused_score),
                    source=self.name,
                    vector_score=float(dscore),
                    graph_score=0.0,
                    via="dense_sparse_weighted",
                    metadata={
                        "dense_score": float(dscore),
                        "sparse_score": float(sscore),
                        "fusion": "weighted",
                    },
                )
            )

        merged.sort(key=lambda row: row.score, reverse=True)
        return merged[:k]

    def _rrf_fusion(
        self,
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
        k: int,
    ) -> list[RetrievedChunk]:
        rrf_scores: dict[str, float] = {}
        rows: dict[str, RetrievedChunk] = {}

        for rank, row in enumerate(dense_results, start=1):
            rrf_scores[row.chunk_id] = rrf_scores.get(row.chunk_id, 0.0) + 1.0 / (
                self.config.rrf_k + rank
            )
            rows[row.chunk_id] = row

        for rank, row in enumerate(sparse_results, start=1):
            rrf_scores[row.chunk_id] = rrf_scores.get(row.chunk_id, 0.0) + 1.0 / (
                self.config.rrf_k + rank
            )
            rows.setdefault(row.chunk_id, row)

        ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:k]

        out: list[RetrievedChunk] = []
        for chunk_id, score in ranked:
            base = rows[chunk_id]
            out.append(
                RetrievedChunk(
                    chunk_id=base.chunk_id,
                    filing_id=base.filing_id,
                    ticker=base.ticker,
                    company_name=base.company_name,
                    section=base.section,
                    text=base.text,
                    sentence_ids=base.sentence_ids,
                    score=float(score),
                    source=self.name,
                    vector_score=base.vector_score,
                    graph_score=0.0,
                    via="dense_sparse_rrf",
                    metadata={
                        "fusion": "rrf",
                    },
                )
            )
        return out

    def retrieve(
        self,
        query: str,
        k: int = 5,
        fusion_mode: str = "weighted",
        filter_ticker: str | None = None,
        filter_section: str | None = None,
    ) -> list[RetrievedChunk]:
        dense_results = self.dense.retrieve(query, k=max(k * 3, 10))
        if filter_ticker:
            dense_results = [r for r in dense_results if r.ticker == filter_ticker]
        if filter_section:
            dense_results = [r for r in dense_results if r.section.lower() == filter_section.lower()]

        sparse_results = self.sparse.retrieve(
            query,
            k=max(k * 3, 10),
            filter_ticker=filter_ticker,
            filter_section=filter_section,
        )

        if fusion_mode == "rrf":
            return self._rrf_fusion(dense_results, sparse_results, k)
        return self._weighted_fusion(dense_results, sparse_results, k)
