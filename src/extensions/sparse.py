"""Sparse retrieval extensions for startup/company intelligence.

This module adds BM25 retrieval over filing chunks as an additive capability.
It does not alter existing dense or graph retrievers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from src.retrievers import RetrievedChunk
from src.vectorstore import FaissVectorStore


def _tokenize(text: str) -> list[str]:
    """Lightweight tokenizer suitable for SEC filing prose."""
    return [tok for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\./]*", text.lower()) if tok]


@dataclass
class SparseSearchResult:
    chunk_id: str
    filing_id: str
    ticker: str
    company_name: str
    section: str
    text: str
    sentence_ids: list[int]
    score: float
    source: str = "sparse_bm25"


class SparseBM25Retriever:
    """BM25 sparse retrieval over existing chunk metadata.

    Design goals:
    - Leverage exact lexical signal for ticker names, products, and risk terms.
    - Offer optional metadata filters for company-intelligence workflows.
    - Stay additive by reusing chunk metadata from the existing FAISS store.
    """

    name = "sparse_bm25"

    def __init__(self, store: FaissVectorStore):
        self.store = store
        self.metadata = list(store.metadata)

        self._tokenized_docs: list[list[str]] = []
        for row in self.metadata:
            # Include section and ticker in sparse text to improve domain relevance.
            text = f"{row.get('ticker', '')} {row.get('section', '')} {row.get('text', '')}"
            self._tokenized_docs.append(_tokenize(text))

        self._bm25 = BM25Okapi(self._tokenized_docs) if self._tokenized_docs else None

        self._ticker_boost_terms = {
            "ticker",
            "listed",
            "exchange",
            "nasdaq",
            "nyse",
        }
        self._risk_boost_terms = {
            "risk",
            "cybersecurity",
            "regulatory",
            "litigation",
            "supply",
            "commodity",
            "inflation",
            "interest",
        }

    def _query_boost(self, query_tokens: set[str], row: dict[str, Any]) -> float:
        """Apply small domain-aware boosts for sparse ranking stability."""
        boost = 1.0
        section = str(row.get("section", "")).lower()

        if query_tokens & self._risk_boost_terms and "risk" in section:
            boost *= 1.15

        if query_tokens & self._ticker_boost_terms and "market" in section:
            boost *= 1.08

        return boost

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_ticker: str | None = None,
        filter_section: str | None = None,
    ) -> list[RetrievedChunk]:
        if not self._bm25:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self._bm25.get_scores(query_tokens)
        query_token_set = set(query_tokens)

        candidates: list[tuple[int, float]] = []
        for idx, score in enumerate(raw_scores):
            row = self.metadata[idx]
            if filter_ticker and row.get("ticker") != filter_ticker:
                continue
            if filter_section and str(row.get("section", "")).lower() != filter_section.lower():
                continue

            boosted_score = float(score) * self._query_boost(query_token_set, row)
            candidates.append((idx, boosted_score))

        if not candidates:
            return []

        # Normalize into [0,1] for fusion compatibility.
        max_score = max(score for _, score in candidates)
        min_score = min(score for _, score in candidates)
        denom = max(max_score - min_score, 1e-9)

        ranked = sorted(candidates, key=lambda item: item[1], reverse=True)[:k]
        results: list[RetrievedChunk] = []
        for idx, score in ranked:
            row = self.metadata[idx]
            norm_score = (score - min_score) / denom
            results.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    filing_id=row["filing_id"],
                    ticker=row["ticker"],
                    company_name=row["company_name"],
                    section=row["section"],
                    text=row["text"],
                    sentence_ids=row.get("sentence_ids", []),
                    score=float(norm_score),
                    source=self.name,
                    vector_score=0.0,
                    graph_score=0.0,
                    via="sparse",
                    metadata={
                        "sparse_raw_score": float(score),
                        "query": query,
                    },
                )
            )
        return results
