"""FAISS-backed vector store for filing chunks."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from loguru import logger

from src.chunking import Chunk, load_chunks
from src.config import EMBEDDINGS_DIR, SETTINGS, ensure_dirs
from src.embeddings import embed_text, embed_texts, embedding_dim


@dataclass
class SearchResult:
    """Dense retrieval result row."""

    chunk_id: str
    filing_id: str
    ticker: str
    company_name: str
    section: str
    text: str
    sentence_ids: list[int]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FaissVectorStore:
    """Thin wrapper around FAISS index + metadata sidecar."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict[str, Any]] = []

    def add(self, chunks: list[Chunk], model: str | None = None) -> None:
        if not chunks:
            logger.warning("No chunks provided to vector store add().")
            return

        model = model or SETTINGS.embed_model
        vectors = embed_texts([c.text for c in chunks], model=model)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch: got {vectors.shape[1]}, expected {self.dim}")

        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.index.add(vectors)

        for chunk in chunks:
            self.metadata.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "filing_id": chunk.filing_id,
                    "ticker": chunk.ticker,
                    "company_name": chunk.company_name,
                    "section": chunk.section,
                    "text": chunk.text,
                    "sentence_ids": chunk.sentence_ids,
                    "token_count": chunk.token_count,
                }
            )

        logger.info("FAISS index now contains {} vectors", self.index.ntotal)

    def search(
        self,
        query: str,
        k: int = 5,
        model: str | None = None,
        filter_ticker: str | None = None,
        filter_section: str | None = None,
    ) -> list[SearchResult]:
        if self.index.ntotal == 0:
            return []

        model = model or SETTINGS.embed_model
        overfetch = min(self.index.ntotal, max(k, 1) * (5 if (filter_ticker or filter_section) else 1))

        qvec = embed_text(query, model=model)
        qvec = np.ascontiguousarray(qvec.reshape(1, -1).astype(np.float32))
        scores, indices = self.index.search(qvec, overfetch)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            md = self.metadata[idx]
            if filter_ticker and md["ticker"] != filter_ticker:
                continue
            if filter_section and md["section"] != filter_section:
                continue
            results.append(
                SearchResult(
                    chunk_id=md["chunk_id"],
                    filing_id=md["filing_id"],
                    ticker=md["ticker"],
                    company_name=md["company_name"],
                    section=md["section"],
                    text=md["text"],
                    sentence_ids=md.get("sentence_ids", []),
                    score=float(score),
                )
            )
            if len(results) >= k:
                break

        return results

    def save(self, output_dir: Path | None = None) -> Path:
        ensure_dirs()
        output_dir = output_dir or EMBEDDINGS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(output_dir / "faiss.index"))
        with open(output_dir / "metadata.pkl", "wb") as file:
            pickle.dump(self.metadata, file)
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
            json.dump(self.metadata, file, indent=2)

        logger.info("Saved FAISS index and metadata to {}", output_dir)
        return output_dir

    @classmethod
    def load(cls, input_dir: Path | None = None) -> "FaissVectorStore":
        input_dir = input_dir or EMBEDDINGS_DIR
        index = faiss.read_index(str(input_dir / "faiss.index"))
        with open(input_dir / "metadata.pkl", "rb") as file:
            metadata = pickle.load(file)

        store = cls(dim=index.d)
        store.index = index
        store.metadata = metadata
        logger.info("Loaded FAISS index with {} vectors from {}", index.ntotal, input_dir)
        return store

    def stats(self) -> dict[str, Any]:
        ticker_counts: dict[str, int] = {}
        section_counts: dict[str, int] = {}

        for row in self.metadata:
            ticker = row["ticker"]
            section = row["section"]
            ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
            section_counts[section] = section_counts.get(section, 0) + 1

        return {
            "n_vectors": int(self.index.ntotal),
            "dim": int(self.dim),
            "approx_size_mb": float(self.index.ntotal * self.dim * 4 / 1e6),
            "chunks_per_ticker": dict(sorted(ticker_counts.items())),
            "chunks_per_section": dict(sorted(section_counts.items())),
        }


def build_from_chunks(
    chunks: list[Chunk] | None = None,
    model: str | None = None,
    save_dir: Path | None = None,
) -> FaissVectorStore:
    """Convenience builder to embed chunks and persist index."""

    model = model or SETTINGS.embed_model
    chunks = chunks or load_chunks()
    if not chunks:
        raise ValueError("No chunks found to index.")

    dim = embedding_dim(model=model)
    store = FaissVectorStore(dim=dim)
    store.add(chunks, model=model)
    store.save(save_dir)
    return store


if __name__ == "__main__":
    from src.chunking import chunk_corpus
    from src.ingest import build_corpus

    filings = build_corpus()
    chunks = chunk_corpus(filings)
    store = build_from_chunks(chunks)
    print(json.dumps(store.stats(), indent=2))
