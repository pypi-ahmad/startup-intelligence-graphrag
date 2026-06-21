"""Retrieval strategies for dense and graph-aware GraphRAG behavior."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from loguru import logger

from src.config import SETTINGS
from src.embeddings import embed_text
from src.graph import CommunitySummary
from src.vectorstore import FaissVectorStore


@dataclass
class RetrievedChunk:
    """Unified retrieval contract for downstream generation/evaluation."""

    chunk_id: str
    filing_id: str
    ticker: str
    company_name: str
    section: str
    text: str
    sentence_ids: list[int]
    score: float
    source: str
    graph_score: float = 0.0
    vector_score: float = 0.0
    via: str = "vector"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "filing_id": self.filing_id,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "section": self.section,
            "text": self.text,
            "sentence_ids": self.sentence_ids,
            "score": float(self.score),
            "source": self.source,
            "graph_score": float(self.graph_score),
            "vector_score": float(self.vector_score),
            "via": self.via,
            "metadata": self.metadata,
        }


class BaseRetriever:
    name = "base"

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError


class DenseVectorRetriever(BaseRetriever):
    name = "vector"

    def __init__(self, store: FaissVectorStore, model: str | None = None):
        self.store = store
        self.model = model or SETTINGS.embed_model

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        rows = self.store.search(query=query, k=k, model=self.model)
        return [
            RetrievedChunk(
                chunk_id=row.chunk_id,
                filing_id=row.filing_id,
                ticker=row.ticker,
                company_name=row.company_name,
                section=row.section,
                text=row.text,
                sentence_ids=row.sentence_ids,
                score=row.score,
                source=self.name,
                vector_score=row.score,
                via="vector",
            )
            for row in rows
        ]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _section_node(filing_id: str, section: str) -> str:
    return f"section::{filing_id}::{_slug(section)}"


class GraphLocalRetriever(BaseRetriever):
    """Local graph retrieval: dense seeds + entity-hop expansion."""

    name = "graph_local"

    def __init__(
        self,
        store: FaissVectorStore,
        graph: nx.Graph,
        model: str | None = None,
        vector_weight: float = 0.65,
        graph_weight: float = 0.35,
    ):
        self.store = store
        self.graph = graph
        self.model = model or SETTINGS.embed_model
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

        self._metadata_by_chunk = {row["chunk_id"]: row for row in self.store.metadata}
        self._section_to_chunks: dict[str, list[str]] = {}
        for row in self.store.metadata:
            section_node = _section_node(row["filing_id"], row["section"])
            self._section_to_chunks.setdefault(section_node, []).append(row["chunk_id"])

    def _expand_sections(self, seed_chunks: list[RetrievedChunk]) -> dict[str, float]:
        section_scores: dict[str, float] = {}

        for seed in seed_chunks:
            seed_section = _section_node(seed.filing_id, seed.section)
            if not self.graph.has_node(seed_section):
                continue

            entity_neighbors = [
                node
                for node in self.graph.neighbors(seed_section)
                if self.graph.nodes[node].get("node_type") == "entity"
            ]

            for entity_node in entity_neighbors:
                entity_degree = max(1, self.graph.degree(entity_node))
                entity_boost = 1.0 / np.sqrt(entity_degree)

                for section_node in self.graph.neighbors(entity_node):
                    if self.graph.nodes[section_node].get("node_type") != "section":
                        continue

                    edge_weight = self.graph[entity_node][section_node].get("weight", 1.0)
                    contribution = float(edge_weight) * float(entity_boost) * max(seed.score, 0.0)
                    section_scores[section_node] = section_scores.get(section_node, 0.0) + contribution

        return section_scores

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        seed_chunks = DenseVectorRetriever(self.store, model=self.model).retrieve(query=query, k=max(k * 2, k))
        if not seed_chunks:
            return []

        vector_scores: dict[str, float] = {chunk.chunk_id: chunk.score for chunk in seed_chunks}
        section_scores = self._expand_sections(seed_chunks)

        # Convert expanded section scores to chunk-level graph scores.
        chunk_graph_scores: dict[str, float] = {}
        for section_node, score in section_scores.items():
            chunk_ids = self._section_to_chunks.get(section_node, [])
            for chunk_id in chunk_ids:
                chunk_graph_scores[chunk_id] = chunk_graph_scores.get(chunk_id, 0.0) + score

        max_vec = max(max(vector_scores.values()), 1e-9)
        max_graph = max(chunk_graph_scores.values()) if chunk_graph_scores else 0.0
        if max_graph <= 0:
            max_graph = 1e-9

        candidate_ids = set(vector_scores.keys()) | set(chunk_graph_scores.keys())
        retrieved: list[RetrievedChunk] = []

        for chunk_id in candidate_ids:
            md = self._metadata_by_chunk.get(chunk_id)
            if md is None:
                continue
            vec = vector_scores.get(chunk_id, 0.0) / max_vec
            gph = chunk_graph_scores.get(chunk_id, 0.0) / max_graph
            combined = (self.vector_weight * vec) + (self.graph_weight * gph)

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    filing_id=md["filing_id"],
                    ticker=md["ticker"],
                    company_name=md["company_name"],
                    section=md["section"],
                    text=md["text"],
                    sentence_ids=md.get("sentence_ids", []),
                    score=combined,
                    source=self.name,
                    vector_score=vec,
                    graph_score=gph,
                    via="graph_expand" if chunk_id in chunk_graph_scores and chunk_id not in vector_scores else "vector",
                    metadata={"query": query},
                )
            )

        retrieved.sort(key=lambda row: row.score, reverse=True)
        return retrieved[:k]


class GraphGlobalRetriever(BaseRetriever):
    """Global retrieval over community summaries for cross-company themes."""

    name = "graph_global"

    def __init__(
        self,
        store: FaissVectorStore,
        graph: nx.Graph,
        partition: dict[str, int],
        summaries: list[CommunitySummary],
        model: str | None = None,
    ):
        self.store = store
        self.graph = graph
        self.partition = partition
        self.summaries = summaries
        self.model = model or SETTINGS.embed_model

        self._metadata_by_chunk = {row["chunk_id"]: row for row in self.store.metadata}
        self._section_to_chunks: dict[str, list[str]] = {}
        for row in self.store.metadata:
            node = _section_node(row["filing_id"], row["section"])
            self._section_to_chunks.setdefault(node, []).append(row["chunk_id"])

        self._summary_vectors: dict[int, np.ndarray] = {}
        for summary in self.summaries:
            text = f"{summary.summary}\nTickers: {', '.join(summary.member_tickers)}"
            self._summary_vectors[summary.community_id] = embed_text(text, model=self.model)

    def _fallback_summary_candidates(self) -> list[CommunitySummary]:
        by_community: dict[int, list[str]] = {}
        for node, community_id in self.partition.items():
            if community_id < 0:
                continue
            if self.graph.nodes[node].get("node_type") != "entity":
                continue
            by_community.setdefault(community_id, []).append(node)

        candidates: list[CommunitySummary] = []
        for community_id, nodes in by_community.items():
            labels = [self.graph.nodes[node].get("name", node.replace("entity::", "")) for node in nodes[:8]]
            tickers: set[str] = set()
            for node in nodes:
                tickers.update(self.graph.nodes[node].get("tickers", []))
            candidates.append(
                CommunitySummary(
                    community_id=community_id,
                    size=len(nodes),
                    member_entities=nodes,
                    member_tickers=sorted(tickers),
                    summary="Theme entities: " + ", ".join(labels),
                )
            )
        candidates.sort(key=lambda item: item.size, reverse=True)
        return candidates

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        summaries = self.summaries or self._fallback_summary_candidates()
        if not summaries:
            logger.warning("GraphGlobalRetriever has no communities/summaries.")
            return []

        # Lazily embed fallback summaries if needed.
        for summary in summaries:
            if summary.community_id not in self._summary_vectors:
                text = f"{summary.summary}\nTickers: {', '.join(summary.member_tickers)}"
                self._summary_vectors[summary.community_id] = embed_text(text, model=self.model)

        query_vec = embed_text(query, model=self.model)

        community_scores: list[tuple[int, float]] = []
        for summary in summaries:
            vector = self._summary_vectors[summary.community_id]
            score = float(np.dot(query_vec, vector))
            community_scores.append((summary.community_id, score))
        community_scores.sort(key=lambda item: item[1], reverse=True)

        score_lookup = {cid: score for cid, score in community_scores}
        summary_lookup = {summary.community_id: summary for summary in summaries}

        chunk_scores: dict[str, float] = {}
        chunk_meta: dict[str, dict[str, Any]] = {}

        for community_id, community_score in community_scores[: max(3, k)]:
            summary = summary_lookup[community_id]
            for entity_node in summary.member_entities:
                if not self.graph.has_node(entity_node):
                    continue
                for neighbor in self.graph.neighbors(entity_node):
                    if self.graph.nodes[neighbor].get("node_type") != "section":
                        continue
                    section_weight = self.graph[entity_node][neighbor].get("weight", 1.0)
                    for chunk_id in self._section_to_chunks.get(neighbor, []):
                        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + float(community_score) * (1.0 + 0.1 * float(section_weight))
                        md = self._metadata_by_chunk.get(chunk_id)
                        if md is not None:
                            chunk_meta[chunk_id] = {
                                "community_id": community_id,
                                "community_score": community_score,
                                "section_weight": section_weight,
                                "summary": summary.summary,
                            }

        if not chunk_scores:
            return []

        max_score = max(chunk_scores.values()) if chunk_scores else 1.0
        if max_score <= 0:
            max_score = 1.0

        ranked = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)
        output: list[RetrievedChunk] = []
        for chunk_id, raw_score in ranked[:k]:
            md = self._metadata_by_chunk.get(chunk_id)
            if md is None:
                continue
            norm_score = raw_score / max_score
            output.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    filing_id=md["filing_id"],
                    ticker=md["ticker"],
                    company_name=md["company_name"],
                    section=md["section"],
                    text=md["text"],
                    sentence_ids=md.get("sentence_ids", []),
                    score=norm_score,
                    source=self.name,
                    graph_score=norm_score,
                    vector_score=0.0,
                    via="community",
                    metadata=chunk_meta.get(chunk_id, {}),
                )
            )

        return output


class HybridRetriever(BaseRetriever):
    """Reciprocal-rank fusion over dense + local + global retrievers."""

    name = "hybrid"

    def __init__(
        self,
        store: FaissVectorStore,
        graph: nx.Graph,
        partition: dict[str, int],
        summaries: list[CommunitySummary],
        model: str | None = None,
        k_rrf: int = 60,
    ):
        self.vector = DenseVectorRetriever(store, model=model)
        self.local = GraphLocalRetriever(store, graph, model=model)
        self.global_ = GraphGlobalRetriever(store, graph, partition, summaries, model=model)
        self.k_rrf = k_rrf

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        runs = [
            self.vector.retrieve(query, k=k * 3),
            self.local.retrieve(query, k=k * 3),
            self.global_.retrieve(query, k=k * 3),
        ]

        rrf_scores: dict[str, float] = {}
        chunk_rows: dict[str, RetrievedChunk] = {}

        for result_list in runs:
            for rank, chunk in enumerate(result_list, start=1):
                rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (self.k_rrf + rank)
                chunk_rows[chunk.chunk_id] = chunk

        ranked_ids = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:k]

        output: list[RetrievedChunk] = []
        for chunk_id, score in ranked_ids:
            row = chunk_rows[chunk_id]
            output.append(
                RetrievedChunk(
                    chunk_id=row.chunk_id,
                    filing_id=row.filing_id,
                    ticker=row.ticker,
                    company_name=row.company_name,
                    section=row.section,
                    text=row.text,
                    sentence_ids=row.sentence_ids,
                    score=float(score),
                    source=self.name,
                    vector_score=row.vector_score,
                    graph_score=row.graph_score,
                    via=row.via,
                    metadata={**row.metadata, "fused": True},
                )
            )

        return output
