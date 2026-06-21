"""Agentic GraphRAG orchestration with corrective retrieval loop."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

from src.config import SETTINGS
from src.generation import generate_answer
from src.ollama_client import get_client
from src.retrievers import (
    BaseRetriever,
    DenseVectorRetriever,
    GraphGlobalRetriever,
    GraphLocalRetriever,
    HybridRetriever,
    RetrievedChunk,
)


@dataclass
class AgentTraceStep:
    step: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    query: str
    query_type: str
    answer: str
    citations: list[dict[str, Any]]
    chunks: list[RetrievedChunk]
    iterations: int
    trace: list[AgentTraceStep]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "answer": self.answer,
            "citations": self.citations,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "iterations": self.iterations,
            "trace": [
                {
                    "step": step.step,
                    "details": step.details,
                }
                for step in self.trace
            ],
        }


CLASSIFY_PROMPT = """Classify the SEC-filing intelligence question as one of:
- local: specific company or entity question
- global: cross-company thematic question
- factual: narrow factual lookup

Return strict JSON: {{"type": "local|global|factual"}}

Question: {query}
"""


DOC_GRADE_PROMPT = """You are grading retrieval relevance for a SEC filing question.
For each chunk id, mark relevant true/false.
Return strict JSON: {{"grades": [{{"id": <int>, "relevant": true|false}}]}}

Question:
{query}

Chunks:
{chunks}
"""


class AgenticGraphRAG:
    """Corrective GraphRAG loop with retrieval grading and fallback routing."""

    def __init__(
        self,
        retrievers: dict[str, BaseRetriever],
        max_iterations: int = 3,
    ):
        self.retrievers = retrievers
        self.max_iterations = max_iterations

    def _chat_json(self, prompt: str, model: str | None = None) -> dict[str, Any]:
        model = model or SETTINGS.generator_model
        if SETTINGS.ollama_host:
            os.environ.setdefault("OLLAMA_HOST", SETTINGS.ollama_host)

        try:
            client = get_client()
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0},
            )
            return json.loads(response["message"]["content"])
        except Exception as exc:
            logger.warning("Agent JSON call failed: {}", exc)
            return {}

    def classify_query(self, query: str) -> str:
        raw = self._chat_json(CLASSIFY_PROMPT.format(query=query))
        qtype = str(raw.get("type", "local")).lower().strip()
        if qtype not in {"local", "global", "factual"}:
            qtype = "local"
        return qtype

    def _primary_retriever_order(self, query_type: str) -> list[str]:
        if query_type == "global":
            return ["graph_global", "hybrid", "graph_local", "vector"]
        if query_type == "factual":
            return ["vector", "graph_local", "hybrid"]
        return ["graph_local", "hybrid", "vector", "graph_global"]

    def _grade_chunks(self, query: str, chunks: list[RetrievedChunk]) -> list[int]:
        if not chunks:
            return []

        rows = []
        for idx, chunk in enumerate(chunks, start=1):
            rows.append(
                {
                    "id": idx,
                    "ticker": chunk.ticker,
                    "section": chunk.section,
                    "snippet": chunk.text[:320],
                }
            )

        raw = self._chat_json(DOC_GRADE_PROMPT.format(query=query, chunks=json.dumps(rows, ensure_ascii=True)))
        grades = raw.get("grades", []) if isinstance(raw, dict) else []
        relevant_ids = [int(row["id"]) for row in grades if row.get("relevant") is True and isinstance(row.get("id"), int)]
        if not relevant_ids:
            # fallback heuristic: keep top 3 by score
            return list(range(min(3, len(chunks))))
        return [idx - 1 for idx in relevant_ids if 1 <= idx <= len(chunks)]

    def run(self, query: str, k: int | None = None) -> AgentResult:
        k = k or SETTINGS.default_top_k
        query_type = self.classify_query(query)

        trace: list[AgentTraceStep] = [
            AgentTraceStep(step="classify", details={"query_type": query_type})
        ]

        ordered_retrievers = self._primary_retriever_order(query_type)
        best_chunks: list[RetrievedChunk] = []

        iteration = 0
        for iteration in range(1, self.max_iterations + 1):
            retriever_name = ordered_retrievers[min(iteration - 1, len(ordered_retrievers) - 1)]
            retriever = self.retrievers[retriever_name]
            chunks = retriever.retrieve(query, k=max(k, 5))
            trace.append(
                AgentTraceStep(
                    step="retrieve",
                    details={
                        "iteration": iteration,
                        "retriever": retriever_name,
                        "n_chunks": len(chunks),
                    },
                )
            )

            relevant_idx = self._grade_chunks(query, chunks)
            filtered = [chunks[idx] for idx in relevant_idx if 0 <= idx < len(chunks)]
            trace.append(
                AgentTraceStep(
                    step="grade_docs",
                    details={
                        "iteration": iteration,
                        "relevant_count": len(filtered),
                    },
                )
            )

            if len(filtered) >= min(2, k):
                best_chunks = filtered[:k]
                break

            if len(chunks) > len(best_chunks):
                best_chunks = chunks[:k]

        if not best_chunks:
            best_chunks = self.retrievers["vector"].retrieve(query, k=k)

        generation = generate_answer(
            query=query,
            chunks=best_chunks,
            model=SETTINGS.generator_model,
            temperature=SETTINGS.generation_temperature,
        )

        trace.append(
            AgentTraceStep(
                step="generate",
                details={
                    "iterations": iteration,
                    "citation_count": len(generation.citations),
                },
            )
        )

        return AgentResult(
            query=query,
            query_type=query_type,
            answer=generation.answer,
            citations=generation.citations,
            chunks=best_chunks,
            iterations=iteration,
            trace=trace,
        )


def build_default_agent(store, graph, partition, summaries, model: str | None = None) -> AgenticGraphRAG:
    """Build default retriever bundle for agentic GraphRAG."""

    retrievers: dict[str, BaseRetriever] = {
        "vector": DenseVectorRetriever(store=store, model=model),
        "graph_local": GraphLocalRetriever(store=store, graph=graph, model=model),
        "graph_global": GraphGlobalRetriever(
            store=store,
            graph=graph,
            partition=partition,
            summaries=summaries,
            model=model,
        ),
        "hybrid": HybridRetriever(
            store=store,
            graph=graph,
            partition=partition,
            summaries=summaries,
            model=model,
        ),
    }
    return AgenticGraphRAG(retrievers=retrievers)
