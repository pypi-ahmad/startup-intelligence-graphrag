"""Benchmark helpers for additive extension techniques.

Provides side-by-side evaluation wrappers for sparse hybrid and multimodal flows
without changing existing pipeline code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evaluator import evaluate_generation, evaluate_retrieval
from src.extensions.rag_metrics import evaluate_rag_quality
from src.generation import answer_query


@dataclass
class ExtensionBenchmarkResult:
    retrieval_metrics: dict[str, Any]
    generation_metrics: dict[str, Any]
    rag_metrics: dict[str, Any]
    judge_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_metrics": self.retrieval_metrics,
            "generation_metrics": self.generation_metrics,
            "rag_metrics": self.rag_metrics,
            "judge_metrics": self.judge_metrics,
        }


def evaluate_retriever_end_to_end(
    retriever,
    queries: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    k: int,
    judge_fn,
    judge_model: str,
) -> ExtensionBenchmarkResult:
    """Run retrieval, generation, judge, and RAG metrics for any retriever.

    This function is designed for later execution. In the current implementation
    phase it is present as runnable infrastructure but should not be executed.
    """

    retrieval = evaluate_retrieval(queries, retriever, metadata=metadata, k=k).to_dict()

    predictions: list[str] = []
    references: list[str] = []
    context_texts: list[list[str]] = []
    query_texts: list[str] = []

    judge_rows: list[dict[str, Any]] = []

    for row in queries:
        q = row["query"]
        ref = row.get("reference_answer", "")

        generation, retrieved = answer_query(q, retriever=retriever, k=k)
        predictions.append(generation.answer)
        references.append(ref)
        context_texts.append([chunk.text for chunk in retrieved])
        query_texts.append(q)

        judge = judge_fn(
            query=q,
            answer=generation.answer,
            contexts=generation.citations,
            reference=ref,
            model=judge_model,
        )
        judge_rows.append(judge.to_dict())

    generation = evaluate_generation(predictions, references).to_dict()
    rag = evaluate_rag_quality(
        predictions=predictions,
        contexts=context_texts,
        queries=query_texts,
        references=references,
        model=judge_model,
    )

    return ExtensionBenchmarkResult(
        retrieval_metrics=retrieval,
        generation_metrics=generation,
        rag_metrics=rag,
        judge_metrics={"rows": judge_rows, "model": judge_model},
    )
