"""End-to-end pipeline orchestration for Startup Intelligence GraphRAG."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.agent import build_default_agent
from src.chunking import chunk_corpus, save_chunks
from src.config import (
    ARCHIVE_DIR,
    ARTIFACTS_DIR,
    EVAL_DIR,
    FIGURES_DIR,
    GENERATIONS_DIR,
    RETRIEVALS_DIR,
    SETTINGS,
    as_dict,
    ensure_dirs,
)
from src.eval_queries import load_eval_queries, save_eval_queries
from src.evaluator import evaluate_generation, evaluate_retrieval, save_metrics
from src.extractor import extract_from_filings
from src.generation import answer_query
from src.graph import build_graph, detect_communities, save_graph, summarise_all_communities
from src.ingest import build_corpus
from src.judge import judge_answer
from src.retrievers import DenseVectorRetriever, GraphGlobalRetriever, GraphLocalRetriever, HybridRetriever
from src.vectorstore import build_from_chunks


@dataclass
class PipelineSummary:
    mode: str
    timestamp_utc: str
    settings: dict[str, Any]
    notes: list[str]


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def archive_existing_artifacts() -> Path:
    """Archive current artifact content before writing new placeholder outputs."""

    ensure_dirs()
    stamp = _timestamp()
    archive_target = ARCHIVE_DIR / f"run_{stamp}"
    archive_target.mkdir(parents=True, exist_ok=True)

    for child in ARTIFACTS_DIR.iterdir():
        if child.name == "archive":
            continue
        destination = archive_target / child.name
        shutil.move(str(child), str(destination))

    # Recreate base artifact directories after archive.
    ensure_dirs()
    logger.info("Archived existing artifacts to {}", archive_target)
    return archive_target


def seed_placeholder_artifacts() -> dict[str, Path]:
    """Write explicit placeholder files for notebook/README rendering."""

    ensure_dirs()

    placeholder_files: dict[str, tuple[Path, dict[str, Any]]] = {
        "retrieval_metrics": (
            EVAL_DIR / "retrieval_metrics_placeholder.json",
            {
                "status": "placeholder",
                "note": "Real retrieval metrics will be populated after execution phase.",
                "metrics": {
                    "precision_at_k": "TBD",
                    "recall_at_k": "TBD",
                    "f1_at_k": "TBD",
                    "mrr": "TBD",
                    "ndcg_at_k": "TBD",
                },
            },
        ),
        "generation_metrics": (
            EVAL_DIR / "generation_metrics_placeholder.json",
            {
                "status": "placeholder",
                "note": "Real generation metrics will be populated after execution phase.",
                "metrics": {
                    "exact_match": "TBD",
                    "bleu": "TBD",
                    "rouge_l": "TBD",
                    "meteor": "TBD",
                    "bert_score_f1": "TBD",
                },
            },
        ),
        "judge_metrics": (
            EVAL_DIR / "llm_judge_placeholder.json",
            {
                "status": "placeholder",
                "note": "LLM judge results will be populated after execution phase.",
                "metrics": {
                    "correctness": "TBD",
                    "relevance": "TBD",
                    "completeness": "TBD",
                    "groundedness": "TBD",
                    "hallucination_risk": "TBD",
                },
            },
        ),
        "retrieval_samples": (
            RETRIEVALS_DIR / "retrieval_samples_placeholder.json",
            {
                "status": "placeholder",
                "note": "Representative retrieval outputs will be filled after running retrieval experiments.",
                "samples": [],
            },
        ),
        "generation_samples": (
            GENERATIONS_DIR / "generation_samples_placeholder.json",
            {
                "status": "placeholder",
                "note": "Generated answers and citations will be filled after execution.",
                "samples": [],
            },
        ),
    }

    written_paths: dict[str, Path] = {}
    for key, (path, payload) in placeholder_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        written_paths[key] = path

    figure_manifest = FIGURES_DIR / "placeholder_manifest.md"
    figure_manifest.write_text(
        "# Placeholder Figure Manifest\n\n"
        "This project intentionally leaves figure outputs as placeholders until explicit execution.\n\n"
        "Planned figures:\n"
        "1. corpus_overview.png\n"
        "2. chunk_distribution.png\n"
        "3. graph_topology.png\n"
        "4. retrieval_metrics_bar.png\n"
        "5. generation_metrics_bar.png\n"
        "6. llm_judge_scores_bar.png\n",
        encoding="utf-8",
    )
    written_paths["figure_manifest"] = figure_manifest

    run_summary = PipelineSummary(
        mode="placeholder",
        timestamp_utc=datetime.utcnow().isoformat(),
        settings=as_dict(),
        notes=[
            "No ingestion, embedding, extraction, retrieval, generation, or evaluation executed.",
            "All outputs are placeholders by design.",
        ],
    )
    summary_path = ARTIFACTS_DIR / "run_summary_placeholder.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(asdict(run_summary), file, indent=2)
    written_paths["run_summary"] = summary_path

    return written_paths


def run_execution_pipeline(n_companies: int | None = None) -> dict[str, Any]:
    """Real execution path (for later explicit run request)."""

    n_companies = n_companies or SETTINGS.default_n_companies
    ensure_dirs()
    save_eval_queries()
    queries = load_eval_queries()

    filings = build_corpus(n_companies=n_companies, force_download=False)
    chunks = chunk_corpus(filings)
    save_chunks(chunks)

    store = build_from_chunks(chunks)
    extractions = extract_from_filings(filings)

    graph = build_graph(filings, extractions)
    partition = detect_communities(graph)
    summaries = summarise_all_communities(graph, partition)
    save_graph(graph, partition=partition, summaries=summaries)

    dense = DenseVectorRetriever(store)
    local = GraphLocalRetriever(store, graph)
    global_retriever = GraphGlobalRetriever(store, graph, partition, summaries)
    hybrid = HybridRetriever(store, graph, partition, summaries)

    retrieval_results = {
        "vector": evaluate_retrieval(queries, dense, metadata=store.metadata, k=SETTINGS.default_top_k).to_dict(),
        "graph_local": evaluate_retrieval(queries, local, metadata=store.metadata, k=SETTINGS.default_top_k).to_dict(),
        "graph_global": evaluate_retrieval(queries, global_retriever, metadata=store.metadata, k=SETTINGS.default_top_k).to_dict(),
        "hybrid": evaluate_retrieval(queries, hybrid, metadata=store.metadata, k=SETTINGS.default_top_k).to_dict(),
    }
    save_metrics(retrieval_results, EVAL_DIR / "retrieval_metrics.json")

    predictions: list[str] = []
    references: list[str] = []
    judge_rows: list[dict[str, Any]] = []

    for query in queries:
        question = query["query"]
        reference = query.get("reference_answer", "")
        generation, _ = answer_query(question, retriever=hybrid, k=SETTINGS.default_top_k)
        predictions.append(generation.answer)
        references.append(reference)

        judge = judge_answer(
            query=question,
            answer=generation.answer,
            contexts=generation.citations,
            reference=reference,
        )
        judge_rows.append(judge.to_dict())

    generation_metrics = evaluate_generation(predictions, references).to_dict()
    save_metrics(generation_metrics, EVAL_DIR / "generation_metrics.json")
    save_metrics({"rows": judge_rows}, EVAL_DIR / "llm_judge_metrics.json")

    agent = build_default_agent(store, graph, partition, summaries)
    demo_queries = [query["query"] for query in queries[:3]]
    demo_runs = [agent.run(question).to_dict() for question in demo_queries]
    save_metrics({"demo_runs": demo_runs}, GENERATIONS_DIR / "agentic_demo_runs.json")

    summary = {
        "mode": "execute",
        "timestamp_utc": datetime.utcnow().isoformat(),
        "n_filings": len(filings),
        "n_chunks": len(chunks),
        "n_queries": len(queries),
    }
    save_metrics(summary, ARTIFACTS_DIR / "run_summary.json")
    return summary
