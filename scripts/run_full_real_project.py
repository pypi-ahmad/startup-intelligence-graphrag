#!/usr/bin/env python3
"""Run full real end-to-end project execution under strict llama-duo policy."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import build_default_agent
from src.chunking import chunk_corpus, save_chunks
from src.config import (
    ARTIFACTS_DIR,
    EVAL_DIR,
    EVAL_QUERIES_PATH,
    GENERATIONS_DIR,
    RAW_DIR,
    RETRIEVALS_DIR,
    SETTINGS,
    as_dict,
    ensure_dirs,
)
from src.eval_query_builder import build_eval_queries_from_filings, save_eval_queries
from src.evaluator import evaluate_generation, save_metrics
from src.extensions.benchmark import evaluate_retriever_end_to_end
from src.extensions.hybrid_sparse_dense import HybridSparseDenseRetriever
from src.extensions.multimodal import build_multimodal_units_from_html_map
from src.extensions.multimodal_ocr import build_ocr_units_from_image_map
from src.extensions.multimodal_retriever import MultimodalRetriever
from src.extensions.multimodal_v2 import build_multimodal_units_v2
from src.extensions.multimodal_vision import build_vision_units_from_image_map
from src.extensions.rag_metrics import evaluate_rag_quality
from src.extractor import extract_from_filings
from src.graph import CommunitySummary, build_graph, detect_communities, save_graph
from src.ingest import build_corpus
from src.judge import judge_answer
from src.multimodal_assets import (
    build_table_image_sources,
    build_text_snapshot_images,
    fetch_filing_html_map,
    save_html_map_manifest,
)
from src.real_run_reporting import (
    make_generation_figure,
    make_graph_figure,
    make_judge_figure,
    make_retrieval_figure,
    save_json,
    write_root_metric_compat_files,
    write_sample_files,
    write_technique_metric_files,
)
from src.retrievers import DenseVectorRetriever, GraphGlobalRetriever, GraphLocalRetriever, HybridRetriever
from src.vectorstore import build_from_chunks


@dataclass
class TechniqueRun:
    name: str
    report: dict[str, Any]
    elapsed_seconds: float


class AgentRetrieverAdapter:
    """Adapter so agentic retrieval can be evaluated with retrieval metrics."""

    name = "agentic_crag"

    def __init__(self, agent):
        self.agent = agent

    def retrieve(self, query: str, k: int = 5):
        return self.agent.run(query, k=k).chunks


def _build_fast_community_summaries(
    graph,
    partition: dict[str, int],
    max_communities: int = 6,
) -> list[CommunitySummary]:
    """Create deterministic community summaries without LLM calls."""

    buckets: dict[int, list[str]] = {}
    for node, cid in partition.items():
        if cid < 0:
            continue
        attrs = graph.nodes[node]
        if attrs.get("node_type") != "entity":
            continue
        buckets.setdefault(cid, []).append(node)

    ranked = sorted(buckets.items(), key=lambda x: len(x[1]), reverse=True)[:max_communities]
    summaries: list[CommunitySummary] = []
    for cid, nodes in ranked:
        labels = []
        tickers = set()
        for node in nodes[:12]:
            attrs = graph.nodes[node]
            labels.append(str(attrs.get("name", node)).strip())
            for ticker in attrs.get("tickers", []):
                tickers.add(ticker)
        summary = "Community theme entities: " + ", ".join(labels[:10])
        summaries.append(
            CommunitySummary(
                community_id=cid,
                size=len(nodes),
                member_entities=nodes,
                member_tickers=sorted(tickers),
                summary=summary,
            )
        )
    return summaries


def _query_subset(queries: list[dict[str, Any]], max_queries: int | None) -> list[dict[str, Any]]:
    if max_queries is None:
        return queries
    if len(queries) <= max_queries:
        return queries
    # keep a mix of local/global/factual
    local = [q for q in queries if q.get("query_type") == "local"]
    global_q = [q for q in queries if q.get("query_type") == "global"]
    factual = [q for q in queries if q.get("query_type") == "factual"]
    out: list[dict[str, Any]] = []
    for bucket in [local, global_q, factual]:
        for row in bucket:
            if len(out) >= max_queries:
                break
            out.append(row)
    if len(out) < max_queries:
        for row in queries:
            if row not in out:
                out.append(row)
                if len(out) >= max_queries:
                    break
    return out[:max_queries]


def _aggregate_judge_rows(judge_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not judge_rows:
        return {
            "overall": 0.0,
            "correctness": 0.0,
            "relevance": 0.0,
            "completeness": 0.0,
            "groundedness": 0.0,
            "hallucination_risk": 0.0,
        }
    fields = [
        "overall",
        "correctness",
        "relevance",
        "completeness",
        "groundedness",
        "hallucination_risk",
    ]
    return {
        key: round(statistics.mean(float(row.get(key, 0.0)) for row in judge_rows), 4)
        for key in fields
    }


def _metadata_from_units(units: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in units:
        rows.append(
            {
                "chunk_id": unit.unit_id,
                "filing_id": unit.filing_id,
                "ticker": unit.ticker,
                "company_name": unit.company_name,
                "section": unit.section,
                "text": unit.text,
            }
        )
    return rows


def _extract_samples(report: dict[str, Any], queries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retrieval_rows = report.get("retrieval_metrics", {}).get("query_level", [])
    judge_rows = report.get("judge_metrics", {}).get("rows", [])
    retrieval_samples: list[dict[str, Any]] = []
    generation_samples: list[dict[str, Any]] = []

    q_by_id = {q["query_id"]: q for q in queries}
    for row in retrieval_rows[:3]:
        retrieval_samples.append(
            {
                "query_id": row.get("query_id"),
                "query": q_by_id.get(row.get("query_id"), {}).get("query", row.get("query")),
                "retrieved_ids": row.get("retrieved_ids", [])[:6],
                "precision": row.get("precision"),
                "recall": row.get("recall"),
                "ndcg": row.get("ndcg"),
            }
        )

    for row in judge_rows[:3]:
        generation_samples.append(
            {
                "query": row.get("query"),
                "answer": row.get("answer", "")[:1200],
                "correctness": row.get("correctness"),
                "relevance": row.get("relevance"),
                "groundedness": row.get("groundedness"),
                "overall": row.get("overall"),
            }
        )
    return retrieval_samples, generation_samples


def evaluate_agentic_end_to_end(
    agent,
    queries: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    """Run retrieval/generation/RAG/judge metrics on true agentic path."""
    adapter = AgentRetrieverAdapter(agent)
    retrieval_metrics = evaluate_retrieval(queries, adapter, metadata=metadata, k=k).to_dict()

    answers: list[str] = []
    refs: list[str] = []
    contexts: list[list[str]] = []
    q_texts: list[str] = []
    judge_rows: list[dict[str, Any]] = []
    latencies: list[float] = []

    for row in queries:
        q = row["query"]
        ref = row.get("reference_answer", "")
        start = time.perf_counter()
        result = agent.run(q, k=k)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        answers.append(result.answer)
        refs.append(ref)
        contexts.append([c.get("text_preview", "") for c in result.citations])
        q_texts.append(q)

        judge = judge_answer(
            query=q,
            answer=result.answer,
            contexts=result.citations,
            reference=ref,
            model=SETTINGS.judge_model,
        )
        judge_rows.append(judge.to_dict())

    generation_metrics = evaluate_generation(answers, refs).to_dict()
    rag_metrics = evaluate_rag_quality(
        predictions=answers,
        contexts=contexts,
        queries=q_texts,
        references=refs,
        model=SETTINGS.extension_judge_model,
    )
    return {
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "rag_metrics": rag_metrics,
        "judge_metrics": {
            "rows": judge_rows,
            "model": SETTINGS.judge_model,
            "aggregate": _aggregate_judge_rows(judge_rows),
        },
        "latency_seconds": {
            "mean": round(statistics.mean(latencies), 4) if latencies else 0.0,
            "p95": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 4) if latencies else 0.0,
        },
    }


def _run_technique(
    name: str,
    retriever,
    queries: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    k: int,
) -> TechniqueRun:
    start = time.perf_counter()
    report = evaluate_retriever_end_to_end(
        retriever=retriever,
        queries=queries,
        metadata=metadata,
        k=k,
        judge_fn=judge_answer,
        judge_model=SETTINGS.extension_judge_model,
    ).to_dict()
    elapsed = time.perf_counter() - start
    report["latency_seconds"] = {"total": round(elapsed, 4)}
    judge_rows = report.get("judge_metrics", {}).get("rows", [])
    report.setdefault("judge_metrics", {})
    report["judge_metrics"]["aggregate"] = _aggregate_judge_rows(judge_rows)
    return TechniqueRun(name=name, report=report, elapsed_seconds=elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full real end-to-end project run")
    parser.add_argument("--n-companies", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-eval-queries", type=int, default=None)
    parser.add_argument("--max-local-eval-companies", type=int, default=10)
    parser.add_argument("--max-multimodal-filings", type=int, default=4)
    parser.add_argument("--max-tables-per-filing", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=450)
    parser.add_argument("--chunk-overlap", type=int, default=60)
    parser.add_argument("--max-community-summaries", type=int, default=6)
    parser.add_argument("--archive-existing", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    run_started = datetime.now(timezone.utc)
    logger.info("Starting full real project run")

    filings = build_corpus(
        n_companies=args.n_companies,
        force_download=args.force_download,
    )
    logger.info("Corpus ready with {} filings", len(filings))

    schema: dict[str, Any] = {}
    manifest_path = RAW_DIR / "manifest.json"
    if manifest_path.exists():
        schema = json.loads(manifest_path.read_text(encoding="utf-8"))

    filing_url_map: dict[str, str] = {}
    url_map_path = RAW_DIR / "filing_url_map.json"
    if url_map_path.exists():
        try:
            raw_map = json.loads(url_map_path.read_text(encoding="utf-8"))
            filing_ids = {f.filing_id for f in filings}
            filing_url_map = {fid: url for fid, url in raw_map.items() if fid in filing_ids}
        except json.JSONDecodeError:
            filing_url_map = {}

    queries = build_eval_queries_from_filings(
        filings,
        max_local_companies=args.max_local_eval_companies,
    )
    queries = _query_subset(queries, max_queries=args.max_eval_queries)
    eval_path = save_eval_queries(EVAL_QUERIES_PATH, queries)
    logger.info("Saved {} eval queries to {}", len(queries), eval_path)

    chunks = chunk_corpus(
        filings,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    save_chunks(chunks)
    store = build_from_chunks(chunks)

    extractions = extract_from_filings(filings)
    graph = build_graph(filings, extractions)
    partition = detect_communities(graph)
    summaries = _build_fast_community_summaries(
        graph,
        partition,
        max_communities=args.max_community_summaries,
    )
    save_graph(graph, partition=partition, summaries=summaries)

    dense = DenseVectorRetriever(store)
    graph_local = GraphLocalRetriever(store, graph)
    graph_global = GraphGlobalRetriever(store, graph, partition, summaries)
    graphrag_hybrid = HybridRetriever(store, graph, partition, summaries)
    sparse_dense_hybrid = HybridSparseDenseRetriever(store)
    agent = build_default_agent(store, graph, partition, summaries)

    technique_runs: list[TechniqueRun] = []
    technique_runs.append(_run_technique("vector_baseline", dense, queries, store.metadata, args.top_k))
    technique_runs.append(_run_technique("graphrag_local", graph_local, queries, store.metadata, args.top_k))
    technique_runs.append(_run_technique("graphrag_global", graph_global, queries, store.metadata, args.top_k))
    technique_runs.append(_run_technique("graphrag_hybrid", graphrag_hybrid, queries, store.metadata, args.top_k))
    technique_runs.append(_run_technique("hybrid_sparse_dense", sparse_dense_hybrid, queries, store.metadata, args.top_k))

    agent_start = time.perf_counter()
    agentic_report = evaluate_agentic_end_to_end(
        agent=agent,
        queries=queries,
        metadata=store.metadata,
        k=args.top_k,
    )
    technique_runs.append(
        TechniqueRun(
            name="agentic_crag",
            report=agentic_report,
            elapsed_seconds=time.perf_counter() - agent_start,
        )
    )

    filing_html_map = fetch_filing_html_map(
        filings=filings,
        filing_url_map=filing_url_map,
        max_filings=args.max_multimodal_filings,
    )
    save_html_map_manifest(filing_html_map)

    multimodal_units = build_multimodal_units_from_html_map(filings, filing_html_map)
    multimodal_retriever = MultimodalRetriever(multimodal_units)
    technique_runs.append(
        _run_technique(
            "multimodal_rag",
            multimodal_retriever,
            queries,
            _metadata_from_units(multimodal_units),
            args.top_k,
        )
    )

    image_map = build_table_image_sources(
        filings=filings,
        filing_html_map=filing_html_map,
        max_tables_per_filing=args.max_tables_per_filing,
        max_filings=args.max_multimodal_filings,
    )
    if not image_map:
        logger.warning(
            "No table-image assets were generated from HTML. Falling back to filing text snapshot images."
        )
        image_map = build_text_snapshot_images(
            filings=filings,
            max_filings=args.max_multimodal_filings,
        )

    ocr_units = build_ocr_units_from_image_map(filings, image_map)
    ocr_retriever = MultimodalRetriever(ocr_units) if ocr_units else None
    if ocr_retriever is not None:
        technique_runs.append(
            _run_technique(
                "multimodal_ocr_rag",
                ocr_retriever,
                queries,
                _metadata_from_units(ocr_units),
                args.top_k,
            )
        )

    vision_units = build_vision_units_from_image_map(filings, image_map)
    vision_retriever = MultimodalRetriever(vision_units) if vision_units else None
    if vision_retriever is not None:
        technique_runs.append(
            _run_technique(
                "multimodal_vision_rag",
                vision_retriever,
                queries,
                _metadata_from_units(vision_units),
                args.top_k,
            )
        )

    multimodal_v2_units = build_multimodal_units_v2(
        filings=filings,
        filing_html_sources=filing_html_map,
        filing_image_sources=image_map,
        include_html_channels=True,
        include_ocr_channels=bool(ocr_units),
        include_vision_channels=bool(vision_units),
    )
    if multimodal_v2_units:
        unified_retriever = MultimodalRetriever(multimodal_v2_units)
        technique_runs.append(
            _run_technique(
                "multimodal_unified_v2",
                unified_retriever,
                queries,
                _metadata_from_units(multimodal_v2_units),
                args.top_k,
            )
        )

    retrieval_fig_input: dict[str, dict[str, Any]] = {}
    generation_fig_input: dict[str, dict[str, Any]] = {}
    judge_fig_input: dict[str, float] = {}

    for run in technique_runs:
        write_technique_metric_files(run.name, run.report)
        retrieval_samples, generation_samples = _extract_samples(run.report, queries)
        write_sample_files(run.name, retrieval_samples, generation_samples)

        retrieval_fig_input[run.name] = run.report.get("retrieval_metrics", {})
        generation_fig_input[run.name] = run.report.get("generation_metrics", {})
        judge_rows = run.report.get("judge_metrics", {}).get("rows", [])
        if judge_rows:
            judge_fig_input[run.name] = round(
                statistics.mean(float(row.get("overall", 0.0)) for row in judge_rows),
                4,
            )
        else:
            judge_fig_input[run.name] = 0.0

    retrieval_fig = make_retrieval_figure(retrieval_fig_input)
    generation_fig = make_generation_figure(generation_fig_input)
    judge_fig = make_judge_figure(judge_fig_input)
    graph_fig = make_graph_figure(graph)

    root_retrieval = next((r.report["retrieval_metrics"] for r in technique_runs if r.name == "graphrag_hybrid"), {})
    root_generation = next((r.report["generation_metrics"] for r in technique_runs if r.name == "graphrag_hybrid"), {})
    root_judge = next((r.report["judge_metrics"] for r in technique_runs if r.name == "graphrag_hybrid"), {})
    write_root_metric_compat_files(root_retrieval, root_generation, root_judge)

    retrieval_sample_path = RETRIEVALS_DIR / "retrieval_samples_placeholder.json"
    generation_sample_path = GENERATIONS_DIR / "generation_samples_placeholder.json"
    hybrid_retrieval_samples = json.loads(
        (RETRIEVALS_DIR / "graphrag_hybrid_retrieval_samples_placeholder.json").read_text(encoding="utf-8")
    ).get("samples", [])
    hybrid_generation_samples = json.loads(
        (GENERATIONS_DIR / "graphrag_hybrid_generation_samples_placeholder.json").read_text(encoding="utf-8")
    ).get("samples", [])
    retrieval_sample_path.write_text(
        json.dumps(
            {
                "status": "executed",
                "technique": "graphrag_hybrid",
                "samples": hybrid_retrieval_samples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    generation_sample_path.write_text(
        json.dumps(
            {
                "status": "executed",
                "technique": "graphrag_hybrid",
                "samples": hybrid_generation_samples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ended = datetime.now(timezone.utc)
    summary = {
        "mode": "execute_real",
        "timestamp_utc": ended.isoformat(),
        "started_utc": run_started.isoformat(),
        "duration_seconds": round((ended - run_started).total_seconds(), 2),
        "dataset_repo": SETTINGS.dataset_repo,
        "strict_dataset": SETTINGS.strict_dataset,
        "n_filings": len(filings),
        "n_chunks": len(chunks),
        "n_eval_queries": len(queries),
        "n_graph_nodes": graph.number_of_nodes(),
        "n_graph_edges": graph.number_of_edges(),
        "n_multimodal_units": len(multimodal_units),
        "n_ocr_units": len(ocr_units),
        "n_vision_units": len(vision_units),
        "settings": as_dict(),
        "schema_snapshot": schema,
        "techniques": [
            {
                "name": run.name,
                "elapsed_seconds": round(run.elapsed_seconds, 2),
                "retrieval_metrics": run.report.get("retrieval_metrics", {}),
                "generation_metrics": run.report.get("generation_metrics", {}),
                "rag_metrics": run.report.get("rag_metrics", {}),
                "judge_metrics": run.report.get("judge_metrics", {}),
            }
            for run in technique_runs
        ],
        "figures": [
            str(retrieval_fig),
            str(generation_fig),
            str(judge_fig),
            str(graph_fig),
        ],
    }
    save_metrics(summary, ARTIFACTS_DIR / "run_summary.json")
    save_json(ARTIFACTS_DIR / "run_summary_placeholder.json", summary)
    save_json(ARTIFACTS_DIR / "run_summary_extensions_placeholder.json", summary)

    logger.info("Full real run completed in {:.1f}s", summary["duration_seconds"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
