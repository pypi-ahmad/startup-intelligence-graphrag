#!/usr/bin/env python3
"""Optional domain-adapter CLI (Unsloth + PEFT + TRL stage).

This script is intentionally separate from the default pipeline entrypoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import chunk_corpus, save_chunks
from src.config import ARTIFACTS_DIR, EVAL_DIR, GENERATIONS_DIR, SETTINGS, ensure_dirs
from src.eval_queries import load_eval_queries
from src.evaluator import save_metrics
from src.extensions.domain_adapter import (
    build_eval_rows_from_retriever,
    evaluate_base_vs_adapter,
    seed_domain_adapter_placeholders,
    train_domain_adapter_from_chunks,
)
from src.extractor import extract_from_filings
from src.graph import build_graph, detect_communities, summarise_all_communities
from src.ingest import build_corpus
from src.retrievers import HybridRetriever
from src.vectorstore import build_from_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional domain adapter runner")
    parser.add_argument(
        "--mode",
        choices=["placeholder", "execute"],
        default="placeholder",
        help="placeholder seeds adapter placeholders; execute runs adapter train+eval",
    )
    parser.add_argument(
        "--n-companies",
        type=int,
        default=None,
        help="override default number of companies for corpus build",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR / "adapter",
        help="directory for adapter checkpoints and adapter-stage outputs",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="override adapter base model",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="top-k contexts for benchmark prompt construction",
    )
    parser.add_argument(
        "--merge-adapter",
        action="store_true",
        help="merge adapter into base model during adapter evaluation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="run even if SIRAG_ADAPTER_ENABLE=false",
    )
    parser.add_argument(
        "--required",
        action="store_true",
        help="exit non-zero unless adapter stage completes successfully",
    )
    return parser.parse_args()


def run_placeholder_mode() -> dict[str, str]:
    ensure_dirs()
    return seed_domain_adapter_placeholders()


def run_execute_mode(args: argparse.Namespace) -> dict[str, object]:
    ensure_dirs()
    if not SETTINGS.adapter_enable and not args.force:
        return {
            "status": "skipped",
            "message": (
                "Adapter stage is disabled by configuration. "
                "Set SIRAG_ADAPTER_ENABLE=true or pass --force."
            ),
        }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    n_companies = args.n_companies or SETTINGS.default_n_companies
    queries = load_eval_queries()

    logger.info("Building corpus for domain-adapter stage (n_companies={})", n_companies)
    filings = build_corpus(n_companies=n_companies, force_download=False)
    chunks = chunk_corpus(filings)
    save_chunks(chunks)

    logger.info("Building retrieval stack for evaluation prompt construction")
    store = build_from_chunks(chunks)
    extractions = extract_from_filings(filings)
    graph = build_graph(filings, extractions)
    partition = detect_communities(graph)
    summaries = summarise_all_communities(graph, partition)
    retriever = HybridRetriever(store, graph, partition, summaries)

    train_summary = train_domain_adapter_from_chunks(
        chunks=chunks,
        output_dir=output_dir,
        base_model=args.base_model or SETTINGS.adapter_base_model,
    )
    save_metrics(train_summary.to_dict(), EVAL_DIR / "domain_adapter_training_summary.json")

    if train_summary.status != "trained":
        logger.warning("Adapter stage did not train. Status={}", train_summary.status)
        return {
            "status": train_summary.status,
            "message": train_summary.message,
            "training": train_summary.to_dict(),
        }

    k = args.k or SETTINGS.default_top_k
    eval_rows = build_eval_rows_from_retriever(queries=queries, retriever=retriever, k=k)
    save_metrics(
        {"rows": [row.to_dict() for row in eval_rows]},
        EVAL_DIR / "domain_adapter_eval_rows.json",
    )

    benchmark = evaluate_base_vs_adapter(
        eval_rows=eval_rows,
        base_model=args.base_model or SETTINGS.adapter_base_model,
        adapter_path=Path(train_summary.payload["adapter_dir"]),
        judge_model=SETTINGS.extension_judge_model,
        max_new_tokens=SETTINGS.adapter_max_new_tokens,
        merge_adapter=args.merge_adapter,
    )

    save_metrics(benchmark["generation"], EVAL_DIR / "domain_adapter_generation_comparison.json")
    save_metrics(benchmark["rag"], EVAL_DIR / "domain_adapter_rag_comparison.json")
    save_metrics(benchmark["judge"], EVAL_DIR / "domain_adapter_judge_comparison.json")
    save_metrics(benchmark["latency"], EVAL_DIR / "domain_adapter_latency.json")
    save_metrics({"examples": benchmark["examples"]}, GENERATIONS_DIR / "domain_adapter_examples.json")

    summary = {
        "status": benchmark["status"],
        "n_queries": benchmark["n_queries"],
        "base_model": benchmark["base_model"],
        "adapter_path": benchmark["adapter_path"],
        "judge_model": benchmark["judge_model"],
    }
    save_metrics(summary, ARTIFACTS_DIR / "run_summary_domain_adapter.json")
    return summary


def main() -> None:
    args = parse_args()
    if args.mode == "placeholder":
        print(json.dumps(run_placeholder_mode(), indent=2))
        return
    result = run_execute_mode(args)
    print(json.dumps(result, indent=2))

    status = str(result.get("status", "unknown"))
    if status == "failed":
        raise SystemExit(1)
    if args.required and status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
