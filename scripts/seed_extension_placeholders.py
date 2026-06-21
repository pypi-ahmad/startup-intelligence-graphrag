#!/usr/bin/env python3
"""Seed placeholder artifacts for additive extension techniques.

This script is intentionally separate from the main pipeline. It is not run
automatically in this phase.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def seed_domain_adapter_placeholders(root: Path) -> None:
    eval_dir = root / "artifacts" / "eval"
    generations_dir = root / "artifacts" / "generations"
    figures_dir = root / "artifacts" / "figures"

    write_json(
        eval_dir / "domain_adapter_training_placeholder.json",
        {
            "status": "placeholder",
            "note": "Populate after explicit adapter training run.",
            "metrics": {
                "train_loss": "TBD",
                "eval_loss": "TBD",
                "steps": "TBD",
            },
        },
    )
    write_json(
        eval_dir / "domain_adapter_generation_comparison_placeholder.json",
        {
            "status": "placeholder",
            "note": "Populate after explicit base-vs-adapter benchmark run.",
            "metrics": {
                "base": {
                    "em": "TBD",
                    "bleu": "TBD",
                    "rouge_l": "TBD",
                    "meteor": "TBD",
                    "bert_score_f1": "TBD",
                },
                "adapter": {
                    "em": "TBD",
                    "bleu": "TBD",
                    "rouge_l": "TBD",
                    "meteor": "TBD",
                    "bert_score_f1": "TBD",
                },
                "delta": {
                    "em": "TBD",
                    "bleu": "TBD",
                    "rouge_l": "TBD",
                    "meteor": "TBD",
                    "bert_score_f1": "TBD",
                },
            },
        },
    )
    write_json(
        eval_dir / "domain_adapter_rag_comparison_placeholder.json",
        {
            "status": "placeholder",
            "note": "Populate after explicit base-vs-adapter benchmark run.",
            "metrics": {
                "base": {
                    "faithfulness": "TBD",
                    "context_precision": "TBD",
                    "context_recall": "TBD",
                    "answer_relevancy": "TBD",
                },
                "adapter": {
                    "faithfulness": "TBD",
                    "context_precision": "TBD",
                    "context_recall": "TBD",
                    "answer_relevancy": "TBD",
                },
            },
        },
    )
    write_json(
        eval_dir / "domain_adapter_judge_comparison_placeholder.json",
        {
            "status": "placeholder",
            "judge_model": "granite4.1:8b",
            "note": "Populate after explicit base-vs-adapter benchmark run.",
            "metrics": {
                "correctness": "TBD",
                "relevance": "TBD",
                "completeness": "TBD",
                "groundedness": "TBD",
                "hallucination_risk": "TBD",
            },
        },
    )
    write_json(
        eval_dir / "domain_adapter_latency_placeholder.json",
        {
            "status": "placeholder",
            "note": "Populate after explicit base-vs-adapter benchmark run.",
            "metrics": {
                "base_mean_seconds": "TBD",
                "adapter_mean_seconds": "TBD",
                "delta_seconds": "TBD",
            },
        },
    )
    write_json(
        generations_dir / "domain_adapter_samples_placeholder.json",
        {
            "status": "placeholder",
            "samples": [],
            "note": "Populate with base-vs-adapter output examples after execution.",
        },
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "domain_adapter_placeholder_manifest.md").write_text(
        "# Domain Adapter Placeholder Figure Manifest\n\n"
        "Planned figures for optional Unsloth+PEFT+TRL stage:\n"
        "1. domain_adapter_architecture.png\n"
        "2. domain_adapter_training_curve.png\n"
        "3. domain_adapter_generation_delta.png\n"
        "4. domain_adapter_latency_comparison.png\n",
        encoding="utf-8",
    )


def main() -> None:
    eval_dir = ROOT / "artifacts" / "eval"
    retrievals_dir = ROOT / "artifacts" / "retrievals"
    generations_dir = ROOT / "artifacts" / "generations"

    techniques = [
        "hybrid_sparse_dense",
        "multimodal_rag",
        "multimodal_ocr_rag",
        "multimodal_vision_rag",
    ]

    for tech in techniques:
        write_json(
            eval_dir / f"{tech}_retrieval_metrics_placeholder.json",
            {
                "status": "placeholder",
                "technique": tech,
                "metrics": {
                    "precision_at_k": "TBD",
                    "recall_at_k": "TBD",
                    "f1_at_k": "TBD",
                    "mrr": "TBD",
                    "ndcg_at_k": "TBD",
                },
                "note": "Populate after explicit execution.",
            },
        )
        write_json(
            eval_dir / f"{tech}_generation_metrics_placeholder.json",
            {
                "status": "placeholder",
                "technique": tech,
                "metrics": {
                    "exact_match": "TBD",
                    "bleu": "TBD",
                    "rouge_l": "TBD",
                    "meteor": "TBD",
                    "bert_score_f1": "TBD",
                },
                "note": "Populate after explicit execution.",
            },
        )
        write_json(
            eval_dir / f"{tech}_rag_metrics_placeholder.json",
            {
                "status": "placeholder",
                "technique": tech,
                "metrics": {
                    "faithfulness": "TBD",
                    "context_precision": "TBD",
                    "context_recall": "TBD",
                    "answer_relevancy": "TBD",
                },
                "note": "Populate after explicit execution.",
            },
        )
        write_json(
            eval_dir / f"{tech}_judge_placeholder.json",
            {
                "status": "placeholder",
                "technique": tech,
                "judge_model": "granite4.1:8b",
                "metrics": {
                    "correctness": "TBD",
                    "relevance": "TBD",
                    "completeness": "TBD",
                    "groundedness": "TBD",
                    "hallucination_risk": "TBD",
                },
                "note": "Populate after explicit execution.",
            },
        )

        write_json(
            retrievals_dir / f"{tech}_retrieval_samples_placeholder.json",
            {
                "status": "placeholder",
                "technique": tech,
                "samples": [],
            },
        )
        write_json(
            generations_dir / f"{tech}_generation_samples_placeholder.json",
            {
                "status": "placeholder",
                "technique": tech,
                "samples": [],
            },
        )

    # Optional advanced stage placeholders.
    seed_domain_adapter_placeholders(root=ROOT)


if __name__ == "__main__":
    main()
