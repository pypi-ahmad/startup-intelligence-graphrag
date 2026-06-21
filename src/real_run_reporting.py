"""Reporting helpers for real execution outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx

from src.config import ARTIFACTS_DIR, EVAL_DIR, FIGURES_DIR, GENERATIONS_DIR, RETRIEVALS_DIR, ensure_dirs


def save_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _normalize_generation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    payload = dict(metrics)
    if "em" in payload and "exact_match" not in payload:
        payload["exact_match"] = payload["em"]
    return payload


def write_technique_metric_files(
    technique: str,
    report: dict[str, Any],
) -> None:
    """Write full metric bundle and compatibility metric files."""

    ensure_dirs()
    save_json(EVAL_DIR / f"{technique}_full_metrics.json", report)

    retrieval = report.get("retrieval_metrics", {})
    generation = _normalize_generation_metrics(report.get("generation_metrics", {}))
    rag_metrics = report.get("rag_metrics", {})
    judge = report.get("judge_metrics", {})

    save_json(
        EVAL_DIR / f"{technique}_retrieval_metrics_placeholder.json",
        {"status": "executed", "technique": technique, "metrics": retrieval},
    )
    save_json(
        EVAL_DIR / f"{technique}_generation_metrics_placeholder.json",
        {"status": "executed", "technique": technique, "metrics": generation},
    )
    save_json(
        EVAL_DIR / f"{technique}_rag_metrics_placeholder.json",
        {"status": "executed", "technique": technique, "metrics": rag_metrics},
    )
    save_json(
        EVAL_DIR / f"{technique}_judge_placeholder.json",
        {"status": "executed", "technique": technique, "metrics": judge},
    )


def write_sample_files(
    technique: str,
    retrieval_samples: list[dict[str, Any]],
    generation_samples: list[dict[str, Any]],
) -> None:
    save_json(
        RETRIEVALS_DIR / f"{technique}_retrieval_samples_placeholder.json",
        {"status": "executed", "technique": technique, "samples": retrieval_samples},
    )
    save_json(
        GENERATIONS_DIR / f"{technique}_generation_samples_placeholder.json",
        {"status": "executed", "technique": technique, "samples": generation_samples},
    )


def _bar_plot(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_retrieval_figure(metrics_by_name: dict[str, dict[str, Any]]) -> Path:
    labels = []
    values = []
    for name, payload in metrics_by_name.items():
        labels.append(name)
        values.append(float(payload.get("ndcg_at_k", 0.0)))
    output = FIGURES_DIR / "retrieval_ndcg_comparison.png"
    _bar_plot(labels, values, "Retrieval NDCG@K by Technique", "NDCG@K", output)
    return output


def make_generation_figure(metrics_by_name: dict[str, dict[str, Any]]) -> Path:
    labels = []
    values = []
    for name, payload in metrics_by_name.items():
        labels.append(name)
        values.append(float(payload.get("rouge_l", 0.0)))
    output = FIGURES_DIR / "generation_rougel_comparison.png"
    _bar_plot(labels, values, "Generation ROUGE-L by Technique", "ROUGE-L", output)
    return output


def make_judge_figure(scores_by_name: dict[str, float]) -> Path:
    labels = list(scores_by_name.keys())
    values = [float(scores_by_name[name]) for name in labels]
    output = FIGURES_DIR / "judge_overall_comparison.png"
    _bar_plot(labels, values, "LLM Judge Overall Score by Technique", "Overall (1-5)", output)
    return output


def make_graph_figure(graph: nx.Graph, max_nodes: int = 180) -> Path:
    """Render a compact graph topology snapshot."""
    sub_nodes = list(graph.nodes())[:max_nodes]
    subgraph = graph.subgraph(sub_nodes).copy()
    fig, ax = plt.subplots(figsize=(11, 8))
    pos = nx.spring_layout(subgraph, seed=42, k=0.45)
    nx.draw_networkx_nodes(subgraph, pos, node_size=40, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.25, width=0.4, ax=ax)
    ax.set_title("Graph Topology Snapshot")
    ax.axis("off")
    fig.tight_layout()
    output = FIGURES_DIR / "graph_topology_snapshot.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def write_root_metric_compat_files(
    retrieval_metrics: dict[str, Any],
    generation_metrics: dict[str, Any],
    judge_payload: dict[str, Any],
) -> None:
    """Populate existing root placeholder contracts with real values."""
    generation_payload = _normalize_generation_metrics(generation_metrics)
    save_json(
        EVAL_DIR / "retrieval_metrics_placeholder.json",
        {"status": "executed", "metrics": retrieval_metrics},
    )
    save_json(
        EVAL_DIR / "generation_metrics_placeholder.json",
        {"status": "executed", "metrics": generation_payload},
    )
    save_json(
        EVAL_DIR / "llm_judge_placeholder.json",
        {"status": "executed", "metrics": judge_payload},
    )
