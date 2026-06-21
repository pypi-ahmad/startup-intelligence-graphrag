"""Knowledge graph construction and community analytics for SEC filing intelligence."""

from __future__ import annotations

import json
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from loguru import logger

try:
    import community as community_louvain
except Exception:  # pragma: no cover - optional dependency behavior
    community_louvain = None

from src.config import GRAPH_DIR, SETTINGS, ensure_dirs
from src.extractor import FilingExtraction
from src.ollama_client import get_client


@dataclass
class CommunitySummary:
    community_id: int
    size: int
    member_entities: list[str]
    member_tickers: list[str]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _company_node(ticker: str) -> str:
    return f"company::{ticker.upper()}"


def _filing_node(filing_id: str) -> str:
    return f"filing::{filing_id}"


def _section_node(filing_id: str, section: str) -> str:
    return f"section::{filing_id}::{_slug(section)}"


def _entity_node(name: str) -> str:
    return f"entity::{_slug(name)}"


def _entity_display(node_id: str) -> str:
    return node_id.replace("entity::", "").replace("_", " ").title()


def build_graph(filings: list, extractions: dict[str, FilingExtraction]) -> nx.Graph:
    """Build multi-layer graph: company, filing, section, and extracted entities."""

    graph = nx.Graph()

    # Structural layers from filings.
    filing_lookup = {f.filing_id: f for f in filings}
    for filing in filings:
        company_id = _company_node(filing.ticker)
        filing_id = _filing_node(filing.filing_id)

        graph.add_node(
            company_id,
            node_type="company",
            ticker=filing.ticker,
            name=filing.company_name,
        )
        graph.add_node(
            filing_id,
            node_type="filing",
            filing_id=filing.filing_id,
            ticker=filing.ticker,
            company_name=filing.company_name,
            filing_date=filing.filing_date,
            report_date=filing.report_date,
            form=filing.form,
        )
        graph.add_edge(company_id, filing_id, edge_type="filed", weight=1.0)

        for section_name, section_sentences in filing.sections.items():
            section_id = _section_node(filing.filing_id, section_name)
            graph.add_node(
                section_id,
                node_type="section",
                filing_id=filing.filing_id,
                ticker=filing.ticker,
                section=section_name,
                sentence_count=len(section_sentences),
            )
            graph.add_edge(filing_id, section_id, edge_type="contains", weight=1.0)

    # Extracted entity and relationship layer.
    for filing_id, extraction in extractions.items():
        if filing_id not in filing_lookup:
            continue

        filing = filing_lookup[filing_id]
        filing_node = _filing_node(filing_id)

        for entity in extraction.entities:
            entity_id = _entity_node(entity.name)
            if not graph.has_node(entity_id):
                graph.add_node(
                    entity_id,
                    node_type="entity",
                    entity_type=entity.type,
                    name=entity.name,
                    description=entity.description,
                    tickers=[filing.ticker],
                )
            else:
                existing_tickers = set(graph.nodes[entity_id].get("tickers", []))
                existing_tickers.add(filing.ticker)
                graph.nodes[entity_id]["tickers"] = sorted(existing_tickers)
                if not graph.nodes[entity_id].get("description"):
                    graph.nodes[entity_id]["description"] = entity.description

            if graph.has_edge(filing_node, entity_id):
                graph[filing_node][entity_id]["weight"] += 1.0
            else:
                graph.add_edge(filing_node, entity_id, edge_type="mentions_entity", weight=1.0)

            # Section-level provenance: attach entity to first matching section.
            entity_name = entity.name.lower()
            linked = False
            for section_name, sentences in filing.sections.items():
                if any(entity_name in sentence.lower() for sentence in sentences):
                    section_id = _section_node(filing_id, section_name)
                    if graph.has_edge(section_id, entity_id):
                        graph[section_id][entity_id]["weight"] += 1.0
                    else:
                        graph.add_edge(section_id, entity_id, edge_type="mentions", weight=1.0)
                    linked = True
                    break

            if not linked:
                # Fallback provenance edge keeps the graph connected.
                section_names = list(filing.sections.keys())
                if section_names:
                    fallback_section = _section_node(filing_id, section_names[0])
                    if graph.has_edge(fallback_section, entity_id):
                        graph[fallback_section][entity_id]["weight"] += 0.5
                    else:
                        graph.add_edge(fallback_section, entity_id, edge_type="mentions", weight=0.5)

        for relation in extraction.relationships:
            source_id = _entity_node(relation.source)
            target_id = _entity_node(relation.target)
            if not graph.has_node(source_id) or not graph.has_node(target_id):
                continue

            if graph.has_edge(source_id, target_id):
                rels = graph[source_id][target_id].setdefault("relation_types", [])
                evs = graph[source_id][target_id].setdefault("evidences", [])
                rels.append(relation.type)
                evs.append(relation.evidence)
                graph[source_id][target_id]["weight"] = graph[source_id][target_id].get("weight", 1.0) + 1.0
            else:
                graph.add_edge(
                    source_id,
                    target_id,
                    edge_type="entity_relation",
                    relation_types=[relation.type],
                    evidences=[relation.evidence],
                    weight=1.0,
                )

        # Co-occurrence edges boost local retrieval expansion.
        entity_ids = [_entity_node(entity.name) for entity in extraction.entities]
        for idx, left in enumerate(entity_ids):
            for right in entity_ids[idx + 1 :]:
                if left == right:
                    continue
                if graph.has_edge(left, right):
                    graph[left][right]["co_occurrence"] = graph[left][right].get("co_occurrence", 0) + 1
                    graph[left][right]["weight"] = graph[left][right].get("weight", 1.0) + 0.2
                else:
                    graph.add_edge(
                        left,
                        right,
                        edge_type="co_occurs",
                        co_occurrence=1,
                        weight=0.2,
                    )

    logger.info("Built graph with {} nodes and {} edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def detect_communities(graph: nx.Graph, resolution: float | None = None) -> dict[str, int]:
    """Run Louvain over entity-only subgraph. Non-entity nodes get community -1."""

    resolution = resolution or SETTINGS.louvain_resolution

    entity_nodes = [
        node
        for node, attrs in graph.nodes(data=True)
        if attrs.get("node_type") == "entity"
    ]

    if not entity_nodes:
        return {node: -1 for node in graph.nodes()}

    entity_subgraph = graph.subgraph(entity_nodes).copy()

    if community_louvain is None:
        logger.warning("python-louvain not available; assigning a single community.")
        partition = {node: 0 for node in entity_subgraph.nodes()}
    else:
        partition = {}
        community_offset = 0
        for component_nodes in nx.connected_components(entity_subgraph):
            component = entity_subgraph.subgraph(component_nodes).copy()
            if component.number_of_nodes() == 1:
                only = next(iter(component.nodes()))
                partition[only] = community_offset
                community_offset += 1
                continue

            local = community_louvain.best_partition(
                component,
                weight="weight",
                resolution=resolution,
                random_state=SETTINGS.random_seed,
            )
            unique_local_ids = sorted(set(local.values()))
            id_map = {local_id: community_offset + pos for pos, local_id in enumerate(unique_local_ids)}
            for node, local_id in local.items():
                partition[node] = id_map[local_id]
            community_offset += len(unique_local_ids)

    full_partition = {node: -1 for node in graph.nodes()}
    full_partition.update(partition)
    return full_partition


COMMUNITY_SUMMARY_PROMPT = """You are a startup intelligence analyst.

Given this entity cluster from SEC filings, summarize what strategic theme it represents.
Focus on cross-company insights an investor/operator would care about.
Use 3-5 concise sentences.

Cluster:
{cluster}
"""


def _summarize_community_with_llm(cluster_text: str, model: str) -> str:
    if SETTINGS.ollama_host:
        os.environ.setdefault("OLLAMA_HOST", SETTINGS.ollama_host)

    try:
        client = get_client()
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": COMMUNITY_SUMMARY_PROMPT.format(cluster=cluster_text)}],
            options={"temperature": 0.2},
        )
        return response["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Community summary generation failed: {}", exc)
        return "Summary generation failed; inspect member entities directly."


def summarise_all_communities(
    graph: nx.Graph,
    partition: dict[str, int],
    model: str | None = None,
    min_size: int = 2,
    max_communities: int = 30,
) -> list[CommunitySummary]:
    """Generate summaries for largest entity communities."""

    model = model or SETTINGS.generator_model

    by_community: dict[int, list[str]] = defaultdict(list)
    for node, community_id in partition.items():
        if community_id < 0:
            continue
        if graph.nodes[node].get("node_type") != "entity":
            continue
        by_community[community_id].append(node)

    ranked_ids = sorted(by_community.keys(), key=lambda cid: len(by_community[cid]), reverse=True)
    ranked_ids = [cid for cid in ranked_ids if len(by_community[cid]) >= min_size][:max_communities]

    summaries: list[CommunitySummary] = []
    for community_id in ranked_ids:
        members = by_community[community_id]
        cluster_lines: list[str] = []
        tickers: set[str] = set()
        for node in members[:40]:
            attrs = graph.nodes[node]
            tickers.update(attrs.get("tickers", []))
            cluster_lines.append(
                f"- [{attrs.get('entity_type', 'entity')}] {attrs.get('name', _entity_display(node))}: "
                f"{attrs.get('description', '')[:140]}"
            )
        cluster_text = "\n".join(cluster_lines)

        summary_text = _summarize_community_with_llm(cluster_text, model=model)
        summaries.append(
            CommunitySummary(
                community_id=community_id,
                size=len(members),
                member_entities=members,
                member_tickers=sorted(tickers),
                summary=summary_text,
            )
        )

    return summaries


def graph_stats(graph: nx.Graph) -> dict[str, Any]:
    """Quick diagnostics for notebook/report output."""

    node_types = Counter(attrs.get("node_type", "unknown") for _, attrs in graph.nodes(data=True))
    degrees = [degree for _, degree in graph.degree()]

    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "node_types": dict(node_types),
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": int(max(degrees)) if degrees else 0,
        "connected_components": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
    }


def save_graph(
    graph: nx.Graph,
    partition: dict[str, int] | None = None,
    summaries: list[CommunitySummary] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Persist graph and optional community outputs."""

    ensure_dirs()
    output_dir = output_dir or GRAPH_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "graph.pkl", "wb") as file:
        pickle.dump(graph, file)

    if partition is not None:
        with open(output_dir / "partition.json", "w", encoding="utf-8") as file:
            json.dump({k: int(v) for k, v in partition.items()}, file, indent=2)

    if summaries is not None:
        with open(output_dir / "community_summaries.json", "w", encoding="utf-8") as file:
            json.dump([summary.to_dict() for summary in summaries], file, indent=2)

    return output_dir


def load_graph(output_dir: Path | None = None) -> tuple[nx.Graph, dict[str, int], list[CommunitySummary]]:
    """Load graph and optional partition/summaries if present."""

    output_dir = output_dir or GRAPH_DIR
    with open(output_dir / "graph.pkl", "rb") as file:
        graph = pickle.load(file)

    partition: dict[str, int] = {}
    partition_path = output_dir / "partition.json"
    if partition_path.exists():
        with open(partition_path, "r", encoding="utf-8") as file:
            partition = {k: int(v) for k, v in json.load(file).items()}

    summaries: list[CommunitySummary] = []
    summary_path = output_dir / "community_summaries.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as file:
            summaries = [CommunitySummary(**row) for row in json.load(file)]

    return graph, partition, summaries
