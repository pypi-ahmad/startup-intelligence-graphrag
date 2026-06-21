"""LLM-based extraction of entities and relationships from SEC filing text."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from src.config import GRAPH_DIR, SETTINGS, ensure_dirs
from src.ingest import Filing
from src.ollama_client import get_client


ENTITY_TYPES: tuple[str, ...] = (
    "company",
    "founder",
    "executive",
    "product",
    "competitor",
    "risk_factor",
    "financial_indicator",
    "strategic_signal",
)

RELATION_TYPES: tuple[str, ...] = (
    "competes_with",
    "founded_by",
    "led_by",
    "produces",
    "mentions_competitor",
    "exposed_to_risk",
    "reports_metric",
    "signals_strategy",
    "partners_with",
    "acquires",
)

EXTRACTION_PROMPT = """You are a senior financial intelligence analyst.

Extract high-value structured intelligence from the SEC filing excerpt.

Company context:
- Company: {company_name}
- Ticker: {ticker}

Allowed entity types:
{entity_types}

Allowed relationship types:
{relation_types}

Output strict JSON only with this schema:
{{
  "entities": [
    {{
      "name": "string",
      "type": "one of allowed entity types",
      "description": "brief filing-grounded description"
    }}
  ],
  "relationships": [
    {{
      "source": "entity name",
      "target": "entity name",
      "type": "one of allowed relationship types",
      "evidence": "direct quote <= 25 words"
    }}
  ]
}}

Rules:
- Keep only important startup/company intelligence signals.
- Maximum 18 entities, maximum 20 relationships.
- Every relationship source and target must exist in entities.
- If evidence is weak, skip that relationship.

Excerpt:
<<<SEC_EXCERPT>>>
{context}
<<<END_SEC_EXCERPT>>>
"""


@dataclass
class Entity:
    name: str
    type: str
    description: str

    def key(self) -> str:
        return " ".join(self.name.lower().split())


@dataclass
class Relationship:
    source: str
    target: str
    type: str
    evidence: str

    def key(self) -> str:
        s = " ".join(self.source.lower().split())
        t = " ".join(self.target.lower().split())
        return f"{s}||{self.type}||{t}"


@dataclass
class FilingExtraction:
    filing_id: str
    ticker: str
    company_name: str
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filing_id": self.filing_id,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "entities": [asdict(x) for x in self.entities],
            "relationships": [asdict(x) for x in self.relationships],
        }


def _cache_path(model: str) -> Path:
    safe = model.replace(":", "_").replace("/", "_")
    return GRAPH_DIR.parent / "_llm_cache" / f"extract_{safe}.json"


def _load_cache(model: str) -> dict[str, dict[str, Any]]:
    path = _cache_path(model)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return {}


def _save_cache(model: str, cache: dict[str, dict[str, Any]]) -> None:
    ensure_dirs()
    path = _cache_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(cache, file, indent=2)
    tmp.replace(path)


def _hash_payload(model: str, prompt: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(model.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(prompt.encode("utf-8"))
    return hasher.hexdigest()


def _build_context(filing: Filing, max_chars: int) -> str:
    priority = [
        "Business",
        "Risk Factors",
        "Management's Discussion and Analysis (MD&A)",
        "Legal Proceedings",
        "Directors, Officers and Compensation",
    ]

    blocks: list[str] = []
    for section in priority:
        sentences = filing.sections.get(section, [])
        if not sentences:
            continue
        for sentence in sentences:
            blocks.append(f"[{section}] {sentence}")
            if sum(len(x) for x in blocks) >= max_chars:
                return " ".join(blocks)[:max_chars]

    if not blocks:
        for section, sentences in filing.sections.items():
            for sentence in sentences:
                blocks.append(f"[{section}] {sentence}")
                if sum(len(x) for x in blocks) >= max_chars:
                    return " ".join(blocks)[:max_chars]

    return " ".join(blocks)[:max_chars]


def _call_llm_json(prompt: str, model: str, max_retries: int = 2) -> dict[str, Any]:
    if SETTINGS.ollama_host:
        os.environ.setdefault("OLLAMA_HOST", SETTINGS.ollama_host)

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
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
            last_exc = exc
            logger.warning("Extraction call failed ({}/{}): {}", attempt + 1, max_retries + 1, exc)
            time.sleep(1.0)

    logger.error("Extraction failed after retries: {}", last_exc)
    return {"entities": [], "relationships": []}


def _validate_extraction(raw: dict[str, Any]) -> tuple[list[Entity], list[Relationship]]:
    entities: list[Entity] = []
    relationships: list[Relationship] = []

    for row in raw.get("entities", []) or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        etype = str(row.get("type", "")).strip()
        description = str(row.get("description", "")).strip()
        if not name or etype not in ENTITY_TYPES:
            continue
        entities.append(Entity(name=name, type=etype, description=description))

    entity_keys = {entity.key() for entity in entities}

    for row in raw.get("relationships", []) or []:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", "")).strip()
        target = str(row.get("target", "")).strip()
        rtype = str(row.get("type", "")).strip()
        evidence = str(row.get("evidence", "")).strip()
        if not source or not target:
            continue
        if rtype not in RELATION_TYPES:
            continue
        if " ".join(source.lower().split()) not in entity_keys:
            continue
        if " ".join(target.lower().split()) not in entity_keys:
            continue
        relationships.append(
            Relationship(source=source, target=target, type=rtype, evidence=evidence[:180])
        )

    # Deduplicate while preserving order.
    uniq_entities: list[Entity] = []
    seen_entities: set[str] = set()
    for entity in entities:
        key = entity.key()
        if key in seen_entities:
            continue
        seen_entities.add(key)
        uniq_entities.append(entity)

    uniq_relationships: list[Relationship] = []
    seen_relationships: set[str] = set()
    for rel in relationships:
        key = rel.key()
        if key in seen_relationships:
            continue
        seen_relationships.add(key)
        uniq_relationships.append(rel)

    return uniq_entities, uniq_relationships


def extract_from_filings(
    filings: list[Filing],
    model: str | None = None,
    max_prompts: int | None = None,
    save_path: Path | None = None,
) -> dict[str, FilingExtraction]:
    """Extract entities/relationships from filings with persistent caching."""

    ensure_dirs()
    model = model or SETTINGS.generator_model
    max_prompts = max_prompts or SETTINGS.entity_extraction_max_prompts
    save_path = save_path or (GRAPH_DIR / "extractions.json")

    if max_prompts < len(filings):
        logger.warning("Extraction prompt cap applied: processing first {} of {} filings", max_prompts, len(filings))

    cache = _load_cache(model) if SETTINGS.use_llm_cache else {}
    extractions: dict[str, FilingExtraction] = {}

    for filing in tqdm(filings[:max_prompts], desc=f"Extracting [{model}]"):
        context = _build_context(filing, max_chars=SETTINGS.entity_extraction_max_chars)
        prompt = EXTRACTION_PROMPT.format(
            company_name=filing.company_name,
            ticker=filing.ticker,
            entity_types=", ".join(ENTITY_TYPES),
            relation_types=", ".join(RELATION_TYPES),
            context=context,
        )
        key = _hash_payload(model=model, prompt=prompt)

        if SETTINGS.use_llm_cache and key in cache:
            raw = cache[key]
        else:
            raw = _call_llm_json(prompt=prompt, model=model)
            if SETTINGS.use_llm_cache:
                cache[key] = raw
                _save_cache(model, cache)

        entities, relationships = _validate_extraction(raw)
        extractions[filing.filing_id] = FilingExtraction(
            filing_id=filing.filing_id,
            ticker=filing.ticker,
            company_name=filing.company_name,
            entities=entities,
            relationships=relationships,
            raw_response=raw,
        )

    with open(save_path, "w", encoding="utf-8") as file:
        json.dump({k: v.to_dict() for k, v in extractions.items()}, file, indent=2)
    logger.info("Saved extractions for {} filings to {}", len(extractions), save_path)

    return extractions


def load_extractions(path: Path | None = None) -> dict[str, FilingExtraction]:
    path = path or (GRAPH_DIR / "extractions.json")
    with open(path, "r", encoding="utf-8") as file:
        rows = json.load(file)

    output: dict[str, FilingExtraction] = {}
    for filing_id, payload in rows.items():
        output[filing_id] = FilingExtraction(
            filing_id=payload["filing_id"],
            ticker=payload["ticker"],
            company_name=payload["company_name"],
            entities=[Entity(**row) for row in payload.get("entities", [])],
            relationships=[Relationship(**row) for row in payload.get("relationships", [])],
            raw_response={},
        )
    return output


if __name__ == "__main__":
    from src.ingest import build_corpus

    filings = build_corpus()
    extracted = extract_from_filings(filings)
    total_entities = sum(len(x.entities) for x in extracted.values())
    total_rels = sum(len(x.relationships) for x in extracted.values())
    print(f"Extracted entities={total_entities}, relationships={total_rels}")
