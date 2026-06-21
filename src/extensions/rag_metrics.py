"""RAG-level metrics extension (faithfulness/context quality/relevancy).

Includes both heuristic and LLM-as-a-judge scoring paths, pinned by default to
`granite4.1:8b` for this extension workflow.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import ARTIFACTS_DIR, SETTINGS, ensure_dirs
from src.ollama_client import get_client


DEFAULT_EXTENSION_JUDGE_MODEL = SETTINGS.extension_judge_model
# Backward-compatible alias retained for existing imports.
DEFAULT_GUARDIAN_MODEL = DEFAULT_EXTENSION_JUDGE_MODEL
RAG_CACHE_DIR = ARTIFACTS_DIR / "_llm_cache"


@dataclass
class RAGQualityScore:
    query: str
    answer: str
    faithfulness: float
    context_precision: float
    context_recall: float
    answer_relevancy: float
    rationale: str
    model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cache_path(model: str) -> Path:
    safe = model.replace(":", "_").replace("/", "_")
    return RAG_CACHE_DIR / f"rag_quality_{safe}.json"


def _hash_payload(model: str, payload: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"|")
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


def _load_cache(model: str) -> dict[str, dict[str, Any]]:
    path = _cache_path(model)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_cache(model: str, cache: dict[str, dict[str, Any]]) -> None:
    ensure_dirs()
    path = _cache_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    tmp.replace(path)


def _heuristic_context_precision(answer: str, contexts: list[str]) -> float:
    answer_terms = set(answer.lower().split())
    if not contexts:
        return 0.0
    supporting = 0
    for ctx in contexts:
        ctx_terms = set(ctx.lower().split())
        overlap = answer_terms & ctx_terms
        if len(overlap) >= max(2, int(0.03 * max(len(answer_terms), 1))):
            supporting += 1
    return supporting / len(contexts)


def _heuristic_context_recall(answer: str, contexts: list[str]) -> float:
    answer_terms = set(answer.lower().split())
    if not answer_terms:
        return 0.0
    if not contexts:
        return 0.0
    context_union = set()
    for ctx in contexts:
        context_union |= set(ctx.lower().split())
    return len(answer_terms & context_union) / max(len(answer_terms), 1)


RAG_JUDGE_PROMPT = """You are an evaluator for Retrieval-Augmented Generation quality.

Score each metric from 0.0 to 1.0:
- faithfulness: how well answer claims are supported by provided context
- context_precision: proportion of retrieved context that is relevant
- context_recall: proportion of needed evidence covered by retrieved context
- answer_relevancy: how directly answer addresses the question

Return strict JSON:
{{
  "faithfulness": float,
  "context_precision": float,
  "context_recall": float,
  "answer_relevancy": float,
  "rationale": "short explanation"
}}

Question:
{query}

Answer:
{answer}

Reference answer (optional):
{reference}

Retrieved context:
{contexts}
"""


def judge_rag_quality(
    query: str,
    answer: str,
    contexts: list[str],
    reference: str | None = None,
    model: str = DEFAULT_EXTENSION_JUDGE_MODEL,
    use_cache: bool | None = None,
) -> RAGQualityScore:
    """LLM-as-a-judge quality evaluation for RAG-specific metrics."""
    use_cache = SETTINGS.use_llm_cache if use_cache is None else use_cache

    if SETTINGS.ollama_host:
        os.environ.setdefault("OLLAMA_HOST", SETTINGS.ollama_host)

    joined_context = "\n\n".join(
        f"[{i+1}] {ctx[:1200]}" for i, ctx in enumerate(contexts)
    ) or "(no contexts provided)"

    prompt = RAG_JUDGE_PROMPT.format(
        query=query,
        answer=answer,
        reference=reference or "(not provided)",
        contexts=joined_context,
    )

    cache = _load_cache(model) if use_cache else {}
    key = _hash_payload(model, prompt)

    raw: dict[str, Any] | None = cache.get(key) if use_cache else None
    if raw is None:
        try:
            client = get_client()
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0},
            )
            raw = json.loads(response["message"]["content"])
        except Exception:
            raw = {}

        if use_cache:
            cache[key] = raw
            _save_cache(model, cache)

    def _clamp(value: Any, fallback: float) -> float:
        try:
            v = float(value)
        except Exception:
            return fallback
        return min(max(v, 0.0), 1.0)

    heuristic_prec = _heuristic_context_precision(answer, contexts)
    heuristic_rec = _heuristic_context_recall(answer, contexts)

    faithfulness = _clamp(raw.get("faithfulness"), heuristic_prec)
    ctx_precision = _clamp(raw.get("context_precision"), heuristic_prec)
    ctx_recall = _clamp(raw.get("context_recall"), heuristic_rec)
    answer_rel = _clamp(raw.get("answer_relevancy"), 0.5)

    return RAGQualityScore(
        query=query,
        answer=answer,
        faithfulness=faithfulness,
        context_precision=ctx_precision,
        context_recall=ctx_recall,
        answer_relevancy=answer_rel,
        rationale=str(raw.get("rationale", "")),
        model=model,
    )


def evaluate_rag_quality(
    predictions: list[str],
    contexts: list[list[str]],
    queries: list[str],
    references: list[str] | None = None,
    model: str = DEFAULT_EXTENSION_JUDGE_MODEL,
) -> dict[str, Any]:
    """Aggregate RAG-level metrics across multiple QA examples."""
    if not (len(predictions) == len(contexts) == len(queries)):
        raise ValueError("predictions, contexts, and queries must have equal length")

    references = references or [None] * len(predictions)

    rows: list[RAGQualityScore] = []
    for pred, ctx, query, ref in zip(predictions, contexts, queries, references):
        rows.append(
            judge_rag_quality(
                query=query,
                answer=pred,
                contexts=ctx,
                reference=ref,
                model=model,
            )
        )

    n = max(len(rows), 1)
    payload = {
        "model": model,
        "n_queries": len(rows),
        "faithfulness": round(sum(r.faithfulness for r in rows) / n, 4),
        "context_precision": round(sum(r.context_precision for r in rows) / n, 4),
        "context_recall": round(sum(r.context_recall for r in rows) / n, 4),
        "answer_relevancy": round(sum(r.answer_relevancy for r in rows) / n, 4),
        "rows": [r.to_dict() for r in rows],
    }
    return payload
