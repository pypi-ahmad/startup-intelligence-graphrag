"""LLM-as-a-judge scoring for RAG outputs."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import ARTIFACTS_DIR, SETTINGS, ensure_dirs
from src.generation import GenerationResult
from src.ollama_client import get_client


LLM_CACHE_DIR = ARTIFACTS_DIR / "_llm_cache"


@dataclass
class JudgeScore:
    query: str
    answer: str
    reference: str | None
    correctness: int
    relevance: int
    completeness: int
    groundedness: int
    hallucination_risk: int
    rationale: str = ""
    contexts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "reference": self.reference,
            "correctness": self.correctness,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "groundedness": self.groundedness,
            "hallucination_risk": self.hallucination_risk,
            "rationale": self.rationale,
            "contexts": self.contexts,
            "overall": self.overall(),
        }

    def overall(self) -> float:
        return (
            self.correctness
            + self.relevance
            + self.completeness
            + self.groundedness
            + (6 - self.hallucination_risk)
        ) / 5.0


JUDGE_PROMPT = """You are evaluating a startup intelligence RAG answer.

Score each dimension from 1 to 5.

1) correctness
2) relevance
3) completeness
4) groundedness
5) hallucination_risk (1 is best, 5 is worst)

Return strict JSON with keys:
{{
  "correctness": int,
  "relevance": int,
  "completeness": int,
  "groundedness": int,
  "hallucination_risk": int,
  "rationale": string
}}

Question:
{query}

Answer:
{answer}

Reference answer:
{reference}

Retrieved context:
{contexts}
"""


def _cache_path(model: str) -> Path:
    safe = model.replace(":", "_").replace("/", "_")
    return LLM_CACHE_DIR / f"judge_{safe}.json"


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


def _hash_prompt(model: str, prompt: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(model.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(prompt.encode("utf-8"))
    return hasher.hexdigest()


def _format_contexts(contexts: list[dict[str, Any]], limit: int = 10) -> str:
    lines: list[str] = []
    for row in contexts[:limit]:
        lines.append(
            f"[{row.get('id', '?')}] [{row.get('ticker', '?')} | {row.get('section', '?')}] "
            f"{str(row.get('text_preview', ''))[:500]}"
        )
    return "\n".join(lines) if lines else "(no contexts provided)"


def judge_answer(
    query: str,
    answer: str,
    contexts: list[dict[str, Any]],
    reference: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_retries: int = 2,
) -> JudgeScore:
    """Score one generated answer with LLM-as-a-judge."""

    model = model or SETTINGS.judge_model
    temperature = SETTINGS.judge_temperature if temperature is None else temperature

    if SETTINGS.ollama_host:
        os.environ.setdefault("OLLAMA_HOST", SETTINGS.ollama_host)

    prompt = JUDGE_PROMPT.format(
        query=query,
        answer=answer,
        reference=reference or "(not provided)",
        contexts=_format_contexts(contexts),
    )

    cache = _load_cache(model) if SETTINGS.use_llm_cache else {}
    key = _hash_prompt(model, prompt)

    raw: dict[str, Any] = {}
    if SETTINGS.use_llm_cache and key in cache:
        raw = cache[key]
    else:
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                client = get_client()
                response = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    format="json",
                    options={"temperature": temperature},
                )
                raw = json.loads(response["message"]["content"])
                if all(
                    isinstance(raw.get(field), int) and 1 <= int(raw[field]) <= 5
                    for field in [
                        "correctness",
                        "relevance",
                        "completeness",
                        "groundedness",
                        "hallucination_risk",
                    ]
                ):
                    break
                raw = {}
            except Exception as exc:
                last_exc = exc
                logger.warning("Judge call failed ({}/{}): {}", attempt + 1, max_retries + 1, exc)
                time.sleep(1.0)

        if not raw:
            logger.error("Judge scoring failed; returning zero-like scores")
            raw = {
                "correctness": 1,
                "relevance": 1,
                "completeness": 1,
                "groundedness": 1,
                "hallucination_risk": 5,
                "rationale": "Judge call failed",
            }

        if SETTINGS.use_llm_cache:
            cache[key] = raw
            _save_cache(model, cache)

    return JudgeScore(
        query=query,
        answer=answer,
        reference=reference,
        correctness=int(raw.get("correctness", 1)),
        relevance=int(raw.get("relevance", 1)),
        completeness=int(raw.get("completeness", 1)),
        groundedness=int(raw.get("groundedness", 1)),
        hallucination_risk=int(raw.get("hallucination_risk", 5)),
        rationale=str(raw.get("rationale", "")),
        contexts=contexts,
    )


def judge_generation(
    generation: GenerationResult,
    reference: str | None = None,
    model: str | None = None,
) -> JudgeScore:
    """Evaluate a `GenerationResult` object with citation-derived context."""

    contexts = generation.citations
    return judge_answer(
        query=generation.query,
        answer=generation.answer,
        contexts=contexts,
        reference=reference,
        model=model,
    )
