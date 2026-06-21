"""Answer generation over retrieved filing chunks."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from loguru import logger

from src.config import ARTIFACTS_DIR, SETTINGS, ensure_dirs
from src.ollama_client import get_client
from src.retrievers import RetrievedChunk


LLM_CACHE_DIR = ARTIFACTS_DIR / "_llm_cache"


@dataclass
class GenerationResult:
    query: str
    answer: str
    citations: list[dict]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


SYSTEM_PROMPT = """You are a startup/company intelligence analyst.
Answer the question using ONLY the provided SEC filing excerpts.

Rules:
- Every factual claim must cite one or more excerpt IDs in brackets, e.g. [1].
- If evidence is missing, explicitly say information is insufficient in provided excerpts.
- Prefer precision over verbosity.
- Do not invent numbers, entities, or events.
"""


USER_PROMPT = """Question:
{query}

Retrieved SEC filing excerpts:
{context}

Provide a concise answer with citations."""


def _cache_path(model: str) -> Path:
    safe = model.replace(":", "_").replace("/", "_")
    return LLM_CACHE_DIR / f"gen_{safe}.json"


def _load_cache(model: str) -> dict[str, str]:
    path = _cache_path(model)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return {}


def _save_cache(model: str, cache: dict[str, str]) -> None:
    ensure_dirs()
    path = _cache_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(cache, file)
    tmp.replace(path)


def _hash_prompt(model: str, prompt: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(model.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(prompt.encode("utf-8"))
    return hasher.hexdigest()


def build_context_block(chunks: Iterable[RetrievedChunk]) -> tuple[str, list[dict]]:
    """Format retrieved chunks into a numbered prompt context."""

    blocks: list[str] = []
    citations: list[dict] = []

    for idx, chunk in enumerate(chunks, start=1):
        snippet = chunk.text[:700] + ("..." if len(chunk.text) > 700 else "")
        blocks.append(
            f"[{idx}] [{chunk.ticker} | {chunk.section}] {snippet}\n"
            f"    source={chunk.source}, score={chunk.score:.4f}, filing_id={chunk.filing_id}"
        )
        citations.append(
            {
                "id": idx,
                "chunk_id": chunk.chunk_id,
                "filing_id": chunk.filing_id,
                "ticker": chunk.ticker,
                "company_name": chunk.company_name,
                "section": chunk.section,
                "score": float(chunk.score),
                "source": chunk.source,
                "via": chunk.via,
                "text_preview": snippet,
            }
        )

    return "\n\n".join(blocks), citations


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    model: str | None = None,
    temperature: float | None = None,
    max_retries: int = 2,
) -> GenerationResult:
    """Generate grounded answer from retrieved chunks."""

    model = model or SETTINGS.generator_model
    temperature = SETTINGS.generation_temperature if temperature is None else temperature

    if SETTINGS.ollama_host:
        os.environ.setdefault("OLLAMA_HOST", SETTINGS.ollama_host)

    context, citations = build_context_block(chunks)
    user_prompt = USER_PROMPT.format(query=query, context=context)

    cache = _load_cache(model) if SETTINGS.use_llm_cache else {}
    key = _hash_prompt(model, user_prompt)

    if SETTINGS.use_llm_cache and key in cache:
        answer = cache[key]
        return GenerationResult(query=query, answer=answer, citations=citations, model=model)

    answer = ""
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            client = get_client()
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": temperature},
            )
            answer = response["message"]["content"].strip()
            if answer:
                break
        except Exception as exc:
            last_exc = exc
            logger.warning("Generation call failed ({}/{}): {}", attempt + 1, max_retries + 1, exc)
            time.sleep(1.0)

    if not answer:
        answer = f"Generation failed: {last_exc}" if last_exc else "Generation failed with empty response."

    if SETTINGS.use_llm_cache:
        cache[key] = answer
        _save_cache(model, cache)

    return GenerationResult(
        query=query,
        answer=answer,
        citations=citations,
        model=model,
    )


def answer_query(
    query: str,
    retriever,
    k: int | None = None,
    model: str | None = None,
) -> tuple[GenerationResult, list[RetrievedChunk]]:
    """Retrieve then generate answer."""

    k = k or SETTINGS.default_top_k
    chunks = retriever.retrieve(query, k=k)
    result = generate_answer(query=query, chunks=chunks, model=model)
    return result, chunks
