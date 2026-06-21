"""Embedding client with batching, normalization, and disk caching."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.config import ARTIFACTS_DIR, SETTINGS, ensure_dirs
from src.ollama_client import get_client


DEFAULT_EMBED_DIM = 2560
EMBED_CACHE_DIR = ARTIFACTS_DIR / "_embed_cache"


def _cache_path(model: str) -> Path:
    safe = model.replace(":", "_").replace("/", "_")
    return EMBED_CACHE_DIR / f"{safe}.json"


def _cache_key(text: str, model: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(model.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def _load_cache(model: str) -> dict[str, list[float]]:
    path = _cache_path(model)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        logger.warning("Embedding cache is invalid JSON at {}. Rebuilding cache.", path)
        return {}


def _save_cache(model: str, cache: dict[str, list[float]]) -> None:
    ensure_dirs()
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(model)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(cache, file)
    tmp.replace(path)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec


def embed_text(
    text: str,
    model: str | None = None,
    use_cache: bool | None = None,
) -> np.ndarray:
    """Embed one text string into a normalized float32 vector."""

    model = model or SETTINGS.embed_model
    use_cache = SETTINGS.use_llm_cache if use_cache is None else use_cache

    text = text.strip()
    if not text:
        return np.zeros(DEFAULT_EMBED_DIM, dtype=np.float32)

    if SETTINGS.ollama_host:
        os.environ.setdefault("OLLAMA_HOST", SETTINGS.ollama_host)

    cache = _load_cache(model) if use_cache else {}
    key = _cache_key(text, model)
    if use_cache and key in cache:
        return np.array(cache[key], dtype=np.float32)

    client = get_client()
    response = client.embed(model=model, input=[text])
    vec = np.array(response["embeddings"][0], dtype=np.float32)
    vec = _normalize(vec)

    if use_cache:
        cache[key] = vec.tolist()
        _save_cache(model, cache)

    return vec


def _embed_batch_ollama(texts: list[str], model: str) -> list[np.ndarray]:
    if not texts:
        return []
    client = get_client()
    response = client.embed(model=model, input=texts)
    vectors: list[np.ndarray] = []
    for row in response["embeddings"]:
        arr = np.array(row, dtype=np.float32)
        vectors.append(_normalize(arr))
    return vectors


def embed_texts(
    texts: list[str],
    model: str | None = None,
    batch_size: int | None = None,
    show_progress: bool = True,
    use_cache: bool | None = None,
) -> np.ndarray:
    """Embed many texts with cache-aware batching."""

    model = model or SETTINGS.embed_model
    batch_size = batch_size or SETTINGS.embed_batch_size
    use_cache = SETTINGS.use_llm_cache if use_cache is None else use_cache

    if SETTINGS.ollama_host:
        os.environ.setdefault("OLLAMA_HOST", SETTINGS.ollama_host)

    if not texts:
        return np.zeros((0, DEFAULT_EMBED_DIM), dtype=np.float32)

    cache = _load_cache(model) if use_cache else {}
    out: list[np.ndarray | None] = [None] * len(texts)

    missing_indices: list[int] = []
    missing_texts: list[str] = []

    for idx, raw in enumerate(texts):
        text = raw.strip()
        if not text:
            out[idx] = np.zeros(DEFAULT_EMBED_DIM, dtype=np.float32)
            continue
        key = _cache_key(text, model)
        if use_cache and key in cache:
            out[idx] = np.array(cache[key], dtype=np.float32)
            continue
        missing_indices.append(idx)
        missing_texts.append(text)

    logger.info(
        "Embedding {} texts (cached={}, uncached={})",
        len(texts),
        len(texts) - len(missing_indices),
        len(missing_indices),
    )

    if missing_texts:
        start = time.time()
        batches = [missing_texts[i : i + batch_size] for i in range(0, len(missing_texts), batch_size)]
        iterator = tqdm(batches, desc=f"embed[{model}]") if show_progress else batches
        computed: list[np.ndarray] = []
        for batch in iterator:
            computed.extend(_embed_batch_ollama(batch, model))

        for idx, vec in zip(missing_indices, computed):
            out[idx] = vec

        elapsed = max(time.time() - start, 1e-6)
        logger.info(
            "Computed {} embeddings in {:.1f}s ({:.2f} embeds/s)",
            len(missing_texts),
            elapsed,
            len(missing_texts) / elapsed,
        )

    if use_cache and missing_texts:
        for idx, text in zip(missing_indices, missing_texts):
            vec = out[idx]
            assert vec is not None
            cache[_cache_key(text, model)] = vec.tolist()
        _save_cache(model, cache)

    rows: list[np.ndarray] = []
    for vec in out:
        if vec is None:
            rows.append(np.zeros(DEFAULT_EMBED_DIM, dtype=np.float32))
        else:
            rows.append(vec.astype(np.float32))
    return np.vstack(rows)


def embedding_dim(model: str | None = None) -> int:
    """Detect embedding dimensionality by probing the model once."""

    model = model or SETTINGS.embed_model
    vec = embed_text("dimension-probe", model=model, use_cache=False)
    return int(vec.shape[0])
