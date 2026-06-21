"""Token-aware chunking for SEC filings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken
from loguru import logger
from tqdm import tqdm

from src.config import CHUNKS_DIR, SETTINGS, ensure_dirs
from src.ingest import Filing


_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """Single retrievable chunk with provenance metadata."""

    chunk_id: str
    filing_id: str
    ticker: str
    company_name: str
    section: str
    text: str
    sentence_ids: list[int]
    token_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def token_count(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def chunk_section(
    filing: Filing,
    section_name: str,
    sentences: list[str],
    sentence_offset: int,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Chunk one filing section with overlap in token space."""

    if not sentences:
        return []

    tokenized = [_ENC.encode(sent, disallowed_special=()) for sent in sentences]
    chunks: list[Chunk] = []

    start = 0
    index = 0
    n = len(sentences)

    while start < n:
        total_tokens = 0
        end = start

        while end < n and total_tokens + len(tokenized[end]) <= chunk_size:
            total_tokens += len(tokenized[end])
            end += 1

        if end == start:
            total_tokens = len(tokenized[start])
            end = start + 1

        text = " ".join(sentences[start:end])
        sentence_ids = list(range(sentence_offset + start, sentence_offset + end))

        chunks.append(
            Chunk(
                chunk_id=f"{filing.filing_id}__{section_name}__{index:04d}",
                filing_id=filing.filing_id,
                ticker=filing.ticker,
                company_name=filing.company_name,
                section=section_name,
                text=text,
                sentence_ids=sentence_ids,
                token_count=total_tokens,
            )
        )
        index += 1

        if end >= n:
            break

        overlap_tokens = 0
        next_start = end - 1
        while next_start > start and overlap_tokens < overlap:
            overlap_tokens += len(tokenized[next_start])
            next_start -= 1
        start = max(start + 1, next_start + 1)

    return chunks


def chunk_filing(
    filing: Filing,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Chunk all sections in a filing."""

    chunk_size = chunk_size or SETTINGS.chunk_size_tokens
    chunk_overlap = chunk_overlap or SETTINGS.chunk_overlap_tokens

    chunks: list[Chunk] = []
    sentence_offset = 0

    for section_name, sentences in filing.sections.items():
        section_chunks = chunk_section(
            filing=filing,
            section_name=section_name,
            sentences=sentences,
            sentence_offset=sentence_offset,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
        chunks.extend(section_chunks)
        sentence_offset += len(sentences)

    return chunks


def chunk_corpus(
    filings: list[Filing],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Chunk every filing in the corpus."""

    all_chunks: list[Chunk] = []
    for filing in tqdm(filings, desc="Chunking filings"):
        all_chunks.extend(
            chunk_filing(
                filing=filing,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    mean_tokens = 0.0
    if all_chunks:
        mean_tokens = sum(c.token_count for c in all_chunks) / len(all_chunks)

    logger.info(
        "Chunking complete: {} chunks from {} filings (mean tokens/chunk={:.1f})",
        len(all_chunks),
        len(filings),
        mean_tokens,
    )
    return all_chunks


def save_chunks(chunks: list[Chunk], path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or (CHUNKS_DIR / "chunks.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump([c.to_dict() for c in chunks], file, indent=2)
    logger.info("Saved {} chunks to {}", len(chunks), path)
    return path


def load_chunks(path: Path | None = None) -> list[Chunk]:
    path = path or (CHUNKS_DIR / "chunks.json")
    with open(path, "r", encoding="utf-8") as file:
        rows = json.load(file)
    chunks = [Chunk(**row) for row in rows]
    logger.info("Loaded {} chunks from {}", len(chunks), path)
    return chunks


if __name__ == "__main__":
    from src.ingest import build_corpus

    filings = build_corpus()
    chunks = chunk_corpus(filings)
    save_chunks(chunks)
    print(f"Created {len(chunks)} chunks")
