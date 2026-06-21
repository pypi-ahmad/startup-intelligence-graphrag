"""Dynamic evaluation query builder for real corpus runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.ingest import Filing


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "will",
    "its",
    "our",
    "their",
    "about",
    "under",
    "into",
    "than",
    "which",
    "also",
    "these",
    "those",
}


def _top_keywords(sentences: list[str], k: int = 5) -> list[str]:
    counts: dict[str, int] = {}
    for sentence in sentences[:200]:
        for token in _WORD_RE.findall(sentence.lower()):
            if token in _STOPWORDS or len(token) < 4:
                continue
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in ranked[:k]]


def _first_section_text(filing: Filing, section_candidates: list[str]) -> str:
    for section in section_candidates:
        if section in filing.sections and filing.sections[section]:
            return " ".join(filing.sections[section][:3])
    all_sentences = filing.all_sentences()
    return " ".join(all_sentences[:3])


def build_eval_queries_from_filings(
    filings: list[Filing],
    max_local_companies: int = 6,
) -> list[dict[str, Any]]:
    """Construct weakly supervised eval set from available filings."""

    local_candidates = sorted(filings, key=lambda x: x.sentence_count(), reverse=True)[:max_local_companies]
    queries: list[dict[str, Any]] = []

    for idx, filing in enumerate(local_candidates, start=1):
        business_text = _first_section_text(filing, ["Business", "Management's Discussion and Analysis (MD&A)"])
        risk_text = _first_section_text(filing, ["Risk Factors", "Management's Discussion and Analysis (MD&A)"])
        business_keywords = _top_keywords(filing.sections.get("Business", filing.all_sentences()), k=5)
        risk_keywords = _top_keywords(filing.sections.get("Risk Factors", filing.all_sentences()), k=5)

        queries.append(
            {
                "query_id": f"q{idx:02d}_local_business_{filing.ticker.lower()}",
                "query": f"What are the core business activities and strategic focus areas of {filing.company_name} ({filing.ticker})?",
                "query_type": "local",
                "relevant_tickers": [filing.ticker],
                "relevant_sections": ["Business", "Management's Discussion and Analysis (MD&A)"],
                "keyword_hints": business_keywords,
                "reference_answer": business_text[:600],
                "retrieval_relevant_chunk_ids": [],
            }
        )
        queries.append(
            {
                "query_id": f"q{idx:02d}_local_risk_{filing.ticker.lower()}",
                "query": f"What major risk factors does {filing.company_name} ({filing.ticker}) emphasize?",
                "query_type": "local",
                "relevant_tickers": [filing.ticker],
                "relevant_sections": ["Risk Factors", "Management's Discussion and Analysis (MD&A)"],
                "keyword_hints": risk_keywords,
                "reference_answer": risk_text[:600],
                "retrieval_relevant_chunk_ids": [],
            }
        )

    global_tickers = [f.ticker for f in local_candidates]
    global_keywords: list[str] = []
    for filing in local_candidates:
        global_keywords.extend(_top_keywords(filing.all_sentences(), k=3))
    global_keywords = list(dict.fromkeys(global_keywords))[:8]

    queries.extend(
        [
            {
                "query_id": "q_global_risk_themes",
                "query": "What recurring risk themes appear across these companies?",
                "query_type": "global",
                "relevant_tickers": global_tickers,
                "relevant_sections": ["Risk Factors", "Management's Discussion and Analysis (MD&A)"],
                "keyword_hints": global_keywords,
                "reference_answer": "Common risk patterns include regulatory, macroeconomic, operational, and technology-related pressures across filings.",
                "retrieval_relevant_chunk_ids": [],
            },
            {
                "query_id": "q_global_strategy_signals",
                "query": "Which strategic signals (expansion, restructuring, partnerships, acquisitions) are visible across the corpus?",
                "query_type": "global",
                "relevant_tickers": global_tickers,
                "relevant_sections": ["Business", "Management's Discussion and Analysis (MD&A)"],
                "keyword_hints": ["acquisition", "expansion", "restructuring", "partnership", "investment"],
                "reference_answer": "Filings show mixed strategic signals around growth investments, portfolio adjustments, and market expansion initiatives.",
                "retrieval_relevant_chunk_ids": [],
            },
            {
                "query_id": "q_global_supply_chain",
                "query": "Which companies disclose supply-chain or raw-material exposure?",
                "query_type": "global",
                "relevant_tickers": global_tickers,
                "relevant_sections": ["Risk Factors", "Management's Discussion and Analysis (MD&A)"],
                "keyword_hints": ["supply", "raw", "material", "vendor", "cost"],
                "reference_answer": "Multiple filings mention supplier dependencies, input-cost volatility, and procurement constraints.",
                "retrieval_relevant_chunk_ids": [],
            },
            {
                "query_id": "q_factual_listing_check",
                "query": f"What ticker symbol is used for {local_candidates[0].company_name} in this corpus?",
                "query_type": "factual",
                "relevant_tickers": [local_candidates[0].ticker],
                "relevant_sections": ["Business"],
                "keyword_hints": [local_candidates[0].ticker.lower(), "ticker", "symbol"],
                "reference_answer": f"The filing identifies the company ticker as {local_candidates[0].ticker}.",
                "retrieval_relevant_chunk_ids": [],
            },
        ]
    )

    return queries


def save_eval_queries(path: Path, queries: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in queries:
            file.write(json.dumps(row) + "\n")
    return path

