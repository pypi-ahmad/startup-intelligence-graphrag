"""Evaluation query bank and persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import EVAL_QUERIES_PATH, ensure_dirs


# Query schema contract:
# - query_id: stable identifier
# - query: natural-language question
# - query_type: local | global | factual
# - relevant_tickers: ticker-level relevance labels
# - relevant_sections: optional section hints
# - keyword_hints: lexical hints for weak supervision relevance mapping
# - reference_answer: expected answer text for generation metrics
# - retrieval_relevant_chunk_ids: optional explicit chunk labels (filled post-run)
EVAL_QUERIES: list[dict[str, Any]] = [
    {
        "query_id": "q01_amd_products",
        "query": "What are AMD's major product lines and business segments?",
        "query_type": "local",
        "relevant_tickers": ["AMD"],
        "relevant_sections": ["Business", "Management's Discussion and Analysis (MD&A)"],
        "keyword_hints": ["Ryzen", "Radeon", "EPYC", "segment", "Computing and Graphics"],
        "reference_answer": "AMD discusses CPUs, GPUs, data-center and embedded offerings, with segment narratives in Business and MD&A sections.",
        "retrieval_relevant_chunk_ids": [],
    },
    {
        "query_id": "q02_abbott_risks",
        "query": "What risk factors are emphasized by Abbott Laboratories?",
        "query_type": "local",
        "relevant_tickers": ["ABT"],
        "relevant_sections": ["Risk Factors"],
        "keyword_hints": ["regulatory", "recall", "reimbursement", "litigation", "supply chain"],
        "reference_answer": "Abbott highlights regulatory, product-liability, reimbursement, and supply-chain related risks in filing risk disclosures.",
        "retrieval_relevant_chunk_ids": [],
    },
    {
        "query_id": "q03_matson_segments",
        "query": "Describe Matson's core operations and segments.",
        "query_type": "local",
        "relevant_tickers": ["MATX"],
        "relevant_sections": ["Business"],
        "keyword_hints": ["Ocean Transportation", "Logistics", "Hawaii", "Guam", "Pacific"],
        "reference_answer": "Matson positions itself around Ocean Transportation and Logistics services in Pacific trade lanes.",
        "retrieval_relevant_chunk_ids": [],
    },
    {
        "query_id": "q04_ingersoll_risks",
        "query": "What risks does Ingersoll Rand disclose around operations and supply chain?",
        "query_type": "local",
        "relevant_tickers": ["IR"],
        "relevant_sections": ["Risk Factors"],
        "keyword_hints": ["raw materials", "supply chain", "FX", "cybersecurity", "acquisition integration"],
        "reference_answer": "Ingersoll Rand filing language covers cyclicality, commodity/supply-chain exposure, cyber risk, and acquisition integration risk.",
        "retrieval_relevant_chunk_ids": [],
    },
    {
        "query_id": "q05_nextera_business",
        "query": "What is NextEra Energy Partners' business model?",
        "query_type": "local",
        "relevant_tickers": ["NEP"],
        "relevant_sections": ["Business", "MD&A"],
        "keyword_hints": ["renewable", "wind", "solar", "contracted", "cash distribution"],
        "reference_answer": "NEP describes owning/operating contracted renewable assets, with wind and solar portfolio focus and cash distribution goals.",
        "retrieval_relevant_chunk_ids": [],
    },
    {
        "query_id": "q06_aar_business",
        "query": "Summarize AAR Corp's primary business activities.",
        "query_type": "local",
        "relevant_tickers": ["AIR"],
        "relevant_sections": ["Business"],
        "keyword_hints": ["MRO", "aftermarket", "aerospace", "defense", "supply chain"],
        "reference_answer": "AAR filing text frames the company as aerospace aftermarket, MRO, and government/commercial aviation services provider.",
        "retrieval_relevant_chunk_ids": [],
    },
    {
        "query_id": "q07_supply_chain_global",
        "query": "Which companies in this corpus show strong supply-chain or raw-material exposure?",
        "query_type": "global",
        "relevant_tickers": ["AMD", "IR", "MATX", "APD", "CECE"],
        "relevant_sections": ["Risk Factors", "MD&A"],
        "keyword_hints": ["supply chain", "raw material", "commodity", "foundry", "fuel"],
        "reference_answer": "Cross-company risk disclosures indicate supply-chain/commodity exposure across hardware, industrial, shipping, and chemicals firms.",
        "retrieval_relevant_chunk_ids": [],
    },
    {
        "query_id": "q08_common_risk_theme",
        "query": "What recurring risk themes appear across multiple filings?",
        "query_type": "global",
        "relevant_tickers": ["AMD", "ABT", "IR", "NEP", "APD", "MATX", "ISTR"],
        "relevant_sections": ["Risk Factors"],
        "keyword_hints": ["macroeconomic", "cybersecurity", "regulatory", "interest rate", "pandemic"],
        "reference_answer": "Shared themes include macroeconomic uncertainty, cyber threats, regulatory pressure, and operational disruptions.",
        "retrieval_relevant_chunk_ids": [],
    },
    {
        "query_id": "q09_strategic_signals_global",
        "query": "What strategic signals (M&A, expansion, restructuring) are visible across the filings?",
        "query_type": "global",
        "relevant_tickers": ["AMD", "IR", "NEP", "APD", "ABT"],
        "relevant_sections": ["Business", "MD&A", "Risk Factors"],
        "keyword_hints": ["acquisition", "divestiture", "expansion", "portfolio", "restructuring"],
        "reference_answer": "Filings repeatedly reference M&A-driven portfolio changes, capacity expansion, and restructuring-oriented strategic moves.",
        "retrieval_relevant_chunk_ids": [],
    },
    {
        "query_id": "q10_sustainability_global",
        "query": "Which companies explicitly discuss sustainability or energy transition initiatives?",
        "query_type": "global",
        "relevant_tickers": ["NEP", "APD", "IR", "AE"],
        "relevant_sections": ["Business", "MD&A"],
        "keyword_hints": ["sustainability", "renewable", "hydrogen", "emissions", "clean energy"],
        "reference_answer": "Energy and industrial filings highlight renewable, hydrogen, and broader emissions/sustainability initiatives.",
        "retrieval_relevant_chunk_ids": [],
    },
    {
        "query_id": "q11_aar_incorporation",
        "query": "Where is AAR Corp incorporated?",
        "query_type": "factual",
        "relevant_tickers": ["AIR"],
        "relevant_sections": ["Business"],
        "keyword_hints": ["incorporated", "Delaware", "organized"],
        "reference_answer": "AAR filing text identifies Delaware incorporation.",
        "retrieval_relevant_chunk_ids": [],
    },
    {
        "query_id": "q12_amd_exchange",
        "query": "What stock exchange is AMD listed on?",
        "query_type": "factual",
        "relevant_tickers": ["AMD"],
        "relevant_sections": ["Market for Registrant's Common Equity", "Business"],
        "keyword_hints": ["NASDAQ", "exchange", "ticker"],
        "reference_answer": "AMD is listed on NASDAQ under ticker AMD.",
        "retrieval_relevant_chunk_ids": [],
    },
    {
        "query_id": "q13_cactus_state",
        "query": "Which state is Cactus, Inc. incorporated in?",
        "query_type": "factual",
        "relevant_tickers": ["WHD"],
        "relevant_sections": ["Business"],
        "keyword_hints": ["incorporated", "Delaware", "headquartered"],
        "reference_answer": "Cactus filing text indicates Delaware incorporation.",
        "retrieval_relevant_chunk_ids": [],
    },
    {
        "query_id": "q14_apd_report_date",
        "query": "What is the report date for Air Products and Chemicals in this filing set?",
        "query_type": "factual",
        "relevant_tickers": ["APD"],
        "relevant_sections": ["Business", "MD&A"],
        "keyword_hints": ["report date", "fiscal year", "September"],
        "reference_answer": "Air Products filing metadata includes a report date aligned with the fiscal year end.",
        "retrieval_relevant_chunk_ids": [],
    },
]


def save_eval_queries(path: Path | None = None, queries: list[dict[str, Any]] | None = None) -> Path:
    """Persist evaluation query set as JSONL."""

    ensure_dirs()
    path = path or EVAL_QUERIES_PATH
    rows = queries or EVAL_QUERIES

    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")

    logger.info("Saved {} evaluation queries to {}", len(rows), path)
    return path


def load_eval_queries(path: Path | None = None) -> list[dict[str, Any]]:
    """Load evaluation query JSONL; create default set on first use."""

    path = path or EVAL_QUERIES_PATH
    if not path.exists():
        save_eval_queries(path=path)

    with open(path, "r", encoding="utf-8") as file:
        queries = [json.loads(line) for line in file if line.strip()]

    logger.info("Loaded {} evaluation queries from {}", len(queries), path)
    return queries


if __name__ == "__main__":
    save_eval_queries()
    for row in load_eval_queries()[:5]:
        print(f"{row['query_id']}: {row['query']}")
