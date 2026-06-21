"""Ingestion adapter for `deerfieldgreen/stk-sec-filings`.

This module provides filing-aware normalization for deerfield SEC records and
is used by the strict ingestion path.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import RAW_DIR, SETTINGS, ensure_dirs
from src.ingest import Filing


DEERFIELD_DATASET_ID = "deerfieldgreen/stk-sec-filings"

_ITEM_SECTION_MAP: dict[str, str] = {
    "1": "Business",
    "1a": "Risk Factors",
    "1b": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis (MD&A)",
    "7a": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9a": "Controls and Procedures",
    "9b": "Other Information",
    "10": "Directors, Officers and Compensation",
    "11": "Security Ownership",
    "12": "Related Transactions",
    "13": "Accountant Fees",
    "14": "Exhibits and Financial Statement Schedules",
}

_ITEM_SPLIT_RE = re.compile(
    r"(?is)\bitem\s+([0-9]{1,2}[a-z]?)\s*[\.\-:]?\s+",
)
_WS_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z\(\[])")


@dataclass
class DeerfieldRecord:
    """Subset of raw dataset fields needed downstream."""

    symbol: str
    cik: str
    form_type: str
    filing_date: str
    accepted_date: str
    final_link: str
    text: str


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_date(value: str) -> str:
    text = _safe_text(value)
    if not text:
        return ""
    text = text.replace("T", " ").split(" ")[0]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text[:10]


def _normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text.replace("\x00", " ")).strip()


def _split_into_sentences(text: str) -> list[str]:
    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return []
    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    out = [p.strip() for p in parts if len(p.strip()) > 20]
    return out


def _split_sections_from_text(text: str) -> dict[str, list[str]]:
    """Best-effort split of full filing text into SEC item sections."""

    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return {}

    matches = list(_ITEM_SPLIT_RE.finditer(cleaned))
    if not matches:
        return {"Business": _split_into_sentences(cleaned)}

    sections: "OrderedDict[str, list[str]]" = OrderedDict()
    for idx, match in enumerate(matches):
        item_key = match.group(1).lower().strip()
        section_name = _ITEM_SECTION_MAP.get(item_key, f"Item {item_key.upper()}")
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        chunk = cleaned[start:end].strip()
        sentences = _split_into_sentences(chunk)
        if not sentences:
            continue
        existing = sections.get(section_name, [])
        existing.extend(sentences)
        sections[section_name] = existing

    if not sections:
        return {"Business": _split_into_sentences(cleaned)}
    return dict(sections)


def _filing_id_from_record(symbol: str, form_type: str, report_date: str, final_link: str) -> str:
    digest = hashlib.sha1(final_link.encode("utf-8")).hexdigest()[:8]
    year = report_date[:4] if report_date else "unknown"
    form_safe = (form_type or "10-K").replace("/", "-").replace(" ", "")
    sym = symbol or "UNK"
    return f"{sym}_{form_safe}_{year}_{digest}"


def _record_from_row(row: dict[str, Any]) -> DeerfieldRecord:
    return DeerfieldRecord(
        symbol=_safe_text(row.get("symbol")),
        cik=_safe_text(row.get("cik")),
        form_type=_safe_text(row.get("type")),
        filing_date=_normalize_date(_safe_text(row.get("fillingDate"))),
        accepted_date=_normalize_date(_safe_text(row.get("acceptedDate"))),
        final_link=_safe_text(row.get("finalLink")),
        text=_safe_text(row.get("text")),
    )


def download_deerfield_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Download raw rows from Hugging Face dataset."""

    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("`datasets` package unavailable. Run `uv sync` first.") from exc

    logger.info("Loading dataset {}", DEERFIELD_DATASET_ID)
    ds = load_dataset(DEERFIELD_DATASET_ID, split="train")
    rows = [dict(r) for r in ds]
    if not rows:
        raise RuntimeError(f"Dataset {DEERFIELD_DATASET_ID} returned zero rows.")

    schema = {
        "dataset_repo": DEERFIELD_DATASET_ID,
        "n_rows": len(rows),
        "columns": sorted(list(rows[0].keys())),
    }
    return rows, schema


def _to_filing(record: DeerfieldRecord) -> Filing | None:
    if not record.symbol or not record.text:
        return None
    form = record.form_type or "10-K"
    if "10-k" not in form.lower():
        return None

    sections = _split_sections_from_text(record.text)
    max_sentences = int(SETTINGS.max_sentences_per_filing)
    total_sentences = sum(len(v) for v in sections.values())
    if total_sentences > max_sentences and total_sentences > 0:
        ratio = max_sentences / total_sentences
        trimmed: dict[str, list[str]] = {}
        for section, rows in sections.items():
            keep = max(1, int(len(rows) * ratio))
            trimmed[section] = rows[:keep]
        sections = trimmed
    sentence_count = sum(len(v) for v in sections.values())
    if sentence_count < SETTINGS.min_sentences_per_filing:
        return None

    report_date = record.filing_date or record.accepted_date
    filing_id = _filing_id_from_record(
        symbol=record.symbol,
        form_type=form,
        report_date=report_date,
        final_link=record.final_link or f"{record.symbol}-{report_date}",
    )

    return Filing(
        company_name=record.symbol,
        cik=record.cik,
        ticker=record.symbol,
        exchange="",
        state_of_incorporation="",
        sic="",
        form=form,
        filing_date=record.filing_date,
        report_date=report_date,
        filing_id=filing_id,
        source_split="train",
        source_row_count=1,
        sections=sections,
    )


def _latest_by_ticker(filings: list[Filing]) -> list[Filing]:
    by_ticker: dict[str, Filing] = {}
    for filing in filings:
        current = by_ticker.get(filing.ticker)
        if current is None:
            by_ticker[filing.ticker] = filing
            continue
        current_key = current.report_date or current.filing_date
        candidate_key = filing.report_date or filing.filing_date
        if candidate_key > current_key:
            by_ticker[filing.ticker] = filing
    return list(by_ticker.values())


def _sample_filings(filings: list[Filing], n_companies: int) -> list[Filing]:
    candidates = _latest_by_ticker(filings)
    if len(candidates) <= n_companies:
        return sorted(candidates, key=lambda x: x.ticker)

    ranked = sorted(candidates, key=lambda x: x.sentence_count(), reverse=True)
    pool = ranked[: max(n_companies * 3, n_companies)]
    rng = random.Random(SETTINGS.company_selection_seed)
    rng.shuffle(pool)
    selected = sorted(pool[:n_companies], key=lambda x: x.ticker)
    return selected


def save_deerfield_artifacts(
    filings: list[Filing],
    url_map: dict[str, str],
    schema: dict[str, Any],
) -> tuple[Path, Path, Path]:
    ensure_dirs()
    filings_path = RAW_DIR / "filings.json"
    manifest_path = RAW_DIR / "manifest.json"
    url_map_path = RAW_DIR / "filing_url_map.json"

    filings_path.write_text(
        json.dumps([f.to_dict() for f in filings], indent=2),
        encoding="utf-8",
    )
    manifest = {
        "dataset_repo": DEERFIELD_DATASET_ID,
        "strict_dataset": True,
        "n_filings": len(filings),
        "tickers": sorted({f.ticker for f in filings}),
        "schema_snapshot": schema,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    url_map_path.write_text(json.dumps(url_map, indent=2), encoding="utf-8")
    return filings_path, manifest_path, url_map_path


def build_deerfield_corpus(
    n_companies: int = 30,
    force_download: bool = False,
) -> tuple[list[Filing], dict[str, str], dict[str, Any]]:
    """Build normalized corpus and filing_id->finalLink mapping."""

    ensure_dirs()
    filings_path = RAW_DIR / "filings.json"
    url_map_path = RAW_DIR / "filing_url_map.json"
    manifest_path = RAW_DIR / "manifest.json"

    if (
        not force_download
        and filings_path.exists()
        and url_map_path.exists()
        and manifest_path.exists()
    ):
        rows = json.loads(filings_path.read_text(encoding="utf-8"))
        filings = [Filing(**row) for row in rows]
        url_map = json.loads(url_map_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_repo") == DEERFIELD_DATASET_ID and len(filings) >= n_companies:
            logger.info("Using cached deerfield corpus from {}", filings_path)
            return filings[:n_companies], url_map, manifest

    raw_rows, schema = download_deerfield_rows()
    records = [_record_from_row(row) for row in raw_rows]

    filings: list[Filing] = []
    url_map: dict[str, str] = {}
    for record in records:
        filing = _to_filing(record)
        if filing is None:
            continue
        filings.append(filing)
        if record.final_link:
            url_map[filing.filing_id] = record.final_link

    if not filings:
        raise RuntimeError("No valid 10-K filings were produced from deerfield dataset.")

    sampled = _sample_filings(filings, n_companies=n_companies)
    sampled_url_map = {f.filing_id: url_map.get(f.filing_id, "") for f in sampled}

    schema["n_valid_10k_filings"] = len(filings)
    schema["n_sampled_filings"] = len(sampled)

    save_deerfield_artifacts(sampled, sampled_url_map, schema)
    logger.info("Built deerfield corpus with {} sampled filings", len(sampled))
    return sampled, sampled_url_map, schema
