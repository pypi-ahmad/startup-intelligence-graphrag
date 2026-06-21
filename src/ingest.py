"""Ingestion pipeline for SEC filings from Hugging Face.

Dataset policy is strict by default:
- source must be `deerfieldgreen/stk-sec-filings`
- no synthetic fallback
- hard fail with actionable guidance if dataset access fails
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from src.config import (
    RAW_DIR,
    SEC_SECTION_LABELS,
    SETTINGS,
    USEFUL_SECTIONS,
    ensure_dirs,
)

STRICT_DATASET_REPO = "deerfieldgreen/stk-sec-filings"


@dataclass
class Filing:
    """Normalized filing object consumed by downstream GraphRAG modules."""

    company_name: str
    cik: str
    ticker: str
    exchange: str
    state_of_incorporation: str
    sic: str
    form: str
    filing_date: str
    report_date: str
    filing_id: str
    source_split: str
    source_row_count: int
    sections: dict[str, list[str]] = field(default_factory=dict)

    def all_sentences(self) -> list[str]:
        items: list[str] = []
        for section_sentences in self.sections.values():
            items.extend(section_sentences)
        return items

    def sentence_count(self) -> int:
        return sum(len(v) for v in self.sections.values())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetAccessError(RuntimeError):
    """Raised when strict dataset access policy cannot be satisfied."""


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_safe_str(v) for v in value if _safe_str(v)]
    if isinstance(value, tuple):
        return [_safe_str(v) for v in value if _safe_str(v)]
    text = _safe_str(value)
    if not text:
        return []
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]
    return [text]


def _first_non_empty(record: dict[str, Any], candidates: Iterable[str]) -> str:
    for key in candidates:
        if key in record:
            value = record.get(key)
            if isinstance(value, (list, tuple)):
                values = _as_list(value)
                text = values[0] if values else ""
            else:
                text = _safe_str(value)
            if text:
                return text
    return ""


def _date_or_empty(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text[:10], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text[:10]


def _normalize_section(record: dict[str, Any]) -> str:
    sec_name = _first_non_empty(record, [
        "section_name",
        "section",
        "item_name",
        "item",
        "heading",
        "topic",
    ])
    if sec_name:
        sec_name = sec_name.strip()
        if sec_name.isdigit():
            sec_id = int(sec_name)
            return SEC_SECTION_LABELS.get(sec_id, f"Section_{sec_id}")
        if sec_name.lower().startswith("item "):
            remainder = sec_name.lower().replace("item", "", 1).strip().split(" ")[0]
            try:
                sec_id = int(remainder.replace(".", ""))
                return SEC_SECTION_LABELS.get(sec_id, sec_name)
            except ValueError:
                return sec_name
        return sec_name

    sec_id_raw = _first_non_empty(record, ["section_id", "section_idx", "item_id"])
    if sec_id_raw and sec_id_raw.isdigit():
        sec_id = int(sec_id_raw)
        return SEC_SECTION_LABELS.get(sec_id, f"Section_{sec_id}")

    return "Unknown Section"


def _extract_text(record: dict[str, Any]) -> str:
    text = _first_non_empty(record, [
        "sentence",
        "text",
        "content",
        "chunk_text",
        "paragraph",
        "body",
    ])
    if text:
        return text

    # Some schemas store full sections under nested dict/list.
    possible = record.get("sections")
    if isinstance(possible, list):
        joined = " ".join(_safe_str(x) for x in possible if _safe_str(x))
        return joined
    if isinstance(possible, dict):
        joined = " ".join(_safe_str(v) for v in possible.values() if _safe_str(v))
        return joined
    return ""


def _infer_schema_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({k for r in records[:5000] for k in r.keys()})
    return {
        "sampled_rows": min(len(records), 5000),
        "observed_keys": keys,
    }


def download_raw_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load all requested splits from the configured HF dataset repository."""

    if SETTINGS.dataset_repo != STRICT_DATASET_REPO and SETTINGS.strict_dataset:
        raise DatasetAccessError(
            f"Strict dataset policy violation: dataset_repo must stay '{STRICT_DATASET_REPO}'."
        )

    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - import failure environment-specific
        raise DatasetAccessError(
            "Failed to import datasets package. Install dependencies with `uv sync` before running ingestion."
        ) from exc

    all_rows: list[dict[str, Any]] = []
    per_split_counts: dict[str, int] = {}

    for split in SETTINGS.dataset_splits:
        try:
            logger.info(
                "Loading HF dataset repo='{}' config='{}' split='{}'",
                SETTINGS.dataset_repo,
                SETTINGS.dataset_config,
                split,
            )
            ds = load_dataset(
                SETTINGS.dataset_repo,
                name=SETTINGS.dataset_config,
                split=split,
                trust_remote_code=SETTINGS.dataset_trust_remote_code,
            )
        except Exception as exc:
            raise DatasetAccessError(
                f"Could not load {STRICT_DATASET_REPO} under strict dataset policy. "
                "No fallback dataset is allowed. Verify Hugging Face auth/access and retry."
            ) from exc

        rows = []
        for row in ds:
            item = dict(row)
            item["_source_split"] = split
            rows.append(item)
        per_split_counts[split] = len(rows)
        all_rows.extend(rows)

    if not all_rows:
        raise DatasetAccessError("Dataset loaded successfully but returned zero rows.")

    schema = _infer_schema_stats(all_rows)
    schema["split_counts"] = per_split_counts
    return all_rows, schema


def _filing_key(record: dict[str, Any]) -> str:
    cik = _first_non_empty(record, ["cik", "cik_number", "issuer_cik"])
    ticker = _first_non_empty(record, ["ticker", "symbol", "trading_symbol", "tickers"])
    filing_id = _first_non_empty(record, ["filing_id", "docID", "accession_number", "accessionNumber"])
    report_date = _date_or_empty(_first_non_empty(record, ["report_date", "reportDate", "period_end", "periodEndDate"]))
    form = _first_non_empty(record, ["form", "form_type", "filing_type", "document_type"])

    parts = [x for x in [filing_id, cik, ticker, report_date, form] if x]
    if parts:
        return "||".join(parts)

    # last-resort key: stable hash over available fields
    name = _first_non_empty(record, ["company_name", "name", "registrant_name", "issuer_name"])
    return f"fallback||{name[:60]}"


def _make_filing_id(ticker: str, form: str, report_date: str, fallback_key: str) -> str:
    ticker_safe = ticker or "UNK"
    form_safe = (form or "FORM").replace("/", "-").replace(" ", "")
    year = report_date[:4] if report_date else "unknown"
    import hashlib

    digest = hashlib.sha1(fallback_key.encode("utf-8")).hexdigest()
    suffix = int(digest[:8], 16) % 100000
    return f"{ticker_safe}_{form_safe}_{year}_{suffix:05d}"


def group_into_filings(records: list[dict[str, Any]]) -> list[Filing]:
    """Normalize row-level dataset records into filing-level documents."""

    grouped: dict[str, dict[str, Any]] = {}

    for row in records:
        key = _filing_key(row)
        filing = grouped.setdefault(
            key,
            {
                "meta": {
                    "company_name": _first_non_empty(row, ["company_name", "name", "registrant_name", "issuer_name"]),
                    "cik": _first_non_empty(row, ["cik", "cik_number", "issuer_cik"]),
                    "ticker": _first_non_empty(row, ["ticker", "symbol", "trading_symbol"]),
                    "exchange": _first_non_empty(row, ["exchange", "primary_exchange", "exchanges"]),
                    "state_of_incorporation": _first_non_empty(row, ["state_of_incorporation", "stateOfIncorporation", "state"]),
                    "sic": _first_non_empty(row, ["sic", "sic_code"]),
                    "form": _first_non_empty(row, ["form", "form_type", "filing_type", "document_type"]),
                    "filing_date": _date_or_empty(_first_non_empty(row, ["filing_date", "filingDate", "filed_at", "filedAt"])),
                    "report_date": _date_or_empty(_first_non_empty(row, ["report_date", "reportDate", "period_end", "periodEndDate"])),
                    "source_split": _first_non_empty(row, ["_source_split", "split"]) or "unknown",
                },
                "sections": defaultdict(list),
                "row_count": 0,
            },
        )

        # Fill missing metadata opportunistically from later rows.
        for mkey, candidates in {
            "company_name": ["company_name", "name", "registrant_name", "issuer_name"],
            "cik": ["cik", "cik_number", "issuer_cik"],
            "ticker": ["ticker", "symbol", "trading_symbol"],
            "exchange": ["exchange", "primary_exchange", "exchanges"],
            "state_of_incorporation": ["state_of_incorporation", "stateOfIncorporation", "state"],
            "sic": ["sic", "sic_code"],
            "form": ["form", "form_type", "filing_type", "document_type"],
            "filing_date": ["filing_date", "filingDate", "filed_at", "filedAt"],
            "report_date": ["report_date", "reportDate", "period_end", "periodEndDate"],
        }.items():
            if not filing["meta"].get(mkey):
                value = _first_non_empty(row, candidates)
                filing["meta"][mkey] = _date_or_empty(value) if mkey.endswith("date") else value

        text = _extract_text(row)
        if not text:
            continue

        section_label = _normalize_section(row)
        section_id_raw = _first_non_empty(row, ["section", "section_id", "item_id"])
        if section_id_raw.isdigit() and int(section_id_raw) not in USEFUL_SECTIONS:
            continue

        filing["sections"][section_label].append(text)
        filing["row_count"] += 1

    filings: list[Filing] = []
    invalid_rows = 0

    for key, payload in grouped.items():
        meta = payload["meta"]
        sections = dict(payload["sections"])

        if not sections:
            continue
        if not meta.get("company_name") or not meta.get("ticker"):
            invalid_rows += 1
            continue

        form = meta.get("form") or ""
        if form and "10-k" not in form.lower():
            # Keep 10-K centric corpus for predictable tutorial behavior.
            continue

        filing_id = _make_filing_id(
            ticker=meta.get("ticker", ""),
            form=form or "10-K",
            report_date=meta.get("report_date", ""),
            fallback_key=key,
        )

        filing = Filing(
            company_name=meta.get("company_name", ""),
            cik=meta.get("cik", ""),
            ticker=meta.get("ticker", ""),
            exchange=meta.get("exchange", ""),
            state_of_incorporation=meta.get("state_of_incorporation", ""),
            sic=meta.get("sic", ""),
            form=form or "10-K",
            filing_date=meta.get("filing_date", ""),
            report_date=meta.get("report_date", ""),
            filing_id=filing_id,
            source_split=meta.get("source_split", "unknown"),
            source_row_count=int(payload["row_count"]),
            sections=sections,
        )
        filings.append(filing)

    if invalid_rows > 0:
        logger.warning("Dropped {} grouped filings due to missing company/ticker metadata", invalid_rows)

    if not filings:
        raise DatasetAccessError(
            "No usable filings after normalization. Verify dataset schema mapping in src/ingest.py."
        )

    return filings


def truncate_and_filter(
    filings: list[Filing],
    max_sentences: int | None = None,
    min_sentences: int | None = None,
) -> list[Filing]:
    """Drop tiny filings and proportionally trim very large filings."""

    max_sentences = max_sentences or SETTINGS.max_sentences_per_filing
    min_sentences = min_sentences or SETTINGS.min_sentences_per_filing

    output: list[Filing] = []
    for filing in filings:
        total = filing.sentence_count()
        if total < min_sentences:
            continue

        if total > max_sentences:
            ratio = max_sentences / total
            new_sections: dict[str, list[str]] = {}
            for section_name, sentences in filing.sections.items():
                keep = max(1, int(len(sentences) * ratio))
                new_sections[section_name] = sentences[:keep]
            filing.sections = new_sections

        output.append(filing)

    if not output:
        raise DatasetAccessError(
            "All filings were filtered out by min/max sentence constraints. Adjust settings and retry."
        )

    return output


def _latest_per_ticker(filings: list[Filing]) -> dict[str, Filing]:
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
    return by_ticker


def sample_companies(filings: list[Filing], n_companies: int | None = None) -> list[Filing]:
    """Select one filing per ticker, then sample deterministic subset."""

    n_companies = n_companies or SETTINGS.default_n_companies
    latest = _latest_per_ticker(filings)
    candidates = list(latest.values())

    # rank by section richness then sample from top pool for diversity.
    ranked = sorted(candidates, key=lambda x: x.sentence_count(), reverse=True)
    if n_companies >= len(ranked):
        return ranked

    pool_size = min(len(ranked), max(n_companies * 2, n_companies))
    pool = ranked[:pool_size]
    rng = random.Random(SETTINGS.company_selection_seed)
    rng.shuffle(pool)
    selected = sorted(pool[:n_companies], key=lambda x: x.ticker)
    return selected


def save_filings(
    filings: list[Filing],
    schema_snapshot: dict[str, Any],
    path: Path | None = None,
) -> tuple[Path, Path]:
    """Persist filings and manifest metadata."""

    ensure_dirs()
    path = path or (RAW_DIR / "filings.json")
    manifest_path = RAW_DIR / "manifest.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump([f.to_dict() for f in filings], file, indent=2)

    manifest = {
        "dataset_repo": SETTINGS.dataset_repo,
        "dataset_config": SETTINGS.dataset_config,
        "dataset_splits": list(SETTINGS.dataset_splits),
        "strict_dataset": SETTINGS.strict_dataset,
        "n_filings": len(filings),
        "tickers": sorted({f.ticker for f in filings}),
        "schema_snapshot": schema_snapshot,
        "placeholder_mode": SETTINGS.placeholder_mode,
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    logger.info("Saved {} filings to {}", len(filings), path)
    logger.info("Saved manifest to {}", manifest_path)
    return path, manifest_path


def load_filings(path: Path | None = None) -> list[Filing]:
    """Load normalized filings from disk."""

    path = path or (RAW_DIR / "filings.json")
    with open(path, "r", encoding="utf-8") as file:
        rows = json.load(file)
    filings = [Filing(**row) for row in rows]
    logger.info("Loaded {} filings from {}", len(filings), path)
    return filings


def build_corpus(
    n_companies: int | None = None,
    force_download: bool = False,
) -> list[Filing]:
    """End-to-end corpus build under strict dataset policy."""

    ensure_dirs()
    if SETTINGS.strict_dataset and SETTINGS.dataset_repo != STRICT_DATASET_REPO:
        raise DatasetAccessError(
            f"Strict dataset policy violation: expected dataset_repo='{STRICT_DATASET_REPO}'."
        )

    if SETTINGS.dataset_repo == STRICT_DATASET_REPO:
        # Delegate to deerfield-specific normalization to preserve filing-level section parsing.
        from src.ingest_deerfield import build_deerfield_corpus

        filings, _, _ = build_deerfield_corpus(
            n_companies=n_companies or SETTINGS.default_n_companies,
            force_download=force_download,
        )
        return filings

    output_path = RAW_DIR / "filings.json"
    if output_path.exists() and not force_download:
        logger.info("Using cached filings from {}", output_path)
        return load_filings(output_path)

    raw_records, schema_snapshot = download_raw_records()
    filings = group_into_filings(raw_records)
    filings = truncate_and_filter(filings)
    filings = sample_companies(filings, n_companies=n_companies)
    save_filings(filings, schema_snapshot=schema_snapshot, path=output_path)
    return filings


if __name__ == "__main__":
    corpus = build_corpus()
    print(f"Built corpus with {len(corpus)} filings")
    for filing in corpus[:10]:
        print(
            f"- {filing.ticker:>6} | {filing.company_name[:38]:38} | "
            f"{filing.report_date or filing.filing_date} | sentences={filing.sentence_count()}"
        )
