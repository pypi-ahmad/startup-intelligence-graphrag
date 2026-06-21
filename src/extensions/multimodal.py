"""Multimodal unit construction for SEC filing intelligence.

This extension creates additional retrievable units from filing HTML:
- table units (financial tables and structured disclosures)
- figure-text units (captions/alt text/diagram labels)

No existing ingestion code is modified.
"""

from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from src.ingest import Filing


@dataclass
class MultimodalUnit:
    unit_id: str
    filing_id: str
    ticker: str
    company_name: str
    modality: str  # text | table | figure_text | ocr_text | vision_text
    section: str
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _table_to_text(df: pd.DataFrame, max_rows: int = 25, max_cols: int = 12) -> str:
    """Convert dataframe into compact retrieval-friendly narrative text."""
    clipped = df.iloc[:max_rows, :max_cols].copy()
    clipped.columns = [str(col).strip() for col in clipped.columns]

    lines = ["Table headers: " + " | ".join(clipped.columns)]
    for idx, row in clipped.iterrows():
        vals = [str(v).strip() for v in row.tolist()]
        lines.append(f"row_{idx}: " + " | ".join(vals))
    return "\n".join(lines)


def _extract_tables(html_text: str, filing: Filing) -> list[MultimodalUnit]:
    units: list[MultimodalUnit] = []

    try:
        tables = pd.read_html(io.StringIO(html_text))
    except ValueError:
        tables = []

    for idx, df in enumerate(tables):
        if df.empty:
            continue
        table_text = _table_to_text(df)
        units.append(
            MultimodalUnit(
                unit_id=f"{filing.filing_id}__table__{idx:03d}",
                filing_id=filing.filing_id,
                ticker=filing.ticker,
                company_name=filing.company_name,
                modality="table",
                section="Financial Tables",
                text=table_text,
                metadata={
                    "table_index": idx,
                    "n_rows": int(df.shape[0]),
                    "n_cols": int(df.shape[1]),
                },
            )
        )
    return units


def _extract_figure_text(html_text: str, filing: Filing) -> list[MultimodalUnit]:
    """Extract figure and diagram text channels from filing HTML."""
    soup = BeautifulSoup(html_text, "html.parser")
    units: list[MultimodalUnit] = []

    figure_idx = 0

    for figure in soup.find_all("figure"):
        caption = " ".join(figure.stripped_strings)
        if not caption:
            continue
        units.append(
            MultimodalUnit(
                unit_id=f"{filing.filing_id}__figure__{figure_idx:03d}",
                filing_id=filing.filing_id,
                ticker=filing.ticker,
                company_name=filing.company_name,
                modality="figure_text",
                section="Figure / Diagram",
                text=caption,
                metadata={"source": "figure"},
            )
        )
        figure_idx += 1

    for image in soup.find_all("img"):
        alt = image.get("alt", "")
        title = image.get("title", "")
        aria = image.get("aria-label", "")
        text = " ".join(x for x in [alt, title, aria] if x)
        if not text.strip():
            continue

        units.append(
            MultimodalUnit(
                unit_id=f"{filing.filing_id}__image_text__{figure_idx:03d}",
                filing_id=filing.filing_id,
                ticker=filing.ticker,
                company_name=filing.company_name,
                modality="figure_text",
                section="Image Alt/Title",
                text=text.strip(),
                metadata={"source": "img", "tag_attrs": dict(image.attrs)},
            )
        )
        figure_idx += 1

    return units


def filing_html_to_multimodal_units(
    filing: Filing,
    html_text: str,
) -> list[MultimodalUnit]:
    """Extract multimodal units from one filing HTML document."""
    units: list[MultimodalUnit] = []
    units.extend(_extract_tables(html_text=html_text, filing=filing))
    units.extend(_extract_figure_text(html_text=html_text, filing=filing))
    return units


def _read_html_input(value: str | Path) -> str:
    path = Path(value)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8", errors="ignore")
    return str(value)


def build_multimodal_units_from_html_map(
    filings: list[Filing],
    filing_html_sources: dict[str, str | Path],
) -> list[MultimodalUnit]:
    """Build multimodal units from map: filing_id -> html string/path."""
    filing_by_id = {f.filing_id: f for f in filings}
    output: list[MultimodalUnit] = []

    for filing_id, html_source in filing_html_sources.items():
        filing = filing_by_id.get(filing_id)
        if not filing:
            continue
        html_text = _read_html_input(html_source)
        output.extend(filing_html_to_multimodal_units(filing=filing, html_text=html_text))

    return output
