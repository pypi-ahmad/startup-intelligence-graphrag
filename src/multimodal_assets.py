"""Multimodal asset preparation for real SEC filing runs."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from src.config import ARTIFACTS_DIR
from src.ingest import Filing


MULTIMODAL_DIR = ARTIFACTS_DIR / "multimodal"
HTML_DIR = MULTIMODAL_DIR / "html"
TABLE_IMG_DIR = MULTIMODAL_DIR / "table_images"


def _http_get(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "StartupIntelligenceGraphRAG/1.0"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content = resp.read()
    return content.decode("utf-8", errors="ignore")


def fetch_filing_html_map(
    filings: list[Filing],
    filing_url_map: dict[str, str],
    max_filings: int | None = None,
) -> dict[str, str]:
    """Download filing HTML pages and return map filing_id->html text."""

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    selected = filings[: max_filings or len(filings)]
    output: dict[str, str] = {}

    for filing in selected:
        url = filing_url_map.get(filing.filing_id, "")
        if not url:
            continue
        html_path = HTML_DIR / f"{filing.filing_id}.html"
        if html_path.exists():
            html = html_path.read_text(encoding="utf-8", errors="ignore")
            if html.strip():
                output[filing.filing_id] = html
                continue
        try:
            html = _http_get(url)
        except urllib.error.URLError as exc:
            logger.warning("Failed to fetch HTML for {}: {}", filing.filing_id, exc)
            continue
        html_path.write_text(html, encoding="utf-8")
        output[filing.filing_id] = html
    logger.info("Fetched HTML for {} filings", len(output))
    return output


def _clean_df(df: pd.DataFrame, max_rows: int = 12, max_cols: int = 8) -> pd.DataFrame:
    clipped = df.iloc[:max_rows, :max_cols].copy()
    clipped.columns = [str(c)[:40] for c in clipped.columns]
    clipped = clipped.fillna("")
    for col in clipped.columns:
        clipped[col] = clipped[col].astype(str).str.slice(0, 80)
    return clipped


def _render_table_png(df: pd.DataFrame, title: str, output_path: Path) -> None:
    rows, cols = df.shape
    width = max(10, min(24, cols * 2.2))
    height = max(3, min(16, rows * 0.5 + 1.5))

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax.set_title(title[:140], fontsize=11, pad=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_table_image_sources(
    filings: list[Filing],
    filing_html_map: dict[str, str],
    max_tables_per_filing: int = 1,
    max_filings: int | None = None,
) -> dict[str, list[str]]:
    """Create image assets from filing HTML tables for OCR/vision channels."""

    TABLE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    selected = filings[: max_filings or len(filings)]
    image_map: dict[str, list[str]] = {}

    for filing in selected:
        html = filing_html_map.get(filing.filing_id, "")
        if not html:
            continue
        try:
            tables = pd.read_html(io.StringIO(html))
        except ValueError:
            tables = []
        if not tables:
            continue

        paths: list[str] = []
        kept = 0
        for idx, df in enumerate(tables):
            if df.empty:
                continue
            cleaned = _clean_df(df)
            if cleaned.shape[0] < 2 or cleaned.shape[1] < 2:
                continue
            out_path = TABLE_IMG_DIR / f"{filing.filing_id}__table_{idx:03d}.png"
            if not out_path.exists():
                _render_table_png(
                    cleaned,
                    title=f"{filing.ticker} filing table {idx}",
                    output_path=out_path,
                )
            paths.append(str(out_path))
            kept += 1
            if kept >= max_tables_per_filing:
                break

        if paths:
            image_map[filing.filing_id] = paths

    manifest = MULTIMODAL_DIR / "image_map.json"
    manifest.write_text(json.dumps(image_map, indent=2), encoding="utf-8")
    logger.info("Prepared table-image assets for {} filings", len(image_map))
    return image_map


def build_text_snapshot_images(
    filings: list[Filing],
    max_filings: int | None = None,
) -> dict[str, list[str]]:
    """Fallback visual asset builder when HTML/table sources are unavailable.

    Renders real filing text snippets into image cards so OCR/vision stages still
    execute on filing-derived visual inputs.
    """
    TABLE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    selected = filings[: max_filings or len(filings)]
    output: dict[str, list[str]] = {}

    for filing in selected:
        business = filing.sections.get("Business", [])
        risk = filing.sections.get("Risk Factors", [])
        lines = (business[:6] + risk[:4])[:8]
        if not lines:
            lines = filing.all_sentences()[:8]
        if not lines:
            continue

        text = "\n".join(f"- {line[:160]}" for line in lines)
        out_path = TABLE_IMG_DIR / f"{filing.filing_id}__text_snapshot_000.png"

        if not out_path.exists():
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis("off")
            ax.set_title(f"{filing.ticker} filing text snapshot", fontsize=13, pad=12)
            ax.text(
                0.01,
                0.98,
                text,
                ha="left",
                va="top",
                fontsize=9,
                wrap=True,
                transform=ax.transAxes,
            )
            fig.tight_layout()
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close(fig)

        output[filing.filing_id] = [str(out_path)]

    manifest = MULTIMODAL_DIR / "text_snapshot_image_map.json"
    manifest.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Prepared text-snapshot images for {} filings", len(output))
    return output


def save_html_map_manifest(filing_html_map: dict[str, str]) -> Path:
    MULTIMODAL_DIR.mkdir(parents=True, exist_ok=True)
    path = MULTIMODAL_DIR / "html_map_manifest.json"
    payload: dict[str, Any] = {
        "n_filings": len(filing_html_map),
        "filing_ids": sorted(filing_html_map.keys()),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
