"""Composable multimodal unit builder (HTML + OCR + Vision)."""

from __future__ import annotations

from pathlib import Path

from src.extensions.multimodal import MultimodalUnit, build_multimodal_units_from_html_map
from src.extensions.multimodal_ocr import build_ocr_units_from_image_map
from src.extensions.multimodal_vision import build_vision_units_from_image_map
from src.ingest import Filing


def build_multimodal_units_v2(
    filings: list[Filing],
    filing_html_sources: dict[str, str | Path] | None = None,
    filing_image_sources: dict[str, list[str | Path]] | None = None,
    include_html_channels: bool = True,
    include_ocr_channels: bool = True,
    include_vision_channels: bool = True,
) -> list[MultimodalUnit]:
    """Build a unified multimodal corpus with optional channel toggles."""
    units: list[MultimodalUnit] = []

    if include_html_channels and filing_html_sources:
        units.extend(
            build_multimodal_units_from_html_map(
                filings=filings,
                filing_html_sources=filing_html_sources,
            )
        )

    if include_ocr_channels and filing_image_sources:
        units.extend(
            build_ocr_units_from_image_map(
                filings=filings,
                filing_image_sources=filing_image_sources,
            )
        )

    if include_vision_channels and filing_image_sources:
        units.extend(
            build_vision_units_from_image_map(
                filings=filings,
                filing_image_sources=filing_image_sources,
            )
        )

    return units
