"""OCR-backed multimodal extraction using local `ollama run glm-ocr`.

This extension is additive and does not alter existing multimodal HTML parsing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from src.config import SETTINGS
from src.extensions.multimodal import MultimodalUnit
from src.ingest import Filing

DEFAULT_OCR_PROMPT = (
    "Extract all visible text from this filing visual. "
    "Preserve numbers, table rows, and labels. Return plain text only."
)


def build_glm_ocr_command(
    image_path: str | Path,
    prompt: str = DEFAULT_OCR_PROMPT,
    model: str | None = None,
) -> list[str]:
    """Create a safe argv command for `ollama run` OCR execution."""
    model = model or SETTINGS.ocr_model
    return ["ollama", "run", model, str(Path(image_path)), prompt]


def run_glm_ocr(
    image_path: str | Path,
    prompt: str = DEFAULT_OCR_PROMPT,
    model: str | None = None,
    timeout_seconds: int = 180,
) -> str:
    """Run OCR on a single image via `ollama run glm-ocr`.

    Returns OCR text. Raises `RuntimeError` on command failures.
    """
    cmd = build_glm_ocr_command(image_path=image_path, prompt=prompt, model=model)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"glm-ocr command failed: {stderr or 'unknown error'}")
    return (proc.stdout or "").strip()


def _normalize_image_list(values: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for value in values:
        path = Path(value)
        if path.exists() and path.is_file():
            out.append(path)
    return out


def build_ocr_units_for_filing(
    filing: Filing,
    image_paths: Iterable[str | Path],
    model: str | None = None,
) -> list[MultimodalUnit]:
    """Build OCR text units for one filing from related image paths."""
    paths = _normalize_image_list(image_paths)
    units: list[MultimodalUnit] = []

    for idx, image_path in enumerate(paths):
        text = run_glm_ocr(image_path=image_path, model=model)
        if not text.strip():
            continue

        units.append(
            MultimodalUnit(
                unit_id=f"{filing.filing_id}__ocr__{idx:03d}",
                filing_id=filing.filing_id,
                ticker=filing.ticker,
                company_name=filing.company_name,
                modality="ocr_text",
                section="OCR Visual Evidence",
                text=text,
                metadata={
                    "image_path": str(image_path),
                    "ocr_model": model or SETTINGS.ocr_model,
                    "source": "ollama_run_glm_ocr",
                },
            )
        )
    return units


def build_ocr_units_from_image_map(
    filings: list[Filing],
    filing_image_sources: dict[str, list[str | Path]],
    model: str | None = None,
) -> list[MultimodalUnit]:
    """Build OCR units from `filing_id -> image paths` mapping."""
    filing_by_id = {f.filing_id: f for f in filings}
    output: list[MultimodalUnit] = []

    for filing_id, image_paths in filing_image_sources.items():
        filing = filing_by_id.get(filing_id)
        if not filing:
            continue
        output.extend(
            build_ocr_units_for_filing(
                filing=filing,
                image_paths=image_paths,
                model=model,
            )
        )
    return output
