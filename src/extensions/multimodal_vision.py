"""Vision-backed multimodal extraction using `qwen3.5:4b` via Ollama chat.

This extension is additive and does not alter existing multimodal HTML parsing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import ollama

from src.config import SETTINGS
from src.extensions.multimodal import MultimodalUnit
from src.ingest import Filing
from src.ollama_client import get_client

VISION_PROMPT = """You are extracting startup/company intelligence from a SEC filing visual.
Analyze the image and return strict JSON with keys:
{
  "summary": "short plain-language summary",
  "entities": ["company/product/person/competitor names"],
  "numeric_signals": ["important numbers with units or context"],
  "risk_signals": ["risk clues from chart/table/diagram labels"]
}
"""


def run_qwen_vision(
    image_path: str | Path,
    prompt: str = VISION_PROMPT,
    model: str | None = None,
) -> dict[str, object]:
    """Analyze one image with a local vision model and return parsed JSON."""
    model = model or SETTINGS.vision_model
    if SETTINGS.ollama_host:
        os.environ.setdefault("OLLAMA_HOST", SETTINGS.ollama_host)

    client = get_client()
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(Path(image_path))],
            }
        ],
        format="json",
        options={"temperature": 0},
    )
    raw = response["message"]["content"]
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"summary": str(raw), "entities": [], "numeric_signals": [], "risk_signals": []}
    return parsed


def _normalize_image_list(values: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for value in values:
        path = Path(value)
        if path.exists() and path.is_file():
            out.append(path)
    return out


def _vision_payload_to_text(payload: dict[str, object]) -> str:
    summary = str(payload.get("summary", "")).strip()
    entities = payload.get("entities", [])
    numeric = payload.get("numeric_signals", [])
    risk = payload.get("risk_signals", [])

    def _join(value: object) -> str:
        if isinstance(value, list):
            return "; ".join(str(v) for v in value if str(v).strip())
        return str(value or "")

    lines = [
        f"Summary: {summary}",
        f"Entities: {_join(entities)}",
        f"Numeric signals: {_join(numeric)}",
        f"Risk signals: {_join(risk)}",
    ]
    return "\n".join(lines).strip()


def build_vision_units_for_filing(
    filing: Filing,
    image_paths: Iterable[str | Path],
    model: str | None = None,
) -> list[MultimodalUnit]:
    """Build vision-text units for one filing from related image paths."""
    paths = _normalize_image_list(image_paths)
    units: list[MultimodalUnit] = []

    for idx, image_path in enumerate(paths):
        payload = run_qwen_vision(image_path=image_path, model=model)
        text = _vision_payload_to_text(payload)
        if not text.strip():
            continue

        units.append(
            MultimodalUnit(
                unit_id=f"{filing.filing_id}__vision__{idx:03d}",
                filing_id=filing.filing_id,
                ticker=filing.ticker,
                company_name=filing.company_name,
                modality="vision_text",
                section="Vision Visual Evidence",
                text=text,
                metadata={
                    "image_path": str(image_path),
                    "vision_model": model or SETTINGS.vision_model,
                    "vision_payload": payload,
                },
            )
        )
    return units


def build_vision_units_from_image_map(
    filings: list[Filing],
    filing_image_sources: dict[str, list[str | Path]],
    model: str | None = None,
) -> list[MultimodalUnit]:
    """Build vision units from `filing_id -> image paths` mapping."""
    filing_by_id = {f.filing_id: f for f in filings}
    output: list[MultimodalUnit] = []

    for filing_id, image_paths in filing_image_sources.items():
        filing = filing_by_id.get(filing_id)
        if not filing:
            continue
        output.extend(
            build_vision_units_for_filing(
                filing=filing,
                image_paths=image_paths,
                model=model,
            )
        )
    return output
