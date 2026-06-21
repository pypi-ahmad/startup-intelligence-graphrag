#!/usr/bin/env python3
"""Execute project notebooks end-to-end and save executed copies.

This runner executes notebooks with the project root as cwd so relative
artifact paths (for example ``artifacts/eval/...``) resolve consistently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from loguru import logger
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DEFAULT_OUTPUT_DIR = NOTEBOOKS_DIR / "executed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute tutorial notebooks")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory where executed notebooks are written",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="per-cell timeout in seconds (use -1 to disable)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.ipynb",
        help="glob pattern for notebook selection inside notebooks/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    notebooks = sorted(p for p in NOTEBOOKS_DIR.glob(args.pattern) if p.is_file())
    if not notebooks:
        raise FileNotFoundError(f"No notebooks found for pattern: {args.pattern}")

    logger.info("Executing {} notebook(s)", len(notebooks))
    ep = ExecutePreprocessor(timeout=args.timeout, kernel_name="python3")

    for nb_path in notebooks:
        logger.info("Executing {}", nb_path.name)
        with nb_path.open("r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        ep.preprocess(nb, {"metadata": {"path": str(PROJECT_ROOT)}})

        out_path = output_dir / nb_path.name
        with out_path.open("w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        logger.info("Wrote {}", out_path)

    logger.info("Notebook execution complete")


if __name__ == "__main__":
    main()
