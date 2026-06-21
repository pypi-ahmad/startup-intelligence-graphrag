#!/usr/bin/env python3
"""Pipeline CLI for placeholder seeding and real execution modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import archive_existing_artifacts, run_execution_pipeline, seed_placeholder_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Startup Intelligence GraphRAG pipeline runner")
    parser.add_argument(
        "--mode",
        choices=["placeholder", "execute"],
        default="placeholder",
        help="placeholder seeds artifacts without model execution; execute runs full pipeline",
    )
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        help="archive existing artifacts before writing new outputs",
    )
    parser.add_argument(
        "--n-companies",
        type=int,
        default=None,
        help="override default number of companies for execute mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.archive_existing:
        archive_path = archive_existing_artifacts()
        logger.info("Archived existing artifacts to {}", archive_path)

    if args.mode == "placeholder":
        paths = seed_placeholder_artifacts()
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
        return

    summary = run_execution_pipeline(n_companies=args.n_companies)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
