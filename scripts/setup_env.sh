#!/usr/bin/env bash
set -euo pipefail

# Reproducible local environment bootstrap (not auto-run by repository).
cd "$(dirname "$0")/.."

uv python install 3.12.10
uv venv --python 3.12.10 .venv
source .venv/bin/activate
uv sync

echo "Environment ready. Activate with: source .venv/bin/activate"
echo "Optional adapter stage deps: uv sync --extra adapter"
