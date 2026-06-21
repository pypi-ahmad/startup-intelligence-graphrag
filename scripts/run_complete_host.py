#!/usr/bin/env python3
"""Single-command strict host execution for full real project completion."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARTIFACTS_DIR, SETTINGS
from src.extensions.domain_adapter import is_gpu_available
from src.ollama_client import get_client


RUN_MANIFEST_PATH = ARTIFACTS_DIR / "run_completion_manifest.json"
REQUIRED_TECHNIQUES = [
    "vector_baseline",
    "graphrag_local",
    "graphrag_global",
    "graphrag_hybrid",
    "hybrid_sparse_dense",
    "agentic_crag",
    "multimodal_rag",
    "multimodal_ocr_rag",
    "multimodal_vision_rag",
    "multimodal_unified_v2",
]
REQUIRED_OLLAMA_MODELS = [
    SETTINGS.embed_model,
    SETTINGS.generator_model,
    SETTINGS.judge_model,
    SETTINGS.ocr_model,
    SETTINGS.vision_model,
]
PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "comprehensive": {
        "n_companies": 30,
        "top_k": 6,
        "max_local_eval_companies": 10,
        "max_multimodal_filings": 4,
        "max_tables_per_filing": 1,
        "chunk_size": 450,
        "chunk_overlap": 60,
        "max_community_summaries": 6,
        "notebook_timeout": 3600,
    }
}


@dataclass
class StageResult:
    name: str
    status: str
    command: list[str]
    duration_seconds: float
    exit_code: int
    started_utc: str
    ended_utc: str


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _write_manifest(payload: dict[str, Any], path: Path = RUN_MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_model_name(name: str) -> str:
    return name.strip()


def _model_present(required: str, available: set[str]) -> bool:
    if required in available:
        return True
    # Accept exact base-name match if list omits tag (rare).
    required_base = required.split(":", 1)[0]
    return any(item.split(":", 1)[0] == required_base for item in available)


def _assert_hf_access() -> dict[str, Any]:
    if SETTINGS.dataset_repo != "deerfieldgreen/stk-sec-filings" or not SETTINGS.strict_dataset:
        raise RuntimeError(
            "Strict dataset policy must be enabled with dataset_repo='deerfieldgreen/stk-sec-filings'."
        )

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Missing HUGGINGFACE_HUB_TOKEN (or HF_TOKEN).")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.dataset_info(SETTINGS.dataset_repo)
    return {
        "token_env": "HUGGINGFACE_HUB_TOKEN" if os.environ.get("HUGGINGFACE_HUB_TOKEN") else "HF_TOKEN",
        "dataset_repo": SETTINGS.dataset_repo,
        "private": bool(getattr(info, "private", False)),
        "sha": getattr(info, "sha", None),
    }


def _assert_ollama_models() -> dict[str, Any]:
    client = get_client(timeout_seconds=20)
    listing = client.list()
    rows: list[Any] = []
    if isinstance(listing, dict):
        rows = list(listing.get("models", []))
    elif hasattr(listing, "models"):
        rows = list(getattr(listing, "models"))

    available: set[str] = set()
    for row in rows:
        if isinstance(row, dict):
            model_name = str(row.get("model") or row.get("name") or "").strip()
        else:
            model_name = str(getattr(row, "model", "") or getattr(row, "name", "")).strip()
        if model_name:
            available.add(_normalize_model_name(model_name))

    required = list(dict.fromkeys(REQUIRED_OLLAMA_MODELS))
    missing = [name for name in required if not _model_present(name, available)]
    if missing:
        raise RuntimeError(f"Missing Ollama models: {missing}")
    return {"required": required, "available_count": len(available)}


def _assert_adapter_prereqs() -> dict[str, Any]:
    gpu_ok, gpu_msg = is_gpu_available()
    if not gpu_ok:
        raise RuntimeError(f"CUDA preflight failed: {gpu_msg}")

    for module_name in ("unsloth", "peft", "trl"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise RuntimeError(f"Missing adapter dependency '{module_name}': {exc}") from exc

    return {"cuda": gpu_msg, "deps": ["unsloth", "peft", "trl"]}


def _run_stage(name: str, command: list[str]) -> StageResult:
    started = _now()
    start_perf = time.perf_counter()
    logger.info("Running stage '{}' -> {}", name, " ".join(command))
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    ended = _now()
    elapsed = round(time.perf_counter() - start_perf, 2)
    status = "completed" if proc.returncode == 0 else "failed"
    result = StageResult(
        name=name,
        status=status,
        command=command,
        duration_seconds=elapsed,
        exit_code=proc.returncode,
        started_utc=started.isoformat(),
        ended_utc=ended.isoformat(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Stage '{name}' failed with exit code {proc.returncode}")
    return result


def _assert_required_outputs() -> dict[str, Any]:
    eval_dir = ARTIFACTS_DIR / "eval"
    missing_metrics: list[str] = []
    for technique in REQUIRED_TECHNIQUES:
        path = eval_dir / f"{technique}_full_metrics.json"
        if not path.exists():
            missing_metrics.append(str(path))
    if missing_metrics:
        raise RuntimeError(f"Missing required technique metrics: {missing_metrics}")

    run_summary_path = ARTIFACTS_DIR / "run_summary.json"
    if not run_summary_path.exists():
        raise RuntimeError(f"Missing core run summary: {run_summary_path}")
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    if run_summary.get("dataset_repo") != SETTINGS.dataset_repo:
        raise RuntimeError(
            "Strict dataset proof failed: run_summary dataset_repo does not match strict config."
        )

    adapter_summary_path = ARTIFACTS_DIR / "run_summary_domain_adapter.json"
    if not adapter_summary_path.exists():
        raise RuntimeError(f"Missing adapter summary: {adapter_summary_path}")
    adapter_summary = json.loads(adapter_summary_path.read_text(encoding="utf-8"))
    if adapter_summary.get("status") != "completed":
        raise RuntimeError(f"Adapter stage not completed: {adapter_summary}")

    executed_dir = PROJECT_ROOT / "notebooks" / "executed"
    notebook_outputs = sorted(p.name for p in executed_dir.glob("*.ipynb"))
    source_notebooks = sorted(p.name for p in (PROJECT_ROOT / "notebooks").glob("*.ipynb"))
    missing_notebooks = [name for name in source_notebooks if name not in notebook_outputs]
    if missing_notebooks:
        raise RuntimeError(f"Missing executed notebooks: {missing_notebooks}")

    return {
        "required_techniques": REQUIRED_TECHNIQUES,
        "strict_dataset_repo": run_summary.get("dataset_repo"),
        "strict_dataset_verified": True,
        "adapter_status": adapter_summary.get("status"),
        "executed_notebooks": notebook_outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-command host execution for full strict completion")
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default="comprehensive")
    parser.add_argument("--archive-existing", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--manifest-path", type=Path, default=RUN_MANIFEST_PATH)
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    profile = PROFILE_CONFIGS[args.profile]

    started = _now()
    stage_results: list[StageResult] = []
    preflight: dict[str, Any] = {}
    final_status = "failed"
    failure_message = ""

    try:
        preflight["hf_access"] = _assert_hf_access()
        preflight["ollama"] = _assert_ollama_models()
        preflight["adapter"] = _assert_adapter_prereqs()

        core_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_full_real_project.py"),
            "--n-companies",
            str(profile["n_companies"]),
            "--top-k",
            str(profile["top_k"]),
            "--max-local-eval-companies",
            str(profile["max_local_eval_companies"]),
            "--max-multimodal-filings",
            str(profile["max_multimodal_filings"]),
            "--max-tables-per-filing",
            str(profile["max_tables_per_filing"]),
            "--chunk-size",
            str(profile["chunk_size"]),
            "--chunk-overlap",
            str(profile["chunk_overlap"]),
            "--max-community-summaries",
            str(profile["max_community_summaries"]),
        ]
        if args.archive_existing:
            core_cmd.append("--archive-existing")
        if args.force_download:
            core_cmd.append("--force-download")
        stage_results.append(_run_stage("core_pipeline", core_cmd))

        adapter_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_domain_adapter.py"),
            "--mode",
            "execute",
            "--n-companies",
            str(profile["n_companies"]),
            "--k",
            str(profile["top_k"]),
            "--force",
            "--required",
        ]
        stage_results.append(_run_stage("adapter_pipeline", adapter_cmd))

        notebook_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "execute_notebooks.py"),
            "--timeout",
            str(profile["notebook_timeout"]),
        ]
        stage_results.append(_run_stage("notebook_execution", notebook_cmd))

        tests_cmd = [sys.executable, "-m", "pytest", "-q"]
        stage_results.append(_run_stage("tests", tests_cmd))

        post_checks = _assert_required_outputs()
        final_status = "completed"
    except Exception as exc:
        failure_message = str(exc)
        post_checks = {}
        logger.error("Completion orchestration failed: {}", exc)
    finally:
        ended = _now()
        manifest = {
            "status": final_status,
            "profile": args.profile,
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "duration_seconds": round((ended - started).total_seconds(), 2),
            "strict_dataset_expected": SETTINGS.dataset_repo,
            "preflight": preflight,
            "stages": [asdict(stage) for stage in stage_results],
            "post_checks": post_checks,
            "failure_message": failure_message,
        }
        _write_manifest(manifest, path=args.manifest_path)
        print(json.dumps(manifest, indent=2))

    if final_status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
