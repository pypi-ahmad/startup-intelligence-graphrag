"""Contract checks for single-command host completion runner."""

from pathlib import Path


def test_complete_host_runner_contains_strict_and_mandatory_stage_contract() -> None:
    script = Path("scripts/run_complete_host.py").read_text(encoding="utf-8")
    assert "HUGGINGFACE_HUB_TOKEN" in script
    assert "REQUIRED_OLLAMA_MODELS" in script
    assert "is_gpu_available" in script
    assert "run_full_real_project.py" in script
    assert "run_domain_adapter.py" in script
    assert "--required" in script
    assert "execute_notebooks.py" in script
    assert '["-m", "pytest", "-q"]' in script or "pytest" in script
    assert "run_completion_manifest.json" in script
