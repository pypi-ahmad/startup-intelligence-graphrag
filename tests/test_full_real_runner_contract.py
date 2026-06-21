"""Contract checks for full real-run orchestrator script."""

from pathlib import Path


def test_full_runner_references_selected_dataset_and_models() -> None:
    script = Path("scripts/run_full_real_project.py").read_text(encoding="utf-8")
    assert "build_corpus(" in script
    assert "SETTINGS.dataset_repo" in script
    assert "build_deerfield_corpus" not in script
    assert "glm-ocr" in script or "multimodal_ocr_rag" in script
    assert "qwen3.5:4b" in script or "multimodal_vision_rag" in script
    assert "granite4.1:8b" in script or "SETTINGS.judge_model" in script
