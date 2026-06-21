"""Contract tests for additive OCR and vision multimodal helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.extensions.multimodal_ocr import build_glm_ocr_command, run_glm_ocr
from src.extensions.multimodal_vision import run_qwen_vision


def test_build_glm_ocr_command_shape() -> None:
    cmd = build_glm_ocr_command(image_path="/tmp/sample.png", model="glm-ocr:latest")
    assert cmd[:3] == ["ollama", "run", "glm-ocr:latest"]
    assert "/tmp/sample.png" in cmd


def test_run_glm_ocr_raises_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyProc:
        returncode = 1
        stderr = "mock failure"
        stdout = ""

    def _fake_run(*args: Any, **kwargs: Any) -> DummyProc:
        return DummyProc()

    monkeypatch.setattr("src.extensions.multimodal_ocr.subprocess.run", _fake_run)

    with pytest.raises(RuntimeError, match="glm-ocr command failed"):
        run_glm_ocr(image_path=Path("/tmp/sample.png"), model="glm-ocr:latest")


def test_run_qwen_vision_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _DummyClient:
        def chat(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {
                "message": {
                    "content": (
                        '{"summary":"ok","entities":["A"],'
                        '"numeric_signals":["10%"],"risk_signals":["volatility"]}'
                    )
                }
            }

    monkeypatch.setattr("src.extensions.multimodal_vision.get_client", lambda: _DummyClient())
    payload = run_qwen_vision(image_path="/tmp/sample.png", model="qwen3.5:4b")

    assert payload["summary"] == "ok"
    assert captured["model"] == "qwen3.5:4b"
    assert captured["messages"][0]["images"] == ["/tmp/sample.png"]
