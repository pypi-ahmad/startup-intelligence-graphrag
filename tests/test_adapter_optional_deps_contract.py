"""Contract test for optional adapter dependency group."""

from pathlib import Path


def test_adapter_extra_declared_in_pyproject() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "adapter = [" in text
    assert '"unsloth"' in text
    assert '"trl==0.22.2"' in text
    assert '"peft>=' in text
