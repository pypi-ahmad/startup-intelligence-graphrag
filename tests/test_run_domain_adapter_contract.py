"""Contract checks for optional domain-adapter CLI."""

from pathlib import Path


def test_run_domain_adapter_script_has_expected_modes() -> None:
    script = Path("scripts/run_domain_adapter.py").read_text(encoding="utf-8")
    assert 'choices=["placeholder", "execute"]' in script
    assert "--force" in script
    assert "--required" in script
    assert "SIRAG_ADAPTER_ENABLE" in script
