"""Contract tests for optional domain-adapter settings."""

from src.config import SETTINGS, as_dict


def test_adapter_stage_defaults_are_opt_in() -> None:
    assert SETTINGS.adapter_enable is False
    assert SETTINGS.adapter_base_model


def test_adapter_settings_are_exported_in_config_manifest() -> None:
    payload = as_dict()
    assert "adapter_base_model" in payload
    assert "adapter_enable" in payload
    assert "adapter_lora_r" in payload
