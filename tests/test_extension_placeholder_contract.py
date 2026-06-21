"""Contract checks for extension placeholder seeding script."""

from pathlib import Path


def test_seed_extensions_includes_new_multimodal_variants() -> None:
    script = Path("scripts/seed_extension_placeholders.py").read_text(encoding="utf-8")
    assert "multimodal_ocr_rag" in script
    assert "multimodal_vision_rag" in script
    assert "seed_domain_adapter_placeholders" in script
    assert '"judge_model": "granite4.1:8b"' in script
