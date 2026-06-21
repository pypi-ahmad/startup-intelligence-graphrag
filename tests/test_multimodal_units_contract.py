"""Contract tests for multimodal unit extraction helpers."""

from src.extensions.multimodal import MultimodalUnit


def test_multimodal_unit_has_expected_modalities() -> None:
    unit = MultimodalUnit(
        unit_id="u1",
        filing_id="f1",
        ticker="AMD",
        company_name="Advanced Micro Devices",
        modality="table",
        section="Financial Tables",
        text="Table headers: revenue | year",
        metadata={"table_index": 0},
    )
    assert unit.modality in {"text", "table", "figure_text", "ocr_text", "vision_text"}
