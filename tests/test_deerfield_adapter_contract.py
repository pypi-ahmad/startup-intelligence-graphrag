"""Contract tests for additive deerfield ingestion adapter."""

from src.ingest_deerfield import _split_sections_from_text


def test_split_sections_from_item_text() -> None:
    text = (
        "ITEM 1. Business We design products for industrial customers. "
        "ITEM 1A. Risk Factors Supply chain disruptions and inflation can affect margins. "
        "ITEM 7. Management's Discussion We are investing in expansion."
    )
    sections = _split_sections_from_text(text)
    assert "Business" in sections
    assert "Risk Factors" in sections
    assert any("industrial customers" in s.lower() for s in sections["Business"])

