"""Unit tests for ingestion contract helpers."""

from src.ingest import _make_filing_id


def test_make_filing_id_is_deterministic() -> None:
    a = _make_filing_id("AMD", "10-K", "2024-12-31", "key-a")
    b = _make_filing_id("AMD", "10-K", "2024-12-31", "key-a")
    assert a == b


def test_make_filing_id_changes_with_key() -> None:
    a = _make_filing_id("AMD", "10-K", "2024-12-31", "key-a")
    b = _make_filing_id("AMD", "10-K", "2024-12-31", "key-b")
    assert a != b
