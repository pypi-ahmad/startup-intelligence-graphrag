"""Contract tests for dynamic eval query builder."""

from src.eval_query_builder import build_eval_queries_from_filings
from src.ingest import Filing


def _mock_filing(ticker: str) -> Filing:
    return Filing(
        company_name=ticker,
        cik="1",
        ticker=ticker,
        exchange="",
        state_of_incorporation="",
        sic="",
        form="10-K",
        filing_date="2025-12-31",
        report_date="2025-12-31",
        filing_id=f"{ticker}_10-K_2025_0001",
        source_split="train",
        source_row_count=1,
        sections={
            "Business": [
                "The company builds software products for enterprise customers and cloud markets."
            ]
            * 40,
            "Risk Factors": [
                "Supply chain and regulatory changes can impact costs and operating performance."
            ]
            * 40,
        },
    )


def test_build_eval_queries_has_required_fields() -> None:
    queries = build_eval_queries_from_filings([_mock_filing("AAA"), _mock_filing("BBB")], max_local_companies=2)
    assert len(queries) >= 6
    row = queries[0]
    assert "query_id" in row
    assert "query" in row
    assert "relevant_tickers" in row
    assert "reference_answer" in row

