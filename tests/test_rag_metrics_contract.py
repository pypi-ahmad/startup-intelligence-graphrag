"""Contract tests for RAG-level metric extensions."""

from src.extensions.rag_metrics import (
    DEFAULT_EXTENSION_JUDGE_MODEL,
    DEFAULT_GUARDIAN_MODEL,
    RAGQualityScore,
)


def test_extension_judge_default_model_is_pinned() -> None:
    assert DEFAULT_EXTENSION_JUDGE_MODEL == "granite4.1:8b"
    assert DEFAULT_GUARDIAN_MODEL == "granite4.1:8b"


def test_rag_quality_score_to_dict_fields() -> None:
    score = RAGQualityScore(
        query="q",
        answer="a",
        faithfulness=0.7,
        context_precision=0.6,
        context_recall=0.5,
        answer_relevancy=0.8,
        rationale="ok",
        model=DEFAULT_EXTENSION_JUDGE_MODEL,
    )
    payload = score.to_dict()
    assert "faithfulness" in payload and "context_recall" in payload
