"""Unit tests for retrieval metric helpers."""

from src.evaluator import (
    f1_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_recall_f1_at_k_basic() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d"}

    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    assert recall_at_k(retrieved, relevant, k=2) == 0.5
    assert f1_at_k(retrieved, relevant, k=2) == 0.5


def test_mrr_basic() -> None:
    retrieved = ["x", "y", "z"]
    relevant = {"y"}
    assert mean_reciprocal_rank(retrieved, relevant) == 0.5


def test_ndcg_bounds() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "e"}
    score = ndcg_at_k(retrieved, relevant, k=5)
    assert 0.0 <= score <= 1.0
