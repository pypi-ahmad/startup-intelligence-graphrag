"""Contract tests for sparse and sparse+dense retriever additions."""

from src.extensions.hybrid_sparse_dense import HybridSparseDenseRetriever


def test_hybrid_sparse_dense_retriever_name_constant() -> None:
    assert HybridSparseDenseRetriever.name == "hybrid_sparse_dense"
