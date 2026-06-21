"""Additive extension modules for advanced tutorial techniques.

These modules are intentionally isolated from the core `src/` pipeline so
existing working code remains untouched.
"""

from src.extensions.hybrid_sparse_dense import HybridSparseDenseRetriever
from src.extensions.domain_adapter import (
    AdapterEvalRow,
    AdapterRunSummary,
    build_eval_rows_from_retriever,
    build_sft_datasets_from_chunks,
    evaluate_base_vs_adapter,
    is_gpu_available,
    seed_domain_adapter_placeholders,
    train_domain_adapter_from_chunks,
)
from src.extensions.multimodal import (
    MultimodalUnit,
    build_multimodal_units_from_html_map,
    filing_html_to_multimodal_units,
)
from src.extensions.multimodal_ocr import (
    build_glm_ocr_command,
    build_ocr_units_for_filing,
    build_ocr_units_from_image_map,
    run_glm_ocr,
)
from src.extensions.multimodal_retriever import MultimodalRetriever
from src.extensions.multimodal_v2 import build_multimodal_units_v2
from src.extensions.multimodal_vision import (
    build_vision_units_for_filing,
    build_vision_units_from_image_map,
    run_qwen_vision,
)
from src.extensions.rag_metrics import (
    RAGQualityScore,
    evaluate_rag_quality,
    judge_rag_quality,
)
from src.extensions.sparse import SparseBM25Retriever

__all__ = [
    "HybridSparseDenseRetriever",
    "MultimodalRetriever",
    "MultimodalUnit",
    "SparseBM25Retriever",
    "RAGQualityScore",
    "evaluate_rag_quality",
    "judge_rag_quality",
    "build_multimodal_units_from_html_map",
    "filing_html_to_multimodal_units",
    "build_glm_ocr_command",
    "run_glm_ocr",
    "build_ocr_units_for_filing",
    "build_ocr_units_from_image_map",
    "run_qwen_vision",
    "build_vision_units_for_filing",
    "build_vision_units_from_image_map",
    "build_multimodal_units_v2",
    "AdapterEvalRow",
    "AdapterRunSummary",
    "is_gpu_available",
    "build_sft_datasets_from_chunks",
    "train_domain_adapter_from_chunks",
    "build_eval_rows_from_retriever",
    "evaluate_base_vs_adapter",
    "seed_domain_adapter_placeholders",
]
