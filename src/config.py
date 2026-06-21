"""Central configuration for Startup Intelligence GraphRAG.

This module defines a single, reproducible settings contract used by notebooks,
CLI scripts, and library code. Environment variables are prefixed with
`SIRAG_` (for example `SIRAG_DEFAULT_N_COMPANIES=40`).
"""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
DATA_DIR: Path = PROJECT_ROOT / "data"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"

RAW_DIR: Path = ARTIFACTS_DIR / "raw"
CHUNKS_DIR: Path = ARTIFACTS_DIR / "chunks"
EMBEDDINGS_DIR: Path = ARTIFACTS_DIR / "embeddings"
GRAPH_DIR: Path = ARTIFACTS_DIR / "graph"
RETRIEVALS_DIR: Path = ARTIFACTS_DIR / "retrievals"
GENERATIONS_DIR: Path = ARTIFACTS_DIR / "generations"
EVAL_DIR: Path = ARTIFACTS_DIR / "eval"
FIGURES_DIR: Path = ARTIFACTS_DIR / "figures"
ARCHIVE_DIR: Path = ARTIFACTS_DIR / "archive"

EVAL_QUERIES_PATH: Path = DATA_DIR / "eval" / "eval_queries.jsonl"


class Settings(BaseSettings):
    """Runtime settings for the full project.

    All defaults are production-safe for local execution on Ubuntu with Ollama.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SIRAG_",
        extra="ignore",
    )

    # Dataset policy: strict, no fallback.
    dataset_repo: str = "deerfieldgreen/stk-sec-filings"
    dataset_config: str | None = None
    dataset_splits: tuple[str, ...] = ("train",)
    dataset_trust_remote_code: bool = True
    strict_dataset: bool = True

    # Corpus shaping.
    default_n_companies: int = 30
    company_selection_seed: int = 42
    min_sentences_per_filing: int = 80
    max_sentences_per_filing: int = 5000

    # Models (all local via Ollama).
    ollama_host: str = "http://localhost:11434"
    ollama_timeout_seconds: int = 180
    embed_model: str = "qwen3-embedding:4b"
    generator_model: str = "granite4.1:8b"
    judge_model: str = "granite4.1:8b"
    extension_judge_model: str = "granite4.1:8b"
    ocr_model: str = "glm-ocr:latest"
    vision_model: str = "qwen3.5:4b"
    adapter_base_model: str = "unsloth/granite-4.1-3b"
    adapter_enable: bool = False
    adapter_use_4bit: bool = True
    adapter_max_seq_length: int = 2048
    adapter_max_train_examples: int = 4000
    adapter_max_eval_examples: int = 400
    adapter_train_split: float = 0.9
    adapter_lora_r: int = 16
    adapter_lora_alpha: int = 32
    adapter_lora_dropout: float = 0.05
    adapter_learning_rate: float = 2e-4
    adapter_per_device_batch_size: int = 1
    adapter_gradient_accumulation_steps: int = 8
    adapter_max_steps: int = 300
    adapter_eval_steps: int = 50
    adapter_save_steps: int = 50
    adapter_log_steps: int = 10
    adapter_max_new_tokens: int = 256

    # Chunking.
    chunk_size_tokens: int = 300
    chunk_overlap_tokens: int = 40

    # Graph/extraction.
    entity_extraction_max_chars: int = 4500
    entity_extraction_max_prompts: int = 120
    louvain_resolution: float = 1.0

    # Retrieval/eval.
    default_top_k: int = 6
    retrieval_k_values: tuple[int, ...] = (3, 5, 10)
    embed_batch_size: int = 64
    generation_temperature: float = 0.2
    judge_temperature: float = 0.0

    # Reliability / deterministic behavior.
    use_llm_cache: bool = True
    random_seed: int = 42

    # Artifact policy for this implementation stage.
    placeholder_mode: bool = True


SETTINGS = Settings()


SEC_SECTION_LABELS: dict[int, str] = {
    0: "Business",
    1: "Risk Factors",
    2: "Unresolved Staff Comments",
    3: "Properties",
    4: "Legal Proceedings",
    5: "Mine Safety Disclosures",
    6: "Market for Registrant's Common Equity",
    7: "Selected Financial Data",
    8: "Management's Discussion and Analysis (MD&A)",
    9: "Quantitative and Qualitative Disclosures About Market Risk",
    10: "Financial Statements and Supplementary Data",
    11: "Changes in and Disagreements with Accountants",
    12: "Controls and Procedures",
    13: "Other Information",
    14: "Directors, Officers and Compensation",
    15: "Security Ownership",
    16: "Related Transactions",
    17: "Accountant Fees",
    18: "Exhibits and Financial Statement Schedules",
    19: "Form 10-K Summary",
}

USEFUL_SECTIONS: tuple[int, ...] = (
    0,
    1,
    3,
    4,
    6,
    7,
    8,
    9,
    12,
    14,
)


def ensure_dirs() -> None:
    """Create required project directories if missing."""
    for directory in [
        RAW_DIR,
        CHUNKS_DIR,
        EMBEDDINGS_DIR,
        GRAPH_DIR,
        RETRIEVALS_DIR,
        GENERATIONS_DIR,
        EVAL_DIR,
        FIGURES_DIR,
        ARCHIVE_DIR,
        DATA_DIR / "raw",
        DATA_DIR / "eval",
        NOTEBOOKS_DIR,
        DOCS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def as_dict() -> dict[str, object]:
    """Return serializable settings and key paths for manifests."""
    return {
        "dataset_repo": SETTINGS.dataset_repo,
        "dataset_config": SETTINGS.dataset_config,
        "dataset_splits": list(SETTINGS.dataset_splits),
        "default_n_companies": SETTINGS.default_n_companies,
        "embed_model": SETTINGS.embed_model,
        "generator_model": SETTINGS.generator_model,
        "judge_model": SETTINGS.judge_model,
        "ollama_timeout_seconds": SETTINGS.ollama_timeout_seconds,
        "extension_judge_model": SETTINGS.extension_judge_model,
        "ocr_model": SETTINGS.ocr_model,
        "vision_model": SETTINGS.vision_model,
        "adapter_base_model": SETTINGS.adapter_base_model,
        "adapter_enable": SETTINGS.adapter_enable,
        "adapter_use_4bit": SETTINGS.adapter_use_4bit,
        "adapter_max_seq_length": SETTINGS.adapter_max_seq_length,
        "adapter_max_train_examples": SETTINGS.adapter_max_train_examples,
        "adapter_max_eval_examples": SETTINGS.adapter_max_eval_examples,
        "adapter_lora_r": SETTINGS.adapter_lora_r,
        "adapter_lora_alpha": SETTINGS.adapter_lora_alpha,
        "adapter_lora_dropout": SETTINGS.adapter_lora_dropout,
        "adapter_learning_rate": SETTINGS.adapter_learning_rate,
        "chunk_size_tokens": SETTINGS.chunk_size_tokens,
        "chunk_overlap_tokens": SETTINGS.chunk_overlap_tokens,
        "default_top_k": SETTINGS.default_top_k,
        "placeholder_mode": SETTINGS.placeholder_mode,
        "project_root": str(PROJECT_ROOT),
    }
