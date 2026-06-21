"""Contract tests for optional Unsloth+PEFT+TRL domain adapter stage."""

from pathlib import Path

from src.chunking import Chunk
from src.extensions.domain_adapter import (
    build_sft_rows_from_chunks,
    is_gpu_available,
    seed_domain_adapter_placeholders,
)


def _make_chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="c1",
        filing_id="f1",
        ticker="AMD",
        company_name="Advanced Micro Devices",
        section="Business",
        text=text,
        sentence_ids=[0, 1],
        token_count=100,
    )


def test_build_sft_rows_from_chunks_uses_real_chunk_text() -> None:
    text = (
        "Advanced Micro Devices reported expanded data center demand and highlighted "
        "AI accelerator momentum across enterprise customers. "
        "The company also discussed pricing discipline, supply constraints, and "
        "longer-term investment plans in server roadmap execution."
    )
    rows = build_sft_rows_from_chunks(chunks=[_make_chunk(text)], max_examples=10, seed=42)
    assert rows
    assert "Passage prefix" in rows[0]["prompt"]
    assert len(rows[0]["completion"]) > 50


def test_is_gpu_available_contract_shape() -> None:
    ok, message = is_gpu_available()
    assert isinstance(ok, bool)
    assert isinstance(message, str)


def test_seed_domain_adapter_placeholders_creates_expected_files(tmp_path: Path) -> None:
    written = seed_domain_adapter_placeholders(output_root=tmp_path)
    assert "domain_adapter_training_placeholder.json" in written
    assert (tmp_path / "eval" / "domain_adapter_training_placeholder.json").exists()
