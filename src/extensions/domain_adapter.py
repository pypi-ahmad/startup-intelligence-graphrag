"""Optional domain-adaptation stage using Unsloth + PEFT + TRL.

This module is intentionally isolated from the default GraphRAG pipeline.
It introduces an opt-in advanced stage for generator domain adaptation only.
"""

from __future__ import annotations

import inspect
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from src.chunking import Chunk
from src.config import ARTIFACTS_DIR, SETTINGS
from src.evaluator import evaluate_generation
from src.extensions.rag_metrics import evaluate_rag_quality
from src.generation import USER_PROMPT, build_context_block
from src.judge import judge_answer


@dataclass
class AdapterEvalRow:
    """Single evaluation row for base-vs-adapter comparison."""

    query_id: str
    query: str
    prompt: str
    reference_answer: str
    contexts: list[str]
    citations: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdapterRunSummary:
    """Outcome for training/evaluation entrypoints."""

    status: str
    message: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def is_gpu_available() -> tuple[bool, str]:
    """Return GPU availability and diagnostic message."""
    try:
        import torch
    except Exception as exc:
        return False, f"torch unavailable: {exc}"

    ok = bool(torch.cuda.is_available())
    if ok:
        name = torch.cuda.get_device_name(0)
        return True, f"cuda available: {name}"
    return False, "cuda unavailable"


def build_sft_rows_from_chunks(
    chunks: list[Chunk],
    max_examples: int,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Build prompt-completion SFT rows from real SEC chunk text.

    This creates self-supervised continuation tasks from real filing passages.
    """
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []

    shuffled = list(chunks)
    rng.shuffle(shuffled)

    for chunk in shuffled:
        text = " ".join(chunk.text.split())
        if len(text) < 220:
            continue

        split_at = int(len(text) * 0.58)
        prefix = text[:split_at].strip()
        suffix = text[split_at:].strip()
        if len(prefix) < 80 or len(suffix) < 80:
            continue

        prompt = (
            "You are learning SEC filing style for company intelligence.\n"
            "Continue the following filing passage faithfully and in-domain.\n\n"
            f"Filing metadata: ticker={chunk.ticker}, section={chunk.section}\n\n"
            f"Passage prefix:\n{prefix}\n\n"
            "Continuation:\n"
        )
        completion = suffix
        rows.append(
            {
                "text": prompt + completion,
                "prompt": prompt,
                "completion": completion,
            }
        )
        if len(rows) >= max_examples:
            break

    return rows


def build_sft_datasets_from_chunks(
    chunks: list[Chunk],
    train_split: float | None = None,
    max_train_examples: int | None = None,
    max_eval_examples: int | None = None,
    seed: int | None = None,
) -> tuple[Any, Any, dict[str, int]]:
    """Create train/eval HF datasets for SFT from real SEC chunks."""
    train_split = float(SETTINGS.adapter_train_split if train_split is None else train_split)
    max_train_examples = int(
        SETTINGS.adapter_max_train_examples if max_train_examples is None else max_train_examples
    )
    max_eval_examples = int(
        SETTINGS.adapter_max_eval_examples if max_eval_examples is None else max_eval_examples
    )
    seed = SETTINGS.random_seed if seed is None else int(seed)

    try:
        from datasets import Dataset
    except Exception as exc:
        raise RuntimeError("datasets library is required for adapter dataset building.") from exc

    target_rows = max(50, max_train_examples + max_eval_examples)
    rows = build_sft_rows_from_chunks(chunks=chunks, max_examples=target_rows, seed=seed)
    if len(rows) < 20:
        raise RuntimeError(
            "Insufficient SFT rows from SEC chunks. Increase corpus size or lower minimum chunk constraints."
        )

    n_train = max(1, int(len(rows) * train_split))
    train_rows = rows[:n_train][:max_train_examples]
    eval_rows = rows[n_train : n_train + max_eval_examples]
    if not eval_rows:
        eval_rows = rows[max(0, len(train_rows) - min(32, len(train_rows))) : len(train_rows)]

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)
    stats = {
        "n_total_rows": len(rows),
        "n_train_rows": len(train_rows),
        "n_eval_rows": len(eval_rows),
    }
    return train_ds, eval_ds, stats


def _filtered_kwargs(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs by constructor signature for version resilience."""
    sig = inspect.signature(cls)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def train_domain_adapter_from_chunks(
    chunks: list[Chunk],
    output_dir: Path,
    base_model: str | None = None,
) -> AdapterRunSummary:
    """Train LoRA adapter with Unsloth + TRL over real SEC chunk continuations.

    If CUDA is unavailable, the run is skipped with a clear summary.
    """
    gpu_ok, gpu_msg = is_gpu_available()
    if not gpu_ok:
        return AdapterRunSummary(
            status="skipped",
            message=f"Domain adapter training skipped: {gpu_msg}",
            payload={"gpu": gpu_msg},
        )

    try:
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        return AdapterRunSummary(
            status="skipped",
            message="Unsloth/TRL stack unavailable. Install optional adapter dependencies first.",
            payload={"import_error": str(exc)},
        )

    base_model = base_model or SETTINGS.adapter_base_model
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_ds, eval_ds, ds_stats = build_sft_datasets_from_chunks(chunks=chunks)
    except Exception as exc:
        return AdapterRunSummary(
            status="failed",
            message=f"Failed to construct SFT dataset: {exc}",
            payload={},
        )

    logger.info("Loading Unsloth model {} for adapter training", base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=SETTINGS.adapter_max_seq_length,
        load_in_4bit=_to_bool(SETTINGS.adapter_use_4bit),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=SETTINGS.adapter_lora_r,
        lora_alpha=SETTINGS.adapter_lora_alpha,
        lora_dropout=SETTINGS.adapter_lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SETTINGS.random_seed,
        max_seq_length=SETTINGS.adapter_max_seq_length,
    )

    sft_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": SETTINGS.adapter_learning_rate,
        "per_device_train_batch_size": SETTINGS.adapter_per_device_batch_size,
        "gradient_accumulation_steps": SETTINGS.adapter_gradient_accumulation_steps,
        "max_steps": SETTINGS.adapter_max_steps,
        "eval_strategy": "steps",
        "eval_steps": SETTINGS.adapter_eval_steps,
        "save_strategy": "steps",
        "save_steps": SETTINGS.adapter_save_steps,
        "logging_steps": SETTINGS.adapter_log_steps,
        "max_seq_length": SETTINGS.adapter_max_seq_length,
        "optim": "adamw_8bit",
        "seed": SETTINGS.random_seed,
        "report_to": "none",
    }

    config = SFTConfig(**_filtered_kwargs(SFTConfig, sft_kwargs))

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": config,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
    }
    trainer_sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    train_output = trainer.train()
    trainer.save_model(str(adapter_dir))

    summary_payload = {
        "status": "trained",
        "base_model": base_model,
        "adapter_dir": str(adapter_dir),
        "dataset_stats": ds_stats,
        "gpu": gpu_msg,
        "training_loss": float(getattr(train_output, "training_loss", 0.0)),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    return AdapterRunSummary(
        status="trained",
        message="Domain adapter training completed.",
        payload=summary_payload,
    )


def build_eval_rows_from_retriever(
    queries: list[dict[str, Any]],
    retriever: Any,
    k: int | None = None,
) -> list[AdapterEvalRow]:
    """Construct generation prompts from retrieval outputs for model comparison."""
    k = SETTINGS.default_top_k if k is None else int(k)

    rows: list[AdapterEvalRow] = []
    for row in queries:
        query = str(row.get("query", ""))
        query_id = str(row.get("query_id", "unknown"))
        reference = str(row.get("reference_answer", ""))
        chunks = retriever.retrieve(query, k=k)
        context, citations = build_context_block(chunks)
        prompt = USER_PROMPT.format(query=query, context=context)
        rows.append(
            AdapterEvalRow(
                query_id=query_id,
                query=query,
                prompt=prompt,
                reference_answer=reference,
                contexts=[chunk.text for chunk in chunks],
                citations=citations,
            )
        )
    return rows


def _load_transformers_base_model(model_name: str) -> tuple[Any, Any]:
    """Load base model/tokenizer for inference benchmark."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers/torch stack unavailable. Install adapter optional dependencies."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return model, tokenizer


def _generate_text(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    """Generate output and return latency in seconds."""
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for adapter inference benchmarking.") from exc

    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    input_len = int(encoded["input_ids"].shape[-1])

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    latency = time.perf_counter() - start
    generated = out[0][input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, latency


def evaluate_base_vs_adapter(
    eval_rows: list[AdapterEvalRow],
    base_model: str,
    adapter_path: str | Path,
    judge_model: str | None = None,
    max_new_tokens: int | None = None,
    merge_adapter: bool = False,
) -> dict[str, Any]:
    """Benchmark base model vs PEFT adapter on the same retrieval-grounded prompts."""
    if not eval_rows:
        raise ValueError("eval_rows must not be empty")

    max_new_tokens = SETTINGS.adapter_max_new_tokens if max_new_tokens is None else int(max_new_tokens)
    judge_model = judge_model or SETTINGS.extension_judge_model

    try:
        from peft import PeftModel
    except Exception as exc:
        raise RuntimeError("peft is required for adapter evaluation.") from exc

    base_model_obj, tokenizer = _load_transformers_base_model(base_model)
    adapter_base_obj, adapter_tokenizer = _load_transformers_base_model(base_model)
    adapter_model_obj = PeftModel.from_pretrained(adapter_base_obj, str(adapter_path))
    if merge_adapter:
        adapter_model_obj = adapter_model_obj.merge_and_unload()

    base_preds: list[str] = []
    adapter_preds: list[str] = []
    refs: list[str] = []
    query_texts: list[str] = []
    contexts: list[list[str]] = []
    base_latencies: list[float] = []
    adapter_latencies: list[float] = []

    base_judge_rows: list[dict[str, Any]] = []
    adapter_judge_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for row in eval_rows:
        refs.append(row.reference_answer)
        query_texts.append(row.query)
        contexts.append(row.contexts)

        base_answer, base_latency = _generate_text(
            model=base_model_obj,
            tokenizer=tokenizer,
            prompt=row.prompt,
            max_new_tokens=max_new_tokens,
        )
        adapter_answer, adapter_latency = _generate_text(
            model=adapter_model_obj,
            tokenizer=adapter_tokenizer,
            prompt=row.prompt,
            max_new_tokens=max_new_tokens,
        )

        base_preds.append(base_answer)
        adapter_preds.append(adapter_answer)
        base_latencies.append(base_latency)
        adapter_latencies.append(adapter_latency)

        base_judge_rows.append(
            judge_answer(
                query=row.query,
                answer=base_answer,
                contexts=row.citations,
                reference=row.reference_answer,
                model=judge_model,
            ).to_dict()
        )
        adapter_judge_rows.append(
            judge_answer(
                query=row.query,
                answer=adapter_answer,
                contexts=row.citations,
                reference=row.reference_answer,
                model=judge_model,
            ).to_dict()
        )
        examples.append(
            {
                "query_id": row.query_id,
                "query": row.query,
                "reference": row.reference_answer,
                "base_answer": base_answer,
                "adapter_answer": adapter_answer,
                "base_latency_seconds": round(base_latency, 4),
                "adapter_latency_seconds": round(adapter_latency, 4),
            }
        )

    base_gen = evaluate_generation(base_preds, refs).to_dict()
    adapter_gen = evaluate_generation(adapter_preds, refs).to_dict()
    base_rag = evaluate_rag_quality(
        predictions=base_preds,
        contexts=contexts,
        queries=query_texts,
        references=refs,
        model=judge_model,
    )
    adapter_rag = evaluate_rag_quality(
        predictions=adapter_preds,
        contexts=contexts,
        queries=query_texts,
        references=refs,
        model=judge_model,
    )

    def _mean(values: list[float]) -> float:
        return float(sum(values) / max(len(values), 1))

    latency = {
        "base_mean_seconds": round(_mean(base_latencies), 4),
        "adapter_mean_seconds": round(_mean(adapter_latencies), 4),
        "delta_seconds": round(_mean(adapter_latencies) - _mean(base_latencies), 4),
    }

    keys = ["em", "bleu", "rouge_l", "meteor", "bert_score_f1"]
    gen_delta = {f"delta_{k}": round(float(adapter_gen[k]) - float(base_gen[k]), 4) for k in keys}

    return {
        "status": "completed",
        "base_model": base_model,
        "adapter_path": str(adapter_path),
        "judge_model": judge_model,
        "n_queries": len(eval_rows),
        "generation": {
            "base": base_gen,
            "adapter": adapter_gen,
            "delta": gen_delta,
        },
        "rag": {
            "base": base_rag,
            "adapter": adapter_rag,
            "delta": {
                "faithfulness": round(adapter_rag["faithfulness"] - base_rag["faithfulness"], 4),
                "context_precision": round(adapter_rag["context_precision"] - base_rag["context_precision"], 4),
                "context_recall": round(adapter_rag["context_recall"] - base_rag["context_recall"], 4),
                "answer_relevancy": round(adapter_rag["answer_relevancy"] - base_rag["answer_relevancy"], 4),
            },
        },
        "judge": {
            "base_rows": base_judge_rows,
            "adapter_rows": adapter_judge_rows,
        },
        "latency": latency,
        "examples": examples,
    }


def seed_domain_adapter_placeholders(output_root: Path | None = None) -> dict[str, str]:
    """Seed placeholder artifacts for optional domain-adapter stage."""
    root = output_root or ARTIFACTS_DIR
    eval_dir = root / "eval"
    generations_dir = root / "generations"
    figures_dir = root / "figures"

    payloads: dict[Path, dict[str, Any]] = {
        eval_dir / "domain_adapter_training_placeholder.json": {
            "status": "placeholder",
            "note": "Populate after explicit adapter training run.",
            "metrics": {
                "train_loss": "TBD",
                "eval_loss": "TBD",
                "steps": "TBD",
            },
        },
        eval_dir / "domain_adapter_generation_comparison_placeholder.json": {
            "status": "placeholder",
            "note": "Populate after explicit base-vs-adapter benchmark run.",
            "metrics": {
                "base": {"em": "TBD", "bleu": "TBD", "rouge_l": "TBD", "meteor": "TBD", "bert_score_f1": "TBD"},
                "adapter": {"em": "TBD", "bleu": "TBD", "rouge_l": "TBD", "meteor": "TBD", "bert_score_f1": "TBD"},
                "delta": {"em": "TBD", "bleu": "TBD", "rouge_l": "TBD", "meteor": "TBD", "bert_score_f1": "TBD"},
            },
        },
        eval_dir / "domain_adapter_rag_comparison_placeholder.json": {
            "status": "placeholder",
            "note": "Populate after explicit base-vs-adapter benchmark run.",
            "metrics": {
                "base": {
                    "faithfulness": "TBD",
                    "context_precision": "TBD",
                    "context_recall": "TBD",
                    "answer_relevancy": "TBD",
                },
                "adapter": {
                    "faithfulness": "TBD",
                    "context_precision": "TBD",
                    "context_recall": "TBD",
                    "answer_relevancy": "TBD",
                },
            },
        },
        eval_dir / "domain_adapter_judge_comparison_placeholder.json": {
            "status": "placeholder",
            "judge_model": SETTINGS.extension_judge_model,
            "note": "Populate after explicit base-vs-adapter benchmark run.",
            "metrics": {
                "correctness": "TBD",
                "relevance": "TBD",
                "completeness": "TBD",
                "groundedness": "TBD",
                "hallucination_risk": "TBD",
            },
        },
        eval_dir / "domain_adapter_latency_placeholder.json": {
            "status": "placeholder",
            "note": "Populate after explicit base-vs-adapter benchmark run.",
            "metrics": {
                "base_mean_seconds": "TBD",
                "adapter_mean_seconds": "TBD",
                "delta_seconds": "TBD",
            },
        },
        generations_dir / "domain_adapter_samples_placeholder.json": {
            "status": "placeholder",
            "samples": [],
            "note": "Populate with base-vs-adapter output examples after execution.",
        },
    }

    written: dict[str, str] = {}
    for path, payload in payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written[path.name] = str(path)

    figures_dir.mkdir(parents=True, exist_ok=True)
    adapter_fig_manifest = figures_dir / "domain_adapter_placeholder_manifest.md"
    adapter_fig_manifest.write_text(
        "# Domain Adapter Placeholder Figure Manifest\n\n"
        "Planned figures for optional Unsloth+PEFT+TRL stage:\n"
        "1. domain_adapter_architecture.png\n"
        "2. domain_adapter_training_curve.png\n"
        "3. domain_adapter_generation_delta.png\n"
        "4. domain_adapter_latency_comparison.png\n",
        encoding="utf-8",
    )
    written[adapter_fig_manifest.name] = str(adapter_fig_manifest)
    return written
