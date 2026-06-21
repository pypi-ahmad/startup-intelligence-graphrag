# Startup / Company Intelligence GraphRAG (SEC Filings + Ollama)

A local-first, tutorial-grade GraphRAG system for company intelligence on real SEC filings.

This project builds a complete intelligence pipeline over filing text and visuals:
- strict SEC ingestion from Hugging Face (`deerfieldgreen/stk-sec-filings`)
- dense, graph-aware, hybrid, agentic, and multimodal retrieval
- grounded answer generation with local Ollama models
- automated + judge-based evaluation with saved artifacts

## Current Execution Status (Artifact-Grounded)

The repository contains **real run outputs** from `2026-06-21`:
- Core run summary exists at `artifacts/run_summary.json` with real metrics/figures.
- Strict dataset snapshot exists at `artifacts/raw/manifest.json`.
- Per-technique evaluation bundles exist in `artifacts/eval/*_full_metrics.json`.

The strict single-command host orchestrator status is currently:
- `artifacts/run_completion_manifest.json` → `"status": "failed"`
- recorded failure: `Stage 'core_pipeline' failed with exit code 1`

So the codebase and core artifacts are real and complete for analysis, while the last strict host-completion record is not in a `completed` state.

## Technique Coverage

Implemented in code and represented in run artifacts:
- Vector RAG baseline
- GraphRAG local retrieval
- GraphRAG global retrieval
- GraphRAG hybrid retrieval
- Hybrid sparse+dense retrieval (BM25 + dense fusion)
- Agentic RAG with corrective retrieval grading (CRAG-style)
- Multimodal RAG from filing HTML/table/figure text
- Multimodal OCR RAG (`glm-ocr` via Ollama)
- Multimodal Vision RAG (`qwen3.5:4b` via Ollama)
- Unified multimodal v2 (HTML + OCR + vision channels)
- Optional domain adapter stage (Unsloth + PEFT + TRL)

## Model Stack

- Embeddings: `qwen3-embedding:4b`
- Generator: `granite4.1:8b`
- Judge: `granite4.1:8b`
- OCR: `glm-ocr:latest`
- Vision: `qwen3.5:4b`

## Environment

- Ubuntu local execution target
- Python `3.12.10`
- `uv` package/workflow manager
- local virtual environment in project folder

## Quickstart

```bash
cd /home/ahmad/AI/startup-intelligence-graphrag

uv python install 3.12.10
uv venv --python 3.12.10 .venv
source .venv/bin/activate
uv sync

ollama pull qwen3-embedding:4b
ollama pull granite4.1:8b
ollama pull glm-ocr:latest
ollama pull qwen3.5:4b
```

Optional adapter extras:

```bash
uv sync --extra adapter
```

## Run Commands

Core real pipeline:

```bash
source .venv/bin/activate
python scripts/run_full_real_project.py --n-companies 30 --force-download
```

Strict host orchestration (core + adapter + notebooks + tests + manifest):

```bash
source .venv/bin/activate
python scripts/run_complete_host.py --profile comprehensive --force-download
```

Optional adapter stage only:

```bash
source .venv/bin/activate
export SIRAG_ADAPTER_ENABLE=true
python scripts/run_domain_adapter.py --mode execute --n-companies 30 --force
```

Notebook execution:

```bash
source .venv/bin/activate
python scripts/execute_notebooks.py --timeout 3600
```

## Real Run Snapshot (`artifacts/run_summary.json`)

- `dataset_repo`: `deerfieldgreen/stk-sec-filings`
- `n_filings`: `6`
- `n_chunks`: `1193`
- `n_eval_queries`: `1`
- `n_graph_nodes`: `178`
- `n_graph_edges`: `602`
- `n_ocr_units`: `2`
- `n_vision_units`: `2`
- `duration_seconds`: `1094.24`

### Retrieval Metrics

| Technique | Precision@3 | Recall@3 | F1@3 | MRR | NDCG@3 |
|---|---:|---:|---:|---:|---:|
| vector_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| graphrag_local | 0.3333 | 0.0256 | 0.0476 | 1.0000 | 0.4693 |
| graphrag_global | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| graphrag_hybrid | 0.3333 | 0.0256 | 0.0476 | 0.3333 | 0.2346 |
| hybrid_sparse_dense | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| agentic_crag | 0.3333 | 0.0256 | 0.0476 | 0.5000 | 0.2961 |
| multimodal_rag | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| multimodal_ocr_rag | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| multimodal_vision_rag | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| multimodal_unified_v2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Generation Metrics

| Technique | EM | BLEU | ROUGE-L | METEOR | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| vector_baseline | - | - | - | - | - |
| graphrag_local | - | - | - | - | - |
| graphrag_global | - | - | - | - | - |
| graphrag_hybrid | 0.0000 | 0.0136 | 0.1053 | 0.1303 | -0.0076 |
| hybrid_sparse_dense | 0.0000 | 0.0118 | 0.1000 | 0.0880 | 0.0446 |
| agentic_crag | 0.0000 | 0.0087 | 0.1207 | 0.1642 | 0.0312 |
| multimodal_rag | 0.0000 | 0.0054 | 0.0826 | 0.0763 | -0.0037 |
| multimodal_ocr_rag | 0.0000 | 0.0140 | 0.1176 | 0.1698 | -0.0092 |
| multimodal_vision_rag | 0.0000 | 0.0072 | 0.1053 | 0.0868 | -0.0078 |
| multimodal_unified_v2 | 0.0000 | 0.0127 | 0.0881 | 0.1033 | 0.0362 |

### RAG Quality Metrics

| Technique | Faithfulness | Context Precision | Context Recall | Answer Relevancy |
|---|---:|---:|---:|---:|
| vector_baseline | - | - | - | - |
| graphrag_local | - | - | - | - |
| graphrag_global | - | - | - | - |
| graphrag_hybrid | 0.9500 | 0.8500 | 0.9000 | 0.9200 |
| hybrid_sparse_dense | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| agentic_crag | 0.9500 | 1.0000 | 0.9000 | 1.0000 |
| multimodal_rag | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| multimodal_ocr_rag | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| multimodal_vision_rag | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| multimodal_unified_v2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Repository Map

- `src/`: production pipeline modules
- `src/extensions/`: additive technique modules (hybrid, multimodal, adapter, rag metrics)
- `scripts/`: execution entrypoints (`run_full_real_project.py`, `run_complete_host.py`, etc.)
- `notebooks/`: tutorial notebooks (source)
- `notebooks/executed/`: historical executed notebook artifacts
- `artifacts/`: run outputs, metrics, retrieval/generation samples, figures, manifests
- `docs/`: handbook/tutorial documentation

## Tutorial Entry Points

Primary notebooks:
- `notebooks/startup_intelligence_graphrag_zero_to_hero.ipynb`
- `notebooks/02_hybrid_sparse_dense_rag_startup_intelligence.ipynb`
- `notebooks/03_multimodal_rag_startup_intelligence.ipynb`
- `notebooks/04_multimodal_rag_ocr_vision_startup_intelligence.ipynb`
- `notebooks/05_optional_domain_adapter_unsloth_peft_trl.ipynb`

Companion handbook docs:
- `docs/documentation.md`
- `docs/tutorials/`

## Artifact Paths You Will Use Most

- Run state:
  - `artifacts/run_summary.json`
  - `artifacts/run_completion_manifest.json`
  - `artifacts/raw/manifest.json`
- Metrics:
  - `artifacts/eval/*_full_metrics.json`
  - `artifacts/eval/*_retrieval_metrics_placeholder.json` (compat payloads, status=`executed`)
- Samples:
  - `artifacts/retrievals/*_retrieval_samples_placeholder.json`
  - `artifacts/generations/*_generation_samples_placeholder.json`
- Figures:
  - `artifacts/figures/retrieval_ndcg_comparison.png`
  - `artifacts/figures/generation_rougel_comparison.png`
  - `artifacts/figures/judge_overall_comparison.png`
  - `artifacts/figures/graph_topology_snapshot.png`

## Optional Unsloth/PEFT/TRL Stage

This stack is intentionally scoped to optional generator adaptation only.
It is not forced into ingestion, retrieval, graph construction, CRAG routing, OCR, or vision layers.

Reference doc:
- `docs/unsloth_peft_trl_sources.md`

## README Writing Standards Used

This README structure follows current public guidance from:
- GitHub Docs: About repository READMEs and repository documentation best practices
- Open Source Guides: README purpose and onboarding questions
- Google Documentation Style Guide: README content and readability conventions

(Linked in `docs/documentation.md` references section.)
