# Zero to Hero Study Handbook: startup-intelligence-graphrag

This handbook is built from static analysis of the repository files only.
It is designed to help a new learner understand the system end-to-end, from code structure to runtime flow.

## Module 1: Foundations & Architecture

### 1) What this project does

`startup-intelligence-graphrag` is a local-first GraphRAG system over SEC filings.
It builds a corpus from a strict Hugging Face dataset, chunks filing text, embeds chunks, builds a graph from extracted entities/relationships, and evaluates multiple retrieval-generation techniques (dense, graph-aware, hybrid, agentic, and multimodal).

Primary use cases supported by the current code:

- Grounded Q&A over 10-K filings.
- Cross-company risk and strategy analysis.
- Retrieval technique benchmarking (`vector`, `graph_local`, `graph_global`, `hybrid`, `hybrid_sparse_dense`, `agentic_crag`, multimodal variants).
- Optional domain adaptation for generation (Unsloth + PEFT + TRL), isolated as an opt-in stage.

### 2) Core paradigms and patterns used in this repository

Definitions first, then where each appears:

- **Pipeline orchestration**: A fixed, ordered set of processing steps from raw data to artifacts.
  - Main implementations: [scripts/run_full_real_project.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/scripts/run_full_real_project.py), [src/pipeline.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/pipeline.py), [scripts/run_complete_host.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/scripts/run_complete_host.py).
- **Dataclass-centric domain modeling**: Structured records for core entities.
  - Examples: `Filing`, `Chunk`, `RetrievedChunk`, `GenerationResult`, `JudgeScore`, `CommunitySummary`.
- **RAG (Retrieval-Augmented Generation)**: Retrieve relevant context chunks, then generate grounded answers with citations.
  - Main flow: [src/generation.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/generation.py), [src/retrievers.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/retrievers.py).
- **GraphRAG**: Add graph structure (company, filing, section, entity nodes and relation/co-occurrence edges) to improve retrieval context.
  - Main flow: [src/graph.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/graph.py), graph-aware retrievers in [src/retrievers.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/retrievers.py).
- **Hybrid retrieval**: Fuse multiple retrieval channels.
  - RRF over dense + graph channels in `HybridRetriever`.
  - Dense + BM25 fusion in `HybridSparseDenseRetriever`.
- **Agentic corrective loop (CRAG-style)**: Classify query type, route retriever, grade retrieved chunks, retry with fallback retrievers.
  - Main class: `AgenticGraphRAG` in [src/agent.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/agent.py).
- **Multimodal extension pattern**: Add HTML table/figure text, OCR text, and vision text as retrievable units without changing base ingestion/retrieval modules.
  - Modules in [src/extensions](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/extensions).
- **Extension isolation**: Optional components are placed in `src/extensions/` and separate scripts (for example, adapter stage).
  - Keeps base runtime path stable.

### 3) Architecture and interactions

High-level component map:

- **Configuration layer**: all runtime settings from `SIRAG_` env vars via `Settings`.
- **Data layer**: strict SEC ingestion (`deerfieldgreen/stk-sec-filings`) and filing normalization.
- **Retrieval substrate**: chunking + embeddings + FAISS + graph.
- **Retrieval strategies**: dense, graph-local, graph-global, hybrid, sparse+dense, multimodal.
- **Generation and evaluation**: answer generation, judge scoring, retrieval/generation/RAG metrics.
- **Orchestration layer**: project runners that sequence stages and write manifests/artifacts.

Text-based ASCII architecture:

```text
                    +------------------------------+
                    | scripts/run_complete_host.py |
                    +--------------+---------------+
                                   |
             +---------------------+----------------------+
             |                                            |
     preflight checks                              stage runners
 (HF token, Ollama models, GPU)      (core, adapter, notebooks, tests)
             |
             v
   +-------------------------------+
   | scripts/run_full_real_project |
   +-------------------------------+
             |
             v
   +--------------------+     +-------------------+
   | src/ingest*.py     | --> | src/chunking.py   |
   +--------------------+     +---------+---------+
                                         |
                                         v
                               +--------------------+
                               | src/vectorstore.py |
                               +---------+----------+
                                         |
                +------------------------+-------------------------+
                |                                                  |
                v                                                  v
      +-------------------+                             +--------------------+
      | src/extractor.py  | --> src/graph.py --> graph | src/retrievers.py  |
      +-------------------+                             +--------------------+
                                                                  |
                                                                  v
                                                        +-------------------+
                                                        | src/generation.py |
                                                        +---------+---------+
                                                                  |
                                                                  v
                                                        +-------------------+
                                                        | evaluator/judge/  |
                                                        | rag_metrics        |
                                                        +---------+---------+
                                                                  |
                                                                  v
                                                        artifacts/* (metrics,
                                                        figures, samples,
                                                        run summaries)
```

## Module 2: Repository Map

Focus first on these files and directories:

| File/Directory Path | Primary Responsibility | Key Classes/Functions | Important Configs/Variables |
|---|---|---|---|
| `pyproject.toml` | Package metadata and dependency lock targets | `[project]`, `[project.optional-dependencies]` | Python `>=3.12.10,<3.13`; optional `adapter` extras |
| `.env.example` | Template runtime environment values | N/A | `SIRAG_*` keys (models, dataset, adapter, defaults) |
| [src/config.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/config.py) | Central settings and project paths | `Settings`, `ensure_dirs()`, `as_dict()` | `dataset_repo`, `strict_dataset`, model names, chunk/eval/adapter settings |
| [src/ingest.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/ingest.py) | Generic strict ingestion and normalization | `Filing`, `build_corpus()`, `group_into_filings()` | `STRICT_DATASET_REPO`, `USEFUL_SECTIONS`, sentence bounds |
| [src/ingest_deerfield.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/ingest_deerfield.py) | Deerfield dataset-specific adapter (current strict path) | `DeerfieldRecord`, `build_deerfield_corpus()` | `DEERFIELD_DATASET_ID`, `_ITEM_SECTION_MAP` |
| [src/chunking.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/chunking.py) | Token-aware chunk creation and persistence | `Chunk`, `chunk_corpus()`, `chunk_section()` | `chunk_size_tokens`, `chunk_overlap_tokens` |
| [src/embeddings.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/embeddings.py) | Ollama embedding calls with cache | `embed_text()`, `embed_texts()`, `embedding_dim()` | `SETTINGS.embed_model`, `use_llm_cache` |
| [src/vectorstore.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/vectorstore.py) | FAISS index wrapper and dense search | `FaissVectorStore`, `SearchResult`, `build_from_chunks()` | `EMBEDDINGS_DIR`, `SETTINGS.embed_model` |
| [src/extractor.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/extractor.py) | LLM entity/relationship extraction | `Entity`, `Relationship`, `FilingExtraction`, `extract_from_filings()` | `ENTITY_TYPES`, `RELATION_TYPES`, extraction prompt and limits |
| [src/graph.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/graph.py) | Graph construction, communities, summaries | `build_graph()`, `detect_communities()`, `CommunitySummary`, `save_graph()` | `louvain_resolution`, graph node-id conventions |
| [src/retrievers.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/retrievers.py) | Dense and graph-aware retrieval strategies | `RetrievedChunk`, `DenseVectorRetriever`, `GraphLocalRetriever`, `GraphGlobalRetriever`, `HybridRetriever` | fusion weights, `k_rrf` |
| [src/agent.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/agent.py) | Agentic CRAG-style orchestration | `AgenticGraphRAG`, `AgentResult`, `build_default_agent()` | `CLASSIFY_PROMPT`, `DOC_GRADE_PROMPT`, `max_iterations` |
| [src/generation.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/generation.py) | Citation-grounded answer generation | `GenerationResult`, `build_context_block()`, `generate_answer()` | `SYSTEM_PROMPT`, `USER_PROMPT`, `generation_temperature` |
| [src/evaluator.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/evaluator.py) | Retrieval and generation metric computation | `evaluate_retrieval()`, `evaluate_generation()`, metric helpers | K, relevance derivation fields in query rows |
| [src/judge.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/judge.py) | LLM-as-a-judge scoring | `JudgeScore`, `judge_answer()`, `judge_generation()` | `judge_model`, `judge_temperature` |
| [src/extensions/sparse.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/extensions/sparse.py) | BM25 sparse retrieval extension | `SparseBM25Retriever` | domain boost terms, BM25 scoring |
| [src/extensions/hybrid_sparse_dense.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/extensions/hybrid_sparse_dense.py) | Dense+BM25 fusion retriever | `HybridSparseDenseRetriever`, `HybridConfig` | `dense_weight`, `sparse_weight`, `rrf_k` |
| [src/extensions/multimodal.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/extensions/multimodal.py) | HTML table/figure-text unit extraction | `MultimodalUnit`, `build_multimodal_units_from_html_map()` | modality tags: `table`, `figure_text` |
| [src/extensions/multimodal_ocr.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/extensions/multimodal_ocr.py) | OCR unit creation via Ollama command | `build_glm_ocr_command()`, `run_glm_ocr()` | `SIRAG_OCR_MODEL` / `SETTINGS.ocr_model` |
| [src/extensions/multimodal_vision.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/extensions/multimodal_vision.py) | Vision unit creation via Ollama chat | `run_qwen_vision()`, `build_vision_units_from_image_map()` | `SIRAG_VISION_MODEL` / `SETTINGS.vision_model` |
| [src/extensions/multimodal_retriever.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/extensions/multimodal_retriever.py) | Retrieval over multimodal units | `MultimodalRetriever` | modality weights, dense/sparse fusion ratio |
| [src/extensions/rag_metrics.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/extensions/rag_metrics.py) | RAG-level quality metrics | `RAGQualityScore`, `evaluate_rag_quality()` | `DEFAULT_EXTENSION_JUDGE_MODEL` |
| [src/extensions/domain_adapter.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/extensions/domain_adapter.py) | Optional Unsloth+PEFT+TRL training and base-vs-adapter benchmark | `train_domain_adapter_from_chunks()`, `evaluate_base_vs_adapter()` | `adapter_*` settings, GPU availability |
| [src/multimodal_assets.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/multimodal_assets.py) | Build multimodal source assets (HTML and image maps) | `fetch_filing_html_map()`, `build_table_image_sources()`, `build_text_snapshot_images()` | `artifacts/multimodal/*` paths |
| [scripts/run_full_real_project.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/scripts/run_full_real_project.py) | Main real runtime path for full technique evaluation | `main()`, `_run_technique()`, `evaluate_agentic_end_to_end()` | CLI args: `--n-companies`, `--top-k`, chunking/multimodal caps |
| [scripts/run_complete_host.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/scripts/run_complete_host.py) | Single-command host orchestrator with strict preflight and manifest | `_assert_hf_access()`, `_assert_ollama_models()`, `_run_stage()` | `REQUIRED_OLLAMA_MODELS`, profile config, HF token env |
| [scripts/run_domain_adapter.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/scripts/run_domain_adapter.py) | Optional adapter-stage CLI | `run_placeholder_mode()`, `run_execute_mode()` | `--mode`, `--force`, `--required` |
| [scripts/run_pipeline.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/scripts/run_pipeline.py) | Simple placeholder or execute pipeline runner | `main()` | `--mode placeholder|execute` |
| [scripts/execute_notebooks.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/scripts/execute_notebooks.py) | Execute notebooks and write executed copies | `main()` | `--timeout`, `--pattern`, output dir |
| [tests](/home/ahmad/AI/Github/startup-intelligence-graphrag/tests) | Contract-oriented safety rails for key behavior | contract tests per module | validates script contracts, model pinning, settings exposure |

## Module 3: Core Execution Flows

### Flow A: Full real project execution (`scripts/run_full_real_project.py`)

Entrypoint: `main()`.

Step-by-step:

1. Parse CLI args (`--n-companies`, `--top-k`, chunk and multimodal limits).
2. Ensure artifact directories (`ensure_dirs()`).
3. Build corpus:
   - `build_corpus(...)` in [src/ingest.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/ingest.py).
   - Under strict defaults, this dispatches to `build_deerfield_corpus(...)` in [src/ingest_deerfield.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/ingest_deerfield.py).
4. Build evaluation queries:
   - `build_eval_queries_from_filings(...)` then `save_eval_queries(...)`.
5. Build retrieval substrate:
   - `chunk_corpus(...)` and `save_chunks(...)`.
   - `build_from_chunks(...)` to create FAISS + metadata.
6. Build graph substrate:
   - `extract_from_filings(...)` -> `build_graph(...)` -> `detect_communities(...)` -> fast summaries -> `save_graph(...)`.
7. Run retrieval/evaluation techniques:
   - Dense, graph local/global/hybrid, sparse+dense hybrid, agentic CRAG.
8. Run multimodal techniques:
   - `fetch_filing_html_map(...)`
   - `build_multimodal_units_from_html_map(...)`
   - image map from `build_table_image_sources(...)` or fallback `build_text_snapshot_images(...)`
   - OCR and vision units + unified v2 units.
9. Persist outputs:
   - per-technique metrics and sample files
   - comparison figures
   - root compatibility placeholders
   - final `artifacts/run_summary.json`.

Core control snippet:

```python
filings = build_corpus(...)
chunks = chunk_corpus(filings, ...)
store = build_from_chunks(chunks)
extractions = extract_from_filings(filings)
graph = build_graph(filings, extractions)
partition = detect_communities(graph)
```

### Flow B: Single-query answer path (retrieve -> generate)

This logic is encapsulated by `answer_query(...)` in [src/generation.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/generation.py):

1. Call `retriever.retrieve(query, k=k)`.
2. Format retrieved chunks into numbered context and citation metadata (`build_context_block(...)`).
3. Call generator model with `SYSTEM_PROMPT` + `USER_PROMPT`.
4. Return `GenerationResult` and retrieved chunks.

Key output shape (`GenerationResult.to_dict()`):

```json
{
  "query": "string",
  "answer": "string",
  "citations": [
    {
      "id": 1,
      "chunk_id": "string",
      "filing_id": "string",
      "ticker": "string",
      "company_name": "string",
      "section": "string",
      "score": 0.1234,
      "source": "vector|hybrid|...",
      "via": "vector|graph_expand|community|...",
      "text_preview": "string"
    }
  ],
  "model": "granite4.1:8b",
  "prompt_tokens": 0,
  "completion_tokens": 0
}
```

### Flow C: Agentic CRAG loop (`src/agent.py`)

Entrypoint: `AgenticGraphRAG.run(query, k)`.

Steps:

1. `classify_query(...)` -> `local`, `global`, or `factual`.
2. Select retriever priority order (`_primary_retriever_order(...)`).
3. Iteratively retrieve and grade:
   - retrieve chunks with current retriever
   - grade chunk relevance via LLM JSON (`_grade_chunks(...)`)
   - if enough relevant chunks, stop; else fallback to next retriever.
4. Generate final answer with `generate_answer(...)`.
5. Return `AgentResult` with trace steps.

Minimal trace semantics:

```text
classify -> retrieve(iteration N, retriever X) -> grade_docs -> ... -> generate
```

### Flow D: Multimodal execution path

Main modules:

- Asset preparation: [src/multimodal_assets.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/multimodal_assets.py)
- Unit extraction: [src/extensions/multimodal.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/extensions/multimodal.py), [src/extensions/multimodal_ocr.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/extensions/multimodal_ocr.py), [src/extensions/multimodal_vision.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/extensions/multimodal_vision.py)
- Retrieval: [src/extensions/multimodal_retriever.py](/home/ahmad/AI/Github/startup-intelligence-graphrag/src/extensions/multimodal_retriever.py)

Steps:

1. Build `filing_id -> html` map (`fetch_filing_html_map`).
2. Extract table and figure text `MultimodalUnit`s from HTML.
3. Build image sources for OCR/vision from tables or fallback text snapshots.
4. OCR channel (`run_glm_ocr`) and vision channel (`run_qwen_vision`).
5. Retrieve via multimodal dense+sparse fusion with modality weights.

### Key input/output data structures you should know

#### Filing object (`src/ingest.py`)

```text
Filing(
  company_name, cik, ticker, exchange, state_of_incorporation, sic,
  form, filing_date, report_date, filing_id, source_split, source_row_count,
  sections: dict[str, list[str]]
)
```

#### Chunk object (`src/chunking.py`)

```text
Chunk(
  chunk_id, filing_id, ticker, company_name, section,
  text, sentence_ids: list[int], token_count: int
)
```

#### RetrievedChunk object (`src/retrievers.py`)

```text
RetrievedChunk(
  chunk_id, filing_id, ticker, company_name, section, text, sentence_ids,
  score, source, graph_score, vector_score, via, metadata
)
```

#### Eval query row (JSONL in `data/eval/eval_queries.jsonl`)

```json
{
  "query_id": "q01_local_business_es",
  "query": "...",
  "query_type": "local|global|factual",
  "relevant_tickers": ["ES"],
  "relevant_sections": ["Business", "Management's Discussion and Analysis (MD&A)"],
  "keyword_hints": ["..."],
  "reference_answer": "...",
  "retrieval_relevant_chunk_ids": []
}
```

#### Run summary (`artifacts/run_summary.json`)

Top-level keys include:

- `mode`, `timestamp_utc`, `started_utc`, `duration_seconds`
- `dataset_repo`
- `n_filings`, `n_chunks`, `n_eval_queries`, `n_graph_nodes`, `n_graph_edges`
- `n_multimodal_units`, `n_ocr_units`, `n_vision_units`
- `settings`, `schema_snapshot`, `techniques`, `figures`

## Module 4: Setup & Run Guide

### 1) Clean-machine setup

This repository expects `uv` workflows and Python `3.12.10`.

```bash
cd /home/ahmad/AI/Github/startup-intelligence-graphrag
uv python install 3.12.10
uv venv --python 3.12.10 .venv
source .venv/bin/activate
uv sync
```

Optional adapter dependencies:

```bash
uv sync --extra adapter
```

### 2) Required external services

- **Ollama** must be running and models must be available:

```bash
ollama pull qwen3-embedding:4b
ollama pull granite4.1:8b
ollama pull glm-ocr:latest
ollama pull qwen3.5:4b
```

- **Hugging Face token** is required for strict host run preflight:
  - `HUGGINGFACE_HUB_TOKEN` or `HF_TOKEN` (checked in `scripts/run_complete_host.py`).

### 3) Environment configuration

Recommended:

1. Copy `.env.example` to `.env`.
2. Keep strict dataset defaults unless intentionally changing behavior.

Key env vars from `.env.example`:

- Dataset and strictness:
  - `SIRAG_DATASET_REPO`
  - `SIRAG_DATASET_CONFIG`
  - `SIRAG_DATASET_SPLITS`
  - `SIRAG_STRICT_DATASET`
- Local model endpoints and model IDs:
  - `SIRAG_OLLAMA_HOST`
  - `SIRAG_EMBED_MODEL`
  - `SIRAG_GENERATOR_MODEL`
  - `SIRAG_JUDGE_MODEL`
  - `SIRAG_EXTENSION_JUDGE_MODEL`
  - `SIRAG_OCR_MODEL`
  - `SIRAG_VISION_MODEL`
- Optional adapter controls:
  - `SIRAG_ADAPTER_ENABLE`
  - `SIRAG_ADAPTER_BASE_MODEL`
  - `SIRAG_ADAPTER_USE_4BIT`
  - `SIRAG_ADAPTER_MAX_SEQ_LENGTH`
  - `SIRAG_ADAPTER_MAX_TRAIN_EXAMPLES`
  - `SIRAG_ADAPTER_MAX_EVAL_EXAMPLES`
  - `SIRAG_ADAPTER_TRAIN_SPLIT`
  - `SIRAG_ADAPTER_LORA_R`
  - `SIRAG_ADAPTER_LORA_ALPHA`
  - `SIRAG_ADAPTER_LORA_DROPOUT`
  - `SIRAG_ADAPTER_LEARNING_RATE`
  - `SIRAG_ADAPTER_PER_DEVICE_BATCH_SIZE`
  - `SIRAG_ADAPTER_GRADIENT_ACCUMULATION_STEPS`
  - `SIRAG_ADAPTER_MAX_STEPS`
  - `SIRAG_ADAPTER_EVAL_STEPS`
  - `SIRAG_ADAPTER_SAVE_STEPS`
  - `SIRAG_ADAPTER_LOG_STEPS`
  - `SIRAG_ADAPTER_MAX_NEW_TOKENS`
- Runtime sizing:
  - `SIRAG_DEFAULT_N_COMPANIES`
  - `SIRAG_PLACEHOLDER_MODE`

### 4) Typical command sequences

Core real pipeline:

```bash
source .venv/bin/activate
python scripts/run_full_real_project.py --n-companies 30 --force-download
```

Strict host orchestration (core + adapter + notebooks + tests + completion manifest):

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

Placeholder seeding path:

```bash
source .venv/bin/activate
python scripts/run_pipeline.py --mode placeholder
python scripts/seed_extension_placeholders.py
```

### 5) Migration/seeding steps for databases or external services

- There is **no relational database layer** and no migration framework in this repository.
- Artifact seeding is file-based:
  - Base placeholders: `src/pipeline.py -> seed_placeholder_artifacts()`
  - Extension placeholders: `scripts/seed_extension_placeholders.py`
  - Domain adapter placeholders: `src/extensions/domain_adapter.py -> seed_domain_adapter_placeholders()`

## Module 5: Study Plan & Practice Exercises

### Ordered self-study plan

Use this reading order:

1. `README.md` and `pyproject.toml` to understand project scope, dependencies, and run commands.
2. `src/config.py` to learn the settings contract and artifact path layout.
3. Ingestion and corpus shaping:
   - `src/ingest.py`
   - `src/ingest_deerfield.py`
4. Retrieval substrate:
   - `src/chunking.py`
   - `src/embeddings.py`
   - `src/vectorstore.py`
5. Graph and retrieval logic:
   - `src/extractor.py`
   - `src/graph.py`
   - `src/retrievers.py`
6. Generation, judging, and metrics:
   - `src/generation.py`
   - `src/judge.py`
   - `src/evaluator.py`
   - `src/extensions/rag_metrics.py`
7. Runtime orchestration:
   - `scripts/run_full_real_project.py`
   - `scripts/run_complete_host.py`
   - `scripts/run_domain_adapter.py`
8. Optional techniques:
   - sparse/hybrid extensions
   - multimodal extensions
   - domain adapter module
9. Finally, use `tests/` as behavior contracts to validate your understanding of interfaces.

### Practice exercises (with solution outlines)

#### Exercise 1
Question: In this repository, what exact function chain turns raw SEC data into `Filing` objects under strict defaults?

Solution outline:
- Start at `build_corpus()` in `src/ingest.py`.
- Because strict repo defaults to deerfield, it delegates to `build_deerfield_corpus()` in `src/ingest_deerfield.py`.
- Deerfield flow: `download_deerfield_rows()` -> `_record_from_row()` -> `_to_filing()` -> `_sample_filings()` -> `save_deerfield_artifacts()`.

#### Exercise 2
Question: Where is chunk overlap implemented, and is it sentence-based or token-based?

Solution outline:
- See `chunk_section()` in `src/chunking.py`.
- Overlap is token-based (`overlap_tokens`) while moving sentence boundaries.
- `tokenized` sentence arrays are used to walk and backtrack starts.

#### Exercise 3
Question: Which object fields are required for retrieval evaluation relevance derivation?

Solution outline:
- Read `_derive_relevant_ids()` in `src/evaluator.py`.
- Query-side fields: `retrieval_relevant_chunk_ids`, `relevant_tickers`, `relevant_sections`, `keyword_hints`.
- Metadata-side fields: `chunk_id`, `ticker`, `section`, `text`.

#### Exercise 4
Question: How does `GraphLocalRetriever` compute graph expansion influence?

Solution outline:
- In `GraphLocalRetriever._expand_sections()`, seed sections map to neighboring entity nodes.
- Entity boost is scaled by inverse sqrt degree.
- Section contribution uses edge weight and seed score.
- Section scores are converted to chunk-level graph scores via `_section_to_chunks`.

#### Exercise 5
Question: Explain the exact difference between `GraphGlobalRetriever` and `GraphLocalRetriever`.

Solution outline:
- Local retriever expands from seed chunks through entity hops.
- Global retriever embeds community summaries and scores communities directly against query embedding, then collects chunks via section neighbors.
- Local is query-neighborhood expansion; global is community-theme routing.

#### Exercise 6
Question: In the agentic flow, what causes retriever switching?

Solution outline:
- In `AgenticGraphRAG.run()`, each iteration grades chunks via `_grade_chunks()`.
- If filtered relevant chunks are below threshold (`min(2, k)`), it falls back to next retriever in `_primary_retriever_order(query_type)`.

#### Exercise 7
Question: Track how multimodal image assets are generated when HTML table extraction fails.

Solution outline:
- In `scripts/run_full_real_project.py`, `build_table_image_sources(...)` is called first.
- If `image_map` is empty, fallback is `build_text_snapshot_images(...)`.
- OCR/vision units are then built from this fallback `image_map`.

#### Exercise 8
Question: What files prove a strict host run is complete vs failed?

Solution outline:
- `artifacts/run_completion_manifest.json` is the authority.
- `status == "completed"` indicates full pass.
- In current artifacts, status is `"failed"` with failure message on `core_pipeline`.

## Understanding Checklist

Use this to self-verify:

- Can you explain how `build_corpus()` reaches `build_deerfield_corpus()` under current strict settings?
- Can you describe how `Chunk` and `RetrievedChunk` differ, and why both are needed?
- Can you trace one query through `HybridRetriever` and into `generate_answer()` with citations?
- Can you explain node and edge types added by `build_graph()` and why section provenance edges exist?
- Can you explain the agentic retry condition in `AgenticGraphRAG.run()`?
- Can you list which files are written by `run_full_real_project.py` vs `run_complete_host.py`?
- Can you identify all required runtime dependencies (HF token + Ollama models) before strict host orchestration?
- Can you explain where multimodal channels are additive (not replacing core text pipeline)?

