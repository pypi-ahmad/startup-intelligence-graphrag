# Zero-to-Hero Roadmap: Startup Intelligence GraphRAG

## 1) What this project is

This project is a local, end-to-end RAG system for startup/company intelligence over real SEC filings.
It uses a strict ingestion source (`deerfieldgreen/stk-sec-filings`), builds retrieval + graph + agentic + multimodal stages, and evaluates outputs with automated and judge-based metrics.

## 2) Why SEC filings for startup/company intelligence

SEC filings are high-signal business documents with:
- business model and segment disclosures
- risk factors and strategic signals
- management narratives and financial context

They are valuable for intelligence use cases such as:
- competitor and risk mapping
- operational and strategic trend analysis
- grounded QA for research workflows

## 3) End-to-end workflow in this repository

```mermaid
flowchart LR
    A[Hugging Face SEC Dataset] --> B[Ingestion + Normalization]
    B --> C[Token-Aware Chunking]
    C --> D[Dense Embeddings + FAISS]
    B --> E[Entity/Relation Extraction]
    E --> F[Knowledge Graph + Communities]
    D --> G[Retrieval Variants]
    F --> G
    G --> H[Generation]
    H --> I[LLM Judge + RAG Metrics]
    G --> I
    I --> J[Artifacts + Figures + Tutorial Notebooks]
```

## 4) Code-first map

- Ingestion and strict dataset contract:
  - `src/ingest.py`
  - `src/ingest_deerfield.py`
- Chunking, embeddings, vector index:
  - `src/chunking.py`
  - `src/embeddings.py`
  - `src/vectorstore.py`
- Knowledge graph and extraction:
  - `src/extractor.py`
  - `src/graph.py`
- Retrieval, generation, judging:
  - `src/retrievers.py`
  - `src/generation.py`
  - `src/judge.py`
- Advanced techniques:
  - `src/extensions/hybrid_sparse_dense.py`
  - `src/agent.py`
  - `src/extensions/multimodal*.py`
  - `src/extensions/domain_adapter.py`
- Run orchestration:
  - `scripts/run_full_real_project.py`
  - `scripts/run_complete_host.py`

## 5) What is already executed vs pending

From current artifacts:
- Executed and available: `artifacts/run_summary.json` with real outputs and metrics.
- Strict host completion manifest currently failed: `artifacts/run_completion_manifest.json`.
- Domain-adapter comparison artifacts are still placeholders unless adapter execute mode succeeds.

## 6) Learning order (recommended)

1. Core GraphRAG foundations (`01_core_graphrag_pipeline.md`)
2. Hybrid sparse+dense retrieval (`02_hybrid_rag_sparse_dense.md`)
3. Agentic corrective routing (`03_agentic_rag_crag.md`)
4. Multimodal pipeline (`04_multimodal_rag.md`)
5. Optional adaptation stage (`05_optional_domain_adapter_unsloth_peft_trl.md`)
6. Evaluation interpretation (`06_evaluation_results_and_interpretation.md`)
