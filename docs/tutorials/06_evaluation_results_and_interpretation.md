# Tutorial 06: Evaluation, Results, and Interpretation

## What is this section?

A practical guide to reading this repository’s real evaluation artifacts and interpreting what they prove.

## Why this evaluation stack is used

RAG quality has multiple failure surfaces. This project evaluates:
- retrieval quality
- text generation quality
- RAG-groundedness quality
- LLM-as-a-judge quality

## Metric families in this repository

### A) Retrieval metrics

Implemented in `src/evaluator.py`:
- Precision@K
- Recall@K
- F1@K
- MRR
- NDCG@K

### B) Generation metrics

Implemented in `src/evaluator.py`:
- Exact Match (EM)
- BLEU
- ROUGE-L
- METEOR
- BERTScore F1

### C) RAG metrics

Implemented in `src/extensions/rag_metrics.py`:
- Faithfulness
- Context Precision
- Context Recall
- Answer Relevancy

### D) Judge metrics (LLM-as-a-Judge)

Implemented in `src/judge.py` using `granite4.1:8b`:
- Correctness
- Relevance
- Completeness
- Groundedness
- Hallucination Risk
- Overall score (derived)

## Where metrics are written

- Per-technique full bundles:
  - `artifacts/eval/*_full_metrics.json`
- Compatibility metric payloads:
  - `artifacts/eval/*_placeholder.json` with `status: executed` where available
- Unified run summary:
  - `artifacts/run_summary.json`

## Real run result highlights

Source of truth: `artifacts/run_summary.json` and per-technique full metrics.

1. Best retrieval NDCG in this run:
- `graphrag_local`: `0.4693`

2. Strongest end-to-end answer quality in this run:
- `agentic_crag`
  - `ROUGE-L=0.1207`
  - `METEOR=0.1642`
  - RAG: `faithfulness=0.95`, `context_precision=1.0`, `answer_relevancy=1.0`
  - judge overall row value: `4.8`

3. Multimodal channels integrated but low impact on this query slice:
- multimodal variants retrieved off-target context for evaluated query
- resulting RAG/judge scores near minimum for those variants in this run

## Practical interpretation guidelines

- Non-zero generation metrics do not imply groundedness; always read RAG + judge scores.
- A high retrieval NDCG on local queries does not imply global thematic performance.
- Multimodal effectiveness must be validated with multimodal-relevant query sets.
- Adapter claims must wait for completed adapter execute artifacts.

## Observed limitations in the executed benchmark slice

- `n_eval_queries=1` in current run summary; this is too small for broad ranking claims.
- Several techniques have sparse metric coverage due run profile and query mix.
- Last strict host completion manifest is failed; treat it as an orchestration status signal, not as evidence that all techniques are invalid.

## Recommended next execution priorities

1. expand evaluated query set (local + global + visual evidence targets)
2. re-run strict host orchestration to completion
3. run adapter execute mode and publish base-vs-adapter deltas
4. compare latency distributions by technique under identical query batches
