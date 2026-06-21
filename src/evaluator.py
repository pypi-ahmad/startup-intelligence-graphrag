"""Evaluation metrics and harnesses for GraphRAG retrieval and generation."""

from __future__ import annotations

import json
import math
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / len(relevant)


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0

    gains = [1.0 if item in relevant else 0.0 for item in retrieved[:k]]
    dcg = sum(gain / math.log2(idx + 2) for idx, gain in enumerate(gains))

    ideal_hits = min(len(relevant), k)
    ideal = [1.0] * ideal_hits + [0.0] * (k - ideal_hits)
    idcg = sum(gain / math.log2(idx + 2) for idx, gain in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[" + re.escape(string.punctuation) + "]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


def bleu_score(prediction: str, references: list[str]) -> float:
    try:
        import sacrebleu
    except ImportError:
        logger.warning("sacrebleu unavailable, returning 0.0")
        return 0.0
    score = sacrebleu.sentence_bleu(prediction, references)
    return float(score.score) / 100.0


def rouge_l(prediction: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge-score unavailable, returning 0.0")
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    result = scorer.score(reference, prediction)
    return float(result["rougeL"].fmeasure)


def meteor_score(prediction: str, reference: str) -> float:
    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
    except Exception:
        logger.warning("nltk meteor unavailable, returning 0.0")
        return 0.0

    try:
        return float(nltk_meteor([_normalize(reference).split()], _normalize(prediction).split()))
    except Exception as exc:
        logger.warning("METEOR failed: {}", exc)
        return 0.0


def bert_score_f1(
    prediction: str,
    reference: str,
    model_type: str = "microsoft/deberta-base-mnli",
) -> float:
    try:
        from bert_score import score as bert_score_compute
    except ImportError:
        logger.warning("bert-score unavailable, returning 0.0")
        return 0.0

    try:
        _, _, f1 = bert_score_compute(
            cands=[prediction],
            refs=[reference],
            model_type=model_type,
            lang="en",
            device="cpu",
            rescale_with_baseline=True,
            verbose=False,
        )
        return float(f1.mean().item())
    except Exception as exc:
        logger.warning("BERTScore failed: {}", exc)
        return 0.0


@dataclass
class RetrievalQueryMetrics:
    query_id: str
    query: str
    k: int
    n_relevant: int
    precision: float
    recall: float
    f1: float
    mrr: float
    ndcg: float
    retrieved_ids: list[str] = field(default_factory=list)
    relevant_ids: list[str] = field(default_factory=list)


@dataclass
class RetrievalMetrics:
    k: int
    n_queries: int
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    mrr: float
    ndcg_at_k: float
    query_level: list[RetrievalQueryMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "n_queries": self.n_queries,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "f1_at_k": self.f1_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "query_level": [q.__dict__ for q in self.query_level],
        }


@dataclass
class GenerationMetrics:
    n_queries: int
    em: float
    bleu: float
    rouge_l: float
    meteor: float
    bert_score_f1: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _derive_relevant_ids(query: dict[str, Any], metadata: list[dict[str, Any]]) -> set[str]:
    explicit = query.get("retrieval_relevant_chunk_ids", [])
    if explicit:
        return set(explicit)

    tickers = {ticker.upper() for ticker in query.get("relevant_tickers", [])}
    sections = {section.lower() for section in query.get("relevant_sections", [])}
    keyword_hints = [kw.lower() for kw in query.get("keyword_hints", []) if kw]

    candidates: set[str] = set()
    for row in metadata:
        ticker_ok = not tickers or row.get("ticker", "").upper() in tickers
        if not ticker_ok:
            continue

        section_ok = True
        if sections:
            section_ok = row.get("section", "").lower() in sections
        if not section_ok:
            continue

        text = str(row.get("text", "")).lower()
        if keyword_hints:
            if not any(kw in text for kw in keyword_hints):
                continue

        candidates.add(row.get("chunk_id", ""))

    if candidates:
        return candidates

    # Fallback: ticker-only weak labels.
    for row in metadata:
        if not tickers or row.get("ticker", "").upper() in tickers:
            candidates.add(row.get("chunk_id", ""))

    return {cid for cid in candidates if cid}


def evaluate_retrieval(
    queries: list[dict[str, Any]],
    retriever,
    metadata: list[dict[str, Any]],
    k: int,
) -> RetrievalMetrics:
    """Evaluate retrieval with weakly/explicitly labeled relevance sets."""

    query_metrics: list[RetrievalQueryMetrics] = []

    for query in queries:
        query_id = str(query.get("query_id", "unknown"))
        text = str(query.get("query", ""))
        relevant_ids = _derive_relevant_ids(query=query, metadata=metadata)

        retrieved = retriever.retrieve(text, k=k)
        retrieved_ids = [row.chunk_id for row in retrieved]

        metric = RetrievalQueryMetrics(
            query_id=query_id,
            query=text,
            k=k,
            n_relevant=len(relevant_ids),
            precision=precision_at_k(retrieved_ids, relevant_ids, k),
            recall=recall_at_k(retrieved_ids, relevant_ids, k),
            f1=f1_at_k(retrieved_ids, relevant_ids, k),
            mrr=mean_reciprocal_rank(retrieved_ids, relevant_ids),
            ndcg=ndcg_at_k(retrieved_ids, relevant_ids, k),
            retrieved_ids=retrieved_ids,
            relevant_ids=sorted(relevant_ids),
        )
        query_metrics.append(metric)

    n = max(len(query_metrics), 1)

    return RetrievalMetrics(
        k=k,
        n_queries=len(query_metrics),
        precision_at_k=round(sum(x.precision for x in query_metrics) / n, 4),
        recall_at_k=round(sum(x.recall for x in query_metrics) / n, 4),
        f1_at_k=round(sum(x.f1 for x in query_metrics) / n, 4),
        mrr=round(sum(x.mrr for x in query_metrics) / n, 4),
        ndcg_at_k=round(sum(x.ndcg for x in query_metrics) / n, 4),
        query_level=query_metrics,
    )


def evaluate_generation(predictions: list[str], references: list[str]) -> GenerationMetrics:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have identical length")

    if not predictions:
        return GenerationMetrics(n_queries=0, em=0.0, bleu=0.0, rouge_l=0.0, meteor=0.0, bert_score_f1=0.0)

    em_scores: list[float] = []
    bleu_scores: list[float] = []
    rouge_scores: list[float] = []
    meteor_scores: list[float] = []
    bert_scores: list[float] = []

    for pred, ref in zip(predictions, references):
        em_scores.append(exact_match(pred, ref))
        bleu_scores.append(bleu_score(pred, [ref]))
        rouge_scores.append(rouge_l(pred, ref))
        meteor_scores.append(meteor_score(pred, ref))
        bert_scores.append(bert_score_f1(pred, ref))

    n = len(predictions)
    return GenerationMetrics(
        n_queries=n,
        em=round(sum(em_scores) / n, 4),
        bleu=round(sum(bleu_scores) / n, 4),
        rouge_l=round(sum(rouge_scores) / n, 4),
        meteor=round(sum(meteor_scores) / n, 4),
        bert_score_f1=round(sum(bert_scores) / n, 4),
    )


def save_metrics(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return path
