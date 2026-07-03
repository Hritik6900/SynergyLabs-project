"""Information-retrieval metrics computed against gold-relevant chunk ids.

All functions take:
  * ``retrieved``: the ranked list of retrieved chunk ids (best first, length <= k)
  * ``gold``: the set of chunk ids that are relevant for the question
  * ``k``: the cutoff

Metrics implemented (standard IR definitions):

  Recall@k        fraction of gold chunks that appear in the top-k
  Hit Rate@k      1 if at least one gold chunk is in the top-k, else 0
  MRR@k           1 / (rank of first relevant chunk), 0 if none in top-k
  nDCG@k          DCG@k / IDCG@k with binary relevance (1 if gold, else 0)
  Context Prec@k  fraction of the retrieved (up to k) chunks that are relevant

These are the metrics the brief asks for; aggregates are simple means across
questions.
"""

from __future__ import annotations

import math
from statistics import mean


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    topk = retrieved[:k]
    hit = len(set(topk) & gold)
    return hit / len(gold)


def hit_rate_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def mrr_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    for rank, cid in enumerate(retrieved[:k], start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    # DCG with binary relevance.
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k]):
        rel = 1.0 if cid in gold else 0.0
        dcg += rel / math.log2(i + 2)  # positions are 1-indexed -> log2(i+2)
    # Ideal DCG: all relevant items ranked first (up to k of them).
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def context_precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    topk = retrieved[:k]
    if not topk:
        return 0.0
    relevant = sum(1 for cid in topk if cid in gold)
    return relevant / len(topk)


def per_question_metrics(retrieved: list[str], gold: set[str], k: int) -> dict:
    """All metrics for a single question."""
    return {
        "recall@k": recall_at_k(retrieved, gold, k),
        "hit_rate@k": hit_rate_at_k(retrieved, gold, k),
        "mrr@k": mrr_at_k(retrieved, gold, k),
        "ndcg@k": ndcg_at_k(retrieved, gold, k),
        "context_precision@k": context_precision_at_k(retrieved, gold, k),
    }


def aggregate_metrics(per_question: list[dict]) -> dict:
    """Mean of each metric across questions (ignores questions with no gold)."""
    if not per_question:
        return {}
    keys = ["recall@k", "hit_rate@k", "mrr@k", "ndcg@k", "context_precision@k"]
    return {key: mean(q[key] for q in per_question) for key in keys}
