"""IR metrics: verified against hand-computed values."""

from __future__ import annotations

import math

import pytest

from eval import retrieval_metrics as rm


def test_all_metrics_hand_computed():
    # retrieved ranked ids; gold = {b, d}. positions (0-indexed): a,b,c,d
    retrieved = ["a", "b", "c", "d"]
    gold = {"b", "d"}
    k = 4

    assert rm.recall_at_k(retrieved, gold, k) == 1.0          # both gold in top-4
    assert rm.hit_rate_at_k(retrieved, gold, k) == 1.0
    assert rm.mrr_at_k(retrieved, gold, k) == pytest.approx(0.5)  # first gold at rank 2
    assert rm.context_precision_at_k(retrieved, gold, k) == pytest.approx(0.5)  # 2 of 4

    # nDCG: gains at ranks 2 and 4 (1-indexed) -> 1/log2(3) + 1/log2(5)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)  # ideal: 2 gold ranked first
    assert rm.ndcg_at_k(retrieved, gold, k) == pytest.approx(dcg / idcg)


def test_no_relevant_retrieved():
    retrieved = ["x", "y", "z"]
    gold = {"g"}
    assert rm.recall_at_k(retrieved, gold, 3) == 0.0
    assert rm.hit_rate_at_k(retrieved, gold, 3) == 0.0
    assert rm.mrr_at_k(retrieved, gold, 3) == 0.0
    assert rm.ndcg_at_k(retrieved, gold, 3) == 0.0
    assert rm.context_precision_at_k(retrieved, gold, 3) == 0.0


def test_k_cutoff_excludes_later_hits():
    retrieved = ["x", "x", "g"]  # gold only at rank 3
    gold = {"g"}
    assert rm.hit_rate_at_k(retrieved, gold, 2) == 0.0   # cut off before the hit
    assert rm.hit_rate_at_k(retrieved, gold, 3) == 1.0


def test_perfect_ranking_scores_one():
    retrieved = ["g1", "g2", "x"]
    gold = {"g1", "g2"}
    assert rm.mrr_at_k(retrieved, gold, 3) == 1.0
    assert rm.ndcg_at_k(retrieved, gold, 3) == pytest.approx(1.0)


def test_aggregate_is_mean():
    rows = [
        {"recall@k": 1.0, "hit_rate@k": 1.0, "mrr@k": 1.0, "ndcg@k": 1.0, "context_precision@k": 0.5},
        {"recall@k": 0.0, "hit_rate@k": 0.0, "mrr@k": 0.0, "ndcg@k": 0.0, "context_precision@k": 0.0},
    ]
    agg = rm.aggregate_metrics(rows)
    assert agg["recall@k"] == pytest.approx(0.5)
    assert agg["context_precision@k"] == pytest.approx(0.25)
