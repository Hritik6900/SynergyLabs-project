"""Aggregate per-case verdicts into a suite report, and compare TWO configs.

Suite report: pass rate, mean per-criterion scores, mean overall, parse-failure
rate. A/B comparison: pointwise mean-score delta AND head-to-head pairwise win
rate, then declare a winner.
"""

from __future__ import annotations

import statistics
from typing import Any, Optional

from . import rubric as rubric_mod
from .judge import Judge

PASS_THRESHOLD = 3.5  # overall weighted score >= this counts as a pass


def score_suite(judge: Judge, cases: list[dict], config_name: str = "config") -> dict:
    """Pointwise-score every case; return per-case results + aggregates."""
    per_case = []
    for case in cases:
        r = judge.score_pointwise(case)
        per_case.append(
            {
                "id": case.get("id"),
                "scores": r["scores"],
                "overall": r["overall_score_weighted"],
                "verdict": r.get("verdict"),
                "passed": (r["overall_score_weighted"] is not None and r["overall_score_weighted"] >= PASS_THRESHOLD),
                "parse_ok": r["_parse"]["ok"],
                "parse_strategy": r["_parse"]["strategy"],
                "overall_rationale": r.get("overall_rationale"),
            }
        )
    return {"config_name": config_name, "per_case": per_case, **_aggregate(per_case)}


def _aggregate(per_case: list[dict]) -> dict:
    n = len(per_case)
    overalls = [c["overall"] for c in per_case if c["overall"] is not None]
    passes = sum(1 for c in per_case if c["passed"])
    parse_fail = sum(1 for c in per_case if not c["parse_ok"])
    per_crit: dict[str, list[float]] = {k: [] for k in rubric_mod.criteria_keys()}
    for c in per_case:
        for k, v in c["scores"].items():
            if isinstance(v, (int, float)):
                per_crit[k].append(v)
    mean_per_crit = {
        k: (round(statistics.fmean(v), 3) if v else None) for k, v in per_crit.items()
    }
    return {
        "n_cases": n,
        "pass_rate": round(passes / n, 3) if n else None,
        "mean_overall": round(statistics.fmean(overalls), 3) if overalls else None,
        "std_overall": round(statistics.pstdev(overalls), 3) if len(overalls) > 1 else 0.0,
        "mean_per_criterion": mean_per_crit,
        "parse_failure_rate": round(parse_fail / n, 3) if n else None,
        "pass_threshold": PASS_THRESHOLD,
    }


def compare_configs(
    judge: Judge,
    cases_a: list[dict],
    cases_b: list[dict],
    *,
    name_a: str = "A",
    name_b: str = "B",
    do_pairwise: bool = True,
) -> dict:
    """Compare two configs producing outputs for the same inputs.

    cases_a / cases_b are aligned by `id` (same input, different model_output).
    Reports pointwise means for each plus, optionally, order-averaged pairwise
    win rate; declares a winner.
    """
    rep_a = score_suite(judge, cases_a, name_a)
    rep_b = score_suite(judge, cases_b, name_b)

    pointwise_delta = None
    if rep_a["mean_overall"] is not None and rep_b["mean_overall"] is not None:
        pointwise_delta = round(rep_a["mean_overall"] - rep_b["mean_overall"], 3)

    pairwise = None
    if do_pairwise:
        pairwise = _pairwise_winrate(judge, cases_a, cases_b, name_a, name_b)

    winner = _declare_winner(rep_a, rep_b, name_a, name_b, pairwise)
    return {
        "name_a": name_a,
        "name_b": name_b,
        "report_a": {k: rep_a[k] for k in ("mean_overall", "std_overall", "pass_rate", "mean_per_criterion")},
        "report_b": {k: rep_b[k] for k in ("mean_overall", "std_overall", "pass_rate", "mean_per_criterion")},
        "pointwise_mean_delta_a_minus_b": pointwise_delta,
        "pairwise": pairwise,
        "winner": winner,
        "full_report_a": rep_a,
        "full_report_b": rep_b,
    }


def _pairwise_winrate(
    judge: Judge, cases_a: list[dict], cases_b: list[dict], name_a: str, name_b: str
) -> dict:
    """Order-averaged head-to-head: run each aligned pair in both orders."""
    by_id_b = {c.get("id"): c for c in cases_b}
    a_wins = b_wins = ties = undecided = 0
    per_pair = []
    for ca in cases_a:
        cb = by_id_b.get(ca.get("id"))
        if cb is None:
            continue
        case = {
            "id": ca.get("id"),
            "input": ca.get("input", ""),
            "system_prompt": ca.get("system_prompt", ""),
            "expected_output": ca.get("expected_output"),
        }
        out_a, out_b = ca.get("model_output", ""), cb.get("model_output", "")
        r1 = judge.compare_pairwise(case, out_a, out_b, meta={"ab": f"{name_a}v{name_b}", "order": "AB"})
        r2 = judge.compare_pairwise(case, out_b, out_a, meta={"ab": f"{name_a}v{name_b}", "order": "BA"})
        # map both to A(name_a)/B(name_b)
        w1 = r1["winner"]  # A=name_a
        w2 = {"A": "B", "B": "A", "TIE": "TIE", None: None}.get(r2["winner"])  # swapped
        if w1 == w2 == "A":
            a_wins += 1; decision = name_a
        elif w1 == w2 == "B":
            b_wins += 1; decision = name_b
        elif w1 == w2 == "TIE":
            ties += 1; decision = "tie"
        else:
            undecided += 1; decision = "undecided(order-inconsistent)"
        per_pair.append({"id": ca.get("id"), "order_AB": w1, "order_BA_mapped": w2, "decision": decision})
    decided = a_wins + b_wins
    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "undecided_order_inconsistent": undecided,
        "a_win_rate_of_decided": round(a_wins / decided, 3) if decided else None,
        "b_win_rate_of_decided": round(b_wins / decided, 3) if decided else None,
        "per_pair": per_pair,
    }


def _declare_winner(
    rep_a: dict, rep_b: dict, name_a: str, name_b: str, pairwise: Optional[dict]
) -> dict:
    reasons = []
    # Prefer pairwise decision when available and decisive.
    if pairwise and (pairwise["a_wins"] + pairwise["b_wins"]) > 0:
        if pairwise["a_wins"] > pairwise["b_wins"]:
            win = name_a
        elif pairwise["b_wins"] > pairwise["a_wins"]:
            win = name_b
        else:
            win = "tie"
        reasons.append(f"pairwise head-to-head: {name_a}={pairwise['a_wins']} vs {name_b}={pairwise['b_wins']}")
    else:
        ma, mb = rep_a["mean_overall"], rep_b["mean_overall"]
        if ma is None or mb is None:
            win = "undetermined"
        elif abs(ma - mb) < 0.1:
            win = "tie"
        elif ma > mb:
            win = name_a
        else:
            win = name_b
        reasons.append(f"pointwise mean overall: {name_a}={ma} vs {name_b}={mb}")
    return {"winner": win, "basis": reasons}
