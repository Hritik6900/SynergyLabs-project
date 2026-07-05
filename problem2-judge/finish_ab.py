"""Lean pairwise-only A/B to produce results/ab_comparison.json fast under free-tier limits.

Skips the redundant double pointwise re-scoring in aggregate.compare_configs and runs
ONLY the order-averaged pairwise head-to-head that actually declares the winner.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.groq_client import get_client, judge_model, generator_model
from src.judge import Judge
from src.aggregate import _pairwise_winrate, _declare_winner
from src.groq_client import USAGE

SUBSET = None  # None = all pairs; or a list of ids

suite = json.load(open("suites/test_suite.json"))
cases = [c for c in suite["cases"]
         if not str(c["id"]).startswith("FILL_ME")
         and str(c.get("model_output", "")).strip()
         and str(c.get("model_output_b", "")).strip()]
if SUBSET:
    cases = [c for c in cases if c["id"] in SUBSET]

a = cases
b = [{**c, "model_output": c["model_output_b"]} for c in cases]

print(f"A/B pairwise over {len(a)} pairs | judge={judge_model()}")
judge = Judge(get_client())
pw = _pairwise_winrate(judge, a, b, "config_A_v1", "config_B_v2")
winner = _declare_winner(
    {"mean_overall": None}, {"mean_overall": None}, "config_A_v1", "config_B_v2", pw
)
report = {
    "name_a": "config_A_v1",
    "name_b": "config_B_v2",
    "mode": "pairwise_only (order-averaged, both A-B and B-A)",
    "n_pairs": len(a),
    "pairwise": pw,
    "winner": winner,
    "usage": USAGE.summary(),
}
json.dump(report, open("results/ab_comparison.json", "w"), indent=2)
print("WINNER:", winner["winner"], "|", winner["basis"])
print(f"a_wins={pw['a_wins']} b_wins={pw['b_wins']} ties={pw['ties']} undecided={pw['undecided_order_inconsistent']}")
print(f"calls={USAGE.calls} cost=${USAGE.estimated_cost_usd()}")
