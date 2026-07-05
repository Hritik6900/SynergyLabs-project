"""CLI runner for the LLM-as-Judge pipeline.

Usage
-----
  # full run: suite report + bias report + A/B comparison
  python main.py --suite suites/test_suite.json --probes suites/adversarial_probes.json

  # only score the suite
  python main.py --suite suites/test_suite.json --only score

  # subcommands: score | bias | validate | ab | all (default)

Outputs (written to results/):
  suite_report.json   — pass rate, mean per-criterion, per-case verdicts
  bias_report.json     — before/after numbers for all 5 biases + position flip rate
  validation.json      — test-retest, adversarial probes, gold agreement/kappa
  ab_comparison.json   — two-config comparison declaring a winner
  usage.json           — judge token/call/cost totals
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# allow `python main.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import aggregate, bias_mitigations as bias, validate
from src.groq_client import USAGE, generator_model, get_client, judge_model
from src.judge import Judge

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _load_suite(path: str) -> dict:
    """Load a suite/probes file as JSON or YAML (by extension)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        import yaml  # PyYAML; only imported when a YAML file is used
        return yaml.safe_load(text)
    return json.loads(text)


# Back-compat alias
_load_json = _load_suite


def _real_cases(suite: dict) -> list[dict]:
    """Return cases that are actually filled in (skip FILL_ME / empty inputs)."""
    cases = suite.get("cases", suite) if isinstance(suite, dict) else suite
    out = []
    for c in cases:
        if not isinstance(c, dict):
            continue
        if str(c.get("id", "")).startswith("FILL_ME"):
            continue
        if not str(c.get("input", "")).strip() or not str(c.get("model_output", "")).strip():
            continue
        out.append(c)
    return out


def _split_ab(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """Cases with model_output_b -> (config A uses model_output, B uses model_output_b)."""
    a, b = [], []
    for c in cases:
        if str(c.get("model_output_b", "")).strip():
            a.append(c)
            b.append({**c, "model_output": c["model_output_b"]})
    return a, b


def _write(name: str, obj: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def run_score(judge: Judge, cases: list[dict]) -> dict:
    print(f"[score] pointwise-scoring {len(cases)} cases ...")
    report = aggregate.score_suite(judge, cases, config_name="suite")
    report["usage"] = USAGE.summary()
    p = _write("suite_report.json", report)
    print(f"[score] pass_rate={report['pass_rate']} mean_overall={report['mean_overall']} -> {p}")
    return report


def run_bias(judge: Judge, cases: list[dict], probes: dict) -> dict:
    print("[bias] measuring 5 biases (before/after) ...")
    ab_a, _ = _split_ab(cases)
    # position bias needs pairs; build from A/B outputs
    pairs = [
        {
            "id": c["id"],
            "input": c["input"],
            "system_prompt": c.get("system_prompt", ""),
            "expected_output": c.get("expected_output"),
            "output_a": c["model_output"],
            "output_b": c["model_output_b"],
        }
        for c in cases
        if str(c.get("model_output_b", "")).strip()
    ]
    report = {
        "judge_model": judge_model(),
        "generator_model": generator_model(),
        "position": bias.position_bias_check(judge, pairs) if pairs else {"bias": "position", "status": "no A/B pairs in suite"},
        "verbosity": bias.verbosity_bias_check(judge, probes["verbosity_probe"]),
        "self_enhancement": bias.self_enhancement_check(judge),
        "sycophancy": bias.sycophancy_check(judge, probes["sycophancy_probe"]),
        "score_clustering": bias.score_clustering_check(judge, cases),
    }
    report["usage"] = USAGE.summary()
    p = _write("bias_report.json", report)
    pr = report["position"]
    print(f"[bias] position flip_rate={pr.get('flip_rate')} | "
          f"verbosity improved={report['verbosity']['improved']} | "
          f"self_enh={report['self_enhancement']['status']} | "
          f"sycophancy fooled_after={report['sycophancy']['fooled_after']} | "
          f"clustering improved={report['score_clustering']['improved']} -> {p}")
    return report


def run_validate(judge: Judge, cases: list[dict], probes: dict, n_runs: int) -> dict:
    print(f"[validate] test-retest ({n_runs} runs) + adversarial probes + gold agreement ...")
    report = {
        "test_retest": validate.test_retest(judge, cases, n_runs=n_runs),
        "adversarial": validate.adversarial_probes(judge, probes["validation_probes"]),
        "gold_agreement": validate.gold_agreement(judge, cases),
    }
    report["usage"] = USAGE.summary()
    p = _write("validation.json", report)
    print(f"[validate] retest_flip_rate={report['test_retest']['verdict_flip_rate']} | "
          f"judge_fooled={report['adversarial']['judge_fooled_overall']} | "
          f"gold={report['gold_agreement'].get('status')} -> {p}")
    return report


def run_ab(judge: Judge, cases: list[dict]) -> dict:
    a, b = _split_ab(cases)
    if not a:
        print("[ab] no cases with model_output_b; skipping A/B.")
        return {"status": "no A/B cases (need model_output_b)"}
    print(f"[ab] comparing config A vs B over {len(a)} aligned cases ...")
    report = aggregate.compare_configs(judge, a, b, name_a="config_A_v1", name_b="config_B_v2")
    report["usage"] = USAGE.summary()
    p = _write("ab_comparison.json", report)
    print(f"[ab] winner={report['winner']['winner']} ({report['winner']['basis']}) -> {p}")
    return report


def main():
    ap = argparse.ArgumentParser(description="LLM-as-Judge pipeline (Groq-backed).")
    ap.add_argument("--suite", default="suites/test_suite.json")
    ap.add_argument("--probes", default="suites/adversarial_probes.json")
    ap.add_argument("--only", choices=["score", "bias", "validate", "ab", "all"], default="all")
    ap.add_argument("--retest-runs", type=int, default=3)
    ap.add_argument(
        "--generate",
        action="store_true",
        help="Actively call the GENERATOR model to produce missing model_output "
        "(and model_output_b from system_prompt_b) before judging.",
    )
    args = ap.parse_args()

    suite = _load_suite(args.suite)
    probes = _load_suite(args.probes)
    cases_raw = suite.get("cases", suite) if isinstance(suite, dict) else suite

    if args.generate:
        from src.generator import enrich_case
        client0 = get_client()
        enriched = []
        for c in cases_raw:
            if not isinstance(c, dict) or str(c.get("id", "")).startswith("FILL_ME"):
                continue
            if not str(c.get("input", "")).strip():
                continue
            enriched.append(enrich_case(c, client=client0))
        suite = {"cases": enriched}
        out_path = _write("generated_suite.json", suite)
        n_gen = sum(1 for c in enriched if c.get("_generated_output") or c.get("_generated_output_b"))
        print(f"[generate] produced outputs via {generator_model()} for {n_gen} case(s) -> {out_path}\n")

    cases = _real_cases(suite)
    if not cases:
        print("No filled-in cases found in the suite. Add cases (see FILL_ME entries).")
        sys.exit(1)

    print(f"Judge     = {judge_model()}")
    print(f"Generator = {generator_model()}")
    print(f"Cases     = {len(cases)} filled  |  mode = {args.only}\n")

    judge = Judge(get_client())

    if args.only in ("score", "all"):
        run_score(judge, cases)
    if args.only in ("bias", "all"):
        run_bias(judge, cases, probes)
    if args.only in ("validate", "all"):
        run_validate(judge, cases, probes, args.retest_runs)
    if args.only in ("ab", "all"):
        run_ab(judge, cases)

    _write("usage.json", USAGE.summary())
    print(f"\n[usage] calls={USAGE.calls} total_tokens={USAGE.total_tokens} "
          f"est_cost_usd=${USAGE.estimated_cost_usd()}")


if __name__ == "__main__":
    main()
