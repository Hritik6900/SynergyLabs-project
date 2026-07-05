"""Judge validation: does the judge deserve trust?

Three artifacts (the assignment asks for at least one; we ship all three):

  A. test-retest consistency — re-run the same cases N times at temperature>0,
     report how often the pass/fail verdict flips and score variance per case.
  B. adversarial probes — verbose-but-wrong and terse-but-correct; report whether
     the judge was fooled.
  C. agreement with gold labels (Cohen's kappa / agreement rate) — PLUGGABLE:
     runs only over cases that carry a `gold_label` (pass/fail) or `gold_overall`.
     If no labels are filled in, it reports "no_labels" instead of failing.
"""

from __future__ import annotations

import os
import statistics
from typing import Any, Optional

from .aggregate import PASS_THRESHOLD
from .judge import Judge


# --------------------------------------------------------------------------- #
# A. Test-retest consistency
# --------------------------------------------------------------------------- #
def test_retest(judge: Judge, cases: list[dict], n_runs: int = 3, temperature: Optional[float] = None) -> dict:
    """Re-score each case n_runs times at temperature>0; measure verdict flips."""
    if temperature is None:
        temperature = float(os.getenv("GROQ_RETEST_TEMPERATURE", "0.7"))
    per_case = []
    flip_count = 0
    for case in cases:
        overalls: list[float] = []
        passes: list[bool] = []
        for _ in range(n_runs):
            r = judge.score_pointwise(case, temperature=temperature, meta={"probe": "retest"})
            ov = r["overall_score_weighted"]
            if ov is not None:
                overalls.append(ov)
                passes.append(ov >= PASS_THRESHOLD)
        # verdict flipped if not all pass-decisions agree
        flipped = len(set(passes)) > 1 if passes else False
        if flipped:
            flip_count += 1
        per_case.append(
            {
                "id": case.get("id"),
                "overalls": overalls,
                "score_std": round(statistics.pstdev(overalls), 3) if len(overalls) > 1 else 0.0,
                "pass_decisions": passes,
                "verdict_flipped": flipped,
            }
        )
    n = len(cases)
    mean_std = (
        round(statistics.fmean([c["score_std"] for c in per_case]), 3) if per_case else None
    )
    return {
        "artifact": "test_retest_consistency",
        "n_cases": n,
        "n_runs": n_runs,
        "temperature": temperature,
        "verdict_flip_rate": round(flip_count / n, 3) if n else None,
        "mean_score_std": mean_std,
        "interpretation": (
            "verdict_flip_rate = fraction of cases whose pass/fail decision was not "
            "unanimous across re-runs. Lower is more reliable. mean_score_std is the "
            "average per-case spread of the overall score."
        ),
        "per_case": per_case,
    }


# --------------------------------------------------------------------------- #
# B. Adversarial probes
# --------------------------------------------------------------------------- #
def adversarial_probes(judge: Judge, probes: dict) -> dict:
    """Run verbose-but-wrong and terse-but-correct; was the judge fooled?

    `probes` = {"verbose_but_wrong": {case...}, "terse_but_correct": {case...}}.
    Fooled if: verbose-but-wrong scores HIGH (>=3.5) OR terse-but-correct scores LOW (<3.5).
    """
    results = {}
    fooled = {}
    for key, expect_high in [("verbose_but_wrong", False), ("terse_but_correct", True)]:
        probe = probes.get(key)
        if not probe:
            results[key] = {"status": "missing"}
            continue
        r = judge.score_pointwise({**probe, "id": key}, meta={"probe": "adversarial", "arm": key})
        ov = r["overall_score_weighted"]
        results[key] = {
            "overall_score": ov,
            "scores": r["scores"],
            "verdict": r.get("verdict"),
            "correctness_rationale": r["criteria"].get("correctness", {}).get("rationale"),
        }
        if ov is None:
            fooled[key] = None
        elif expect_high:
            fooled[key] = ov < PASS_THRESHOLD  # good answer scored low => fooled
        else:
            fooled[key] = ov >= PASS_THRESHOLD  # bad answer scored high => fooled
    any_fooled = any(v is True for v in fooled.values())
    return {
        "artifact": "adversarial_probes",
        "pass_threshold": PASS_THRESHOLD,
        "results": results,
        "fooled": fooled,
        "judge_fooled_overall": any_fooled,
        "interpretation": (
            "The judge should give the verbose-but-wrong answer a LOW score and the "
            "terse-but-correct answer a HIGH score. 'fooled'=true on either means it failed."
        ),
    }


# --------------------------------------------------------------------------- #
# C. Agreement with gold labels (pluggable Cohen's kappa)
# --------------------------------------------------------------------------- #
def _cohen_kappa(a: list[str], b: list[str]) -> Optional[float]:
    """Cohen's kappa for two lists of categorical labels (no sklearn dependency)."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for lab in labels:
        pa = sum(1 for x in a if x == lab) / n
        pb = sum(1 for y in b if y == lab) / n
        pe += pa * pb
    if pe == 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 3)


def gold_agreement(judge: Judge, cases: list[dict]) -> dict:
    """Agreement + Cohen's kappa vs human/gold labels, over labeled cases only.

    A case is labeled if it has `gold_label` in {"pass","fail"} (preferred) or a
    numeric `gold_overall` (thresholded at PASS_THRESHOLD). Unlabeled cases are
    skipped so this stays a no-op until you fill labels in.
    """
    judge_labels: list[str] = []
    gold_labels: list[str] = []
    per_case = []
    for case in cases:
        gl = case.get("gold_label")
        if gl not in {"pass", "fail"} and "gold_overall" in case:
            try:
                gl = "pass" if float(case["gold_overall"]) >= PASS_THRESHOLD else "fail"
            except (TypeError, ValueError):
                gl = None
        if gl not in {"pass", "fail"}:
            continue
        r = judge.score_pointwise(case, meta={"probe": "gold"})
        ov = r["overall_score_weighted"]
        jl = "pass" if (ov is not None and ov >= PASS_THRESHOLD) else "fail"
        judge_labels.append(jl)
        gold_labels.append(gl)
        per_case.append({"id": case.get("id"), "judge": jl, "gold": gl, "agree": jl == gl})

    if not judge_labels:
        return {
            "artifact": "gold_agreement",
            "status": "no_labels",
            "note": "Add gold_label ('pass'/'fail') or gold_overall to cases to enable this.",
        }
    n = len(judge_labels)
    agree = sum(1 for c in per_case if c["agree"])
    return {
        "artifact": "gold_agreement",
        "status": "ok",
        "n_labeled": n,
        "agreement_rate": round(agree / n, 3),
        "cohen_kappa": _cohen_kappa(judge_labels, gold_labels),
        "per_case": per_case,
        "interpretation": (
            "agreement_rate = raw match with gold pass/fail. cohen_kappa corrects for "
            "chance agreement (0=chance, 1=perfect, <0=worse than chance)."
        ),
    }
