"""The five judge biases: each named, mitigated in code, and MEASURED (before vs after).

Every function here returns a dict with concrete numbers so main.py can assemble
results/bias_report.json. Nothing is a comment-only mitigation.

  1. Position (A/B order) ....... run each pair in BOTH orders -> flip rate.
  2. Verbosity / length ......... length-control instruction; padded probe over-score delta.
  3. Self-enhancement .......... judge family != generator family; config assertion (+ optional gap probe).
  4. Sycophancy / style ........ per-criterion grounding; confidently-wrong probe correctness delta.
  5. Score clustering .......... few-shot anchors; score spread (std / distinct values) with vs without.

"before" = mitigation OFF, "after" = mitigation ON, on the SAME probes/cases.
"""

from __future__ import annotations

import statistics
from typing import Any, Optional

from .groq_client import generator_model, judge_model
from .judge import Judge, MitigationConfig


# --------------------------------------------------------------------------- #
# 1. POSITION BIAS
# --------------------------------------------------------------------------- #
def _winner_in_natural_order(verdict_winner: Optional[str], swapped: bool) -> Optional[str]:
    """Map a pairwise winner back to the ORIGINAL A/B labels.

    In the swapped run we present original-B as "A" and original-A as "B", so a
    reported "A" actually means original-B won. TIE stays TIE.
    """
    if verdict_winner is None:
        return None
    if verdict_winner == "TIE":
        return "TIE"
    if not swapped:
        return verdict_winner
    return "B" if verdict_winner == "A" else "A"


def position_bias_check(
    judge: Judge, pairs: list[dict]
) -> dict:
    """Run each (input, output_a, output_b) pair in BOTH orders; report flip rate.

    A 'flip' = the winner (mapped to original labels) disagrees between the two
    orders, OR one order ties while the other doesn't. Consistent verdicts are
    the mitigated output; the flip rate quantifies residual position bias.
    """
    per_pair = []
    flips = 0
    consistent = 0
    for p in pairs:
        case = {
            "id": p.get("id"),
            "input": p.get("input", ""),
            "system_prompt": p.get("system_prompt", ""),
            "expected_output": p.get("expected_output"),
        }
        a, b = p.get("output_a", ""), p.get("output_b", "")
        # Order 1: A=a, B=b
        r1 = judge.compare_pairwise(case, a, b, meta={"order": "AB"})
        w1 = _winner_in_natural_order(r1["winner"], swapped=False)
        # Order 2: A=b, B=a  (swapped)
        r2 = judge.compare_pairwise(case, b, a, meta={"order": "BA"})
        w2 = _winner_in_natural_order(r2["winner"], swapped=True)

        flipped = (w1 != w2)
        if flipped:
            flips += 1
        else:
            consistent += 1
        per_pair.append(
            {
                "id": p.get("id"),
                "winner_order_AB": w1,
                "winner_order_BA": w2,
                "flipped": flipped,
                "agreed_winner": (w1 if (not flipped and w1 in {"A", "B"}) else None),
            }
        )
    n = len(pairs)
    return {
        "bias": "position",
        "mitigation": "run each pair in both orders; require order-agreement for a decision",
        "n_pairs": n,
        "flips": flips,
        "flip_rate": round(flips / n, 3) if n else None,
        "consistent": consistent,
        "consistency_rate": round(consistent / n, 3) if n else None,
        "interpretation": (
            "flip_rate is the residual position bias: fraction of pairs whose winner "
            "changed when we swapped presentation order. Only order-agreed winners are trusted."
        ),
        "per_pair": per_pair,
    }


# --------------------------------------------------------------------------- #
# 2. VERBOSITY / LENGTH BIAS
# --------------------------------------------------------------------------- #
def verbosity_bias_check(judge: Judge, probe: dict) -> dict:
    """Score a concise-correct vs verbose-padded-low-quality answer to the SAME input.

    A length-biased judge scores the padded answer at least as high as the concise
    one. We measure the overall-score gap (concise - padded) WITHOUT the length-
    control instruction (before) and WITH it (after). Mitigation works if the gap
    increases (concise pulls ahead) after the instruction is added.
    """
    input_text = probe["input"]
    sys = probe.get("system_prompt", "")
    concise = probe["concise_correct"]
    padded = probe["verbose_padded"]

    def _pair_scores(mit: MitigationConfig) -> dict:
        c = judge.score_pointwise(
            {"id": "concise", "input": input_text, "system_prompt": sys, "model_output": concise,
             "expected_output": probe.get("expected_output")},
            mit=mit, meta={"probe": "verbosity", "arm": "concise"},
        )
        p = judge.score_pointwise(
            {"id": "padded", "input": input_text, "system_prompt": sys, "model_output": padded,
             "expected_output": probe.get("expected_output")},
            mit=mit, meta={"probe": "verbosity", "arm": "padded"},
        )
        cs = c["overall_score_weighted"]
        ps = p["overall_score_weighted"]
        return {
            "concise_score": cs,
            "padded_score": ps,
            "gap_concise_minus_padded": (round(cs - ps, 3) if (cs is not None and ps is not None) else None),
            "padded_len_chars": len(padded),
            "concise_len_chars": len(concise),
        }

    before = _pair_scores(MitigationConfig(length_control=False))
    after = _pair_scores(MitigationConfig(length_control=True))
    return {
        "bias": "verbosity",
        "mitigation": "length-control instruction in rubric prompt ('do not reward length alone')",
        "before_no_length_control": before,
        "after_length_control": after,
        "improved": (
            after["gap_concise_minus_padded"] is not None
            and before["gap_concise_minus_padded"] is not None
            and after["gap_concise_minus_padded"] > before["gap_concise_minus_padded"]
        ),
        "interpretation": (
            "A positive gap means the concise-correct answer scored higher than the padded one. "
            "Mitigation succeeds when the gap rises from 'before' to 'after'."
        ),
    }


# --------------------------------------------------------------------------- #
# 3. SELF-ENHANCEMENT BIAS
# --------------------------------------------------------------------------- #
def _family_of(model_id: str) -> str:
    m = model_id.lower()
    if "gpt-oss" in m or m.startswith("openai/"):
        return "gpt-oss (OpenAI open-weight)"
    if "qwen" in m:
        return "qwen (Alibaba)"
    if "gemma" in m:
        return "gemma (Google)"
    if "llama" in m:
        return "llama (Meta)"
    if "kimi" in m:
        return "kimi (Moonshot)"
    if "deepseek" in m:
        return "deepseek"
    if "mixtral" in m or "mistral" in m:
        return "mistral"
    return model_id.split("/")[0] if "/" in model_id else "unknown"


def self_enhancement_check(
    judge: Judge,
    optional_same_family_output: Optional[dict] = None,
) -> dict:
    """Primary mitigation = judge and generator are DIFFERENT families (config-level).

    We assert the two configured families differ and document why they are
    meaningfully different (different orgs, pretraining corpora, tokenizers,
    RLHF lineage — not two sizes of one base).

    Optional quantitative gap probe: if `optional_same_family_output` is provided
    (an output the caller believes was produced by the judge's own family plus a
    matched output from a different family), we score both and report the delta.
    Off by default because it requires generating same-family outputs.
    """
    jm, gm = judge_model(), generator_model()
    jf, gf = _family_of(jm), _family_of(gm)
    result: dict[str, Any] = {
        "bias": "self_enhancement",
        "mitigation": "judge and generator drawn from different model families on Groq",
        "judge_model": jm,
        "judge_family": jf,
        "generator_model": gm,
        "generator_family": gf,
        "families_differ": jf != gf,
        "why_meaningfully_different": (
            "gpt-oss (OpenAI open-weight MoE) and qwen (Alibaba) come from different "
            "organizations with different pretraining corpora, tokenizers, and RLHF "
            "pipelines — not two sizes of one base model. A judge cannot self-prefer "
            "outputs from a family it did not produce."
        ),
        "status": "PASS" if jf != gf else "FAIL — judge and generator share a family; change GROQ_*_MODEL",
    }
    if optional_same_family_output:
        same = judge.score_pointwise(
            {**optional_same_family_output["same_family_case"], "id": "same_family"},
            meta={"probe": "self_enhancement", "arm": "same_family"},
        )
        diff = judge.score_pointwise(
            {**optional_same_family_output["diff_family_case"], "id": "diff_family"},
            meta={"probe": "self_enhancement", "arm": "diff_family"},
        )
        ss = same["overall_score_weighted"]
        ds = diff["overall_score_weighted"]
        result["gap_probe"] = {
            "same_family_score": ss,
            "diff_family_score": ds,
            "self_enhancement_gap": (round(ss - ds, 3) if (ss is not None and ds is not None) else None),
            "note": "positive gap on matched-quality outputs would indicate self-preference",
        }
    return result


# --------------------------------------------------------------------------- #
# 4. SYCOPHANCY / STYLE BIAS
# --------------------------------------------------------------------------- #
def sycophancy_check(judge: Judge, probe: dict) -> dict:
    """Confidently-wrong probe: a fluent, assertive answer that is factually wrong.

    A sycophantic/style-swayed judge rewards confident tone. Mitigation = require
    each score to quote/reference the output (grounding). We score the probe
    WITHOUT grounding (before) and WITH grounding (after) and report the
    correctness score. Mitigation works if correctness drops toward 1 when the
    judge is forced to ground its rationale in the (wrong) content.
    """
    case = {
        "input": probe["input"],
        "system_prompt": probe.get("system_prompt", ""),
        "model_output": probe["confidently_wrong"],
        "expected_output": probe.get("expected_output"),
    }

    def _score(mit: MitigationConfig, arm: str) -> dict:
        r = judge.score_pointwise({**case, "id": arm}, mit=mit, meta={"probe": "sycophancy", "arm": arm})
        return {
            "correctness_score": r["scores"].get("correctness"),
            "faithfulness_score": r["scores"].get("faithfulness"),
            "overall_score": r["overall_score_weighted"],
            "correctness_rationale": r["criteria"].get("correctness", {}).get("rationale"),
        }

    before = _score(MitigationConfig(grounding=False), "no_grounding")
    after = _score(MitigationConfig(grounding=True), "grounding")
    bc = before["correctness_score"]
    ac = after["correctness_score"]
    return {
        "bias": "sycophancy",
        "mitigation": "force per-criterion grounding (quote/reference the output in every rationale)",
        "before_no_grounding": before,
        "after_grounding": after,
        "improved": (bc is not None and ac is not None and ac <= bc),
        "fooled_after": (ac is not None and ac >= 4),
        "interpretation": (
            "The probe is factually wrong, so a good judge gives LOW correctness. "
            "Mitigation succeeds if grounding lowers (or holds low) the correctness score. "
            "'fooled_after'=true means the judge still rated a wrong answer highly."
        ),
    }


# --------------------------------------------------------------------------- #
# 5. SCORE CLUSTERING
# --------------------------------------------------------------------------- #
def _spread_stats(scores: list[float]) -> dict:
    vals = [s for s in scores if isinstance(s, (int, float))]
    if not vals:
        return {"n": 0, "std": None, "min": None, "max": None, "range": None, "distinct": 0}
    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 3),
        "std": round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
        "range": round(max(vals) - min(vals), 3),
        "distinct": len({round(v, 1) for v in vals}),
    }


def score_clustering_check(judge: Judge, cases: list[dict]) -> dict:
    """Score the suite WITHOUT few-shot anchors (before) and WITH them (after).

    Open models bunch scores in a narrow band (e.g. everything 3-4). Anchors
    calibrate the scale. We measure spread (std, range, distinct values) of the
    overall weighted scores. Higher spread after anchors = less clustering.
    """
    def _run(mit: MitigationConfig, arm: str) -> list[float]:
        out = []
        for c in cases:
            r = judge.score_pointwise(c, mit=mit, meta={"probe": "clustering", "arm": arm})
            if r["overall_score_weighted"] is not None:
                out.append(r["overall_score_weighted"])
        return out

    before_scores = _run(MitigationConfig(fewshot_anchors=False), "no_anchors")
    after_scores = _run(MitigationConfig(fewshot_anchors=True), "anchors")
    before = _spread_stats(before_scores)
    after = _spread_stats(after_scores)
    return {
        "bias": "score_clustering",
        "mitigation": "few-shot anchor examples (clear 1, 3, 5) calibrate the 1-5 scale",
        "n_cases": len(cases),
        "before_no_anchors": {**before, "scores": before_scores},
        "after_anchors": {**after, "scores": after_scores},
        "improved": (
            after["std"] is not None and before["std"] is not None and after["std"] >= before["std"]
        ),
        "interpretation": (
            "Wider spread (higher std / range / distinct values) means the judge is "
            "using the full scale instead of clustering. Mitigation succeeds if spread rises."
        ),
    }
