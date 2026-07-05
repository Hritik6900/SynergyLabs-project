"""The judge: builds structured judging prompts and calls the Groq judge model.

Modes
-----
pointwise  : score ONE model_output against the rubric (primary mode).
             Reference-based when expected_output is provided, reference-free
             otherwise. Best for absolute quality gating and per-criterion
             diagnostics.
pairwise   : compare output A vs output B for the same input and pick a winner.
             Best for "is v2 better than v1?" A/B decisions where absolute
             scores are noisy but relative preference is stable. Also the mode
             where POSITION bias is measured (see bias_mitigations.py).

Every prompt embeds:
  * the explicit rubric (definitions + 1-5 anchors)          -> anti score-clustering
  * few-shot anchor examples (a clear 1, 3, and 5)           -> anti score-clustering
  * a length-control instruction                              -> anti verbosity bias
  * a per-criterion grounding requirement (quote the output) -> anti sycophancy

The judge is asked to return a strict JSON object; we request JSON mode and fall
back to prompt-only JSON if the model rejects response_format (handled in
groq_client + parser).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from . import logging_utils
from . import rubric as rubric_mod
from .groq_client import GroqClient, get_client, judge_model
from .parser import parse_verdict

# --------------------------------------------------------------------------- #
# Few-shot anchor examples (calibrate the 1-5 scale; anti score-clustering).
# Kept short and generic so they calibrate WITHOUT leaking task-specific answers.
# --------------------------------------------------------------------------- #
FEWSHOT_ANCHORS = [
    {
        "note": "A clear 1 — confidently wrong + fabricated.",
        "input": "What is the boiling point of water at sea level in Celsius?",
        "model_output": "Water boils at 50°C at sea level, a well-known fact.",
        "verdict": {
            "criteria": {
                "correctness": {"score": 1, "rationale": "States 50°C; the correct value is 100°C — central claim is wrong."},
                "faithfulness": {"score": 1, "rationale": "Invents 'well-known fact' framing for a false value."},
                "completeness": {"score": 3, "rationale": "Answers the asked quantity, but the value is wrong."},
                "instruction_following": {"score": 4, "rationale": "Gives a Celsius number as asked."},
                "tone_safety": {"score": 3, "rationale": "Neutral tone; no safety issue but misleadingly confident."},
            },
            "overall_score": 1.6,
            "verdict": "fail",
        },
    },
    {
        "note": "A clear 3 — partially correct, missing a requested part.",
        "input": "List the three primary additive colors and give a one-line use.",
        "model_output": "The primary additive colors are red and green. They are used in screens.",
        "verdict": {
            "criteria": {
                "correctness": {"score": 3, "rationale": "Red and green are correct but blue is omitted."},
                "faithfulness": {"score": 4, "rationale": "'used in screens' is accurate and grounded."},
                "completeness": {"score": 2, "rationale": "Asked for three colors; only two given, use-line is thin."},
                "instruction_following": {"score": 3, "rationale": "Partially follows the 'three + one-line use' format."},
                "tone_safety": {"score": 5, "rationale": "Appropriate, neutral, safe."},
            },
            "overall_score": 3.2,
            "verdict": "borderline",
        },
    },
    {
        "note": "A clear 5 — correct, grounded, complete, concise (NOT long).",
        "input": "In one sentence, why does ice float on water?",
        "model_output": "Ice floats because water expands as it freezes, making solid ice less dense than liquid water.",
        "verdict": {
            "criteria": {
                "correctness": {"score": 5, "rationale": "Density-from-expansion explanation is correct."},
                "faithfulness": {"score": 5, "rationale": "No unsupported claims; 'less dense than liquid water' is exact."},
                "completeness": {"score": 5, "rationale": "Fully answers the one-sentence 'why'."},
                "instruction_following": {"score": 5, "rationale": "Exactly one sentence as requested."},
                "tone_safety": {"score": 5, "rationale": "Clear, appropriate, safe."},
            },
            "overall_score": 5.0,
            "verdict": "pass",
        },
    },
]


def _fewshot_block() -> str:
    lines = ["## Calibration examples (match new outputs to this 1-5 scale)"]
    for ex in FEWSHOT_ANCHORS:
        lines.append(f"\n{ex['note']}")
        lines.append(f"input: {ex['input']}")
        lines.append(f"model_output: {ex['model_output']}")
        lines.append("verdict: " + json.dumps(ex["verdict"], ensure_ascii=False))
    return "\n".join(lines)


JUDGE_SYSTEM = (
    "You are a rigorous, calibrated evaluation judge. You score model outputs "
    "against an explicit rubric and return STRICT JSON only. You are skeptical of "
    "confident tone and long answers; you reward correct, grounded, complete work. "
    "You never output anything except the JSON object."
)


def _pointwise_schema_hint() -> str:
    keys = rubric_mod.criteria_keys()
    crit_obj = ", ".join(
        f'"{k}": {{"score": <int 1-5>, "rationale": "<quote/reference from model_output>"}}'
        for k in keys
    )
    return (
        "{\n"
        f'  "criteria": {{ {crit_obj} }},\n'
        '  "overall_score": <float 1-5, roughly the weighted mean>,\n'
        '  "verdict": "pass" | "borderline" | "fail",\n'
        '  "overall_rationale": "<2-3 sentence justification>"\n'
        "}"
    )


class MitigationConfig:
    """Which prompt-level bias mitigations are ON.

    Toggling these lets bias_mitigations.py measure before/after numbers by
    running the SAME cases with a mitigation off, then on. Default = all on.
    """

    def __init__(
        self,
        length_control: bool = True,   # verbosity bias
        grounding: bool = True,        # sycophancy / style bias
        fewshot_anchors: bool = True,  # score clustering
    ):
        self.length_control = length_control
        self.grounding = grounding
        self.fewshot_anchors = fewshot_anchors

    def label(self) -> str:
        return (
            f"len={int(self.length_control)},"
            f"ground={int(self.grounding)},"
            f"fewshot={int(self.fewshot_anchors)}"
        )


ALL_ON = MitigationConfig()


def build_pointwise_prompt(case: dict, mit: MitigationConfig = ALL_ON) -> list[dict]:
    """Build messages for pointwise scoring of a single case."""
    rules = ["- Use the FULL 1-5 range. Reserve 5 for genuinely excellent, 1 for clearly bad."]
    if mit.length_control:
        rules.insert(0, f"- {rubric_mod.LENGTH_CONTROL_INSTRUCTION}")
    if mit.grounding:
        rules.insert(0, f"- {rubric_mod.GROUNDING_INSTRUCTION}")
    parts = [
        "# Task: Score the model_output against the rubric below.",
        "",
        "## Rubric",
        rubric_mod.render_rubric_text(),
        "",
        "## Scoring rules",
        *rules,
        "",
        _fewshot_block() if mit.fewshot_anchors else "",
        "",
        "## Case to score",
        f"input:\n{case.get('input','')}",
        "",
        f"system_prompt (that produced the output):\n{case.get('system_prompt','') or '(none)'}",
        "",
        f"model_output:\n{case.get('model_output','')}",
    ]
    if case.get("expected_output"):
        parts += [
            "",
            "reference expected_output (judge faithfulness/correctness against this):",
            str(case["expected_output"]),
        ]
    parts += [
        "",
        "## Output format — return ONLY this JSON object:",
        _pointwise_schema_hint(),
    ]
    user = "\n".join(parts)
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _pairwise_schema_hint() -> str:
    keys = rubric_mod.criteria_keys()
    crit_obj = ", ".join(
        f'"{k}": {{"winner": "A"|"B"|"tie", "rationale": "<grounded reason>"}}'
        for k in keys
    )
    return (
        "{\n"
        f'  "criteria": {{ {crit_obj} }},\n'
        '  "winner": "A" | "B" | "tie",\n'
        '  "confidence": <float 0-1>,\n'
        '  "rationale": "<2-3 sentence justification citing both outputs>"\n'
        "}"
    )


def build_pairwise_prompt(case: dict, output_a: str, output_b: str) -> list[dict]:
    """Build messages comparing output A vs B for the same input."""
    parts = [
        "# Task: Decide which output (A or B) better satisfies the rubric for the input.",
        "",
        "## Rubric",
        rubric_mod.render_rubric_text(),
        "",
        "## Judging rules",
        f"- {rubric_mod.LENGTH_CONTROL_INSTRUCTION}",
        f"- {rubric_mod.GROUNDING_INSTRUCTION}",
        "- Judge on substance. Do NOT prefer an output because it is longer or listed first.",
        "- 'tie' is allowed when neither is clearly better.",
        "",
        "## Case",
        f"input:\n{case.get('input','')}",
        "",
        f"system_prompt:\n{case.get('system_prompt','') or '(none)'}",
    ]
    if case.get("expected_output"):
        parts += ["", f"reference expected_output:\n{case['expected_output']}"]
    parts += [
        "",
        f"### Output A:\n{output_a}",
        "",
        f"### Output B:\n{output_b}",
        "",
        "## Output format — return ONLY this JSON object:",
        _pairwise_schema_hint(),
    ]
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


def _normalize_pointwise(obj: Any) -> dict:
    """Coerce the parsed object into a stable shape; fill weighted overall."""
    criteria_in = (obj or {}).get("criteria", {}) if isinstance(obj, dict) else {}
    scores = {}
    criteria_out = {}
    for c in rubric_mod.RUBRIC:
        entry = criteria_in.get(c.key) if isinstance(criteria_in, dict) else None
        score = None
        rationale = ""
        if isinstance(entry, dict):
            score = entry.get("score")
            rationale = entry.get("rationale", "")
        elif isinstance(entry, (int, float)):
            score = entry
        try:
            score = int(round(float(score))) if score is not None else None
            if score is not None:
                score = max(1, min(5, score))
        except (TypeError, ValueError):
            score = None
        criteria_out[c.key] = {"score": score, "rationale": rationale}
        if score is not None:
            scores[c.key] = score
    weighted = rubric_mod.weighted_overall(scores)
    reported = obj.get("overall_score") if isinstance(obj, dict) else None
    try:
        reported = float(reported) if reported is not None else None
    except (TypeError, ValueError):
        reported = None
    return {
        "criteria": criteria_out,
        "scores": scores,
        "overall_score_weighted": weighted,
        "overall_score_reported": reported,
        "verdict": (obj.get("verdict") if isinstance(obj, dict) else None),
        "overall_rationale": (obj.get("overall_rationale") if isinstance(obj, dict) else None),
    }


class Judge:
    def __init__(
        self,
        client: Optional[GroqClient] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        reasoning_effort: Optional[str] = None,
    ):
        self.client = client or get_client()
        self.model = model or judge_model()
        self.temperature = temperature
        # GPT-OSS models reason before answering; "low" cuts reasoning tokens
        # (much faster/cheaper) which is plenty for rubric scoring. Ignored by
        # models that don't support it (groq_client drops it on error).
        import os as _os
        self.reasoning_effort = reasoning_effort or _os.getenv("GROQ_JUDGE_REASONING_EFFORT", "low")

    def _extra(self) -> dict:
        return {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else None

    # ------------------------------------------------------------------ #
    def score_pointwise(
        self,
        case: dict,
        *,
        temperature: Optional[float] = None,
        meta: dict | None = None,
        mit: MitigationConfig = ALL_ON,
    ) -> dict:
        messages = build_pointwise_prompt(case, mit)
        call_id = logging_utils.next_id("pw")
        out = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=700,
            force_json=True,
            extra=self._extra(),
        )
        raw = out["content"]
        obj, info = parse_verdict(
            raw, call_id=call_id, client=self.client, repair_model=self.model
        )
        result = _normalize_pointwise(obj)
        result["_parse"] = info
        result["_raw"] = raw
        result["_id"] = case.get("id")
        logging_utils.log_judge_call(
            call_id=call_id,
            mode="pointwise",
            model=self.model,
            messages=messages,
            raw_response=raw,
            parsed=obj,
            usage=out["usage"],
            meta={**(meta or {}), "case_id": case.get("id"), "parse": info, "mitigations": mit.label()},
        )
        return result

    # ------------------------------------------------------------------ #
    def compare_pairwise(
        self,
        case: dict,
        output_a: str,
        output_b: str,
        *,
        temperature: Optional[float] = None,
        meta: dict | None = None,
    ) -> dict:
        messages = build_pairwise_prompt(case, output_a, output_b)
        call_id = logging_utils.next_id("cmp")
        out = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=600,
            force_json=True,
            extra=self._extra(),
        )
        raw = out["content"]
        obj, info = parse_verdict(
            raw, call_id=call_id, client=self.client, repair_model=self.model
        )
        winner = None
        confidence = None
        rationale = None
        if isinstance(obj, dict):
            winner = obj.get("winner")
            if isinstance(winner, str):
                winner = winner.strip().upper()
                if winner not in {"A", "B", "TIE"}:
                    winner = None
            confidence = obj.get("confidence")
            rationale = obj.get("rationale")
        result = {
            "winner": winner,
            "confidence": confidence,
            "rationale": rationale,
            "criteria": obj.get("criteria") if isinstance(obj, dict) else None,
            "_parse": info,
            "_raw": raw,
        }
        logging_utils.log_judge_call(
            call_id=call_id,
            mode="pairwise",
            model=self.model,
            messages=messages,
            raw_response=raw,
            parsed=obj,
            usage=out["usage"],
            meta={**(meta or {}), "case_id": case.get("id"), "parse": info},
        )
        return result
