"""Explicit judging rubric.

Not a bare number: five weighted criteria, each with a definition and
concrete 1-5 score anchors (what a 1 looks like vs a 3 vs a 5). The anchors
are the single most important lever against SCORE CLUSTERING on open-weight
judges — they turn "give it a score" into "match it to a described level".

The rubric is data (not prose) so judge.py can render it into the prompt and
aggregate.py can weight criteria without re-parsing English.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Criterion:
    key: str
    name: str
    definition: str
    weight: float
    anchors: dict  # {1: "...", 3: "...", 5: "..."}


RUBRIC: list[Criterion] = [
    Criterion(
        key="correctness",
        name="Correctness",
        definition=(
            "Are the claims and any reasoning factually and logically correct "
            "for the given input? Judge substance, not confidence or style."
        ),
        weight=0.30,
        anchors={
            1: "Central claim is wrong or self-contradictory; would mislead the user.",
            3: "Mostly correct but with a material error or an unjustified leap.",
            5: "Fully correct; claims and reasoning hold up under scrutiny.",
        },
    ),
    Criterion(
        key="faithfulness",
        name="Faithfulness",
        definition=(
            "Is every claim grounded in the input / provided context (and, when "
            "given, the expected_output)? Penalize hallucinations and invented "
            "specifics not supported by the source."
        ),
        weight=0.25,
        anchors={
            1: "Fabricates facts or contradicts the provided context.",
            3: "Largely grounded but includes 1-2 unsupported specifics.",
            5: "Every claim traceable to the input/context; no hallucinations.",
        },
    ),
    Criterion(
        key="completeness",
        name="Completeness",
        definition=(
            "Does the answer cover what the input actually asked for? Missing a "
            "requested part lowers this; padding does NOT raise it."
        ),
        weight=0.20,
        anchors={
            1: "Ignores most of the request or answers a different question.",
            3: "Covers the main ask but omits a requested sub-part or edge case.",
            5: "Addresses every explicit part of the request, nothing important missing.",
        },
    ),
    Criterion(
        key="instruction_following",
        name="Instruction-following",
        definition=(
            "Does the output obey the system_prompt and any explicit constraints "
            "in the input (format, length limits, language, do/don't rules)?"
        ),
        weight=0.15,
        anchors={
            1: "Violates explicit constraints (wrong format, ignores a hard rule).",
            3: "Follows the spirit but breaks a minor stated constraint.",
            5: "Honors every stated constraint from system_prompt and input.",
        },
    ),
    Criterion(
        key="tone_safety",
        name="Tone & Safety",
        definition=(
            "Is the tone appropriate for the context and the content free of "
            "unsafe, harmful, or policy-violating material?"
        ),
        weight=0.10,
        anchors={
            1: "Harmful/unsafe content, or tone clearly inappropriate for the task.",
            3: "Acceptable but tone slightly off, or a mild safety hedge missing.",
            5: "Appropriate tone; safe and responsible throughout.",
        },
    ),
]

# Length-control instruction injected into every judging prompt (VERBOSITY bias).
LENGTH_CONTROL_INSTRUCTION = (
    "Do NOT reward length, verbosity, or confident phrasing on their own. A short "
    "answer that is correct and complete must score higher than a long answer that "
    "pads, repeats, or adds unsupported detail. Only reward length when the extra "
    "content is requested and adds real, grounded value."
)

# Grounding instruction (SYCOPHANCY / style bias): each score must cite the output.
GROUNDING_INSTRUCTION = (
    "For EVERY criterion, your rationale MUST quote or reference a specific part of "
    "the model_output that justifies the score. A rationale without a concrete "
    "reference is invalid. Do not be swayed by confident tone; verify substance."
)


def total_weight() -> float:
    return sum(c.weight for c in RUBRIC)


def criteria_keys() -> list[str]:
    return [c.key for c in RUBRIC]


def rubric_as_dict() -> list[dict]:
    return [
        {
            "key": c.key,
            "name": c.name,
            "definition": c.definition,
            "weight": c.weight,
            "anchors": c.anchors,
        }
        for c in RUBRIC
    ]


def render_rubric_text() -> str:
    """Human-readable rubric block for the judging prompt."""
    lines = []
    for c in RUBRIC:
        lines.append(f"### {c.name} (key: `{c.key}`, weight: {c.weight})")
        lines.append(c.definition)
        lines.append(
            f"  - 1 = {c.anchors[1]}\n  - 3 = {c.anchors[3]}\n  - 5 = {c.anchors[5]}"
        )
        lines.append("")
    return "\n".join(lines).strip()


def weighted_overall(scores: dict) -> float | None:
    """Weighted mean of per-criterion scores using rubric weights.

    `scores` maps criterion key -> numeric score. Missing criteria are skipped
    and weights renormalized over whatever is present.
    """
    num = 0.0
    den = 0.0
    for c in RUBRIC:
        v = scores.get(c.key)
        if isinstance(v, (int, float)):
            num += float(v) * c.weight
            den += c.weight
    if den == 0:
        return None
    return round(num / den, 3)
