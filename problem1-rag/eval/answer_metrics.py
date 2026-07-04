"""Answer-quality metrics.

Two families:

1. LLM-as-judge (needs an LLM key + provider configured via env):
     * faithfulness / groundedness (1-5): is every claim in the answer supported
       by the retrieved context chunks?
     * answer relevance (1-5): does the answer actually address the question?
   The judge is asked for STRUCTURED JSON with a 1-5 score and a rationale for
   each. We parse that JSON; on any failure we return a null score with the raw
   text so the run is never silently corrupted.

2. Lexical overlap vs a gold answer (no LLM needed), computed only when the
   question provides a ``gold_answer``:
     * Exact Match (EM): normalized strings equal.
     * token F1: SQuAD-style token overlap F1.

The judge reuses the same provider/model as generation (config.py), so the eval
honours LLM_PROVIDER=openai|anthropic.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter

# Import from the app package. Works whether run as a module or a script.
try:
    from src.llm_client import chat_complete
except ImportError:  # pragma: no cover - fallback for direct execution
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.llm_client import chat_complete


# --------------------------------------------------------------------------- #
# EM / F1 (lexical, no LLM)                                                    #
# --------------------------------------------------------------------------- #
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles/extra ws."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# LLM-as-judge                                                                 #
# --------------------------------------------------------------------------- #
_JUDGE_SYSTEM = (
    "You are a strict evaluator of a RAG system's answers. You are given a "
    "question, the context chunks that were retrieved, and the system's answer. "
    "Score two dimensions on an integer scale of 1 to 5:\n"
    "  * faithfulness: 5 = every claim is directly supported by the context; "
    "1 = the answer contradicts or invents facts not in the context.\n"
    "  * answer_relevance: 5 = fully and directly answers the question; "
    "1 = off-topic or non-responsive.\n"
    "Respond with ONLY a JSON object of the form:\n"
    '{"faithfulness": <1-5>, "faithfulness_rationale": "<one sentence>", '
    '"answer_relevance": <1-5>, "answer_relevance_rationale": "<one sentence>"}'
)


def _judge_prompt(question: str, context: str, answer: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Retrieved context chunks:\n{context}\n\n"
        f"System answer:\n{answer}\n\n"
        "Return only the JSON object."
    )


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def llm_judge(question: str, context: str, answer: str) -> dict:
    """Score faithfulness + relevance via the configured LLM (openai/groq/anthropic).

    Returns a dict with integer 1-5 scores (or None on parse failure) plus
    rationales."""
    raw, _usage = chat_complete(
        _JUDGE_SYSTEM,
        _judge_prompt(question, context, answer),
        json_mode=True,
        max_tokens=512,
    )
    parsed = _extract_json(raw)
    if not parsed:
        return {
            "faithfulness": None,
            "faithfulness_rationale": "judge output could not be parsed",
            "answer_relevance": None,
            "answer_relevance_rationale": "judge output could not be parsed",
            "raw": raw,
        }

    def _coerce(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    return {
        "faithfulness": _coerce(parsed.get("faithfulness")),
        "faithfulness_rationale": parsed.get("faithfulness_rationale", ""),
        "answer_relevance": _coerce(parsed.get("answer_relevance")),
        "answer_relevance_rationale": parsed.get("answer_relevance_rationale", ""),
    }
