"""Robust JSON parsing for judge responses.

Open-weight models on Groq emit malformed JSON far more often than closed
frontier models: markdown fences, leading prose ("Sure! Here is..."), trailing
commentary, single quotes, trailing commas, smart quotes. This module tries a
ladder of increasingly aggressive strategies before giving up:

  1. json.loads on the raw string
  2. strip markdown ``` fences, retry
  3. extract the first balanced {...} object, retry
  4. light repairs (trailing commas, single->double quotes, smart quotes), retry
  5. LLM repair prompt: ask the judge model to reformat its own text as valid JSON

Every failure is logged. parse_verdict() returns (obj, info) where info records
which strategy succeeded so we can measure malformed-JSON rates in reports.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from . import logging_utils

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _try_json(s: str) -> tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(s), None
    except json.JSONDecodeError as e:
        return None, str(e)


def _strip_fences(s: str) -> str:
    m = _FENCE_RE.search(s)
    if m:
        return m.group(1).strip()
    # also handle a lone leading ```json with no closing fence
    return re.sub(r"^```(?:json)?", "", s.strip(), flags=re.IGNORECASE).strip("`").strip()


def _extract_first_object(s: str) -> Optional[str]:
    """Return the first balanced {...} block (brace-aware, string-aware)."""
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _light_repairs(s: str) -> str:
    t = s
    # normalize smart quotes
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("‘", "'").replace("’", "'")
    # remove trailing commas before } or ]
    t = re.sub(r",\s*([}\]])", r"\1", t)
    # convert single-quoted keys/strings to double where it looks safe
    # (only when there are no double quotes already doing the job)
    if '"' not in t and "'" in t:
        t = t.replace("'", '"')
    return t


def parse_json_best_effort(raw: str) -> tuple[Optional[Any], str]:
    """Try strategies 1-4 (no LLM). Returns (obj_or_None, strategy_name)."""
    if raw is None:
        return None, "empty"

    obj, _ = _try_json(raw)
    if obj is not None:
        return obj, "direct"

    stripped = _strip_fences(raw)
    obj, _ = _try_json(stripped)
    if obj is not None:
        return obj, "fenced"

    block = _extract_first_object(stripped) or _extract_first_object(raw)
    if block:
        obj, _ = _try_json(block)
        if obj is not None:
            return obj, "extracted"
        obj, _ = _try_json(_light_repairs(block))
        if obj is not None:
            return obj, "repaired"

    obj, _ = _try_json(_light_repairs(stripped))
    if obj is not None:
        return obj, "repaired"

    return None, "failed"


REPAIR_SYSTEM = (
    "You are a JSON repair tool. The user gives you text that was supposed to be "
    "a single JSON object but is malformed or wrapped in prose. Output ONLY the "
    "corrected, valid JSON object. No markdown, no commentary, no code fences. "
    "Preserve all information; do not invent fields or values."
)


def parse_verdict(
    raw: str,
    *,
    call_id: str,
    client=None,
    repair_model: Optional[str] = None,
    allow_llm_repair: bool = True,
) -> tuple[Optional[Any], dict]:
    """Full ladder including optional LLM repair.

    Returns (obj_or_None, info) where info = {
        "strategy": <how it parsed>, "llm_repair_used": bool, "ok": bool
    }.
    """
    obj, strategy = parse_json_best_effort(raw)
    if obj is not None:
        return obj, {"strategy": strategy, "llm_repair_used": False, "ok": True}

    logging_utils.log_parse_failure(
        call_id=call_id, raw=raw, error="best-effort parse failed", stage="salvage"
    )

    if not (allow_llm_repair and client is not None and repair_model):
        return None, {"strategy": "failed", "llm_repair_used": False, "ok": False}

    # Strategy 5: ask the model to repair its own output.
    try:
        out = client.chat(
            model=repair_model,
            messages=[
                {"role": "system", "content": REPAIR_SYSTEM},
                {"role": "user", "content": raw},
            ],
            temperature=0.0,
            max_tokens=1200,
            force_json=True,
        )
        repaired_raw = out["content"]
        obj, strategy = parse_json_best_effort(repaired_raw)
        if obj is not None:
            return obj, {
                "strategy": f"llm_repair:{strategy}",
                "llm_repair_used": True,
                "ok": True,
            }
        logging_utils.log_parse_failure(
            call_id=call_id,
            raw=repaired_raw,
            error="llm repair still unparseable",
            stage="repair",
        )
    except Exception as e:  # network/API failure during repair
        logging_utils.log_parse_failure(
            call_id=call_id, raw=raw, error=f"repair call failed: {e}", stage="repair"
        )

    return None, {"strategy": "failed", "llm_repair_used": True, "ok": False}
