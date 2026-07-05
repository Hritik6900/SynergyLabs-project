"""Auditable/replayable logging: every judge prompt + raw response to logs/.

One JSONL line per judge call in logs/judge_calls.jsonl, plus a human-readable
per-call dump under logs/calls/<id>.json. Parse failures go to logs/parse_failures.jsonl.

We deliberately keep a monotonic counter instead of timestamps for the id so the
module stays import-safe in environments where wall-clock is unavailable; a
timestamp field is still added best-effort.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

LOG_DIR = os.getenv("JUDGE_LOG_DIR", os.path.join(os.getcwd(), "logs"))
CALLS_DIR = os.path.join(LOG_DIR, "calls")
_CALLS_JSONL = os.path.join(LOG_DIR, "judge_calls.jsonl")
_PARSE_FAIL_JSONL = os.path.join(LOG_DIR, "parse_failures.jsonl")

_lock = threading.Lock()
_counter = 0


def _ensure_dirs() -> None:
    os.makedirs(CALLS_DIR, exist_ok=True)


def _now_iso() -> str:
    try:
        import datetime

        return datetime.datetime.now().isoformat(timespec="seconds")
    except Exception:
        return ""


def next_id(prefix: str = "call") -> str:
    global _counter
    with _lock:
        _counter += 1
        return f"{prefix}-{_counter:05d}"


def log_judge_call(
    *,
    call_id: str,
    mode: str,
    model: str,
    messages: list[dict],
    raw_response: str,
    parsed: Any | None,
    usage: Any | None,
    meta: dict | None = None,
) -> None:
    """Persist a full judge interaction for audit/replay."""
    _ensure_dirs()
    record = {
        "id": call_id,
        "ts": _now_iso(),
        "mode": mode,
        "model": model,
        "messages": messages,
        "raw_response": raw_response,
        "parsed": parsed,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        if usage is not None
        else None,
        "meta": meta or {},
    }
    with _lock:
        with open(_CALLS_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(os.path.join(CALLS_DIR, f"{call_id}.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def log_parse_failure(*, call_id: str, raw: str, error: str, stage: str) -> None:
    _ensure_dirs()
    rec = {
        "id": call_id,
        "ts": _now_iso(),
        "stage": stage,  # "primary" | "salvage" | "repair"
        "error": error,
        "raw": raw,
    }
    with _lock:
        with open(_PARSE_FAIL_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
