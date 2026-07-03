"""Append-only per-query logging to a local JSONL file.

Each query writes one JSON line capturing latency, chunk count, and token usage,
so operational cost/latency can be audited after the fact.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

from .config import settings

_lock = threading.Lock()


def log_query(record: dict) -> None:
    """Append one query record as a JSON line to the configured log file."""
    path = settings.query_log_path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    line = json.dumps(record, ensure_ascii=False)
    with _lock:  # serialize writes across threads/requests
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
