"""Single entry point for every Groq call in the pipeline.

Why one wrapper:
  * centralizes GROQ_API_KEY handling (never re-read os.environ elsewhere),
  * makes swapping models/providers a one-file change,
  * gives us ONE place to add retry-with-backoff (Groq's free/dev tier
    returns 429s) and ONE place to accumulate token usage for cost tracking.

All calls go through Groq's OpenAI-compatible endpoint using the `openai` SDK.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError

load_dotenv()

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


# --------------------------------------------------------------------------- #
# Usage / cost tracking
# --------------------------------------------------------------------------- #
# Groq per-token pricing (USD per 1M tokens), keep in sync with
# https://groq.com/pricing . Used only for an *estimate* in reports; the
# authoritative token counts come from the API `usage` field on every call.
GROQ_PRICING_USD_PER_1M = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.75},
    "openai/gpt-oss-20b": {"input": 0.10, "output": 0.50},
    "qwen/qwen3.6-27b": {"input": 0.20, "output": 0.80},
    "qwen/qwen3-32b": {"input": 0.29, "output": 0.59},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
}


@dataclass
class UsageTracker:
    """Running count of judge (and any) token usage + call count."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    per_model: dict = field(default_factory=dict)

    def record(self, model: str, usage: Any) -> None:
        self.calls += 1
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        tt = int(getattr(usage, "total_tokens", 0) or (pt + ct))
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.total_tokens += tt
        m = self.per_model.setdefault(
            model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        )
        m["calls"] += 1
        m["prompt_tokens"] += pt
        m["completion_tokens"] += ct

    def estimated_cost_usd(self) -> float:
        total = 0.0
        for model, m in self.per_model.items():
            price = GROQ_PRICING_USD_PER_1M.get(model)
            if not price:
                continue
            total += m["prompt_tokens"] / 1_000_000 * price["input"]
            total += m["completion_tokens"] / 1_000_000 * price["output"]
        return round(total, 6)

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd(),
            "per_model": self.per_model,
        }


# One shared tracker for the whole process (judge + generator + repair calls).
USAGE = UsageTracker()


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class GroqClient:
    """Thin retrying wrapper around the OpenAI SDK pointed at Groq."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: Optional[int] = None,
        tracker: Optional[UsageTracker] = None,
    ):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self.base_url = base_url or os.getenv("GROQ_BASE_URL", DEFAULT_BASE_URL)
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        self.max_retries = (
            max_retries
            if max_retries is not None
            else int(os.getenv("GROQ_MAX_RETRIES", "6"))
        )
        self.tracker = tracker or USAGE

    def chat(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        force_json: bool = False,
        extra: Optional[dict] = None,
    ) -> dict:
        """Call chat.completions with retry/backoff.

        Returns {"content": str, "usage": <usage obj>, "raw": <response>}.

        If force_json is True we first try response_format={"type":"json_object"}.
        Not every Groq model accepts it; on a 400 that mentions json/response_format
        we transparently retry WITHOUT it (the caller's prompt still instructs JSON,
        and parser.py handles the rest). This is the "graceful either way" path.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra:
            kwargs.update(extra)
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            return self._with_retry(kwargs, model)
        except APIError as e:
            msg = str(getattr(e, "message", e)).lower()
            # A model may reject reasoning_effort (non-reasoning models) -> drop it and retry.
            if "reasoning_effort" in kwargs.get("extra_body", {}) or "reasoning_effort" in kwargs:
                if "reasoning_effort" in msg or "unrecognized" in msg or "unknown" in msg:
                    kwargs.pop("reasoning_effort", None)
                    kwargs.get("extra_body", {}).pop("reasoning_effort", None)
                    return self._with_retry(kwargs, model)
            if force_json and (
                "json" in msg
                or "response_format" in msg
                or "not supported" in msg
                or getattr(e, "status_code", None) == 400
            ):
                # Model doesn't support JSON mode -> fall back to prompt-only JSON.
                kwargs.pop("response_format", None)
                return self._with_retry(kwargs, model)
            raise

    def _pace(self) -> None:
        """Optional steady spacing between calls to stay under a per-minute token
        limit. Bursting then eating a 30s backoff is far slower than pacing, so on
        a throttled tier set GROQ_MIN_CALL_INTERVAL (seconds) to smooth it out.
        """
        interval = float(os.getenv("GROQ_MIN_CALL_INTERVAL", "0") or 0)
        if interval <= 0:
            return
        now = time.time()
        wait = interval - (now - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.time()

    _last_call_ts: float = 0.0

    def _with_retry(self, kwargs: dict, model: str) -> dict:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                self._pace()
                resp = self.client.chat.completions.create(**kwargs)
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    self.tracker.record(model, usage)
                content = resp.choices[0].message.content or ""
                return {"content": content, "usage": usage, "raw": resp}
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_exc = e
                sleep_s = self._backoff_seconds(e, attempt)
                time.sleep(sleep_s)
            except APIError as e:
                # 5xx are retryable; 4xx (other than 429) are not.
                status = getattr(e, "status_code", None)
                if status is not None and 500 <= status < 600:
                    last_exc = e
                    time.sleep(self._backoff_seconds(e, attempt))
                else:
                    raise
        raise RuntimeError(
            f"Groq call failed after {self.max_retries} retries: {last_exc}"
        )

    def _backoff_seconds(self, exc: Exception, attempt: int) -> float:
        """Exponential backoff with jitter; honor Retry-After if Groq sends it."""
        retry_after = None
        resp = getattr(exc, "response", None)
        if resp is not None:
            hdr = getattr(resp, "headers", {}) or {}
            ra = hdr.get("retry-after") or hdr.get("Retry-After")
            if ra:
                try:
                    retry_after = float(ra)
                except (TypeError, ValueError):
                    retry_after = None
        if retry_after is not None:
            return retry_after + random.uniform(0, 0.5)
        return min(2.0 ** attempt, 30.0) + random.uniform(0, 0.75)


# Convenience singleton (lazy) so callers can `from groq_client import get_client`.
_CLIENT: Optional[GroqClient] = None


def get_client() -> GroqClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = GroqClient()
    return _CLIENT


def judge_model() -> str:
    return os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")


def generator_model() -> str:
    return os.getenv("GROQ_GENERATOR_MODEL", "qwen/qwen3.6-27b")


if __name__ == "__main__":
    # Tiny connectivity smoke test: `python -m src.groq_client`
    c = get_client()
    out = c.chat(
        model=judge_model(),
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=8,
    )
    print("judge_model  :", judge_model())
    print("generator    :", generator_model())
    print("response     :", out["content"].strip())
    print("usage        :", json.dumps(USAGE.summary(), indent=2))
