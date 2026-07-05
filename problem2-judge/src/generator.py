"""Active generator: produce model_output(s) with the GENERATOR model.

This closes the loop so judge and generator are not just *configurable* but
*independently exercised*: the generator (a different family than the judge)
actually writes the answers, and the judge scores them. Used by `main.py
--generate` to fill in missing outputs and to build A/B pairs from two system
prompts (prompt v1 vs v2).
"""

from __future__ import annotations

from typing import Optional

from .groq_client import GroqClient, get_client, generator_model


def generate_output(
    input_text: str,
    system_prompt: str = "",
    *,
    client: Optional[GroqClient] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """Call the generator model to answer `input_text` under `system_prompt`."""
    client = client or get_client()
    model = model or generator_model()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": input_text})
    out = client.chat(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return (out["content"] or "").strip()


def enrich_case(case: dict, *, client: Optional[GroqClient] = None, model: Optional[str] = None) -> dict:
    """Fill missing model_output (and model_output_b, if system_prompt_b given).

    - If `model_output` is empty, generate it from (input, system_prompt).
    - If `system_prompt_b` is present and `model_output_b` is empty, generate a
      second answer under that prompt -> gives a real prompt-v1-vs-v2 A/B pair.
    Idempotent: existing outputs are left untouched.
    """
    c = dict(case)
    if not str(c.get("model_output", "")).strip():
        c["model_output"] = generate_output(
            c.get("input", ""), c.get("system_prompt", ""), client=client, model=model
        )
        c["_generated_output"] = True
    if str(c.get("system_prompt_b", "")).strip() and not str(c.get("model_output_b", "")).strip():
        c["model_output_b"] = generate_output(
            c.get("input", ""), c.get("system_prompt_b", ""), client=client, model=model
        )
        c["_generated_output_b"] = True
    return c
