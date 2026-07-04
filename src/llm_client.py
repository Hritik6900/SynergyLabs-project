"""Single entry point for chat completion across LLM providers.

Supported providers (env ``LLM_PROVIDER``):
  * ``openai``    -> OpenAI chat completions (gpt-4o-mini).
  * ``groq``      -> Groq (OpenAI-compatible chat API) via the OpenAI SDK with a
                     custom base_url. Groq is chat-only; it has no embeddings API.
  * ``anthropic`` -> Anthropic Messages API (claude-haiku).

Both generation (generate.py) and the LLM-as-judge (eval/answer_metrics.py) go
through :func:`chat_complete`, so adding a provider is a one-place change and the
two stay consistent.
"""

from __future__ import annotations

from .config import settings


def _openai_compatible(base_url: str | None, api_key: str | None, provider: str):
    from openai import OpenAI

    if not api_key:
        raise RuntimeError(
            f"LLM_PROVIDER={provider} but its API key is not set "
            f"({'GROQ_API_KEY' if provider == 'groq' else 'OPENAI_API_KEY'})."
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def chat_complete(
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> tuple[str, dict]:
    """Run one system+user chat turn and return (text, token_usage).

    token_usage is normalized to {"prompt_tokens", "completion_tokens",
    "total_tokens"} across all providers.

    ``json_mode`` requests a JSON object response where the provider supports it
    (OpenAI/Groq); callers should still defensively parse, and for Anthropic the
    prompt must itself instruct JSON (we rely on the caller's parsing fallback).
    """
    provider = settings.llm_provider

    if provider in ("openai", "groq"):
        if provider == "openai":
            client = _openai_compatible(None, settings.openai_api_key, "openai")
            model = settings.openai_llm_model
        else:
            client = _openai_compatible(settings.groq_base_url, settings.groq_api_key, "groq")
            model = settings.groq_llm_model

        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            **kwargs,
        )
        text = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
        return text.strip(), usage

    if provider == "anthropic":
        import anthropic

        if not settings.anthropic_api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=settings.anthropic_llm_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
        }
        return text.strip(), usage

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def llm_available() -> bool:
    """True if the configured provider has the credentials it needs."""
    return {
        "openai": bool(settings.openai_api_key),
        "groq": bool(settings.groq_api_key),
        "anthropic": bool(settings.anthropic_api_key),
    }.get(settings.llm_provider, False)
