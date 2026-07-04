"""Grounded answer generation with citations.

Flow:
  1. Retrieve top-k chunks for the question (optionally metadata-filtered).
  2. Keep only chunks that clear the similarity threshold.
  3. If none clear it -> return "no relevant context found" WITHOUT calling the
     LLM (no hallucination, no spend).
  4. Otherwise build a grounded prompt that numbers the chunks and instructs the
     model to answer only from them and to cite the chunks it uses inline as
     [source #chunk_index]. Return the answer, the cited chunks, and token usage.

The LLM provider is configurable: OpenAI (gpt-4o-mini) or Anthropic (claude-haiku).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import settings
from .embed_store import RetrievedChunk, VectorStore
from .llm_client import chat_complete
from .retrieve import relevant_hits, retrieve

NO_CONTEXT_MESSAGE = "no relevant context found"

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer ONLY using the "
    "numbered context chunks provided by the user. If the answer is not "
    "contained in the chunks, say you don't have enough information. Do not use "
    "outside knowledge. After each claim, cite the chunk(s) you used inline using "
    "the format [source #chunk_index] exactly as labelled in the context. Keep "
    "the answer concise."
)


@dataclass
class GenerationResult:
    answer: str
    cited_chunks: list[dict]
    chunk_count: int
    token_usage: dict
    retrieval_latency_ms: float
    generation_latency_ms: float
    no_relevant_context: bool
    all_retrieved: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "cited_chunks": self.cited_chunks,
            "chunk_count": self.chunk_count,
            "token_usage": self.token_usage,
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "generation_latency_ms": round(self.generation_latency_ms, 2),
            "no_relevant_context": self.no_relevant_context,
        }


def _format_context(hits: list[RetrievedChunk]) -> str:
    cap = settings.max_context_chars_per_chunk
    blocks = []
    for h in hits:
        label = f"[{h.metadata.get('source')} #{h.metadata.get('chunk_index')}]"
        text = h.text
        if cap and len(text) > cap:
            text = text[:cap] + " …[truncated]"
        blocks.append(f"{label}\n{text}")
    return "\n\n".join(blocks)


def _call_llm(context: str, question: str) -> tuple[str, dict]:
    """Grounded generation via the configured provider (openai/groq/anthropic)."""
    user_prompt = f"Context chunks:\n\n{context}\n\nQuestion: {question}"
    return chat_complete(SYSTEM_PROMPT, user_prompt)


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def generate_answer(
    question: str,
    k: int | None = None,
    source_filter: str | None = None,
    where: dict | None = None,
    threshold: float | None = None,
    store: VectorStore | None = None,
) -> GenerationResult:
    """Retrieve, gate on the threshold, and generate a grounded, cited answer."""
    store = store or VectorStore()

    t0 = time.perf_counter()
    hits = retrieve(question, k=k, source_filter=source_filter, where=where, store=store)
    retrieval_ms = (time.perf_counter() - t0) * 1000.0

    all_retrieved = [h.as_dict() for h in hits]
    kept = relevant_hits(hits, threshold=threshold)

    # No chunk cleared the threshold -> refuse instead of hallucinating.
    if not kept:
        return GenerationResult(
            answer=NO_CONTEXT_MESSAGE,
            cited_chunks=[],
            chunk_count=0,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=0.0,
            no_relevant_context=True,
            all_retrieved=all_retrieved,
        )

    context = _format_context(kept)
    t1 = time.perf_counter()
    answer, usage = _call_llm(context, question)
    generation_ms = (time.perf_counter() - t1) * 1000.0

    return GenerationResult(
        answer=answer,
        cited_chunks=[h.as_dict() for h in kept],
        chunk_count=len(kept),
        token_usage=usage,
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=generation_ms,
        no_relevant_context=False,
        all_retrieved=all_retrieved,
    )
