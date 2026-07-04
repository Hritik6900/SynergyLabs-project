"""The no-relevant-context guard: refuse instead of hallucinating (no LLM call)."""

from __future__ import annotations

from src.generate import NO_CONTEXT_MESSAGE, generate_answer


def test_returns_no_context_when_nothing_clears_threshold(tmp_path, store):
    store.add(
        ids=["a"],
        texts=["an unrelated note about gardening tomatoes"],
        metadatas=[{"source": "garden.md", "section": "full", "chunk_index": 0}],
    )
    # Threshold of 1.01 is impossible to clear -> must refuse without calling an LLM.
    result = generate_answer(
        "What is the capital of France?",
        k=3,
        threshold=1.01,
        store=store,
    )
    assert result.no_relevant_context is True
    assert result.answer == NO_CONTEXT_MESSAGE
    assert result.cited_chunks == []
    assert result.chunk_count == 0
    assert result.token_usage["total_tokens"] == 0


def test_empty_store_returns_no_context(store):
    result = generate_answer("anything", k=3, threshold=0.0, store=store)
    assert result.no_relevant_context is True
    assert result.answer == NO_CONTEXT_MESSAGE
