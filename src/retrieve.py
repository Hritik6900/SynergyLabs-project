"""Retrieval: top-k nearest chunks with an optional metadata filter.

The similarity threshold is *not* applied here (retrieval always returns the
top-k nearest hits). The threshold is a separate policy used by generation to
decide whether any hit is relevant enough to answer from — see generate.py.
"""

from __future__ import annotations

from .config import settings
from .embed_store import RetrievedChunk, VectorStore


def retrieve(
    question: str,
    k: int | None = None,
    source_filter: str | None = None,
    where: dict | None = None,
    store: VectorStore | None = None,
) -> list[RetrievedChunk]:
    """Return the top-``k`` chunks most similar to ``question``.

    Args:
        question: the natural-language query.
        k: number of chunks to return (defaults to settings.top_k).
        source_filter: convenience filter restricting to chunks whose ``source``
            metadata equals this filename.
        where: a raw Chroma metadata filter (takes precedence over
            ``source_filter``), e.g. {"source": "notes.md"} or
            {"chunk_index": {"$gte": 3}}.
        store: an existing VectorStore (a new one is created if omitted).
    """
    k = k if k is not None else settings.top_k
    store = store or VectorStore()
    if where is None and source_filter:
        where = {"source": source_filter}
    return store.query(question, k=k, where=where)


def max_similarity(hits: list[RetrievedChunk]) -> float:
    """Best (highest) similarity among hits, or -inf if there are none."""
    return max((h.similarity for h in hits), default=float("-inf"))


def relevant_hits(
    hits: list[RetrievedChunk],
    threshold: float | None = None,
) -> list[RetrievedChunk]:
    """Subset of ``hits`` whose similarity clears the relevance threshold."""
    threshold = threshold if threshold is not None else settings.similarity_threshold
    return [h for h in hits if h.similarity >= threshold]
