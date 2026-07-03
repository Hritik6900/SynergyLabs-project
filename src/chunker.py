"""Token-based chunking with configurable size and overlap.

Why token-based (not character- or word-based): embedding models operate on
tokens and have token input limits, so chunking on tokens gives predictable,
model-aligned chunk sizes. We use tiktoken's ``cl100k_base`` encoding, which is
the tokenizer used by OpenAI's embedding + chat models.

Defaults (see config): 500 tokens/chunk, 50 tokens overlap.
  - ~500 tokens is a coherent passage (~350-400 words) that keeps retrieval
    granular while staying well under the embedding input limit.
  - 50-token (10%) overlap preserves continuity so a fact that straddles a chunk
    boundary is still fully present in at least one chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

# cl100k_base is shared by text-embedding-3-* and gpt-4o-mini, so token counts
# here line up with what the models actually see.
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """A single chunk of text plus its position within its source document."""

    text: str
    chunk_index: int  # 0-based index of this chunk within its source document
    token_count: int


def count_tokens(text: str) -> int:
    """Return the number of tokens in ``text`` (cl100k_base)."""
    return len(_ENCODING.encode(text))


def chunk_text(
    text: str,
    *,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Split ``text`` into overlapping token windows.

    Args:
        text: the full document text.
        chunk_size_tokens: target tokens per chunk (must be > 0).
        overlap_tokens: tokens shared between consecutive chunks
            (must be >= 0 and < chunk_size_tokens).

    Returns:
        Ordered list of :class:`Chunk`. Empty/whitespace-only input yields [].
    """
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be > 0")
    if overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must satisfy 0 <= overlap < chunk_size_tokens")

    if not text or not text.strip():
        return []

    token_ids = _ENCODING.encode(text)
    total = len(token_ids)
    step = chunk_size_tokens - overlap_tokens  # how far the window advances

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < total:
        end = min(start + chunk_size_tokens, total)
        window = token_ids[start:end]
        piece = _ENCODING.decode(window).strip()
        if piece:  # skip windows that decode to only whitespace
            chunks.append(Chunk(text=piece, chunk_index=index, token_count=len(window)))
            index += 1
        if end == total:
            break
        start += step

    return chunks
