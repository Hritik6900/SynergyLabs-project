"""Chunker: sizing, overlap, edge cases."""

from __future__ import annotations

import pytest

from src.chunker import chunk_text, count_tokens


def test_empty_and_whitespace_yield_no_chunks():
    assert chunk_text("", chunk_size_tokens=100, overlap_tokens=10) == []
    assert chunk_text("   \n  ", chunk_size_tokens=100, overlap_tokens=10) == []


def test_short_text_is_single_chunk():
    chunks = chunk_text("hello world", chunk_size_tokens=100, overlap_tokens=10)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].token_count == count_tokens("hello world")


def test_full_windows_size_indices_and_full_coverage():
    # Words are not 1 token each, so assert token-level properties, not a word count.
    text = " ".join(f"w{i}" for i in range(400))
    size, overlap = 100, 20
    chunks = chunk_text(text, chunk_size_tokens=size, overlap_tokens=overlap)

    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.token_count <= size for c in chunks)
    # every chunk except the last is a full-size window
    assert all(c.token_count == size for c in chunks[:-1])
    # nothing is dropped: the union of chunk words covers the whole document
    covered = set()
    for c in chunks:
        covered.update(c.text.split())
    assert covered == set(text.split())


def test_overlap_shares_words_and_zero_overlap_does_not():
    # Globally-unique words, so any word shared between consecutive chunks can
    # ONLY come from the overlap region.
    text = " ".join(f"w{i}" for i in range(250))

    with_overlap = chunk_text(text, chunk_size_tokens=100, overlap_tokens=30)
    shared = set(with_overlap[0].text.split()) & set(with_overlap[1].text.split())
    assert len(shared) >= 3

    no_overlap = chunk_text(text, chunk_size_tokens=100, overlap_tokens=0)
    assert set(no_overlap[0].text.split()) & set(no_overlap[1].text.split()) == set()


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size_tokens=0, overlap_tokens=0)
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size_tokens=100, overlap_tokens=100)  # overlap >= size
