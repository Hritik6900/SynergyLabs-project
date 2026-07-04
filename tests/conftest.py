"""Shared pytest fixtures.

Tests are hermetic: they force the deterministic *local* embedding provider (no
API key, no network) and use throwaway on-disk Chroma stores under pytest's
tmp_path, so they never touch a user's corpus, keys, or persisted data.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _force_local_embeddings(monkeypatch):
    """Every test runs with the deterministic local hash embedder (small dim)."""
    from src.config import settings

    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(settings, "embedding_dim", 64)
    yield


@pytest.fixture
def store(tmp_path):
    """A fresh, isolated VectorStore backed by a temp directory."""
    from src.embed_store import VectorStore

    return VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )
