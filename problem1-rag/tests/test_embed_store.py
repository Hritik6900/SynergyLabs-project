"""Embeddings + ChromaDB store behaviour (local provider)."""

from __future__ import annotations

from src.embed_store import embed_texts


def test_local_embedding_is_deterministic_and_sized():
    v1 = embed_texts(["idempotent ingestion"])[0]
    v2 = embed_texts(["idempotent ingestion"])[0]
    v3 = embed_texts(["something entirely different"])[0]
    assert v1 == v2                 # deterministic
    assert v1 != v3                 # content-sensitive
    assert len(v1) == 64            # dim forced by the test fixture


def test_existing_ids_roundtrip(store):
    store.add(
        ids=["id1", "id2"],
        texts=["alpha content", "beta content"],
        metadatas=[{"source": "a"}, {"source": "b"}],
    )
    found = store.existing_ids(["id1", "id2", "missing"])
    assert found == {"id1", "id2"}
    assert store.count() == 2


def test_query_returns_ranked_hits(store):
    store.add(
        ids=["a", "b", "c"],
        texts=[
            "cosine similarity measures the angle between vectors",
            "bananas are a yellow fruit",
            "the weather today is sunny and warm",
        ],
        metadatas=[{"source": "x"}, {"source": "y"}, {"source": "z"}],
    )
    hits = store.query("angle between vectors cosine", k=3)
    assert hits
    assert hits[0].id == "a"                       # most lexically similar
    # similarities are sorted descending
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)


def test_metadata_filter_restricts_source(store):
    store.add(
        ids=["a", "b"],
        texts=["shared vocabulary here", "shared vocabulary here too"],
        metadatas=[{"source": "keep.md"}, {"source": "drop.md"}],
    )
    hits = store.query("shared vocabulary", k=5, where={"source": "keep.md"})
    assert hits and all(h.metadata["source"] == "keep.md" for h in hits)
