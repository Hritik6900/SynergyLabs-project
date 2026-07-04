"""FAISS second-store backend: agrees with exact search and mirrors the interface."""

from __future__ import annotations

from src.embed_store import VectorStore, embed_texts
from src.faiss_store import FaissStore


def test_faiss_returns_correct_top_result():
    fs = FaissStore()
    fs.add(
        ids=["a", "b", "c"],
        texts=[
            "cosine similarity measures the angle between vectors",
            "bananas are a yellow fruit",
            "the weather is sunny",
        ],
        metadatas=[{"source": "x"}, {"source": "y"}, {"source": "z"}],
    )
    assert fs.count() == 3
    hits = fs.query("angle between vectors cosine", k=2)
    assert hits[0].id == "a"
    assert len(hits) == 2


def test_faiss_metadata_filter():
    fs = FaissStore()
    fs.add(
        ids=["a", "b"],
        texts=["shared vocab here", "shared vocab here too"],
        metadatas=[{"source": "keep"}, {"source": "drop"}],
    )
    hits = fs.query("shared vocab", k=5, where={"source": "keep"})
    assert hits and all(h.metadata["source"] == "keep" for h in hits)


def test_faiss_and_chroma_agree_on_top_k(tmp_path):
    ids = [f"id{i}" for i in range(6)]
    texts = [
        "vector databases store embeddings for similarity search",
        "chunking splits documents before embedding",
        "cosine similarity compares vector direction",
        "an ANN index trades recall for speed",
        "metadata filtering restricts results by field",
        "a grounded answer cites its sources",
    ]
    metas = [{"source": f"s{i}"} for i in range(6)]
    embeddings = embed_texts(texts)

    chroma = VectorStore(persist_dir=str(tmp_path / "c"), collection_name="bench")
    chroma.reset()
    chroma.add(ids=ids, texts=texts, metadatas=metas, embeddings=embeddings)

    fs = FaissStore()
    fs.add_precomputed(ids=ids, embeddings=embeddings, texts=texts, metadatas=metas)

    q = "how does an approximate nearest neighbor index work"
    qemb = embed_texts([q])[0]
    c_ids = [h.id for h in chroma.query(q, k=3, query_embedding=qemb)]
    f_ids = [h.id for h in fs.query(q, k=3, query_embedding=qemb)]
    # Chroma (approximate HNSW) and FAISS (exact) are built from identical vectors,
    # so their top-k should largely agree. With the weak local hash embedding a
    # boundary rank can differ, so require a strong-majority overlap rather than
    # exact equality (the real ST-embedding benchmark shows full 1.0 agreement).
    assert len(set(c_ids) & set(f_ids)) >= 2
    assert len(c_ids) == 3 and len(f_ids) == 3
