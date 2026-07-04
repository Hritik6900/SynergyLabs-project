"""FAISS-backed vector store — a second, alternative backend to ChromaDB.

Purpose: benchmarking bonus. FAISS is an in-memory ANN library (no server, no
persistence layer of its own). Here we use an exact `IndexFlatIP` over
L2-normalized vectors, so inner product == cosine similarity and results are
*exact* (ground truth). That makes it a useful yardstick: we can measure how much
recall ChromaDB's approximate HNSW index gives up versus exact search, and compare
per-query latency.

It mirrors the subset of :class:`src.embed_store.VectorStore` used for retrieval
(``add`` / ``query`` / ``count``) and returns the same :class:`RetrievedChunk`
objects, so it is a drop-in for read paths.
"""

from __future__ import annotations

import numpy as np

from .embed_store import RetrievedChunk, embed_texts


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class FaissStore:
    """In-memory exact cosine store backed by faiss.IndexFlatIP."""

    def __init__(self, dim: int | None = None):
        import faiss

        self._faiss = faiss
        self._dim = dim
        self._index = None  # created on first add once we know the dimension
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._metas: list[dict] = []

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self._dim = dim
            self._index = self._faiss.IndexFlatIP(dim)  # inner product on unit vecs = cosine

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        if not ids:
            return
        vecs = _normalize(np.asarray(embed_texts(texts), dtype="float32"))
        self._ensure_index(vecs.shape[1])
        self._index.add(vecs)
        self._ids.extend(ids)
        self._docs.extend(texts)
        self._metas.extend(metadatas)

    def add_precomputed(
        self, ids: list[str], embeddings: list[list[float]], texts: list[str], metadatas: list[dict]
    ) -> None:
        """Add with embeddings already computed (so a benchmark embeds only once)."""
        if not ids:
            return
        vecs = _normalize(np.asarray(embeddings, dtype="float32"))
        self._ensure_index(vecs.shape[1])
        self._index.add(vecs)
        self._ids.extend(ids)
        self._docs.extend(texts)
        self._metas.extend(metadatas)

    def count(self) -> int:
        return len(self._ids)

    def query(
        self,
        query_text: str,
        k: int,
        where: dict | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        """Top-k exact cosine search, with an optional equality metadata filter.

        ``where`` supports simple field==value equality (e.g. {"source": "x.md"}),
        applied by over-fetching then filtering — enough to mirror the Chroma path.
        """
        if self.count() == 0:
            return []
        emb = query_embedding if query_embedding is not None else embed_texts([query_text])[0]
        q = _normalize(np.asarray([emb], dtype="float32"))
        # Over-fetch when filtering so the post-filter still yields up to k hits.
        fetch = k if not where else min(self.count(), k * 10)
        scores, idxs = self._index.search(q, fetch)

        hits: list[RetrievedChunk] = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            meta = self._metas[i]
            if where and any(meta.get(f) != v for f, v in where.items()):
                continue
            hits.append(
                RetrievedChunk(
                    id=self._ids[i],
                    text=self._docs[i],
                    metadata=meta,
                    similarity=float(score),  # already cosine for unit vectors
                )
            )
            if len(hits) >= k:
                break
        return hits
