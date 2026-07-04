"""Embeddings and the ChromaDB-backed vector store.

Embedding providers (env ``EMBEDDING_PROVIDER``):
  * ``openai``  -> text-embedding-3-small, 1536 dimensions (needs OPENAI_API_KEY).
  * ``local``   -> a deterministic hashing-trick embedding, no API key or network.
                   Same text always maps to the same vector, and texts sharing
                   words get positive cosine similarity, so retrieval, idempotency,
                   and the IR metrics can be exercised end-to-end offline with zero
                   spend. Not as good as a learned model — for tests/CI, not prod.

The store uses a Chroma ``PersistentClient`` with cosine space, so nothing runs
as an always-on server; data lives on the local filesystem.
"""

from __future__ import annotations

import hashlib
import logging
import math

import chromadb
from chromadb.config import Settings as ChromaClientSettings

from .config import settings

# ChromaDB 0.5.x emits noisy "Failed to send telemetry event" lines even with
# telemetry disabled (a known upstream bug). Silence that logger; we also disable
# telemetry via ChromaClientSettings below.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# --------------------------------------------------------------------------- #
# Embeddings                                                                   #
# --------------------------------------------------------------------------- #
_openai_client = None  # lazily constructed singleton


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Set it in .env, or use EMBEDDING_PROVIDER=local for offline runs."
            )
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _embed_openai(texts: list[str]) -> list[list[float]]:
    client = _get_openai_client()
    resp = client.embeddings.create(model=settings.embedding_model, input=texts)
    # API preserves input order.
    return [d.embedding for d in resp.data]


_st_model = None  # lazily constructed sentence-transformers model


def _get_st_model():
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "EMBEDDING_PROVIDER=sentence-transformers requires the "
                "'sentence-transformers' package. Install it with:\n"
                "  pip install sentence-transformers"
            ) from exc
        _st_model = SentenceTransformer(settings.st_embedding_model)
    return _st_model


def _embed_sentence_transformers(texts: list[str]) -> list[list[float]]:
    """Local CPU embeddings via sentence-transformers (free, no API key)."""
    model = _get_st_model()
    # normalize so cosine space behaves well; returns a numpy array.
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.tolist()


def _embed_local(texts: list[str]) -> list[list[float]]:
    """Deterministic hashing-trick embedding.

    For each whitespace token we derive one or more (index, sign) pairs from a
    sha1 hash and accumulate into a fixed-size vector, then L2-normalize. Shared
    vocabulary -> positive cosine similarity. Fully deterministic.
    """
    dim = settings.embedding_dim
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        tokens = text.lower().split()
        for tok in tokens:
            digest = hashlib.sha1(tok.encode("utf-8")).digest()
            # Two features per token to reduce collisions.
            idx1 = int.from_bytes(digest[0:4], "big") % dim
            sign1 = 1.0 if digest[4] & 1 else -1.0
            idx2 = int.from_bytes(digest[5:9], "big") % dim
            sign2 = 1.0 if digest[9] & 1 else -1.0
            vec[idx1] += sign1
            vec[idx2] += sign2
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        vectors.append(vec)
    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the configured provider."""
    if not texts:
        return []
    if settings.embedding_provider == "openai":
        return _embed_openai(texts)
    if settings.embedding_provider == "sentence-transformers":
        return _embed_sentence_transformers(texts)
    if settings.embedding_provider == "local":
        return _embed_local(texts)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")


def embedding_info() -> dict:
    """Human-readable record of the embedding model + dimensionality."""
    provider = settings.embedding_provider
    if provider == "openai":
        model = settings.embedding_model
        dim = 1536
    elif provider == "sentence-transformers":
        model = settings.st_embedding_model
        # Ask the loaded model for its true output dimension.
        dim = _get_st_model().get_sentence_embedding_dimension()
    else:
        model = f"local-hash-{settings.embedding_dim}d"
        dim = settings.embedding_dim
    return {"provider": provider, "model": model, "dimensionality": dim}


# --------------------------------------------------------------------------- #
# Vector store                                                                 #
# --------------------------------------------------------------------------- #
class RetrievedChunk:
    """A single retrieval hit."""

    __slots__ = ("id", "text", "metadata", "similarity")

    def __init__(self, id: str, text: str, metadata: dict, similarity: float):
        self.id = id
        self.text = text
        self.metadata = metadata
        self.similarity = similarity

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.id,
            "source": self.metadata.get("source"),
            "section": self.metadata.get("section"),
            "chunk_index": self.metadata.get("chunk_index"),
            "similarity": round(self.similarity, 4),
            "text": self.text,
        }


class VectorStore:
    """Thin wrapper over a persistent Chroma collection (cosine space)."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaClientSettings(anonymized_telemetry=False),
        )
        # cosine space so distance = 1 - cosine_similarity.
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- writes --
    def existing_ids(self, ids: list[str]) -> set[str]:
        """Return the subset of ``ids`` already present in the collection."""
        if not ids:
            return set()
        found = self._collection.get(ids=ids, include=[])
        return set(found.get("ids", []))

    def add(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Add texts under the given ids + metadata.

        Embeds ``texts`` unless precomputed ``embeddings`` are supplied (lets a
        benchmark embed once and share vectors across backends)."""
        if not ids:
            return
        if embeddings is None:
            embeddings = embed_texts(texts)
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    # -- reads --
    def count(self) -> int:
        return self._collection.count()

    def query(
        self,
        query_text: str,
        k: int,
        where: dict | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        """Top-k retrieval for ``query_text`` with an optional metadata filter.

        ``where`` is a Chroma metadata filter, e.g. {"source": "notes.md"}.
        Pass ``query_embedding`` to skip embedding (used by benchmarks to time
        only the search). Returns hits ordered by descending cosine similarity.
        """
        if self.count() == 0:
            return []
        if query_embedding is None:
            query_embedding = embed_texts([query_text])[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[RetrievedChunk] = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            similarity = 1.0 - float(dist)  # cosine distance -> cosine similarity
            hits.append(RetrievedChunk(id=cid, text=doc, metadata=meta, similarity=similarity))
        return hits

    def all_chunks(self, include_text: bool = False) -> list[dict]:
        """Return every stored chunk's id + metadata (optionally its text).

        Used by the eval harness to resolve human-friendly (source, chunk_index)
        gold references into content-addressed chunk ids, and by the CLI
        ``chunks`` command so users can discover chunk ids for their gold set.
        """
        include = ["metadatas"]
        if include_text:
            include.append("documents")
        got = self._collection.get(include=include)
        out: list[dict] = []
        ids = got.get("ids", [])
        metas = got.get("metadatas", []) or []
        docs = got.get("documents", []) if include_text else [None] * len(ids)
        for i, cid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            row = {
                "chunk_id": cid,
                "source": meta.get("source"),
                "section": meta.get("section"),
                "chunk_index": meta.get("chunk_index"),
            }
            if include_text:
                text = docs[i] if i < len(docs) else ""
                row["text_preview"] = (text or "")[:160].replace("\n", " ")
            out.append(row)
        # Stable ordering: by source then chunk_index.
        out.sort(key=lambda r: (r["source"] or "", r["chunk_index"] if r["chunk_index"] is not None else 0))
        return out

    def reset(self) -> None:
        """Delete and recreate the collection (used by tests / fresh ingests)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
