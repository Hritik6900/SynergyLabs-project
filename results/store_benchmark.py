"""Bonus: benchmark a SECOND vector store (FAISS) against ChromaDB.

Both backends are built from the *same* embeddings of the same corpus, so the
comparison is apples-to-apples. We then, for a set of query embeddings (also
computed once), measure per-search latency for each store and how well ChromaDB's
approximate HNSW index agrees with FAISS exact search (our ground truth).

What this shows:
  * Retrieval latency (p50/p95) of each store's search step alone (embedding is
    excluded — it is identical for both).
  * Recall agreement @k: ChromaDB HNSW vs FAISS exact — i.e. how much recall the
    approximate index gives up (on this corpus, typically none at small scale).

Run:  python results/store_benchmark.py [--k 5 --repeats 50]
Writes results/store_benchmark.md.

Uses the configured embedding provider (default: local sentence-transformers), so
it runs offline with zero API cost.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

# Make `src` importable regardless of CWD.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json  # noqa: E402

from src.embed_store import VectorStore, embed_texts, embedding_info  # noqa: E402
from src.faiss_store import FaissStore  # noqa: E402
from src.ingest import _chunk_document, load_document  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "results")
DEFAULT_CORPUS = os.path.join(_ROOT, "data", "sample_corpus")
QUESTIONS = os.path.join(_ROOT, "eval", "questions.json")


def _load_chunks(corpus_dir: str):
    """Chunk every supported file into (id, text, metadata) tuples."""
    from pathlib import Path

    from src.ingest import SUPPORTED_EXTENSIONS

    ids, texts, metas = [], [], []
    for path in sorted(Path(corpus_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        doc = load_document(path)
        if not doc:
            continue
        for cid, chunk, section in _chunk_document(doc):
            ids.append(cid)
            texts.append(chunk.text)
            metas.append({"source": doc.filename, "section": section, "chunk_index": chunk.chunk_index})
    return ids, texts, metas


def _load_queries() -> list[str]:
    """Use the eval questions as the query workload; fall back to a default set."""
    if os.path.exists(QUESTIONS):
        spec = json.load(open(QUESTIONS, encoding="utf-8"))
        qs = [q["question"] for q in spec.get("questions", []) if "question" in q]
        if qs:
            return qs
    return [
        "What is cosine similarity?",
        "How does idempotent ingestion work?",
        "What is an ANN index?",
        "When should a RAG system refuse to answer?",
    ]


def _percentiles(latencies_ms: list[float]) -> dict:
    s = sorted(latencies_ms)
    def pct(p):
        if not s:
            return 0.0
        idx = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
        return s[idx]
    return {
        "p50_ms": round(pct(50), 3),
        "p95_ms": round(pct(95), 3),
        "mean_ms": round(statistics.mean(s), 3) if s else 0.0,
    }


def run(corpus_dir: str, k: int, repeats: int) -> dict:
    ids, texts, metas = _load_chunks(corpus_dir)
    if not ids:
        raise RuntimeError(f"No chunks found under {corpus_dir}")

    # Embed the corpus ONCE and share the vectors across both backends.
    doc_embeddings = embed_texts(texts)

    chroma = VectorStore(persist_dir="./_bench_chroma", collection_name="bench")
    chroma.reset()
    chroma.add(ids=ids, texts=texts, metadatas=metas, embeddings=doc_embeddings)

    faiss_store = FaissStore()
    faiss_store.add_precomputed(ids=ids, embeddings=doc_embeddings, texts=texts, metadatas=metas)

    queries = _load_queries()
    query_embeddings = embed_texts(queries)  # embed queries once, shared + untimed

    # Warm up both search paths (exclude one-time init from timings).
    chroma.query(queries[0], k=k, query_embedding=query_embeddings[0])
    faiss_store.query(queries[0], k=k, query_embedding=query_embeddings[0])

    chroma_lat, faiss_lat = [], []
    overlap_scores = []
    for _ in range(repeats):
        for qtext, qemb in zip(queries, query_embeddings):
            t0 = time.perf_counter()
            c_hits = chroma.query(qtext, k=k, query_embedding=qemb)
            chroma_lat.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            f_hits = faiss_store.query(qtext, k=k, query_embedding=qemb)
            faiss_lat.append((time.perf_counter() - t0) * 1000)

    # Recall agreement (compute once per query, not per repeat): FAISS exact = truth.
    for qtext, qemb in zip(queries, query_embeddings):
        c_ids = {h.id for h in chroma.query(qtext, k=k, query_embedding=qemb)}
        f_ids = {h.id for h in faiss_store.query(qtext, k=k, query_embedding=qemb)}
        if f_ids:
            overlap_scores.append(len(c_ids & f_ids) / len(f_ids))

    chroma.reset()  # clean up bench collection

    return {
        "config": {
            "corpus_dir": corpus_dir,
            "num_chunks": len(ids),
            "num_queries": len(queries),
            "k": k,
            "repeats": repeats,
            "total_searches_per_store": len(chroma_lat),
            "embedding": embedding_info(),
        },
        "chromadb": {"index": "HNSW (approximate, cosine)", **_percentiles(chroma_lat)},
        "faiss": {"index": "IndexFlatIP (exact, cosine)", **_percentiles(faiss_lat)},
        "recall_agreement_at_k": round(statistics.mean(overlap_scores), 4) if overlap_scores else None,
    }


def _write_md(report: dict, path: str) -> None:
    c = report["config"]
    ch, fa = report["chromadb"], report["faiss"]
    lines = [
        "# Store Benchmark: ChromaDB vs FAISS (bonus)\n",
        "Both stores are built from the **same embeddings** of the same corpus and "
        "queried with the **same pre-computed query vectors**, so only the search "
        "step differs. FAISS uses an exact flat index (ground truth); ChromaDB uses "
        "approximate HNSW.\n",
        "## Setup\n",
        f"- Corpus: `{c['corpus_dir']}` — **{c['num_chunks']}** chunks",
        f"- Queries: **{c['num_queries']}**, repeated {c['repeats']}x = "
        f"**{c['total_searches_per_store']}** searches/store",
        f"- k = **{c['k']}**",
        f"- Embedding: **{c['embedding']['model']}** ({c['embedding']['dimensionality']}-dim, "
        f"provider={c['embedding']['provider']})\n",
        "## Search latency (embedding excluded — identical for both)\n",
        "| Store | Index | p50 (ms) | p95 (ms) | mean (ms) |",
        "| --- | --- | --- | --- | --- |",
        f"| ChromaDB | {ch['index']} | {ch['p50_ms']} | {ch['p95_ms']} | {ch['mean_ms']} |",
        f"| FAISS | {fa['index']} | {fa['p50_ms']} | {fa['p95_ms']} | {fa['mean_ms']} |",
        "",
        "## Recall agreement @k (ChromaDB HNSW vs FAISS exact)\n",
        f"- **{report['recall_agreement_at_k']}** — fraction of FAISS's exact top-k that "
        "ChromaDB's approximate index also returns (1.0 = no recall lost to approximation).\n",
        "## Takeaways\n",
        "- At this corpus size both stores are sub-millisecond-to-low-ms per search; "
        "the embedding step (excluded here) dominates end-to-end query latency in practice.",
        "- FAISS (in-memory, exact) is a great raw-speed / ground-truth baseline but is "
        "*just* an index: no persistence, metadata store, or filtering out of the box — "
        "you build those yourself. ChromaDB bundles persistence + metadata + filtering, "
        "which is why it's the primary store here; FAISS is the benchmark yardstick.",
        "- Recall agreement quantifies exactly what ChromaDB's HNSW approximation costs "
        "in retrieval quality versus exact search.\n",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark ChromaDB vs FAISS.")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()

    report = run(args.corpus, k=args.k, repeats=args.repeats)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    md_path = os.path.join(RESULTS_DIR, "store_benchmark.md")
    json_path = os.path.join(RESULTS_DIR, "store_benchmark.json")
    _write_md(report, md_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nWrote {md_path}\nWrote {json_path}")


if __name__ == "__main__":
    main()
