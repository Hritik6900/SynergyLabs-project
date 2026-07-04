# Store Benchmark: ChromaDB vs FAISS (bonus)

Both stores are built from the **same embeddings** of the same corpus and queried with the **same pre-computed query vectors**, so only the search step differs. FAISS uses an exact flat index (ground truth); ChromaDB uses approximate HNSW.

## Setup

- Corpus: `/home/hritik/Desktop/SynergyLabs-project-1/problem1-rag/data/sample_corpus` — **63** chunks
- Queries: **5**, repeated 50x = **250** searches/store
- k = **5**
- Embedding: **sentence-transformers/all-MiniLM-L6-v2** (384-dim, provider=sentence-transformers)

## Search latency (embedding excluded — identical for both)

| Store | Index | p50 (ms) | p95 (ms) | mean (ms) |
| --- | --- | --- | --- | --- |
| ChromaDB | HNSW (approximate, cosine) | 6.861 | 7.563 | 6.217 |
| FAISS | IndexFlatIP (exact, cosine) | 0.165 | 0.22 | 0.16 |

## Recall agreement @k (ChromaDB HNSW vs FAISS exact)

- **1.0** — fraction of FAISS's exact top-k that ChromaDB's approximate index also returns (1.0 = no recall lost to approximation).

## Takeaways

- At this corpus size both stores are sub-millisecond-to-low-ms per search; the embedding step (excluded here) dominates end-to-end query latency in practice.
- FAISS (in-memory, exact) is a great raw-speed / ground-truth baseline but is *just* an index: no persistence, metadata store, or filtering out of the box — you build those yourself. ChromaDB bundles persistence + metadata + filtering, which is why it's the primary store here; FAISS is the benchmark yardstick.
- Recall agreement quantifies exactly what ChromaDB's HNSW approximation costs in retrieval quality versus exact search.
