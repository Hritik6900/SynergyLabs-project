# RAG Evaluation Summary

## Configuration

- Questions: **30**  |  Chunks in store: **63**
- Retrieval k: **5**  |  Similarity threshold: **0.25**
- Embedding: **sentence-transformers/all-MiniLM-L6-v2** (384-dim, provider=sentence-transformers)
- LLM: **groq / llama-3.1-8b-instant**

## Retrieval quality (mean across questions)

| Metric | Value |
| --- | --- |
| Recall@5 | 0.7333 |
| Hit Rate@5 | 0.7333 |
| MRR@5 | 0.4922 |
| nDCG@5 | 0.5526 |
| Context Precision@5 | 0.1467 |

## Answer quality

| Metric | Value |
| --- | --- |
| Faithfulness (1-5) | 5.0 |
| Answer relevance (1-5) | 5.0 |
| Exact Match | 0.0 |
| Token F1 | 0.407 |

_Judged 8 question(s) — an evenly-spread sample of 8 (retrieval metrics above cover ALL questions; answer-eval was sampled to fit the LLM rate limit)._

## Retrieval latency

| Metric | ms |
| --- | --- |
| p50 | 21.34 |
| p95 | 28.2 |
| mean | 20.59 |
| max | 28.81 |
