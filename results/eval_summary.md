# RAG Evaluation Summary

## Configuration

- Questions: **35**  |  Chunks in store: **63**
- Retrieval k: **5**  |  Similarity threshold: **0.25**
- Embedding: **sentence-transformers/all-MiniLM-L6-v2** (384-dim, provider=sentence-transformers)
- LLM: **groq / llama-3.1-8b-instant**

## Retrieval quality (mean across questions)

| Metric | Value |
| --- | --- |
| Recall@5 | 0.7143 |
| Hit Rate@5 | 0.7143 |
| MRR@5 | 0.4933 |
| nDCG@5 | 0.5488 |
| Context Precision@5 | 0.1429 |

## Answer quality

| Metric | Value |
| --- | --- |
| Faithfulness (1-5) | 4.125 |
| Answer relevance (1-5) | 4.5 |
| Exact Match | 0.0 |
| Token F1 | 0.356 |

_Judged 8 question(s) — an evenly-spread sample of 8 (retrieval metrics above cover ALL questions; answer-eval was sampled to fit the LLM rate limit)._

## Retrieval latency

| Metric | ms |
| --- | --- |
| p50 | 21.2 |
| p95 | 26.96 |
| mean | 20.53 |
| max | 27.63 |
