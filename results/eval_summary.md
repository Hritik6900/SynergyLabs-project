# RAG Evaluation Summary

## Configuration

- Questions: **8**  |  Chunks in store: **63**
- Retrieval k: **5**  |  Similarity threshold: **0.25**
- Embedding: **sentence-transformers/all-MiniLM-L6-v2** (384-dim, provider=sentence-transformers)
- LLM: **groq / llama-3.3-70b-versatile**

## Retrieval quality (mean across questions)

| Metric | Value |
| --- | --- |
| Recall@5 | 0.8750 |
| Hit Rate@5 | 0.8750 |
| MRR@5 | 0.6042 |
| nDCG@5 | 0.6741 |
| Context Precision@5 | 0.1750 |

## Answer quality

| Metric | Value |
| --- | --- |
| Faithfulness (1-5) | 5.0 |
| Answer relevance (1-5) | 5.0 |
| Exact Match | 0.0 |
| Token F1 | 0.34 |

_Judged 8 question(s)._

## Retrieval latency

| Metric | ms |
| --- | --- |
| p50 | 21.3 |
| p95 | 28.16 |
| mean | 22.49 |
| max | 28.34 |
