# RAG Evaluation Summary

## Configuration

- Questions: **5**  |  Chunks in store: **3**
- Retrieval k: **3**  |  Similarity threshold: **0.25**
- Embedding: **sentence-transformers/all-MiniLM-L6-v2** (384-dim, provider=sentence-transformers)
- LLM: **groq / llama-3.3-70b-versatile**

## Retrieval quality (mean across questions)

| Metric | Value |
| --- | --- |
| Recall@3 | 1.0000 |
| Hit Rate@3 | 1.0000 |
| MRR@3 | 0.9000 |
| nDCG@3 | 0.9262 |
| Context Precision@3 | 0.3333 |

## Answer quality

| Metric | Value |
| --- | --- |
| Faithfulness (1-5) | 5.0 |
| Answer relevance (1-5) | 5.0 |
| Exact Match | 0.0 |
| Token F1 | 0.543 |

_Judged 5 question(s)._

## Retrieval latency

| Metric | ms |
| --- | --- |
| p50 | 22.62 |
| p95 | 26.81 |
| mean | 22.36 |
| max | 27.41 |
