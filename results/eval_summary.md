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

_LLM-as-judge disabled (no LLM API key configured). Retrieval + latency were still evaluated._

## Retrieval latency

| Metric | ms |
| --- | --- |
| p50 | 23.01 |
| p95 | 35.04 |
| mean | 26.12 |
| max | 35.65 |
