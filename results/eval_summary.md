# RAG Evaluation Summary

## Configuration

- Questions: **5**  |  Chunks in store: **3**
- Retrieval k: **3**  |  Similarity threshold: **0.25**
- Embedding: **local-hash-1536d** (1536-dim, provider=local)
- LLM: **openai / gpt-4o-mini**

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
| p50 | 2.78 |
| p95 | 12.0 |
| mean | 5.12 |
| max | 14.16 |
