# Vector Databases

## What a vector database is

A vector database stores high-dimensional vectors (embeddings) and supports
approximate nearest-neighbor (ANN) search over them. Instead of matching exact
keywords, it finds records whose embeddings are geometrically close to a query
embedding, which makes it well suited to semantic search and retrieval-augmented
generation (RAG).

## Distance metrics

The most common similarity measures are cosine similarity, dot product, and
Euclidean (L2) distance. Cosine similarity is popular for text embeddings because
it is insensitive to vector magnitude and only compares direction. A cosine
similarity of 1.0 means the two vectors point in the same direction; 0.0 means
they are orthogonal (unrelated).

## ANN indexes

Exact nearest-neighbor search is O(n) per query, which does not scale. ANN indexes
trade a small amount of recall for large speedups. Two widely used index families
are HNSW (Hierarchical Navigable Small World graphs) and IVF (inverted file with
coarse quantization). HNSW gives excellent recall and latency at the cost of higher
memory use, while IVF-based indexes are more memory efficient at very large scale.

## Metadata filtering

Production vector search almost always combines vector similarity with metadata
filters, for example restricting results to a particular source document, tenant,
language, or date range. Filtering before or alongside the ANN search is what makes
multi-tenant and permissioned retrieval possible.
