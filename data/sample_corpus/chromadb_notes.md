# ChromaDB Notes

## Why ChromaDB for a low-cost store

ChromaDB is an open-source, embeddable vector database. In its persistent-client
mode it stores vectors and metadata on the local filesystem (backed by SQLite plus
a vector index) and requires no separate server process. This means there are no
always-on pods or hourly pod fees: cost is essentially the disk and the VM you
already run, which is the core reason a lightly-queried but large index is far
cheaper on ChromaDB than on a fully managed vector database.

## Persistence model

A `PersistentClient` writes data under a directory you choose. Re-opening the same
directory restores the collection, its embeddings, and metadata. Because the data
lives on disk, ingestion is a one-time cost and subsequent process restarts are
cheap.

## Idempotent ingestion

Chroma addresses records by an `id` you supply. If you insert a record with an id
that already exists, you can detect and skip it. Deriving the id deterministically
from the content (for example, a hash of the source path plus the chunk text) makes
re-ingestion idempotent: running ingestion twice does not create duplicate vectors.

## Querying and filters

Chroma's `query` accepts `query_embeddings`, an `n_results` value (top-k), and an
optional `where` clause for metadata filtering. The `where` clause supports equality
and simple operators on stored metadata fields such as the source filename.
