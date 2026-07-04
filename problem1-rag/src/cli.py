"""Command-line interface for the RAG service (an alternative to the HTTP API).

Examples:
    python -m src.cli ingest data/sample_corpus
    python -m src.cli query "What is cosine similarity?" --k 3
    python -m src.cli query "..." --source vector_databases.md
    python -m src.cli stats
"""

from __future__ import annotations

import argparse
import json

from .config import settings
from .embed_store import VectorStore, embedding_info
from .generate import generate_answer
from .ingest import ingest_folder


def _cmd_ingest(args: argparse.Namespace) -> None:
    store = VectorStore()
    result = ingest_folder(args.folder, store=store)
    print(json.dumps(result.as_dict(), indent=2))
    print(f"\nCollection now holds {store.count()} chunks.")


def _cmd_query(args: argparse.Namespace) -> None:
    where = {"source": args.source} if args.source else None
    result = generate_answer(args.question, k=args.k, where=where)
    print("\n=== ANSWER ===")
    print(result.answer)
    print("\n=== CITED CHUNKS ===")
    for c in result.cited_chunks:
        print(f"  [{c['source']} #{c['chunk_index']}] sim={c['similarity']} id={c['chunk_id'][:12]}...")
    print("\n=== METRICS ===")
    print(json.dumps(
        {
            "chunk_count": result.chunk_count,
            "token_usage": result.token_usage,
            "retrieval_latency_ms": round(result.retrieval_latency_ms, 2),
            "generation_latency_ms": round(result.generation_latency_ms, 2),
            "no_relevant_context": result.no_relevant_context,
        },
        indent=2,
    ))


def _cmd_chunks(_args: argparse.Namespace) -> None:
    """Dump every stored chunk so you can pick gold chunk ids for eval."""
    store = VectorStore()
    rows = store.all_chunks(include_text=True)
    for r in rows:
        print(f"[{r['source']} #{r['chunk_index']}]  id={r['chunk_id']}")
        print(f"    {r['text_preview']}")
    print(f"\n{len(rows)} chunks total.")


def _cmd_stats(_args: argparse.Namespace) -> None:
    store = VectorStore()
    print(json.dumps(
        {
            "collection": settings.chroma_collection,
            "persist_dir": settings.chroma_persist_dir,
            "chunk_count": store.count(),
            "embedding": embedding_info(),
            "llm_provider": settings.llm_provider,
        },
        indent=2,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG service CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a corpus folder")
    p_ingest.add_argument("folder")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("question")
    p_query.add_argument("--k", type=int, default=None, help="Top-k chunks")
    p_query.add_argument("--source", default=None, help="Filter to a source filename")
    p_query.set_defaults(func=_cmd_query)

    p_stats = sub.add_parser("stats", help="Show collection stats")
    p_stats.set_defaults(func=_cmd_stats)

    p_chunks = sub.add_parser("chunks", help="List all chunks (find gold ids for eval)")
    p_chunks.set_defaults(func=_cmd_chunks)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
