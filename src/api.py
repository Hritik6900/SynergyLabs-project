"""FastAPI HTTP surface for the RAG service.

Endpoints:
  POST /ingest  {folder}                       -> ingestion stats
  POST /query   {question, k?, filter?}         -> {answer, cited_chunks,
                                                    latency_ms, chunk_count,
                                                    token_usage, ...}
  GET  /health                                  -> liveness + config summary
  GET  /stats                                   -> collection size + embedding info

Every /query call is logged (latency, chunk count, token usage) to a local file.
Config comes from env vars only (see config.py); no secrets are hardcoded.

Run:  uvicorn src.api:app --reload --port 8000
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .embed_store import VectorStore, embedding_info
from .generate import generate_answer
from .ingest import ingest_folder
from .logging_utils import log_query

app = FastAPI(
    title="Cost-Efficient RAG Service",
    description="RAG over a document corpus backed by ChromaDB (low-cost store).",
    version="1.0.0",
)

# One store instance reused across requests (opens the persistent client once).
_store = VectorStore()


# --------------------------------------------------------------------------- #
# Request/response models                                                      #
# --------------------------------------------------------------------------- #
class IngestRequest(BaseModel):
    folder: str = Field(..., description="Path to the corpus folder to ingest.")


class QueryRequest(BaseModel):
    question: str = Field(..., description="The natural-language question.")
    k: int | None = Field(None, description="Top-k chunks to retrieve.")
    filter: dict | None = Field(
        None,
        description='Optional Chroma metadata filter, e.g. {"source": "notes.md"}.',
    )


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "embedding": embedding_info(),
        "llm_provider": settings.llm_provider,
        "llm_model": (
            settings.openai_llm_model
            if settings.llm_provider == "openai"
            else settings.anthropic_llm_model
        ),
        "top_k": settings.top_k,
        "similarity_threshold": settings.similarity_threshold,
    }


@app.get("/stats")
def stats() -> dict:
    return {
        "collection": settings.chroma_collection,
        "persist_dir": settings.chroma_persist_dir,
        "chunk_count": _store.count(),
        "embedding": embedding_info(),
    }


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    try:
        result = ingest_folder(req.folder, store=_store)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.as_dict()


@app.post("/query")
def query(req: QueryRequest) -> dict:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    t0 = time.perf_counter()
    try:
        result = generate_answer(
            req.question,
            k=req.k,
            where=req.filter,
            store=_store,
        )
    except RuntimeError as exc:
        # Misconfiguration (e.g. provider selected but API key missing).
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    total_latency_ms = (time.perf_counter() - t0) * 1000.0

    response = {
        "answer": result.answer,
        "cited_chunks": result.cited_chunks,
        "latency_ms": round(total_latency_ms, 2),
        "chunk_count": result.chunk_count,
        "token_usage": result.token_usage,
        "no_relevant_context": result.no_relevant_context,
        "retrieval_latency_ms": round(result.retrieval_latency_ms, 2),
        "generation_latency_ms": round(result.generation_latency_ms, 2),
    }

    # Persist an operational log line for this query.
    log_query(
        {
            "question": req.question,
            "k": req.k if req.k is not None else settings.top_k,
            "filter": req.filter,
            "latency_ms": round(total_latency_ms, 2),
            "retrieval_latency_ms": round(result.retrieval_latency_ms, 2),
            "generation_latency_ms": round(result.generation_latency_ms, 2),
            "chunk_count": result.chunk_count,
            "token_usage": result.token_usage,
            "no_relevant_context": result.no_relevant_context,
            "cited_chunk_ids": [c["chunk_id"] for c in result.cited_chunks],
        }
    )
    return response
