"""Central configuration, driven entirely by environment variables.

No secrets are hardcoded. In local dev, values are read from a `.env` file
(via python-dotenv); in production they come from the real environment.
Import `settings` (a singleton) everywhere else in the codebase.
"""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into the process environment if present (no-op if the file is absent).
load_dotenv()


class Settings(BaseSettings):
    """Typed view over the environment. Field names map to UPPER_CASE env vars."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- Vector store ----
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "rag_chunks"

    # ---- Chunking ----
    # 500/50 is a deliberate default: ~500 tokens is roughly 350-400 words, a
    # coherent passage that still keeps retrieval granular, and stays well within
    # the embedding model's input limit. A 50-token (10%) overlap preserves
    # continuity so facts spanning a chunk boundary are not lost.
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # ---- Embeddings ----
    # "openai" uses text-embedding-3-small (1536-dim). "local" uses a deterministic
    # hash embedding so the whole pipeline runs offline with zero API spend — useful
    # for tests, CI, and proving idempotency without a key.
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # ---- Generation ----
    llm_provider: str = "openai"  # "openai" | "anthropic"
    openai_llm_model: str = "gpt-4o-mini"
    anthropic_llm_model: str = "claude-haiku-4-5"

    # ---- Retrieval ----
    top_k: int = 5
    similarity_threshold: float = 0.25

    # ---- Secrets ----
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # ---- Logging ----
    query_log_path: str = "./logs/queries.log"

    def embedding_dimensionality(self) -> int:
        """Effective embedding dimensionality (openai fixes this at 1536)."""
        if self.embedding_provider == "openai":
            return 1536
        return self.embedding_dim


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Convenient module-level singleton.
settings = get_settings()
