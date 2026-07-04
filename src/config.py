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
    # Providers:
    #   "openai"                -> text-embedding-3-small (1536-dim), needs OPENAI_API_KEY.
    #   "sentence-transformers" -> a local CPU model (default 384-dim), free, no key.
    #   "local"                 -> deterministic hash embedding, offline, zero deps,
    #                              low quality (smoke tests / CI only).
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536  # used by the "local" hash provider
    # Model used when embedding_provider == "sentence-transformers".
    st_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- Generation ----
    # "openai" | "groq" | "anthropic". Groq is OpenAI-compatible (chat only) and is
    # used via the OpenAI SDK with GROQ_BASE_URL.
    llm_provider: str = "openai"
    openai_llm_model: str = "gpt-4o-mini"
    anthropic_llm_model: str = "claude-haiku-4-5"
    groq_llm_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # ---- Retrieval ----
    top_k: int = 5
    similarity_threshold: float = 0.25

    # ---- Secrets ----
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None

    # ---- Logging ----
    query_log_path: str = "./logs/queries.log"

    def embedding_dimensionality(self) -> int:
        """Effective embedding dimensionality.

        openai fixes 1536; sentence-transformers is model-dependent (resolved at
        runtime by the store); the local hash provider uses embedding_dim.
        """
        if self.embedding_provider == "openai":
            return 1536
        return self.embedding_dim

    def active_llm_model(self) -> str:
        """The model name for the configured LLM provider."""
        return {
            "openai": self.openai_llm_model,
            "groq": self.groq_llm_model,
            "anthropic": self.anthropic_llm_model,
        }.get(self.llm_provider, self.llm_provider)


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Convenient module-level singleton.
settings = get_settings()
