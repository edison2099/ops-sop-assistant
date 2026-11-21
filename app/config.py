"""App settings, loaded from environment variables or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True)

    # Providers: "anthropic" | "openai" | "fake" (offline, for tests and demos)
    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-5-5"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 800
    # USD per million tokens, used only for the cost estimate in logs. Set to your model's price.
    llm_input_price_per_mtok: float = 1.0
    llm_output_price_per_mtok: float = 5.0

    # Embeddings: "local" (free, runs on your computer) | "openai" | "hash" (keyword-only, for tests)
    # local model: BAAI/bge-small-en-v1.5 ; openai model: text-embedding-3-small
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Who the assistant is, used in the system prompt. Change it to match your documents, e.g.
    # "a motorcycle service assistant for the Suzuki GSX-S750 service manual"
    assistant_role: str = "an operations SOP assistant"

    # Retrieval
    top_k: int = 5
    # Below this similarity a chunk is treated as not relevant. Leave unset for the provider default
    # (local 0.60, openai 0.30, hash 0.20), then tune with eval/run_eval.py.
    min_similarity: float | None = None
    chunk_words: int = 450  # roughly 600 tokens
    chunk_overlap_words: int = 75  # roughly 100 tokens

    # Storage
    data_dir: Path = Path("data")
    collection_name: str = "sop_chunks"

    # Security: admin endpoints need header X-Admin-Key
    admin_api_key: str = "change-me"
    max_upload_mb: int = 50

    @property
    def similarity_threshold(self) -> float:
        if self.min_similarity is not None:
            return self.min_similarity
        return {"local": 0.60, "openai": 0.30, "hash": 0.20}.get(self.embedding_provider, 0.30)

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "app.db"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    return s
