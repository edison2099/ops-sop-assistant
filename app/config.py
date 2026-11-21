"""App settings, loaded from environment variables or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Providers: "anthropic" | "openai" | "fake" (offline, for tests and demos)
    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-5-5"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 800
    # USD per million tokens, used only for the cost estimate in logs. Set to your model's price.
    llm_input_price_per_mtok: float = 1.0
    llm_output_price_per_mtok: float = 5.0

    # Embeddings: "openai" | "hash" (offline, keyword-based, for tests and demos)
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Who the assistant is, used in the system prompt. Change it to match your documents, e.g.
    # "a motorcycle service assistant for the Suzuki GSX-S750 service manual"
    assistant_role: str = "an operations SOP assistant"

    # Retrieval
    top_k: int = 5
    min_similarity: float = 0.30  # below this, the chunk is treated as not relevant
    chunk_words: int = 450  # roughly 600 tokens
    chunk_overlap_words: int = 75  # roughly 100 tokens

    # Storage
    data_dir: Path = Path("data")
    collection_name: str = "sop_chunks"

    # Security: admin endpoints need header X-Admin-Key
    admin_api_key: str = "change-me"
    max_upload_mb: int = 50

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
