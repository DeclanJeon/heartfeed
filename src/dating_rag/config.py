"""Application configuration using Pydantic settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class QdrantSettings(BaseSettings):
    """Qdrant connection settings."""

    url: str = "http://localhost:6333"
    api_key: str = ""

    model_config = {"env_prefix": "QDRANT_"}


class EmbeddingSettings(BaseSettings):
    """Embedding model settings."""

    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"

    model_config = {"env_prefix": "EMBEDDING_"}


class LLMSettings(BaseSettings):
    """LLM API settings with fallback support."""

    api_url: str = "https://api.xiaomimimo.com/v1"
    api_key: str = ""
    model: str = "mimo-v2.5"
    fallback_api_url: str = "https://openrouter.ai/api/v1"
    fallback_api_key: str = ""
    fallback_model: str = "google/gemma-4-31b-it:free"

    model_config = {"env_prefix": "LLM_"}


class APISettings(BaseSettings):
    """API server settings."""

    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "API_"}


class Settings(BaseSettings):
    """Root application settings."""

    data_dir: Path = Path("./data")
    collection_name: str = "datewise_transcripts"
    auto_category_filters: bool = False

    model_config = {"env_prefix": "DATEWISE_"}
    config_dir: Path = Path("./config")

    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    api: APISettings = Field(default_factory=APISettings)

    def load_yaml_config(self, name: str) -> dict[str, Any]:
        """Load a YAML config file from the config directory."""
        path = self.config_dir / f"{name}.yaml"
        with open(path) as f:
            return yaml.safe_load(f)

    @property
    def categories(self) -> list[dict[str, str]]:
        """Load category definitions."""
        data = self.load_yaml_config("categories")
        return data.get("categories", [])

    @property
    def retrieval_config(self) -> dict[str, Any]:
        """Load retrieval parameters."""
        return self.load_yaml_config("retrieval")

    @property
    def prompts_config(self) -> dict[str, str]:
        """Load prompt templates."""
        return self.load_yaml_config("prompts")


def get_settings() -> Settings:
    """Create settings instance from environment variables."""
    return Settings()
