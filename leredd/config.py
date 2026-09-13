"""Configuration settings for LEREDD using Pydantic Settings."""

import os
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LEREDDConfig(BaseSettings):
    """LEREDD Configuration Parameters based on Paper RQ2 Optimal Settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Embedding and Similarity Settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="SBERT embedding model name or path"
    )
    similarity_metric: Literal["euclidean", "cosine"] = Field(
        default="euclidean",
        description="Distance/Similarity metric"
    )
    aggregation: Literal["average", "max"] = Field(
        default="average",
        description="Formula for aggregating pair similarities (average Eq. 1 vs max Eq. 2)"
    )
    k_examples: int = Field(
        default=6,
        ge=1,
        le=10,
        description="Number of dynamic in-context examples per dependency type"
    )

    # RAG Settings
    use_rag: bool = Field(
        default=False,
        description="Whether to augment prompt with RAG context chunks from SRS"
    )
    rag_chunk_size: int = Field(
        default=1000,
        description="SRS document fixed chunk size in characters"
    )
    rag_chunk_overlap: int = Field(
        default=200,
        description="SRS document chunk overlap in characters"
    )
    rag_k_chunks: int = Field(
        default=6,
        description="Top k SRS context chunks to retrieve for RAG"
    )

    # LLM Settings
    llm_provider: Literal["openai", "ollama", "mock"] = Field(
        default="mock",
        description="LLM provider backend"
    )
    llm_model: str = Field(
        default="gpt-4",
        description="LLM model identifier"
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for LLM generation"
    )
    confidence_threshold: int = Field(
        default=4,
        ge=0,
        le=5,
        description="Re-annotation threshold: predictions with confidence <= threshold are set to No_dependency"
    )

    # API Keys & URLs
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama local service"
    )

    # Directory Paths
    data_dir: str = Field(
        default="data",
        description="Base directory for data files"
    )
    cache_dir: str = Field(
        default="data/embeddings",
        description="Directory for caching SBERT embeddings and LLM responses"
    )

    def get_data_path(self, relative_path: str) -> str:
        """Resolve a path relative to the project data directory."""
        return os.path.join(self.data_dir, relative_path)


_config_instance: LEREDDConfig | None = None


def get_config() -> LEREDDConfig:
    """Get singleton instance of LEREDDConfig."""
    global _config_instance
    if _config_instance is None:
        _config_instance = LEREDDConfig()
    return _config_instance
