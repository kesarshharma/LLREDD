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
        ge=0,
        le=10,
        description="Number of dynamic in-context examples per dependency type"
    )
    retrieval_mode: Literal["dense", "bm25", "hybrid", "hybrid_no_metadata", "hybrid_metadata"] = Field(
        default="dense",
        description=(
            "Example retrieval strategy: 'dense' (SBERT), 'bm25' (lexical), "
            "'hybrid' (RRF fusion), 'hybrid_no_metadata' (dense+bm25 only), "
            "or 'hybrid_metadata' (dense+bm25+metadata)"
        ),
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Reciprocal Rank Fusion constant parameter"
    )
    dense_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="RRF weight for SBERT dense retrieval in hybrid mode",
    )
    bm25_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="RRF weight for BM25 lexical retrieval in hybrid mode",
    )
    metadata_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="RRF weight for structured metadata overlap in hybrid mode (0.0 disables metadata signal)",
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
    rag_chunking_mode: Literal["fixed", "requirement_boundary"] = Field(
        default="fixed",
        description="SRS chunking strategy: 'fixed' character windows or 'requirement_boundary' aware chunking",
    )
    rag_deduplicate_chunks: bool = Field(
        default=False,
        description="Whether to filter exact and near-duplicate SRS context chunks prior to top-k selection",
    )
    rag_near_duplicate_threshold: float = Field(
        default=0.90,
        gt=0.0,
        le=1.0,
        description="Token Jaccard similarity threshold above which chunks are treated as near-duplicates",
    )
    rag_cite_id_boost: float = Field(
        default=0.0,
        ge=0.0,
        description="Optional additive similarity boost for chunks containing requirement IDs cited by target pair",
    )
    rag_min_similarity: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Diagnostic similarity floor below which selected chunks are flagged as potentially irrelevant",
    )

    # LLM Settings
    llm_provider: Literal["openai", "ollama", "gemini", "mock"] = Field(
        default="mock",
        description="LLM provider backend (openai, ollama, gemini, or mock)"
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
        ge=-1,
        le=5,
        description=(
            "Self-reported confidence cutoff: predictions at or below it are flagged "
            "for review without changing the predicted class. Set -1 to disable."
        )
    )
    legacy_rewrite_low_confidence_to_no_dependency: bool = Field(
        default=False,
        description=(
            "DEPRECATED compatibility option. Keep False so low-confidence predictions "
            "preserve their predicted DependencyType and emit validation_status='requires_review'. "
            "Only set True when reproducing pre-audit legacy behavior."
        ),
    )

    # API Keys & URLs
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key"
    )
    gemini_api_key: str | None = Field(
        default=None,
        description="Google Gemini API key"
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


def reset_config() -> None:
    """Reset the cached global configuration singleton.

    Sets the module-level singleton instance to None, forcing the next
    call to get_config() to reload configuration from environment variables,
    .env files, or default values. Intended primarily for testing and
    controlled reinitialization.
    """
    global _config_instance
    _config_instance = None
