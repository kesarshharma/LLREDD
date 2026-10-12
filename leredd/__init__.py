"""LEREDD: LLM-Enabled REquirement Dependency Detection package."""

__version__ = "1.0.0"
__author__ = "LEREDD Research Team"

from leredd.config import LEREDDConfig, get_config
from leredd.data_models import (
    Requirement,
    RequirementPair,
    DependencyType,
    Prediction,
    EvaluationReport,
    AnnotationSource,
    VerificationStatus,
    AnnotationProvenance,
)
from leredd.dataset_splitter import (
    DatasetSplitter,
    SplitConfig,
    DatasetSplit,
    SplitStrategy,
    LeakageReport,
)
from leredd.bm25_retrieval import BM25Retriever
from leredd.hybrid_retrieval import HybridRetriever
from leredd.candidate_generator import (
    CandidateGenerator,
    CandidateGenerationManifest,
    GenerationStrategy,
    get_canonical_pair_key,
)
from leredd.provisional_labeler import (
    HeuristicLabeler,
    LLMLabeler,
    ProvisionalLabeler,
    LabelingRunManifest,
    LabelingMode,
)
from leredd.review_queue import (
    ReviewReasonCode,
    QueueStatus,
    ReviewPriority,
    HumanDecision,
    ReviewItem,
    ReviewQueueConfig,
    ReviewQueueManifest,
    ReviewQueueManager,
)

__all__ = [
    "LEREDDConfig",
    "get_config",
    "Requirement",
    "RequirementPair",
    "DependencyType",
    "Prediction",
    "EvaluationReport",
    "AnnotationSource",
    "VerificationStatus",
    "AnnotationProvenance",
    "DatasetSplitter",
    "SplitConfig",
    "DatasetSplit",
    "SplitStrategy",
    "LeakageReport",
    "BM25Retriever",
    "HybridRetriever",
    "CandidateGenerator",
    "CandidateGenerationManifest",
    "GenerationStrategy",
    "get_canonical_pair_key",
    "HeuristicLabeler",
    "LLMLabeler",
    "ProvisionalLabeler",
    "LabelingRunManifest",
    "LabelingMode",
    "ReviewReasonCode",
    "QueueStatus",
    "ReviewPriority",
    "HumanDecision",
    "ReviewItem",
    "ReviewQueueConfig",
    "ReviewQueueManifest",
    "ReviewQueueManager",
]
