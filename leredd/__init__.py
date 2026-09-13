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
)

__all__ = [
    "LEREDDConfig",
    "get_config",
    "Requirement",
    "RequirementPair",
    "DependencyType",
    "Prediction",
    "EvaluationReport",
]
