"""Baseline implementations for requirement dependency detection comparison."""

from leredd.baselines.tfidf_lsa import TFIDF_LSABaseline
from leredd.baselines.bert_classifier import BERTBaseline

__all__ = ["TFIDF_LSABaseline", "BERTBaseline"]
