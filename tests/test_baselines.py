"""Tests for TF-IDF & LSA and BERT Baselines."""

from leredd.baselines.tfidf_lsa import TFIDF_LSABaseline
from leredd.baselines.bert_classifier import BERTBaseline
from leredd.data_models import DependencyType


def test_tfidf_lsa_baseline(sample_pair, sample_annotated_pool):
    baseline = TFIDF_LSABaseline(n_components=5)
    baseline.fit(sample_annotated_pool)
    pred = baseline.predict_pair(sample_pair)

    assert pred.pair_id == sample_pair.id
    assert pred.dependency_type in [DependencyType.REQUIRES, DependencyType.NO_DEPENDENCY]


def test_bert_baseline(sample_pair, sample_annotated_pool):
    baseline = BERTBaseline()
    baseline.fit(sample_annotated_pool)
    pred = baseline.predict_pair(sample_pair)

    assert pred.pair_id == sample_pair.id
    assert isinstance(pred.dependency_type, DependencyType)
