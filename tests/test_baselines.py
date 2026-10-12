"""Tests for TF-IDF & LSA and BERT Baselines."""

from leredd.baselines.tfidf_lsa import TFIDF_LSABaseline
from leredd.baselines.bert_classifier import BERTBaseline
from leredd.data_models import DependencyType
import pytest


class DeterministicEmbedder:
    model_name = "test-embedder"

    def encode(self, texts):
        import numpy as np

        vectors = []
        for text in texts:
            vector = np.zeros(384, dtype=np.float32)
            for index, byte in enumerate(text.encode("utf-8")):
                vector[index % 384] += byte
            vectors.append(vector)
        return np.vstack(vectors)


def test_tfidf_lsa_baseline(sample_pair, sample_annotated_pool):
    baseline = TFIDF_LSABaseline(n_components=5)
    baseline.fit(sample_annotated_pool)
    pred = baseline.predict_pair(sample_pair)

    assert pred.pair_id == sample_pair.id
    assert pred.dependency_type in [DependencyType.REQUIRES, DependencyType.NO_DEPENDENCY]


def test_bert_baseline(sample_pair, sample_annotated_pool):
    baseline = BERTBaseline(embedder=DeterministicEmbedder())
    baseline.fit(sample_annotated_pool)
    pred = baseline.predict_pair(sample_pair)

    assert pred.pair_id == sample_pair.id
    assert isinstance(pred.dependency_type, DependencyType)


def test_bert_baseline_rejects_unlabeled_training_pair(sample_pair):
    sample_pair.dependency_type = None
    baseline = BERTBaseline(embedder=object())
    with pytest.raises(ValueError, match="unlabeled or unverified"):
        baseline.fit([sample_pair])


def test_bert_baseline_artifact_roundtrip(tmp_path, sample_pair, sample_annotated_pool):
    embedder = DeterministicEmbedder()
    baseline = BERTBaseline(embedder=embedder).fit(sample_annotated_pool)
    artifact_path = tmp_path / "bert_baseline.joblib"
    baseline.save(str(artifact_path))
    original_artifact = artifact_path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        baseline.save(str(artifact_path))
    assert artifact_path.read_bytes() == original_artifact

    restored = BERTBaseline.load(str(artifact_path), embedder=embedder)

    assert restored.is_fitted
    assert restored.predict_pair(sample_pair).dependency_type == baseline.predict_pair(sample_pair).dependency_type
    baseline.save(str(artifact_path), overwrite=True)


def test_bert_baseline_cannot_save_before_fit(tmp_path):
    baseline = BERTBaseline(embedder=object())
    with pytest.raises(ValueError, match="before fitting"):
        baseline.save(str(tmp_path / "model.joblib"))


@pytest.mark.parametrize("baseline_class", [BERTBaseline, TFIDF_LSABaseline])
def test_baselines_require_separate_training_before_prediction(
    baseline_class, sample_pair
):
    baseline = baseline_class(embedder=object()) if baseline_class is BERTBaseline else baseline_class()
    with pytest.raises(ValueError, match="fit on training pairs"):
        baseline.predict_pair(sample_pair)
    with pytest.raises(ValueError, match="fit on training pairs"):
        baseline.predict_batch([sample_pair])
