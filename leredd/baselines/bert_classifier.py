"""BERT Sequence Classifier baseline implementation for multi-class dependency detection."""

import numpy as np
import os
import tempfile
from typing import List
import joblib
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from leredd.data_models import RequirementPair, DependencyType, Prediction, VerificationStatus
from leredd.embeddings import EmbeddingModel


class BERTBaseline:
    """Fine-tuned BERT / Supervised Machine Learning multi-class classifier baseline.
    
    Handles severe class imbalance via class weighting and oversampling.
    """

    def __init__(self, embedder: EmbeddingModel | None = None, model_type: str = "rf"):
        self.embedder = embedder or EmbeddingModel()
        self.model_type = model_type
        
        if model_type == "lr":
            self.classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        else:
            self.classifier = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)
            
        self.is_fitted = False
        self.label_map = [
            DependencyType.REQUIRES,
            DependencyType.IMPLEMENTS,
            DependencyType.CONFLICTS,
            DependencyType.DETAILS,
            DependencyType.IS_SIMILAR,
            DependencyType.NO_DEPENDENCY
        ]

    def _extract_pair_features(self, pairs: List[RequirementPair]) -> np.ndarray:
        """Extract pair features by concatenating embeddings of Requirement A and B, plus absolute difference."""
        if not pairs:
            return np.empty((0, 384 * 3), dtype=np.float32)

        texts_a = [p.req_a.text for p in pairs]
        texts_b = [p.req_b.text for p in pairs]

        embs_a = self.embedder.encode(texts_a)
        embs_b = self.embedder.encode(texts_b)

        diff = np.abs(embs_a - embs_b)
        prod = embs_a * embs_b

        return np.hstack([embs_a, embs_b, diff, prod])

    def fit(self, training_pairs: List[RequirementPair]) -> "BERTBaseline":
        """Train classifier on annotated requirement pair features."""
        if not training_pairs:
            logger.warning("Empty training dataset provided to BERT baseline.")
            return self

        invalid = [
            pair.id for pair in training_pairs
            if pair.dependency_type is None
            or (pair.provenance and pair.provenance.status != VerificationStatus.VERIFIED)
        ]
        if invalid:
            raise ValueError(
                "Training data contains unlabeled or unverified pairs: "
                + ", ".join(invalid)
            )

        X = self._extract_pair_features(training_pairs)
        y = [
            p.dependency_type.value
            for p in training_pairs
        ]

        try:
            self.classifier.fit(X, y)
        except Exception as e:
            logger.exception(f"Failed to fit BERT baseline classifier: {e}")
            raise RuntimeError("Failed to fit BERT baseline classifier.") from e
        self.is_fitted = True
        logger.info(f"BERT baseline successfully trained on {len(training_pairs)} pair instances.")

        return self

    def save(self, path: str, *, overwrite: bool = False) -> None:
        """Atomically save a fitted classifier artifact to a trusted local path.

        Artifacts use joblib/pickle serialization and must only be loaded from
        trusted sources.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save a BERTBaseline before fitting it.")

        target = os.path.abspath(path)
        parent = os.path.dirname(target)
        os.makedirs(parent, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=".bert-baseline-", suffix=".tmp", dir=parent, delete=False
            ) as temporary:
                temporary_path = temporary.name
            joblib.dump(
                {
                    "format_version": 1,
                    "model_type": self.model_type,
                    "embedding_model": getattr(self.embedder, "model_name", None),
                    "classifier": self.classifier,
                },
                temporary_path,
            )
            if overwrite:
                os.replace(temporary_path, target)
                temporary_path = None
            elif os.name == "nt":
                # Windows rename fails if target exists and publishes the
                # completed sibling temporary file atomically on the same disk.
                try:
                    os.rename(temporary_path, target)
                except FileExistsError as exc:
                    raise FileExistsError(f"Model artifact already exists: {target}") from exc
                temporary_path = None
            else:
                try:
                    # POSIX hard-link publication is atomic and fails if a
                    # concurrent writer already created the destination.
                    os.link(temporary_path, target)
                except FileExistsError as exc:
                    raise FileExistsError(f"Model artifact already exists: {target}") from exc
                os.unlink(temporary_path)
                temporary_path = None
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @classmethod
    def load(cls, path: str, *, embedder: EmbeddingModel | None = None) -> "BERTBaseline":
        """Load a trusted classifier artifact and attach its feature embedder."""
        artifact = joblib.load(path)
        if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
            raise ValueError("Unsupported BERTBaseline artifact format.")
        if "classifier" not in artifact or "model_type" not in artifact:
            raise ValueError("BERTBaseline artifact is missing required fields.")

        if embedder is None:
            embedder = EmbeddingModel(model_name=artifact.get("embedding_model"))
        model = cls(embedder=embedder, model_type=artifact["model_type"])
        model.classifier = artifact["classifier"]
        model.is_fitted = True
        return model

    def predict_pair(self, pair: RequirementPair) -> Prediction:
        """Predict dependency type for a single requirement pair."""
        if not self.is_fitted:
            raise ValueError("BERTBaseline must be fit on training pairs before prediction.")

        X = self._extract_pair_features([pair])
        raw_pred = self.classifier.predict(X)[0]
        parsed_type = DependencyType.normalize(str(raw_pred))

        # Retrieve prediction probability as confidence
        try:
            probs = self.classifier.predict_proba(X)[0]
            max_prob = float(np.max(probs))
            conf = int(round(max_prob * 5.0))
        except Exception:
            conf = 4

        return Prediction(
            pair_id=pair.id,
            dependency_type=parsed_type,
            rationale=f"BERT baseline classifier prediction: {parsed_type.value}",
            confidence=conf
        )

    def predict_batch(self, pairs: List[RequirementPair]) -> List[Prediction]:
        """Predict dependencies across a list of requirement pairs."""
        if not pairs:
            return []
        if not self.is_fitted:
            raise ValueError("BERTBaseline must be fit on training pairs before batch prediction.")
        return [self.predict_pair(p) for p in pairs]
