"""TF-IDF & LSA Baseline model for requirement dependency detection."""

from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from leredd.data_models import DependencyType, Prediction, RequirementPair, VerificationStatus
from leredd.synthetic import is_synthetic


class TFIDF_LSABaseline:
    """TF-IDF & Latent Semantic Analysis (LSA) baseline model.
    
    Recommends 'Requires' dependencies based on TF-IDF + LSA cosine similarity thresholding.
    """

    def __init__(
        self,
        n_components: int = 20,
        similarity_threshold: float = 0.45,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.similarity_threshold = similarity_threshold
        self.random_state = random_state
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.is_fitted = False

    def fit(self, training_pairs: List[RequirementPair]) -> "TFIDF_LSABaseline":
        """Fit TF-IDF vectorizer and SVD matrix on training requirements text."""
        invalid = [
            pair.id
            for pair in training_pairs
            if is_synthetic(pair)
            or (pair.provenance and pair.provenance.status != VerificationStatus.VERIFIED)
        ]
        if invalid:
            raise ValueError(
                "Training data contains synthetic or unverified pairs: " + ", ".join(invalid)
            )

        corpus = []
        for pair in training_pairs:
            corpus.append(pair.req_a.text)
            corpus.append(pair.req_b.text)

        if not corpus:
            return self

        tfidf_matrix = self.tfidf.fit_transform(corpus)
        actual_components = min(self.n_components, tfidf_matrix.shape[1] - 1)
        if actual_components > 0:
            self.svd = TruncatedSVD(n_components=actual_components, random_state=self.random_state)
            self.svd.fit(tfidf_matrix)
        self.is_fitted = True
        return self

    def predict_pair(self, pair: RequirementPair) -> Prediction:
        """Predict dependency for a single pair using TF-IDF + LSA similarity threshold."""
        text_a = pair.req_a.text
        text_b = pair.req_b.text

        if not self.is_fitted:
            raise ValueError("TFIDF_LSABaseline must be fit on training pairs before prediction.")

        try:
            vec_a = self.tfidf.transform([text_a])
            vec_b = self.tfidf.transform([text_b])

            if hasattr(self.svd, "components_"):
                lsa_a = self.svd.transform(vec_a)
                lsa_b = self.svd.transform(vec_b)
                sim = float(cosine_similarity(lsa_a, lsa_b)[0][0])
            else:
                sim = float(cosine_similarity(vec_a, vec_b)[0][0])
        except Exception:
            sim = 0.0

        predicted_type = DependencyType.REQUIRES if sim >= self.similarity_threshold else DependencyType.NO_DEPENDENCY

        return Prediction(
            pair_id=pair.id,
            dependency_type=predicted_type,
            rationale=f"TF-IDF & LSA similarity score: {sim:.4f} (threshold: {self.similarity_threshold})",
            confidence=5 if sim >= self.similarity_threshold else 4
        )

    def predict_batch(self, pairs: List[RequirementPair]) -> List[Prediction]:
        """Predict dependencies across a batch of requirement pairs."""
        if not self.is_fitted and pairs:
            raise ValueError("TFIDF_LSABaseline must be fit on training pairs before batch prediction.")
        return [self.predict_pair(p) for p in pairs]
