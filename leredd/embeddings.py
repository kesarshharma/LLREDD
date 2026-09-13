"""SBERT Embeddings wrapper with caching and lightweight fallback."""

import os
import hashlib
import pickle
import numpy as np
from typing import List
from loguru import logger
from leredd.config import get_config


class EmbeddingModel:
    """Embedding model wrapper supporting SBERT sentence-transformers and caching."""

    def __init__(self, model_name: str | None = None, cache_dir: str | None = None):
        config = get_config()
        self.model_name = model_name or config.embedding_model
        self.cache_dir = cache_dir or config.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self._model = None
        self._fallback_tfidf = None
        self._init_model()

    def _init_model(self) -> None:
        """Lazily initialize sentence-transformers model or fallback vectorizer."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer ({e}). Falling back to TF-IDF vectorizer.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._fallback_tfidf = TfidfVectorizer(ngram_range=(1, 2))

    def _get_cache_key(self, text: str) -> str:
        """Compute MD5 hash for a string for embedding cache lookups."""
        hash_str = f"{self.model_name}:{text}"
        return hashlib.md5(hash_str.encode("utf-8")).hexdigest()

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text strings into an embedding matrix (N, D)."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings: List[np.ndarray | None] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check disk cache first
        for idx, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cache_file = os.path.join(self.cache_dir, f"emb_{cache_key}.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        embeddings[idx] = pickle.load(f)
                    continue
                except Exception:
                    pass
            uncached_indices.append(idx)
            uncached_texts.append(text)

        # Compute embeddings for uncached texts
        if uncached_texts:
            if self._model is not None:
                new_vectors = self._model.encode(
                    uncached_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
            elif self._fallback_tfidf is not None:
                # Fallback implementation
                try:
                    fitted = self._fallback_tfidf.fit_transform(uncached_texts).toarray()
                    # Pad to 384 dim if necessary for vector shape consistency
                    if fitted.shape[1] < 384:
                        pad = np.zeros((fitted.shape[0], 384 - fitted.shape[1]), dtype=np.float32)
                        new_vectors = np.hstack([fitted, pad])
                    else:
                        new_vectors = fitted[:, :384]
                except Exception:
                    # Deterministic hash fallback vector
                    new_vectors = []
                    for t in uncached_texts:
                        seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % (2**32)
                        np.random.seed(seed)
                        vec = np.random.randn(384).astype(np.float32)
                        new_vectors.append(vec / (np.linalg.norm(vec) + 1e-8))
                    new_vectors = np.array(new_vectors)
            else:
                new_vectors = np.zeros((len(uncached_texts), 384), dtype=np.float32)

            # Ensure 2D float32
            new_vectors = np.asarray(new_vectors, dtype=np.float32)
            if new_vectors.ndim == 1:
                new_vectors = new_vectors.reshape(1, -1)

            # Save to disk cache
            for i, orig_idx in enumerate(uncached_indices):
                vec = new_vectors[i]
                embeddings[orig_idx] = vec
                cache_key = self._get_cache_key(uncached_texts[i])
                cache_file = os.path.join(self.cache_dir, f"emb_{cache_key}.pkl")
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump(vec, f)
                except Exception as e:
                    logger.debug(f"Failed to cache embedding: {e}")

        # Combine results into single numpy array
        res_list = [v if v is not None else np.zeros((384,), dtype=np.float32) for v in embeddings]
        return np.vstack(res_list)
