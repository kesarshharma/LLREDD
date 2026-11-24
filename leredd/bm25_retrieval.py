"""BM25 Lexical Retrieval module for Requirement Dependency Detection.

Provides deterministic BM25Plus lexical retrieval for requirements and requirement pairs.
Complements SBERT dense retrieval without network access or API dependencies.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from rank_bm25 import BM25Okapi, BM25Plus

from leredd.config import LEREDDConfig, get_config
from leredd.data_models import DependencyType, Requirement, RequirementPair
from leredd.retrieval import is_eligible_retrieval_example


class BM25Retriever:
    """Lexical retriever using BM25Plus for requirement statements and pairs.

    Features:
    - Independent lexical retrieval (requires zero network/API access or SBERT models).
    - Supports requirement text strings, Requirement objects, or RequirementPair objects.
    - Deterministic tokenization (lowercasing, punctuation stripping, case normalization).
    - Stable tie-breaking for identical scores.
    - Pure corpus isolation and strict verified/non-synthetic example filtering.
    """

    def __init__(
        self,
        corpus: Optional[List[Union[RequirementPair, Requirement, str]]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        config: Optional[LEREDDConfig] = None,
    ):
        self.config = config or get_config()
        self.k1 = k1
        self.b = b
        self._corpus: List[Any] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

        if corpus is not None and len(corpus) > 0:
            self.fit(corpus)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize input text into clean, lowercased alphanumeric tokens."""
        if not text:
            return []
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def _extract_text(item: Union[RequirementPair, Requirement, str]) -> str:
        """Extract searchable text representation from an item."""
        if isinstance(item, str):
            return item
        if isinstance(item, Requirement):
            return item.text
        if isinstance(item, RequirementPair):
            return f"{item.req_a.text} {item.req_b.text}"
        return str(item)

    def fit(self, corpus: List[Union[RequirementPair, Requirement, str]]) -> None:
        """Build the BM25 index over an explicitly provided corpus."""
        if not corpus:
            logger.warning("Empty corpus provided to BM25Retriever.fit(). Index will be empty.")
            self._corpus = []
            self._tokenized_corpus = []
            self._bm25 = None
            return

        self._corpus = list(corpus)
        self._tokenized_corpus = [
            self.tokenize(self._extract_text(item)) for item in self._corpus
        ]
        self._bm25 = BM25Plus(self._tokenized_corpus, k1=self.k1, b=self.b)
        logger.info(f"Built BM25 index over {len(self._corpus)} corpus items.")

    @property
    def corpus(self) -> List[Any]:
        """Inspectable copy of the indexed corpus."""
        return list(self._corpus)

    @property
    def is_indexed(self) -> bool:
        """True if the BM25 index is built and non-empty."""
        return self._bm25 is not None and len(self._corpus) > 0

    def score(self, query: str, candidate_idx: int) -> float:
        """Calculate BM25 score for a query string against candidate index."""
        if not self.is_indexed or self._bm25 is None or candidate_idx < 0 or candidate_idx >= len(self._corpus):
            return 0.0
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return 0.0

        scores = self._bm25.get_scores(query_tokens)
        return float(scores[candidate_idx])

    def retrieve(
        self,
        query: Union[str, Requirement, RequirementPair],
        k: int = 5,
    ) -> List[Tuple[float, Any]]:
        """Retrieve top-k candidates from the indexed corpus with BM25 scores."""
        if k <= 0 or not self.is_indexed or self._bm25 is None:
            return []

        query_text = self._extract_text(query)
        query_tokens = self.tokenize(query_text)

        if not query_tokens:
            return [(0.0, item) for item in self._corpus[:k]]

        raw_scores = self._bm25.get_scores(query_tokens)

        scored_items = []
        for idx, (score, item) in enumerate(zip(raw_scores, self._corpus)):
            item_id = getattr(item, "id", str(idx))
            scored_items.append((float(score), item, str(item_id), idx))

        scored_items.sort(key=lambda x: (-x[0], x[2], x[3]))

        return [(score, item) for score, item, _, _ in scored_items[:k]]

    def retrieve_example_pairs(
        self,
        target_pair: RequirementPair,
        candidate_pool: Optional[List[RequirementPair]] = None,
        k: int | None = None,
    ) -> Dict[DependencyType, List[RequirementPair]]:
        """Retrieve top-k verified, non-synthetic example pairs per DependencyType using BM25."""
        k = k if k is not None else self.config.k_examples

        if k <= 0:
            return {}

        pool = candidate_pool if candidate_pool is not None else self._corpus
        pairs_pool = [
            item for item in pool
            if is_eligible_retrieval_example(item, target_pair)
        ]

        if not pairs_pool:
            logger.warning("No eligible RequirementPair items in candidate pool for BM25 example retrieval.")
            return {}

        retriever = BM25Retriever(pairs_pool, k1=self.k1, b=self.b, config=self.config)

        query_text = f"{target_pair.req_a.text} {target_pair.req_b.text}"
        query_tokens = self.tokenize(query_text)

        typed_candidates: Dict[DependencyType, List[Tuple[float, str, int, RequirementPair]]] = {
            dep_type: [] for dep_type in DependencyType
        }

        if not query_tokens or retriever._bm25 is None:
            for idx, pair in enumerate(pairs_pool):
                dep_type = pair.dependency_type
                assert dep_type is not None
                typed_candidates[dep_type].append((0.0, str(pair.id), idx, pair))
        else:
            raw_scores = retriever._bm25.get_scores(query_tokens)
            for idx, candidate_pair in enumerate(pairs_pool):
                score = float(raw_scores[idx])
                dep_type = candidate_pair.dependency_type
                assert dep_type is not None
                typed_candidates[dep_type].append((score, str(candidate_pair.id), idx, candidate_pair))

        retrieved_examples: Dict[DependencyType, List[RequirementPair]] = {}
        for dep_type, scored_list in typed_candidates.items():
            scored_list.sort(key=lambda x: (-x[0], x[1], x[2]))
            retrieved_examples[dep_type] = [pair for _, _, _, pair in scored_list[:k]]

        return retrieved_examples
