"""BM25 Lexical Retrieval module for Requirement Dependency Detection.

Provides deterministic BM25Okapi lexical retrieval for requirements and requirement pairs.
Complements SBERT dense retrieval without network access or API dependencies.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from rank_bm25 import BM25Okapi, BM25Plus

from leredd.config import LEREDDConfig, get_config
from leredd.data_models import DependencyType, Requirement, RequirementPair, VerificationStatus


class BM25Retriever:
    """Lexical retriever using BM25Okapi for requirement statements and pairs.

    Features:
    - Independent lexical retrieval (requires zero network/API access or SBERT models).
    - Supports requirement text strings, Requirement objects, or RequirementPair objects.
    - Deterministic tokenization (lowercasing, punctuation stripping, case normalization).
    - Stable tie-breaking for identical scores.
    - Pure corpus isolation (indexes only explicitly provided corpus items).
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
        """Tokenize input text into clean, lowercased alphanumeric tokens.

        Strips punctuation, normalizes case, and handles empty strings gracefully.
        """
        if not text:
            return []
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def _extract_text(item: Union[RequirementPair, Requirement, str]) -> str:
        """Extract searchable text representation from an item."""
        if isinstance(item, str):
            return item
        elif isinstance(item, Requirement):
            return item.text
        elif isinstance(item, RequirementPair):
            return f"{item.req_a.text} {item.req_b.text}"
        else:
            return str(item)

    def fit(self, corpus: List[Union[RequirementPair, Requirement, str]]) -> None:
        """Build the BM25 index over an explicitly provided corpus.

        Does NOT load global datasets or implicit dev/test sets.
        """
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
        if not self.is_indexed or candidate_idx < 0 or candidate_idx >= len(self._corpus):
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
        """Retrieve top-k candidates from the indexed corpus with BM25 scores.

        Returns a list of (score, candidate_item) tuples sorted descending by score,
        with stable tie-breaking on candidate identifier / insertion index.
        """
        if k <= 0 or not self.is_indexed:
            return []

        query_text = self._extract_text(query)
        query_tokens = self.tokenize(query_text)

        if not query_tokens:
            # Empty query returns top k items with 0.0 score deterministically
            res = []
            for idx, item in enumerate(self._corpus[:k]):
                res.append((0.0, item))
            return res

        raw_scores = self._bm25.get_scores(query_tokens)

        # Build list of (score, item, candidate_id, insertion_idx) for stable sorting
        scored_items = []
        for idx, (score, item) in enumerate(zip(raw_scores, self._corpus)):
            item_id = getattr(item, "id", str(idx))
            scored_items.append((float(score), item, str(item_id), idx))

        # Stable sort: score descending (-x[0]), then item_id ascending (x[2]), then idx ascending (x[3])
        scored_items.sort(key=lambda x: (-x[0], x[2], x[3]))

        return [(score, item) for score, item, _, _ in scored_items[:k]]

    def retrieve_example_pairs(
        self,
        target_pair: RequirementPair,
        candidate_pool: Optional[List[RequirementPair]] = None,
        k: int | None = None,
    ) -> Dict[DependencyType, List[RequirementPair]]:
        """Retrieve top-k example pairs for each DependencyType using BM25 pair matching.

        Compatible with ExampleRetriever.retrieve interface.
        If candidate_pool is provided, temporarily indexes candidate_pool for retrieval.
        If candidate_pool is None, uses the currently indexed corpus (if it contains pairs).
        """
        k = k if k is not None else self.config.k_examples

        if k <= 0:
            return {}

        pool = candidate_pool if candidate_pool is not None else self._corpus
        # Filter pool to ensure RequirementPair items only
        pairs_pool = [
            item for item in pool
            if isinstance(item, RequirementPair)
            and item.dependency_type is not None
            and (item.provenance is None or item.provenance.status == VerificationStatus.VERIFIED)
        ]

        if not pairs_pool:
            logger.warning("No RequirementPair items in candidate pool for example retrieval.")
            return {}

        # Create temporary BM25 index for candidate pool if candidate_pool was passed
        if candidate_pool is not None or not self.is_indexed:
            retriever = BM25Retriever(pairs_pool, k1=self.k1, b=self.b, config=self.config)
        else:
            retriever = self

        query_text = f"{target_pair.req_a.text} {target_pair.req_b.text}"
        query_tokens = self.tokenize(query_text)

        typed_candidates: Dict[DependencyType, List[Tuple[float, str, RequirementPair]]] = {
            dep_type: [] for dep_type in DependencyType
        }

        if not query_tokens:
            for pair in pairs_pool:
                if pair.id == target_pair.id or (
                    pair.req_a.text == target_pair.req_a.text
                    and pair.req_b.text == target_pair.req_b.text
                ):
                    continue
                dep_type = pair.dependency_type
                typed_candidates[dep_type].append((0.0, pair.id, pair))
        else:
            raw_scores = retriever._bm25.get_scores(query_tokens)
            for idx, candidate_pair in enumerate(pairs_pool):
                # Exclude exact target pair
                if candidate_pair.id == target_pair.id or (
                    candidate_pair.req_a.text == target_pair.req_a.text
                    and candidate_pair.req_b.text == target_pair.req_b.text
                ):
                    continue

                score = float(raw_scores[idx])
                dep_type = candidate_pair.dependency_type
                typed_candidates[dep_type].append((score, candidate_pair.id, candidate_pair))

        retrieved_examples: Dict[DependencyType, List[RequirementPair]] = {}
        for dep_type, scored_list in typed_candidates.items():
            # Stable sort by score descending, then pair ID
            scored_list.sort(key=lambda x: (-x[0], x[1]))
            retrieved_examples[dep_type] = [pair for _, _, pair in scored_list[:k]]

        return retrieved_examples
