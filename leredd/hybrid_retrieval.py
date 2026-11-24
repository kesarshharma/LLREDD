"""Hybrid Retrieval module using Reciprocal Rank Fusion (RRF) over BM25 and SBERT.

Combines lexical keyword matching (BM25Plus) and dense semantic similarity (SBERT)
without modifying arbitrary score scales or breaking existing interfaces.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from loguru import logger

from leredd.bm25_retrieval import BM25Retriever
from leredd.config import LEREDDConfig, get_config
from leredd.data_models import DependencyType, Requirement, RequirementPair
from leredd.retrieval import ExampleRetriever


def get_candidate_id(item: Any) -> str:
    """Derive a stable, deterministic unique identifier for candidate items.

    Preserves object metadata, labels, and provenance.
    """
    if isinstance(item, RequirementPair):
        if item.id:
            return str(item.id)
        id_a = getattr(item.req_a, "id", item.req_a.text)
        id_b = getattr(item.req_b, "id", item.req_b.text)
        return f"PAIR_{id_a}_{id_b}"
    elif isinstance(item, Requirement):
        return str(item.id) if item.id else item.text
    elif hasattr(item, "id") and getattr(item, "id"):
        return str(getattr(item, "id"))
    else:
        return str(item)


class HybridRetriever:
    """Hybrid Retriever combining SBERT dense embeddings and BM25 lexical matching via RRF."""

    def __init__(
        self,
        dense_retriever: Optional[ExampleRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        metadata_weight: float = 1.0,
        config: Optional[LEREDDConfig] = None,
    ):
        self.config = config or get_config()
        self.rrf_k = rrf_k or self.config.rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.metadata_weight = metadata_weight

        self._dense_retriever = dense_retriever
        self._bm25_retriever = bm25_retriever

    @property
    def dense_retriever(self) -> ExampleRetriever:
        """Lazy-instantiated dense SBERT retriever."""
        if self._dense_retriever is None:
            self._dense_retriever = ExampleRetriever(config=self.config)
        return self._dense_retriever

    @property
    def bm25_retriever(self) -> BM25Retriever:
        """Lazy-instantiated BM25 lexical retriever."""
        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever(config=self.config)
        return self._bm25_retriever

    @staticmethod
    def reciprocal_rank_fusion(
        rankings: List[List[Any]],
        weights: Optional[List[float]] = None,
        rrf_k: int = 60,
        top_k: int = 5,
        id_fn: Optional[Callable[[Any], str]] = None,
    ) -> List[Tuple[float, Any]]:
        """Perform Reciprocal Rank Fusion (RRF) over multiple ranked candidate lists.

        Formula:
            RRF_Score(d) = sum_m [ w_m / (rrf_k + r(m, d)) ]

        where r(m, d) is the 1-indexed rank of candidate d in ranker m.

        Features:
        - Deterministic candidate deduplication via id_fn.
        - Preserves original candidate objects and metadata.
        - Stable tie-breaking ordering by (-rrf_score, item_id, insertion_idx).
        - Handles candidates appearing in 1 or all rankers.
        """
        if top_k <= 0 or not rankings:
            return []

        id_mapper = id_fn or get_candidate_id
        weights = weights or [1.0] * len(rankings)

        scores: Dict[str, float] = {}
        candidate_map: Dict[str, Any] = {}
        first_seen_idx: Dict[str, int] = {}

        global_counter = 0
        for ranker_idx, ranking in enumerate(rankings):
            w = weights[ranker_idx] if ranker_idx < len(weights) else 1.0
            for rank_0, item in enumerate(ranking):
                item_id = id_mapper(item)
                rank_1 = rank_0 + 1  # 1-indexed rank

                rrf_term = w / (rrf_k + rank_1)
                scores[item_id] = scores.get(item_id, 0.0) + rrf_term

                if item_id not in candidate_map:
                    candidate_map[item_id] = item
                    first_seen_idx[item_id] = global_counter
                    global_counter += 1

        # Build list of (fused_score, item, item_id, insertion_idx)
        fused = [
            (scores[item_id], candidate_map[item_id], item_id, first_seen_idx[item_id])
            for item_id in scores
        ]

        # Stable sort: score descending (-x[0]), item_id ascending (x[2]), insertion_idx (x[3])
        fused.sort(key=lambda x: (-x[0], x[2], x[3]))

        return [(score, item) for score, item, _, _ in fused[:top_k]]

    def retrieve_example_pairs(
        self,
        target_pair: RequirementPair,
        annotated_pool: List[RequirementPair],
        k: int | None = None,
    ) -> Dict[DependencyType, List[RequirementPair]]:
        """Retrieve top-k hybrid example pairs per DependencyType using RRF fusion over SBERT and BM25.

        Interface compatible with ExampleRetriever.retrieve and BM25Retriever.retrieve_example_pairs.
        Enforces strict corpus isolation and target self-exclusion.
        """
        k = k if k is not None else self.config.k_examples
        if k <= 0 or not annotated_pool:
            return {}

        # 1. Get dense SBERT rankings per DependencyType
        dense_results = self.dense_retriever.retrieve(
            target_pair, annotated_pool, k=len(annotated_pool)
        )

        # 2. Get BM25 lexical rankings per DependencyType
        bm25_results = self.bm25_retriever.retrieve_example_pairs(
            target_pair, candidate_pool=annotated_pool, k=len(annotated_pool)
        )

        fused_examples: Dict[DependencyType, List[RequirementPair]] = {}

        for dep_type in DependencyType:
            dense_list = dense_results.get(dep_type, [])
            bm25_list = bm25_results.get(dep_type, [])

            # Filter out target self-matches from both lists
            def filter_self(lst: List[RequirementPair]) -> List[RequirementPair]:
                return [
                    p for p in lst
                    if not (
                        p.id == target_pair.id
                        or (p.req_a.text == target_pair.req_a.text and p.req_b.text == target_pair.req_b.text)
                    )
                ]

            clean_dense = filter_self(dense_list)
            clean_bm25 = filter_self(bm25_list)

            metadata_ranking = self._metadata_ranking(
                target_pair,
                [pair for pair in annotated_pool if pair.dependency_type == dep_type],
                filter_fn=filter_self,
            )

            if not clean_dense and not clean_bm25 and not metadata_ranking:
                fused_examples[dep_type] = []
                continue

            rankings = [clean_dense, clean_bm25]
            weights = [self.dense_weight, self.bm25_weight]
            if metadata_ranking and self.metadata_weight > 0:
                rankings.append(metadata_ranking)
                weights.append(self.metadata_weight)

            # Fuse dense, lexical, and available structured metadata rankings.
            fused_tuples = self.reciprocal_rank_fusion(
                rankings=rankings,
                weights=weights,
                rrf_k=self.rrf_k,
                top_k=k,
                id_fn=get_candidate_id,
            )

            fused_examples[dep_type] = [item for _, item in fused_tuples]

        return fused_examples

    @staticmethod
    def _metadata_ranking(
        target_pair: RequirementPair,
        candidates: List[RequirementPair],
        filter_fn: Optional[Callable[[List[RequirementPair]], List[RequirementPair]]] = None,
    ) -> List[RequirementPair]:
        """Rank pairs by exact component/subsystem/system metadata overlap.

        Metadata is a supplemental ranking signal: it is omitted when the target
        has no usable metadata or no candidate shares a value. Source document,
        requirement identifiers, and free text are intentionally not treated as
        metadata so they cannot leak identity or duplicate lexical retrieval.
        """
        fields = ("component", "subsystem", "system")

        def values(pair: RequirementPair) -> set[tuple[str, str]]:
            result = set()
            for requirement in (pair.req_a, pair.req_b):
                for field in fields:
                    raw = getattr(requirement, field, None)
                    if not isinstance(raw, str) or not raw.strip():
                        continue
                    value = " ".join(raw.casefold().split())
                    if field == "system" and value == "genericsystem":
                        continue
                    result.add((field, value))
            return result

        target_values = values(target_pair)
        if not target_values:
            return []

        eligible = filter_fn(candidates) if filter_fn else candidates
        scored = [(len(target_values & values(pair)), pair) for pair in eligible]
        if not any(score for score, _ in scored):
            return []
        scored.sort(key=lambda item: (-item[0], get_candidate_id(item[1])))
        return [pair for score, pair in scored if score > 0]

    def retrieve(
        self,
        target_pair: RequirementPair,
        annotated_pool: List[RequirementPair],
        k: int = 5,
    ) -> List[Tuple[float, RequirementPair]]:
        """Retrieve overall top-k hybrid requirement pair candidates across all dependency types."""
        if k <= 0 or not annotated_pool:
            return []

        examples_by_type = self.retrieve_example_pairs(target_pair, annotated_pool, k=k)
        all_candidates: List[RequirementPair] = []
        for pair_list in examples_by_type.values():
            all_candidates.extend(pair_list)

        # Return RRF scored candidates
        dense_list = [
            p for sub in self.dense_retriever.retrieve(target_pair, annotated_pool, k=len(annotated_pool)).values() for p in sub
        ]
        bm25_list = [
            p for sub in self.bm25_retriever.retrieve_example_pairs(target_pair, candidate_pool=annotated_pool, k=len(annotated_pool)).values() for p in sub
        ]

        metadata_ranking = self._metadata_ranking(
            target_pair,
            annotated_pool,
            filter_fn=lambda pairs: [
                pair for pair in pairs
                if pair.id != target_pair.id
                and not (
                    pair.req_a.text == target_pair.req_a.text
                    and pair.req_b.text == target_pair.req_b.text
                )
            ],
        )
        rankings = [dense_list, bm25_list]
        weights = [self.dense_weight, self.bm25_weight]
        if metadata_ranking and self.metadata_weight > 0:
            rankings.append(metadata_ranking)
            weights.append(self.metadata_weight)

        return self.reciprocal_rank_fusion(
            rankings=rankings,
            weights=weights,
            rrf_k=self.rrf_k,
            top_k=k,
            id_fn=get_candidate_id,
        )
