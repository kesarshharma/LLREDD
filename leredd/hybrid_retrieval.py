"""Hybrid Retrieval module using Reciprocal Rank Fusion (RRF) over BM25, SBERT, and Metadata.

Combines lexical keyword matching (BM25Plus), dense semantic similarity (SBERT), and
optional structured metadata similarity without modifying arbitrary score scales or
breaking existing interfaces.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from leredd.bm25_retrieval import BM25Retriever
from leredd.config import LEREDDConfig, get_config
from leredd.data_models import DependencyType, Requirement, RequirementPair
from leredd.retrieval import ExampleRetriever, is_eligible_retrieval_example, is_equivalent_pair

_EXCLUDED_METADATA_DICT_KEYS = {
    "id",
    "req_id",
    "requirement_id",
    "text",
    "statement",
    "source_location",
    "dependency_type",
    "label",
}


def get_candidate_id(item: Any) -> str:
    """Derive a stable, deterministic unique identifier for candidate items."""
    if isinstance(item, RequirementPair):
        if item.id:
            return str(item.id)
        id_a = getattr(item.req_a, "id", item.req_a.text)
        id_b = getattr(item.req_b, "id", item.req_b.text)
        return f"PAIR_{id_a}_{id_b}"
    if isinstance(item, Requirement):
        return str(item.id) if item.id else item.text
    if hasattr(item, "id") and item.id:
        return str(item.id)
    return str(item)


class HybridRetriever:
    """Hybrid Retriever combining SBERT dense embeddings, BM25 lexical matching, and metadata via RRF."""

    def __init__(
        self,
        dense_retriever: Optional[ExampleRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        rrf_k: Optional[int] = None,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        metadata_weight: Optional[float] = None,
        config: Optional[LEREDDConfig] = None,
    ):
        self.config = config or get_config()
        resolved_rrf_k = self.config.rrf_k if rrf_k is None else rrf_k
        if isinstance(resolved_rrf_k, bool) or not isinstance(resolved_rrf_k, int) or resolved_rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer >= 1.")
        self.rrf_k = resolved_rrf_k

        self.dense_weight = float(
            self.config.dense_weight if dense_weight is None else dense_weight
        )
        self.bm25_weight = float(
            self.config.bm25_weight if bm25_weight is None else bm25_weight
        )
        self.metadata_weight = float(
            self.config.metadata_weight if metadata_weight is None else metadata_weight
        )
        if any(w < 0.0 for w in (self.dense_weight, self.bm25_weight, self.metadata_weight)):
            raise ValueError("Retrieval weights must be non-negative.")

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
        """
        if top_k <= 0 or not rankings:
            return []

        id_mapper = id_fn or get_candidate_id
        resolved_weights = [1.0] * len(rankings) if weights is None else list(weights)

        scores: Dict[str, float] = {}
        candidate_map: Dict[str, Any] = {}
        first_seen_idx: Dict[str, int] = {}

        global_counter = 0
        for ranker_idx, ranking in enumerate(rankings):
            w = resolved_weights[ranker_idx] if ranker_idx < len(resolved_weights) else 1.0
            if w <= 0.0:
                continue
            seen_in_ranker: set[str] = set()
            rank_1 = 0
            for item in ranking:
                item_id = id_mapper(item)
                if item_id in seen_in_ranker:
                    continue
                seen_in_ranker.add(item_id)
                rank_1 += 1

                rrf_term = w / (rrf_k + rank_1)
                scores[item_id] = scores.get(item_id, 0.0) + rrf_term

                if item_id not in candidate_map:
                    candidate_map[item_id] = item
                    first_seen_idx[item_id] = global_counter
                    global_counter += 1

        fused = [
            (scores[item_id], candidate_map[item_id], item_id, first_seen_idx[item_id])
            for item_id in scores
        ]

        # Deterministic tie-breaking: score descending, item_id ascending, insertion_idx ascending
        fused.sort(key=lambda x: (-x[0], x[2], x[3]))

        return [(score, item) for score, item, _, _ in fused[:top_k]]

    def retrieve_example_pairs(
        self,
        target_pair: RequirementPair,
        annotated_pool: List[RequirementPair],
        k: int | None = None,
    ) -> Dict[DependencyType, List[RequirementPair]]:
        """Retrieve top-k verified, non-synthetic hybrid example pairs per DependencyType."""
        k = k if k is not None else self.config.k_examples
        if k <= 0 or not annotated_pool:
            return {}

        eligible_pool = [
            pair for pair in annotated_pool
            if is_eligible_retrieval_example(pair, target_pair)
        ]
        if not eligible_pool:
            return {}

        dense_results = (
            self.dense_retriever.retrieve(target_pair, eligible_pool, k=len(eligible_pool))
            if self.dense_weight > 0.0
            else {}
        )
        bm25_results = (
            self.bm25_retriever.retrieve_example_pairs(
                target_pair, candidate_pool=eligible_pool, k=len(eligible_pool)
            )
            if self.bm25_weight > 0.0
            else {}
        )

        def filter_self(lst: List[RequirementPair]) -> List[RequirementPair]:
            return [
                p for p in lst
                if is_eligible_retrieval_example(p, target_pair)
            ]

        fused_examples: Dict[DependencyType, List[RequirementPair]] = {}

        for dep_type in DependencyType:
            clean_dense = filter_self(dense_results.get(dep_type, []))
            clean_bm25 = filter_self(bm25_results.get(dep_type, []))

            metadata_ranking = (
                self._metadata_ranking(
                    target_pair,
                    [pair for pair in eligible_pool if pair.dependency_type == dep_type],
                    filter_fn=filter_self,
                )
                if self.metadata_weight > 0.0
                else []
            )

            rankings: List[List[RequirementPair]] = []
            weights: List[float] = []
            if clean_dense and self.dense_weight > 0.0:
                rankings.append(clean_dense)
                weights.append(self.dense_weight)
            if clean_bm25 and self.bm25_weight > 0.0:
                rankings.append(clean_bm25)
                weights.append(self.bm25_weight)
            if metadata_ranking and self.metadata_weight > 0.0:
                rankings.append(metadata_ranking)
                weights.append(self.metadata_weight)

            if not rankings:
                fused_examples[dep_type] = []
                continue

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
        """Rank pairs by exact component/subsystem/module/system and structured metadata overlap.

        Metadata is a supplemental ranking signal: it is omitted when the target
        has no usable metadata or no candidate shares a value. Source document,
        requirement identifiers, and free text are intentionally not treated as
        metadata so they cannot leak identity or duplicate lexical retrieval.
        """
        fields = ("component", "subsystem", "module", "system")

        def values(pair: RequirementPair) -> set[Tuple[str, str]]:
            result: set[Tuple[str, str]] = set()
            for requirement in (pair.req_a, pair.req_b):
                for field in fields:
                    raw = getattr(requirement, field, None)
                    if not isinstance(raw, str) or not raw.strip():
                        continue
                    value = " ".join(raw.casefold().split())
                    if field == "system" and value == "genericsystem":
                        continue
                    result.add((field, value))

                extra_meta = getattr(requirement, "metadata", None)
                if isinstance(extra_meta, dict):
                    for key, raw_val in extra_meta.items():
                        norm_key = str(key).strip().casefold()
                        if not norm_key or norm_key in _EXCLUDED_METADATA_DICT_KEYS:
                            continue
                        if isinstance(raw_val, (str, int, float)) and not isinstance(raw_val, bool):
                            str_val = " ".join(str(raw_val).casefold().split())
                            if str_val:
                                result.add((f"meta:{norm_key}", str_val))
            return result

        target_values = values(target_pair)
        if not target_values:
            return []

        base_eligible = filter_fn(candidates) if filter_fn else [
            p for p in candidates if not is_equivalent_pair(p, target_pair)
        ]
        eligible = [p for p in base_eligible if is_eligible_retrieval_example(p, target_pair)]

        scored = [
            (len(target_values & values(pair)), get_candidate_id(pair), idx, pair)
            for idx, pair in enumerate(eligible)
        ]
        if not any(score > 0 for score, _, _, _ in scored):
            return []
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [pair for score, _, _, pair in scored if score > 0]

    def retrieve(
        self,
        target_pair: RequirementPair,
        annotated_pool: List[RequirementPair],
        k: int = 5,
    ) -> List[Tuple[float, RequirementPair]]:
        """Retrieve overall top-k verified hybrid requirement pair candidates across all dependency types."""
        if k <= 0 or not annotated_pool:
            return []

        eligible_pool = [
            pair for pair in annotated_pool
            if is_eligible_retrieval_example(pair, target_pair)
        ]
        if not eligible_pool:
            return []

        dense_list = (
            [
                p
                for sub in self.dense_retriever.retrieve(
                    target_pair, eligible_pool, k=len(eligible_pool)
                ).values()
                for p in sub
            ]
            if self.dense_weight > 0.0
            else []
        )
        bm25_list = (
            [
                p
                for sub in self.bm25_retriever.retrieve_example_pairs(
                    target_pair, candidate_pool=eligible_pool, k=len(eligible_pool)
                ).values()
                for p in sub
            ]
            if self.bm25_weight > 0.0
            else []
        )
        metadata_ranking = (
            self._metadata_ranking(target_pair, eligible_pool)
            if self.metadata_weight > 0.0
            else []
        )

        rankings: List[List[RequirementPair]] = []
        weights: List[float] = []
        if dense_list and self.dense_weight > 0.0:
            rankings.append(dense_list)
            weights.append(self.dense_weight)
        if bm25_list and self.bm25_weight > 0.0:
            rankings.append(bm25_list)
            weights.append(self.bm25_weight)
        if metadata_ranking and self.metadata_weight > 0.0:
            rankings.append(metadata_ranking)
            weights.append(self.metadata_weight)

        if not rankings:
            return []

        return self.reciprocal_rank_fusion(
            rankings=rankings,
            weights=weights,
            rrf_k=self.rrf_k,
            top_k=k,
            id_fn=get_candidate_id,
        )
