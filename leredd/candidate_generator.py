"""Candidate Requirement Pair Generation module for Semi-Automated Dataset Generation (Phase 4).

Provides reproducible candidate pair generation using:
1. ALL_PAIRS: Exhaustive n(n-1)/2 distinct unordered requirement pairs.
2. BM25 / DENSE / HYBRID retrieval-based candidate generation.

Features:
- Canonical unordered pair deduplication ((A,B) and (B,A) deduplicated).
- Target self-exclusion (prevents (A,A) self-pairs).
- Explicit candidate limits with truncation reporting in audit manifest.
- Pure corpus isolation (operates strictly on explicit input requirement collections).
- Full audit manifest generation for reproducibility.
"""

import datetime
import json
import os
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger
from pydantic import BaseModel, Field

from leredd.bm25_retrieval import BM25Retriever
from leredd.config import get_config
from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    DependencyType,
    Requirement,
    RequirementPair,
    VerificationStatus,
)
from leredd.synthetic import is_synthetic, synthetic_source_identity
from leredd.hybrid_retrieval import HybridRetriever
from leredd.retrieval import RequirementRetriever


class GenerationStrategy(str, Enum):
    """Supported candidate generation strategies."""
    ALL_PAIRS = "all_pairs"
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


class CandidateGenerationManifest(BaseModel):
    """Audit manifest detailing candidate generation run configuration and statistics."""
    run_id: str = Field(default_factory=lambda: f"GEN-{uuid.uuid4().hex[:8].upper()}")
    source_identity: str = Field(description="SRS file path, system identifier, or dataset source")
    input_requirement_count: int = Field(description="Number of input requirements evaluated")
    strategy: GenerationStrategy = Field(description="Generation strategy used")
    top_k: Optional[int] = Field(default=None, description="Top-k retrieval parameter if retrieval strategy used")
    max_candidate_limit: Optional[int] = Field(default=None, description="Configured maximum candidate limit if set")
    generated_count: int = Field(description="Actual number of candidate pairs generated")
    pair_universe_count: int = Field(
        default=0,
        description="Number of possible unordered pairs among unique input requirements",
    )
    pair_space_recall: Optional[float] = Field(
        default=None,
        description=(
            "Fraction of the complete unordered pair universe emitted by this run; "
            "this is candidate coverage, not recall of labeled dependencies"
        ),
    )
    uncovered_pair_count: int = Field(
        default=0,
        description="Possible unordered pairs not present in this run's candidate output",
    )
    is_exhaustive: bool = Field(description="True if generation produced all candidate pairs without truncation")
    truncated_count: int = Field(default=0, description="Number of candidate pairs omitted due to safety limits")
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


def get_canonical_pair_key(req_a: Requirement, req_b: Requirement) -> str:
    """Derive canonical string key for an unordered requirement pair.

    Ensures (A, B) and (B, A) generate identical keys regardless of order.
    """
    id_a = (req_a.id or req_a.text.strip().lower()).upper()
    id_b = (req_b.id or req_b.text.strip().lower()).upper()
    return f"{id_a}|||{id_b}" if id_a <= id_b else f"{id_b}|||{id_a}"


class CandidateGenerator:
    """Generates candidate requirement pairs for semi-automated dataset labeling."""

    @staticmethod
    def generate_candidates(
        requirements: List[Requirement],
        strategy: GenerationStrategy = GenerationStrategy.ALL_PAIRS,
        source_identity: str = "SRS_Document",
        top_k: Optional[int] = 5,
        max_candidates: Optional[int] = None,
        retriever: Optional[Any] = None,
    ) -> Tuple[List[RequirementPair], CandidateGenerationManifest]:
        """Generate candidate requirement pairs using the specified strategy.

        Returns:
            Tuple of (List[RequirementPair], CandidateGenerationManifest)

        The manifest's pair_space_recall measures coverage against all possible
        unordered pairs in the deduplicated input requirement set. It is not a
        positive-class recall metric because no dependency labels are consulted.
        """
        if strategy != GenerationStrategy.ALL_PAIRS and (
            not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer for retrieval strategies.")
        if not requirements:
            logger.warning("Empty requirement list provided to CandidateGenerator.")
            manifest = CandidateGenerationManifest(
                source_identity=source_identity,
                input_requirement_count=0,
                strategy=strategy,
                top_k=top_k,
                max_candidate_limit=max_candidates,
                generated_count=0,
                pair_universe_count=0,
                pair_space_recall=None,
                uncovered_pair_count=0,
                is_exhaustive=True,
                truncated_count=0,
            )
            return [], manifest

        # Deduplicate requirements by ID/text first
        unique_reqs: List[Requirement] = []
        seen_req_ids: Set[str] = set()
        for r in requirements:
            rid = (r.id or r.text.strip().lower()).upper()
            if rid not in seen_req_ids:
                seen_req_ids.add(rid)
                unique_reqs.append(r)

        n = len(unique_reqs)

        if strategy == GenerationStrategy.ALL_PAIRS:
            raw_candidates = CandidateGenerator._generate_all_pairs(unique_reqs, source_identity)
            theoretical_total = (n * (n - 1)) // 2
        else:
            raw_candidates = CandidateGenerator._generate_retrieval_candidates(
                unique_reqs, strategy=strategy, top_k=top_k, source_identity=source_identity, retriever=retriever
            )
            theoretical_total = (n * (n - 1)) // 2

        # Truncate if max_candidates limit is set
        truncated_count = 0
        is_exhaustive = True

        if max_candidates is not None and max_candidates > 0 and len(raw_candidates) > max_candidates:
            truncated_count = len(raw_candidates) - max_candidates
            raw_candidates = raw_candidates[:max_candidates]
            is_exhaustive = False
            logger.warning(
                f"Candidate generation truncated at max_candidates={max_candidates} "
                f"({truncated_count} candidates omitted)."
            )

        candidate_keys = set()
        for pair in raw_candidates:
            pair_req_a_id = (pair.req_a.id or pair.req_a.text.strip().lower()).upper()
            pair_req_b_id = (pair.req_b.id or pair.req_b.text.strip().lower()).upper()
            if (
                pair_req_a_id in seen_req_ids
                and pair_req_b_id in seen_req_ids
                and pair_req_a_id != pair_req_b_id
            ):
                candidate_keys.add(get_canonical_pair_key(pair.req_a, pair.req_b))

        covered_pair_count = len(candidate_keys)
        manifest = CandidateGenerationManifest(
            source_identity=source_identity,
            input_requirement_count=n,
            strategy=strategy,
            top_k=top_k if strategy != GenerationStrategy.ALL_PAIRS else None,
            max_candidate_limit=max_candidates,
            generated_count=len(raw_candidates),
            pair_universe_count=theoretical_total,
            pair_space_recall=(covered_pair_count / theoretical_total if theoretical_total else None),
            uncovered_pair_count=theoretical_total - covered_pair_count,
            is_exhaustive=(
                is_exhaustive
                and len(raw_candidates) == theoretical_total
            ),
            truncated_count=truncated_count,
            metadata={
                "unique_requirement_count": n,
                "theoretical_all_pairs_total": (n * (n - 1)) // 2,
            }
        )

        logger.info(
            f"Generated {len(raw_candidates)} candidate pairs for {n} requirements using strategy '{strategy.value}'."
        )
        return raw_candidates, manifest

    @staticmethod
    def _generate_all_pairs(
        requirements: List[Requirement],
        source_identity: str
    ) -> List[RequirementPair]:
        """Exhaustively generate all n(n-1)/2 unique unordered requirement pairs."""
        n = len(requirements)
        candidates: List[RequirementPair] = []
        pair_counter = 1

        for i in range(n):
            for j in range(i + 1, n):
                req_a = requirements[i]
                req_b = requirements[j]

                # Target self-exclusion check
                id_a = req_a.id or req_a.text
                id_b = req_b.id or req_b.text
                if id_a == id_b:
                    continue

                prov = AnnotationProvenance(
                    source=AnnotationSource.CANDIDATE_GENERATOR,
                    status=VerificationStatus.PROVISIONAL,
                    annotator_id="candidate_generator_all_pairs",
                    evidence_text=f"Exhaustive candidate generation from {source_identity}"
                )
                synthetic = is_synthetic(req_a) or is_synthetic(req_b)
                origin = synthetic_source_identity(req_a, req_b) if synthetic else None
                prov.synthetic = synthetic
                prov.source_identity = origin

                pair_id = f"CAND-{source_identity.upper()}-{pair_counter:05d}"
                candidates.append(
                    RequirementPair(
                        id=pair_id,
                        req_a=req_a,
                        req_b=req_b,
                        dependency_type=None,  # Explicitly unlabeled candidate
                        provenance=prov,
                        synthetic=synthetic,
                        source_identity=origin,
                    )
                )
                pair_counter += 1

        return candidates

    @staticmethod
    def _generate_retrieval_candidates(
        requirements: List[Requirement],
        strategy: GenerationStrategy,
        top_k: int,
        source_identity: str,
        retriever: Optional[Any] = None
    ) -> List[RequirementPair]:
        """Generate candidate pairs by retrieving top-k matching requirements for each requirement."""
        if strategy not in {
            GenerationStrategy.BM25,
            GenerationStrategy.DENSE,
            GenerationStrategy.HYBRID,
        }:
            raise ValueError(f"Unsupported retrieval strategy: {strategy}")

        bm25_retriever = (
            BM25Retriever(requirements)
            if strategy in (GenerationStrategy.BM25, GenerationStrategy.HYBRID)
            else None
        )
        if strategy == GenerationStrategy.DENSE:
            dense_retriever = retriever or RequirementRetriever(requirements)
        elif strategy == GenerationStrategy.HYBRID:
            # HybridRetriever is accepted as an RRF configuration carrier; its
            # pair-level dense retriever cannot rank individual Requirements.
            dense_retriever = (
                RequirementRetriever(requirements)
                if isinstance(retriever, HybridRetriever)
                else (retriever or RequirementRetriever(requirements))
            )
        else:
            dense_retriever = None
        rrf_k = get_config().rrf_k
        dense_weight = 1.0
        bm25_weight = 1.0
        if isinstance(retriever, HybridRetriever):
            rrf_k = retriever.rrf_k
            dense_weight = retriever.dense_weight
            bm25_weight = retriever.bm25_weight

        seen_canonical_keys: Set[str] = set()
        candidates: List[RequirementPair] = []
        pair_counter = 1

        for req in requirements:
            if strategy == GenerationStrategy.BM25:
                scored_results = bm25_retriever.retrieve(req, k=top_k + 1)
            elif strategy == GenerationStrategy.DENSE:
                scored_results = dense_retriever.retrieve(req, k=top_k + 1)
            else:
                dense_results = dense_retriever.retrieve(req, k=top_k + 1)
                bm25_results = bm25_retriever.retrieve(req, k=top_k + 1)
                scored_results = HybridRetriever.reciprocal_rank_fusion(
                    rankings=[
                        [candidate for _, candidate in dense_results],
                        [candidate for _, candidate in bm25_results],
                    ],
                    weights=[dense_weight, bm25_weight],
                    rrf_k=rrf_k,
                    top_k=top_k + 1,
                )

            for score, cand_req in scored_results:
                if not isinstance(cand_req, Requirement):
                    continue

                # Exclude self-matches
                if cand_req.id == req.id or cand_req.text == req.text:
                    continue

                canon_key = get_canonical_pair_key(req, cand_req)
                if canon_key in seen_canonical_keys:
                    continue
                seen_canonical_keys.add(canon_key)

                prov = AnnotationProvenance(
                    source=AnnotationSource.CANDIDATE_GENERATOR,
                    status=VerificationStatus.PROVISIONAL,
                    annotator_id=f"candidate_generator_{strategy.value}",
                    evidence_text=f"Retrieved via {strategy.value} strategy (top_{top_k}, score={score:.4f})"
                )
                synthetic = is_synthetic(req) or is_synthetic(cand_req)
                origin = synthetic_source_identity(req, cand_req) if synthetic else None
                prov.synthetic = synthetic
                prov.source_identity = origin

                pair_id = f"CAND-{source_identity.upper()}-{pair_counter:05d}"
                candidates.append(
                    RequirementPair(
                        id=pair_id,
                        req_a=req,
                        req_b=cand_req,
                        dependency_type=None,  # Unlabeled
                        provenance=prov,
                        synthetic=synthetic,
                        source_identity=origin,
                    )
                )
                pair_counter += 1

        return candidates

    @staticmethod
    def save_candidates(
        candidates: List[RequirementPair],
        manifest: CandidateGenerationManifest,
        output_pairs_file: str,
        output_manifest_file: Optional[str] = None
    ) -> None:
        """Save generated candidate pairs and audit manifest to JSON files."""
        os.makedirs(os.path.dirname(os.path.abspath(output_pairs_file)), exist_ok=True)

        pairs_data = [p.model_dump(mode="json") for p in candidates]
        with open(output_pairs_file, "w", encoding="utf-8") as f:
            json.dump(pairs_data, f, indent=2)

        if output_manifest_file is None:
            base, ext = os.path.splitext(output_pairs_file)
            output_manifest_file = f"{base}_manifest.json"

        with open(output_manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(mode="json"), f, indent=2)

        logger.info(
            f"Saved {len(candidates)} candidate pairs to '{output_pairs_file}' "
            f"and manifest to '{output_manifest_file}'."
        )
