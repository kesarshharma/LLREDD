"""Dynamic Example Retrieval and Measurable RAG Contextual Retrieval modules."""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Tuple

from loguru import logger
import numpy as np
from pydantic import BaseModel, Field

from leredd.config import LEREDDConfig, get_config
from leredd.data_models import (
    DependencyType,
    Requirement,
    RequirementPair,
    VerificationStatus,
)
from leredd.embeddings import EmbeddingModel
from leredd.synthetic import is_synthetic

_CITED_ID_PATTERN = re.compile(
    r"\b(?:REQ[-_:][A-Za-z0-9:_-]+|[A-Z]{2,10}-\d{1,6}(?:-[A-Za-z0-9]+)*)\b"
)
_REQ_BOUNDARY_PATTERN = re.compile(
    r"(?=^\s*(?:\[?\s*(?:REQ[-_:]|[A-Z]{2,10}-\d+)|(?:\d+\.)+\s+|\*\s+|-\s+))",
    re.MULTILINE,
)


def _norm_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.casefold().split())


def is_equivalent_pair(candidate: RequirementPair, target: RequirementPair) -> bool:
    """Return True if candidate is the target pair or its exact/reversed equivalent."""
    if candidate.id and target.id and candidate.id == target.id:
        return True

    cand_ids = ((candidate.req_a.id or "").strip(), (candidate.req_b.id or "").strip())
    targ_ids = ((target.req_a.id or "").strip(), (target.req_b.id or "").strip())
    if all(cand_ids) and all(targ_ids):
        if cand_ids == targ_ids or cand_ids == (targ_ids[1], targ_ids[0]):
            return True

    cand_texts = (_norm_text(candidate.req_a.text), _norm_text(candidate.req_b.text))
    targ_texts = (_norm_text(target.req_a.text), _norm_text(target.req_b.text))
    if all(cand_texts) and all(targ_texts):
        if cand_texts == targ_texts or cand_texts == (targ_texts[1], targ_texts[0]):
            return True

    return False


def is_eligible_retrieval_example(
    pair: object,
    target_pair: RequirementPair | None = None,
) -> bool:
    """Return True iff pair is a non-synthetic, verified (or legacy) labeled example."""
    if not isinstance(pair, RequirementPair):
        return False
    if pair.dependency_type is None:
        return False
    if is_synthetic(pair):
        return False
    if pair.provenance is not None and pair.provenance.status != VerificationStatus.VERIFIED:
        return False
    if target_pair is not None and is_equivalent_pair(pair, target_pair):
        return False
    return True


def compute_vector_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray,
    metric: str = "euclidean"
) -> float:
    """Compute similarity between two embedding vectors.

    If metric == 'euclidean': sim = 1 / (1 + EuclideanDistance)
    If metric == 'cosine': sim = (vec1 . vec2) / (||vec1|| * ||vec2||)
    """
    if metric.lower() == "euclidean":
        dist = float(np.linalg.norm(vec1 - vec2))
        return 1.0 / (1.0 + dist)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cosine = float(np.dot(vec1, vec2) / (norm1 * norm2))
    return max(0.0, cosine)


def compute_pair_similarity(
    emb_r1: np.ndarray,
    emb_r2: np.ndarray,
    emb_ra: np.ndarray,
    emb_rb: np.ndarray,
    metric: str = "euclidean",
    aggregation: str = "average"
) -> float:
    """Compute pair-to-pair similarity score using Paper Eq. 1 or Eq. 2.

    Target pair: (R1, R2)
    Candidate pair: (Ra, Rb)

    Eq. 1 (Average): Score_avg = (sim(R1, Ra) + sim(R1, Rb) + sim(R2, Ra) + sim(R2, Rb)) / 4
    Eq. 2 (Max-Avg): Score_max_avg = (max(sim(R1, Ra), sim(R1, Rb)) + max(sim(R2, Ra), sim(R2, Rb))) / 2
    """
    sim_1a = compute_vector_similarity(emb_r1, emb_ra, metric)
    sim_1b = compute_vector_similarity(emb_r1, emb_rb, metric)
    sim_2a = compute_vector_similarity(emb_r2, emb_ra, metric)
    sim_2b = compute_vector_similarity(emb_r2, emb_rb, metric)

    if aggregation.lower() == "max":
        return (max(sim_1a, sim_1b) + max(sim_2a, sim_2b)) / 2.0
    return (sim_1a + sim_1b + sim_2a + sim_2b) / 4.0


class ExampleRetriever:
    """Dynamic In-Context Learning Example Retriever."""

    def __init__(self, embedder: EmbeddingModel | None = None, config: LEREDDConfig | None = None):
        self.config = config or get_config()
        self.embedder = embedder or EmbeddingModel(self.config.embedding_model)

    def retrieve(
        self,
        target_pair: RequirementPair,
        annotated_pool: List[RequirementPair],
        k: int | None = None
    ) -> Dict[DependencyType, List[RequirementPair]]:
        """Retrieve top k dynamic in-context example pairs for each dependency type."""
        k = self.config.k_examples if k is None else k
        if k <= 0:
            return {}

        eligible_pool = [
            pair for pair in annotated_pool
            if is_eligible_retrieval_example(pair, target_pair)
        ]
        if not eligible_pool:
            logger.warning("Annotated candidate pool is empty. Returning empty examples dictionary.")
            return {}

        r1_text = target_pair.req_a.text
        r2_text = target_pair.req_b.text
        target_embs = self.embedder.encode([r1_text, r2_text])
        emb_r1, emb_r2 = target_embs[0], target_embs[1]

        candidate_texts: List[str] = []
        for pair in eligible_pool:
            candidate_texts.append(pair.req_a.text)
            candidate_texts.append(pair.req_b.text)

        candidate_embs = self.embedder.encode(candidate_texts)

        typed_candidates: Dict[DependencyType, List[Tuple[float, str, int, RequirementPair]]] = {
            dep_type: [] for dep_type in DependencyType
        }

        for idx, candidate_pair in enumerate(eligible_pool):
            emb_ra = candidate_embs[2 * idx]
            emb_rb = candidate_embs[2 * idx + 1]

            score = compute_pair_similarity(
                emb_r1, emb_r2, emb_ra, emb_rb,
                metric=self.config.similarity_metric,
                aggregation=self.config.aggregation
            )

            dep_type = candidate_pair.dependency_type
            assert dep_type is not None
            typed_candidates[dep_type].append((score, str(candidate_pair.id), idx, candidate_pair))

        retrieved_examples: Dict[DependencyType, List[RequirementPair]] = {}
        for dep_type, scored_list in typed_candidates.items():
            scored_list.sort(key=lambda x: (-x[0], x[1], x[2]))
            retrieved_examples[dep_type] = [pair for _, _, _, pair in scored_list[:k]]

        return retrieved_examples


class RequirementRetriever:
    """Rank a requirement corpus by dense embedding similarity."""

    def __init__(
        self,
        requirements: List[Requirement],
        embedder: EmbeddingModel | None = None,
        config: LEREDDConfig | None = None,
    ):
        self.config = config or get_config()
        self.requirements = list(requirements)
        self.embedder = embedder or EmbeddingModel(self.config.embedding_model)
        self._embeddings = self.embedder.encode(
            [requirement.text for requirement in self.requirements]
        )

    def retrieve(
        self, query: Requirement, k: int = 5
    ) -> List[Tuple[float, Requirement]]:
        """Return the top ``k`` corpus requirements for a requirement query."""
        if k <= 0 or not self.requirements:
            return []

        query_embedding = self.embedder.encode([query.text])[0]
        scored = [
            (
                compute_vector_similarity(
                    query_embedding,
                    embedding,
                    metric=self.config.similarity_metric,
                ),
                requirement,
                index,
            )
            for index, (requirement, embedding) in enumerate(
                zip(self.requirements, self._embeddings)
            )
        ]
        scored.sort(
            key=lambda item: (
                -item[0],
                str(item[1].id or item[1].text),
                item[2],
            )
        )
        return [(score, requirement) for score, requirement, _ in scored[:k]]


class RAGRetrievalDiagnostics(BaseModel):
    """Measurable diagnostics for SRS context chunking and RAG selection (Part 7)."""

    total_chunks: int = Field(default=0, description="Total chunks produced prior to deduplication")
    unique_exact_chunks: int = Field(default=0, description="Distinct normalized chunks")
    exact_duplicate_count: int = Field(default=0, description="Exact duplicate chunks detected")
    near_duplicate_count: int = Field(default=0, description="Near-duplicate chunks detected via token Jaccard")
    duplicate_ratio: float = Field(default=0.0, description="Fraction of raw chunks that are exact or near duplicates")
    candidate_chunks_after_dedup: int = Field(default=0, description="Chunks retained for ranking")
    cited_ids: List[str] = Field(default_factory=list, description="Explicit requirement IDs cited in target pair text")
    cited_id_chunks_available: int = Field(default=0, description="Candidate chunks mentioning a cited requirement ID")
    cited_id_chunks_selected: int = Field(default=0, description="Selected top-k chunks mentioning a cited requirement ID")
    selected_count: int = Field(default=0, description="Final number of context chunks returned")
    irrelevant_selected_count: int = Field(
        default=0,
        description="Selected chunks whose base vector similarity fell below min_similarity",
    )
    selected_scores: List[float] = Field(default_factory=list, description="Final ranking scores for selected chunks")
    chunking_mode: str = Field(default="fixed")
    deduplicated: bool = Field(default=False)
    cite_id_boost: float = Field(default=0.0)


class RAGRetriever:
    """Measurable SRS chunking and RAG context retrieval for cross-system setting."""

    def __init__(self, embedder: EmbeddingModel | None = None, config: LEREDDConfig | None = None):
        self.config = config or get_config()
        self.embedder = embedder or EmbeddingModel(self.config.embedding_model)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    @staticmethod
    def extract_cited_requirement_ids(target_pair: RequirementPair) -> List[str]:
        """Extract explicit requirement identifiers referenced inside target pair prose."""
        found: List[str] = []
        seen: set[str] = set()
        for text in (target_pair.req_a.text or "", target_pair.req_b.text or ""):
            for match in _CITED_ID_PATTERN.findall(text):
                cleaned = match.strip().upper()
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    found.append(cleaned)
        return found

    @classmethod
    def analyze_chunk_duplicates(
        cls,
        chunks: List[str],
        near_duplicate_threshold: float = 0.90,
    ) -> Tuple[List[str], int, int]:
        """Identify exact and near-duplicate chunks deterministically.

        Returns (deduplicated_chunks, exact_duplicate_count, near_duplicate_count).
        """
        if not chunks:
            return [], 0, 0

        kept: List[str] = []
        kept_norm: set[str] = set()
        kept_tokens: List[set[str]] = []
        exact_dups = 0
        near_dups = 0

        for chunk in chunks:
            norm = _norm_text(chunk)
            if norm in kept_norm:
                exact_dups += 1
                continue
            tokens = cls._tokenize(chunk)
            is_near_dup = False
            if tokens and near_duplicate_threshold < 1.0:
                for prev_tokens in kept_tokens:
                    if not prev_tokens:
                        continue
                    union_len = len(tokens | prev_tokens)
                    if union_len > 0 and (len(tokens & prev_tokens) / union_len) >= near_duplicate_threshold:
                        is_near_dup = True
                        break
            if is_near_dup:
                near_dups += 1
                continue
            kept.append(chunk)
            kept_norm.add(norm)
            kept_tokens.append(tokens)

        return kept, exact_dups, near_dups

    def _chunk_fixed(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        if len(text) <= chunk_size:
            stripped = text.strip()
            return [stripped] if stripped else []

        chunks: List[str] = []
        start = 0
        step = chunk_size - overlap
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start += step
        return chunks

    def _chunk_requirement_boundary(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split SRS text along requirement headers/paragraphs without cutting statements mid-line."""
        raw_blocks = [blk.strip() for blk in re.split(r"\n\s*\n+", text) if blk.strip()]
        units: List[str] = []
        for blk in raw_blocks:
            parts = [p.strip() for p in _REQ_BOUNDARY_PATTERN.split(blk) if p and p.strip()]
            units.extend(parts if parts else [blk])

        if not units:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for unit in units:
            if len(unit) > chunk_size:
                if current:
                    chunks.append("\n".join(current))
                    current = []
                    current_len = 0
                chunks.extend(self._chunk_fixed(unit, chunk_size=chunk_size, overlap=overlap))
                continue

            sep_len = 1 if current else 0
            if current and (current_len + sep_len + len(unit)) > chunk_size:
                chunks.append("\n".join(current))
                # Carry trailing unit(s) up to overlap budget for context continuity
                carry: List[str] = []
                carry_len = 0
                if overlap > 0:
                    for prev in reversed(current):
                        added = len(prev) + (1 if carry else 0)
                        if carry_len + added <= overlap:
                            carry.insert(0, prev)
                            carry_len += added
                        else:
                            break
                current = carry + [unit]
                current_len = sum(len(x) for x in current) + max(0, len(current) - 1)
            else:
                current.append(unit)
                current_len += sep_len + len(unit)

        if current:
            chunks.append("\n".join(current))

        return chunks

    def chunk_document(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
        *,
        mode: Literal["fixed", "requirement_boundary"] | None = None,
    ) -> List[str]:
        """Split text into chunks using fixed windows (default) or requirement-boundary packing."""
        chunk_size = self.config.rag_chunk_size if chunk_size is None else chunk_size
        overlap = self.config.rag_chunk_overlap if overlap is None else overlap
        active_mode = mode or getattr(self.config, "rag_chunking_mode", "fixed")

        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if (
            isinstance(overlap, bool)
            or not isinstance(overlap, int)
            or overlap < 0
            or overlap >= chunk_size
        ):
            raise ValueError("overlap must be an integer from 0 to less than chunk_size.")
        if active_mode not in ("fixed", "requirement_boundary"):
            raise ValueError(f"Unsupported chunking mode: {active_mode!r}")

        if not text or not text.strip():
            return []

        if active_mode == "requirement_boundary":
            return self._chunk_requirement_boundary(text, chunk_size=chunk_size, overlap=overlap)
        return self._chunk_fixed(text, chunk_size=chunk_size, overlap=overlap)

    @staticmethod
    def build_pair_query(target_pair: RequirementPair) -> str:
        """Build deterministic query representation pairing Requirement A and Requirement B."""
        part_a = (target_pair.req_a.text or "").strip()
        part_b = (target_pair.req_b.text or "").strip()
        return f"{part_a} {part_b}".strip()

    def retrieve_with_diagnostics(
        self,
        target_pair: RequirementPair,
        srs_text: str,
        k: int | None = None,
        *,
        chunk_size: int | None = None,
        overlap: int | None = None,
        chunking_mode: Literal["fixed", "requirement_boundary"] | None = None,
        deduplicate: bool | None = None,
        near_duplicate_threshold: float | None = None,
        cite_id_boost: float | None = None,
        min_similarity: float | None = None,
    ) -> Tuple[List[str], RAGRetrievalDiagnostics]:
        """Retrieve top-k SRS chunks together with measurable quality diagnostics."""
        k = self.config.rag_k_chunks if k is None else k
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer.")

        active_mode = chunking_mode or getattr(self.config, "rag_chunking_mode", "fixed")
        active_dedup = (
            getattr(self.config, "rag_deduplicate_chunks", False)
            if deduplicate is None
            else deduplicate
        )
        active_near_thresh = (
            getattr(self.config, "rag_near_duplicate_threshold", 0.90)
            if near_duplicate_threshold is None
            else near_duplicate_threshold
        )
        active_boost = (
            getattr(self.config, "rag_cite_id_boost", 0.0)
            if cite_id_boost is None
            else cite_id_boost
        )
        active_min_sim = (
            getattr(self.config, "rag_min_similarity", 0.10)
            if min_similarity is None
            else min_similarity
        )

        cited_ids = self.extract_cited_requirement_ids(target_pair)

        if k == 0 or not srs_text or not srs_text.strip():
            return [], RAGRetrievalDiagnostics(
                cited_ids=cited_ids,
                chunking_mode=active_mode,
                deduplicated=active_dedup,
                cite_id_boost=active_boost,
            )

        raw_chunks = self.chunk_document(
            srs_text, chunk_size=chunk_size, overlap=overlap, mode=active_mode
        )
        if not raw_chunks:
            return [], RAGRetrievalDiagnostics(
                cited_ids=cited_ids,
                chunking_mode=active_mode,
                deduplicated=active_dedup,
                cite_id_boost=active_boost,
            )

        dedup_chunks, exact_dups, near_dups = self.analyze_chunk_duplicates(
            raw_chunks, near_duplicate_threshold=active_near_thresh
        )
        total_chunks = len(raw_chunks)
        unique_exact = len({_norm_text(c) for c in raw_chunks})
        dup_ratio = (exact_dups + near_dups) / total_chunks if total_chunks > 0 else 0.0

        candidate_chunks = dedup_chunks if active_dedup else raw_chunks

        query_text = self.build_pair_query(target_pair)
        query_emb = self.embedder.encode([query_text])[0]
        chunk_embs = self.embedder.encode(candidate_chunks)

        def mentions_cited_id(chunk_str: str) -> bool:
            if not cited_ids:
                return False
            upper_chunk = chunk_str.upper()
            return any(cid in upper_chunk for cid in cited_ids)

        cited_available = sum(1 for ch in candidate_chunks if mentions_cited_id(ch))

        # Tuple: (final_score, base_sim, has_cited, original_idx, chunk_str)
        scored_chunks: List[Tuple[float, float, bool, int, str]] = []
        for idx, chunk_emb in enumerate(chunk_embs):
            chunk_str = candidate_chunks[idx]
            base_sim = compute_vector_similarity(
                query_emb, chunk_emb, metric=self.config.similarity_metric
            )
            has_cited = mentions_cited_id(chunk_str)
            final_score = base_sim + (active_boost if has_cited and active_boost > 0 else 0.0)
            scored_chunks.append((final_score, base_sim, has_cited, idx, chunk_str))

        # Deterministic tie-breaking: score descending, then original chunk index ascending, then text
        scored_chunks.sort(key=lambda x: (-x[0], x[3], x[4]))
        selected = scored_chunks[:k]

        selected_texts = [item[4] for item in selected]
        selected_scores = [round(float(item[0]), 6) for item in selected]
        cited_selected = sum(1 for item in selected if item[2])
        irrelevant_selected = sum(1 for item in selected if item[1] < active_min_sim)

        diagnostics = RAGRetrievalDiagnostics(
            total_chunks=total_chunks,
            unique_exact_chunks=unique_exact,
            exact_duplicate_count=exact_dups,
            near_duplicate_count=near_dups,
            duplicate_ratio=round(dup_ratio, 6),
            candidate_chunks_after_dedup=len(candidate_chunks),
            cited_ids=cited_ids,
            cited_id_chunks_available=cited_available,
            cited_id_chunks_selected=cited_selected,
            selected_count=len(selected_texts),
            irrelevant_selected_count=irrelevant_selected,
            selected_scores=selected_scores,
            chunking_mode=active_mode,
            deduplicated=active_dedup,
            cite_id_boost=active_boost,
        )
        return selected_texts, diagnostics

    def retrieve(
        self,
        target_pair: RequirementPair,
        srs_text: str,
        k: int | None = None,
        **kwargs,
    ) -> List[str]:
        """Retrieve top k most relevant context chunks from SRS document."""
        chunks, _ = self.retrieve_with_diagnostics(target_pair, srs_text, k=k, **kwargs)
        return chunks
