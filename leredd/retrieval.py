"""Dynamic Example Retrieval and RAG Contextual Retrieval modules."""

import numpy as np
from typing import List, Dict
from loguru import logger
from leredd.config import LEREDDConfig, get_config
from leredd.data_models import RequirementPair, DependencyType
from leredd.embeddings import EmbeddingModel


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
    else:  # Cosine
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
        # Paper Eq. 2
        return (max(sim_1a, sim_1b) + max(sim_2a, sim_2b)) / 2.0
    else:
        # Paper Eq. 1 (Average - Optimal choice from Paper RQ2)
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
        k = k or self.config.k_examples

        if not annotated_pool:
            logger.warning("Annotated candidate pool is empty. Returning empty examples dictionary.")
            return {}

        # 1. Encode target requirement pair statements
        r1_text = target_pair.req_a.text
        r2_text = target_pair.req_b.text
        target_embs = self.embedder.encode([r1_text, r2_text])
        emb_r1, emb_r2 = target_embs[0], target_embs[1]

        # 2. Extract candidate requirement statements and encode batch
        candidate_texts: List[str] = []
        for pair in annotated_pool:
            candidate_texts.append(pair.req_a.text)
            candidate_texts.append(pair.req_b.text)

        candidate_embs = self.embedder.encode(candidate_texts)

        # 3. Group candidates by dependency type and compute similarity scores
        typed_candidates: Dict[DependencyType, List[tuple[float, RequirementPair]]] = {}
        for dep_type in DependencyType:
            typed_candidates[dep_type] = []

        for idx, candidate_pair in enumerate(annotated_pool):
            # Skip if candidate is identical to target pair
            if candidate_pair.id == target_pair.id or (
                candidate_pair.req_a.text == target_pair.req_a.text
                and candidate_pair.req_b.text == target_pair.req_b.text
            ):
                continue

            emb_ra = candidate_embs[2 * idx]
            emb_rb = candidate_embs[2 * idx + 1]

            score = compute_pair_similarity(
                emb_r1, emb_r2, emb_ra, emb_rb,
                metric=self.config.similarity_metric,
                aggregation=self.config.aggregation
            )

            dep_type = candidate_pair.dependency_type or DependencyType.NO_DEPENDENCY
            typed_candidates[dep_type].append((score, candidate_pair))

        # 4. Sort each group by score descending and select top k
        retrieved_examples: Dict[DependencyType, List[RequirementPair]] = {}
        for dep_type, scored_list in typed_candidates.items():
            scored_list.sort(key=lambda x: x[0], reverse=True)
            retrieved_examples[dep_type] = [pair for _, pair in scored_list[:k]]

        return retrieved_examples


class RAGRetriever:
    """Fixed-size chunking and RAG context retrieval for cross-system setting."""

    def __init__(self, embedder: EmbeddingModel | None = None, config: LEREDDConfig | None = None):
        self.config = config or get_config()
        self.embedder = embedder or EmbeddingModel(self.config.embedding_model)

    def chunk_document(self, text: str, chunk_size: int | None = None, overlap: int | None = None) -> List[str]:
        """Split text into fixed-size chunks with specified overlap."""
        chunk_size = chunk_size or self.config.rag_chunk_size
        overlap = overlap or self.config.rag_chunk_overlap

        if not text or len(text) <= chunk_size:
            return [text] if text else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start += (chunk_size - overlap)
        return chunks

    def retrieve(
        self,
        target_pair: RequirementPair,
        srs_text: str,
        k: int | None = None
    ) -> List[str]:
        """Retrieve top k most relevant context chunks from SRS document."""
        k = k or self.config.rag_k_chunks
        chunks = self.chunk_document(srs_text)

        if not chunks:
            return []

        # Target query representation: concatenation of Requirement A and Requirement B
        query_text = f"{target_pair.req_a.text} {target_pair.req_b.text}"
        query_emb = self.embedder.encode([query_text])[0]
        chunk_embs = self.embedder.encode(chunks)

        scored_chunks = []
        for idx, chunk_emb in enumerate(chunk_embs):
            sim = compute_vector_similarity(query_emb, chunk_emb, metric=self.config.similarity_metric)
            scored_chunks.append((sim, chunks[idx]))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:k]]
