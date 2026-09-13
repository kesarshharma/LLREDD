"""Main LEREDD Detection Pipeline orchestrating retrieval, prompts, LLM generation, and parsing."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from loguru import logger
from leredd.config import LEREDDConfig, get_config
from leredd.data_models import (
    RequirementPair,
    DependencyType,
    Prediction,
)
from leredd.embeddings import EmbeddingModel
from leredd.retrieval import ExampleRetriever, RAGRetriever
from leredd.prompts import PromptBuilder
from leredd.llm import LLMClient, LLMClientFactory


class ResponseParser:
    """Parses structured LLM responses into Prediction objects."""

    @staticmethod
    def parse(
        raw_text: str,
        pair_id: str,
        confidence_threshold: int = 4,
        usage_info: Dict[str, Any] | None = None
    ) -> Prediction:
        """Extract Dependency_Type, Rationale, and Confidence Score from LLM text."""
        usage = usage_info or {}

        # 1. Parse Dependency Type
        dep_match = re.search(r"Dependency_Type:\s*\[?(.*?)\]?\*?\*?$", raw_text, re.MULTILINE | re.IGNORECASE)
        if dep_match:
            raw_type_str = dep_match.group(1).strip().strip("*").strip("[]")
            parsed_type = DependencyType.normalize(raw_type_str)
        else:
            # Fallback search anywhere in text
            parsed_type = DependencyType.NO_DEPENDENCY
            for item in DependencyType:
                if item.value.lower() in raw_text.lower():
                    parsed_type = item
                    break

        # 2. Parse Rationale
        rat_match = re.search(r"Rationale:\s*\[?(.*?)\]?\*?$", raw_text, re.MULTILINE | re.IGNORECASE)
        if rat_match:
            rationale = rat_match.group(1).strip().strip("*").strip("[]")
        else:
            rationale = raw_text.strip()

        # 3. Parse Confidence Score (0 to 5)
        conf_match = re.search(r"Confidence\s*Score:\s*\[?(\d+)\]?", raw_text, re.IGNORECASE)
        if conf_match:
            try:
                confidence = int(conf_match.group(1))
                confidence = max(0, min(5, confidence))
            except ValueError:
                confidence = 5
        else:
            confidence = 5

        # 4. Apply paper Threshold-Based Re-annotation (RQ2 optimal setting)
        # Requirement pairs assigned a confidence score of threshold or lower are automatically reclassified as No_dependency
        final_type = parsed_type
        if confidence <= confidence_threshold and final_type != DependencyType.NO_DEPENDENCY:
            logger.info(
                f"Re-annotating pair {pair_id} from {final_type.value} to No_dependency "
                f"(confidence {confidence} <= threshold {confidence_threshold})"
            )
            final_type = DependencyType.NO_DEPENDENCY

        return Prediction(
            pair_id=pair_id,
            dependency_type=final_type,
            rationale=rationale,
            confidence=confidence,
            raw_response=raw_text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            estimated_cost_usd=usage.get("cost_usd", 0.0)
        )


class LEREDDDetector:
    """Core LEREDD Engine executing the 2-phase knowledge retrieval and dependency inference pipeline."""

    def __init__(
        self,
        config: LEREDDConfig | None = None,
        embedder: EmbeddingModel | None = None,
        llm: LLMClient | None = None
    ):
        self.config = config or get_config()
        self.embedder = embedder or EmbeddingModel(self.config.embedding_model)
        self.example_retriever = ExampleRetriever(self.embedder, self.config)
        self.rag_retriever = RAGRetriever(self.embedder, self.config) if self.config.use_rag else None
        self.prompt_builder = PromptBuilder(self.config)
        self.llm = llm or LLMClientFactory.create(self.config)

    def detect_pair(
        self,
        target_pair: RequirementPair,
        annotated_pool: List[RequirementPair] | None = None,
        srs_text: str | None = None,
        domain: str = "automotive software engineering"
    ) -> Prediction:
        """Execute LEREDD detection pipeline on a single requirement pair."""
        pool = annotated_pool or []

        # Phase 1: Knowledge Retrieval
        # 1. Dynamic Examples Retrieval (In-Context Learning)
        retrieved_examples = None
        if pool:
            retrieved_examples = self.example_retriever.retrieve(
                target_pair=target_pair,
                annotated_pool=pool,
                k=self.config.k_examples
            )

        # 2. Contextual Retrieval (RAG) - used for cross-system / inter-dataset setting
        context_chunks = None
        if self.config.use_rag and srs_text and self.rag_retriever:
            context_chunks = self.rag_retriever.retrieve(
                target_pair=target_pair,
                srs_text=srs_text,
                k=self.config.rag_k_chunks
            )

        # Phase 2: Dependency Inference
        prompt = self.prompt_builder.build(
            target_pair=target_pair,
            retrieved_examples=retrieved_examples,
            context_chunks=context_chunks,
            domain=domain,
            system_name=target_pair.req_a.system
        )

        response_text, usage = self.llm.generate(prompt, temperature=self.config.temperature)

        prediction = ResponseParser.parse(
            raw_text=response_text,
            pair_id=target_pair.id,
            confidence_threshold=self.config.confidence_threshold,
            usage_info=usage
        )

        return prediction

    def detect_batch(
        self,
        pairs: List[RequirementPair],
        annotated_pool: List[RequirementPair] | None = None,
        srs_text: str | None = None,
        max_workers: int = 4
    ) -> List[Prediction]:
        """Execute batch detection across a list of requirement pairs."""
        if not pairs:
            return []

        logger.info(f"Starting batch LEREDD detection for {len(pairs)} pairs using {max_workers} worker threads.")

        results: Dict[str, Prediction] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(self.detect_pair, pair, annotated_pool, srs_text): pair
                for pair in pairs
            }
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    pred = future.result()
                    results[pair.id] = pred
                except Exception as e:
                    logger.error(f"Error processing pair {pair.id}: {e}")
                    results[pair.id] = Prediction(
                        pair_id=pair.id,
                        dependency_type=DependencyType.NO_DEPENDENCY,
                        rationale=f"Pipeline error: {str(e)}",
                        confidence=0
                    )

        # Preserve original input list ordering
        return [results[p.id] for p in pairs if p.id in results]
