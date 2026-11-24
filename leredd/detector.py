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
    PredictionStabilityStatus,
    PredictionValidationStatus,
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
            parsed_type = None
            for item in DependencyType:
                if item.value.lower() in raw_text.lower():
                    parsed_type = item
                    break
            if parsed_type is None:
                raise ValueError("Malformed LLM response: no supported dependency label found")

        # 2. Parse Rationale
        rat_match = re.search(
            r"Rationale:\s*(?:\*\*)?\s*\[?(.*?)(?=(?:\n\s*|\s+(?=\*\*))\s*(?:\*\*)?\s*(?:Confidence\s*Score|Dependency_Type)\b|\Z)",
            raw_text,
            re.DOTALL | re.IGNORECASE,
        )
        if rat_match:
            rationale = rat_match.group(1).strip().strip("*").strip("[]").strip()
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

        # Confidence screening is a review signal. A low self-reported score
        # must not change the predicted taxonomy class into No_dependency.
        final_type = parsed_type
        if confidence_threshold < 0:
            validation_status = PredictionValidationStatus.THRESHOLD_DISABLED
        elif confidence <= confidence_threshold:
            validation_status = PredictionValidationStatus.REQUIRES_REVIEW
            logger.info(
                f"Flagging pair {pair_id} for review: self-reported confidence "
                f"{confidence} <= threshold {confidence_threshold}"
            )
        else:
            validation_status = PredictionValidationStatus.PASSED_THRESHOLD

        return Prediction(
            pair_id=pair_id,
            dependency_type=final_type,
            rationale=rationale,
            confidence=confidence,
            validation_status=validation_status,
            validation_threshold=confidence_threshold,
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
        # Keep model startup off the prediction path unless retrieval actually
        # needs embeddings. Zero-shot detection and BM25 retrieval do not use it.
        self._embedder = embedder
        self._example_retriever: ExampleRetriever | None = None
        self._bm25_retriever = None
        self._hybrid_retriever = None
        self._rag_retriever: RAGRetriever | None = None
        self.prompt_builder = PromptBuilder(self.config)
        self.llm = llm or LLMClientFactory.create(self.config)

    @property
    def embedder(self) -> EmbeddingModel:
        """Initialize dense embeddings only when a workflow first needs them."""
        if self._embedder is None:
            self._embedder = EmbeddingModel(self.config.embedding_model)
        return self._embedder

    @property
    def loaded_embedder(self) -> EmbeddingModel | None:
        """Return the initialized embedder without triggering model startup."""
        return self._embedder

    @embedder.setter
    def embedder(self, value: EmbeddingModel) -> None:
        self._embedder = value
        self._example_retriever = None
        self._hybrid_retriever = None
        self._rag_retriever = None

    @property
    def example_retriever(self) -> ExampleRetriever:
        if self._example_retriever is None:
            self._example_retriever = ExampleRetriever(self.embedder, self.config)
        return self._example_retriever

    @example_retriever.setter
    def example_retriever(self, value: ExampleRetriever) -> None:
        self._example_retriever = value

    @property
    def rag_retriever(self) -> RAGRetriever | None:
        if not self.config.use_rag:
            return None
        if self._rag_retriever is None:
            self._rag_retriever = RAGRetriever(self.embedder, self.config)
        return self._rag_retriever

    @rag_retriever.setter
    def rag_retriever(self, value: RAGRetriever | None) -> None:
        self._rag_retriever = value

    @property
    def bm25_retriever(self):
        if self._bm25_retriever is None:
            from leredd.bm25_retrieval import BM25Retriever
            self._bm25_retriever = BM25Retriever(config=self.config)
        return self._bm25_retriever

    @property
    def hybrid_retriever(self):
        if self._hybrid_retriever is None:
            from leredd.hybrid_retrieval import HybridRetriever
            self._hybrid_retriever = HybridRetriever(
                dense_retriever=self.example_retriever,
                config=self.config
            )
        return self._hybrid_retriever

    def detect_pair(
        self,
        target_pair: RequirementPair,
        annotated_pool: List[RequirementPair] | None = None,
        srs_text: str | None = None,
        domain: str = "automotive software engineering",
        retrieval_mode: str | None = None,
        confidence_threshold: int | None = None,
        stability_runs: int = 1,
        use_rag: bool | None = None,
    ) -> Prediction:
        """Execute LEREDD detection pipeline on a single requirement pair."""
        if (
            isinstance(stability_runs, bool)
            or not isinstance(stability_runs, int)
            or not 1 <= stability_runs <= 5
        ):
            raise ValueError("stability_runs must be an integer from 1 to 5.")
        pool = annotated_pool or []

        # Phase 1: Knowledge Retrieval
        # 1. Dynamic Examples Retrieval (In-Context Learning)
        retrieved_examples = None
        mode = retrieval_mode or self.config.retrieval_mode
        if pool and self.config.k_examples > 0:
            if mode == "bm25":
                retrieved_examples = self.bm25_retriever.retrieve_example_pairs(
                    target_pair=target_pair, candidate_pool=pool, k=self.config.k_examples
                )
            elif mode == "hybrid":
                retrieved_examples = self.hybrid_retriever.retrieve_example_pairs(
                    target_pair=target_pair, annotated_pool=pool, k=self.config.k_examples
                )
            else:
                retrieved_examples = self.example_retriever.retrieve(
                    target_pair=target_pair, annotated_pool=pool, k=self.config.k_examples
                )

        # 2. Contextual Retrieval (RAG) - used for cross-system / inter-dataset setting
        context_chunks = None
        rag_enabled = self.config.use_rag if use_rag is None else use_rag
        rag_retriever = self._rag_retriever
        if rag_enabled and srs_text:
            if rag_retriever is None:
                rag_retriever = RAGRetriever(self.embedder, self.config)
                self._rag_retriever = rag_retriever
            context_chunks = rag_retriever.retrieve(
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

        generator = self.llm.generate
        if stability_runs > 1:
            generator = getattr(self.llm, "generate_uncached", generator)
        responses = [
            generator(prompt, temperature=self.config.temperature)
            for _ in range(stability_runs)
        ]
        response_text = responses[0][0]
        usage = {
            "prompt_tokens": sum(int(item.get("prompt_tokens", 0) or 0) for _, item in responses),
            "completion_tokens": sum(int(item.get("completion_tokens", 0) or 0) for _, item in responses),
            "cost_usd": sum(float(item.get("cost_usd", 0.0) or 0.0) for _, item in responses),
        }

        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.config.confidence_threshold
        )

        prediction = ResponseParser.parse(
            raw_text=response_text,
            pair_id=target_pair.id,
            confidence_threshold=threshold,
            usage_info=usage
        )

        if stability_runs == 1:
            return prediction

        repeated = [
            ResponseParser.parse(
                raw_text=raw_text,
                pair_id=target_pair.id,
                confidence_threshold=threshold,
            )
            for raw_text, _ in responses
        ]
        labels = [item.dependency_type for item in repeated]
        confidences = [item.confidence for item in repeated]
        reasons = []
        if len(set(labels)) > 1:
            reasons.append("dependency_label_disagreement")
        if len(set(confidences)) > 1:
            reasons.append("self_reported_confidence_variation")

        return prediction.model_copy(update={
            "stability_status": (
                PredictionStabilityStatus.INCONSISTENT
                if reasons
                else PredictionStabilityStatus.CONSISTENT
            ),
            "stability_run_count": stability_runs,
            "stability_labels": labels,
            "stability_confidences": confidences,
            "stability_reasons": reasons,
        })

    def detect_batch(
        self,
        pairs: List[RequirementPair],
        annotated_pool: List[RequirementPair] | None = None,
        srs_text: str | None = None,
        max_workers: int = 4,
        retrieval_mode: str | None = None,
        confidence_threshold: int | None = None,
        stability_runs: int = 1,
        use_rag: bool | None = None,
    ) -> List[Prediction]:
        """Execute batch detection across a list of requirement pairs."""
        if not pairs:
            return []

        logger.info(f"Starting batch LEREDD detection for {len(pairs)} pairs using {max_workers} worker threads.")

        results: Dict[str, Prediction] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(
                    self.detect_pair,
                    pair,
                    annotated_pool,
                    srs_text,
                    "automotive software engineering",
                    retrieval_mode,
                    confidence_threshold,
                    stability_runs,
                    use_rag,
                ): pair
                for pair in pairs
            }
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    pred = future.result()
                    results[pair.id] = pred
                except Exception as e:
                    logger.error(f"Error processing pair {pair.id}: {e}")
                    raise RuntimeError(
                        f"Failed to detect dependency for pair {pair.id}: {e}"
                    ) from e

        # Preserve original input list ordering
        return [results[p.id] for p in pairs if p.id in results]
