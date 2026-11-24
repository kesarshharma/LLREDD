"""Main LEREDD Detection Pipeline orchestrating retrieval, prompts, LLM generation, and parsing."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
import re
from typing import Any, Dict, List
import warnings
from loguru import logger
from leredd.config import LEREDDConfig, get_config
from leredd.data_models import (
    AnnotationSource,
    DependencyType,
    Prediction,
    PredictionStabilityStatus,
    PredictionValidationStatus,
    RequirementPair,
)
from leredd.embeddings import EmbeddingModel
from leredd.llm import LLMClient, LLMClientFactory
from leredd.prompts import PromptBuilder
from leredd.retrieval import ExampleRetriever, RAGRetriever


class ResponseParser:
    """Parses structured LLM responses into Prediction objects.

    CONFIDENCE POLICY (Part 4):
    - Raw predicted dependency classes are preserved by default regardless of confidence.
    - Self-reported Likert confidence (0-5) is an ordinal screening cue, NEVER a
      calibrated probability.
    - Predictions with confidence <= confidence_threshold receive
      validation_status = PredictionValidationStatus.REQUIRES_REVIEW.
    - Malformed responses missing an explicit Dependency_Type header or valid JSON
      field raise ValueError rather than guessing labels from free-text substrings.
    """

    @staticmethod
    def parse(
        raw_text: str,
        pair_id: str,
        confidence_threshold: int = 4,
        usage_info: Dict[str, Any] | None = None,
        *,
        legacy_rewrite_low_confidence: bool = False,
    ) -> Prediction:
        """Extract Dependency_Type, Rationale, and Confidence Score from LLM text."""
        usage = usage_info or {}
        if not raw_text or not raw_text.strip():
            raise ValueError("Malformed LLM response: empty response text")

        parsed_type: DependencyType | None = None
        rationale: str | None = None
        confidence: int | None = None
        missing_or_invalid_confidence = False

        # 0. Support structured JSON responses when provided
        stripped = raw_text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed LLM JSON response: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError("Malformed LLM JSON response: expected JSON object")
            raw_dep = payload.get("dependency_type") or payload.get("Dependency_Type")
            if raw_dep is None or not str(raw_dep).strip():
                raise ValueError("Malformed LLM JSON response: missing dependency_type field")
            parsed_type = DependencyType.normalize(str(raw_dep))
            rationale = str(
                payload.get("rationale")
                or payload.get("Rationale")
                or payload.get("explanation")
                or ""
            ).strip()
            raw_conf = (
                payload.get("confidence")
                if "confidence" in payload
                else payload.get("confidence_score", payload.get("Confidence Score"))
            )
            if raw_conf is None or isinstance(raw_conf, bool):
                missing_or_invalid_confidence = True
                confidence = 0
            else:
                try:
                    conf_int = int(raw_conf)
                except (TypeError, ValueError):
                    missing_or_invalid_confidence = True
                    confidence = 0
                else:
                    if 0 <= conf_int <= 5:
                        confidence = conf_int
                    else:
                        missing_or_invalid_confidence = True
                        confidence = max(0, min(5, conf_int))

        # 1. Parse explicit Dependency_Type field from structured Markdown/text
        if parsed_type is None:
            dep_match = re.search(
                r"(?:^\s*(?:\*\*)?\s*Dependency_Type\s*(?:\*\*)?\s*:\s*)(.*?)\s*$",
                raw_text,
                re.MULTILINE | re.IGNORECASE,
            )
            if not dep_match:
                raise ValueError(
                    "Malformed LLM response: missing explicit Dependency_Type field"
                )
            raw_type_str = dep_match.group(1).strip().strip("*").strip("[]").strip("*").strip()
            if not raw_type_str:
                raise ValueError("Malformed LLM response: empty Dependency_Type value")
            parsed_type = DependencyType.normalize(raw_type_str)

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
            conf_header_match = re.search(
                r"Confidence\s*Score\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*\[?([^\n\r\*\]]+)\]?",
                raw_text,
                re.IGNORECASE,
            )
            if conf_header_match:
                raw_conf_token = conf_header_match.group(1).strip()
                int_match = re.match(r"^(-?\d+)(?:\s*/\s*5)?$", raw_conf_token)
                if int_match:
                    conf_val = int(int_match.group(1))
                    if 0 <= conf_val <= 5:
                        confidence = conf_val
                    else:
                        missing_or_invalid_confidence = True
                        confidence = max(0, min(5, conf_val))
                else:
                    missing_or_invalid_confidence = True
                    confidence = 0
            else:
                missing_or_invalid_confidence = True
                confidence = 0

        assert confidence is not None
        assert rationale is not None

        # Confidence screening is a review signal. By default, a low self-reported
        # score preserves the predicted taxonomy class and sets REQUIRES_REVIEW.
        final_type = parsed_type
        if confidence_threshold < 0:
            validation_status = PredictionValidationStatus.THRESHOLD_DISABLED
        elif missing_or_invalid_confidence or confidence <= confidence_threshold:
            validation_status = PredictionValidationStatus.REQUIRES_REVIEW
            logger.info(
                f"Flagging pair {pair_id} for review: self-reported confidence "
                f"{confidence} <= threshold {confidence_threshold}"
                + (" (missing/invalid confidence field)" if missing_or_invalid_confidence else "")
            )
            if legacy_rewrite_low_confidence:
                warnings.warn(
                    "legacy_rewrite_low_confidence is deprecated; low-confidence "
                    "predictions should preserve dependency_type and use validation_status.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                final_type = DependencyType.NO_DEPENDENCY
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
            estimated_cost_usd=usage.get("cost_usd", 0.0),
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
                bm25_retriever=self.bm25_retriever,
                config=self.config,
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
            elif mode == "hybrid_no_metadata":
                from leredd.hybrid_retrieval import HybridRetriever
                no_meta_hybrid = HybridRetriever(
                    dense_retriever=self.example_retriever,
                    bm25_retriever=self.bm25_retriever,
                    metadata_weight=0.0,
                    config=self.config,
                )
                retrieved_examples = no_meta_hybrid.retrieve_example_pairs(
                    target_pair=target_pair, annotated_pool=pool, k=self.config.k_examples
                )
            elif mode == "hybrid_metadata":
                from leredd.hybrid_retrieval import HybridRetriever
                meta_hybrid = HybridRetriever(
                    dense_retriever=self.example_retriever,
                    bm25_retriever=self.bm25_retriever,
                    metadata_weight=self.config.metadata_weight if self.config.metadata_weight > 0 else 1.0,
                    config=self.config,
                )
                retrieved_examples = meta_hybrid.retrieve_example_pairs(
                    target_pair=target_pair, annotated_pool=pool, k=self.config.k_examples
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
            usage_info=usage,
            legacy_rewrite_low_confidence=self.config.legacy_rewrite_low_confidence_to_no_dependency,
        )
        prediction = prediction.model_copy(
            update={
                "annotation_source": AnnotationSource.LLM,
                "model_id": self.config.llm_model,
                "config_hash": self.prompt_builder.template_hash(domain=domain),
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )

        if stability_runs == 1:
            return prediction

        repeated = [
            ResponseParser.parse(
                raw_text=raw_text,
                pair_id=target_pair.id,
                confidence_threshold=threshold,
                legacy_rewrite_low_confidence=self.config.legacy_rewrite_low_confidence_to_no_dependency,
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
