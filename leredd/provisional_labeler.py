"""Provisional Requirement Dependency Labeling module (Phase 4, TASK-P4-02).

Implements provisional dataset labeling using:
1. HeuristicLabeler: Rule-based baseline detecting explicit references, lexical similarity, and negation/conflict indicators.
2. LLMLabeler: LLM-based classifier using existing LLM infrastructure (and MockLLMClient for offline testing).
3. ProvisionalLabeler: Coordinator combining heuristic and LLM outputs, recording agreement/disagreement, and producing auditable provisional dataset records with full provenance.

CORE PRINCIPLE: Machine-generated labels are PROVISIONAL predictions (VerificationStatus.PROVISIONAL).
They are NEVER silently promoted to verified ground-truth annotations.
"""

import datetime
import json
import os
import re
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from pydantic import BaseModel, Field

from leredd.config import LEREDDConfig, get_config
from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    DependencyType,
    RequirementPair,
    VerificationStatus,
)
from leredd.detector import ResponseParser
from leredd.llm import LLMClient, LLMClientFactory
from leredd.prompts import PromptBuilder


class LabelingMode(str, Enum):
    """Supported provisional labeling modes."""
    HEURISTIC = "heuristic"
    LLM = "llm"
    HYBRID = "hybrid"


class HeuristicRuleResult(BaseModel):
    """Result of a heuristic rule evaluation."""
    suggested_label: Optional[DependencyType] = None
    rule_id: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str


class HeuristicLabeler:
    """Rule-based baseline for provisional requirement dependency labeling.

    Rules:
    - RULE-REF: Explicit requirement ID citation or explicit dependency phrase -> Requires / Implements / Details / Conflicts.
    - RULE-NEG: Negation + entity overlap -> Conflicts.
    - RULE-SIM: High lexical Jaccard overlap -> Is similar.
    """

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    @classmethod
    def evaluate_pair(cls, pair: RequirementPair) -> HeuristicRuleResult:
        """Evaluate a requirement pair using deterministic rules.

        Returns HeuristicRuleResult with suggested_label=None if evidence is insufficient (abstention).
        """
        text_a = pair.req_a.text
        text_b = pair.req_b.text
        id_a = (pair.req_a.id or "").upper()
        id_b = (pair.req_b.id or "").upper()

        lower_a = text_a.lower()
        lower_b = text_b.lower()

        # 1. Check explicit requirement ID reference
        if id_b and id_b in text_a.upper():
            return HeuristicRuleResult(
                suggested_label=DependencyType.REQUIRES,
                rule_id="RULE-REF-ID-DIRECT",
                confidence=0.9,
                evidence=f"Requirement A ({pair.req_a.id}) explicitly cites Requirement B ({id_b})."
            )
        if id_a and id_a in text_b.upper():
            return HeuristicRuleResult(
                suggested_label=DependencyType.REQUIRES,
                rule_id="RULE-REF-ID-REVERSE",
                confidence=0.9,
                evidence=f"Requirement B ({pair.req_b.id}) explicitly cites Requirement A ({id_a})."
            )

        # 2. Check explicit structural dependency keywords
        if "implements" in lower_a or "subsystem" in lower_b and "include" in lower_b:
            if "subsystem" in lower_a or "include" in lower_a or "implements" in lower_b:
                return HeuristicRuleResult(
                    suggested_label=DependencyType.IMPLEMENTS,
                    rule_id="RULE-REF-IMPLEMENTS",
                    confidence=0.8,
                    evidence="Explicit subsystem architecture or implementation relation detected."
                )

        if "conflicts with" in lower_a or "contradicts" in lower_a or "conflicts with" in lower_b or "contradicts" in lower_b:
            return HeuristicRuleResult(
                suggested_label=DependencyType.CONFLICTS,
                rule_id="RULE-REF-CONFLICTS",
                confidence=0.85,
                evidence="Explicit conflict/contradiction terminology detected."
            )

        # 3. Check negation / conflict indicators with entity overlap
        tokens_a = cls._tokenize(text_a)
        tokens_b = cls._tokenize(text_b)
        intersection = tokens_a.intersection(tokens_b)

        negation_terms = {"not", "no", "never", "cannot", "prohibit", "prohibited", "disable", "deactivate"}
        has_neg_a = bool(tokens_a.intersection(negation_terms))
        has_neg_b = bool(tokens_b.intersection(negation_terms))

        if (has_neg_a != has_neg_b) and len(intersection) >= 3:
            return HeuristicRuleResult(
                suggested_label=DependencyType.CONFLICTS,
                rule_id="RULE-NEG-CONFLICT",
                confidence=0.75,
                evidence=f"Asymmetric negation detected with {len(intersection)} shared terms: {sorted(list(intersection))[:4]}."
            )

        # 4. Check high lexical similarity (Jaccard)
        union = tokens_a.union(tokens_b)
        jaccard = len(intersection) / len(union) if union else 0.0

        if jaccard >= 0.45:
            return HeuristicRuleResult(
                suggested_label=DependencyType.IS_SIMILAR,
                rule_id="RULE-SIM-JACCARD",
                confidence=float(min(1.0, jaccard)),
                evidence=f"High lexical similarity detected (Jaccard score = {jaccard:.3f})."
            )

        # 5. Insufficient evidence -> Abstain (return None, NOT No_dependency)
        return HeuristicRuleResult(
            suggested_label=None,
            rule_id="RULE-ABSTAIN",
            confidence=0.0,
            evidence="Insufficient heuristic evidence to assert a dependency label."
        )


class LLMLabeler:
    """LLM-based classifier for provisional requirement dependency labeling."""

    def __init__(self, llm_client: Optional[LLMClient] = None, config: Optional[LEREDDConfig] = None):
        self.config = config or get_config()
        self.llm_client = llm_client or LLMClientFactory.create(self.config)
        self.prompt_builder = PromptBuilder()

    def label_pair(
        self,
        target_pair: RequirementPair,
        examples: Optional[Dict[DependencyType, List[RequirementPair]]] = None,
        srs_context: Optional[List[str]] = None,
    ) -> Tuple[Optional[DependencyType], Optional[int], str]:
        """Classify a requirement pair using LLM client.

        Returns Tuple of (predicted_label, confidence_score_0_5, rationale_text).
        Returns (None, None, error_message) if parsing fails or LLM returns malformed label.
        """
        try:
            prompt = self.prompt_builder.build(target_pair, retrieved_examples=examples, context_chunks=srs_context)
            raw_res = self.llm_client.generate(prompt)
            raw_response = raw_res[0] if isinstance(raw_res, tuple) else raw_res

            # Try JSON parsing first
            try:
                data = json.loads(raw_response)
                if isinstance(data, dict):
                    dep_str = data.get("dependency_type") or data.get("dependency")
                    conf = data.get("confidence") or data.get("confidence_score")
                    rat = data.get("explanation") or data.get("rationale", "")
                    if dep_str:
                        norm_label = DependencyType.normalize(str(dep_str))
                        return norm_label, int(conf) if conf is not None else None, str(rat)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

            # Fallback to standard ResponseParser if explicit tag exists
            dep_match = re.search(r"Dependency_Type:\s*\[?(.*?)\]?\*?\*?$", raw_response, re.MULTILINE | re.IGNORECASE)
            if not dep_match:
                return None, None, f"Malformed LLM response (missing explicit Dependency_Type tag): {raw_response[:100]}"

            parsed = ResponseParser.parse(raw_response, pair_id=target_pair.id)
            return parsed.dependency_type, parsed.confidence, parsed.rationale

        except Exception as e:
            logger.error(f"Error during LLM labeling for pair {target_pair.id}: {e}")
            return None, None, f"LLM provider error: {str(e)}"


class LabelingRunManifest(BaseModel):
    """Audit manifest for a provisional labeling run."""
    run_id: str = Field(default_factory=lambda: f"LABEL-{uuid.uuid4().hex[:8].upper()}")
    source_identity: str = Field(description="Input candidates dataset path or system identifier")
    input_candidate_count: int = Field(description="Total candidate pairs processed")
    mode: LabelingMode = Field(description="Labeling mode used")
    labeled_count: int = Field(description="Pairs assigned a valid provisional label")
    agreed_count: int = Field(description="Pairs where heuristic and LLM agreed")
    disagreed_count: int = Field(description="Pairs where heuristic and LLM produced conflicting valid labels")
    unresolved_count: int = Field(description="Pairs where neither method or only abstentions occurred")
    failed_count: int = Field(description="Pairs that failed execution")
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProvisionalLabeler:
    """Coordinator combining Heuristic and LLM labeling outputs into auditable provisional records."""

    def __init__(
        self,
        heuristic_labeler: Optional[HeuristicLabeler] = None,
        llm_labeler: Optional[LLMLabeler] = None,
        config: Optional[LEREDDConfig] = None,
    ):
        self.config = config or get_config()
        self.heuristic_labeler = heuristic_labeler or HeuristicLabeler()
        self.llm_labeler = llm_labeler or LLMLabeler(config=self.config)

    def process_pair(
        self,
        pair: RequirementPair,
        mode: LabelingMode = LabelingMode.HYBRID,
        examples: Optional[Dict[DependencyType, List[RequirementPair]]] = None,
    ) -> RequirementPair:
        """Process a single requirement pair and return a new RequirementPair with provisional provenance.

        Never overwrites existing VERIFIED human annotations.
        """
        # Safeguard: Do not overwrite verified annotations
        if pair.provenance and pair.provenance.status == VerificationStatus.VERIFIED:
            logger.info(f"Pair {pair.id} is human-verified. Skipping provisional labeling.")
            return pair

        heuristic_res: Optional[HeuristicRuleResult] = None
        llm_label: Optional[DependencyType] = None
        llm_conf: Optional[int] = None
        llm_rationale: str = ""

        if mode in (LabelingMode.HEURISTIC, LabelingMode.HYBRID):
            heuristic_res = self.heuristic_labeler.evaluate_pair(pair)

        if mode in (LabelingMode.LLM, LabelingMode.HYBRID):
            llm_label, llm_conf, llm_rationale = self.llm_labeler.label_pair(pair, examples=examples)

        # Combine outputs and determine provisional outcome
        h_label = heuristic_res.suggested_label if heuristic_res else None
        final_provisional_label: Optional[DependencyType] = None
        source: AnnotationSource = AnnotationSource.HEURISTIC
        review_notes = ""
        evidence_parts = []

        if mode == LabelingMode.HEURISTIC:
            final_provisional_label = h_label
            source = AnnotationSource.HEURISTIC
            if heuristic_res:
                evidence_parts.append(f"Heuristic [{heuristic_res.rule_id}]: {heuristic_res.evidence}")

        elif mode == LabelingMode.LLM:
            final_provisional_label = llm_label
            source = AnnotationSource.LLM
            if llm_rationale:
                evidence_parts.append(f"LLM Rationale: {llm_rationale}")

        else:  # HYBRID mode
            if h_label is not None and llm_label is not None:
                if h_label == llm_label:
                    final_provisional_label = h_label
                    source = AnnotationSource.ENSEMBLE
                    evidence_parts.append(f"AGREEMENT (Heuristic & LLM): {h_label.value}")
                    evidence_parts.append(f"Heuristic [{heuristic_res.rule_id}]: {heuristic_res.evidence}")
                    evidence_parts.append(f"LLM Rationale: {llm_rationale}")
                else:
                    # Disagreement -> Keep provisional label as None (Unresolved) for human review
                    final_provisional_label = None
                    source = AnnotationSource.ENSEMBLE
                    review_notes = f"DISAGREEMENT: Heuristic predicted '{h_label.value}', LLM predicted '{llm_label.value}'."
                    evidence_parts.append(f"Heuristic [{heuristic_res.rule_id}]: {h_label.value}")
                    evidence_parts.append(f"LLM Rationale: {llm_label.value} ({llm_rationale})")
            elif h_label is not None:
                final_provisional_label = h_label
                source = AnnotationSource.HEURISTIC
                evidence_parts.append(f"Heuristic [{heuristic_res.rule_id}]: {heuristic_res.evidence}")
            elif llm_label is not None:
                final_provisional_label = llm_label
                source = AnnotationSource.LLM
                evidence_parts.append(f"LLM Rationale: {llm_rationale}")
            else:
                final_provisional_label = None
                source = AnnotationSource.HEURISTIC
                review_notes = "UNRESOLVED: Both heuristic and LLM abstained or failed."

        prov = AnnotationProvenance(
            source=source,
            status=VerificationStatus.PROVISIONAL,  # Always provisional!
            annotator_id=f"provisional_labeler_{mode.value}",
            model_id=self.config.llm_model if mode != LabelingMode.HEURISTIC else None,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            predicted_label=llm_label or h_label,
            accepted_label=final_provisional_label,
            confidence_score=llm_conf,
            evidence=" | ".join(evidence_parts),
            review_notes=review_notes,
            dataset_version="provisional-v1.0",
        )

        return RequirementPair(
            id=pair.id,
            req_a=pair.req_a,
            req_b=pair.req_b,
            dependency_type=final_provisional_label,
            provenance=prov,
        )

    def process_batch(
        self,
        candidates: List[RequirementPair],
        mode: LabelingMode = LabelingMode.HYBRID,
        source_identity: str = "Candidates_Dataset",
    ) -> Tuple[List[RequirementPair], LabelingRunManifest]:
        """Process a batch of candidate pairs and return labeled pairs + audit manifest."""
        labeled_pairs: List[RequirementPair] = []

        labeled_count = 0
        agreed_count = 0
        disagreed_count = 0
        unresolved_count = 0
        failed_count = 0

        for candidate in candidates:
            try:
                processed = self.process_pair(candidate, mode=mode)
                labeled_pairs.append(processed)

                if processed.dependency_type is not None:
                    labeled_count += 1

                prov = processed.provenance
                if prov:
                    if "AGREEMENT" in (prov.evidence or ""):
                        agreed_count += 1
                    elif "DISAGREEMENT" in (prov.review_notes or ""):
                        disagreed_count += 1
                    elif processed.dependency_type is None:
                        unresolved_count += 1

            except Exception as e:
                logger.error(f"Failed to process candidate {candidate.id}: {e}")
                failed_count += 1
                # Preserve candidate in failed state
                failed_prov = AnnotationProvenance(
                    source=AnnotationSource.HEURISTIC,
                    status=VerificationStatus.PROVISIONAL,
                    annotator_id="provisional_labeler_error",
                    review_notes=f"FAILED: Execution error {str(e)}"
                )
                labeled_pairs.append(
                    RequirementPair(
                        id=candidate.id,
                        req_a=candidate.req_a,
                        req_b=candidate.req_b,
                        dependency_type=None,
                        provenance=failed_prov
                    )
                )

        manifest = LabelingRunManifest(
            source_identity=source_identity,
            input_candidate_count=len(candidates),
            mode=mode,
            labeled_count=labeled_count,
            agreed_count=agreed_count,
            disagreed_count=disagreed_count,
            unresolved_count=unresolved_count,
            failed_count=failed_count,
        )

        logger.info(
            f"Provisional labeling complete ({len(candidates)} candidates). "
            f"Labeled: {labeled_count}, Agreed: {agreed_count}, Disagreed: {disagreed_count}, Unresolved: {unresolved_count}."
        )
        return labeled_pairs, manifest

    @staticmethod
    def save_provisional_dataset(
        pairs: List[RequirementPair],
        manifest: LabelingRunManifest,
        output_pairs_file: str,
        output_manifest_file: Optional[str] = None
    ) -> None:
        """Save provisional labeled dataset and run manifest to JSON files."""
        os.makedirs(os.path.dirname(os.path.abspath(output_pairs_file)), exist_ok=True)

        data = [p.model_dump(mode="json") for p in pairs]
        with open(output_pairs_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        if output_manifest_file is None:
            base, ext = os.path.splitext(output_pairs_file)
            output_manifest_file = f"{base}_manifest.json"

        with open(output_manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(mode="json"), f, indent=2)

        logger.info(f"Saved {len(pairs)} provisional pairs to '{output_pairs_file}'.")
