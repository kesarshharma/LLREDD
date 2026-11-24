"""Provisional Requirement Dependency Labeling module (Phase 4, TASK-P4-02 & Part 5).

Implements provisional dataset labeling using:
1. HeuristicLabeler: Rule-based baseline detecting explicit references, lexical similarity, and negation/conflict indicators.
2. LLMLabeler: LLM-based classifier using existing LLM infrastructure (and MockLLMClient for offline testing).
3. ProvisionalLabeler: Coordinator combining heuristic and LLM outputs, recording structured agreement/disagreement/failure/abstention fields, and producing auditable provisional dataset records with full provenance.

CORE PRINCIPLE: Machine-generated labels are PROVISIONAL predictions (VerificationStatus.PROVISIONAL).
They are NEVER silently promoted to verified ground-truth annotations.
"""

import datetime
from enum import Enum
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import uuid

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
from leredd.synthetic import is_synthetic, synthetic_source_identity


class LabelingMode(str, Enum):
    """Supported provisional labeling modes."""
    HEURISTIC = "heuristic"
    LLM = "llm"
    MULTI_LLM = "multi_llm"
    HYBRID = "hybrid"


class HeuristicRuleResult(BaseModel):
    """Result of a heuristic rule evaluation."""
    suggested_label: Optional[DependencyType] = None
    rule_id: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str


class LLMLabelResult(BaseModel):
    """Structured result of an LLM provisional labeling evaluation."""
    suggested_label: Optional[DependencyType] = None
    confidence: Optional[int] = Field(default=None, ge=0, le=5)
    rationale: str = ""
    raw_response: Optional[str] = None
    failed: bool = False
    error: Optional[str] = None
    disagreement: bool = False


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
        if "implements" in lower_a or ("subsystem" in lower_b and "include" in lower_b):
            if "subsystem" in lower_a or "include" in lower_a or "implements" in lower_b:
                return HeuristicRuleResult(
                    suggested_label=DependencyType.IMPLEMENTS,
                    rule_id="RULE-REF-IMPLEMENTS",
                    confidence=0.8,
                    evidence="Explicit subsystem architecture or implementation relation detected."
                )

        if (
            "conflicts with" in lower_a
            or "contradicts" in lower_a
            or "conflicts with" in lower_b
            or "contradicts" in lower_b
        ):
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
                evidence=f"Asymmetric negation detected with {len(intersection)} shared terms: {sorted(intersection)[:4]}."
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

    def evaluate_pair(
        self,
        target_pair: RequirementPair,
        examples: Optional[Dict[DependencyType, List[RequirementPair]]] = None,
        srs_context: Optional[List[str]] = None,
    ) -> LLMLabelResult:
        """Classify a requirement pair and return a structured LLMLabelResult."""
        raw_response: Optional[str] = None
        try:
            prompt = self.prompt_builder.build(
                target_pair, retrieved_examples=examples, context_chunks=srs_context
            )
            raw_res = self.llm_client.generate(prompt)
            raw_response = raw_res[0] if isinstance(raw_res, tuple) else raw_res

            # Try JSON parsing first
            try:
                data = json.loads(raw_response)
                if isinstance(data, dict):
                    if data.get("abstain") is True:
                        rat = str(data.get("explanation") or data.get("rationale") or "LLM abstained.")
                        return LLMLabelResult(
                            suggested_label=None,
                            confidence=None,
                            rationale=rat,
                            raw_response=raw_response,
                            failed=False,
                        )
                    dep_str = data.get("dependency_type") or data.get("dependency")
                    conf = (
                        data.get("confidence")
                        if "confidence" in data
                        else data.get("confidence_score")
                    )
                    rat = data.get("explanation") or data.get("rationale", "")
                    if dep_str:
                        norm_label = DependencyType.normalize(str(dep_str))
                        parsed_conf = (
                            max(0, min(5, int(conf)))
                            if conf is not None and not isinstance(conf, bool)
                            else None
                        )
                        return LLMLabelResult(
                            suggested_label=norm_label,
                            confidence=parsed_conf,
                            rationale=str(rat),
                            raw_response=raw_response,
                            failed=False,
                        )
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

            # Fallback to standard ResponseParser requiring explicit Dependency_Type tag
            dep_match = re.search(
                r"(?:^\s*(?:\*\*)?\s*Dependency_Type\s*(?:\*\*)?\s*:\s*)(.*?)\s*$",
                raw_response,
                re.MULTILINE | re.IGNORECASE,
            )
            if not dep_match:
                err_msg = f"Malformed LLM response (missing explicit Dependency_Type tag): {raw_response[:100]}"
                return LLMLabelResult(
                    suggested_label=None,
                    confidence=None,
                    rationale=err_msg,
                    raw_response=raw_response,
                    failed=True,
                    error=err_msg,
                )

            raw_val = dep_match.group(1).strip().strip("*").strip("[]").strip("*").strip()
            if raw_val.upper() in {"ABSTAIN", "NONE", "UNKNOWN"}:
                return LLMLabelResult(
                    suggested_label=None,
                    confidence=None,
                    rationale=f"LLM abstained ({raw_val}).",
                    raw_response=raw_response,
                    failed=False,
                )

            parsed = ResponseParser.parse(raw_response, pair_id=target_pair.id)
            has_conf_header = bool(
                re.search(r"Confidence\s*Score\s*(?:\*\*)?\s*:", raw_response, re.IGNORECASE)
            )
            return LLMLabelResult(
                suggested_label=parsed.dependency_type,
                confidence=parsed.confidence if has_conf_header else None,
                rationale=parsed.rationale,
                raw_response=raw_response,
                failed=False,
            )

        except Exception as e:
            logger.error(f"Error during LLM labeling for pair {target_pair.id}: {e}")
            err_msg = f"LLM provider error: {e}"
            return LLMLabelResult(
                suggested_label=None,
                confidence=None,
                rationale=err_msg,
                raw_response=raw_response,
                failed=True,
                error=err_msg,
            )

    def label_pair(
        self,
        target_pair: RequirementPair,
        examples: Optional[Dict[DependencyType, List[RequirementPair]]] = None,
        srs_context: Optional[List[str]] = None,
    ) -> Tuple[Optional[DependencyType], Optional[int], str]:
        """Classify a requirement pair using LLM client (tuple-returning compatibility API)."""
        res = self.evaluate_pair(target_pair, examples=examples, srs_context=srs_context)
        return res.suggested_label, res.confidence, res.rationale


class MultiModelLLMLabeler:
    """Collect independent model labels; accept only unanimous valid votes."""

    def __init__(self, labelers: List[LLMLabeler]):
        if len(labelers) < 2:
            raise ValueError("Multi-model labeling requires at least two model labelers.")
        self.labelers = list(labelers)
        self.model_ids = [labeler.config.llm_model for labeler in self.labelers]
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("Multi-model labeling requires distinct model identifiers.")
        self.model_id = "ensemble[" + ",".join(self.model_ids) + "]"

    def evaluate_pair(
        self,
        target_pair: RequirementPair,
        examples: Optional[Dict[DependencyType, List[RequirementPair]]] = None,
        srs_context: Optional[List[str]] = None,
    ) -> LLMLabelResult:
        votes: List[Tuple[str, LLMLabelResult]] = []
        for model_id, labeler in zip(self.model_ids, self.labelers):
            res = labeler.evaluate_pair(
                target_pair, examples=examples, srs_context=srs_context
            )
            votes.append((model_id, res))

        vote_evidence = "; ".join(
            f"model={model_id}, label={res.suggested_label.value if res.suggested_label else 'ABSTAIN'}, "
            f"confidence={res.confidence if res.confidence is not None else 'unknown'}, "
            f"rationale={res.rationale}"
            for model_id, res in votes
        )
        labels = [res.suggested_label for _, res in votes]
        any_failed = any(res.failed for _, res in votes)
        if all(label is not None for label in labels) and len(set(labels)) == 1:
            confidences = [res.confidence for _, res in votes if res.confidence is not None]
            confidence = min(confidences) if confidences else None
            return LLMLabelResult(
                suggested_label=labels[0],
                confidence=confidence,
                rationale=f"MULTI_LLM_UNANIMOUS: {vote_evidence}",
                failed=False,
            )

        differing_labels = {label for label in labels if label is not None}
        if len(differing_labels) > 1:
            msg = f"MULTI_LLM_DISAGREEMENT: {vote_evidence}"
            return LLMLabelResult(
                suggested_label=None,
                confidence=None,
                rationale=msg,
                failed=False,
                disagreement=True,
            )
        msg = f"MULTI_LLM_INCOMPLETE: {vote_evidence}"
        return LLMLabelResult(
            suggested_label=None,
            confidence=None,
            rationale=msg,
            failed=any_failed,
            error=msg if any_failed else None,
        )

    def label_pair(
        self,
        target_pair: RequirementPair,
        examples: Optional[Dict[DependencyType, List[RequirementPair]]] = None,
        srs_context: Optional[List[str]] = None,
    ) -> Tuple[Optional[DependencyType], Optional[int], str]:
        res = self.evaluate_pair(target_pair, examples=examples, srs_context=srs_context)
        return res.suggested_label, res.confidence, res.rationale


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
        llm_labeler: Optional[LLMLabeler | MultiModelLLMLabeler] = None,
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
        """Process a single requirement pair and return a new RequirementPair with structured provisional provenance.

        Never overwrites existing VERIFIED human annotations.
        """
        if pair.provenance and pair.provenance.status == VerificationStatus.VERIFIED:
            logger.info(f"Pair {pair.id} is human-verified. Skipping provisional labeling.")
            return pair

        heuristic_res: Optional[HeuristicRuleResult] = None
        llm_res: Optional[LLMLabelResult] = None
        is_multi_model = isinstance(self.llm_labeler, MultiModelLLMLabeler)
        if mode == LabelingMode.MULTI_LLM and not is_multi_model:
            raise ValueError("multi_llm mode requires a MultiModelLLMLabeler.")

        if mode in (LabelingMode.HEURISTIC, LabelingMode.HYBRID):
            heuristic_res = self.heuristic_labeler.evaluate_pair(pair)

        if mode in (LabelingMode.LLM, LabelingMode.MULTI_LLM, LabelingMode.HYBRID):
            if hasattr(self.llm_labeler, "evaluate_pair"):
                llm_res = self.llm_labeler.evaluate_pair(pair, examples=examples)
            else:
                l_lbl, l_cf, l_rt = self.llm_labeler.label_pair(pair, examples=examples)
                failed_hint = l_lbl is None and l_rt.startswith(("Malformed LLM", "LLM provider error"))
                llm_res = LLMLabelResult(
                    suggested_label=l_lbl,
                    confidence=l_cf,
                    rationale=l_rt,
                    failed=failed_hint,
                    error=l_rt if failed_hint else None,
                )

        h_label = heuristic_res.suggested_label if heuristic_res else None
        llm_label = llm_res.suggested_label if llm_res else None
        llm_conf = llm_res.confidence if llm_res else None
        llm_rationale = llm_res.rationale if llm_res else ""
        llm_failed = bool(llm_res and llm_res.failed)
        llm_error = llm_res.error if llm_res else None
        llm_needs_review = bool(
            is_multi_model
            and llm_rationale.startswith(("MULTI_LLM_DISAGREEMENT:", "MULTI_LLM_INCOMPLETE:"))
        )

        final_provisional_label: Optional[DependencyType] = None
        source: AnnotationSource = AnnotationSource.HEURISTIC
        review_notes = ""
        evidence_parts: List[str] = []
        structured_reasons: List[str] = []

        if mode == LabelingMode.HEURISTIC:
            final_provisional_label = h_label
            source = AnnotationSource.HEURISTIC
            if heuristic_res:
                evidence_parts.append(f"Heuristic [{heuristic_res.rule_id}]: {heuristic_res.evidence}")
            if h_label is not None:
                structured_reasons.append("heuristic_only")
            else:
                structured_reasons.append("both_abstained")
                review_notes = "UNRESOLVED: Heuristic abstained."

        elif mode in (LabelingMode.LLM, LabelingMode.MULTI_LLM):
            final_provisional_label = None if llm_needs_review else llm_label
            source = AnnotationSource.ENSEMBLE if is_multi_model else AnnotationSource.LLM
            if llm_rationale:
                evidence_parts.append(f"LLM Rationale: {llm_rationale}")
            if llm_failed:
                structured_reasons.append("llm_failure")
                review_notes = f"FAILED: {llm_error or llm_rationale}"
            elif llm_res and llm_res.disagreement:
                structured_reasons.append("disagreement")
                review_notes = llm_rationale
            elif llm_needs_review:
                structured_reasons.append("both_abstained")
                review_notes = llm_rationale
            elif llm_label is not None:
                if not is_multi_model:
                    structured_reasons.append("llm_only")
            else:
                structured_reasons.append("both_abstained")
                review_notes = "UNRESOLVED: LLM abstained."

        else:  # HYBRID mode
            if llm_needs_review:
                final_provisional_label = None
                source = AnnotationSource.ENSEMBLE
                review_notes = llm_rationale
                if heuristic_res:
                    evidence_parts.append(
                        f"Heuristic [{heuristic_res.rule_id}]: "
                        f"{h_label.value if h_label else 'ABSTAIN'}"
                    )
                evidence_parts.append(f"LLM Votes: {llm_rationale}")
                if llm_failed:
                    structured_reasons.append("llm_failure")
                if llm_res and llm_res.disagreement:
                    structured_reasons.append("disagreement")
                elif not llm_failed:
                    structured_reasons.append("both_abstained")
            elif h_label is not None and llm_label is not None:
                if h_label == llm_label:
                    final_provisional_label = h_label
                    source = AnnotationSource.ENSEMBLE
                    evidence_parts.append(f"AGREEMENT (Heuristic & LLM): {h_label.value}")
                    if heuristic_res:
                        evidence_parts.append(f"Heuristic [{heuristic_res.rule_id}]: {heuristic_res.evidence}")
                    evidence_parts.append(f"LLM Rationale: {llm_rationale}")
                else:
                    final_provisional_label = None
                    source = AnnotationSource.ENSEMBLE
                    structured_reasons.append("disagreement")
                    review_notes = f"DISAGREEMENT: Heuristic predicted '{h_label.value}', LLM predicted '{llm_label.value}'."
                    if heuristic_res:
                        evidence_parts.append(f"Heuristic [{heuristic_res.rule_id}]: {h_label.value}")
                    evidence_parts.append(f"LLM Rationale: {llm_label.value} ({llm_rationale})")
            elif h_label is not None:
                final_provisional_label = h_label
                source = AnnotationSource.HEURISTIC
                structured_reasons.append("heuristic_only")
                if llm_failed:
                    structured_reasons.append("llm_failure")
                    review_notes = f"FAILED: {llm_error or llm_rationale}"
                if heuristic_res:
                    evidence_parts.append(f"Heuristic [{heuristic_res.rule_id}]: {heuristic_res.evidence}")
            elif llm_label is not None:
                final_provisional_label = llm_label
                source = AnnotationSource.ENSEMBLE if is_multi_model else AnnotationSource.LLM
                if not is_multi_model:
                    structured_reasons.append("llm_only")
                evidence_parts.append(f"LLM Rationale: {llm_rationale}")
            else:
                final_provisional_label = None
                if llm_failed:
                    source = AnnotationSource.LLM
                    structured_reasons.append("llm_failure")
                    review_notes = f"FAILED: {llm_error or llm_rationale}"
                    if llm_rationale:
                        evidence_parts.append(f"LLM Error: {llm_rationale}")
                else:
                    source = AnnotationSource.HEURISTIC
                    structured_reasons.append("both_abstained")
                    review_notes = "UNRESOLVED: Both heuristic and LLM abstained or failed."

        existing_explicit_flag = bool(pair.provenance and pair.provenance.explicit_review_flag)
        if existing_explicit_flag and "explicit_flag" not in structured_reasons:
            structured_reasons.append("explicit_flag")

        synth_flag = is_synthetic(pair)
        synth_identity = synthetic_source_identity(pair)

        prov = AnnotationProvenance(
            source=source,
            status=VerificationStatus.PROVISIONAL,  # Always provisional!
            annotator_id=f"provisional_labeler_{mode.value}",
            model_id=(
                getattr(self.llm_labeler, "model_id", self.config.llm_model)
                if mode != LabelingMode.HEURISTIC else None
            ),
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            predicted_label=llm_label or h_label,
            accepted_label=final_provisional_label,
            confidence_score=llm_conf,
            evidence=" | ".join(evidence_parts),
            review_notes=review_notes,
            dataset_version="provisional-v1.0",
            heuristic_label=h_label,
            heuristic_rule_id=heuristic_res.rule_id if heuristic_res else None,
            heuristic_confidence=heuristic_res.confidence if heuristic_res else None,
            heuristic_evidence=heuristic_res.evidence if heuristic_res else None,
            llm_label=llm_label,
            llm_rationale=llm_rationale if mode != LabelingMode.HEURISTIC else None,
            llm_raw_response=llm_res.raw_response if llm_res else None,
            llm_failed=llm_failed,
            llm_error=llm_error,
            explicit_review_flag=existing_explicit_flag,
            structured_review_reasons=structured_reasons,
            synthetic=synth_flag,
            source_identity=synth_identity,
        )

        return RequirementPair(
            id=pair.id,
            req_a=pair.req_a,
            req_b=pair.req_b,
            dependency_type=final_provisional_label,
            provenance=prov,
            synthetic=synth_flag,
            source_identity=synth_identity,
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
                    if prov.llm_failed:
                        failed_count += 1
                    if (
                        (prov.heuristic_label is not None and prov.heuristic_label == prov.llm_label)
                        or "MULTI_LLM_UNANIMOUS" in (prov.evidence or "")
                        or "AGREEMENT" in (prov.evidence or "")
                    ):
                        agreed_count += 1
                    elif (
                        "disagreement" in prov.structured_review_reasons
                        or "DISAGREEMENT" in (prov.review_notes or "")
                    ):
                        disagreed_count += 1
                    elif processed.dependency_type is None and not prov.llm_failed:
                        unresolved_count += 1

            except Exception as e:
                logger.error(f"Failed to process candidate {candidate.id}: {e}")
                failed_count += 1
                synth_flag = is_synthetic(candidate)
                synth_id = synthetic_source_identity(candidate)
                failed_prov = AnnotationProvenance(
                    source=AnnotationSource.HEURISTIC,
                    status=VerificationStatus.PROVISIONAL,
                    annotator_id="provisional_labeler_error",
                    review_notes=f"FAILED: Execution error {e}",
                    llm_failed=True,
                    llm_error=str(e),
                    structured_review_reasons=["llm_failure"],
                    synthetic=synth_flag,
                    source_identity=synth_id,
                )
                labeled_pairs.append(
                    RequirementPair(
                        id=candidate.id,
                        req_a=candidate.req_a,
                        req_b=candidate.req_b,
                        dependency_type=None,
                        provenance=failed_prov,
                        synthetic=synth_flag,
                        source_identity=synth_id,
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
            metadata=(
                {
                    "llm_provider": self.config.llm_provider,
                    "llm_models": self.llm_labeler.model_ids,
                    "aggregation": "unanimous_only",
                }
                if isinstance(self.llm_labeler, MultiModelLLMLabeler)
                else {}
            ),
        )

        logger.info(
            f"Provisional labeling complete ({len(candidates)} candidates). "
            f"Labeled: {labeled_count}, Agreed: {agreed_count}, Disagreed: {disagreed_count}, "
            f"Unresolved: {unresolved_count}, Failed: {failed_count}."
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
            base, _ = os.path.splitext(output_pairs_file)
            output_manifest_file = f"{base}_manifest.json"

        with open(output_manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(mode="json"), f, indent=2)

        logger.info(f"Saved {len(pairs)} provisional pairs to '{output_pairs_file}'.")
