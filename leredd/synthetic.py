"""Synthetic Review Fixture loading and isolation utilities.

Provides deterministic test fixtures for review workflows (disagreements, low confidence,
abstentions, parsing failures, ambiguous pairs, missing metadata, and six categories).

CORE PRINCIPLE:
Synthetic fixtures are strictly designated for testing and review workflow verification.
They must NEVER enter production ground truth, benchmark evaluations, or example retrieval pools.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    DependencyType,
    Requirement,
    RequirementPair,
    VerificationStatus,
)
from leredd.review_queue import QueueStatus, ReviewItem, ReviewReasonCode

SYNTHETIC_SOURCE_IDENTITY = "synthetic-review-tests"

DEFAULT_FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "fixtures",
    "synthetic",
)


def get_synthetic_fixtures_dir() -> str:
    """Return the absolute path to the synthetic fixtures directory."""
    return os.environ.get("LEREDD_SYNTHETIC_FIXTURES_DIR", DEFAULT_FIXTURES_DIR)


def list_synthetic_fixtures(fixtures_dir: Optional[str] = None) -> List[str]:
    """List all available synthetic fixture filenames."""
    fdir = fixtures_dir or get_synthetic_fixtures_dir()
    if not os.path.exists(fdir):
        return []
    manifest_path = os.path.join(fdir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("fixtures", [])
    return [f for f in os.listdir(fdir) if f.endswith(".json") and f != "manifest.json"]


def load_synthetic_fixture(
    name_or_path: str, fixtures_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Load and validate raw synthetic fixture JSON.

    Enforces that 'synthetic': true and 'source_identity': 'synthetic-review-tests'
    are explicitly declared in the fixture.
    """
    fdir = fixtures_dir or get_synthetic_fixtures_dir()
    if os.path.isabs(name_or_path):
        target_path = name_or_path
    elif os.path.exists(name_or_path):
        target_path = os.path.abspath(name_or_path)
    else:
        filename = name_or_path if name_or_path.endswith(".json") else f"{name_or_path}.json"
        target_path = os.path.join(fdir, filename)

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Synthetic fixture not found at '{target_path}'.")

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Synthetic fixture '{target_path}' must be a JSON object.")

    if not data.get("synthetic"):
        raise ValueError(
            f"Invalid synthetic fixture '{target_path}': missing required 'synthetic: true' declaration."
        )

    if data.get("source_identity") != SYNTHETIC_SOURCE_IDENTITY:
        raise ValueError(
            f"Invalid synthetic fixture '{target_path}': 'source_identity' must be '{SYNTHETIC_SOURCE_IDENTITY}'."
        )

    return data


def is_synthetic(obj: Any) -> bool:
    """Return True if the object, pair, or provenance is marked as synthetic."""
    if obj is None:
        return False
    if isinstance(obj, dict):
        if obj.get("synthetic") is True:
            return True
        if obj.get("source_identity") == SYNTHETIC_SOURCE_IDENTITY:
            return True
        prov = obj.get("provenance")
        if isinstance(prov, dict):
            return prov.get("synthetic") is True or prov.get("source") == "synthetic"
        return False

    if getattr(obj, "synthetic", False) is True:
        return True
    if getattr(obj, "source_identity", None) == SYNTHETIC_SOURCE_IDENTITY:
        return True

    prov = getattr(obj, "provenance", None)
    if prov is not None:
        if getattr(prov, "synthetic", False) is True:
            return True
        if getattr(prov, "source_identity", None) == SYNTHETIC_SOURCE_IDENTITY:
            return True

    return False


def load_synthetic_pairs(
    name_or_path: str, fixtures_dir: Optional[str] = None
) -> List[RequirementPair]:
    """Load a synthetic fixture and parse into a list of RequirementPair models.

    All generated RequirementPair instances are guaranteed to have synthetic=True
    and source_identity='synthetic-review-tests'.
    """
    data = load_synthetic_fixture(name_or_path, fixtures_dir=fixtures_dir)
    raw_pairs = data.get("pairs", [])
    pairs: List[RequirementPair] = []

    for item in raw_pairs:
        pair_dict = dict(item)
        pair_dict["synthetic"] = True
        pair_dict["source_identity"] = SYNTHETIC_SOURCE_IDENTITY

        prov_dict = pair_dict.get("provenance")
        if isinstance(prov_dict, dict):
            prov_dict["synthetic"] = True
            prov_dict["source_identity"] = SYNTHETIC_SOURCE_IDENTITY
            pair_dict["provenance"] = AnnotationProvenance.model_validate(prov_dict)

        pairs.append(RequirementPair.model_validate(pair_dict))

    return pairs


def load_synthetic_review_items(
    name_or_path: str, fixtures_dir: Optional[str] = None
) -> List[ReviewItem]:
    """Load a synthetic fixture and transform into ReviewItem models for review queue testing."""
    data = load_synthetic_fixture(name_or_path, fixtures_dir=fixtures_dir)
    raw_pairs = data.get("pairs", [])
    items: List[ReviewItem] = []

    for idx, item in enumerate(raw_pairs, start=1):
        pair_id = item.get("id", f"SYNTH-PAIR-{idx:03d}")
        req_a_data = item.get("req_a", {})
        req_b_data = item.get("req_b", {})

        review_reasons = []
        for r in item.get("review_reasons", []):
            try:
                review_reasons.append(ReviewReasonCode(r))
            except ValueError:
                pass

        prov_dict = item.get("provenance")
        provenance = (
            AnnotationProvenance.model_validate(prov_dict)
            if isinstance(prov_dict, dict)
            else None
        )

        h_pred = item.get("heuristic_prediction")
        l_pred = item.get("llm_prediction")
        p_label = item.get("proposed_label") or l_pred or h_pred

        review_item = ReviewItem(
            queue_id=f"QITEM-{pair_id}",
            pair_id=pair_id,
            req_a_id=req_a_data.get("id", f"REQ-A-{idx}"),
            req_b_id=req_b_data.get("id", f"REQ-B-{idx}"),
            req_a_text=req_a_data.get("text", ""),
            req_b_text=req_b_data.get("text", ""),
            system_name=req_a_data.get("system") or "SyntheticSystem",
            orientation=(
                req_a_data.get("id", f"REQ-A-{idx}"),
                req_b_data.get("id", f"REQ-B-{idx}"),
            ),
            heuristic_prediction=h_pred,
            heuristic_evidence=item.get("heuristic_evidence", ""),
            llm_prediction=l_pred,
            llm_rationale=item.get("llm_rationale", ""),
            llm_confidence=item.get("llm_confidence"),
            proposed_label=p_label,
            provenance=provenance,
            review_reasons=review_reasons,
            review_notes=item.get("llm_rationale") or item.get("heuristic_evidence") or "Synthetic review test pair.",
            status=QueueStatus.PENDING,
            priority=item.get("priority", 3),
            human_decision=None,
            synthetic=True,
            source_identity=SYNTHETIC_SOURCE_IDENTITY,
        )
        items.append(review_item)

    return items
