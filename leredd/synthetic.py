"""Synthetic fixture utilities and synthetic-data isolation helpers.

Provides deterministic test fixtures for review workflows (disagreements, low confidence,
abstentions, parsing failures, ambiguous pairs, missing metadata, and six categories).

CORE PRINCIPLE:
Synthetic fixtures are strictly designated for testing and review workflow verification.
They must NEVER enter production ground truth, benchmark evaluations, or example retrieval pools.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    RequirementPair,
    VerificationStatus,
)

if TYPE_CHECKING:
    from leredd.review_queue import ReviewItem

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
    return sorted(
        f for f in os.listdir(fdir) if f.endswith(".json") and f != "manifest.json"
    )


def load_synthetic_fixture(
    name_or_path: str, fixtures_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Load and validate raw synthetic fixture JSON.

    Enforces that 'synthetic': true and 'source_identity': 'synthetic-review-tests'
    are explicitly declared in the fixture, and that no simulated human decisions exist.
    """
    fdir = fixtures_dir or get_synthetic_fixtures_dir()
    if os.path.isabs(name_or_path):
        target_path = name_or_path
    elif os.path.exists(name_or_path):
        target_path = os.path.abspath(name_or_path)
    else:
        filename = (
            name_or_path if name_or_path.endswith(".json") else f"{name_or_path}.json"
        )
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

    for raw_pair in data.get("pairs", []):
        if isinstance(raw_pair, Mapping):
            if raw_pair.get("human_decision") is not None:
                raise ValueError(
                    f"Invalid synthetic fixture '{target_path}': simulated human_decision is forbidden."
                )
            prov = raw_pair.get("provenance")
            if isinstance(prov, Mapping) and (
                prov.get("status") == VerificationStatus.VERIFIED.value
                or prov.get("source") == AnnotationSource.HUMAN.value
                or prov.get("reviewer_id")
            ):
                raise ValueError(
                    f"Invalid synthetic fixture '{target_path}': simulated human verification provenance is forbidden."
                )

    return data


def is_synthetic(obj: Any) -> bool:
    """Return whether a record or any nested record is explicitly synthetic."""
    if obj is None:
        return False
    if isinstance(obj, Mapping):
        if obj.get("synthetic") is True:
            return True
        identity = obj.get("source_identity")
        if identity == SYNTHETIC_SOURCE_IDENTITY or (
            isinstance(identity, str) and identity.startswith("synthetic:")
        ):
            return True
        if obj.get("source") == "synthetic":
            return True
        return any(is_synthetic(obj.get(key)) for key in ("provenance", "req_a", "req_b"))

    if getattr(obj, "synthetic", False) is True:
        return True
    identity = getattr(obj, "source_identity", None)
    if identity == SYNTHETIC_SOURCE_IDENTITY or (
        isinstance(identity, str) and identity.startswith("synthetic:")
    ):
        return True
    if getattr(obj, "source", None) == "synthetic":
        return True
    return any(
        is_synthetic(getattr(obj, key, None))
        for key in ("provenance", "req_a", "req_b")
    )


def synthetic_source_identity(*records: Any) -> str | None:
    """Return the first source identity attached to a synthetic record."""
    for record in records:
        if record is None:
            continue
        identity = (
            record.get("source_identity")
            if isinstance(record, Mapping)
            else getattr(record, "source_identity", None)
        )
        if isinstance(identity, str) and identity.strip():
            return identity.strip()
        nested = (
            (record.get(key) for key in ("provenance", "req_a", "req_b"))
            if isinstance(record, Mapping)
            else (getattr(record, key, None) for key in ("provenance", "req_a", "req_b"))
        )
        for child in nested:
            identity = (
                (
                    child.get("source_identity")
                    if isinstance(child, Mapping)
                    else getattr(child, "source_identity", None)
                )
                if child is not None
                else None
            )
            if isinstance(identity, str) and identity.strip():
                return identity.strip()
    return None


def load_synthetic_pairs(
    name_or_path: str, fixtures_dir: Optional[str] = None
) -> List[RequirementPair]:
    """Load a synthetic fixture and parse into a list of RequirementPair models.

    All generated RequirementPair instances and nested requirements/provenance are
    guaranteed to have synthetic=True, source_identity='synthetic-review-tests', and
    preserved structured review fields on AnnotationProvenance.
    """
    data = load_synthetic_fixture(name_or_path, fixtures_dir=fixtures_dir)
    raw_pairs = data.get("pairs", [])
    pairs: List[RequirementPair] = []

    for item in raw_pairs:
        pair_dict = dict(item)
        pair_dict["synthetic"] = True
        pair_dict["source_identity"] = SYNTHETIC_SOURCE_IDENTITY

        for req_key in ("req_a", "req_b"):
            if isinstance(pair_dict.get(req_key), dict):
                req_copy = dict(pair_dict[req_key])
                req_copy["synthetic"] = True
                req_copy["source_identity"] = SYNTHETIC_SOURCE_IDENTITY
                pair_dict[req_key] = req_copy

        prov_dict = dict(pair_dict.get("provenance") or {})
        prov_dict["synthetic"] = True
        prov_dict["source_identity"] = SYNTHETIC_SOURCE_IDENTITY

        reasons = list(pair_dict.get("review_reasons") or [])
        if reasons and not prov_dict.get("structured_review_reasons"):
            prov_dict["structured_review_reasons"] = [str(r) for r in reasons]
        if "explicit_flag" in reasons:
            prov_dict["explicit_review_flag"] = True
        if "llm_failure" in reasons or pair_dict.get("llm_raw_response"):
            if pair_dict.get("llm_prediction") is None:
                prov_dict["llm_failed"] = True
                prov_dict.setdefault(
                    "llm_error",
                    str(pair_dict.get("llm_rationale") or "Malformed LLM response"),
                )

        if "heuristic_prediction" in pair_dict and prov_dict.get("heuristic_label") is None:
            prov_dict["heuristic_label"] = pair_dict.get("heuristic_prediction")
        if "heuristic_evidence" in pair_dict and prov_dict.get("heuristic_evidence") is None:
            prov_dict["heuristic_evidence"] = pair_dict.get("heuristic_evidence")
        if "llm_prediction" in pair_dict and prov_dict.get("llm_label") is None:
            prov_dict["llm_label"] = pair_dict.get("llm_prediction")
        if "llm_rationale" in pair_dict and prov_dict.get("llm_rationale") is None:
            prov_dict["llm_rationale"] = pair_dict.get("llm_rationale")
        if "llm_raw_response" in pair_dict and prov_dict.get("llm_raw_response") is None:
            prov_dict["llm_raw_response"] = pair_dict.get("llm_raw_response")
        if "llm_confidence" in pair_dict and prov_dict.get("confidence_score") is None:
            prov_dict["confidence_score"] = pair_dict.get("llm_confidence")

        pair_dict["provenance"] = AnnotationProvenance.model_validate(prov_dict)
        pairs.append(RequirementPair.model_validate(pair_dict))

    return pairs


def load_synthetic_review_items(
    name_or_path: str, fixtures_dir: Optional[str] = None
) -> List[ReviewItem]:
    """Load a synthetic fixture and transform into ReviewItem models for review queue testing."""
    from leredd.review_queue import ReviewQueueConfig, ReviewQueueManager

    pairs = load_synthetic_pairs(name_or_path, fixtures_dir=fixtures_dir)
    manager = ReviewQueueManager(
        config=ReviewQueueConfig(minimum_requirement_characters=0)
    )
    items: List[ReviewItem] = []
    for pair in pairs:
        created = manager.create_review_item(pair)
        if created is not None:
            items.append(created)
    return items
