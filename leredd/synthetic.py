"""Helpers for identifying synthetic/demo records across JSON and model objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SYNTHETIC_SOURCE_IDENTITY = "synthetic-review-tests"


def is_synthetic(record: Any) -> bool:
    """Return whether a record or any nested record is explicitly synthetic.

    Legacy records without the marker remain unchanged. The check follows pair
    requirements and annotation provenance so serialization boundaries cannot
    accidentally drop the synthetic designation.
    """
    if record is None:
        return False

    if isinstance(record, Mapping):
        if record.get("synthetic") is True:
            return True
        identity = record.get("source_identity")
        if identity == SYNTHETIC_SOURCE_IDENTITY or (
            isinstance(identity, str) and identity.startswith("synthetic:")
        ):
            return True
        if record.get("source") == "synthetic":
            return True
        return any(
            is_synthetic(record.get(key))
            for key in ("provenance", "req_a", "req_b")
        )

    if getattr(record, "synthetic", False) is True:
        return True
    identity = getattr(record, "source_identity", None)
    if identity == SYNTHETIC_SOURCE_IDENTITY or (
        isinstance(identity, str) and identity.startswith("synthetic:")
    ):
        return True
    if getattr(record, "source", None) == "synthetic":
        return True

    return any(
        is_synthetic(getattr(record, key, None))
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
                child.get("source_identity")
                if isinstance(child, Mapping)
                else getattr(child, "source_identity", None)
            ) if child is not None else None
            if isinstance(identity, str) and identity.strip():
                return identity.strip()
    return None
