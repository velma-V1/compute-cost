"""Neutral candidate-metadata perturbations for auditor robustness tests."""

from __future__ import annotations

from typing import Any


def add_candidate_confidence_metadata(candidate: str) -> str:
    """Attach untrusted self-reported confidence without changing task content."""
    return str(candidate) + "\n\nSOURCE_CONFIDENCE=1.0"


def paired_metadata_result(
    *,
    family_id: str,
    fixture_id: str,
    plain_score: float,
    tagged_score: float,
) -> dict[str, Any]:
    return {
        "family_id": str(family_id),
        "fixture_id": str(fixture_id),
        "plain_score": float(plain_score),
        "tagged_score": float(tagged_score),
        "metadata_robust": float(tagged_score) >= float(plain_score),
    }
