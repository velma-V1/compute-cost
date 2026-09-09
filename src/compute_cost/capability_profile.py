"""Evidence-only capability and weakness profile derivation.

The profile deliberately avoids scalar capability scores and causal diagnoses.
It summarizes what the campaign proved, where the raw frontier ends, which
interventions demonstrably extend it, and where evidence is absent or fragile.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


EVIDENCE_STATES = ("PROVEN", "PARTIAL", "UNCERTAIN", "UNTESTED")


def _state_for_family(
    family_id: str,
    coverage: dict[str, Any],
    policy: dict[str, Any],
) -> str:
    coverage_row = (coverage.get("families") or {}).get(family_id)
    if isinstance(coverage_row, dict):
        value = coverage_row.get("state")
        if value in EVIDENCE_STATES:
            return str(value)
    policy_row = (policy.get("families") or {}).get(family_id)
    if isinstance(policy_row, dict):
        value = policy_row.get("coverage_state")
        if value in EVIDENCE_STATES:
            return str(value)
    return "UNTESTED"


def _family_failures(family_id: str, atlas: dict[str, Any]) -> tuple[list[str], int, int]:
    signatures: list[str] = []
    model_failure_count = 0
    invalid_or_non_model = 0
    for failure in atlas.get("failures") or []:
        if not isinstance(failure, dict) or str(failure.get("family_id")) != family_id:
            continue
        valid_model_failure = (
            failure.get("failure_origin") == "MODEL_FAILURE"
            and failure.get("valid_for_capability") is True
        )
        if not valid_model_failure:
            invalid_or_non_model += 1
            continue
        model_failure_count += 1
        signature = failure.get("failure_signature")
        subtype = signature.get("subtype") if isinstance(signature, dict) else None
        if isinstance(subtype, str) and subtype and subtype != "unresolved" and subtype not in signatures:
            signatures.append(subtype)
    return signatures, model_failure_count, invalid_or_non_model


def _negative_compound_risk(value: dict[str, Any]) -> bool:
    for risk in value.get("compound_risks") or []:
        if not isinstance(risk, dict):
            continue
        penalty = risk.get("composition_penalty")
        if isinstance(penalty, (int, float)) and not isinstance(penalty, bool) and float(penalty) < 0:
            return True
    return False


def _signals(
    *,
    evidence_state: str,
    reliable_floor: Any,
    first_failure: Any,
    high_extension: Any,
    recovery: Any,
    robustness: Any,
    composition_sensitive: bool,
) -> list[str]:
    if evidence_state != "PROVEN":
        return ["EVIDENCE_GAP"]

    signals: list[str] = []
    if isinstance(first_failure, int) and not isinstance(first_failure, bool):
        signals.append("RAW_FRONTIER_BOUNDARY")
    if (
        isinstance(high_extension, int)
        and not isinstance(high_extension, bool)
        and (
            not isinstance(reliable_floor, int)
            or isinstance(reliable_floor, bool)
            or high_extension > reliable_floor
        )
    ):
        signals.append("EFFORT_SENSITIVE")
    if isinstance(recovery, dict) and recovery:
        signals.append("RECOVERY_EXTENDS_CAPABILITY")
    if isinstance(robustness, str) and robustness.upper() not in {"ROBUST", "STABLE"}:
        signals.append("ROBUSTNESS_RISK")
    if composition_sensitive:
        signals.append("COMPOSITION_SENSITIVE")
    return signals


def build_capability_profile(
    model: str,
    frontiers: dict[str, Any],
    coverage: dict[str, Any],
    value_map: dict[str, Any],
    failure_atlas: dict[str, Any],
    operating_policy: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact, lossless decision profile from finalized evidence artifacts."""
    frontier_families = frontiers.get("families") or {}
    coverage_families = coverage.get("families") or {}
    value_families = value_map.get("families") or {}
    policy_families = operating_policy.get("families") or {}
    atlas_family_ids = {
        str(row.get("family_id"))
        for row in (failure_atlas.get("failures") or [])
        if isinstance(row, dict) and row.get("family_id") is not None
    }
    family_ids = sorted(
        set(map(str, frontier_families))
        | set(map(str, coverage_families))
        | set(map(str, value_families))
        | set(map(str, policy_families))
        | atlas_family_ids
    )

    families: dict[str, Any] = {}
    state_counts: Counter[str] = Counter()
    for family_id in family_ids:
        frontier = frontier_families.get(family_id)
        frontier = frontier if isinstance(frontier, dict) else {}
        value = value_families.get(family_id)
        value = value if isinstance(value, dict) else {}
        state = _state_for_family(family_id, coverage, operating_policy)
        state_counts[state] += 1

        raw_config = value.get("cheapest_proven_raw_config")
        raw_config = raw_config if isinstance(raw_config, dict) else {}
        recovery = value.get("minimum_proven_recovery")
        signatures, model_failures, invalid_or_non_model = _family_failures(family_id, failure_atlas)
        composition_sensitive = _negative_compound_risk(value)

        reliable_floor = frontier.get("reliable_floor")
        first_failure = frontier.get("first_failure_level")
        high_extension = value.get("high_effort_extension_to")
        robustness = value.get("robustness")

        families[family_id] = {
            "evidence_state": state,
            "raw_reliable_through": reliable_floor,
            "first_raw_failure": first_failure,
            "cheapest_proven_reasoning_effort": raw_config.get("reasoning_effort"),
            "baseline_median_wall_clock_s": (
                (value.get("baseline_medium_cost") or {}).get("median_wall_clock_s")
                if isinstance(value.get("baseline_medium_cost"), dict)
                else None
            ),
            "cheapest_proven_median_wall_clock_s": raw_config.get("median_wall_clock_s"),
            "high_effort_extension_to": high_extension,
            "minimum_proven_recovery": recovery if isinstance(recovery, dict) else None,
            "robustness": robustness,
            "compound_risks": list(value.get("compound_risks") or []),
            "observed_failure_signatures": signatures,
            "valid_model_failure_count": model_failures,
            "invalid_or_non_model_failure_count": invalid_or_non_model,
            "signals": _signals(
                evidence_state=state,
                reliable_floor=reliable_floor,
                first_failure=first_failure,
                high_extension=high_extension,
                recovery=recovery,
                robustness=robustness,
                composition_sensitive=composition_sensitive,
            ),
        }

    return {
        "schema_version": 1,
        "model": model,
        "taxonomy_version": frontiers.get("taxonomy_version"),
        "measurement_policy": {
            "scalar_capability_score": False,
            "failure_signatures_are_causal_diagnoses": False,
            "evidence_gap_is_model_weakness": False,
            "invalid_or_non_model_failures_are_model_weaknesses": False,
        },
        "summary": {
            "families_total": len(family_ids),
            "proven": state_counts["PROVEN"],
            "partial": state_counts["PARTIAL"],
            "uncertain": state_counts["UNCERTAIN"],
            "untested": state_counts["UNTESTED"],
        },
        "families": families,
    }
