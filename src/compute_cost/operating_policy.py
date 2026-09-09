"""Compile and evaluate conservative routing policy from characterization evidence."""

from __future__ import annotations

import copy
from typing import Any


DECISION_ORDER = [
    "CHEAPEST_PROVEN_RAW",
    "PROVEN_HIGH_EFFORT_EXTENSION",
    "MINIMUM_PROVEN_RECOVERY",
    "ESCALATE",
]
EFFORT_ORDER = {"low": 0, "medium": 1, "high": 2}


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _coverage_state(frontier: dict[str, Any], boundary_repeats: int) -> str:
    levels = frontier.get("levels") or []
    valid_total = sum(int(row.get("valid_count", 0)) for row in levels if isinstance(row, dict))
    invalid_total = sum(int(row.get("invalid_count", 0)) for row in levels if isinstance(row, dict))
    if valid_total == 0:
        return "UNCERTAIN" if invalid_total else "UNTESTED"

    by_level = {
        int(row["level"]): row
        for row in levels
        if isinstance(row, dict)
        and isinstance(row.get("level"), int)
        and not isinstance(row.get("level"), bool)
    }
    bracket = frontier.get("transition_bracket")
    if isinstance(bracket, dict):
        lower_level = _integer(bracket.get("lower_level"))
        upper_level = _integer(bracket.get("upper_level"))
        lower = by_level.get(lower_level) if lower_level is not None else None
        upper = by_level.get(upper_level) if upper_level is not None else None
        if (
            lower is not None
            and upper is not None
            and int(lower.get("valid_count", 0)) >= boundary_repeats
            and int(upper.get("valid_count", 0)) >= boundary_repeats
        ):
            return "PROVEN"

    max_row = by_level.get(10)
    if (
        max_row is not None
        and max_row.get("label") == "reliable"
        and int(max_row.get("valid_count", 0)) >= boundary_repeats
    ):
        return "PROVEN"

    min_row = by_level.get(0)
    if (
        min_row is not None
        and min_row.get("label") == "failure"
        and int(min_row.get("valid_count", 0)) >= boundary_repeats
    ):
        return "PROVEN"
    return "PARTIAL"


def _family_policy(
    family_id: str,
    frontier: dict[str, Any],
    value: dict[str, Any],
    *,
    boundary_repeats: int,
) -> dict[str, Any]:
    cheapest = copy.deepcopy(value.get("cheapest_proven_raw_config") or {})
    effort = str(cheapest.get("reasoning_effort") or "medium").lower()
    if effort not in EFFORT_ORDER:
        effort = "medium"
    cheapest["reasoning_effort"] = effort

    recovery = copy.deepcopy(value.get("minimum_proven_recovery"))
    if not isinstance(recovery, dict):
        recovery = None

    return {
        "family_id": family_id,
        "coverage_state": _coverage_state(frontier, boundary_repeats),
        "reliable_floor": frontier.get("reliable_floor"),
        "first_failure_level": frontier.get("first_failure_level"),
        "cheapest_proven_raw_config": cheapest,
        "high_effort_extension_to": value.get("high_effort_extension_to"),
        "minimum_proven_recovery": recovery,
        "robustness": value.get("robustness"),
        "compound_risks": copy.deepcopy(value.get("compound_risks") or []),
    }


def build_operating_policy(
    model: str,
    frontiers: dict[str, Any],
    value_map: dict[str, Any],
    *,
    compound_map: dict[str, Any] | None = None,
    boundary_repeats: int = 3,
) -> dict[str, Any]:
    """Compile routing rules without extrapolating beyond demonstrated evidence."""
    if boundary_repeats <= 0:
        raise ValueError("boundary_repeats must be positive")

    frontier_families = frontiers.get("families") or {}
    value_families = value_map.get("families") or {}
    family_ids = sorted(set(frontier_families) | set(value_families))
    families: dict[str, Any] = {}
    for family_id in family_ids:
        frontier = frontier_families.get(family_id) or {
            "levels": [],
            "reliable_floor": None,
            "first_failure_level": None,
        }
        value = value_families.get(family_id) or {}
        families[str(family_id)] = _family_policy(
            str(family_id),
            frontier,
            value,
            boundary_repeats=boundary_repeats,
        )

    compounds: dict[str, Any] = {}
    for compound_id, compound in sorted(((compound_map or {}).get("compounds") or {}).items()):
        if not isinstance(compound, dict):
            continue
        compounds[str(compound_id)] = {
            "compound_id": str(compound_id),
            "capabilities_required": list(compound.get("capabilities_required") or []),
            "expected_component_frontier": compound.get("expected_component_frontier"),
            "observed_compound_frontier": compound.get("observed_compound_frontier"),
            "composition_penalty": compound.get("composition_penalty"),
            "status": compound.get("status"),
        }

    return {
        "schema_version": 1,
        "model": model,
        "taxonomy_version": frontiers.get("taxonomy_version"),
        "boundary_repeats": boundary_repeats,
        "decision_order": list(DECISION_ORDER),
        "policy_guards": {
            "partial_or_unresolved_capability": "ESCALATE",
            "untested_capability": "ESCALATE",
            "recovery_extrapolation": False,
            "component_success_implies_compound_success": False,
            "unmeasured_composition": "ESCALATE",
        },
        "families": families,
        "compounds": compounds,
    }


def _validate_request(capabilities: list[str], difficulty: int) -> list[str]:
    if not isinstance(capabilities, list) or not capabilities:
        raise ValueError("capabilities must be a non-empty list")
    normalized: list[str] = []
    for capability in capabilities:
        if not isinstance(capability, str) or not capability:
            raise ValueError("capabilities must contain non-empty strings")
        if capability not in normalized:
            normalized.append(capability)
    if isinstance(difficulty, bool) or not isinstance(difficulty, int) or not 0 <= difficulty <= 10:
        raise ValueError("difficulty must be an integer 0..10")
    return normalized


def _raw_effort_for_family(family: dict[str, Any], difficulty: int) -> tuple[str, str] | None:
    floor = _integer(family.get("reliable_floor"))
    if floor is not None and difficulty <= floor:
        cheapest = family.get("cheapest_proven_raw_config") or {}
        effort = str(cheapest.get("reasoning_effort") or "medium").lower()
        if effort not in EFFORT_ORDER:
            effort = "medium"
        return effort, "RELIABLE_FLOOR"

    extension = _integer(family.get("high_effort_extension_to"))
    if extension is not None and difficulty <= extension:
        return "high", "HIGH_EFFORT_EXTENSION"
    return None


def _escalate(capabilities: list[str], difficulty: int, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "action": "ESCALATE",
        "capabilities_required": list(capabilities),
        "difficulty": difficulty,
        "reason": reason,
        **extra,
    }


def route_task(
    policy: dict[str, Any],
    capabilities: list[str],
    difficulty: int,
    *,
    compound_id: str | None = None,
) -> dict[str, Any]:
    """Return the cheapest route that is directly supported by retained evidence."""
    required = _validate_request(capabilities, difficulty)
    families = policy.get("families") or {}

    if len(required) > 1 or compound_id is not None:
        if compound_id is None:
            return _escalate(required, difficulty, "COMPOSITION_UNMEASURED")
        compound = (policy.get("compounds") or {}).get(compound_id)
        if not isinstance(compound, dict):
            return _escalate(required, difficulty, "COMPOSITION_UNMEASURED", compound_id=compound_id)
        measured_required = list(compound.get("capabilities_required") or [])
        if set(measured_required) != set(required):
            return _escalate(required, difficulty, "COMPOUND_CAPABILITY_MISMATCH", compound_id=compound_id)
        observed = _integer(compound.get("observed_compound_frontier"))
        if compound.get("status") != "MEASURED" or observed is None:
            return _escalate(required, difficulty, "COMPOSITION_UNMEASURED", compound_id=compound_id)
        if difficulty > observed:
            return _escalate(
                required,
                difficulty,
                "COMPOUND_FRONTIER_EXCEEDED",
                compound_id=compound_id,
                observed_compound_frontier=observed,
                composition_penalty=compound.get("composition_penalty"),
            )

        efforts: list[str] = []
        for capability in required:
            family = families.get(capability)
            if not isinstance(family, dict) or family.get("coverage_state") != "PROVEN":
                return _escalate(required, difficulty, "CAPABILITY_NOT_PROVEN", compound_id=compound_id)
            raw = _raw_effort_for_family(family, difficulty)
            if raw is None:
                return _escalate(required, difficulty, "NO_PROVEN_ROUTE", compound_id=compound_id)
            efforts.append(raw[0])
        effort = max(efforts, key=lambda item: EFFORT_ORDER[item])
        return {
            "action": "RAW",
            "capabilities_required": required,
            "difficulty": difficulty,
            "compound_id": compound_id,
            "reasoning_effort": effort,
            "evidence_basis": "MEASURED_COMPOUND_FRONTIER",
            "observed_compound_frontier": observed,
            "composition_penalty": compound.get("composition_penalty"),
        }

    capability = required[0]
    family = families.get(capability)
    if not isinstance(family, dict) or family.get("coverage_state") != "PROVEN":
        return _escalate(required, difficulty, "CAPABILITY_NOT_PROVEN")

    raw = _raw_effort_for_family(family, difficulty)
    if raw is not None:
        effort, basis = raw
        return {
            "action": "RAW",
            "capabilities_required": required,
            "difficulty": difficulty,
            "reasoning_effort": effort,
            "evidence_basis": basis,
            "robustness": family.get("robustness"),
        }

    recovery = family.get("minimum_proven_recovery")
    if isinstance(recovery, dict):
        recovery_difficulty = _integer(recovery.get("difficulty_level"))
        recovery_level = recovery.get("level")
        if recovery_difficulty == difficulty and isinstance(recovery_level, str) and recovery_level:
            return {
                "action": "RECOVERY",
                "capabilities_required": required,
                "difficulty": difficulty,
                "recovery_level": recovery_level,
                "evidence_basis": "PROVEN_RECOVERY",
                "robustness": family.get("robustness"),
            }

    return _escalate(required, difficulty, "NO_PROVEN_ROUTE")
