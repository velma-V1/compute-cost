"""Zero-model-call Test 1.2 preflight planning.

This module computes the mechanism x capability-family matrix before the
campaign spends inference clock.  It is sizing/selection logic only: proof-grade
replication remains owned by Test 2.
"""

from __future__ import annotations

import copy
from typing import Any

from .test12_campaign import (
    COLLECTION_ACTIVE_SECONDS,
    DEFAULT_TEST12_CONFIG,
    PHASES,
    TEST2_CAPABILITY_FAMILIES,
    _estimated_physical_calls,
    _semantic_mechanism_descriptor,
    build_intervention_bank,
    mechanism_applicability,
)

MATRIX_PHASES = frozenset({
    "capability_family_manufacturing_floor",
    "mechanism_coverage_floor",
    "real_tool_execution",
    "failure_phenotype_replay",
    "dose_activation_boundaries",
    "negative_transfer_sentinels",
    "information_gain_reserve",
})

APPLICABILITY_STATES = ("structural_no", "unbuilt", "yes", "untested")
EFFECT_STATES = (
    "verified",
    "conditional",
    "null_verified",
    "null_censored",
    "harmful",
    "unknown",
)


def _semantic_catalog(
    source: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    interventions = build_intervention_bank(
        source,
        max_source_recipes=int(cfg.get("max_source_recipes", 8)),
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for intervention in interventions:
        key = str(_semantic_mechanism_descriptor(intervention)["mechanism_key"])
        grouped.setdefault(key, []).append(intervention)

    result: dict[str, dict[str, Any]] = {}
    for key, members in sorted(grouped.items()):
        ordered = sorted(
            members,
            key=lambda row: (
                _estimated_physical_calls(row),
                str(row.get("id") or ""),
            ),
        )
        representative = ordered[0]
        result[key] = {
            "mechanism_key": key,
            "representative_intervention_id": str(
                representative.get("id") or ""
            ),
            "category": str(representative.get("category") or "UNKNOWN"),
            "estimated_physical_calls_per_application": int(
                _estimated_physical_calls(representative)
            ),
            "implementation_status": str(
                representative.get("implementation_status") or "BUILT"
            ),
            "representative": copy.deepcopy(representative),
            "variant_count": len(members),
        }
    return result


def _applicability(
    mechanism: dict[str, Any],
    family: str,
) -> tuple[str, str]:
    if mechanism.get("implementation_status") == "UNBUILT":
        return "unbuilt", "DECLARED_MECHANISM_LACKS_DELIVERY_PLUMBING"
    decision = mechanism_applicability(
        mechanism["representative"],
        family,
    )
    raw = str(decision.get("status") or "UNKNOWN")
    if raw == "APPLICABLE":
        return "yes", str(decision.get("basis") or "")
    if raw == "NOT_APPLICABLE":
        return "structural_no", str(decision.get("basis") or "")
    return "untested", str(decision.get("basis") or "")


def _family_accuracy(
    classifier_report: dict[str, Any] | None,
    family: str,
) -> float:
    payload = (
        ((classifier_report or {}).get("families") or {}).get(family)
        or {}
    )
    value = payload.get("accuracy")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _rank_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    applicability_rank = {
        "yes": 0,
        "untested": 1,
        "unbuilt": 2,
        "structural_no": 3,
    }
    return (
        applicability_rank.get(str(cell["applicability"]), 9),
        -float(cell.get("runtime_family_classifier_accuracy") or 0.0),
        int(cell["estimated_physical_calls_per_application"]),
        str(cell["family_id"]),
        str(cell["mechanism_key"]),
    )


def _allocate(
    eligible: list[dict[str, Any]],
    call_capacity: int,
    *,
    valid_effect_observations: int,
    harm_sentinel_observations: int,
) -> dict[str, Any]:
    selected: list[str] = []
    calls = 0
    for cell in sorted(eligible, key=_rank_key):
        per_application = int(
            cell["estimated_physical_calls_per_application"]
        )
        needed = per_application * (
            int(valid_effect_observations)
            + int(harm_sentinel_observations)
        )
        if calls + needed > call_capacity:
            continue
        calls += needed
        selected.append(str(cell["cell_key"]))
    return {
        "valid_effect_observations_per_cell": int(valid_effect_observations),
        "harm_sentinel_observations_per_cell": int(harm_sentinel_observations),
        "estimated_physical_call_capacity": int(call_capacity),
        "selected_cell_count": len(selected),
        "selected_cell_fraction": (
            len(selected) / len(eligible) if eligible else 0.0
        ),
        "estimated_physical_calls_used": calls,
        "estimated_physical_calls_unspent": max(0, int(call_capacity) - calls),
        "selected_cell_keys": selected,
    }


def build_test12_cell_budget_plan(
    source: dict[str, Any],
    classifier_report: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_TEST12_CONFIG)
    cfg.update(copy.deepcopy(config or {}))
    mechanisms = _semantic_catalog(source, cfg)

    cells: list[dict[str, Any]] = []
    counts = {state: 0 for state in APPLICABILITY_STATES}
    for family in TEST2_CAPABILITY_FAMILIES:
        for mechanism_key, mechanism in mechanisms.items():
            applicability, basis = _applicability(mechanism, family)
            counts[applicability] += 1
            cells.append({
                "cell_key": f"{mechanism_key}|{family}",
                "mechanism_key": mechanism_key,
                "family_id": family,
                "applicability": applicability,
                "applicability_basis": basis,
                "effect": "unknown",
                "conditions": {
                    "status": "UNMEASURED",
                    "predicate": None,
                },
                "cost": {
                    "status": "UNMEASURED",
                    "effect_observation_binding": None,
                    "operating_point": None,
                },
                "harm": {
                    "status": "UNMEASURED",
                    "population": None,
                    "break_rate": None,
                    "confidence_interval": None,
                },
                "composition": {
                    "status": "unknown",
                    "relations": {},
                },
                "estimated_physical_calls_per_application": int(
                    mechanism["estimated_physical_calls_per_application"]
                ),
                "runtime_family_classifier_accuracy": _family_accuracy(
                    classifier_report,
                    family,
                ),
                "compiler_condition_expressible": True,
            })

    eligible = [
        cell for cell in cells
        if cell["applicability"] in {"yes", "untested"}
    ]
    explicit_yes = [
        cell for cell in cells if cell["applicability"] == "yes"
    ]

    matrix_seconds = sum(
        int(seconds)
        for phase, seconds in PHASES
        if phase in MATRIX_PHASES
    )
    expected_calls = int(cfg.get("expected_calls", 5900))
    call_capacity = int(
        round(
            expected_calls
            * (matrix_seconds / max(1, COLLECTION_ACTIVE_SECONDS))
        )
    )

    breadth = _allocate(
        eligible,
        call_capacity,
        valid_effect_observations=1,
        harm_sentinel_observations=1,
    )
    decision_complete = _allocate(
        eligible,
        call_capacity,
        valid_effect_observations=3,
        harm_sentinel_observations=1,
    )
    yes_decision_complete = _allocate(
        explicit_yes,
        call_capacity,
        valid_effect_observations=3,
        harm_sentinel_observations=1,
    )

    preferred = decision_complete
    gating_decision = (
        "BROAD_DECISION_COMPLETE_SWEEP"
        if float(preferred["selected_cell_fraction"]) >= 0.80
        else "DELIBERATE_HIGH_VALUE_SUBSET"
    )

    return {
        "schema_version": 1,
        "analysis_type": "ZERO_MODEL_CALL_MECHANISM_FAMILY_CELL_BUDGET",
        "model_calls_added": 0,
        "runtime_calls_added": 0,
        "family_count": len(TEST2_CAPABILITY_FAMILIES),
        "semantic_mechanism_count": len(mechanisms),
        "potential_cell_count": len(cells),
        "applicability_counts": counts,
        "eligible_cell_count": len(eligible),
        "explicitly_applicable_cell_count": len(explicit_yes),
        "matrix_phase_seconds": matrix_seconds,
        "active_campaign_seconds": COLLECTION_ACTIVE_SECONDS,
        "expected_campaign_physical_calls": expected_calls,
        "estimated_matrix_physical_call_capacity": call_capacity,
        "capacity_scenarios": {
            "breadth_probe": breadth,
            "decision_complete_estimate": decision_complete,
            "explicit_yes_decision_complete_estimate": yes_decision_complete,
        },
        "preferred_selection_scenario": "decision_complete_estimate",
        "gating_decision": gating_decision,
        "scheduler_contract": {
            "mandatory_family_surface_floor_preserved": True,
            "proof_replication_owner": "TEST2",
            "selected_cells_prioritized_after_mandatory_breadth": True,
            "throughput_is_sizing_not_go_no_go": True,
        },
        "schema_contract": {
            "applicability_states": list(APPLICABILITY_STATES),
            "effect_states": list(EFFECT_STATES),
            "conditional_requires_populated_condition_predicate": True,
            "null_verified_distinct_from_null_censored": True,
            "harm_population_required": True,
            "composition_defaults_unknown": True,
            "cost_and_effect_must_share_observation_set": True,
            "cost_and_effect_must_share_operating_point": True,
            "verified_is_scientific_status": True,
            "compilable_is_separate_engineering_status": True,
        },
        "classifier_dependency": {
            "report_type": (
                (classifier_report or {}).get("analysis_type")
            ),
            "fixture_corpus_accuracy": (
                (classifier_report or {}).get("top1_accuracy")
            ),
            "generalization_claim": False,
        },
        "mechanisms": {
            key: {
                name: copy.deepcopy(value)
                for name, value in payload.items()
                if name != "representative"
            }
            for key, payload in mechanisms.items()
        },
        "cells": cells,
    }
