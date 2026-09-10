"""Protected full-comparability campaign orchestration.

The matched capability matrix runs first, followed by the matched autonomous
matrix. The runner carries the number of untouched mandatory fixed cells so token
retries may spend reserve only when doing so cannot sacrifice a required core call.
"""

from __future__ import annotations

from typing import Any

from .autonomous_matrix import run_autonomous_matrix
from .comparison_matrix import FIXED_LEVELS, fixed_comparison_cells, native_reasoning_conditions, planned_fixed_core
from .failure_atlas import build_failure_atlas
from .full_campaign import run_fixed_capability_matrix


def _families(cases: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(case.get("family_id") or case.get("category"))
            for case in cases
            if case.get("family_id") or case.get("category")
        }
    )


def _declared_comparison_cells(cases: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    cells = fixed_comparison_cells(cases, model)
    roles = {condition.role for condition in native_reasoning_conditions(model)}

    # Keep the standardized MAX_NATIVE slot explicit for Qwen. think=true is its
    # highest supported condition but it is already the ENHANCED matched condition;
    # duplicating the same native state would create fake evidence.
    if model.lower().startswith("qwen") and "MAX_NATIVE" not in roles:
        index: dict[tuple[str, int], dict[str, Any]] = {}
        for case in cases:
            family = str(case.get("family_id") or case.get("category") or "")
            level = case.get("difficulty_level")
            if family and isinstance(level, int) and not isinstance(level, bool):
                index.setdefault((family, level), case)
        for family in _families(cases):
            for level in FIXED_LEVELS:
                fixture = index.get((family, level))
                cells.append(
                    {
                        "family_id": family,
                        "difficulty_level": level,
                        "task_id": None if fixture is None else str(fixture.get("id")),
                        "fixture": None,
                        "status": "UNSUPPORTED_REASONING_CONDITION",
                        "reasoning_role": "MAX_NATIVE",
                        "native_reasoning_control": "think",
                        "native_reasoning_value": True,
                        "comparison_scope": "fixed_core",
                        "note": "Qwen think=true is already ENHANCED; MAX_NATIVE is not duplicated as a second measurement.",
                    }
                )
    return cells


def run_full_comparability_campaign(
    runner: Any,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Execute the mandatory comparable core before any optional diagnostics."""
    family_count = len(_families(cases))
    autonomous_cfg = runner.config.get("autonomous_simulation") or {}
    scenario_count = int(autonomous_cfg.get("scenario_count", 6))
    turns = int(autonomous_cfg.get("steps_per_scenario", 8))
    plan = planned_fixed_core(
        str(runner.model),
        family_count=family_count,
        scenario_count=scenario_count,
        turns=turns,
    )
    hard_limit = int(runner.config.get("limits", {}).get("max_model_calls_per_run", 700))
    if int(plan["total"]) > hard_limit:
        raise ValueError(
            f"mandatory fixed core exceeds model-call ceiling: {plan['total']}>{hard_limit}"
        )

    runner._mandatory_fixed_remaining = int(plan["total"])
    if getattr(runner, "store", None) is not None:
        runner.store.write_json(
            "fixed-core-plan.json",
            {
                "schema_version": 1,
                "model": str(runner.model),
                "family_count": family_count,
                "reasoning_conditions": [
                    {
                        "role": item.role,
                        "native_name": item.native_name,
                        "request_value": item.request_value,
                    }
                    for item in native_reasoning_conditions(str(runner.model))
                ],
                **plan,
            },
            producer="full-comparability",
            stage="plan",
        )
        runner.store.write_json(
            "comparison-cells.json",
            {
                "schema_version": 1,
                "model": str(runner.model),
                "cells": _declared_comparison_cells(cases, str(runner.model)),
            },
            producer="full-comparability",
            stage="plan",
        )

    capability_rows, sequence = run_fixed_capability_matrix(
        runner, cases, sequence_start=0
    )
    autonomous_rows, autonomous_summary, sequence = run_autonomous_matrix(
        runner, sequence_start=sequence
    )
    runner._mandatory_fixed_remaining = max(
        0, int(getattr(runner, "_mandatory_fixed_remaining", 0) or 0)
    )

    all_rows = capability_rows + autonomous_rows
    if getattr(runner, "store", None) is not None:
        runner.store.write_json(
            "autonomous-simulation.json",
            autonomous_summary,
            producer="autonomous-matrix",
            stage="report",
        )
        runner.store.write_json(
            "failure-atlas.json",
            build_failure_atlas(str(runner.model), all_rows),
            producer="failure-atlas",
            stage="report",
        )
        ledger = getattr(runner, "_call_ledger", None)
        remaining = None
        if ledger is not None and hasattr(ledger, "snapshot"):
            remaining = ledger.snapshot().get("remaining_calls")
        runner.store.write_json(
            "diagnostic-reserve.json",
            {
                "schema_version": 1,
                "state": "NOT_SPENT_IN_FIXED_COMPARABILITY_CORE",
                "remaining_calls": remaining,
                "note": "Reserve is preserved for evidence-triggered follow-up diagnostics and truncation retries; it is excluded from official fixed-core scores.",
            },
            producer="full-comparability",
            stage="report",
        )
    return all_rows
