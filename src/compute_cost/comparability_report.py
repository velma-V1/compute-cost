"""Zero-call synthesis for the fixed-core capability comparison matrix.

Official capability aggregates are computed only from matched fixed-core logical
cells. Physical retry attempts remain visible as retry deltas but do not create
extra weight in capability denominators. Adaptive/diagnostic work is reported
separately and never contaminates official matched-core scores.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Iterable

NA = {"NOT_APPLICABLE", "UNAVAILABLE", None}


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _avg(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _numeric(value)) is not None]
    return None if not numbers else round(mean(numbers), 3)


def _comparison(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("comparison")
    return value if isinstance(value, dict) else {}


def _experiment(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("experiment")
    return value if isinstance(value, dict) else {}


def _vector(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("score_vector")
    return value if isinstance(value, dict) else {}


def _cell_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    comp = _comparison(row)
    exp = _experiment(row)
    family = str(comp.get("family_id") or exp.get("task_family") or "unknown")
    task_id = str(comp.get("task_id") or exp.get("task_id") or "unknown")
    level = comp.get("difficulty_level", exp.get("difficulty_level", -1))
    level_i = int(level) if isinstance(level, int) and not isinstance(level, bool) else -1
    role = str(comp.get("reasoning_role") or "UNSPECIFIED")
    return family, task_id, level_i, role


def _terminal_fixed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _comparison(row).get("scope") == "fixed_core":
            grouped[_cell_key(row)].append(row)

    terminal: list[dict[str, Any]] = []
    for selected in grouped.values():
        ids = {
            str(_experiment(row).get("experiment_id"))
            for row in selected
            if _experiment(row).get("experiment_id") is not None
        }
        parents = {
            str(_experiment(row).get("parent_experiment_id"))
            for row in selected
            if _experiment(row).get("parent_experiment_id") is not None
        }
        candidates = [
            row
            for row in selected
            if str(_experiment(row).get("experiment_id")) in (ids - parents)
        ]
        terminal.append(candidates[-1] if candidates else selected[-1])
    return sorted(terminal, key=_cell_key)


def _component_macro(rows: list[dict[str, Any]], component: str) -> float | None:
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        family = _cell_key(row)[0]
        value = _numeric(_vector(row).get(component))
        if value is not None:
            by_family[family].append(value)
    family_means = [mean(values) for values in by_family.values() if values]
    return None if not family_means else round(mean(family_means), 3)


def _components(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    names = sorted({key for row in rows for key in _vector(row)})
    return {name: _component_macro(rows, name) for name in names}


def _retry_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {
        str(_experiment(row).get("experiment_id")): row
        for row in rows
        if _experiment(row).get("experiment_id") is not None
    }
    result: list[dict[str, Any]] = []
    for child in rows:
        child_exp = _experiment(child)
        parent_id = child_exp.get("parent_experiment_id")
        if parent_id is None:
            continue
        parent = by_id.get(str(parent_id))
        if parent is None:
            continue
        parent_vector = _vector(parent)
        child_vector = _vector(child)
        score_delta: dict[str, float] = {}
        for name in sorted(set(parent_vector) | set(child_vector)):
            before = _numeric(parent_vector.get(name))
            after = _numeric(child_vector.get(name))
            if before is not None and after is not None:
                score_delta[name] = round(after - before, 6)
        parent_budget = _experiment(parent).get("generation_budget")
        child_budget = child_exp.get("generation_budget")
        budget_delta = None
        if (
            isinstance(parent_budget, int)
            and not isinstance(parent_budget, bool)
            and isinstance(child_budget, int)
            and not isinstance(child_budget, bool)
        ):
            budget_delta = child_budget - parent_budget
        result.append(
            {
                "parent_experiment_id": str(parent_id),
                "child_experiment_id": str(child_exp.get("experiment_id")),
                "family_id": _cell_key(child)[0],
                "task_id": _cell_key(child)[1],
                "difficulty_level": _cell_key(child)[2],
                "reasoning_role": _cell_key(child)[3],
                "generation_budget_delta": budget_delta,
                "result_class_before": (_experiment(parent) and (parent.get("classification") or {}).get("result_class")),
                "result_class_after": (child.get("classification") or {}).get("result_class"),
                "score_delta": score_delta,
            }
        )
    return result


def _task_scorecard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        family, task_id, level, role = _cell_key(row)
        result.append(
            {
                "family_id": family,
                "task_id": task_id,
                "difficulty_level": level,
                "reasoning_role": role,
                "terminal_experiment_id": _experiment(row).get("experiment_id"),
                "result_class": (row.get("classification") or {}).get("result_class"),
                "valid_for_capability": (row.get("classification") or {}).get("valid_for_capability"),
                "score_vector": dict(_vector(row)),
            }
        )
    return result


def _reasoning_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_role[_cell_key(row)[3]].append(row)
    conditions: dict[str, Any] = {}
    for role, selected in sorted(by_role.items()):
        families = sorted({_cell_key(row)[0] for row in selected})
        conditions[role] = {
            "official_attempts": len(selected),
            "family_count": len(families),
            "macro_semantic_score": _component_macro(selected, "semantic_correctness"),
            "component_macro_scores": _components(selected),
            "families": {
                family: {
                    "logical_cells": sum(_cell_key(row)[0] == family for row in selected),
                    "semantic_score": _avg(
                        _vector(row).get("semantic_correctness")
                        for row in selected
                        if _cell_key(row)[0] == family
                    ),
                }
                for family in families
            },
        }
    return {
        "aggregation": "equal-weight family macro over terminal fixed-core logical cells",
        "conditions": conditions,
    }


def _comparison_cell_summary(cells: Iterable[dict[str, Any]] | None) -> dict[str, Any]:
    materialized = list(cells or [])
    unsupported = sum(str(cell.get("status")) == "UNSUPPORTED_REASONING_CONDITION" for cell in materialized)
    missing = sum(str(cell.get("status")) == "MISSING_FIXTURE_COVERAGE" for cell in materialized)
    ready = sum(str(cell.get("status")) == "READY" for cell in materialized)
    return {
        "declared": len(materialized),
        "ready": ready,
        "unsupported": unsupported,
        "missing_fixture": missing,
        "cells": materialized,
    }


def build_comparability_report(
    model: str,
    rows: Iterable[dict[str, Any]],
    *,
    comparison_cells: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build official matched-core aggregates and retry evidence without inference."""
    materialized = list(rows)
    fixed_physical = [row for row in materialized if _comparison(row).get("scope") == "fixed_core"]
    diagnostics = [
        row
        for row in materialized
        if _comparison(row).get("scope") not in {"fixed_core", "fixed_autonomous"}
    ]
    terminals = _terminal_fixed_rows(materialized)
    retries = _retry_deltas(fixed_physical)
    cells = _comparison_cell_summary(comparison_cells)
    return {
        "schema_version": 1,
        "model": model,
        "score_policy": {
            "official_scope": "matched fixed_core logical cells only",
            "retry_policy": "terminal retry attempt represents one logical cell; all physical attempts remain in retry_deltas",
            "diagnostic_policy": "adaptive/diagnostic rows excluded from official aggregates",
            "macro_weighting": "equal weight per family",
        },
        "model_summary": {
            "official_scope": "fixed_core_only",
            "official_attempts": len(terminals),
            "physical_fixed_attempts": len(fixed_physical),
            "retry_attempts": len(retries),
            "diagnostic_rows_excluded": len(diagnostics),
            "macro_semantic_score_by_reasoning": {
                role: block["macro_semantic_score"]
                for role, block in _reasoning_comparison(terminals)["conditions"].items()
            },
        },
        "task_scorecard": _task_scorecard(terminals),
        "reasoning_comparison": _reasoning_comparison(terminals),
        "retry_deltas": retries,
        "comparison_cells": cells,
    }


def render_reasoning_comparison(report: dict[str, Any]) -> str:
    lines = [
        f"# Reasoning Comparison: {report.get('model')}",
        "",
        "Official scores use terminal matched fixed-core logical cells and equal-family macro weighting.",
        "",
        "| Native reasoning role | Macro semantic | Logical cells | Families |",
        "|---|---:|---:|---:|",
    ]
    for role, block in sorted((report.get("reasoning_comparison") or {}).get("conditions", {}).items()):
        lines.append(
            f"| {role} | {block.get('macro_semantic_score')} | {block.get('official_attempts', 0)} | {block.get('family_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "Adaptive/diagnostic observations are excluded from these official aggregates. Retry deltas and every component vector remain available in JSON artifacts.",
            "",
        ]
    )
    return "\n".join(lines)
