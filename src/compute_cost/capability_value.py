"""Evidence-only phase cost rollups and demonstrated capability value summaries."""

from __future__ import annotations

import copy
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _complete_sum(rows: list[dict[str, Any]], bucket: str, field: str, scale: float = 1.0) -> tuple[float | int | None, dict[str, int]]:
    values: list[float] = []
    for row in rows:
        source = row.get(bucket) or {}
        value = _number(source.get(field)) if isinstance(source, dict) else None
        if value is not None:
            values.append(value)
    coverage = {"observed": len(values), "expected": len(rows)}
    if len(values) != len(rows):
        return None, coverage
    total = sum(values) / scale
    if scale == 1.0 and all(float(value).is_integer() for value in values):
        return int(total), coverage
    return total, coverage


def rollup_experiment_cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up observed runtime/token counters without converting missing values to zero."""
    rows = list(rows)
    generated, generated_cov = _complete_sum(rows, "metrics", "eval_count")
    prompt, prompt_cov = _complete_sum(rows, "metrics", "prompt_eval_count")
    generation_s, generation_cov = _complete_sum(rows, "metrics", "eval_duration_ns", 1_000_000_000.0)
    runtime_s, runtime_cov = _complete_sum(rows, "metrics", "total_duration_ns", 1_000_000_000.0)
    latency_s, latency_cov = _complete_sum(rows, "timing", "client_latency_ns", 1_000_000_000.0)
    return {
        "model_calls": len(rows),
        "generated_tokens": generated,
        "prompt_tokens": prompt,
        "generation_seconds": generation_s,
        "total_runtime_seconds": runtime_s,
        "client_latency_seconds": latency_s,
        "coverage": {
            "generated_tokens": generated_cov,
            "prompt_tokens": prompt_cov,
            "generation_seconds": generation_cov,
            "total_runtime_seconds": runtime_cov,
            "client_latency_seconds": latency_cov,
        },
    }


def _compound_exposure(family_id: str, compound_map: dict[str, Any]) -> list[dict[str, Any]]:
    exposure: list[dict[str, Any]] = []
    for compound_id, row in sorted((compound_map.get("compounds") or {}).items()):
        required = row.get("capabilities_required") or []
        if family_id not in required:
            continue
        exposure.append(
            {
                "compound_id": compound_id,
                "expected_component_frontier": row.get("expected_component_frontier"),
                "observed_compound_frontier": row.get("observed_compound_frontier"),
                "composition_penalty": row.get("composition_penalty"),
            }
        )
    return exposure


def build_capability_value_map(
    model: str,
    *,
    base_rows: list[dict[str, Any]],
    reasoning_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
    compound_rows: list[dict[str, Any]],
    frontiers: dict[str, Any],
    reasoning_curves: dict[str, Any] | None = None,
    recovery_map: dict[str, Any] | None = None,
    robustness_map: dict[str, Any] | None = None,
    compound_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe demonstrated effects and measured call cost without inventing a scalar score."""
    reasoning_curves = reasoning_curves or {"families": {}}
    recovery_map = recovery_map or {"families": {}}
    robustness_map = robustness_map or {"families": {}}
    compound_map = compound_map or {"compounds": {}}
    phases = {
        "baseline": rollup_experiment_cost(base_rows),
        "reasoning": rollup_experiment_cost(reasoning_rows),
        "recovery": rollup_experiment_cost(recovery_rows),
        "robustness": rollup_experiment_cost(robustness_rows),
        "compound": rollup_experiment_cost(compound_rows),
    }
    total_rows = list(base_rows) + list(reasoning_rows) + list(recovery_rows) + list(robustness_rows) + list(compound_rows)

    families: dict[str, Any] = {}
    for family_id, frontier in sorted((frontiers.get("families") or {}).items()):
        reliable_floor = frontier.get("reliable_floor")
        first_failure = frontier.get("first_failure_level")
        reasoning = copy.deepcopy((reasoning_curves.get("families") or {}).get(family_id) or {})
        extension = reasoning.get("demonstrated_high_effort_extension_to")
        if isinstance(extension, int) and isinstance(reliable_floor, int):
            reasoning["frontier_extension_levels"] = extension - reliable_floor
        else:
            reasoning["frontier_extension_levels"] = None
        families[family_id] = {
            "baseline_frontier": {
                "reliable_floor": reliable_floor,
                "first_failure_level": first_failure,
            },
            "reasoning": reasoning,
            "recovery": copy.deepcopy((recovery_map.get("families") or {}).get(family_id) or {}),
            "robustness": copy.deepcopy((robustness_map.get("families") or {}).get(family_id) or {}),
            "compound_exposure": _compound_exposure(family_id, compound_map),
        }

    return {
        "schema_version": 1,
        "model": model,
        "measurement_policy": {
            "cost_basis": "MEASURED_EXPERIMENT_RUNTIME_AND_TOKEN_COUNTERS",
            "value_basis": "DEMONSTRATED_CAPABILITY_EFFECTS_ONLY",
            "scalar_value_score": False,
        },
        "phases": phases,
        "total_cost": rollup_experiment_cost(total_rows),
        "families": families,
    }
