"""Fixed-core capability execution for full cross-model comparability."""

from __future__ import annotations

import copy
from typing import Any

from .comparison_matrix import fixed_comparison_cells, native_reasoning_conditions
from .experiments import ExperimentSpec, make_experiment_id

TRUNCATION = {"THINK_TRUNCATED", "ANSWER_TRUNCATED"}


def _reasoning_fields(model: str, native_name: str, value: Any) -> tuple[bool, str | None]:
    if model.lower().startswith("gpt-oss"):
        return True, str(value)
    if native_name == "think":
        return bool(value), None
    return False, None


def _remaining_calls(runner: Any) -> int | None:
    ledger = getattr(runner, "_call_ledger", None)
    if ledger is None or not hasattr(ledger, "snapshot"):
        return None
    value = ledger.snapshot().get("remaining_calls")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _can_spend_retry(runner: Any) -> bool:
    remaining = _remaining_calls(runner)
    mandatory = int(getattr(runner, "_mandatory_fixed_remaining", 0) or 0)
    return remaining is None or remaining > mandatory


def _decrement_mandatory(runner: Any) -> None:
    current = getattr(runner, "_mandatory_fixed_remaining", None)
    if isinstance(current, int) and current > 0:
        runner._mandatory_fixed_remaining = current - 1


def _append_comparison_row(runner: Any, row: dict[str, Any]) -> None:
    if getattr(runner, "store", None) is not None and hasattr(runner.store, "append_jsonl"):
        runner.store.append_jsonl("fixed-capability-observations.jsonl", row)


def _append_frontier_observation(
    runner: Any,
    *,
    family: str,
    level: int,
    fixture_id: str,
    row: dict[str, Any],
) -> None:
    classification = row.get("classification") or {}
    valid = classification.get("valid_for_capability") is True
    result_class = str(classification.get("result_class") or "UNKNOWN")
    observation = {
        "family_id": family,
        "level": level,
        "passed": None if not valid else result_class == "ANSWER_CORRECT",
        "valid_for_capability": valid,
        "result_class": result_class,
        "experiment_id": (row.get("experiment") or {}).get("experiment_id"),
        "fixture_id": fixture_id,
        "comparison_scope": "fixed_core",
        "reasoning_role": "BASELINE_MINIMAL",
    }
    if getattr(runner, "store", None) is not None and hasattr(runner.store, "append_jsonl"):
        runner.store.append_jsonl("capability-observations.jsonl", observation)


def run_fixed_capability_matrix(
    runner: Any,
    cases: list[dict[str, Any]],
    *,
    sequence_start: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Run identical L2/L5/L8/L10 fixtures at every supported native reasoning mode.

    Each fixed comparison cell is independent. A retry occurs only after actual
    truncation and changes generation budget only. Diagnostic/adaptive work is not
    performed here and therefore cannot influence fixed-core aggregates.
    """
    import compute_cost.capability_campaign as campaign

    base_budget = int(runner.config.get("capability_campaign", {}).get("generation_budget", 256))
    ceiling = int(runner.config.get("characterization", {}).get("max_generation_budget", 2048))
    ceiling = max(base_budget, ceiling)
    sequence = sequence_start
    rows: list[dict[str, Any]] = []
    conditions = {condition.role: condition for condition in native_reasoning_conditions(runner.model)}

    for cell in fixed_comparison_cells(cases, runner.model):
        if cell["status"] != "READY" or cell["fixture"] is None:
            if getattr(runner, "store", None) is not None and hasattr(runner.store, "append_jsonl"):
                runner.store.append_jsonl("comparison-cells.jsonl", copy.deepcopy(cell))
            # The protected-core plan reserved one physical call for this supported
            # standardized cell. If the fixture is missing, the cell cannot execute;
            # retain that absence as evidence and release its otherwise stranded slot.
            _decrement_mandatory(runner)
            continue

        case = cell["fixture"]
        family = str(cell["family_id"])
        level = int(cell["difficulty_level"])
        condition = conditions[str(cell["reasoning_role"])]
        thinking_mode, reasoning_effort = _reasoning_fields(
            runner.model, condition.native_name, condition.request_value
        )
        budget = base_budget
        parent: ExperimentSpec | None = None
        attempt = 0

        while True:
            attempt += 1
            sequence += 1
            label = f"fixed-{condition.role.lower()}-a{attempt}"
            spec = ExperimentSpec(
                experiment_id=make_experiment_id(sequence, str(case["id"]), label),
                parent_experiment_id=None if parent is None else parent.experiment_id,
                task_id=str(case["id"]),
                task_family=family,
                difficulty_level=level,
                hypothesis=(
                    "fixed comparable reasoning-condition measurement"
                    if parent is None
                    else "retry exact fixed task after observed token-ceiling exhaustion"
                ),
                changed_variable="baseline" if parent is None else "generation_budget",
                thinking_mode=thinking_mode,
                generation_budget=budget,
                context_request=None,
                temperature=0.0,
                seed=42,
                prompt_variant="fixed-core",
                recovery_level="R1" if parent is not None else None,
                reasoning_effort=reasoning_effort,
            )

            runner._call_category_context = (
                "truncation_retry" if parent is not None else "fixed_capability"
            )
            runner._call_family_context = family
            try:
                row = campaign.execute_experiment(runner, case, spec, parent=parent)
            finally:
                runner._call_category_context = None
                runner._call_family_context = None

            if parent is None:
                _decrement_mandatory(runner)

            comparison = {
                "scope": "fixed_core",
                "reasoning_role": condition.role,
                "native_reasoning_control": condition.native_name,
                "native_reasoning_value": condition.request_value,
                "task_id": str(case["id"]),
                "family_id": family,
                "difficulty_level": level,
                "attempt": attempt,
                "is_retry": parent is not None,
            }
            row = copy.deepcopy(row)
            row["comparison"] = comparison
            rows.append(row)
            _append_comparison_row(runner, row)

            if condition.role == "BASELINE_MINIMAL" and parent is None:
                _append_frontier_observation(
                    runner,
                    family=family,
                    level=level,
                    fixture_id=str(case["id"]),
                    row=row,
                )

            result_class = str((row.get("classification") or {}).get("result_class") or "")
            if result_class not in TRUNCATION or budget >= ceiling or not _can_spend_retry(runner):
                break
            next_budget = min(ceiling, budget * 2)
            if next_budget <= budget:
                break
            parent = spec
            budget = next_budget

    return rows, sequence
