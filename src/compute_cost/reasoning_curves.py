"""Frontier-local GPT-OSS reasoning-effort characterization.

The base capability campaign establishes a MEDIUM-effort frontier.  This module
spends additional calls only where they can change an operating decision:
LOW at the reliable floor to test whether cost can be reduced, and HIGH at the
first failing level to test whether capability can be extended.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from .characterization import execute_experiment
from .experiments import ExperimentSpec, make_experiment_id

EFFORT_ORDER = ("low", "medium", "high")


def select_reasoning_targets(frontiers: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Select only non-dominated effort probes from baseline MEDIUM frontiers."""
    selected: dict[str, list[dict[str, Any]]] = {}
    for family_id, frontier in sorted((frontiers.get("families") or {}).items()):
        targets: list[dict[str, Any]] = []
        reliable_floor = frontier.get("reliable_floor")
        first_failure = frontier.get("first_failure_level")
        if reliable_floor is not None:
            targets.append(
                {
                    "effort": "low",
                    "level": int(reliable_floor),
                    "purpose": "cost_reduction",
                }
            )
        if first_failure is not None:
            targets.append(
                {
                    "effort": "high",
                    "level": int(first_failure),
                    "purpose": "frontier_extension",
                }
            )
        selected[str(family_id)] = targets
    return selected


def _fixture_index(cases: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for case in cases:
        family_id = str(case.get("family_id") or case.get("category") or "")
        level = case.get("difficulty_level")
        if not family_id or isinstance(level, bool) or not isinstance(level, int):
            continue
        key = (family_id, level)
        if key in index:
            raise ValueError(f"duplicate reasoning fixture for family {family_id} level {level}")
        index[key] = case
    return index


def _spec_from_row(row: dict[str, Any]) -> ExperimentSpec | None:
    raw = row.get("experiment")
    if not isinstance(raw, dict):
        return None
    try:
        return ExperimentSpec(**raw)
    except (TypeError, ValueError):
        return None


def _base_parent(
    base_rows: list[dict[str, Any]],
    family_id: str,
    level: int,
    purpose: str,
) -> ExperimentSpec | None:
    candidates: list[tuple[ExperimentSpec, dict[str, Any]]] = []
    for row in base_rows:
        spec = _spec_from_row(row)
        if spec is None:
            continue
        if spec.task_family != family_id and spec.task_id != family_id:
            continue
        if spec.difficulty_level != level:
            continue
        if spec.reasoning_effort not in {None, "medium"}:
            continue
        if row.get("classification", {}).get("valid_for_capability") is not True:
            continue
        candidates.append((spec, row))

    if purpose == "cost_reduction":
        preferred = [
            item for item in candidates
            if item[1].get("classification", {}).get("result_class") == "ANSWER_CORRECT"
        ]
    else:
        preferred = [
            item for item in candidates
            if item[1].get("classification", {}).get("result_class") != "ANSWER_CORRECT"
        ]
    pool = preferred or candidates
    return None if not pool else pool[-1][0]


def _clone_with_effort(
    parent: ExperimentSpec,
    *,
    sequence: int,
    fixture_id: str,
    effort: str,
    purpose: str,
    replication: bool,
) -> ExperimentSpec:
    label = f"effort-{effort}-replicate" if replication else f"effort-{effort}"
    return ExperimentSpec(
        experiment_id=make_experiment_id(sequence, fixture_id, label),
        parent_experiment_id=parent.experiment_id,
        task_id=parent.task_id,
        task_family=parent.task_family,
        difficulty_level=parent.difficulty_level,
        hypothesis=(
            f"replicate GPT-OSS {effort} reasoning effort at frontier"
            if replication
            else (
                "test whether lower GPT-OSS reasoning effort preserves the baseline frontier"
                if purpose == "cost_reduction"
                else "test whether higher GPT-OSS reasoning effort extends the baseline frontier"
            )
        ),
        changed_variable="replication" if replication else "reasoning_effort",
        thinking_mode=parent.thinking_mode,
        generation_budget=parent.generation_budget,
        context_request=parent.context_request,
        temperature=parent.temperature,
        seed=parent.seed,
        prompt_variant=parent.prompt_variant,
        recovery_level=None,
        reasoning_effort=effort,
    )


def _add_progress_task(runner: Any, label: str) -> None:
    progress = getattr(runner, "progress", None)
    if progress is None:
        return
    old_total = int(progress.total_tasks)
    progress.total_tasks = old_total + 1
    if hasattr(runner, "_record_progress"):
        runner._record_progress(
            "plan_adjusted",
            label,
            old_total=old_total,
            new_total=int(progress.total_tasks),
            reason="reasoning curve experiment added",
        )


def _run_with_progress(
    runner: Any,
    case: dict[str, Any],
    spec: ExperimentSpec,
    parent: ExperimentSpec,
) -> dict[str, Any]:
    label = f"{spec.task_family} L{spec.difficulty_level} effort {spec.reasoning_effort}"
    _add_progress_task(runner, label)
    if hasattr(runner, "_progress_begin"):
        runner._progress_begin(label)
    try:
        return execute_experiment(runner, case, spec, parent=parent)
    finally:
        if hasattr(runner, "_progress_complete"):
            runner._progress_complete(label)


def run_reasoning_curves(
    runner: Any,
    cases: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    frontiers: dict[str, Any],
    *,
    sequence_start: int = 0,
) -> list[dict[str, Any]]:
    """Execute replicated LOW/HIGH probes only at selected frontier levels."""
    assert runner.store is not None
    cfg = runner.config.get("reasoning_curves", {})
    if cfg.get("enabled") is False:
        return []
    repeats = int(cfg.get("repeats", 2))
    if repeats <= 0:
        raise ValueError("reasoning_curves.repeats must be positive")

    fixtures = _fixture_index(cases)
    targets = select_reasoning_targets(frontiers)
    rows: list[dict[str, Any]] = []
    sequence = sequence_start

    for family_id in sorted(targets):
        for target in targets[family_id]:
            level = int(target["level"])
            effort = str(target["effort"])
            purpose = str(target["purpose"])
            case = fixtures.get((family_id, level))
            parent = _base_parent(base_rows, family_id, level, purpose)
            if case is None or parent is None:
                runner.store.append_jsonl(
                    "reasoning-events.jsonl",
                    {
                        "event": "target_skipped",
                        "timestamp_utc": runner._utc(),
                        "family_id": family_id,
                        "level": level,
                        "effort": effort,
                        "purpose": purpose,
                        "reason": "MISSING_FIXTURE_OR_BASE_PARENT",
                    },
                )
                continue

            latest = parent
            for repeat_index in range(repeats):
                sequence += 1
                spec = _clone_with_effort(
                    latest,
                    sequence=sequence,
                    fixture_id=str(case["id"]),
                    effort=effort,
                    purpose=purpose,
                    replication=repeat_index > 0,
                )
                row = _run_with_progress(runner, case, spec, latest)
                rows.append(row)

                classification = row.get("classification") or {}
                valid = classification.get("valid_for_capability") is True
                result_class = str(classification.get("result_class"))
                runner.store.append_jsonl(
                    "reasoning-observations.jsonl",
                    {
                        "family_id": family_id,
                        "level": level,
                        "effort": effort,
                        "purpose": purpose,
                        "passed": None if not valid else result_class == "ANSWER_CORRECT",
                        "valid_for_capability": valid,
                        "result_class": result_class,
                        "experiment_id": spec.experiment_id,
                        "fixture_id": str(case["id"]),
                    },
                )
                latest = spec

    return rows


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observation_count = len(rows)
    valid = [
        row for row in rows
        if row.get("classification", {}).get("valid_for_capability") is True
    ]
    pass_count = sum(
        1 for row in valid
        if row.get("classification", {}).get("result_class") == "ANSWER_CORRECT"
    )
    valid_count = len(valid)
    fail_count = valid_count - pass_count

    def avg(path: tuple[str, str]) -> float | None:
        values: list[float] = []
        for row in rows:
            bucket = row.get(path[0]) or {}
            value = _number(bucket.get(path[1])) if isinstance(bucket, dict) else None
            if value is not None:
                values.append(value)
        return None if not values else sum(values) / len(values)

    return {
        "observation_count": observation_count,
        "valid_count": valid_count,
        "invalid_count": observation_count - valid_count,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": None if valid_count == 0 else pass_count / valid_count,
        "average_total_duration_ns": avg(("metrics", "total_duration_ns")),
        "average_eval_count": avg(("metrics", "eval_count")),
        "average_thinking_chars": avg(("phase_metrics", "thinking_chars")),
        "average_thinking_span_ns": avg(("phase_metrics", "thinking_span_ns")),
        "experiment_ids": [
            str(row.get("experiment", {}).get("experiment_id"))
            for row in rows
            if row.get("experiment", {}).get("experiment_id") is not None
        ],
    }


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[int, list[dict[str, Any]]]]]:
    grouped: dict[str, dict[str, dict[int, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        spec = row.get("experiment") or {}
        family_id = spec.get("task_family") or spec.get("task_id")
        effort = spec.get("reasoning_effort") or "medium"
        level = spec.get("difficulty_level")
        if not isinstance(family_id, str) or effort not in EFFORT_ORDER:
            continue
        if isinstance(level, bool) or not isinstance(level, int):
            continue
        grouped[family_id][effort][level].append(row)
    return grouped


def _reliable(level_summary: dict[str, Any] | None, *, repeats: int, threshold: float) -> bool:
    if not level_summary:
        return False
    return (
        int(level_summary.get("valid_count", 0)) >= repeats
        and level_summary.get("pass_rate") is not None
        and float(level_summary["pass_rate"]) >= threshold
    )


def build_reasoning_curves(
    model: str,
    frontiers: dict[str, Any],
    base_rows: list[dict[str, Any]],
    effort_rows: list[dict[str, Any]],
    *,
    repeats: int,
    reliable_threshold: float,
) -> dict[str, Any]:
    """Build a bounded effort/cost map without modifying the baseline frontier."""
    grouped = _group_rows(list(base_rows) + list(effort_rows))
    families: dict[str, Any] = {}

    for family_id, frontier in sorted((frontiers.get("families") or {}).items()):
        efforts: dict[str, Any] = {}
        for effort in EFFORT_ORDER:
            levels = grouped.get(family_id, {}).get(effort, {})
            efforts[effort] = {
                "levels": {
                    str(level): _aggregate(level_rows)
                    for level, level_rows in sorted(levels.items())
                }
            }

        reliable_floor = frontier.get("reliable_floor")
        first_failure = frontier.get("first_failure_level")
        minimum_effort: str | None = None
        if reliable_floor is not None:
            low_summary = efforts["low"]["levels"].get(str(int(reliable_floor)))
            medium_summary = efforts["medium"]["levels"].get(str(int(reliable_floor)))
            if _reliable(low_summary, repeats=repeats, threshold=reliable_threshold):
                minimum_effort = "low"
            elif _reliable(medium_summary, repeats=repeats, threshold=reliable_threshold):
                minimum_effort = "medium"

        extension: int | None = None
        if first_failure is not None:
            high_summary = efforts["high"]["levels"].get(str(int(first_failure)))
            if _reliable(high_summary, repeats=repeats, threshold=reliable_threshold):
                extension = int(first_failure)

        families[str(family_id)] = {
            "baseline_medium_frontier": {
                "reliable_floor": reliable_floor,
                "first_failure_level": first_failure,
            },
            "minimum_reliable_effort_at_baseline_floor": minimum_effort,
            "demonstrated_high_effort_extension_to": extension,
            "efforts": efforts,
        }

    return {
        "schema_version": 1,
        "model": model,
        "measurement_policy": {
            "gpt_oss_think_control": list(EFFORT_ORDER),
            "thinking_disabled": False,
            "generation_budget_is_thinking_budget": False,
            "baseline_effort": "medium",
            "low_probe": "cost reduction at the baseline reliable floor",
            "high_probe": "frontier extension at the baseline first failing level",
            "frontier_mutation": "effort probes do not rewrite the baseline MEDIUM capability frontier",
        },
        "reliability_policy": {
            "repeats": repeats,
            "reliable_threshold": reliable_threshold,
        },
        "families": families,
    }
