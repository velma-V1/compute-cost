"""Adaptive execution of family-local capability difficulty frontiers."""

from __future__ import annotations

from typing import Any

from .adaptive import AdaptiveDifficultyController, DifficultyObservation
from .capability_ladders import build_ladder_index, resolve_requested_level
from .characterization import execute_experiment
from .experiments import ExperimentSpec, make_experiment_id


def _spec(
    *,
    sequence: int,
    family_id: str,
    fixture: dict[str, Any],
    parent: ExperimentSpec | None,
    changed_variable: str,
    hypothesis: str,
    thinking_mode: bool,
    generation_budget: int,
) -> ExperimentSpec:
    level = int(fixture["difficulty_level"])
    return ExperimentSpec(
        experiment_id=make_experiment_id(
            sequence,
            str(fixture["id"]),
            "capability-replicate" if changed_variable == "replication" else "capability-probe",
        ),
        parent_experiment_id=None if parent is None else parent.experiment_id,
        task_id=family_id,
        task_family=family_id,
        difficulty_level=level,
        hypothesis=hypothesis,
        changed_variable=changed_variable,
        thinking_mode=thinking_mode,
        generation_budget=generation_budget,
        context_request=None,
        temperature=0.0,
        seed=42,
        prompt_variant="base",
        recovery_level=None,
    )


def _record_family_stop(
    runner: Any,
    family_id: str,
    *,
    reason: str,
    experiments: int,
    requested_level: int | None = None,
) -> None:
    row = {
        "event": "family_stop",
        "timestamp_utc": runner._utc(),
        "family_id": family_id,
        "reason": reason,
        "experiments": experiments,
    }
    if requested_level is not None:
        row["requested_level"] = requested_level
    runner.store.append_jsonl("capability-events.jsonl", row)


def _run_with_progress(
    runner: Any,
    family_id: str,
    fixture: dict[str, Any],
    spec: ExperimentSpec,
    parent: ExperimentSpec | None,
) -> dict[str, Any]:
    label = f"{family_id} L{fixture['difficulty_level']}"
    if hasattr(runner, "_progress_begin"):
        runner._progress_begin(label)
    try:
        return execute_experiment(runner, fixture, spec, parent=parent)
    finally:
        if hasattr(runner, "_progress_complete"):
            runner._progress_complete(label)


def run_family_frontier(
    runner: Any,
    family_id: str,
    ladder: dict[int, dict[str, Any]],
    *,
    sequence_start: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Execute only controller-selected fixtures for one family frontier."""
    assert runner.store is not None
    cfg = runner.config["capability_campaign"]
    controller = AdaptiveDifficultyController(
        anchor_level=int(cfg["anchor_level"]),
        jump=int(cfg["jump"]),
        boundary_repeats=int(cfg["boundary_repeats"]),
    )
    thinking_mode = bool(cfg["thinking_mode"])
    generation_budget = int(cfg["generation_budget"])
    max_experiments = int(cfg["max_experiments_per_family"])

    rows: list[dict[str, Any]] = []
    observations: list[DifficultyObservation] = []
    attempted_levels: set[int] = set()
    latest_spec_by_level: dict[int, ExperimentSpec] = {}
    previous_spec: ExperimentSpec | None = None
    sequence = sequence_start

    while len(rows) < max_experiments:
        decision = controller.next(observations)
        if decision.action == "STOP":
            _record_family_stop(
                runner,
                family_id,
                reason=decision.reason,
                experiments=len(rows),
            )
            break
        assert decision.level is not None
        requested_level = int(decision.level)

        if decision.action == "REPLICATE":
            level = requested_level if requested_level in ladder else None
        else:
            level = resolve_requested_level(ladder, requested_level, attempted_levels)

        if level is None:
            _record_family_stop(
                runner,
                family_id,
                reason="MISSING_FIXTURE_COVERAGE",
                experiments=len(rows),
                requested_level=requested_level,
            )
            break

        fixture = ladder[level]
        if decision.action == "REPLICATE":
            parent = latest_spec_by_level.get(level)
            if parent is None:
                raise ValueError(
                    f"cannot replicate family {family_id} level {level} without prior experiment"
                )
            changed_variable = "replication"
        else:
            parent = previous_spec
            changed_variable = "baseline" if parent is None else "difficulty_level"

        sequence += 1
        spec = _spec(
            sequence=sequence,
            family_id=family_id,
            fixture=fixture,
            parent=parent,
            changed_variable=changed_variable,
            hypothesis=decision.reason,
            thinking_mode=thinking_mode,
            generation_budget=generation_budget,
        )
        row = _run_with_progress(runner, family_id, fixture, spec, parent)
        rows.append(row)

        classification = row.get("classification") or {}
        result_class = str(classification.get("result_class"))
        valid = classification.get("valid_for_capability") is True
        passed: bool | None = None if not valid else result_class == "ANSWER_CORRECT"
        observation = {
            "family_id": family_id,
            "level": level,
            "passed": passed,
            "valid_for_capability": valid,
            "result_class": result_class,
            "experiment_id": spec.experiment_id,
            "fixture_id": str(fixture["id"]),
        }
        runner.store.append_jsonl("capability-observations.jsonl", observation)
        observations.append(
            DifficultyObservation(
                level=level,
                passed=passed,
                valid_for_capability=valid,
            )
        )
        attempted_levels.add(level)
        latest_spec_by_level[level] = spec
        previous_spec = spec
    else:
        _record_family_stop(
            runner,
            family_id,
            reason="MAX_EXPERIMENTS_PER_FAMILY",
            experiments=len(rows),
        )

    return rows, sequence


def run_capability_campaign(
    runner: Any,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run adaptive difficulty search independently for every declared fixture family."""
    ladders = build_ladder_index(cases)
    all_rows: list[dict[str, Any]] = []
    sequence = 0
    for family_id, ladder in ladders.items():
        family_rows, sequence = run_family_frontier(
            runner,
            family_id,
            ladder,
            sequence_start=sequence,
        )
        all_rows.extend(family_rows)
    return all_rows
