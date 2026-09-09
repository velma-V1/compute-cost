"""Adaptive execution of family-local capability difficulty frontiers."""

from __future__ import annotations

from typing import Any

from .adaptive import AdaptiveDifficultyController, DifficultyObservation
from .capability_ladders import build_ladder_index, resolve_requested_level
from .characterization import execute_experiment
from .compound_lab import run_compound_lab
from .experiments import ExperimentSpec, make_experiment_id
from .failure_atlas import build_failure_atlas
from .frontier import build_capability_frontiers
from .reasoning_curves import build_reasoning_curves, run_reasoning_curves
from .recovery_lab import run_recovery_lab
from .robustness_lab import run_robustness_lab


def _spec(
    *,
    sequence: int,
    family_id: str,
    fixture: dict[str, Any],
    parent: ExperimentSpec | None,
    changed_variable: str,
    hypothesis: str,
    thinking_mode: bool,
    reasoning_effort: str | None,
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
        reasoning_effort=reasoning_effort,
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


def _add_adaptive_progress_task(runner: Any, label: str) -> None:
    progress = getattr(runner, "progress", None)
    if progress is None:
        return
    old_total = int(progress.total_tasks)
    progress.total_tasks = old_total + 1
    runner._record_progress(
        "plan_adjusted",
        label,
        old_total=old_total,
        new_total=int(progress.total_tasks),
        reason="adaptive experiment added",
    )


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
    reasoning_effort = str(cfg["reasoning_effort"]) if cfg.get("reasoning_effort") is not None else None
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
            reasoning_effort=reasoning_effort,
            generation_budget=generation_budget,
        )
        if rows:
            _add_adaptive_progress_task(
                runner,
                f"{family_id} {decision.action.lower()} L{level}",
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


def _baseline_frontiers(runner: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive the MEDIUM-effort frontier from base campaign rows only."""
    family_observations: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        experiment = row.get("experiment") or {}
        family_id = experiment.get("task_family") or experiment.get("task_id")
        level = experiment.get("difficulty_level")
        if not isinstance(family_id, str):
            continue
        if isinstance(level, bool) or not isinstance(level, int):
            continue
        classification = row.get("classification") or {}
        valid = classification.get("valid_for_capability") is True
        result_class = str(classification.get("result_class"))
        family_observations.setdefault(family_id, []).append(
            {
                "family_id": family_id,
                "level": level,
                "passed": None if not valid else result_class == "ANSWER_CORRECT",
                "valid_for_capability": valid,
                "result_class": result_class,
                "experiment_id": experiment.get("experiment_id"),
            }
        )

    cfg = runner.config["capability_campaign"]
    return build_capability_frontiers(
        str(runner.suite.get("taxonomy_version") or "unknown"),
        family_observations,
        thresholds={
            "reliable": float(cfg.get("reliable_threshold", 0.90)),
            "unstable": float(cfg.get("unstable_threshold", 0.40)),
        },
    )


def run_capability_campaign(
    runner: Any,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run frontiers, effort, recovery, robustness, compounds, then final autopsy."""
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

    # Freeze the MEDIUM frontier before any reasoning, recovery, robustness, or
    # compound intervention. Later phases can add evidence but cannot move it.
    base_rows = list(all_rows)
    frontiers = _baseline_frontiers(runner, base_rows)
    effort_rows: list[dict[str, Any]] = []

    curve_cfg = runner.config.get("reasoning_curves")
    if isinstance(curve_cfg, dict) and curve_cfg.get("enabled") is True:
        effort_rows = run_reasoning_curves(
            runner,
            cases,
            base_rows,
            frontiers,
            sequence_start=sequence,
        )
        sequence += len(effort_rows)
        repeats = int(curve_cfg["repeats"])
        reliable_threshold = float(
            runner.config["capability_campaign"].get("reliable_threshold", 0.90)
        )
        reasoning_summary = build_reasoning_curves(
            str(runner.model),
            frontiers,
            base_rows,
            effort_rows,
            repeats=repeats,
            reliable_threshold=reliable_threshold,
        )
        runner.store.write_json(
            "reasoning-curves.json",
            reasoning_summary,
            producer="reasoning-curves",
            stage="report",
        )
        all_rows.extend(effort_rows)

    # Recovery uses only evidence that existed before recovery began. This prevents
    # the recovery atlas from becoming self-referential and allows R0/R2 reuse.
    pre_recovery_atlas = build_failure_atlas(str(runner.model), all_rows)
    recovery_rows: list[dict[str, Any]] = []
    recovery_map: dict[str, Any] = {"schema_version": 1, "families": {}}
    recovery_cfg = runner.config.get("recovery_lab")
    if isinstance(recovery_cfg, dict):
        recovery_rows, recovery_map, sequence = run_recovery_lab(
            runner,
            cases,
            base_rows,
            effort_rows,
            frontiers,
            pre_recovery_atlas,
            sequence_start=sequence,
        )
        runner.store.write_json(
            "recovery-map.json",
            recovery_map,
            producer="recovery-lab",
            stage="report",
        )
        all_rows.extend(recovery_rows)

    # Robustness consumes the frozen frontier and recovery result. It tests the
    # useful operating point but cannot retroactively redefine either one.
    robustness_cfg = runner.config.get("robustness_lab")
    if isinstance(robustness_cfg, dict):
        robustness_rows, robustness_map, sequence = run_robustness_lab(
            runner,
            cases,
            base_rows,
            effort_rows,
            recovery_rows,
            frontiers,
            recovery_map,
            sequence_start=sequence,
        )
        runner.store.write_json(
            "robustness-map.json",
            robustness_map,
            producer="robustness-lab",
            stage="report",
        )
        all_rows.extend(robustness_rows)

    # Compounds are measured against the same frozen MEDIUM component frontiers.
    # This keeps the composition penalty comparable instead of mixing interventions.
    compound_cfg = runner.config.get("compound_lab")
    if isinstance(compound_cfg, dict):
        compound_rows, compound_map, sequence = run_compound_lab(
            runner,
            frontiers,
            sequence_start=sequence,
        )
        runner.store.write_json(
            "compound-map.json",
            compound_map,
            producer="compound-lab",
            stage="report",
        )
        all_rows.extend(compound_rows)

    runner.store.write_json(
        "failure-atlas.json",
        build_failure_atlas(str(runner.model), all_rows),
        producer="failure-atlas",
        stage="report",
    )
    return all_rows
