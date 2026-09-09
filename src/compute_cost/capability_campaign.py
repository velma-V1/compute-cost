"""Adaptive execution of family-local capability difficulty frontiers."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from typing import Any, Callable

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


SnapshotSink = Callable[[dict[str, Any]], None]


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


def _read_retained_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _retained_snapshot(
    runner: Any,
    fixture: dict[str, Any],
    spec: ExperimentSpec,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    """Rehydrate the just-written exact exchange/scoring evidence for a successful probe.

    Failure snapshots are already persisted by ``execute_experiment``.  Successful
    probes are not, so boundary capture reads the exact raw exchange and scorer
    artifacts immediately after execution rather than inventing model evidence.
    """
    store = getattr(runner, "store", None)
    run_dir = getattr(store, "run_dir", None)
    refs = row.get("evidence_refs") or {}
    request_id = refs.get("request_id")
    if run_dir is None or not isinstance(request_id, str) or not request_id:
        return None

    root = Path(run_dir)
    generation = _read_retained_json(root / "raw" / "runtime" / "exchanges" / f"{request_id}.json")
    safe = spec.experiment_id.replace("/", "-").replace("\\", "-")
    scoring = _read_retained_json(root / "raw" / "scoring" / f"characterize-{safe}.json")
    if generation is None or scoring is None:
        return None

    messages = [{"role": "user", "content": str(fixture["prompt"])}]
    options = runner._generation_options(
        fixture,
        {
            "num_predict": spec.generation_budget,
            "temperature": spec.temperature,
            "seed": spec.seed,
        },
    )
    think_request = spec.reasoning_effort if spec.reasoning_effort is not None else spec.thinking_mode
    invocation = {
        "model": runner.model,
        "messages": copy.deepcopy(messages),
        "options": copy.deepcopy(options),
        "stream": True,
        "request_fields": {"think": think_request},
    }
    return {
        "schema_version": 2,
        "created_at_utc": runner._utc(),
        "benchmark_version": runner.suite.get("benchmark_version"),
        "stage": "characterize",
        "model": runner.model,
        "case": copy.deepcopy(fixture),
        "experiment": spec.to_dict(),
        "classification": copy.deepcopy(row.get("classification") or {}),
        "evidence_key": spec.experiment_id,
        "invocation": invocation,
        "generation": generation,
        "scoring": scoring,
        "telemetry_before_failure": list(copy.deepcopy(getattr(runner, "_recent_telemetry", []))),
        "resolved_config": copy.deepcopy(runner.config),
    }


def _execute_with_snapshot(
    runner: Any,
    fixture: dict[str, Any],
    spec: ExperimentSpec,
    parent: ExperimentSpec | None,
    snapshot_sink: SnapshotSink,
) -> dict[str, Any]:
    """Execute once and retain an exact replay payload without changing model-call count."""
    try:
        supports_sink = "snapshot_sink" in inspect.signature(execute_experiment).parameters
    except (TypeError, ValueError):
        supports_sink = False

    if supports_sink:
        return execute_experiment(
            runner,
            fixture,
            spec,
            parent=parent,
            snapshot_sink=snapshot_sink,
        )

    row = execute_experiment(runner, fixture, spec, parent=parent)
    snapshot = _retained_snapshot(runner, fixture, spec, row)
    if snapshot is not None:
        snapshot_sink(snapshot)
    return row


def _run_with_progress(
    runner: Any,
    family_id: str,
    fixture: dict[str, Any],
    spec: ExperimentSpec,
    parent: ExperimentSpec | None,
    *,
    snapshot_sink: SnapshotSink,
) -> dict[str, Any]:
    label = f"{family_id} L{fixture['difficulty_level']}"
    if hasattr(runner, "_progress_begin"):
        runner._progress_begin(label)
    try:
        return _execute_with_snapshot(runner, fixture, spec, parent, snapshot_sink)
    finally:
        if hasattr(runner, "_progress_complete"):
            runner._progress_complete(label)


def _persist_boundary_replays(
    runner: Any,
    family_id: str,
    observations: list[dict[str, Any]],
    snapshots: dict[str, dict[str, Any]],
    *,
    boundary_repeats: int,
    reliable_threshold: float,
    unstable_threshold: float,
) -> None:
    """Persist every valid observation in a fully reproduced transition pair."""
    if not observations or not snapshots:
        return
    frontier = build_capability_frontiers(
        str(runner.suite.get("taxonomy_version") or "unknown"),
        {family_id: observations},
        thresholds={
            "reliable": reliable_threshold,
            "unstable": unstable_threshold,
        },
    )["families"][family_id]
    bracket = frontier.get("transition_bracket")
    if not isinstance(bracket, dict):
        return
    lower = bracket.get("lower_level")
    upper = bracket.get("upper_level")
    if isinstance(lower, bool) or not isinstance(lower, int):
        return
    if isinstance(upper, bool) or not isinstance(upper, int):
        return

    valid_lower = [
        row for row in observations
        if row.get("level") == lower and row.get("valid_for_capability") is True
    ]
    valid_upper = [
        row for row in observations
        if row.get("level") == upper and row.get("valid_for_capability") is True
    ]
    if len(valid_lower) < boundary_repeats or len(valid_upper) < boundary_repeats:
        return

    for role, level, selected in (
        ("lower_reliable", lower, valid_lower),
        ("upper_transition", upper, valid_upper),
    ):
        for observation in selected:
            source_id = observation.get("experiment_id")
            if not isinstance(source_id, str) or not source_id:
                continue
            source = snapshots.get(source_id)
            if not isinstance(source, dict):
                continue
            replay_id = f"{source_id}--boundary"
            path = f"replay/boundaries/{replay_id}.json"
            snapshot = copy.deepcopy(source)
            snapshot["source_experiment_id"] = source_id
            snapshot["boundary"] = {
                "family_id": family_id,
                "role": role,
                "level": level,
                "transition_bracket": copy.deepcopy(bracket),
                "boundary_repeats": boundary_repeats,
            }
            record = runner.store.write_json(
                path,
                snapshot,
                producer="capability-boundary-replay",
                stage="characterize",
                case_id=(snapshot.get("case") or {}).get("id"),
            )
            experiment = snapshot.get("experiment") or {}
            runner.store.append_jsonl(
                "replay/index.jsonl",
                {
                    "replay_id": replay_id,
                    "category": "boundaries",
                    "path": path,
                    "canonical_sha256": record.get("sha256"),
                    "source_experiment_id": source_id,
                    "family_id": family_id,
                    "task_id": experiment.get("task_id"),
                    "difficulty_level": level,
                    "result_class": observation.get("result_class"),
                    "valid_for_capability": True,
                    "recovery_level": experiment.get("recovery_level"),
                    "parent_experiment_id": experiment.get("parent_experiment_id"),
                    "boundary_role": role,
                },
            )


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
    boundary_repeats = int(cfg["boundary_repeats"])
    reliable_threshold = float(cfg.get("reliable_threshold", 0.90))
    unstable_threshold = float(cfg.get("unstable_threshold", 0.40))
    controller = AdaptiveDifficultyController(
        anchor_level=int(cfg["anchor_level"]),
        jump=int(cfg["jump"]),
        boundary_repeats=boundary_repeats,
    )
    thinking_mode = bool(cfg["thinking_mode"])
    reasoning_effort = str(cfg["reasoning_effort"]) if cfg.get("reasoning_effort") is not None else None
    generation_budget = int(cfg["generation_budget"])
    max_experiments = int(cfg["max_experiments_per_family"])

    rows: list[dict[str, Any]] = []
    observations: list[DifficultyObservation] = []
    observation_rows: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, Any]] = {}
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

        def capture(snapshot: dict[str, Any], *, experiment_id: str = spec.experiment_id) -> None:
            snapshots[experiment_id] = copy.deepcopy(snapshot)

        row = _run_with_progress(
            runner,
            family_id,
            fixture,
            spec,
            parent,
            snapshot_sink=capture,
        )
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
        observation_rows.append(observation)
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

    _persist_boundary_replays(
        runner,
        family_id,
        observation_rows,
        snapshots,
        boundary_repeats=boundary_repeats,
        reliable_threshold=reliable_threshold,
        unstable_threshold=unstable_threshold,
    )
    return rows, sequence


def _baseline_frontiers(runner: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive the MEDIUM-effort frontier while retaining every declared family."""
    declared_families = set((runner.suite.get("coverage") or {}).keys())
    declared_families.update(
        str(case.get("family_id") or case.get("category"))
        for case in (runner.suite.get("cases") or [])
        if case.get("family_id") or case.get("category")
    )
    family_observations: dict[str, list[dict[str, Any]]] = {
        str(family_id): [] for family_id in sorted(declared_families)
    }
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