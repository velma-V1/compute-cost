"""Frontier-local robustness testing for proven raw or recovered capability.

Robustness spends calls only at an already useful operating point: a proven
recovery when one exists, otherwise the MEDIUM-effort reliable floor.  Every
initial perturbation branches from the same proven parent and changes exactly
one controlled variable.  A valid failure is sufficient evidence of fragility
for that perturbation; a passing perturbation must reproduce before it is
called robust.
"""

from __future__ import annotations

import copy
from typing import Any

from .characterization import execute_experiment
from .experiments import ExperimentSpec, make_experiment_id


PROMPT_PERTURBATIONS = {
    "prompt_wording": (
        "The wording below is a semantically equivalent presentation of the same task. "
        "Preserve all stated constraints and return only the requested answer."
    ),
    "format_pressure": (
        "Treat the requested output format as strict: do not add prose, labels, fences, "
        "or fields that the task did not request."
    ),
    "distractor_noise": (
        "Irrelevant context may be present. Ignore facts that do not participate in the "
        "task and answer only from the relevant premises and constraints."
    ),
}

FAMILY_PERTURBATIONS = {
    "strict_structured_output": ("prompt_wording", "format_pressure"),
    "format_robustness": ("prompt_wording", "format_pressure"),
    "distractor_noise_resistance": ("prompt_wording", "distractor_noise"),
    "lost_in_middle_resistance": ("prompt_wording", "distractor_noise"),
    "context_retrieval": ("prompt_wording", "distractor_noise"),
    "context_reasoning": ("prompt_wording", "distractor_noise"),
}


def select_robustness_targets(
    frontiers: dict[str, Any],
    recovery_map: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Prefer the minimum proven recovery; otherwise use the raw reliable floor."""
    selected: dict[str, dict[str, Any]] = {}
    recovery_families = recovery_map.get("families") or {}
    for family_id, frontier in sorted((frontiers.get("families") or {}).items()):
        recovered = recovery_families.get(family_id) if isinstance(recovery_families, dict) else None
        if isinstance(recovered, dict):
            recovery_level = recovered.get("minimum_successful_recovery")
            recovery_difficulty = recovered.get("difficulty_level")
            if (
                isinstance(recovery_level, str)
                and recovery_level
                and isinstance(recovery_difficulty, int)
                and not isinstance(recovery_difficulty, bool)
            ):
                selected[str(family_id)] = {
                    "level": int(recovery_difficulty),
                    "source": f"RECOVERY_{recovery_level}",
                    "recovery_level": recovery_level,
                }
                continue

        reliable_floor = frontier.get("reliable_floor") if isinstance(frontier, dict) else None
        if isinstance(reliable_floor, int) and not isinstance(reliable_floor, bool):
            selected[str(family_id)] = {
                "level": int(reliable_floor),
                "source": "BASELINE_RELIABLE_FLOOR",
                "recovery_level": None,
            }
    return selected


def perturbation_plan(family_id: str, *, max_perturbations: int) -> list[str]:
    """Return a small family-relevant perturbation set, never a broad random sweep."""
    if max_perturbations <= 0:
        return []
    planned = FAMILY_PERTURBATIONS.get(str(family_id), ("prompt_wording", "seed"))
    return list(planned[:max_perturbations])


def perturbation_suffix(kind: str) -> str:
    """Return an oracle-free prompt perturbation instruction."""
    if kind not in PROMPT_PERTURBATIONS:
        raise ValueError(f"unsupported prompt robustness perturbation: {kind}")
    return PROMPT_PERTURBATIONS[kind]


def _spec_from_row(row: dict[str, Any]) -> ExperimentSpec | None:
    raw = row.get("experiment")
    if not isinstance(raw, dict):
        return None
    try:
        return ExperimentSpec(**raw)
    except (TypeError, ValueError):
        return None


def _valid_pass(row: dict[str, Any]) -> tuple[bool, bool]:
    classification = row.get("classification") or {}
    valid = classification.get("valid_for_capability") is True
    passed = valid and classification.get("result_class") == "ANSWER_CORRECT"
    return valid, passed


def _fixture_index(cases: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for case in cases:
        if case.get("robustness_eligible") is False:
            continue
        family_id = str(case.get("family_id") or case.get("category") or "")
        level = case.get("difficulty_level")
        if not family_id or isinstance(level, bool) or not isinstance(level, int):
            continue
        key = (family_id, int(level))
        if key in index:
            raise ValueError(f"duplicate robustness fixture for family {family_id} level {level}")
        index[key] = case
    return index


def _matching_rows(
    rows: list[dict[str, Any]],
    family_id: str,
    level: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        experiment = row.get("experiment") or {}
        family = experiment.get("task_family") or experiment.get("task_id")
        if family == family_id and experiment.get("difficulty_level") == level:
            selected.append(row)
    return selected


def _latest_passing_spec(
    rows: list[dict[str, Any]],
    family_id: str,
    level: int,
    *,
    effort: str | None = None,
    recovery_level: str | None = None,
) -> ExperimentSpec | None:
    found: list[ExperimentSpec] = []
    for row in _matching_rows(rows, family_id, level):
        valid, passed = _valid_pass(row)
        if not valid or not passed:
            continue
        spec = _spec_from_row(row)
        if spec is None:
            continue
        if effort is not None and (spec.reasoning_effort or "medium") != effort:
            continue
        if recovery_level is not None and spec.recovery_level != recovery_level:
            continue
        found.append(spec)
    return None if not found else found[-1]


def _recovery_step(
    recovery_map: dict[str, Any],
    family_id: str,
    recovery_level: str,
) -> dict[str, Any] | None:
    family = (recovery_map.get("families") or {}).get(family_id)
    if not isinstance(family, dict):
        return None
    for step in family.get("steps", []) or []:
        if (
            isinstance(step, dict)
            and step.get("level") == recovery_level
            and step.get("status") == "SUCCESS"
        ):
            return step
    return None


def _target_parent(
    target: dict[str, Any],
    family_id: str,
    base_rows: list[dict[str, Any]],
    reasoning_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
) -> ExperimentSpec | None:
    level = int(target["level"])
    recovery_level = target.get("recovery_level")
    if recovery_level is None:
        return _latest_passing_spec(base_rows, family_id, level, effort="medium")
    if recovery_level == "R2":
        return (
            _latest_passing_spec(recovery_rows, family_id, level, recovery_level="R2")
            or _latest_passing_spec(reasoning_rows, family_id, level, effort="high")
        )
    return _latest_passing_spec(
        recovery_rows,
        family_id,
        level,
        recovery_level=str(recovery_level),
    )


def _target_case(
    fixture: dict[str, Any],
    target: dict[str, Any],
    recovery_map: dict[str, Any],
) -> dict[str, Any] | None:
    case = copy.deepcopy(fixture)
    recovery_level = target.get("recovery_level")
    if recovery_level in {None, "R2"}:
        return case
    family_id = str(fixture.get("family_id") or fixture.get("category"))
    step = _recovery_step(recovery_map, family_id, str(recovery_level))
    intervention = step.get("intervention") if isinstance(step, dict) else None
    suffix = intervention.get("suffix") if isinstance(intervention, dict) else None
    if not isinstance(suffix, str) or not suffix:
        return None
    case["prompt"] = f"{fixture['prompt']}\n\nRECOVERY INSTRUCTION:\n{suffix}"
    return case


def _perturbed_case(case: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "seed":
        return copy.deepcopy(case)
    modified = copy.deepcopy(case)
    modified["prompt"] = (
        f"{case['prompt']}\n\nROBUSTNESS PERTURBATION:\n{perturbation_suffix(kind)}"
    )
    return modified


def _perturbation_spec(
    parent: ExperimentSpec,
    *,
    sequence: int,
    fixture_id: str,
    kind: str,
    replication: bool,
) -> ExperimentSpec:
    if replication:
        changed_variable = "replication"
        seed = parent.seed
        prompt_variant = parent.prompt_variant
    elif kind == "seed":
        changed_variable = "seed"
        seed = parent.seed + 1 if parent.seed < 2_147_483_647 else parent.seed - 1
        prompt_variant = parent.prompt_variant
    else:
        changed_variable = "prompt_variant"
        seed = parent.seed
        prompt_variant = f"{parent.prompt_variant}+robust-{kind}"

    return ExperimentSpec(
        experiment_id=make_experiment_id(
            sequence,
            fixture_id,
            f"robust-{kind}-replicate" if replication else f"robust-{kind}",
        ),
        parent_experiment_id=parent.experiment_id,
        task_id=parent.task_id,
        task_family=parent.task_family,
        difficulty_level=parent.difficulty_level,
        hypothesis=(
            f"replicate {kind} robustness probe"
            if replication
            else f"test {kind} robustness at proven operating point"
        ),
        changed_variable=changed_variable,
        thinking_mode=parent.thinking_mode,
        reasoning_effort=parent.reasoning_effort,
        generation_budget=parent.generation_budget,
        context_request=parent.context_request,
        temperature=parent.temperature,
        seed=seed,
        prompt_variant=prompt_variant,
        # Robustness is a new experiment phase, not another recovery attempt.
        # Recovery ancestry remains preserved by parent_experiment_id and target metadata.
        recovery_level=None,
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
            reason="robustness experiment added",
        )


def _execute(
    runner: Any,
    case: dict[str, Any],
    spec: ExperimentSpec,
    parent: ExperimentSpec,
    *,
    perturbation: str,
    target_source: str,
) -> dict[str, Any]:
    label = f"{spec.task_family} L{spec.difficulty_level} robustness {perturbation}"
    _add_progress_task(runner, label)
    if hasattr(runner, "_progress_begin"):
        runner._progress_begin(label)
    try:
        row = execute_experiment(runner, case, spec, parent=parent)
    finally:
        if hasattr(runner, "_progress_complete"):
            runner._progress_complete(label)
    valid, passed = _valid_pass(row)
    runner.store.append_jsonl(
        "robustness-observations.jsonl",
        {
            "family_id": spec.task_family,
            "difficulty_level": spec.difficulty_level,
            "perturbation": perturbation,
            "target_source": target_source,
            "passed": None if not valid else passed,
            "valid_for_capability": valid,
            "result_class": str((row.get("classification") or {}).get("result_class")),
            "experiment_id": spec.experiment_id,
            "parent_experiment_id": spec.parent_experiment_id,
            "changed_variable": spec.changed_variable,
            "prompt_variant": spec.prompt_variant,
            "seed": spec.seed,
        },
    )
    return row


def _summarize(
    kind: str,
    rows: list[dict[str, Any]],
    *,
    repeats: int,
    reliable_threshold: float,
) -> dict[str, Any]:
    valid_rows = [row for row in rows if _valid_pass(row)[0]]
    pass_count = sum(1 for row in valid_rows if _valid_pass(row)[1])
    fail_count = len(valid_rows) - pass_count
    pass_rate = None if not valid_rows else pass_count / len(valid_rows)
    success = (
        len(valid_rows) >= repeats
        and pass_rate is not None
        and pass_rate >= reliable_threshold
    )
    if fail_count > 0:
        status = "FAILED"
    elif success:
        status = "SUCCESS"
    else:
        status = "UNCERTAIN"
    return {
        "perturbation": kind,
        "status": status,
        "observation_count": len(rows),
        "valid_count": len(valid_rows),
        "invalid_count": len(rows) - len(valid_rows),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": pass_rate,
        "required_repeats": repeats,
        "reliable_threshold": reliable_threshold,
        "experiment_ids": [
            str(row.get("experiment", {}).get("experiment_id"))
            for row in rows
            if row.get("experiment", {}).get("experiment_id") is not None
        ],
    }


def run_robustness_lab(
    runner: Any,
    cases: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    reasoning_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    frontiers: dict[str, Any],
    recovery_map: dict[str, Any],
    *,
    sequence_start: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Probe bounded robustness at each family's proven operating point."""
    assert runner.store is not None
    cfg = runner.config.get("robustness_lab", {})
    if cfg.get("enabled") is False:
        return [], {
            "schema_version": 1,
            "model": runner.model,
            "enabled": False,
            "families": {},
        }, sequence_start

    repeats = int(cfg.get("repeats", 2))
    max_perturbations = int(cfg.get("max_perturbations_per_family", 2))
    max_attempts = int(cfg.get("max_attempts_per_perturbation", max(2, repeats * 2)))
    if repeats <= 0:
        raise ValueError("robustness_lab.repeats must be positive")
    if max_perturbations <= 0:
        raise ValueError("robustness_lab.max_perturbations_per_family must be positive")
    if max_attempts < repeats:
        raise ValueError("robustness_lab.max_attempts_per_perturbation must be >= repeats")
    reliable_threshold = float(
        runner.config.get("capability_campaign", {}).get("reliable_threshold", 0.90)
    )

    fixtures = _fixture_index(cases)
    targets = select_robustness_targets(frontiers, recovery_map)
    generated: list[dict[str, Any]] = []
    families: dict[str, Any] = {}
    sequence = sequence_start

    for family_id, target in targets.items():
        level = int(target["level"])
        fixture = fixtures.get((family_id, level))
        parent = _target_parent(
            target,
            family_id,
            base_rows,
            reasoning_rows,
            recovery_rows,
        )
        target_case = None if fixture is None else _target_case(fixture, target, recovery_map)
        if fixture is None or parent is None or target_case is None:
            families[family_id] = {
                "target": copy.deepcopy(target),
                "status": "SKIPPED",
                "reason": "MISSING_FIXTURE_PARENT_OR_RECOVERY_INSTRUCTION",
                "perturbations": [],
                "robustness": "UNCERTAIN",
            }
            continue

        perturbations: list[dict[str, Any]] = []
        for kind in perturbation_plan(family_id, max_perturbations=max_perturbations):
            case = _perturbed_case(target_case, kind)
            rows: list[dict[str, Any]] = []
            latest = parent
            valid_count = 0
            valid_failure = False

            for attempt in range(max_attempts):
                sequence += 1
                spec = _perturbation_spec(
                    latest if attempt > 0 else parent,
                    sequence=sequence,
                    fixture_id=str(fixture["id"]),
                    kind=kind,
                    replication=attempt > 0,
                )
                row = _execute(
                    runner,
                    case,
                    spec,
                    latest if attempt > 0 else parent,
                    perturbation=kind,
                    target_source=str(target["source"]),
                )
                rows.append(row)
                generated.append(row)
                latest = spec
                valid, passed = _valid_pass(row)
                if not valid:
                    continue
                valid_count += 1
                if not passed:
                    valid_failure = True
                    break
                if valid_count >= repeats:
                    break

            summary = _summarize(
                kind,
                rows,
                repeats=repeats,
                reliable_threshold=reliable_threshold,
            )
            # A valid failure is decisive fragility evidence even if it occurs on
            # the first call; no additional repeats are spent after that point.
            if valid_failure:
                summary["status"] = "FAILED"
            perturbations.append(summary)

        statuses = [item["status"] for item in perturbations]
        if any(status == "FAILED" for status in statuses):
            robustness = "FRAGILE"
        elif perturbations and all(status == "SUCCESS" for status in statuses):
            robustness = "ROBUST"
        else:
            robustness = "UNCERTAIN"
        families[family_id] = {
            "target": copy.deepcopy(target),
            "parent_experiment_id": parent.experiment_id,
            "perturbations": perturbations,
            "robustness": robustness,
        }

    robustness_map = {
        "schema_version": 1,
        "model": runner.model,
        "measurement_policy": {
            "target_selection": "PROVEN_RECOVERY_ELSE_BASELINE_RELIABLE_FLOOR",
            "single_controlled_variable_per_initial_perturbation": True,
            "successful_perturbations_require_replication": True,
            "valid_failure_stops_perturbation_early": True,
            "oracle_material_in_perturbations": False,
        },
        "required_repeats": repeats,
        "reliable_threshold": reliable_threshold,
        "max_perturbations_per_family": max_perturbations,
        "max_attempts_per_perturbation": max_attempts,
        "families": families,
    }
    return generated, robustness_map, sequence
