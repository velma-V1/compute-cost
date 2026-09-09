"""Bounded, evidence-reusing recovery experiments for reproduced capability failures.

Recovery candidates are isolated interventions.  Each initial R2-R7 candidate
branches from the same reproduced MEDIUM-effort baseline failure so a successful
change can be attributed to one controlled variable.  Replications branch from
the successful candidate and preserve its controlled fields.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from .characterization import execute_experiment
from .experiments import ExperimentSpec, make_experiment_id

RECOVERY_LEVELS = ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8")


def select_recovery_targets(
    frontiers: dict[str, Any],
    *,
    boundary_repeats: int,
) -> dict[str, int]:
    """Return only family boundaries with a reproduced valid failure."""
    if boundary_repeats <= 0:
        raise ValueError("boundary_repeats must be positive")
    selected: dict[str, int] = {}
    for family_id, frontier in sorted((frontiers.get("families") or {}).items()):
        first_failure = frontier.get("first_failure_level")
        if isinstance(first_failure, bool) or not isinstance(first_failure, int):
            continue
        level_row = next(
            (
                row
                for row in frontier.get("levels", []) or []
                if isinstance(row, dict) and row.get("level") == first_failure
            ),
            None,
        )
        if not isinstance(level_row, dict):
            continue
        valid_count = int(level_row.get("valid_count", 0) or 0)
        fail_count = int(level_row.get("fail_count", 0) or 0)
        pass_count = int(level_row.get("pass_count", 0) or 0)
        if valid_count >= boundary_repeats and fail_count >= boundary_repeats and pass_count == 0:
            selected[str(family_id)] = int(first_failure)
    return selected


def recovery_intervention(
    level: str,
    family_id: str,
    failure_signature: dict[str, Any] | None,
) -> dict[str, str]:
    """Return an oracle-free intervention description for one recovery level."""
    level = str(level).upper()
    subtype = str((failure_signature or {}).get("subtype") or "unresolved")
    family_id = str(family_id)

    if level == "R3":
        return {
            "kind": "prompt_variant",
            "suffix": (
                "Before answering, check whether any ambiguity prevents a uniquely correct response. "
                "If the request is unambiguous, answer it directly without adding commentary."
            ),
        }
    if level == "R4":
        return {
            "kind": "prompt_variant",
            "suffix": (
                f"A prior attempt had the non-causal failure signature `{subtype}`. "
                "Re-check that failure mode before producing the final answer."
            ),
        }
    if level == "R5":
        return {
            "kind": "prompt_variant",
            "suffix": (
                "Decompose the task into the minimum necessary subproblems, solve them in dependency "
                "order, then return only the requested final output."
            ),
        }
    if level == "R6":
        return {
            "kind": "prompt_variant",
            "suffix": (
                "Maintain a compact structured state of premises, constraints, intermediate results, "
                "and the required output format; verify the state before answering."
            ),
        }
    if level == "R7":
        targeted = {
            "formal_logic_deduction": (
                "Verify each logical implication against the stated premises and do not introduce "
                "unstated assumptions before returning the requested answer."
            ),
            "arithmetic_numerical_reasoning": (
                "Recompute the numerical operations independently, including units and signs, before "
                "returning the requested answer."
            ),
            "algebra_quantitative_reasoning": (
                "Verify the algebraic constraints by substitution before returning the requested answer."
            ),
            "strict_structured_output": (
                "Validate every output field, type, delimiter, and exclusivity constraint before answering."
            ),
            "tool_selection": (
                "Verify that the selected tool is the minimum sufficient tool for the requested operation."
            ),
            "tool_argument_correctness": (
                "Validate every tool argument and all cross-field constraints before issuing the call."
            ),
        }.get(
            family_id,
            "Apply a targeted correction to the identified task-family failure mode, verify the correction, "
            "and return only the requested final output.",
        )
        return {"kind": "prompt_variant", "suffix": targeted}
    if level == "R8":
        return {"kind": "unavailable", "reason": "INVERTED_INTERVENTION_NOT_CONFIGURED"}
    raise ValueError(f"unsupported recovery intervention: {level}")


def _spec_from_row(row: dict[str, Any]) -> ExperimentSpec | None:
    raw = row.get("experiment")
    if not isinstance(raw, dict):
        return None
    try:
        return ExperimentSpec(**raw)
    except (TypeError, ValueError):
        return None


def _valid_result(row: dict[str, Any]) -> tuple[bool, bool]:
    classification = row.get("classification") or {}
    valid = classification.get("valid_for_capability") is True
    passed = valid and classification.get("result_class") == "ANSWER_CORRECT"
    return valid, passed


def _rows_for(
    rows: list[dict[str, Any]],
    family_id: str,
    level: int,
    *,
    effort: str | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        spec = row.get("experiment") or {}
        family = spec.get("task_family") or spec.get("task_id")
        if family != family_id or spec.get("difficulty_level") != level:
            continue
        if effort is not None and (spec.get("reasoning_effort") or "medium") != effort:
            continue
        selected.append(row)
    return selected


def _baseline_parent(
    rows: list[dict[str, Any]],
    family_id: str,
    level: int,
) -> ExperimentSpec | None:
    candidates: list[ExperimentSpec] = []
    for row in _rows_for(rows, family_id, level, effort="medium"):
        valid, passed = _valid_result(row)
        if not valid or passed:
            continue
        spec = _spec_from_row(row)
        if spec is not None:
            candidates.append(spec)
    return None if not candidates else candidates[-1]


def _fixture_index(cases: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for case in cases:
        family_id = str(case.get("family_id") or case.get("category") or "")
        level = case.get("difficulty_level")
        if not family_id or isinstance(level, bool) or not isinstance(level, int):
            continue
        key = (family_id, int(level))
        if key in index:
            raise ValueError(f"duplicate recovery fixture for family {family_id} level {level}")
        index[key] = case
    return index


def _failure_signature(
    failure_atlas: dict[str, Any],
    family_id: str,
    level: int,
) -> dict[str, Any]:
    matching = [
        entry
        for entry in failure_atlas.get("failures", []) or []
        if isinstance(entry, dict)
        and entry.get("family_id") == family_id
        and entry.get("difficulty_level") == level
        and entry.get("failure_origin", "MODEL_FAILURE") == "MODEL_FAILURE"
    ]
    if not matching:
        return {"subtype": "unresolved", "causal_claim": False}
    signature = matching[-1].get("failure_signature")
    return (
        copy.deepcopy(signature)
        if isinstance(signature, dict)
        else {"subtype": "unresolved", "causal_claim": False}
    )


def _experiment_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [
        str(row.get("experiment", {}).get("experiment_id"))
        for row in rows
        if row.get("experiment", {}).get("experiment_id") is not None
    ]


def _summarize_step(
    recovery_level: str,
    rows: list[dict[str, Any]],
    *,
    evidence_source: str,
    repeats: int,
    reliable_threshold: float,
) -> dict[str, Any]:
    valid_rows = [row for row in rows if _valid_result(row)[0]]
    pass_count = sum(1 for row in valid_rows if _valid_result(row)[1])
    valid_count = len(valid_rows)
    pass_rate = None if valid_count == 0 else pass_count / valid_count
    reliable = (
        valid_count >= repeats
        and pass_rate is not None
        and pass_rate >= reliable_threshold
    )
    return {
        "level": recovery_level,
        "status": "SUCCESS" if reliable else "FAILED",
        "evidence_source": evidence_source,
        "observation_count": len(rows),
        "valid_count": valid_count,
        "pass_count": pass_count,
        "fail_count": valid_count - pass_count,
        "pass_rate": pass_rate,
        "reliable_threshold": reliable_threshold,
        "required_repeats": repeats,
        "experiment_ids": _experiment_ids(rows),
    }


def _clone_r2(
    parent: ExperimentSpec,
    *,
    sequence: int,
    fixture_id: str,
    replication: bool,
) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=make_experiment_id(sequence, fixture_id, "recovery-r2-replicate" if replication else "recovery-r2"),
        parent_experiment_id=parent.experiment_id,
        task_id=parent.task_id,
        task_family=parent.task_family,
        difficulty_level=parent.difficulty_level,
        hypothesis=(
            "replicate R2 high-reasoning recovery"
            if replication
            else "test whether higher GPT-OSS reasoning effort recovers the reproduced failure"
        ),
        changed_variable="replication" if replication else "reasoning_effort",
        thinking_mode=parent.thinking_mode,
        reasoning_effort="high",
        generation_budget=parent.generation_budget,
        context_request=parent.context_request,
        temperature=parent.temperature,
        seed=parent.seed,
        prompt_variant=parent.prompt_variant,
        recovery_level="R2",
    )


def _clone_prompt_recovery(
    parent: ExperimentSpec,
    *,
    sequence: int,
    fixture_id: str,
    recovery_level: str,
    replication: bool,
) -> ExperimentSpec:
    variant = f"recovery-{recovery_level.lower()}"
    return ExperimentSpec(
        experiment_id=make_experiment_id(
            sequence,
            fixture_id,
            f"{variant}-replicate" if replication else variant,
        ),
        parent_experiment_id=parent.experiment_id,
        task_id=parent.task_id,
        task_family=parent.task_family,
        difficulty_level=parent.difficulty_level,
        hypothesis=(
            f"replicate {recovery_level} recovery"
            if replication
            else f"test isolated {recovery_level} recovery intervention"
        ),
        changed_variable="replication" if replication else "prompt_variant",
        thinking_mode=parent.thinking_mode,
        reasoning_effort=parent.reasoning_effort,
        generation_budget=parent.generation_budget,
        context_request=parent.context_request,
        temperature=parent.temperature,
        seed=parent.seed,
        prompt_variant=variant,
        recovery_level=recovery_level,
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
            reason="recovery experiment added",
        )


def _execute(
    runner: Any,
    case: dict[str, Any],
    spec: ExperimentSpec,
    parent: ExperimentSpec,
) -> dict[str, Any]:
    label = f"{spec.task_family} L{spec.difficulty_level} recovery {spec.recovery_level}"
    _add_progress_task(runner, label)
    if hasattr(runner, "_progress_begin"):
        runner._progress_begin(label)
    try:
        row = execute_experiment(runner, case, spec, parent=parent)
    finally:
        if hasattr(runner, "_progress_complete"):
            runner._progress_complete(label)
    valid, passed = _valid_result(row)
    runner.store.append_jsonl(
        "recovery-observations.jsonl",
        {
            "family_id": spec.task_family,
            "difficulty_level": spec.difficulty_level,
            "recovery_level": spec.recovery_level,
            "passed": None if not valid else passed,
            "valid_for_capability": valid,
            "result_class": str((row.get("classification") or {}).get("result_class")),
            "experiment_id": spec.experiment_id,
            "parent_experiment_id": spec.parent_experiment_id,
            "prompt_variant": spec.prompt_variant,
            "reasoning_effort": spec.reasoning_effort,
        },
    )
    return row


def _can_still_reach_threshold(
    pass_count: int,
    valid_count: int,
    *,
    repeats: int,
    threshold: float,
) -> bool:
    remaining = max(0, repeats - valid_count)
    required = math.ceil(threshold * repeats - 1e-12)
    return pass_count + remaining >= required


def _prompt_case(case: dict[str, Any], suffix: str) -> dict[str, Any]:
    modified = copy.deepcopy(case)
    modified["prompt"] = f"{case['prompt']}\n\nRECOVERY INSTRUCTION:\n{suffix}"
    return modified


def run_recovery_lab(
    runner: Any,
    cases: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    reasoning_rows: list[dict[str, Any]],
    frontiers: dict[str, Any],
    failure_atlas: dict[str, Any],
    *,
    sequence_start: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Find the minimum reliable recovery while avoiding duplicate model calls."""
    assert runner.store is not None
    cfg = runner.config.get("recovery_lab", {})
    if cfg.get("enabled") is False:
        return [], {
            "schema_version": 1,
            "model": runner.model,
            "enabled": False,
            "families": {},
        }, sequence_start

    repeats = int(cfg.get("repeats", 3))
    if repeats <= 0:
        raise ValueError("recovery_lab.repeats must be positive")
    max_level = str(cfg.get("max_level", "R7")).upper()
    if max_level not in RECOVERY_LEVELS:
        raise ValueError(f"recovery_lab.max_level must be one of {RECOVERY_LEVELS}")
    max_index = RECOVERY_LEVELS.index(max_level)
    reliable_threshold = float(
        runner.config.get("capability_campaign", {}).get("reliable_threshold", 0.90)
    )
    boundary_repeats = int(
        runner.config.get("capability_campaign", {}).get("boundary_repeats", repeats)
    )

    fixtures = _fixture_index(cases)
    targets = select_recovery_targets(frontiers, boundary_repeats=boundary_repeats)
    generated_rows: list[dict[str, Any]] = []
    families: dict[str, Any] = {}
    sequence = sequence_start

    for family_id, level in targets.items():
        case = fixtures.get((family_id, level))
        baseline = _rows_for(base_rows, family_id, level, effort="medium")
        baseline_failures = [
            row for row in baseline
            if _valid_result(row)[0] and not _valid_result(row)[1]
        ]
        parent = _baseline_parent(base_rows, family_id, level)
        if case is None or parent is None:
            families[family_id] = {
                "difficulty_level": level,
                "status": "SKIPPED",
                "reason": "MISSING_FIXTURE_OR_BASELINE_PARENT",
                "steps": [],
                "minimum_successful_recovery": None,
            }
            continue

        signature = _failure_signature(failure_atlas, family_id, level)
        steps: list[dict[str, Any]] = [
            {
                "level": "R0",
                "status": "REPRODUCED_FAILURE",
                "evidence_source": "BASELINE_REPLICATIONS",
                "valid_count": len(baseline_failures),
                "experiment_ids": _experiment_ids(baseline_failures),
            }
        ]
        if max_index >= 1:
            steps.append(
                {
                    "level": "R1",
                    "status": "NOT_APPLICABLE",
                    "evidence_source": "CLASSIFICATION_POLICY",
                    "reason": "SEMANTIC_FAILURE_NOT_GENERATION_HEADROOM_FAILURE",
                }
            )

        minimum_success: str | None = None

        if max_index >= 2:
            existing_high = [
                row
                for row in _rows_for(reasoning_rows, family_id, level, effort="high")
                if _valid_result(row)[0]
            ]
            r2_rows = list(existing_high)
            r2_source = "REASONING_CURVE_REUSE" if existing_high else "RECOVERY_EXECUTION"

            if len(r2_rows) >= repeats:
                step = _summarize_step(
                    "R2",
                    r2_rows,
                    evidence_source="REASONING_CURVE_REUSE",
                    repeats=repeats,
                    reliable_threshold=reliable_threshold,
                )
                steps.append(step)
                if step["status"] == "SUCCESS":
                    minimum_success = "R2"
            else:
                # Existing frontier-local HIGH evidence is reused first.  If it is
                # incomplete, fill the missing replication count.  With no reusable
                # evidence, a failed diagnostic candidate is not needlessly repeated.
                latest: ExperimentSpec = parent
                target_calls = max(1, repeats - len(r2_rows))
                for index in range(target_calls):
                    sequence += 1
                    spec = _clone_r2(
                        latest if index > 0 else parent,
                        sequence=sequence,
                        fixture_id=str(case["id"]),
                        replication=index > 0,
                    )
                    row = _execute(runner, case, spec, latest if index > 0 else parent)
                    generated_rows.append(row)
                    r2_rows.append(row)
                    latest = spec
                    valid, passed = _valid_result(row)
                    if not existing_high and index == 0 and valid and not passed:
                        break
                    valid_rows = [item for item in r2_rows if _valid_result(item)[0]]
                    passes = sum(1 for item in valid_rows if _valid_result(item)[1])
                    if valid_rows and not _can_still_reach_threshold(
                        passes,
                        len(valid_rows),
                        repeats=repeats,
                        threshold=reliable_threshold,
                    ):
                        break
                step = _summarize_step(
                    "R2",
                    r2_rows,
                    evidence_source=r2_source,
                    repeats=repeats,
                    reliable_threshold=reliable_threshold,
                )
                steps.append(step)
                if step["status"] == "SUCCESS":
                    minimum_success = "R2"

        if minimum_success is None:
            for level_index in range(3, max_index + 1):
                recovery_level = RECOVERY_LEVELS[level_index]
                intervention = recovery_intervention(recovery_level, family_id, signature)
                if intervention["kind"] == "unavailable":
                    steps.append(
                        {
                            "level": recovery_level,
                            "status": "UNAVAILABLE",
                            "evidence_source": "CONFIGURATION",
                            "reason": intervention["reason"],
                        }
                    )
                    break

                modified_case = _prompt_case(case, intervention["suffix"])
                candidate_rows: list[dict[str, Any]] = []
                sequence += 1
                candidate = _clone_prompt_recovery(
                    parent,
                    sequence=sequence,
                    fixture_id=str(case["id"]),
                    recovery_level=recovery_level,
                    replication=False,
                )
                first_row = _execute(runner, modified_case, candidate, parent)
                generated_rows.append(first_row)
                candidate_rows.append(first_row)
                valid, passed = _valid_result(first_row)

                if valid and passed:
                    latest = candidate
                    while len([row for row in candidate_rows if _valid_result(row)[0]]) < repeats:
                        valid_rows = [row for row in candidate_rows if _valid_result(row)[0]]
                        passes = sum(1 for row in valid_rows if _valid_result(row)[1])
                        if not _can_still_reach_threshold(
                            passes,
                            len(valid_rows),
                            repeats=repeats,
                            threshold=reliable_threshold,
                        ):
                            break
                        sequence += 1
                        replica = _clone_prompt_recovery(
                            latest,
                            sequence=sequence,
                            fixture_id=str(case["id"]),
                            recovery_level=recovery_level,
                            replication=True,
                        )
                        row = _execute(runner, modified_case, replica, latest)
                        generated_rows.append(row)
                        candidate_rows.append(row)
                        latest = replica

                step = _summarize_step(
                    recovery_level,
                    candidate_rows,
                    evidence_source="RECOVERY_EXECUTION",
                    repeats=repeats,
                    reliable_threshold=reliable_threshold,
                )
                step["intervention"] = copy.deepcopy(intervention)
                step["failure_signature"] = copy.deepcopy(signature)
                steps.append(step)
                if step["status"] == "SUCCESS":
                    minimum_success = recovery_level
                    break

        families[family_id] = {
            "difficulty_level": level,
            "failure_signature": copy.deepcopy(signature),
            "baseline_parent_experiment_id": parent.experiment_id,
            "steps": steps,
            "minimum_successful_recovery": minimum_success,
            "recovered": minimum_success is not None,
        }

    recovery_map = {
        "schema_version": 1,
        "model": runner.model,
        "measurement_policy": {
            "candidate_isolation": "EACH_INITIAL_R2_R7_BRANCHES_FROM_REPRODUCED_BASELINE_FAILURE",
            "successful_candidate_replication": repeats,
            "existing_r0_evidence_reused": True,
            "existing_reasoning_curve_r2_evidence_reused": True,
            "oracle_material_in_interventions": False,
        },
        "reliable_threshold": reliable_threshold,
        "required_repeats": repeats,
        "families": families,
    }
    return generated_rows, recovery_map, sequence
