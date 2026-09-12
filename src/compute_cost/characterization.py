"""Adaptive, evidence-first model characterization built on the stable runner core."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any

from .adaptive import AdaptiveBudgetController, BudgetObservation
from .classification import classify_result
from .experiments import ExperimentSpec, changed_fields, make_experiment_id
from .replay_registry import replay_category
from .scoring import score_case


HARNESS_INVALID = {"SCORER_DEFECT", "TEST_DEFECT", "CAPTURE_GAP"}
RUNTIME_INVALID = {"RUNTIME_FAILURE", "TIMEOUT", "RESOURCE_LIMIT"}
TRUNCATION = {"THINK_TRUNCATED", "ANSWER_TRUNCATED"}


def _validate_lineage(parent: ExperimentSpec | None, child: ExperimentSpec) -> None:
    if parent is None or child.changed_variable == "baseline":
        return
    changes = changed_fields(parent, child)
    if child.changed_variable == "replication":
        if changes:
            raise ValueError(f"replication must preserve controlled fields: {changes}")
        return
    if changes != [child.changed_variable]:
        raise ValueError(f"experiment must change exactly {child.changed_variable}: {changes}")


def _runtime_error_scoring(generation: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": 0.0,
        "status": "RUNTIME_ERROR",
        "checks": [],
        "evidence": {"raw_response": ""},
        "error": copy.deepcopy(generation.get("error") or {"type": "RUNTIME_ERROR"}),
    }


def _write_replay_v2(
    runner: Any,
    *,
    case: dict[str, Any],
    spec: ExperimentSpec,
    invocation: dict[str, Any],
    generation: dict[str, Any],
    scoring: dict[str, Any],
    classification: dict[str, Any],
) -> None:
    """Persist a canonical categorized snapshot plus the historical root alias."""
    assert runner.store is not None
    result_class = str(classification.get("result_class"))
    category = replay_category(result_class, recovery_level=spec.recovery_level)
    canonical_path = f"replay/{category}/{spec.experiment_id}.json"
    compatibility_path = f"replay/{spec.experiment_id}.json"
    snapshot = {
        "schema_version": 2,
        "created_at_utc": runner._utc(),
        "benchmark_version": runner.suite.get("benchmark_version"),
        "stage": "characterize",
        "model": runner.model,
        "case": copy.deepcopy(case),
        "experiment": spec.to_dict(),
        "classification": copy.deepcopy(classification),
        "evidence_key": spec.experiment_id,
        "invocation": copy.deepcopy(invocation),
        "generation": copy.deepcopy(generation),
        "scoring": copy.deepcopy(scoring),
        "telemetry_before_failure": list(copy.deepcopy(runner._recent_telemetry)),
        "resolved_config": copy.deepcopy(runner.config),
    }
    canonical = runner.store.write_json(
        canonical_path,
        snapshot,
        producer="characterization",
        stage="characterize",
        case_id=case.get("id"),
    )
    runner.store.write_json(
        compatibility_path,
        snapshot,
        producer="characterization-compatibility-alias",
        stage="characterize",
        case_id=case.get("id"),
    )
    runner.store.append_jsonl(
        "replay/index.jsonl",
        {
            "replay_id": spec.experiment_id,
            "category": category,
            "path": canonical_path,
            "compatibility_path": compatibility_path,
            "canonical_sha256": canonical.get("sha256"),
            "family_id": spec.task_family,
            "task_id": spec.task_id,
            "difficulty_level": spec.difficulty_level,
            "result_class": result_class,
            "valid_for_capability": classification.get("valid_for_capability") is True,
            "recovery_level": spec.recovery_level,
            "parent_experiment_id": spec.parent_experiment_id,
        },
    )


def execute_experiment(
    runner: Any,
    case: dict[str, Any],
    spec: ExperimentSpec,
    *,
    parent: ExperimentSpec | None = None,
    messages_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute one fully specified model call while preserving unique raw evidence.

    messages_override is an explicit experimental surface used by controlled
    prompt-treatment campaigns. Existing callers omit it and retain historical
    behavior exactly.
    """
    assert runner.store is not None
    _validate_lineage(parent, spec)
    messages = (
        copy.deepcopy(messages_override)
        if messages_override is not None
        else [{"role": "user", "content": str(case["prompt"])}]
    )
    options = runner._generation_options(
        case,
        {
            "num_predict": spec.generation_budget,
            "temperature": spec.temperature,
            "seed": spec.seed,
        },
    )
    if spec.context_request is not None:
        options["num_ctx"] = int(spec.context_request)
    think_request = spec.reasoning_effort if spec.reasoning_effort is not None else spec.thinking_mode
    generation, invocation, refs = runner._invoke_generation(
        stage="characterize",
        case_id=spec.experiment_id,
        messages=messages,
        options=options,
        request_fields={"think": think_request},
    )
    if generation.get("ok", False):
        response = str((generation.get("normalized") or {}).get("text", ""))
        scoring = score_case(case, response)
    else:
        scoring = _runtime_error_scoring(generation)

    runner._persist_scoring(spec.experiment_id, "characterize", scoring)
    classification = classify_result(case, generation, scoring)
    row = {
        "experiment": spec.to_dict(),
        "classification": classification,
        "score": scoring.get("score"),
        "status": scoring.get("status"),
        "response_text": response if generation.get("ok", False) else "",
        "metrics": copy.deepcopy(generation.get("metrics") or {}),
        "timing": copy.deepcopy(generation.get("timing") or {}),
        "phase_metrics": copy.deepcopy(generation.get("phase_metrics") or {}),
        "evidence_key": spec.experiment_id,
        "evidence_refs": refs,
    }
    runner.store.append_jsonl("experiments.jsonl", row)

    if classification.get("result_class") != "ANSWER_CORRECT":
        _write_replay_v2(
            runner,
            case=case,
            spec=spec,
            invocation=invocation,
            generation=generation,
            scoring=scoring,
            classification=classification,
        )
    return row


def _spec(
    *,
    sequence: int,
    case: dict[str, Any],
    parent: ExperimentSpec | None,
    thinking: bool,
    budget: int,
    hypothesis: str,
    changed_variable: str,
    recovery_level: str | None = None,
) -> ExperimentSpec:
    label = f"{'think-on' if thinking else 'think-off'}-{budget}"
    return ExperimentSpec(
        experiment_id=make_experiment_id(sequence, str(case["id"]), label),
        parent_experiment_id=None if parent is None else parent.experiment_id,
        task_id=str(case["id"]),
        task_family=str(case.get("category", "unknown")),
        difficulty_level=int(case.get("difficulty_level", 0)),
        hypothesis=hypothesis,
        changed_variable=changed_variable,
        thinking_mode=thinking,
        generation_budget=budget,
        context_request=None,
        temperature=0.0,
        seed=42,
        prompt_variant="base",
        recovery_level=recovery_level,
    )


def _run_with_progress(
    runner: Any,
    case: dict[str, Any],
    spec: ExperimentSpec,
    parent: ExperimentSpec | None,
) -> dict[str, Any]:
    label = f"{case['id']} {'think on' if spec.thinking_mode else 'think off'} {spec.generation_budget}"
    if hasattr(runner, "_progress_begin"):
        runner._progress_begin(label)
    try:
        return execute_experiment(runner, case, spec, parent=parent)
    finally:
        if hasattr(runner, "_progress_complete"):
            runner._progress_complete(label)


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


def run_characterization(runner: Any, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run OFF/ON baselines then spend calls only around informative generation boundaries."""
    assert runner.store is not None
    cfg = runner.config["characterization"]
    all_rows: list[dict[str, Any]] = []
    sequence = 0

    for case in cases:
        task_rows: list[dict[str, Any]] = []
        sequence += 1
        off_spec = _spec(
            sequence=sequence,
            case=case,
            parent=None,
            thinking=False,
            budget=int(cfg["think_off_generation_budget"]),
            hypothesis="establish thinking-off capability baseline",
            changed_variable="baseline",
        )
        off_row = _run_with_progress(runner, case, off_spec, None)
        task_rows.append(off_row)
        all_rows.append(off_row)

        controller = AdaptiveBudgetController(
            initial_budget=int(cfg["initial_generation_budget"]),
            min_budget=int(cfg["min_generation_budget"]),
            max_budget=int(cfg["max_generation_budget"]),
            granularity=int(cfg["generation_budget_granularity"]),
            boundary_repeats=int(cfg["boundary_repeats"]),
        )
        first = controller.next([])
        assert first.budget is not None
        sequence += 1
        on_spec = _spec(
            sequence=sequence,
            case=case,
            parent=off_spec,
            thinking=True,
            budget=int(first.budget),
            hypothesis="establish thinking-on capability baseline",
            changed_variable="thinking_mode",
        )
        on_row = _run_with_progress(runner, case, on_spec, off_spec)
        task_rows.append(on_row)
        all_rows.append(on_row)
        observations = [
            BudgetObservation(
                budget=on_spec.generation_budget,
                result_class=str(on_row["classification"]["result_class"]),
            )
        ]
        previous_spec = on_spec
        previous_row = on_row
        latest_spec_by_budget: dict[int, ExperimentSpec] = {
            on_spec.generation_budget: on_spec,
        }
        stop_reason = "CONTROLLER_STOP"

        while len(task_rows) < int(cfg["max_experiments_per_task"]):
            decision = controller.next(observations)
            if decision.action == "STOP":
                stop_reason = decision.reason
                break
            assert decision.budget is not None
            next_budget = int(decision.budget)
            _add_adaptive_progress_task(
                runner,
                f"{case['id']} {decision.action.lower()} {next_budget}",
            )
            if decision.action == "REPLICATE":
                parent_spec = latest_spec_by_budget[next_budget]
                changed_variable = "replication"
                hypothesis = "reproduce minimum passing boundary"
                recovery_level = None
            else:
                parent_spec = previous_spec
                previous_class = str(previous_row["classification"]["result_class"])
                increased_after_truncation = (
                    previous_class in TRUNCATION
                    and next_budget > previous_spec.generation_budget
                )
                recovery_level = "R1" if increased_after_truncation else None
                changed_variable = "generation_budget"
                hypothesis = (
                    "insufficient generation headroom caused truncation"
                    if increased_after_truncation
                    else decision.reason
                )
            sequence += 1
            child = _spec(
                sequence=sequence,
                case=case,
                parent=parent_spec,
                thinking=True,
                budget=next_budget,
                hypothesis=hypothesis,
                changed_variable=changed_variable,
                recovery_level=recovery_level,
            )
            row = _run_with_progress(runner, case, child, parent_spec)
            task_rows.append(row)
            all_rows.append(row)
            observations.append(
                BudgetObservation(
                    budget=child.generation_budget,
                    result_class=str(row["classification"]["result_class"]),
                )
            )
            latest_spec_by_budget[child.generation_budget] = child
            previous_spec = child
            previous_row = row
        else:
            stop_reason = "MAX_EXPERIMENTS_PER_TASK"

        runner.store.append_jsonl(
            "characterization-events.jsonl",
            {
                "event": "task_stop",
                "timestamp_utc": runner._utc(),
                "task_id": case["id"],
                "reason": stop_reason,
                "experiments": len(task_rows),
            },
        )

    return all_rows


def build_task_profile(
    task_id: str,
    rows: list[dict[str, Any]],
    *,
    boundary_repeats: int,
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row.get("experiment", {}).get("task_id") == task_id
    ]
    think_off_row = next(
        (row for row in selected if not row.get("experiment", {}).get("thinking_mode")),
        None,
    )
    on_rows = [
        row for row in selected
        if row.get("experiment", {}).get("thinking_mode")
    ]
    capability_rows = [
        row for row in selected
        if row.get("classification", {}).get("valid_for_capability") is True
    ]
    usable_boundary_rows = [
        row for row in on_rows
        if row.get("classification", {}).get("result_class")
        not in HARNESS_INVALID | RUNTIME_INVALID
    ]

    correct_counts = Counter(
        int(row["experiment"]["generation_budget"])
        for row in usable_boundary_rows
        if row.get("classification", {}).get("result_class") == "ANSWER_CORRECT"
    )
    reproduced = sorted(
        budget for budget, count in correct_counts.items()
        if count >= boundary_repeats
    )
    minimum_budget = reproduced[0] if reproduced else None
    lower_fail = None
    if minimum_budget is not None:
        lower = [
            int(row["experiment"]["generation_budget"])
            for row in usable_boundary_rows
            if int(row["experiment"]["generation_budget"]) < minimum_budget
            and row.get("classification", {}).get("result_class") != "ANSWER_CORRECT"
        ]
        lower_fail = max(lower) if lower else None

    return {
        "task_id": task_id,
        "think_off": None if think_off_row is None else {
            "result_class": think_off_row["classification"]["result_class"],
            "score": think_off_row.get("score"),
            "kind": "MEASURED",
        },
        "minimum_reproduced_pass_budget": None if minimum_budget is None else {
            "value": minimum_budget,
            "kind": "DERIVED",
        },
        "transition_bracket": None if minimum_budget is None or lower_fail is None else {
            "lower_fail": lower_fail,
            "upper_pass": minimum_budget,
            "kind": "DERIVED",
        },
        "capability_observations": capability_rows,
        "behavioral_observations": selected,
    }


def build_characterization_summary(
    model: str,
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    boundary_repeats: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": model,
        "measurement_policy": {
            "exposed_thinking": "observable behavioral trace",
            "aggregate_eval_count": (
                "MEASURED aggregate generation count; not an exact per-phase "
                "thinking-token count"
            ),
        },
        "tasks": [
            build_task_profile(
                str(case["id"]),
                rows,
                boundary_repeats=boundary_repeats,
            )
            for case in cases
        ],
    }


def render_characterization_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Adaptive Characterization Report",
        "",
        f"Model: `{summary.get('model')}`",
        "",
        "Exposed thinking is an observable behavioral trace, not hidden cognition.",
        (
            "Aggregate `eval_count` is MEASURED runtime evidence and is not split "
            "into fabricated per-phase token counts."
        ),
        "",
        "## Task profiles",
        "",
    ]
    for task in summary.get("tasks", []):
        lines.append(f"### {task['task_id']}")
        off = task.get("think_off")
        lines.append(
            f"- Think OFF: {None if off is None else off.get('result_class')} (MEASURED)"
        )
        minimum = task.get("minimum_reproduced_pass_budget")
        lines.append(
            "- Minimum reproduced passing generation budget: "
            f"{None if minimum is None else minimum.get('value')} (DERIVED)"
        )
        bracket = task.get("transition_bracket")
        if bracket is None:
            lines.append("- Transition bracket: unavailable")
        else:
            lines.append(
                f"- Transition bracket: {bracket['lower_fail']} fail/truncate -> "
                f"{bracket['upper_pass']} reproduced pass (DERIVED)"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
