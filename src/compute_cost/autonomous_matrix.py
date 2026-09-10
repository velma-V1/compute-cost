"""Fully scored autonomous comparison matrix across native reasoning controls."""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from statistics import median
from typing import Any

from .attempt_dossier import persist_attempt_dossier
from .autonomous_simulation import build_scenarios
from .classification import classify_result
from .comparison_matrix import ReasoningCondition, native_reasoning_conditions
from .experiments import ExperimentSpec, make_experiment_id
from .score_vector import build_score_vector

TRUNCATION = {"THINK_TRUNCATED", "ANSWER_TRUNCATED"}

# Declared before any new benchmark run. These encode operational equivalence,
# while exact action-token compliance remains a separate score dimension.
SEMANTIC_ACTION_EQUIVALENCE: dict[tuple[str, int], frozenset[str]] = {
    ("repo_repair", 6): frozenset({"RETEST", "TEST"}),
    ("diagnostic_root_cause", 4): frozenset({"ELIMINATE", "TEST"}),
}


def semantic_action_equivalent(
    scenario_id: str, step: int, expected: str, actual: str
) -> bool:
    expected_u = str(expected).strip().upper()
    actual_u = str(actual).strip().upper()
    if actual_u == expected_u:
        return True
    allowed = SEMANTIC_ACTION_EQUIVALENCE.get((str(scenario_id), int(step)))
    return allowed is not None and expected_u in allowed and actual_u in allowed


def _check(name: str, passed: bool, **extra: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), **extra}


def _score_turn(
    *,
    text: str,
    scenario: dict[str, Any],
    step_index: int,
    step: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {
            "score": 0.0,
            "status": "SCORED",
            "checks": [_check("valid_json", False)],
            "evidence": {"raw_response": text, "parsed": None},
        }

    valid_object = isinstance(payload, dict)
    checks.append(_check("valid_json", valid_object))
    if not valid_object:
        return {
            "score": 0.0,
            "status": "SCORED",
            "checks": checks,
            "evidence": {"raw_response": text, "parsed": payload},
        }

    for field in ("action", "checkpoint", "rationale", "state"):
        checks.append(_check(f"required:{field}", field in payload))

    actual_action = str(payload.get("action") or "").strip().upper()
    actual_checkpoint = str(payload.get("checkpoint") or "").strip()
    expected_action = str(step["expected_action"]).strip().upper()
    expected_checkpoint = str(step["expected_checkpoint"]).strip()
    allowed_actions = {str(value).strip().upper() for value in scenario["allowed_actions"]}

    semantic_action = semantic_action_equivalent(
        str(scenario["id"]), step_index, expected_action, actual_action
    )
    exact_action = actual_action == expected_action
    checkpoint_ok = actual_checkpoint == expected_checkpoint
    allowed_action = actual_action in allowed_actions
    premature_complete = actual_action == "COMPLETE" and expected_action != "COMPLETE"

    checks.extend(
        [
            _check(
                "decision:semantic_action",
                semantic_action,
                actual=actual_action,
                expected=expected_action,
            ),
            _check(
                "contract:exact_action",
                exact_action,
                actual=actual_action,
                expected=expected_action,
            ),
            _check(
                "state:checkpoint",
                checkpoint_ok,
                actual=actual_checkpoint,
                expected=expected_checkpoint,
            ),
            _check("constraint:allowed_action", allowed_action, actual=actual_action),
            _check("goal:preserved", not premature_complete),
        ]
    )
    if bool(step.get("injected_change")):
        checks.append(_check("recovery:injected_change", semantic_action and checkpoint_ok))
    if expected_action == "COMPLETE":
        checks.append(_check("verification:completion", semantic_action and checkpoint_ok))

    required_ok = all(
        row["pass"] for row in checks if str(row["name"]).startswith("required:")
    )
    # Semantic correctness is intentionally independent of exact action spelling.
    semantic_pass = (
        required_ok
        and semantic_action
        and checkpoint_ok
        and allowed_action
        and not premature_complete
    )
    return {
        "score": 1.0 if semantic_pass else 0.0,
        "status": "SCORED",
        "checks": checks,
        "evidence": {"raw_response": text, "parsed": payload},
    }


def _runtime_error_scoring(generation: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": 0.0,
        "status": "RUNTIME_ERROR",
        "checks": [],
        "evidence": {"raw_response": ""},
        "error": copy.deepcopy(generation.get("error") or {"type": "RUNTIME_ERROR"}),
    }


def _reasoning_fields(model: str, condition: ReasoningCondition) -> tuple[bool, str | None]:
    if model.lower().startswith("gpt-oss"):
        return True, str(condition.request_value)
    if condition.native_name == "think":
        return bool(condition.request_value), None
    return False, None


def _request_fields(condition: ReasoningCondition) -> dict[str, Any]:
    if condition.native_name in {"reasoning_effort", "think"}:
        return {"think": condition.request_value}
    return {}


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


def _median(values: list[float]) -> float | None:
    return None if not values else float(median(values))


def _render_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"# Fully Scored Autonomous Matrix: {summary.get('model')}",
        "",
        f"Semantic turns: {summary.get('semantic_turns', 0)} | attempts: {summary.get('attempts', 0)}",
        "",
        "| Reasoning | Scenario | Semantic | Decision | Exact action | State | Complete | Median s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role, block in sorted((summary.get("reasoning_conditions") or {}).items()):
        for scenario_id, row in sorted((block.get("scenarios") or {}).items()):
            pct = lambda value: "N/A" if value is None else f"{100.0 * float(value):.1f}%"
            med = row.get("median_latency_s")
            lines.append(
                f"| {role} | {scenario_id} | {pct(row.get('semantic_accuracy'))} | "
                f"{pct(row.get('decision_accuracy'))} | {pct(row.get('exact_action_accuracy'))} | "
                f"{pct(row.get('state_accuracy'))} | {'YES' if row.get('completed_correctly') else 'NO'} | "
                f"{'N/A' if med is None else f'{float(med):.3f}'} |"
            )
    return "\n".join(lines).rstrip() + "\n"


def run_autonomous_matrix(
    runner: Any,
    *,
    sequence_start: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Run each scenario independently for every native reasoning condition."""
    cfg = runner.config.get("autonomous_simulation") or {}
    if cfg.get("enabled") is not True:
        return [], {"schema_version": 2, "enabled": False, "reasoning_conditions": {}}, sequence_start

    scenarios = build_scenarios()
    scenarios = scenarios[: max(1, min(int(cfg.get("scenario_count", 6)), len(scenarios)))]
    max_steps = max(1, int(cfg.get("steps_per_scenario", 8)))
    base_budget = int(cfg.get("generation_budget", 512))
    ceiling = max(base_budget, int(cfg.get("max_generation_budget", 2048)))
    conditions = native_reasoning_conditions(runner.model)
    sequence = sequence_start
    rows: list[dict[str, Any]] = []
    summary_by_role: dict[str, Any] = {}

    system = (
        "You are operating autonomously inside a deterministic simulation. Preserve the mission goal and "
        "current authoritative checkpoint across the growing transcript. At each turn choose the best next "
        "action from the allowed actions. Return ONLY one JSON object with keys action, checkpoint, rationale, "
        "state. Never declare COMPLETE before required verification is satisfied."
    )

    for condition in conditions:
        role_key = condition.role.lower()
        thinking_mode, reasoning_effort = _reasoning_fields(runner.model, condition)
        role_summaries: dict[str, Any] = {}

        for scenario in scenarios:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"SCENARIO {scenario['id']}: {scenario['brief']} Initial authoritative "
                        f"checkpoint={scenario['checkpoint']}. Allowed actions: "
                        f"{', '.join(scenario['allowed_actions'])}."
                    ),
                },
            ]
            final_rows: list[dict[str, Any]] = []
            latencies: list[float] = []
            attempt_count = 0
            truncation_retries = 0

            for step_index, step in enumerate(scenario["steps"][:max_steps], start=1):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"TURN {step_index}/{min(max_steps, len(scenario['steps']))}. "
                            f"ENVIRONMENT: {step['event']} Choose the best next autonomous action now."
                        ),
                    }
                )
                budget = base_budget
                parent: ExperimentSpec | None = None
                attempt_index = 0
                final_response = ""
                final_row: dict[str, Any] | None = None

                while True:
                    attempt_index += 1
                    attempt_count += 1
                    sequence += 1
                    case_id = (
                        f"auto-{scenario['id']}-s{step_index:02d}--{role_key}-a{attempt_index:02d}"
                    )
                    case = {
                        "id": case_id,
                        "family_id": "autonomous_simulation",
                        "category": "autonomous_simulation",
                        "difficulty_level": step_index,
                        "prompt": messages[-1]["content"],
                        "scorer": "autonomous_semantic_v2",
                        "expected": {
                            "action": step["expected_action"],
                            "checkpoint": step["expected_checkpoint"],
                        },
                        "required": ["action", "checkpoint", "rationale", "state"],
                        "allowed_actions": list(scenario["allowed_actions"]),
                        "semantic_action_equivalence": sorted(
                            SEMANTIC_ACTION_EQUIVALENCE.get((scenario["id"], step_index), frozenset())
                        ),
                        "tags": ["autonomous_simulation", scenario["id"], role_key, f"step_{step_index}"],
                    }
                    spec = ExperimentSpec(
                        experiment_id=make_experiment_id(
                            sequence, scenario["id"], f"{role_key}-s{step_index}-a{attempt_index}"
                        ),
                        parent_experiment_id=None if parent is None else parent.experiment_id,
                        task_id=scenario["id"],
                        task_family="autonomous_simulation",
                        difficulty_level=step_index,
                        hypothesis=(
                            "fixed autonomous reasoning-condition measurement"
                            if parent is None
                            else "retry exact autonomous turn after token-ceiling exhaustion"
                        ),
                        changed_variable="baseline" if parent is None else "generation_budget",
                        thinking_mode=thinking_mode,
                        reasoning_effort=reasoning_effort,
                        generation_budget=budget,
                        context_request=None,
                        temperature=0.0,
                        seed=42,
                        prompt_variant=f"{scenario['id']}-{role_key}-step-{step_index}",
                        recovery_level="R1" if parent is not None else None,
                    )
                    options = runner._generation_options(
                        case,
                        {"num_predict": budget, "temperature": 0.0, "seed": 42},
                    )
                    runner._call_category_context = (
                        "truncation_retry" if parent is not None else "fixed_autonomous"
                    )
                    runner._call_scenario_context = str(scenario["id"])
                    try:
                        generation, invocation, refs = runner._invoke_generation(
                            stage="autonomous-simulation",
                            case_id=case_id,
                            messages=messages,
                            options=options,
                            request_fields=_request_fields(condition),
                        )
                    finally:
                        runner._call_category_context = None
                        runner._call_scenario_context = None

                    if parent is None:
                        _decrement_mandatory(runner)

                    if generation.get("ok", False):
                        final_response = str((generation.get("normalized") or {}).get("text", ""))
                        scoring = _score_turn(
                            text=final_response,
                            scenario=scenario,
                            step_index=step_index,
                            step=step,
                        )
                    else:
                        final_response = ""
                        scoring = _runtime_error_scoring(generation)
                    runner._persist_scoring(case_id, "autonomous-simulation", scoring)
                    classification = classify_result(case, generation, scoring)
                    score_vector = build_score_vector(
                        case,
                        scoring,
                        classification,
                        generation,
                        telemetry=list(copy.deepcopy(getattr(runner, "_recent_telemetry", []))),
                    )

                    # For autonomous semantic scoring, exact token equality is a
                    # separate contract signal rather than semantic correctness.
                    exact_checks = [
                        c for c in scoring.get("checks", [])
                        if c.get("name") == "contract:exact_action"
                    ]
                    score_vector["exact_action_contract_compliance"] = (
                        100.0 if exact_checks and exact_checks[0].get("pass") is True else 0.0
                    )
                    semantic_checks = [
                        c for c in scoring.get("checks", [])
                        if c.get("name") in {"decision:semantic_action", "state:checkpoint"}
                    ]
                    if semantic_checks:
                        score_vector["semantic_correctness"] = (
                            100.0 if all(c.get("pass") is True for c in semantic_checks) else 0.0
                        )
                    decision_checks = [
                        c for c in scoring.get("checks", [])
                        if c.get("name") == "decision:semantic_action"
                    ]
                    if decision_checks:
                        score_vector["decision_quality"] = (
                            100.0 if decision_checks[0].get("pass") is True else 0.0
                        )

                    row = {
                        "experiment": spec.to_dict(),
                        "classification": classification,
                        "score": scoring.get("score"),
                        "status": scoring.get("status"),
                        "score_vector": score_vector,
                        "metrics": copy.deepcopy(generation.get("metrics") or {}),
                        "timing": copy.deepcopy(generation.get("timing") or {}),
                        "phase_metrics": copy.deepcopy(generation.get("phase_metrics") or {}),
                        "evidence_key": case_id,
                        "evidence_refs": copy.deepcopy(refs),
                        "comparison": {
                            "scope": "fixed_core",
                            "reasoning_role": condition.role,
                            "native_reasoning_control": condition.native_name,
                            "native_reasoning_value": condition.request_value,
                            "scenario_id": scenario["id"],
                            "turn": step_index,
                            "attempt": attempt_index,
                            "is_retry": parent is not None,
                        },
                        "simulation": {
                            "scenario_id": scenario["id"],
                            "step": step_index,
                            "attempt": attempt_index,
                            "expected_action": step["expected_action"],
                            "expected_checkpoint": step["expected_checkpoint"],
                            "injected_change": bool(step.get("injected_change")),
                        },
                    }
                    runner.store.append_jsonl("experiments.jsonl", row)
                    runner.store.append_jsonl("autonomous-observations.jsonl", row)
                    rows.append(row)

                    dossier = persist_attempt_dossier(
                        runner,
                        case=case,
                        spec=spec,
                        invocation=invocation,
                        generation=generation,
                        scoring=scoring,
                        classification=classification,
                        evidence_refs=refs,
                        telemetry_before=[],
                        telemetry_after=list(copy.deepcopy(getattr(runner, "_recent_telemetry", []))),
                    )
                    if dossier is not None and isinstance(dossier.get("score_vector"), dict):
                        # Persist the autonomous semantic overrides too.
                        dossier["score_vector"].update(score_vector)

                    final_row = row
                    result_class = str(classification.get("result_class") or "")
                    if result_class != "ANSWER_CORRECT" and hasattr(runner, "_write_replay"):
                        runner._write_replay(
                            case=case,
                            invocation=invocation,
                            generation=generation,
                            scoring=scoring,
                            stage="autonomous-simulation",
                            replay_name=case_id,
                        )

                    if (
                        result_class not in TRUNCATION
                        or budget >= ceiling
                        or not _can_spend_retry(runner)
                    ):
                        break
                    next_budget = min(ceiling, budget * 2)
                    if next_budget <= budget:
                        break
                    parent = spec
                    budget = next_budget
                    truncation_retries += 1

                if final_row is not None:
                    final_rows.append(final_row)
                    latency_ns = (final_row.get("timing") or {}).get("client_latency_ns")
                    if isinstance(latency_ns, (int, float)) and not isinstance(latency_ns, bool):
                        latencies.append(float(latency_ns) / 1e9)
                messages.append(
                    {
                        "role": "assistant",
                        "content": final_response or '{"action":"NO_RESPONSE","checkpoint":"UNKNOWN"}',
                    }
                )

            transcript_path = f"autonomous-transcripts/{role_key}/{scenario['id']}.json"
            runner.store.write_json(
                transcript_path,
                {
                    "scenario": scenario,
                    "reasoning_role": condition.role,
                    "native_reasoning_control": condition.native_name,
                    "native_reasoning_value": condition.request_value,
                    "messages": messages,
                },
                producer="autonomous-matrix",
                stage="report",
            )

            semantic_ok = [r["score_vector"]["semantic_correctness"] == 100.0 for r in final_rows]
            decisions = [r["score_vector"]["decision_quality"] == 100.0 for r in final_rows]
            exacts = [r["score_vector"]["exact_action_contract_compliance"] == 100.0 for r in final_rows]
            states = [r["score_vector"]["state_checkpoint_accuracy"] == 100.0 for r in final_rows]
            complete = bool(
                final_rows
                and final_rows[-1]["score_vector"]["semantic_correctness"] == 100.0
                and str(final_rows[-1]["simulation"]["expected_action"]).upper() == "COMPLETE"
            )
            role_summaries[scenario["id"]] = {
                "turns": len(final_rows),
                "attempts": attempt_count,
                "semantic_accuracy": None if not semantic_ok else sum(semantic_ok) / len(semantic_ok),
                "decision_accuracy": None if not decisions else sum(decisions) / len(decisions),
                "exact_action_accuracy": None if not exacts else sum(exacts) / len(exacts),
                "state_accuracy": None if not states else sum(states) / len(states),
                "completed_correctly": complete,
                "truncation_retries": truncation_retries,
                "median_latency_s": _median(latencies),
                "result_classes": dict(Counter(str((r.get("classification") or {}).get("result_class")) for r in final_rows)),
            }

        summary_by_role[condition.role] = {
            "native_reasoning_control": condition.native_name,
            "native_reasoning_value": condition.request_value,
            "scenarios": role_summaries,
        }

    summary = {
        "schema_version": 2,
        "model": str(runner.model),
        "semantic_turns": sum(
            int(row.get("turns", 0))
            for block in summary_by_role.values()
            for row in (block.get("scenarios") or {}).values()
        ),
        "attempts": len(rows),
        "reasoning_conditions": summary_by_role,
        "semantic_equivalence_policy": {
            f"{scenario}:{step}": sorted(values)
            for (scenario, step), values in sorted(SEMANTIC_ACTION_EQUIVALENCE.items())
        },
    }
    runner.store.write_json(
        "autonomous-simulation.json",
        summary,
        producer="autonomous-matrix",
        stage="report",
    )
    runner.store.write_text(
        "autonomous-simulation.md",
        _render_summary(summary),
        producer="autonomous-matrix",
        stage="report",
    ) if hasattr(runner.store, "write_text") else None
    return rows, summary, sequence
