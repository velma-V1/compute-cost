"""Bounded multi-turn autonomous simulation for capability characterization.

The phase is intentionally small: six deterministic eight-turn scenarios.  Each
turn preserves the growing transcript, scores the model's next autonomous action
and active state checkpoint, and retains raw runtime/scorer evidence.  Token
budget increases only when the runtime/scorer classifies the attempt as truncated.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from statistics import median
from typing import Any

from .classification import classify_result
from .experiments import ExperimentSpec, make_experiment_id
from .scoring import score_case

TRUNCATION = {"THINK_TRUNCATED", "ANSWER_TRUNCATED"}


def _steps(actions: list[str], *, old: str, new: str, events: list[str], change_at: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (action, event) in enumerate(zip(actions, events, strict=True), start=1):
        rows.append(
            {
                "event": event,
                "expected_action": action,
                "expected_checkpoint": new if index >= change_at else old,
                "injected_change": index == change_at,
            }
        )
    return rows


def build_scenarios() -> list[dict[str, Any]]:
    """Return six deterministic eight-turn long-horizon simulations."""
    return [
        {
            "id": "repo_repair",
            "title": "Repository repair under changing evidence",
            "checkpoint": "BUILD-A",
            "tags": ["planning", "state_retention", "constraint_update", "self_verification"],
            "allowed_actions": ["INSPECT", "HYPOTHESIZE", "TEST", "PATCH", "DIAGNOSE", "RETEST", "AUDIT", "COMPLETE"],
            "brief": "Repair a failing repository with the smallest evidence-backed change. Preserve tests and do not declare completion before verification.",
            "steps": _steps(
                ["INSPECT", "HYPOTHESIZE", "TEST", "PATCH", "DIAGNOSE", "RETEST", "AUDIT", "COMPLETE"],
                old="BUILD-A", new="BUILD-B", change_at=5,
                events=[
                    "A regression report arrives. No root cause is yet established.",
                    "Inspection isolates one suspicious boundary condition but has not proven causality.",
                    "You have a reproducible failing fixture and must establish whether the hypothesis is real before editing.",
                    "The focused test proves the hypothesis. A minimal source change is now justified.",
                    "AUTHORITATIVE UPDATE: checkpoint is now BUILD-B. The first patch still fails one independent fixture, so do not repeat the same patch blindly.",
                    "New evidence isolates the remaining defect and a corrected minimal patch is available; verify it against the focused tests.",
                    "Focused tests pass. Check the broader acceptance contract for omitted regressions before completion.",
                    "The full acceptance audit is green and no required evidence is missing.",
                ],
            ),
        },
        {
            "id": "tool_chain_recovery",
            "title": "Dependent tool chain with injected schema failure",
            "checkpoint": "TOOL-A",
            "tags": ["tool_recovery", "state_retention", "planning", "self_verification"],
            "allowed_actions": ["SEARCH", "FETCH", "CALCULATE", "HANDLE_ERROR", "RETRY", "CROSSCHECK", "SYNTHESIZE", "COMPLETE"],
            "brief": "Complete a dependent tool workflow without inventing tool results. Preserve identifiers and recover from tool errors with the smallest valid retry.",
            "steps": _steps(
                ["SEARCH", "FETCH", "CALCULATE", "HANDLE_ERROR", "RETRY", "CROSSCHECK", "SYNTHESIZE", "COMPLETE"],
                old="TOOL-A", new="TOOL-B", change_at=4,
                events=[
                    "You need the current record for Mira but have no identifier yet.",
                    "Search returned id=42. Retrieve the record before calculating anything derived from it.",
                    "The record supplies value=17 and multiplier=23. Compute the required product.",
                    "AUTHORITATIVE UPDATE: checkpoint is now TOOL-B. The tool rejects id=42 because the schema requires id:string. Preserve the workflow state.",
                    "The schema confirms id must be the string '42'; perform the minimal corrected retry.",
                    "The retry succeeds. A second source is available to verify the resulting record before synthesis.",
                    "Cross-check agrees. Produce the integrated result without losing provenance.",
                    "All dependencies, recovery evidence, and verification checks are satisfied.",
                ],
            ),
        },
        {
            "id": "diagnostic_root_cause",
            "title": "Long diagnostic elimination sequence",
            "checkpoint": "DIAG-A",
            "tags": ["diagnosis", "state_retention", "constraint_update", "self_verification"],
            "allowed_actions": ["OBSERVE", "HYPOTHESIZE", "TEST", "ELIMINATE", "RETEST", "ROOT_CAUSE", "VERIFY", "COMPLETE"],
            "brief": "Diagnose a fault by evidence, not guessing. Keep competing hypotheses alive until tests eliminate them and verify the final root cause independently.",
            "steps": _steps(
                ["OBSERVE", "HYPOTHESIZE", "TEST", "ELIMINATE", "RETEST", "ROOT_CAUSE", "VERIFY", "COMPLETE"],
                old="DIAG-A", new="DIAG-B", change_at=4,
                events=[
                    "A system intermittently loses pressure under load. Only symptoms are known.",
                    "Initial measurements support several plausible causes; formulate the smallest discriminating hypothesis set.",
                    "A controlled test can distinguish supply restriction from sensor error; gather that evidence before replacing anything.",
                    "AUTHORITATIVE UPDATE: checkpoint is now DIAG-B. The test disproves sensor error and contradicts your earliest favorite hypothesis.",
                    "A second controlled test can separate restriction from actuator leakage.",
                    "The second test isolates supply restriction as the only surviving causal mechanism.",
                    "An independent measurement can verify that diagnosis before declaring the job complete.",
                    "Independent verification confirms the root cause and no competing hypothesis remains supported.",
                ],
            ),
        },
        {
            "id": "adaptive_project_plan",
            "title": "Long project plan with mid-run constraint change",
            "checkpoint": "PLAN-A",
            "tags": ["planning", "state_retention", "constraint_update", "self_verification"],
            "allowed_actions": ["DECOMPOSE", "PRIORITIZE", "SCHEDULE", "REPLAN", "RECOVER", "OPTIMIZE", "VERIFY", "COMPLETE"],
            "brief": "Drive a multi-step project toward the objective while minimizing wasted work. Adapt when an authoritative constraint invalidates part of the plan.",
            "steps": _steps(
                ["DECOMPOSE", "PRIORITIZE", "SCHEDULE", "REPLAN", "RECOVER", "OPTIMIZE", "VERIFY", "COMPLETE"],
                old="PLAN-A", new="PLAN-B", change_at=4,
                events=[
                    "The objective has four dependent workstreams and no execution order yet.",
                    "Dependencies are known; identify the work that unlocks the most downstream progress.",
                    "Priorities are accepted. Build an executable order that respects dependencies.",
                    "AUTHORITATIVE UPDATE: checkpoint is now PLAN-B. A key resource disappears, invalidating two scheduled steps.",
                    "A substitute resource exists but requires changing the middle of the plan while preserving completed work.",
                    "The recovered plan is feasible; reduce unnecessary steps without violating constraints.",
                    "The optimized plan is ready. Verify every original and updated constraint before completion.",
                    "Verification confirms all dependencies, updates, and success criteria are satisfied.",
                ],
            ),
        },
        {
            "id": "evidence_research",
            "title": "Research synthesis with contradictory and obsolete evidence",
            "checkpoint": "SRC-A",
            "tags": ["research_conflict", "state_retention", "constraint_update", "self_verification"],
            "allowed_actions": ["GATHER", "COMPARE", "IDENTIFY_CONFLICT", "UPDATE", "WEIGH_EVIDENCE", "EDGE_CASE", "SYNTHESIZE", "COMPLETE"],
            "brief": "Build a conclusion from supplied evidence while distinguishing direct evidence, stale evidence, contradictions, and unresolved edge cases.",
            "steps": _steps(
                ["GATHER", "COMPARE", "IDENTIFY_CONFLICT", "UPDATE", "WEIGH_EVIDENCE", "EDGE_CASE", "SYNTHESIZE", "COMPLETE"],
                old="SRC-A", new="SRC-B", change_at=4,
                events=[
                    "Three sources arrive with different scopes. Collect their claims before choosing a conclusion.",
                    "Claims are extracted; compare scope, date, and directness rather than counting sources.",
                    "Two sources conflict on the central claim and the disagreement must be made explicit.",
                    "AUTHORITATIVE UPDATE: checkpoint is now SRC-B. A newer primary source supersedes one stale conflicting claim.",
                    "The stale source remains in context but is marked obsolete. Weight the remaining evidence by authority and directness.",
                    "A rare edge case is not covered by the primary evidence; test whether it changes the general conclusion.",
                    "The edge case is bounded. Synthesize the conclusion with limitations and provenance intact.",
                    "The synthesis has been checked against every supplied source and unresolved limitation.",
                ],
            ),
        },
        {
            "id": "state_memory_continuation",
            "title": "Persistent state and continuation under compression",
            "checkpoint": "STATE-A",
            "tags": ["state_retention", "constraint_update", "planning", "self_verification"],
            "allowed_actions": ["CAPTURE", "COMPRESS", "CONTINUE", "REJECT_STALE", "RESTORE", "EXECUTE", "VERIFY", "COMPLETE"],
            "brief": "Maintain a compact but sufficient working state across a long continuation. Never resurrect obsolete state after an authoritative update.",
            "steps": _steps(
                ["CAPTURE", "COMPRESS", "CONTINUE", "REJECT_STALE", "RESTORE", "EXECUTE", "VERIFY", "COMPLETE"],
                old="STATE-A", new="STATE-B", change_at=4,
                events=[
                    "A long work session has accumulated goals, constraints, completed work, and pending work; capture the decision-relevant state.",
                    "The full history is too large to carry forever. Compress it while preserving goal, constraints, decisions, and unresolved items.",
                    "Resume from the compact state and identify the next pending action without replaying completed work.",
                    "AUTHORITATIVE UPDATE: checkpoint is now STATE-B. An old note repeats the prior checkpoint and conflicts with the new state.",
                    "A continuation starts from the compact record; restore the current state without resurrecting the obsolete checkpoint.",
                    "The restored state identifies one executable pending action; carry it out while preserving constraints.",
                    "Execution succeeds. Verify that completed work, current state, and remaining obligations are internally consistent.",
                    "The continuation is consistent, current, and has no unresolved required work.",
                ],
            ),
        },
    ]


def planned_semantic_calls(scenarios: list[dict[str, Any]] | None = None) -> int:
    return sum(len(item["steps"]) for item in (scenarios or build_scenarios()))


def _runtime_error_scoring(generation: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": 0.0,
        "status": "RUNTIME_ERROR",
        "checks": [],
        "evidence": {"raw_response": ""},
        "error": copy.deepcopy(generation.get("error") or {"type": "RUNTIME_ERROR"}),
    }


def _check_pass(scoring: dict[str, Any], name: str) -> bool:
    for row in scoring.get("checks", []) or []:
        if row.get("name") == name:
            return row.get("pass") is True
    return False


def _add_progress_task(runner: Any, label: str) -> None:
    progress = getattr(runner, "progress", None)
    if progress is None:
        return
    old_total = int(progress.total_tasks)
    progress.total_tasks = old_total + 1
    if hasattr(runner, "_record_progress"):
        runner._record_progress(
            "plan_adjusted", label, old_total=old_total,
            new_total=int(progress.total_tasks), reason="autonomous simulation turn added",
        )


def _longest_streak(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def run_autonomous_simulation(
    runner: Any,
    *,
    sequence_start: int = 0,
    thinking_mode: bool = False,
    reasoning_effort: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Run bounded multi-turn simulations and return standard experiment rows."""
    assert runner.store is not None
    cfg = runner.config.get("autonomous_simulation") or {}
    if cfg.get("enabled") is not True:
        return [], {"schema_version": 1, "enabled": False, "scenarios": {}}, sequence_start

    all_scenarios = build_scenarios()
    count = max(1, min(int(cfg.get("scenario_count", 6)), len(all_scenarios)))
    max_steps = max(1, int(cfg.get("steps_per_scenario", 8)))
    base_budget = int(cfg.get("generation_budget", 512))
    ceiling = max(base_budget, int(cfg.get("max_generation_budget", 2048)))
    scenarios = all_scenarios[:count]
    sequence = sequence_start
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    system = (
        "You are operating autonomously inside a deterministic simulation. Preserve the mission goal and "
        "current authoritative checkpoint across the entire growing transcript. At each turn choose the "
        "single best next action from the scenario's allowed actions. Return ONLY one JSON object with keys "
        "action, checkpoint, rationale, state. action and checkpoint must be strings. rationale may be concise; "
        "state may contain your compact working state. Never declare COMPLETE before the environment says all "
        "required verification is satisfied."
    )

    for scenario in scenarios:
        allowed = ", ".join(scenario["allowed_actions"])
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"SCENARIO {scenario['id']}: {scenario['brief']} "
                    f"Initial authoritative checkpoint={scenario['checkpoint']}. Allowed actions: {allowed}."
                ),
            },
        ]
        semantic_results: list[dict[str, Any]] = []
        attempts = 0
        truncation_retries = 0

        for step_index, step in enumerate(scenario["steps"][:max_steps], start=1):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"TURN {step_index}/{min(max_steps, len(scenario['steps']))}. ENVIRONMENT: {step['event']} "
                        "Choose the best next autonomous action now."
                    ),
                }
            )
            budget = base_budget
            attempt_index = 0
            final_row: dict[str, Any] | None = None
            final_response = ""

            while True:
                attempt_index += 1
                attempts += 1
                sequence += 1
                case_id = f"auto-{scenario['id']}-s{step_index:02d}-a{attempt_index:02d}"
                case = {
                    "id": case_id,
                    "category": "autonomous_simulation",
                    "difficulty_level": step_index,
                    "prompt": messages[-1]["content"],
                    "scorer": "json",
                    "expected": {
                        "action": step["expected_action"],
                        "checkpoint": step["expected_checkpoint"],
                    },
                    "required": ["action", "checkpoint"],
                    "timeout_s": float(runner.config.get("limits", {}).get("request_timeout_s", 120)),
                }
                label = f"autonomous {scenario['id']} turn {step_index} budget {budget}"
                _add_progress_task(runner, label)
                if hasattr(runner, "_progress_begin"):
                    runner._progress_begin(label)
                try:
                    options = runner._generation_options(
                        case,
                        {"num_predict": budget, "temperature": 0.0, "seed": 42},
                    )
                    generation, invocation, refs = runner._invoke_generation(
                        stage="autonomous-simulation",
                        case_id=case_id,
                        messages=messages,
                        options=options,
                        request_fields={
                            "think": reasoning_effort if reasoning_effort is not None else thinking_mode
                        },
                    )
                finally:
                    if hasattr(runner, "_progress_complete"):
                        runner._progress_complete(label)

                if generation.get("ok", False):
                    final_response = str((generation.get("normalized") or {}).get("text", ""))
                    scoring = score_case(case, final_response)
                else:
                    final_response = ""
                    scoring = _runtime_error_scoring(generation)
                runner._persist_scoring(case_id, "autonomous-simulation", scoring)
                classification = classify_result(case, generation, scoring)

                spec = ExperimentSpec(
                    experiment_id=make_experiment_id(sequence, scenario["id"], f"auto-s{step_index}-a{attempt_index}"),
                    parent_experiment_id=None,
                    task_id=scenario["id"],
                    task_family="autonomous_simulation",
                    difficulty_level=step_index,
                    hypothesis="measure long-horizon autonomous action, state retention, adaptation, and verification",
                    changed_variable="baseline" if attempt_index == 1 else "generation_budget",
                    thinking_mode=thinking_mode,
                    reasoning_effort=reasoning_effort,
                    generation_budget=budget,
                    context_request=None,
                    temperature=0.0,
                    seed=42,
                    prompt_variant=f"{scenario['id']}-step-{step_index}",
                    recovery_level=None,
                )
                row = {
                    "experiment": spec.to_dict(),
                    "classification": classification,
                    "score": scoring.get("score"),
                    "status": scoring.get("status"),
                    "metrics": copy.deepcopy(generation.get("metrics") or {}),
                    "timing": copy.deepcopy(generation.get("timing") or {}),
                    "phase_metrics": copy.deepcopy(generation.get("phase_metrics") or {}),
                    "evidence_key": case_id,
                    "evidence_refs": refs,
                    "simulation": {
                        "scenario_id": scenario["id"],
                        "step": step_index,
                        "attempt": attempt_index,
                        "generation_budget": budget,
                        "expected_action": step["expected_action"],
                        "expected_checkpoint": step["expected_checkpoint"],
                        "injected_change": bool(step.get("injected_change")),
                        "action_correct": _check_pass(scoring, "value:action"),
                        "checkpoint_correct": _check_pass(scoring, "value:checkpoint"),
                    },
                }
                runner.store.append_jsonl("experiments.jsonl", row)
                runner.store.append_jsonl("autonomous-observations.jsonl", row)
                rows.append(row)
                final_row = row

                result_class = str(classification.get("result_class"))
                if result_class != "ANSWER_CORRECT":
                    runner._write_replay(
                        case=case,
                        invocation=invocation,
                        generation=generation,
                        scoring=scoring,
                        stage="autonomous-simulation",
                        replay_name=case_id,
                    )

                if result_class not in TRUNCATION or budget >= ceiling:
                    break
                budget = min(ceiling, budget * 2)
                truncation_retries += 1

            if final_row is None:
                continue
            semantic_results.append(final_row)
            messages.append(
                {
                    "role": "assistant",
                    "content": final_response if final_response else "{\"action\":\"NO_RESPONSE\",\"checkpoint\":\"UNKNOWN\"}",
                }
            )

        runner.store.write_json(
            f"autonomous-transcripts/{scenario['id']}.json",
            {"scenario": scenario, "messages": messages},
            producer="autonomous-simulation",
            stage="report",
        )
        valid = [r for r in semantic_results if r["classification"].get("valid_for_capability") is True]
        both = [
            bool(r["simulation"]["action_correct"] and r["simulation"]["checkpoint_correct"])
            for r in semantic_results
        ]
        action_correct = sum(bool(r["simulation"]["action_correct"]) for r in semantic_results)
        checkpoint_correct = sum(bool(r["simulation"]["checkpoint_correct"]) for r in semantic_results)
        injected = [r for r in semantic_results if r["simulation"].get("injected_change")]
        injected_correct = sum(
            bool(r["simulation"]["action_correct"] and r["simulation"]["checkpoint_correct"])
            for r in injected
        )
        latencies = [
            float(r.get("timing", {}).get("client_latency_ns")) / 1e9
            for r in semantic_results
            if isinstance(r.get("timing", {}).get("client_latency_ns"), (int, float))
        ]
        eval_tokens = sum(
            int(r.get("metrics", {}).get("eval_count", 0) or 0) for r in semantic_results
        )
        final_ok = bool(
            semantic_results
            and semantic_results[-1]["simulation"]["action_correct"]
            and semantic_results[-1]["simulation"]["checkpoint_correct"]
            and semantic_results[-1]["simulation"]["expected_action"] == "COMPLETE"
        )
        summaries[scenario["id"]] = {
            "title": scenario["title"],
            "tags": list(scenario["tags"]),
            "semantic_turns": len(semantic_results),
            "attempts": attempts,
            "valid_turns": len(valid),
            "invalid_turns": len(semantic_results) - len(valid),
            "step_score": None if not semantic_results else sum(both) / len(semantic_results),
            "action_accuracy": None if not semantic_results else action_correct / len(semantic_results),
            "checkpoint_accuracy": None if not semantic_results else checkpoint_correct / len(semantic_results),
            "injected_change_recovery": None if not injected else injected_correct / len(injected),
            "completed_correctly": final_ok,
            "longest_fully_correct_streak": _longest_streak(both),
            "truncation_retries": truncation_retries,
            "median_step_latency_s": None if not latencies else median(latencies),
            "generated_tokens": eval_tokens,
            "max_transcript_messages": len(messages),
            "result_classes": dict(Counter(str(r["classification"].get("result_class")) for r in semantic_results)),
        }

    scenario_rows = list(summaries.values())
    overall_turns = sum(int(s["semantic_turns"]) for s in scenario_rows)
    overall_correct = sum(
        int(round(float(s["step_score"] or 0.0) * int(s["semantic_turns"]))) for s in scenario_rows
    )
    summary = {
        "schema_version": 1,
        "model": str(runner.model),
        "measurement_policy": {
            "kind": "multi_turn_deterministic_autonomous_simulation",
            "semantic_call_target": planned_semantic_calls(scenarios),
            "token_escalation": "only on THINK_TRUNCATED or ANSWER_TRUNCATED",
            "transcript": "grows across all turns within a scenario",
        },
        "scenario_count": len(summaries),
        "semantic_turns": overall_turns,
        "attempts": len(rows),
        "overall_step_score": None if overall_turns == 0 else overall_correct / overall_turns,
        "completion_rate": None if not scenario_rows else sum(bool(s["completed_correctly"]) for s in scenario_rows) / len(scenario_rows),
        "scenarios": summaries,
    }
    return rows, summary, sequence


def render_autonomous_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"# Long Autonomous Simulation: {summary.get('model', 'unknown')}",
        "",
        f"Scenarios: {summary.get('scenario_count', 0)} | semantic turns: {summary.get('semantic_turns', 0)} | actual attempts: {summary.get('attempts', 0)}",
        f"Overall fully-correct step score: {100.0 * float(summary.get('overall_step_score') or 0.0):.1f}% | completion rate: {100.0 * float(summary.get('completion_rate') or 0.0):.1f}%",
        "",
        "| Scenario | Step score | Action | State | Change recovery | Complete | Attempts | Trunc retries | Median s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario_id, row in (summary.get("scenarios") or {}).items():
        pct = lambda value: "N/A" if value is None else f"{100.0 * float(value):.1f}%"
        med = "N/A" if row.get("median_step_latency_s") is None else f"{float(row['median_step_latency_s']):.2f}"
        lines.append(
            f"| {scenario_id} | {pct(row.get('step_score'))} | {pct(row.get('action_accuracy'))} | "
            f"{pct(row.get('checkpoint_accuracy'))} | {pct(row.get('injected_change_recovery'))} | "
            f"{'YES' if row.get('completed_correctly') else 'NO'} | {row.get('attempts', 0)} | "
            f"{row.get('truncation_retries', 0)} | {med} |"
        )
    lines.extend([
        "",
        "Each scenario preserves its full growing transcript plus raw request/response, scorer, telemetry, failure replay, token-budget, and per-turn evidence.",
        "",
    ])
    return "\n".join(lines)
