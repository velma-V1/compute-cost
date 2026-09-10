"""Forensic attempt dossiers for every failure and retry.

This module does not create model calls. It joins already-captured evidence into a
single lossless, queryable record. Hidden model internals that the runtime does not
expose are marked UNOBSERVABLE instead of inferred or fabricated.
"""

from __future__ import annotations

import copy
from typing import Any

RETRY_VARIABLES = {"generation_budget", "replication", "recovery_level", "prompt_variant"}


def _seconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) / 1_000_000_000.0


def _duration(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in metrics:
            return _seconds(metrics.get(name))
    return None


def _count(metrics: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _rate(count: int | None, seconds: float | None) -> float | None:
    if count is None or seconds is None or seconds <= 0:
        return None
    return float(count) / seconds


def should_persist_attempt_dossier(spec: Any, classification: dict[str, Any]) -> bool:
    """Persist every failure and every retry, including successful retries."""
    result_class = str(classification.get("result_class") or "")
    failed = result_class != "ANSWER_CORRECT"
    retry = bool(getattr(spec, "parent_experiment_id", None)) and str(
        getattr(spec, "changed_variable", "")
    ) in RETRY_VARIABLES
    return failed or retry


def build_attempt_dossier(
    *,
    case: dict[str, Any],
    spec: Any,
    invocation: dict[str, Any],
    generation: dict[str, Any],
    scoring: dict[str, Any],
    classification: dict[str, Any],
    evidence_refs: dict[str, Any],
    telemetry_before: list[dict[str, Any]],
    telemetry_after: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized = copy.deepcopy(generation.get("normalized") or {})
    metrics = copy.deepcopy(generation.get("metrics") or {})
    timing = copy.deepcopy(generation.get("timing") or {})
    phase_metrics = copy.deepcopy(generation.get("phase_metrics") or {})
    checks = list(copy.deepcopy(scoring.get("checks") or []))
    right_checks = [row for row in checks if row.get("pass") is True]
    wrong_checks = [row for row in checks if row.get("pass") is False]

    prompt_count = _count(metrics, "prompt_eval_count", "prompt_tokens")
    generation_count = _count(metrics, "eval_count", "completion_tokens", "generated_tokens")
    prompt_seconds = _duration(metrics, "prompt_eval_duration", "prompt_eval_duration_ns")
    generation_seconds = _duration(metrics, "eval_duration", "eval_duration_ns")
    latency_seconds = _seconds(timing.get("client_latency_ns"))

    exposed_thinking = str(normalized.get("thinking") or "")
    parent_id = getattr(spec, "parent_experiment_id", None)
    changed_variable = str(getattr(spec, "changed_variable", "baseline"))
    is_retry = bool(parent_id) and changed_variable in RETRY_VARIABLES

    return {
        "schema_version": 1,
        "experiment_id": str(getattr(spec, "experiment_id", "unknown")),
        "question": {
            "id": str(case.get("id") or getattr(spec, "task_id", "unknown")),
            "family": str(case.get("family_id") or case.get("category") or getattr(spec, "task_family", "unknown")),
            "category": case.get("category"),
            "difficulty_level": case.get("difficulty_level", getattr(spec, "difficulty_level", None)),
            "tags": copy.deepcopy(case.get("tags") or []),
            "scorer": case.get("scorer"),
            "rubric_version": (case.get("capability_map") or {}).get("rubric_version"),
            "full_case_snapshot": copy.deepcopy(case),
            "expected_oracle": copy.deepcopy(case.get("expected")),
            "required_fields": copy.deepcopy(case.get("required")),
            "executable_tests": case.get("tests"),
        },
        "lineage": {
            "is_retry": is_retry,
            "parent_experiment_id": parent_id,
            "changed_variable": changed_variable,
            "hypothesis_or_retry_reason": str(getattr(spec, "hypothesis", "")),
            "generation_budget": getattr(spec, "generation_budget", None),
            "thinking_mode": getattr(spec, "thinking_mode", None),
            "reasoning_effort": getattr(spec, "reasoning_effort", None),
            "context_request": getattr(spec, "context_request", None),
            "temperature": getattr(spec, "temperature", None),
            "seed": getattr(spec, "seed", None),
            "prompt_variant": getattr(spec, "prompt_variant", None),
            "recovery_level": getattr(spec, "recovery_level", None),
        },
        "model_visible": {
            "model": invocation.get("model"),
            "messages": copy.deepcopy(invocation.get("messages") or []),
            "options": copy.deepcopy(invocation.get("options") or {}),
            "request_fields": copy.deepcopy(invocation.get("request_fields") or {}),
            "stream": invocation.get("stream"),
            "exact_serialized_request_ref": copy.deepcopy(evidence_refs.get("request")),
            "request_id": evidence_refs.get("request_id"),
        },
        "model_output": {
            "final_answer": str(normalized.get("text") or ""),
            "exposed_thinking": exposed_thinking,
            "exposed_thinking_status": "CAPTURED" if exposed_thinking else "NOT_EXPOSED_BY_RUNTIME",
            "hidden_internal_reasoning": "UNOBSERVABLE",
            "hidden_internal_reasoning_note": "Only reasoning explicitly returned by the model/runtime can be captured; unexposed internal activations or hidden chain-of-thought are not available.",
            "tool_calls": copy.deepcopy(normalized.get("tool_calls") or []),
            "done_reason": normalized.get("done_reason"),
            "normalized_runtime_output": normalized,
            "raw_response_ref": copy.deepcopy(evidence_refs.get("response")),
            "raw_stream_refs": copy.deepcopy(evidence_refs.get("streams") or []),
            "runtime_error_ref": copy.deepcopy(evidence_refs.get("error")),
            "runtime_ok": bool(generation.get("ok", False)),
            "http_status": generation.get("http_status"),
        },
        "evaluation": {
            "result_class": classification.get("result_class"),
            "classification_basis": classification.get("basis"),
            "valid_for_capability": classification.get("valid_for_capability"),
            "score": scoring.get("score"),
            "status": scoring.get("status"),
            "right_checks": right_checks,
            "wrong_checks": wrong_checks,
            "all_checks": checks,
            "scoring_evidence": copy.deepcopy(scoring.get("evidence") or {}),
            "scoring_error": copy.deepcopy(scoring.get("error")),
            "what_was_right": right_checks,
            "what_was_wrong": wrong_checks,
        },
        "performance": {
            "prompt_tokens": prompt_count,
            "generated_tokens": generation_count,
            "prompt_eval_seconds": prompt_seconds,
            "generation_seconds": generation_seconds,
            "client_latency_seconds": latency_seconds,
            "prompt_tokens_per_second": _rate(prompt_count, prompt_seconds),
            "generation_tokens_per_second": _rate(generation_count, generation_seconds),
            "runtime_metrics": metrics,
            "phase_metrics": phase_metrics,
            "timing": timing,
        },
        "telemetry": {
            "before": copy.deepcopy(telemetry_before),
            "after": copy.deepcopy(telemetry_after),
        },
        "evidence": {
            "request": copy.deepcopy(evidence_refs.get("request")),
            "response": copy.deepcopy(evidence_refs.get("response")),
            "streams": copy.deepcopy(evidence_refs.get("streams") or []),
            "runtime_error": copy.deepcopy(evidence_refs.get("error")),
            "all_refs": copy.deepcopy(evidence_refs),
            "observable_runtime_envelope": {
                "metrics": metrics,
                "timing": timing,
                "phase_metrics": phase_metrics,
                "done_reason": normalized.get("done_reason"),
            },
        },
    }


def persist_attempt_dossier(
    runner: Any,
    *,
    case: dict[str, Any],
    spec: Any,
    invocation: dict[str, Any],
    generation: dict[str, Any],
    scoring: dict[str, Any],
    classification: dict[str, Any],
    evidence_refs: dict[str, Any],
    telemetry_before: list[dict[str, Any]],
    telemetry_after: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not should_persist_attempt_dossier(spec, classification):
        return None
    dossier = build_attempt_dossier(
        case=case,
        spec=spec,
        invocation=invocation,
        generation=generation,
        scoring=scoring,
        classification=classification,
        evidence_refs=evidence_refs,
        telemetry_before=telemetry_before,
        telemetry_after=telemetry_after,
    )
    safe_id = dossier["experiment_id"].replace("/", "-").replace("\\", "-")
    relative = f"attempt-dossiers/{safe_id}.json"
    written = runner.store.write_json(
        relative,
        dossier,
        producer="attempt-dossier",
        stage="forensic",
        case_id=str(case.get("id") or "unknown"),
    )
    runner.store.append_jsonl(
        "attempt-dossiers/index.jsonl",
        {
            "experiment_id": dossier["experiment_id"],
            "question_id": dossier["question"]["id"],
            "family": dossier["question"]["family"],
            "difficulty_level": dossier["question"]["difficulty_level"],
            "result_class": dossier["evaluation"]["result_class"],
            "is_retry": dossier["lineage"]["is_retry"],
            "parent_experiment_id": dossier["lineage"]["parent_experiment_id"],
            "changed_variable": dossier["lineage"]["changed_variable"],
            "path": relative,
            "sha256": written.get("sha256"),
        },
    )
    return dossier
