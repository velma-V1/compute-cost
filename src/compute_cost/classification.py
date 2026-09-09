"""Deterministic behavioral classification for characterization results."""

from __future__ import annotations

import json
from typing import Any

from .schema import MeasurementKind, ResultClass


CAPABILITY_RESULTS = {
    ResultClass.ANSWER_CORRECT,
    ResultClass.ANSWER_WRONG,
    ResultClass.FORMAT_FAILURE,
    ResultClass.TOOL_FAILURE,
    ResultClass.CONTEXT_FAILURE,
}


def classify_result(case: dict[str, Any], generation: dict[str, Any], scoring: dict[str, Any]) -> dict[str, Any]:
    normalized = generation.get("normalized") or {}
    text = str(normalized.get("text") or "")
    thinking = str(normalized.get("thinking") or "")
    done_reason = normalized.get("done_reason")
    scorer = str(case.get("scorer") or "")
    error_text = json.dumps(generation.get("error") or {}).lower()

    if not generation.get("ok", False) and "timeout" in error_text:
        result, basis = ResultClass.TIMEOUT, "runtime request timed out"
    elif not generation.get("ok", False) and any(token in error_text for token in ("out of memory", "oom", "resource")):
        result, basis = ResultClass.RESOURCE_LIMIT, "runtime reported resource exhaustion"
    elif not generation.get("ok", False):
        result, basis = ResultClass.RUNTIME_FAILURE, "runtime generation failed"
    elif scoring.get("status") == "SCORER_ERROR":
        result, basis = ResultClass.SCORER_DEFECT, "scorer reported benchmark defect"
    elif scoring.get("score") == 1.0:
        result, basis = ResultClass.ANSWER_CORRECT, "deterministic scorer passed"
    elif done_reason == "length" and thinking and not text:
        result, basis = ResultClass.THINK_TRUNCATED, "length stop with exposed thinking and no final content"
    elif done_reason == "length" and text:
        result, basis = ResultClass.ANSWER_TRUNCATED, "length stop after final content began"
    elif not text:
        result, basis = ResultClass.NO_FINAL_ANSWER, "runtime succeeded but emitted no final content"
    elif any(check.get("name") == "valid_json" and check.get("pass") is False for check in scoring.get("checks", [])):
        result, basis = ResultClass.FORMAT_FAILURE, "structured output was not valid JSON"
    elif scorer == "tool_call":
        result, basis = ResultClass.TOOL_FAILURE, "tool-call scorer failed"
    elif scorer == "context_retrieval":
        result, basis = ResultClass.CONTEXT_FAILURE, "context-retrieval scorer failed"
    else:
        result, basis = ResultClass.ANSWER_WRONG, "runtime completed and deterministic scorer failed"

    return {
        "result_class": result.value,
        "basis": basis,
        "measurement_kind": MeasurementKind.DERIVED.value,
        "valid_for_capability": result in CAPABILITY_RESULTS,
    }
