"""Stable schema primitives shared by benchmark evidence and reports."""

from __future__ import annotations

from enum import Enum
from typing import Any


class StageStatus(str, Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class FailureCode(str, Enum):
    RUNTIME_UNAVAILABLE = "RUNTIME_UNAVAILABLE"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_PULL_FAILED = "MODEL_PULL_FAILED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    TIMEOUT = "TIMEOUT"
    OOM_OR_RESOURCE_LIMIT = "OOM_OR_RESOURCE_LIMIT"
    INVALID_STRUCTURED_OUTPUT = "INVALID_STRUCTURED_OUTPUT"
    SCORER_ERROR = "SCORER_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    USER_LIMIT_STOP = "USER_LIMIT_STOP"
    CAPTURE_GAP = "CAPTURE_GAP"


class ResultClass(str, Enum):
    ANSWER_CORRECT = "ANSWER_CORRECT"
    ANSWER_WRONG = "ANSWER_WRONG"
    THINK_TRUNCATED = "THINK_TRUNCATED"
    ANSWER_TRUNCATED = "ANSWER_TRUNCATED"
    NO_FINAL_ANSWER = "NO_FINAL_ANSWER"
    FORMAT_FAILURE = "FORMAT_FAILURE"
    TOOL_FAILURE = "TOOL_FAILURE"
    CONTEXT_FAILURE = "CONTEXT_FAILURE"
    SELF_CORRECTED = "SELF_CORRECTED"
    REASONING_LOOP = "REASONING_LOOP"
    OVERTHINK_CORRUPTION = "OVERTHINK_CORRUPTION"
    TIMEOUT = "TIMEOUT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    RUNTIME_FAILURE = "RUNTIME_FAILURE"
    TEST_DEFECT = "TEST_DEFECT"
    SCORER_DEFECT = "SCORER_DEFECT"
    CAPTURE_GAP = "CAPTURE_GAP"


class MeasurementKind(str, Enum):
    MEASURED = "MEASURED"
    DERIVED = "DERIVED"
    ESTIMATED = "ESTIMATED"
    UNAVAILABLE = "UNAVAILABLE"


def unavailable(reason: str, *, collector: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"availability": "unavailable", "reason": reason}
    if collector is not None:
        value["collector"] = collector
    return value
