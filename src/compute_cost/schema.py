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


def unavailable(reason: str, *, collector: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"availability": "unavailable", "reason": reason}
    if collector is not None:
        value["collector"] = collector
    return value
