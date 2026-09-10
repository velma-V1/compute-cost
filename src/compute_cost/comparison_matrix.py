"""Fixed cross-model comparison matrix and native reasoning controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

FIXED_LEVELS = (2, 5, 8, 10)
HARD_CALL_LIMIT = 700


@dataclass(frozen=True)
class ReasoningCondition:
    role: str
    native_name: str
    request_value: Any


def native_reasoning_conditions(model: str) -> tuple[ReasoningCondition, ...]:
    name = model.lower()
    if name.startswith("gpt-oss"):
        return (
            ReasoningCondition("BASELINE_MINIMAL", "reasoning_effort", "low"),
            ReasoningCondition("ENHANCED", "reasoning_effort", "medium"),
            ReasoningCondition("MAX_NATIVE", "reasoning_effort", "high"),
        )
    if name.startswith("qwen"):
        return (
            ReasoningCondition("BASELINE_MINIMAL", "think", False),
            ReasoningCondition("ENHANCED", "think", True),
        )
    if name.startswith("devstral"):
        return (ReasoningCondition("BASELINE_MINIMAL", "think", False),)
    return (ReasoningCondition("BASELINE_MINIMAL", "default", None),)


def planned_fixed_core(
    model: str,
    *,
    family_count: int,
    scenario_count: int,
    turns: int,
) -> dict[str, int]:
    conditions = len(native_reasoning_conditions(model))
    capability = int(family_count) * len(FIXED_LEVELS) * conditions
    autonomous = int(scenario_count) * int(turns) * conditions
    total = capability + autonomous
    return {
        "capability": capability,
        "autonomous": autonomous,
        "total": total,
        "reserve": max(0, HARD_CALL_LIMIT - total),
        "limit": HARD_CALL_LIMIT,
    }


def fixed_comparison_cells(
    cases: Iterable[dict[str, Any]], model: str
) -> list[dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for case in cases:
        family = str(case.get("family_id") or case.get("category") or "")
        level = case.get("difficulty_level")
        if family and isinstance(level, int) and not isinstance(level, bool):
            index.setdefault((family, level), case)

    families = sorted({family for family, _ in index})
    cells: list[dict[str, Any]] = []
    for family in families:
        for level in FIXED_LEVELS:
            case = index.get((family, level))
            for condition in native_reasoning_conditions(model):
                cells.append(
                    {
                        "family_id": family,
                        "difficulty_level": level,
                        "task_id": None if case is None else str(case.get("id")),
                        "fixture": case,
                        "status": "MISSING_FIXTURE_COVERAGE" if case is None else "READY",
                        "reasoning_role": condition.role,
                        "native_reasoning_control": condition.native_name,
                        "native_reasoning_value": condition.request_value,
                        "comparison_scope": "fixed_core",
                    }
                )
    return cells
