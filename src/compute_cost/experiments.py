"""Immutable experiment metadata and lineage helpers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

CONTROLLED_FIELDS = (
    "thinking_mode",
    "generation_budget",
    "context_request",
    "temperature",
    "seed",
    "prompt_variant",
)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "x"


def make_experiment_id(sequence: int, task_id: str, label: str) -> str:
    return f"exp-{sequence:06d}-{_safe(task_id)}-{_safe(label)}"


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    parent_experiment_id: str | None
    task_id: str
    task_family: str
    difficulty_level: int
    hypothesis: str
    changed_variable: str
    thinking_mode: bool
    generation_budget: int
    context_request: int | None
    temperature: float
    seed: int
    prompt_variant: str
    recovery_level: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def changed_fields(parent: ExperimentSpec, child: ExperimentSpec) -> list[str]:
    return [name for name in CONTROLLED_FIELDS if getattr(parent, name) != getattr(child, name)]
