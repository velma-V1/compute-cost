"""Configuration loading with stable defaults and explicit overrides."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG: dict[str, Any] = {
    "runtime": {"endpoint": "http://127.0.0.1:11434"},
    "generation": {"temperature": 0.0, "seed": 42, "max_tokens": 256},
    "telemetry": {"interval_s": 0.25, "nvidia": True},
    "limits": {
        "request_timeout_s": 120,
        "host_ram_percent": 95.0,
        "vram_percent": 98.0,
        "context_schedule": [4096, 8192, 16384, 32768],
        "consecutive_context_failures": 2,
        "sustained_iterations": 8,
    },
    "cost": {"electricity_per_kwh": 0.0, "electricity_configured": False},
    "evidence": {"retain_stream_chunks": True, "retain_raw_collectors": True},
    "characterization": {
        "initial_generation_budget": 256,
        "min_generation_budget": 32,
        "max_generation_budget": 2048,
        "generation_budget_granularity": 32,
        "boundary_repeats": 3,
        "max_experiments_per_task": 12,
        "think_off_generation_budget": 256,
    },
    "capability_campaign": {
        "anchor_level": 2,
        "jump": 3,
        "boundary_repeats": 5,
        "max_experiments_per_family": 24,
        "thinking_mode": True,
        "reasoning_effort": "medium",
        "generation_budget": 256,
        "reliable_threshold": 0.90,
        "unstable_threshold": 0.40,
    },
}


def _deep_merge(target: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    for key, value in incoming.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _apply_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _validate_characterization(config: dict[str, Any]) -> None:
    c = config.get("characterization")
    if not isinstance(c, dict):
        raise ValueError("characterization config must be a table")
    integer_fields = (
        "initial_generation_budget",
        "min_generation_budget",
        "max_generation_budget",
        "generation_budget_granularity",
        "boundary_repeats",
        "max_experiments_per_task",
        "think_off_generation_budget",
    )
    for name in integer_fields:
        value = c.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"characterization.{name} must be a positive integer")
    if not (
        c["min_generation_budget"]
        <= c["initial_generation_budget"]
        <= c["max_generation_budget"]
    ):
        raise ValueError("characterization generation budgets must satisfy min <= initial <= max")
    if c["think_off_generation_budget"] > c["max_generation_budget"]:
        raise ValueError(
            "characterization think_off_generation_budget must be <= max_generation_budget"
        )


def _validate_capability_campaign(config: dict[str, Any]) -> None:
    c = config.get("capability_campaign")
    if not isinstance(c, dict):
        raise ValueError("capability_campaign config must be a table")

    for name in ("anchor_level", "jump", "boundary_repeats", "max_experiments_per_family", "generation_budget"):
        value = c.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"capability_campaign.{name} must be a positive integer")

    if not 0 <= c["anchor_level"] <= 10:
        raise ValueError("capability_campaign.anchor_level must be between 0 and 10")
    if not isinstance(c.get("thinking_mode"), bool):
        raise ValueError("capability_campaign.thinking_mode must be bool")
    reasoning_effort = c.get("reasoning_effort")
    if reasoning_effort not in {"low", "medium", "high"}:
        raise ValueError(
            "capability_campaign.reasoning_effort must be one of: low, medium, high"
        )

    reliable = c.get("reliable_threshold")
    unstable = c.get("unstable_threshold")
    if isinstance(reliable, bool) or not isinstance(reliable, (int, float)):
        raise ValueError("capability_campaign.reliable_threshold must be numeric")
    if isinstance(unstable, bool) or not isinstance(unstable, (int, float)):
        raise ValueError("capability_campaign.unstable_threshold must be numeric")
    if not 0.0 <= float(unstable) <= float(reliable) <= 1.0:
        raise ValueError(
            "capability_campaign thresholds must satisfy 0 <= unstable <= reliable <= 1"
        )


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)

    if path is not None:
        with Path(path).open("rb") as handle:
            _deep_merge(config, tomllib.load(handle))

    if overrides:
        for key, value in overrides.items():
            _apply_dotted(config, key, value)
        if "cost.electricity_per_kwh" in overrides:
            config["cost"]["electricity_configured"] = True

    _validate_characterization(config)
    _validate_capability_campaign(config)
    return config
