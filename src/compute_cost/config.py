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
        "max_model_calls_per_run": 3000,
    },
    "test1_campaign": {
        "expected_calls": 4300,
        "safety_call_cap": 10000,
        "generation_budget": 256,
        "thinking_mode": True,
        "reasoning_effort": "medium",
        "calibration_anchor_count": 64,
        "calibration_repeats": 5,
        "discovery_min_observations": 16,
        "discovery_max_observations": 64,
        "characterization_top_ingredients": 6,
        "pair_top_ingredients": 8,
        "confirmation_candidates": 24,
        "bootstrap_samples": 500,
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
        "thinking_mode": False,
        "reasoning_effort": None,
        "generation_budget": 256,
        "reliable_threshold": 0.90,
        "unstable_threshold": 0.40,
    },
    "reasoning_curves": {
        "enabled": True,
        "repeats": 3,
    },
    "recovery_lab": {
        "enabled": True,
        "repeats": 3,
        "max_level": "R7",
        "max_attempts_per_candidate": 6,
    },
    "robustness_lab": {
        "enabled": True,
        "repeats": 2,
        "max_perturbations_per_family": 2,
        "max_attempts_per_perturbation": 4,
    },
    "compound_lab": {
        "enabled": True,
        "jump": 2,
        "boundary_repeats": 3,
        "max_experiments_per_compound": 16,
        "max_compounds": 6,
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


def _validate_limits(config: dict[str, Any]) -> None:
    limits = config.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("limits config must be a table")
    max_calls = limits.get("max_model_calls_per_run")
    if not isinstance(max_calls, int) or isinstance(max_calls, bool) or max_calls <= 0:
        raise ValueError("limits.max_model_calls_per_run must be a positive integer")


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
    if c["thinking_mode"]:
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError(
                "capability_campaign.reasoning_effort must be one of: low, medium, high when thinking is enabled"
            )
    elif reasoning_effort is not None:
        raise ValueError(
            "capability_campaign.reasoning_effort must be omitted/None when thinking is disabled"
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


def _validate_reasoning_curves(config: dict[str, Any]) -> None:
    c = config.get("reasoning_curves")
    if not isinstance(c, dict):
        raise ValueError("reasoning_curves config must be a table")
    if not isinstance(c.get("enabled"), bool):
        raise ValueError("reasoning_curves.enabled must be bool")
    repeats = c.get("repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ValueError("reasoning_curves.repeats must be a positive integer")


def _validate_recovery_lab(config: dict[str, Any]) -> None:
    c = config.get("recovery_lab")
    if not isinstance(c, dict):
        raise ValueError("recovery_lab config must be a table")
    if not isinstance(c.get("enabled"), bool):
        raise ValueError("recovery_lab.enabled must be bool")
    repeats = c.get("repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ValueError("recovery_lab.repeats must be a positive integer")
    max_attempts = c.get("max_attempts_per_candidate")
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts <= 0:
        raise ValueError("recovery_lab.max_attempts_per_candidate must be a positive integer")
    if max_attempts < repeats:
        raise ValueError("recovery_lab.max_attempts_per_candidate must be >= recovery_lab.repeats")
    max_level = c.get("max_level")
    if max_level not in {f"R{index}" for index in range(9)}:
        raise ValueError("recovery_lab.max_level must be one of R0 through R8")


def _validate_robustness_lab(config: dict[str, Any]) -> None:
    c = config.get("robustness_lab")
    if not isinstance(c, dict):
        raise ValueError("robustness_lab config must be a table")
    if not isinstance(c.get("enabled"), bool):
        raise ValueError("robustness_lab.enabled must be bool")
    for name in ("repeats", "max_perturbations_per_family", "max_attempts_per_perturbation"):
        value = c.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"robustness_lab.{name} must be a positive integer")
    if c["max_attempts_per_perturbation"] < c["repeats"]:
        raise ValueError(
            "robustness_lab.max_attempts_per_perturbation must be >= robustness_lab.repeats"
        )


def _validate_compound_lab(config: dict[str, Any]) -> None:
    c = config.get("compound_lab")
    if not isinstance(c, dict):
        raise ValueError("compound_lab config must be a table")
    if not isinstance(c.get("enabled"), bool):
        raise ValueError("compound_lab.enabled must be bool")
    for name in ("jump", "boundary_repeats", "max_experiments_per_compound", "max_compounds"):
        value = c.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"compound_lab.{name} must be a positive integer")


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

    _validate_limits(config)
    _validate_characterization(config)
    _validate_capability_campaign(config)
    _validate_reasoning_curves(config)
    _validate_recovery_lab(config)
    _validate_robustness_lab(config)
    _validate_compound_lab(config)
    return config