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

    return config
