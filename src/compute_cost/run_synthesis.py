"""Post-run synthesis of capability cost and value evidence.

This layer is intentionally read-only with respect to model execution. It consumes
completed experiment and telemetry evidence after telemetry collection has stopped,
then delegates measurement and routing economics to :mod:`compute_cost.cost_value`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .cost_value import build_cost_map, build_value_map


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def build_cost_value_outputs(
    model: str,
    rows: Iterable[dict[str, Any]],
    frontiers: dict[str, Any],
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build run economics from complete retained evidence without new model calls."""
    root = Path(run_dir)
    materialized_rows = list(rows)
    telemetry = _read_jsonl(root / "telemetry.jsonl")

    cost_map = build_cost_map(model, materialized_rows, telemetry)
    value_map = build_value_map(
        model,
        frontiers,
        cost_map,
        reasoning_curves=_read_json(root / "reasoning-curves.json"),
        recovery_map=_read_json(root / "recovery-map.json"),
        robustness_map=_read_json(root / "robustness-map.json"),
        compound_map=_read_json(root / "compound-map.json"),
    )
    return cost_map, value_map
