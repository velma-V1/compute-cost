"""Post-run synthesis of capability economics, profile, and operating policy.

This layer is read-only with respect to model execution. It consumes completed
experiment and telemetry evidence after telemetry collection has stopped, then
derives cost, value, profile, and conservative routing policy without new model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .capability_profile import build_capability_profile, render_capability_profile
from .cost_value import build_cost_map, build_value_map
from .operating_policy import build_operating_policy


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


def _boundary_repeats(root: Path) -> int:
    config = _read_json(root / "resolved-config.json") or {}
    campaign = config.get("capability_campaign")
    if not isinstance(campaign, dict):
        return 3
    value = campaign.get("boundary_repeats", 3)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return 3
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def build_cost_value_outputs(
    model: str,
    rows: Iterable[dict[str, Any]],
    frontiers: dict[str, Any],
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build finalized economics and persist evidence-gated decision artifacts."""
    root = Path(run_dir)
    materialized_rows = list(rows)
    telemetry = _read_jsonl(root / "telemetry.jsonl")
    compound_map = _read_json(root / "compound-map.json")

    cost_map = build_cost_map(model, materialized_rows, telemetry)
    value_map = build_value_map(
        model,
        frontiers,
        cost_map,
        reasoning_curves=_read_json(root / "reasoning-curves.json"),
        recovery_map=_read_json(root / "recovery-map.json"),
        robustness_map=_read_json(root / "robustness-map.json"),
        compound_map=compound_map,
    )
    policy = build_operating_policy(
        model,
        frontiers,
        value_map,
        compound_map=compound_map,
        boundary_repeats=_boundary_repeats(root),
    )
    _write_json(root / "inverted-operating-policy.json", policy)

    coverage = _read_json(root / "coverage-ledger.json")
    failure_atlas = _read_json(root / "failure-atlas.json")
    if coverage is not None and failure_atlas is not None:
        profile = build_capability_profile(
            model,
            frontiers,
            coverage,
            value_map,
            failure_atlas,
            policy,
        )
        _write_json(root / "capability-profile.json", profile)
        (root / "capability-profile.md").write_text(
            render_capability_profile(profile),
            encoding="utf-8",
        )

    return cost_map, value_map
