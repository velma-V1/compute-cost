"""Categorized replay registry with backward-compatible lookup."""

from __future__ import annotations

import json
from pathlib import Path


ANOMALY_RESULTS = {
    "THINK_TRUNCATED",
    "ANSWER_TRUNCATED",
    "SCORER_DEFECT",
    "TEST_DEFECT",
    "CAPTURE_GAP",
    "RUNTIME_FAILURE",
    "TIMEOUT",
    "RESOURCE_LIMIT",
}


def replay_category(result_class: str, *, recovery_level: str | None) -> str:
    """Select the canonical replay bucket without changing legacy aliases."""
    if recovery_level:
        return "recoveries"
    if str(result_class) in ANOMALY_RESULTS:
        return "anomalies"
    return "failures"


def _safe_replay_id(replay_id: str) -> str:
    value = str(replay_id)
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("replay id must be a single safe path component")
    return value


def _inside_run(run_dir: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("replay index path must stay inside run directory")
    base = run_dir.resolve()
    target = (run_dir / rel).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError("replay index path must stay inside run directory") from exc
    return target


def resolve_replay_snapshot(run_dir: str | Path, replay_id: str) -> Path:
    """Resolve historical root snapshots first, then the v2 replay index."""
    run = Path(run_dir)
    safe_id = _safe_replay_id(replay_id)

    legacy = run / "replay" / f"{safe_id}.json"
    if legacy.is_file():
        return legacy

    index_path = run / "replay" / "index.jsonl"
    if index_path.is_file():
        for raw_line in index_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if str(row.get("replay_id")) != safe_id:
                continue
            relative = row.get("path")
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"replay index entry {safe_id} has no path")
            target = _inside_run(run, relative)
            if not target.is_file():
                raise FileNotFoundError(target)
            return target

    raise FileNotFoundError(run / "replay" / f"{safe_id}.json")
