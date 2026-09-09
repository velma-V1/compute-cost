"""Materialize exact frontier-boundary replays from retained run evidence.

This module performs no model calls. It reconstructs replay snapshots only from
artifacts already persisted by the runner, then rewrites the boundary slice of
the replay registry deterministically. Re-running synthesis is therefore
idempotent and historical runs can be upgraded without re-executing the model.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _encoded_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _encoded_json(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return default
    return value


def _boundary_repeats(root: Path) -> int:
    config = _read_json(root / "resolved-config.json") or {}
    campaign = config.get("capability_campaign")
    if not isinstance(campaign, dict):
        return 3
    return _positive_int(campaign.get("boundary_repeats"), 3)


def _family_level(row: dict[str, Any]) -> tuple[str | None, int | None]:
    experiment = row.get("experiment")
    if not isinstance(experiment, dict):
        return None, None
    family = experiment.get("task_family") or experiment.get("task_id")
    level = experiment.get("difficulty_level")
    if not isinstance(family, str) or not family:
        return None, None
    if isinstance(level, bool) or not isinstance(level, int):
        return family, None
    return family, level


def _case_index(root: Path) -> tuple[str | None, dict[tuple[str, int], dict[str, Any]]]:
    benchmark = _read_json(root / "benchmark-snapshot.json") or {}
    benchmark_version = benchmark.get("benchmark_version")
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for case in benchmark.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        family = case.get("family_id") or case.get("category")
        level = case.get("difficulty_level")
        if not isinstance(family, str) or not family:
            continue
        if isinstance(level, bool) or not isinstance(level, int):
            continue
        index[(family, level)] = copy.deepcopy(case)
    return (str(benchmark_version) if benchmark_version is not None else None), index


def _request_body(exchange: dict[str, Any]) -> dict[str, Any]:
    request = exchange.get("request")
    if not isinstance(request, dict):
        return {}
    body = request.get("body_text")
    if not isinstance(body, str) or not body:
        return {}
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _invocation_from_exchange(
    model: str,
    exchange: dict[str, Any],
    experiment: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    body = _request_body(exchange)
    messages = body.get("messages")
    if not isinstance(messages, list):
        messages = [{"role": "user", "content": str(case.get("prompt", ""))}]
    options = body.get("options")
    if not isinstance(options, dict):
        options = {
            "num_predict": experiment.get("generation_budget"),
            "temperature": experiment.get("temperature"),
            "seed": experiment.get("seed"),
        }
        options = {key: value for key, value in options.items() if value is not None}

    request_fields = {
        key: copy.deepcopy(value)
        for key, value in body.items()
        if key not in {"model", "messages", "options", "stream"}
    }
    if not request_fields:
        effort = experiment.get("reasoning_effort")
        thinking = experiment.get("thinking_mode")
        if effort is not None:
            request_fields["think"] = effort
        elif thinking is not None:
            request_fields["think"] = thinking

    return {
        "model": str(body.get("model") or model),
        "messages": copy.deepcopy(messages),
        "options": copy.deepcopy(options),
        "stream": bool(body.get("stream", True)),
        "request_fields": request_fields,
    }


def _telemetry_for(root: Path, experiment_id: str) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(row)
        for row in _read_jsonl(root / "telemetry.jsonl")
        if row.get("stage") == "characterize" and row.get("case_id") == experiment_id
    ]


def _safe_scoring_id(experiment_id: str) -> str:
    return experiment_id.replace("/", "-").replace("\\", "-")


def _snapshot_from_retained_evidence(
    model: str,
    root: Path,
    row: dict[str, Any],
    case: dict[str, Any],
    *,
    benchmark_version: str | None,
    resolved_config: dict[str, Any],
    family_id: str,
    role: str,
    level: int,
    bracket: dict[str, Any],
    boundary_repeats: int,
) -> tuple[str, dict[str, Any]] | None:
    experiment = row.get("experiment")
    classification = row.get("classification")
    evidence_refs = row.get("evidence_refs")
    if not isinstance(experiment, dict) or not isinstance(classification, dict):
        return None
    if not isinstance(evidence_refs, dict):
        return None

    experiment_id = experiment.get("experiment_id")
    request_id = evidence_refs.get("request_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        return None
    if not isinstance(request_id, str) or not request_id:
        return None

    generation = _read_json(root / "raw" / "runtime" / "exchanges" / f"{request_id}.json")
    scoring = _read_json(
        root / "raw" / "scoring" / f"characterize-{_safe_scoring_id(experiment_id)}.json"
    )
    if generation is None or scoring is None:
        return None

    replay_id = f"{experiment_id}--boundary"
    snapshot = {
        "schema_version": 2,
        "benchmark_version": benchmark_version,
        "stage": "characterize",
        "model": model,
        "case": copy.deepcopy(case),
        "experiment": copy.deepcopy(experiment),
        "classification": copy.deepcopy(classification),
        "evidence_key": row.get("evidence_key") or experiment_id,
        "evidence_refs": copy.deepcopy(evidence_refs),
        "invocation": _invocation_from_exchange(model, generation, experiment, case),
        "generation": copy.deepcopy(generation),
        "scoring": copy.deepcopy(scoring),
        "telemetry_for_experiment": _telemetry_for(root, experiment_id),
        "resolved_config": copy.deepcopy(resolved_config),
        "source_experiment_id": experiment_id,
        "boundary": {
            "family_id": family_id,
            "role": role,
            "level": level,
            "transition_bracket": copy.deepcopy(bracket),
            "boundary_repeats": boundary_repeats,
        },
    }
    return replay_id, snapshot


def materialize_boundary_replays(
    model: str,
    frontiers: dict[str, Any],
    rows: Iterable[dict[str, Any]],
    run_dir: str | Path,
) -> dict[str, Any]:
    """Persist exact lower/upper transition snapshots from retained evidence only."""
    root = Path(run_dir)
    materialized_rows = [row for row in rows if isinstance(row, dict)]
    boundary_repeats = _boundary_repeats(root)
    benchmark_version, cases = _case_index(root)
    resolved_config = _read_json(root / "resolved-config.json") or {}

    by_family_level: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in materialized_rows:
        family, level = _family_level(row)
        classification = row.get("classification")
        if family is None or level is None or not isinstance(classification, dict):
            continue
        if classification.get("valid_for_capability") is not True:
            continue
        by_family_level.setdefault((family, level), []).append(row)

    boundary_dir = root / "replay" / "boundaries"
    boundary_dir.mkdir(parents=True, exist_ok=True)
    for existing in boundary_dir.glob("*.json"):
        existing.unlink()

    boundary_entries: list[dict[str, Any]] = []
    family_summary: dict[str, Any] = {}
    boundary_families = 0

    for family_id, frontier in sorted((frontiers.get("families") or {}).items()):
        if not isinstance(frontier, dict):
            continue
        bracket = frontier.get("transition_bracket")
        if not isinstance(bracket, dict):
            continue
        lower = bracket.get("lower_level")
        upper = bracket.get("upper_level")
        if isinstance(lower, bool) or not isinstance(lower, int):
            continue
        if isinstance(upper, bool) or not isinstance(upper, int):
            continue

        lower_rows = [
            row
            for row in by_family_level.get((str(family_id), lower), [])
            if (row.get("classification") or {}).get("result_class") == "ANSWER_CORRECT"
        ]
        upper_rows = [
            row
            for row in by_family_level.get((str(family_id), upper), [])
            if (row.get("classification") or {}).get("result_class") != "ANSWER_CORRECT"
        ]
        if len(lower_rows) < boundary_repeats or len(upper_rows) < boundary_repeats:
            continue

        lower_case = cases.get((str(family_id), lower))
        upper_case = cases.get((str(family_id), upper))
        if lower_case is None or upper_case is None:
            continue

        written = 0
        for role, level, selected, case in (
            ("lower_reliable", lower, lower_rows, lower_case),
            ("upper_transition", upper, upper_rows, upper_case),
        ):
            for row in selected:
                built = _snapshot_from_retained_evidence(
                    model,
                    root,
                    row,
                    case,
                    benchmark_version=benchmark_version,
                    resolved_config=resolved_config,
                    family_id=str(family_id),
                    role=role,
                    level=level,
                    bracket=bracket,
                    boundary_repeats=boundary_repeats,
                )
                if built is None:
                    continue
                replay_id, snapshot = built
                relative_path = f"replay/boundaries/{replay_id}.json"
                sha256 = _write_json(root / relative_path, snapshot)
                experiment = snapshot.get("experiment") or {}
                classification = snapshot.get("classification") or {}
                boundary_entries.append(
                    {
                        "replay_id": replay_id,
                        "category": "boundaries",
                        "path": relative_path,
                        "canonical_sha256": sha256,
                        "source_experiment_id": snapshot["source_experiment_id"],
                        "family_id": str(family_id),
                        "task_id": experiment.get("task_id"),
                        "difficulty_level": level,
                        "result_class": classification.get("result_class"),
                        "valid_for_capability": True,
                        "recovery_level": experiment.get("recovery_level"),
                        "parent_experiment_id": experiment.get("parent_experiment_id"),
                        "boundary_role": role,
                    }
                )
                written += 1

        if written:
            boundary_families += 1
            family_summary[str(family_id)] = {
                "lower_level": lower,
                "upper_level": upper,
                "snapshots": written,
            }

    index_path = root / "replay" / "index.jsonl"
    preserved = [row for row in _read_jsonl(index_path) if row.get("category") != "boundaries"]
    boundary_entries.sort(key=lambda row: str(row.get("replay_id") or ""))
    _write_jsonl(index_path, [*preserved, *boundary_entries])

    return {
        "schema_version": 1,
        "model": model,
        "executes_model_calls": False,
        "summary": {
            "boundary_snapshots": len(boundary_entries),
            "boundary_families": boundary_families,
        },
        "families": family_summary,
    }
