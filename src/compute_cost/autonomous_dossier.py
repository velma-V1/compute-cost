"""Forensic reconstruction for autonomous-simulation failures and retries."""

from __future__ import annotations

import base64
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .attempt_dossier import persist_attempt_dossier
from .experiments import ExperimentSpec


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def invocation_from_exchange(exchange: dict[str, Any]) -> dict[str, Any]:
    """Recover the exact model-visible request payload from retained raw request bytes."""
    request = exchange.get("request") or {}
    body_b64 = request.get("body_b64")
    payload: dict[str, Any] = {}
    if isinstance(body_b64, str) and body_b64:
        try:
            decoded = base64.b64decode(body_b64)
            candidate = json.loads(decoded.decode("utf-8"))
            if isinstance(candidate, dict):
                payload = candidate
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            payload = {}

    standard = {"model", "messages", "options", "stream"}
    return {
        "model": payload.get("model"),
        "messages": copy.deepcopy(payload.get("messages") or []),
        "options": copy.deepcopy(payload.get("options") or {}),
        "stream": payload.get("stream"),
        "request_fields": {
            key: copy.deepcopy(value)
            for key, value in payload.items()
            if key not in standard
        },
    }


def _telemetry_for_case(run_dir: Path, case_id: str) -> list[dict[str, Any]]:
    path = run_dir / "telemetry.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("case_id") == case_id:
            rows.append(row)
    return rows


def persist_autonomous_dossiers(runner: Any, rows: list[dict[str, Any]]) -> int:
    """Persist every failed autonomous attempt and every retry, including success."""
    run_dir = Path(runner.store.run_dir)
    previous_by_step: dict[tuple[str, int], str] = {}
    written = 0

    for row in rows:
        simulation = row.get("simulation") or {}
        scenario_id = str(simulation.get("scenario_id") or "unknown")
        step = int(simulation.get("step") or 0)
        attempt = int(simulation.get("attempt") or 1)
        key = (scenario_id, step)

        experiment_data = copy.deepcopy(row.get("experiment") or {})
        try:
            spec = ExperimentSpec(**experiment_data)
        except TypeError:
            continue

        parent_id = previous_by_step.get(key)
        if attempt > 1:
            spec = replace(
                spec,
                parent_experiment_id=parent_id,
                changed_variable="generation_budget",
            )

        classification = copy.deepcopy(row.get("classification") or {})
        result_class = str(classification.get("result_class") or "")
        if result_class == "ANSWER_CORRECT" and attempt <= 1:
            previous_by_step[key] = spec.experiment_id
            continue

        refs = copy.deepcopy(row.get("evidence_refs") or {})
        request_id = refs.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            previous_by_step[key] = spec.experiment_id
            continue

        generation = _read_json(
            run_dir / "raw" / "runtime" / "exchanges" / f"{request_id}.json"
        )
        case_id = str(row.get("evidence_key") or "")
        safe_case = case_id.replace("/", "-").replace("\\", "-")
        scoring = _read_json(
            run_dir / "raw" / "scoring" / f"autonomous-simulation-{safe_case}.json"
        )
        if generation is None or scoring is None:
            runner.store.record_capture_gap(
                channel="attempt_dossier",
                collector="persist_autonomous_dossiers",
                error="retained autonomous generation or scoring evidence unavailable",
                affected=case_id or spec.experiment_id,
                continued=True,
            )
            previous_by_step[key] = spec.experiment_id
            continue

        invocation = invocation_from_exchange(generation)
        messages = invocation.get("messages") or []
        last_user = next(
            (
                str(message.get("content") or "")
                for message in reversed(messages)
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            "",
        )
        case = {
            "id": case_id or spec.experiment_id,
            "family_id": "autonomous_simulation",
            "category": "autonomous_simulation",
            "difficulty_level": step,
            "prompt": last_user,
            "scorer": "json",
            "expected": {
                "action": simulation.get("expected_action"),
                "checkpoint": simulation.get("expected_checkpoint"),
            },
            "required": ["action", "checkpoint"],
            "tags": ["autonomous_simulation", scenario_id, f"step_{step}"],
        }
        telemetry = _telemetry_for_case(run_dir, case_id)
        dossier = persist_attempt_dossier(
            runner,
            case=case,
            spec=spec,
            invocation=invocation,
            generation=generation,
            scoring=scoring,
            classification=classification,
            evidence_refs=refs,
            telemetry_before=telemetry[:1],
            telemetry_after=telemetry[1:],
        )
        if dossier is not None:
            written += 1
        previous_by_step[key] = spec.experiment_id

    return written


def install_autonomous_dossier_hook() -> None:
    """Attach dossier reconstruction to the existing autonomous simulation phase."""
    import compute_cost.autonomous_simulation as autonomous

    current = autonomous.run_autonomous_simulation
    if getattr(current, "_autonomous_dossier_hook", False):
        return
    original = current

    def wrapped_run_autonomous_simulation(*args, **kwargs):
        rows, summary, sequence = original(*args, **kwargs)
        runner = args[0] if args else kwargs.get("runner")
        if runner is not None and rows:
            summary = copy.deepcopy(summary)
            summary["forensic_attempt_dossiers"] = persist_autonomous_dossiers(runner, rows)
        return rows, summary, sequence

    wrapped_run_autonomous_simulation._autonomous_dossier_hook = True
    autonomous.run_autonomous_simulation = wrapped_run_autonomous_simulation
