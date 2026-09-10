"""Zero-call integrity reconciliation for completed model-call evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    result: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _runtime_model_request_count(root: Path) -> tuple[int, str]:
    """Count actual authorized model requests, excluding preflight/control exchanges.

    New runs persist an explicit index immediately after each authorized generation
    request. Older/unit-test fixtures fall back to raw request files.
    """
    index = root / "model-call-request-index.jsonl"
    indexed = _read_jsonl(index)
    if indexed:
        request_ids = {
            str(row.get("request_id"))
            for row in indexed
            if row.get("request_id") is not None
        }
        return len(request_ids), "model-call-request-index.jsonl"

    request_dir = root / "raw" / "runtime" / "requests"
    if not request_dir.is_dir():
        return 0, "raw-request-fallback"
    return sum(path.is_file() and path.suffix == ".bin" for path in request_dir.iterdir()), "raw-request-fallback"


def reconcile_run_integrity(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    ledger = _read_json(root / "call-ledger.json")
    dossier_rows = _read_jsonl(root / "attempt-dossiers" / "index.jsonl")
    atlas = _read_json(root / "failure-atlas.json")

    ledger_used = ledger.get("calls_used")
    ledger_used_i = int(ledger_used) if isinstance(ledger_used, int) and not isinstance(ledger_used, bool) else 0
    runtime_requests, request_source = _runtime_model_request_count(root)
    dossier_count = len(dossier_rows)
    failure_dossiers = sum(
        str(row.get("result_class") or "") not in {"", "ANSWER_CORRECT", "SELF_CORRECTED"}
        for row in dossier_rows
    )
    atlas_failures = atlas.get("failures") if isinstance(atlas.get("failures"), list) else []
    atlas_count = len(atlas_failures)

    issues: list[str] = []
    if ledger_used_i != runtime_requests:
        issues.append("LEDGER_RUNTIME_REQUEST_COUNT_MISMATCH")
    if ledger_used_i != dossier_count:
        issues.append("LEDGER_DOSSIER_COUNT_MISMATCH")
    if failure_dossiers != atlas_count:
        issues.append("FAILURE_DOSSIER_ATLAS_COUNT_MISMATCH")

    ceiling = ledger.get("hard_ceiling")
    if isinstance(ceiling, int) and not isinstance(ceiling, bool) and ledger_used_i > ceiling:
        issues.append("CALL_LEDGER_HARD_CEILING_EXCEEDED")

    dossier_ids = {
        str(row.get("experiment_id"))
        for row in dossier_rows
        if row.get("experiment_id") is not None
    }
    atlas_ids = {
        str(row.get("experiment_id"))
        for row in atlas_failures
        if isinstance(row, dict) and row.get("experiment_id") is not None
    }
    failure_dossier_ids = {
        str(row.get("experiment_id"))
        for row in dossier_rows
        if row.get("experiment_id") is not None
        and str(row.get("result_class") or "") not in {"", "ANSWER_CORRECT", "SELF_CORRECTED"}
    }
    if atlas_ids and not atlas_ids.issubset(dossier_ids):
        issues.append("FAILURE_ATLAS_REFERENCES_MISSING_DOSSIER")
    if failure_dossier_ids != atlas_ids and failure_dossiers == atlas_count:
        issues.append("FAILURE_DOSSIER_ATLAS_ID_MISMATCH")

    return {
        "schema_version": 1,
        "ok": not issues,
        "request_count_source": request_source,
        "counts": {
            "ledger_calls_used": ledger_used_i,
            "runtime_requests": runtime_requests,
            "scored_dossiers": dossier_count,
            "failure_dossiers": failure_dossiers,
            "failure_atlas_entries": atlas_count,
        },
        "issues": issues,
    }
