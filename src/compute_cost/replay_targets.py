"""Deterministic selection of high-value exact replay targets.

The complete replay registry remains the source of truth. This module only groups
and ranks already-captured snapshots so a later retest can start with the smallest
set most likely to resolve a capability boundary or recovery question.
"""

from __future__ import annotations

from typing import Any, Iterable


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _failure_signature(failure: dict[str, Any], *, model_failure: bool) -> str | None:
    if not model_failure:
        return None
    signature = failure.get("failure_signature")
    subtype = signature.get("subtype") if isinstance(signature, dict) else None
    if not isinstance(subtype, str) or not subtype or subtype == "unresolved":
        return None
    return subtype


def _reason_and_priority(
    failure: dict[str, Any],
    frontier: dict[str, Any],
    recovery_level: Any,
) -> tuple[str, int]:
    valid_model_failure = (
        failure.get("failure_origin") == "MODEL_FAILURE"
        and failure.get("valid_for_capability") is True
    )
    if not valid_model_failure:
        return "EVIDENCE_INTEGRITY_REPRODUCTION", 4
    if isinstance(recovery_level, str) and recovery_level:
        return "RECOVERY_LIMIT_RETEST", 2

    difficulty = _integer(failure.get("difficulty_level"))
    first_failure = _integer(frontier.get("first_failure_level"))
    if difficulty is not None and first_failure is not None and difficulty == first_failure:
        return "RAW_BOUNDARY_RETEST", 1
    return "MODEL_FAILURE_RETEST", 3


def build_replay_targets(
    model: str,
    frontiers: dict[str, Any],
    failure_atlas: dict[str, Any],
    replay_index: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Group equivalent replay snapshots and rank the groups by decision value."""
    registry = [row for row in replay_index if isinstance(row, dict)]
    failures_by_id = {
        str(row.get("experiment_id")): row
        for row in (failure_atlas.get("failures") or [])
        if isinstance(row, dict) and row.get("experiment_id") is not None
    }
    frontier_families = frontiers.get("families") or {}

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    matched_entries = 0
    unmatched_entries = 0
    seen_replay_ids: set[str] = set()

    for index_row in registry:
        replay_id = index_row.get("replay_id")
        path = index_row.get("path")
        if not isinstance(replay_id, str) or not replay_id or replay_id in seen_replay_ids:
            unmatched_entries += 1
            continue
        seen_replay_ids.add(replay_id)
        failure = failures_by_id.get(replay_id)
        if not isinstance(failure, dict) or not isinstance(path, str) or not path:
            unmatched_entries += 1
            continue
        matched_entries += 1

        family_id = str(failure.get("family_id") or index_row.get("family_id") or "unknown")
        difficulty = _integer(failure.get("difficulty_level"))
        if difficulty is None:
            difficulty = _integer(index_row.get("difficulty_level"))
        result_class = str(failure.get("result_class") or index_row.get("result_class") or "unknown")
        recovery_level = failure.get("recovery_level")
        if recovery_level is None:
            recovery_level = index_row.get("recovery_level")
        frontier = frontier_families.get(family_id)
        frontier = frontier if isinstance(frontier, dict) else {}
        reason, priority = _reason_and_priority(failure, frontier, recovery_level)
        failure_origin = str(failure.get("failure_origin") or "UNRESOLVED")
        signature = _failure_signature(
            failure,
            model_failure=(
                failure_origin == "MODEL_FAILURE"
                and failure.get("valid_for_capability") is True
            ),
        )

        key = (
            priority,
            reason,
            family_id,
            difficulty,
            result_class,
            recovery_level,
            failure_origin,
            signature,
        )
        existing = groups.get(key)
        if existing is None:
            groups[key] = {
                "priority": priority,
                "reason": reason,
                "family_id": family_id,
                "difficulty_level": difficulty,
                "result_class": result_class,
                "recovery_level": recovery_level,
                "failure_origin": failure_origin,
                "failure_signature": signature,
                "primary_replay_id": replay_id,
                "primary_path": path,
                "supporting_replay_ids": [],
                "evidence_count": 1,
            }
        else:
            existing["supporting_replay_ids"].append(replay_id)
            existing["evidence_count"] = int(existing["evidence_count"]) + 1

    targets = sorted(
        groups.values(),
        key=lambda row: (
            int(row["priority"]),
            str(row["family_id"]),
            99 if row["difficulty_level"] is None else int(row["difficulty_level"]),
            str(row["result_class"]),
            str(row["recovery_level"] or ""),
            str(row["primary_replay_id"]),
        ),
    )

    counts = {
        "RAW_BOUNDARY_RETEST": 0,
        "RECOVERY_LIMIT_RETEST": 0,
        "MODEL_FAILURE_RETEST": 0,
        "EVIDENCE_INTEGRITY_REPRODUCTION": 0,
    }
    for target in targets:
        counts[str(target["reason"])] += 1

    return {
        "schema_version": 1,
        "model": model,
        "selection_policy": {
            "full_registry_is_source_of_truth": True,
            "executes_model_calls": False,
            "replicates_are_grouped_not_discarded": True,
            "non_model_failures_are_not_capability_weaknesses": True,
            "priority_order": [
                "RAW_BOUNDARY_RETEST",
                "RECOVERY_LIMIT_RETEST",
                "MODEL_FAILURE_RETEST",
                "EVIDENCE_INTEGRITY_REPRODUCTION",
            ],
        },
        "summary": {
            "registry_entries": len(registry),
            "matched_entries": matched_entries,
            "selected_groups": len(targets),
            "raw_boundary_groups": counts["RAW_BOUNDARY_RETEST"],
            "recovery_limit_groups": counts["RECOVERY_LIMIT_RETEST"],
            "other_model_failure_groups": counts["MODEL_FAILURE_RETEST"],
            "evidence_integrity_groups": counts["EVIDENCE_INTEGRITY_REPRODUCTION"],
            "unmatched_registry_entries": unmatched_entries,
        },
        "targets": targets,
    }
