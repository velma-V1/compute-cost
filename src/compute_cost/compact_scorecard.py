"""Compact numeric scorecard derived from already-retained capability evidence.

This module makes no model calls. It converts the existing experiment/frontier
records into the per-family, per-level, token-budget and runtime measurements
needed for direct model comparison.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import Any, Iterable


TRUNCATION = {"THINK_TRUNCATED", "ANSWER_TRUNCATED"}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _score_percent(correct: int, valid: int) -> float | None:
    if valid <= 0:
        return None
    return round((correct / valid) * 100.0, 3)


def _median_latency(rows: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for row in rows:
        timing = row.get("timing") if isinstance(row.get("timing"), dict) else {}
        ns = _number(timing.get("client_latency_ns"))
        if ns is not None:
            values.append(ns / 1_000_000_000)
    return None if not values else median(values)


def _token_total(rows: list[dict[str, Any]]) -> int | None:
    values: list[int] = []
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        value = metrics.get("eval_count")
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        values.append(value)
    return None if not values else sum(values)


def _budget_curve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        spec = row.get("experiment") if isinstance(row.get("experiment"), dict) else {}
        budget = spec.get("generation_budget")
        if isinstance(budget, bool) or not isinstance(budget, int):
            continue
        grouped[budget].append(row)

    result: dict[str, Any] = {}
    for budget in sorted(grouped):
        selected = grouped[budget]
        classes = [
            str((row.get("classification") or {}).get("result_class") or "UNKNOWN")
            for row in selected
        ]
        valid = [
            row
            for row in selected
            if (row.get("classification") or {}).get("valid_for_capability") is True
        ]
        correct = sum(
            1
            for row in valid
            if (row.get("classification") or {}).get("result_class") == "ANSWER_CORRECT"
        )
        result[str(budget)] = {
            "attempts": len(selected),
            "valid": len(valid),
            "invalid": len(selected) - len(valid),
            "correct": correct,
            "wrong_or_behavioral_failure": len(valid) - correct,
            "truncations": sum(value in TRUNCATION for value in classes),
            "result_classes": dict(sorted(Counter(classes).items())),
            "median_wall_clock_s": _median_latency(selected),
            "generated_tokens_observed": _token_total(selected),
            "experiment_ids": [
                (row.get("experiment") or {}).get("experiment_id") for row in selected
            ],
        }
    return result


def _reasoning_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        spec = row.get("experiment") if isinstance(row.get("experiment"), dict) else {}
        effort = spec.get("reasoning_effort")
        if effort is None:
            label = "on" if spec.get("thinking_mode") is True else "off"
        else:
            label = str(effort)
        grouped[label].append(row)

    answer: dict[str, Any] = {}
    for label in sorted(grouped):
        selected = grouped[label]
        valid = [
            row
            for row in selected
            if (row.get("classification") or {}).get("valid_for_capability") is True
        ]
        correct = sum(
            1
            for row in valid
            if (row.get("classification") or {}).get("result_class") == "ANSWER_CORRECT"
        )
        answer[label] = {
            "attempts": len(selected),
            "valid": len(valid),
            "correct": correct,
            "score_percent": _score_percent(correct, len(valid)),
            "median_wall_clock_s": _median_latency(selected),
        }
    return answer


def _minimum_budget(rows: list[dict[str, Any]], *, require_correct: bool) -> int | None:
    candidates: list[int] = []
    for row in rows:
        spec = row.get("experiment") if isinstance(row.get("experiment"), dict) else {}
        classification = (
            row.get("classification") if isinstance(row.get("classification"), dict) else {}
        )
        budget = spec.get("generation_budget")
        if isinstance(budget, bool) or not isinstance(budget, int):
            continue
        result_class = classification.get("result_class")
        if result_class in TRUNCATION:
            continue
        if require_correct and result_class != "ANSWER_CORRECT":
            continue
        candidates.append(budget)
    return None if not candidates else min(candidates)


def _family_scorecard(
    family_id: str,
    rows: list[dict[str, Any]],
    frontier: dict[str, Any],
) -> dict[str, Any]:
    valid_rows = [
        row
        for row in rows
        if (row.get("classification") or {}).get("valid_for_capability") is True
    ]
    correct = sum(
        1
        for row in valid_rows
        if (row.get("classification") or {}).get("result_class") == "ANSWER_CORRECT"
    )
    classes = Counter(
        str((row.get("classification") or {}).get("result_class") or "UNKNOWN")
        for row in rows
    )

    levels: dict[str, Any] = {}
    for level in frontier.get("levels") or []:
        if not isinstance(level, dict):
            continue
        valid_count = int(level.get("valid_count") or 0)
        pass_count = int(level.get("pass_count") or 0)
        level_num = level.get("level")
        levels[str(level_num)] = {
            "score_percent": _score_percent(pass_count, valid_count),
            "pass_rate": level.get("pass_rate"),
            "pass_count": pass_count,
            "fail_count": int(level.get("fail_count") or 0),
            "valid_count": valid_count,
            "invalid_count": int(level.get("invalid_count") or 0),
            "label": level.get("label"),
        }

    return {
        "family_id": family_id,
        "score_percent": _score_percent(correct, len(valid_rows)),
        "valid_observations": len(valid_rows),
        "invalid_observations": len(rows) - len(valid_rows),
        "correct": correct,
        "wrong": len(valid_rows) - correct,
        "result_classes": dict(sorted(classes.items())),
        "levels": levels,
        "tested_levels": list((frontier.get("coverage") or {}).get("tested_levels") or []),
        "untested_levels": list((frontier.get("coverage") or {}).get("untested_levels") or []),
        "reliable_floor": frontier.get("reliable_floor"),
        "unstable_levels": list(frontier.get("unstable_levels") or []),
        "first_failure_level": frontier.get("first_failure_level"),
        "transition_bracket": frontier.get("transition_bracket"),
        "token_budget_curve": _budget_curve(rows),
        "minimum_observed_completion_budget": _minimum_budget(rows, require_correct=False),
        "minimum_observed_correct_budget": _minimum_budget(rows, require_correct=True),
        "reasoning_modes": _reasoning_breakdown(rows),
        "median_wall_clock_s": _median_latency(rows),
        "generated_tokens_observed": _token_total(rows),
        "experiment_ids": [
            (row.get("experiment") or {}).get("experiment_id") for row in rows
        ],
    }


def build_compact_scorecard(
    model: str,
    rows: Iterable[dict[str, Any]],
    frontiers: dict[str, Any],
) -> dict[str, Any]:
    """Build direct numeric capability answers from the existing run evidence."""
    materialized = list(rows)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        spec = row.get("experiment") if isinstance(row.get("experiment"), dict) else {}
        family_id = spec.get("task_family") or spec.get("task_id")
        if family_id is not None:
            by_family[str(family_id)].append(row)

    frontier_families = frontiers.get("families") or {}
    family_ids = sorted(set(by_family) | set(map(str, frontier_families)))
    families = {
        family_id: _family_scorecard(
            family_id,
            by_family.get(family_id, []),
            frontier_families.get(family_id)
            if isinstance(frontier_families.get(family_id), dict)
            else {},
        )
        for family_id in family_ids
    }

    valid = [
        row
        for row in materialized
        if (row.get("classification") or {}).get("valid_for_capability") is True
    ]
    correct = sum(
        1
        for row in valid
        if (row.get("classification") or {}).get("result_class") == "ANSWER_CORRECT"
    )
    invalid = len(materialized) - len(valid)
    classes = Counter(
        str((row.get("classification") or {}).get("result_class") or "UNKNOWN")
        for row in materialized
    )

    return {
        "schema_version": 1,
        "model": model,
        "taxonomy_version": frontiers.get("taxonomy_version"),
        "score_policy": {
            "family_score": "ANSWER_CORRECT / valid capability observations",
            "level_score": "ANSWER_CORRECT / valid capability observations at that tested level",
            "adaptive_warning": "Scores summarize controller-selected probes; untested L0-L10 levels remain explicit and are never imputed.",
            "invalid_policy": "Invalid harness/runtime/truncation observations are reported but excluded from semantic score denominators.",
        },
        "overall": {
            "score_percent": _score_percent(correct, len(valid)),
            "attempts": len(materialized),
            "valid_observations": len(valid),
            "invalid_observations": invalid,
            "correct": correct,
            "wrong": len(valid) - correct,
            "result_classes": dict(sorted(classes.items())),
            "median_wall_clock_s": _median_latency(materialized),
            "generated_tokens_observed": _token_total(materialized),
        },
        "families": families,
    }


def render_compact_scorecard(scorecard: dict[str, Any]) -> str:
    """Render the numeric scorecard for immediate human inspection."""
    overall = scorecard.get("overall") or {}
    lines = [
        f"# Capability Scorecard: {scorecard.get('model')}",
        "",
        f"Overall valid-observation score: {overall.get('score_percent')}% ",
        f"({overall.get('correct', 0)}/{overall.get('valid_observations', 0)} correct; "
        f"{overall.get('invalid_observations', 0)} invalid reported separately)",
        "",
        "| Family | Score | Correct/Valid | Invalid | Tested levels | Reliable floor | First failure | Min correct budget | Median s |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for family_id, family in sorted((scorecard.get("families") or {}).items()):
        tested = ",".join(f"L{x}" for x in family.get("tested_levels") or []) or "none"
        lines.append(
            f"| {family_id} | {family.get('score_percent')}% | "
            f"{family.get('correct', 0)}/{family.get('valid_observations', 0)} | "
            f"{family.get('invalid_observations', 0)} | {tested} | "
            f"{family.get('reliable_floor')} | {family.get('first_failure_level')} | "
            f"{family.get('minimum_observed_correct_budget')} | "
            f"{family.get('median_wall_clock_s')} |"
        )

    lines.extend(
        [
            "",
            "## Detailed family evidence",
            "",
            "Each family entry in `scorecard.json` includes per-tested-level scores, result-class counts, token-budget curve, reasoning-mode breakdown, frontier data, runtime cost, and source experiment IDs. Untested levels are explicit; no score is fabricated for them.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
