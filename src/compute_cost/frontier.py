"""Deterministic derivation of probabilistic capability frontiers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _validate_thresholds(reliable_threshold: float, unstable_threshold: float) -> None:
    if not 0.0 <= unstable_threshold <= reliable_threshold <= 1.0:
        raise ValueError(
            "thresholds must satisfy 0 <= unstable_threshold <= reliable_threshold <= 1"
        )


def classify_pass_rate(
    pass_rate: float,
    *,
    reliable_threshold: float = 0.90,
    unstable_threshold: float = 0.40,
) -> str:
    """Classify a valid-observation pass rate into the v1 reliability bands."""
    _validate_thresholds(reliable_threshold, unstable_threshold)
    if not 0.0 <= pass_rate <= 1.0:
        raise ValueError("pass_rate must be between 0 and 1")
    if pass_rate >= reliable_threshold:
        return "reliable"
    if pass_rate >= unstable_threshold:
        return "unstable"
    return "failure"


def build_family_frontier(
    family_id: str,
    observations: list[dict[str, Any]],
    *,
    reliable_threshold: float = 0.90,
    unstable_threshold: float = 0.40,
) -> dict[str, Any]:
    """Aggregate raw replicated observations into one family-local frontier."""
    _validate_thresholds(reliable_threshold, unstable_threshold)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        level = row.get("level")
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 10:
            raise ValueError("observation level must be integer 0..10")
        grouped[level].append(row)

    levels: list[dict[str, Any]] = []
    for level in sorted(grouped):
        rows = grouped[level]
        valid = [
            row
            for row in rows
            if row.get("valid_for_capability") is True
            and isinstance(row.get("passed"), bool)
        ]
        pass_count = sum(1 for row in valid if row["passed"] is True)
        fail_count = sum(1 for row in valid if row["passed"] is False)
        valid_count = len(valid)
        invalid_count = len(rows) - valid_count
        if valid_count:
            pass_rate: float | None = pass_count / valid_count
            label = classify_pass_rate(
                pass_rate,
                reliable_threshold=reliable_threshold,
                unstable_threshold=unstable_threshold,
            )
        else:
            pass_rate = None
            label = "unresolved"
        levels.append(
            {
                "level": level,
                "observation_count": len(rows),
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "pass_rate": pass_rate,
                "label": label,
            }
        )

    reliable_floor: int | None = None
    contradicted = False
    for row in levels:
        label = row["label"]
        if label == "unresolved":
            continue
        if label == "reliable" and not contradicted:
            reliable_floor = int(row["level"])
            continue
        if label != "reliable":
            contradicted = True

    unstable_levels = [int(row["level"]) for row in levels if row["label"] == "unstable"]
    failure_levels = [int(row["level"]) for row in levels if row["label"] == "failure"]

    if reliable_floor is None:
        first_failure_level = failure_levels[0] if failure_levels else None
    else:
        later_failures = [level for level in failure_levels if level > reliable_floor]
        first_failure_level = later_failures[0] if later_failures else None

    transition_bracket: dict[str, Any] | None = None
    if reliable_floor is not None:
        upper = next(
            (
                row
                for row in levels
                if int(row["level"]) > reliable_floor
                and row["label"] in {"unstable", "failure"}
            ),
            None,
        )
        if upper is not None:
            transition_bracket = {
                "lower_level": reliable_floor,
                "lower_label": "reliable",
                "upper_level": int(upper["level"]),
                "upper_label": str(upper["label"]),
            }

    tested_levels = sorted(grouped)
    return {
        "family_id": family_id,
        "thresholds": {
            "reliable": reliable_threshold,
            "unstable": unstable_threshold,
        },
        "levels": levels,
        "reliable_floor": reliable_floor,
        "unstable_levels": unstable_levels,
        "failure_levels": failure_levels,
        "first_failure_level": first_failure_level,
        "transition_bracket": transition_bracket,
        "coverage": {
            "tested_levels": tested_levels,
            "untested_levels": [level for level in range(11) if level not in grouped],
            "tested_count": len(tested_levels),
        },
    }


def build_capability_frontiers(
    taxonomy_version: str,
    family_observations: dict[str, list[dict[str, Any]]],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build the versioned aggregate capability-frontiers artifact."""
    resolved = thresholds or {"reliable": 0.90, "unstable": 0.40}
    reliable_threshold = float(resolved["reliable"])
    unstable_threshold = float(resolved["unstable"])
    _validate_thresholds(reliable_threshold, unstable_threshold)

    families = {
        family_id: build_family_frontier(
            family_id,
            family_observations[family_id],
            reliable_threshold=reliable_threshold,
            unstable_threshold=unstable_threshold,
        )
        for family_id in sorted(family_observations)
    }
    return {
        "schema_version": 1,
        "taxonomy_version": taxonomy_version,
        "thresholds": {
            "reliable": reliable_threshold,
            "unstable": unstable_threshold,
        },
        "families": families,
    }
