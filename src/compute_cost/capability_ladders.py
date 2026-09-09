"""Index and resolve declared family-local capability fixtures."""

from __future__ import annotations

from typing import Any


def build_ladder_index(
    cases: list[dict[str, Any]],
) -> dict[str, dict[int, dict[str, Any]]]:
    """Group normalized capability cases by family and difficulty level."""
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for case in cases:
        family_id = str(case["family_id"])
        level = int(case["difficulty_level"])
        family = grouped.setdefault(family_id, {})
        if level in family:
            raise ValueError(f"duplicate fixture for family {family_id} level {level}")
        family[level] = case

    return {
        family_id: dict(sorted(levels.items()))
        for family_id, levels in sorted(grouped.items())
    }


def resolve_requested_level(
    ladder: dict[int, dict[str, Any]],
    requested_level: int,
    attempted_levels: set[int],
) -> int | None:
    """Resolve a controller request to the nearest declared unattempted level."""
    available = [level for level in ladder if level not in attempted_levels]
    if not available:
        return None
    if requested_level in ladder and requested_level not in attempted_levels:
        return requested_level
    return min(available, key=lambda level: (abs(level - requested_level), level))
