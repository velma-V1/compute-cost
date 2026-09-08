"""Validation for versioned capability-map benchmark suites."""

from __future__ import annotations

from typing import Any


ALLOWED_COVERAGE = {"PROVEN", "PARTIAL", "UNCERTAIN", "UNTESTED"}
ALLOWED_SCORERS = {
    "exact",
    "numeric",
    "json",
    "extraction_set",
    "tool_call",
    "ambiguity",
    "context_retrieval",
    "python_function",
}
LEGACY_FIELDS = (
    "id",
    "category",
    "difficulty_level",
    "prompt",
    "scorer",
    "timeout_s",
)


def validate_capability_suite(
    suite: dict[str, Any],
    taxonomy: dict[str, Any],
) -> None:
    """Raise ``ValueError`` when a capability suite violates the v1 contract."""
    taxonomy_version = taxonomy.get("taxonomy_version")
    if taxonomy_version != "capability-taxonomy-v1":
        raise ValueError("unsupported taxonomy version")

    families = taxonomy.get("families")
    if not isinstance(families, list) or len(families) != 40:
        raise ValueError("taxonomy must contain exactly 40 families")

    family_ids: list[str] = []
    rubric_by_family: dict[str, str] = {}
    for family in families:
        if not isinstance(family, dict):
            raise ValueError("taxonomy family must be an object")
        family_id = family.get("id")
        rubric_version = family.get("rubric_version")
        dimensions = family.get("difficulty_dimensions")
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("taxonomy family id must be a non-empty string")
        if not isinstance(rubric_version, str) or not rubric_version:
            raise ValueError(f"family {family_id} missing rubric version")
        if not isinstance(dimensions, list) or not dimensions:
            raise ValueError(f"family {family_id} missing difficulty dimensions")
        if not all(isinstance(item, str) and item for item in dimensions):
            raise ValueError(f"family {family_id} has invalid difficulty dimensions")
        family_ids.append(family_id)
        rubric_by_family[family_id] = rubric_version

    if len(set(family_ids)) != 40:
        raise ValueError("taxonomy family ids must be unique")
    known_families = set(family_ids)

    if suite.get("schema_version") != "capability-suite-v1":
        raise ValueError("unsupported capability suite schema version")
    if suite.get("taxonomy_version") != taxonomy_version:
        raise ValueError("suite taxonomy version mismatch")

    coverage = suite.get("coverage")
    if not isinstance(coverage, dict) or set(coverage) != known_families:
        raise ValueError("coverage ledger must exactly match taxonomy families")
    if any(state not in ALLOWED_COVERAGE for state in coverage.values()):
        raise ValueError("invalid coverage state")

    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("suite cases must be a non-empty list")

    seen_case_ids: set[str] = set()
    covered_families: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case must be an object")
        for field in LEGACY_FIELDS:
            if field not in case:
                raise ValueError(f"case missing legacy field: {field}")

        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case id must be a non-empty string")
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate case id: {case_id}")
        seen_case_ids.add(case_id)

        if not isinstance(case["category"], str) or not case["category"]:
            raise ValueError(f"case {case_id} category must be a non-empty string")
        if not isinstance(case["prompt"], str) or not case["prompt"]:
            raise ValueError(f"case {case_id} prompt must be a non-empty string")
        if case["scorer"] not in ALLOWED_SCORERS:
            raise ValueError(f"case {case_id} uses unknown scorer: {case['scorer']}")

        level = case["difficulty_level"]
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 10:
            raise ValueError(f"case {case_id} difficulty_level must be integer 0..10")
        timeout = case["timeout_s"]
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"case {case_id} timeout_s must be positive")

        meta = case.get("capability_map")
        if not isinstance(meta, dict):
            raise ValueError(f"case {case_id} missing capability_map")
        if meta.get("taxonomy_version") != taxonomy_version:
            raise ValueError(f"case {case_id} taxonomy version mismatch")

        family_id = meta.get("family_id")
        if family_id not in known_families:
            raise ValueError(f"case {case_id} references unknown family: {family_id}")
        covered_families.add(str(family_id))

        if meta.get("rubric_version") != rubric_by_family[family_id]:
            raise ValueError(f"case {case_id} rubric version mismatch")

        difficulty = meta.get("difficulty")
        if not isinstance(difficulty, dict) or difficulty.get("level") != level:
            raise ValueError(f"case {case_id} difficulty level mismatch")
        dimensions = difficulty.get("dimensions")
        if not isinstance(dimensions, dict) or not dimensions:
            raise ValueError(f"case {case_id} difficulty dimensions must be non-empty")

        required = meta.get("capabilities_required")
        if not isinstance(required, list) or not required:
            raise ValueError(f"case {case_id} capabilities_required must be non-empty")
        unknown_required = set(required) - known_families
        if unknown_required:
            raise ValueError(
                f"case {case_id} has unknown required capabilities: {sorted(unknown_required)}"
            )

        if not isinstance(meta.get("recovery_eligible"), bool):
            raise ValueError(f"case {case_id} recovery_eligible must be bool")
        if not isinstance(meta.get("robustness_eligible"), bool):
            raise ValueError(f"case {case_id} robustness_eligible must be bool")

    missing = known_families - covered_families
    if missing:
        raise ValueError(f"families missing anchor cases: {sorted(missing)}")
