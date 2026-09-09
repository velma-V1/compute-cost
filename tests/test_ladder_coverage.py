import copy
import json
from pathlib import Path

from compute_cost import capability_suite


ALL_LEVELS = list(range(11))


def load_json(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_committed_suite_reports_declared_and_missing_levels_for_all_40_families():
    taxonomy = load_json("benchmarks/capability-taxonomy-v1.json")
    suite = load_json("benchmarks/gpt-oss-20b-capability-v1.json")

    assert hasattr(capability_suite, "build_ladder_coverage")
    coverage = capability_suite.build_ladder_coverage(suite, taxonomy)

    family_ids = {family["id"] for family in taxonomy["families"]}
    assert coverage["schema_version"] == 1
    assert coverage["taxonomy_version"] == taxonomy["taxonomy_version"]
    assert set(coverage["families"]) == family_ids
    assert coverage["family_count"] == 40
    assert coverage["complete_family_count"] == 0
    assert coverage["complete"] is False

    for family_id, row in coverage["families"].items():
        assert row["family_id"] == family_id
        assert row["declared_levels"] == sorted(row["declared_levels"])
        assert row["missing_levels"] == [level for level in ALL_LEVELS if level not in row["declared_levels"]]
        assert row["declared_count"] == len(row["declared_levels"])
        assert row["missing_count"] == len(row["missing_levels"])
        assert row["complete"] is False
        assert row["declared_count"] >= 1


def test_ladder_coverage_is_derived_without_mutating_suite_or_requiring_complete_ladders():
    taxonomy = load_json("benchmarks/capability-taxonomy-v1.json")
    suite = load_json("benchmarks/gpt-oss-20b-capability-v1.json")
    original = copy.deepcopy(suite)

    coverage = capability_suite.build_ladder_coverage(suite, taxonomy)

    assert suite == original
    assert any(row["missing_count"] > 0 for row in coverage["families"].values())
    assert capability_suite.validate_capability_suite(suite, taxonomy) is None


def test_ladder_coverage_rejects_duplicate_family_level_even_when_case_ids_differ():
    taxonomy = load_json("benchmarks/capability-taxonomy-v1.json")
    suite = load_json("benchmarks/gpt-oss-20b-capability-v1.json")
    broken = copy.deepcopy(suite)
    duplicate = copy.deepcopy(broken["cases"][0])
    duplicate["id"] = duplicate["id"] + "-duplicate"
    broken["cases"].append(duplicate)

    try:
        capability_suite.build_ladder_coverage(broken, taxonomy)
    except ValueError as exc:
        assert "duplicate fixture" in str(exc)
    else:
        raise AssertionError("duplicate family/level fixture was accepted")


def test_complete_synthetic_ladder_reports_no_missing_levels():
    taxonomy = load_json("benchmarks/capability-taxonomy-v1.json")
    suite = load_json("benchmarks/gpt-oss-20b-capability-v1.json")
    seed = copy.deepcopy(suite["cases"][0])
    family_id = seed["capability_map"]["family_id"]

    synthetic = copy.deepcopy(suite)
    synthetic["cases"] = []
    synthetic["coverage"] = {
        family["id"]: "UNTESTED" for family in taxonomy["families"]
    }
    for family in taxonomy["families"]:
        for level in ALL_LEVELS:
            case = copy.deepcopy(seed)
            case["id"] = f"synthetic-{family['id']}-L{level}"
            case["category"] = family["id"]
            case["difficulty_level"] = level
            meta = case["capability_map"]
            meta["family_id"] = family["id"]
            meta["rubric_version"] = family["rubric_version"]
            meta["difficulty"]["level"] = level
            meta["difficulty"]["dimensions"] = {
                dimension: level for dimension in family["difficulty_dimensions"]
            }
            meta["capabilities_required"] = [family["id"]]
            synthetic["cases"].append(case)

    coverage = capability_suite.build_ladder_coverage(synthetic, taxonomy)
    assert coverage["complete"] is True
    assert coverage["complete_family_count"] == 40
    assert all(row["declared_levels"] == ALL_LEVELS for row in coverage["families"].values())
    assert all(row["missing_levels"] == [] for row in coverage["families"].values())
