import importlib
import importlib.util

import pytest


def _module():
    spec = importlib.util.find_spec("compute_cost.capability_ladders")
    assert spec is not None, "compute_cost.capability_ladders is not implemented"
    return importlib.import_module("compute_cost.capability_ladders")


def case(family: str, level: int, case_id: str) -> dict:
    return {
        "id": case_id,
        "family_id": family,
        "category": family,
        "difficulty_level": level,
        "difficulty": {
            "level": level,
            "rubric_version": f"{family}-v1",
            "dimensions": {"load": level},
        },
        "prompt": f"{family} level {level}",
        "scorer": "exact",
        "expected": "OK",
        "timeout_s": 120,
        "capabilities_required": [family],
        "recovery_eligible": True,
        "robustness_eligible": True,
        "compound": False,
        "tags": [],
    }


def test_build_ladder_index_groups_declared_cases_by_family_and_level():
    module = _module()
    rows = [
        case("math", 1, "math-L1"),
        case("logic", 2, "logic-L2"),
        case("math", 4, "math-L4"),
    ]
    index = module.build_ladder_index(rows)
    assert list(index) == ["logic", "math"]
    assert list(index["math"]) == [1, 4]
    assert index["math"][4]["id"] == "math-L4"


def test_build_ladder_index_rejects_duplicate_family_level_fixture():
    module = _module()
    with pytest.raises(ValueError, match="duplicate fixture for family math level 2"):
        module.build_ladder_index([
            case("math", 2, "math-L2-a"),
            case("math", 2, "math-L2-b"),
        ])


def test_resolve_requested_level_uses_only_declared_unattempted_levels():
    module = _module()
    ladder = {
        1: case("math", 1, "math-L1"),
        4: case("math", 4, "math-L4"),
        7: case("math", 7, "math-L7"),
    }
    assert module.resolve_requested_level(ladder, 4, set()) == 4
    assert module.resolve_requested_level(ladder, 5, {4}) == 7
    assert module.resolve_requested_level(ladder, 2, {1}) == 4
    assert module.resolve_requested_level(ladder, 5, {1, 4, 7}) is None


def test_resolve_requested_level_breaks_equal_distance_ties_toward_lower_level():
    module = _module()
    ladder = {
        2: case("math", 2, "math-L2"),
        4: case("math", 4, "math-L4"),
    }
    assert module.resolve_requested_level(ladder, 3, set()) == 2
