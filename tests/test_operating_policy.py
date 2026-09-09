def _level(level, label, *, valid_count=3, pass_count=None, fail_count=None):
    if pass_count is None:
        pass_count = valid_count if label == "reliable" else 0
    if fail_count is None:
        fail_count = valid_count - pass_count
    return {
        "level": level,
        "observation_count": valid_count,
        "valid_count": valid_count,
        "invalid_count": 0,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": None if valid_count == 0 else pass_count / valid_count,
        "label": label,
    }


def _frontier(family_id, floor, failure):
    levels = []
    if floor is not None:
        levels.append(_level(floor, "reliable"))
    if failure is not None:
        levels.append(_level(failure, "failure"))
    return {
        "family_id": family_id,
        "levels": levels,
        "reliable_floor": floor,
        "first_failure_level": failure,
        "transition_bracket": (
            None
            if floor is None or failure is None
            else {
                "lower_level": floor,
                "lower_label": "reliable",
                "upper_level": failure,
                "upper_label": "failure",
            }
        ),
        "coverage": {
            "tested_levels": [row["level"] for row in levels],
            "untested_levels": [],
            "tested_count": len(levels),
        },
    }


def test_operating_policy_routes_only_proven_evidence_and_guards_composition():
    from compute_cost.operating_policy import build_operating_policy, route_task

    frontiers = {
        "taxonomy_version": "capability-taxonomy-v1",
        "families": {
            "math": _frontier("math", 4, 5),
            "logic": _frontier("logic", 3, 4),
            "unknown": _frontier("unknown", None, None),
        },
    }
    value_map = {
        "families": {
            "math": {
                "cheapest_proven_raw_config": {
                    "reasoning_effort": "low",
                    "median_wall_clock_s": 1.0,
                },
                "high_effort_extension_to": 5,
                "minimum_proven_recovery": {
                    "difficulty_level": 6,
                    "level": "R3",
                },
                "robustness": "ROBUST",
            },
            "logic": {
                "cheapest_proven_raw_config": {
                    "reasoning_effort": "medium",
                    "median_wall_clock_s": 2.0,
                },
                "high_effort_extension_to": None,
                "minimum_proven_recovery": None,
                "robustness": "FRAGILE",
            },
            "unknown": {
                "cheapest_proven_raw_config": {
                    "reasoning_effort": "medium",
                    "median_wall_clock_s": None,
                },
                "high_effort_extension_to": None,
                "minimum_proven_recovery": None,
                "robustness": None,
            },
        }
    }
    compound_map = {
        "compounds": {
            "math_logic": {
                "capabilities_required": ["math", "logic"],
                "expected_component_frontier": 3,
                "observed_compound_frontier": 2,
                "composition_penalty": -1,
                "status": "MEASURED",
            }
        }
    }

    policy = build_operating_policy(
        "gpt-oss:20b",
        frontiers,
        value_map,
        compound_map=compound_map,
        boundary_repeats=3,
    )

    assert policy["schema_version"] == 1
    assert policy["model"] == "gpt-oss:20b"
    assert policy["decision_order"] == [
        "CHEAPEST_PROVEN_RAW",
        "PROVEN_HIGH_EFFORT_EXTENSION",
        "MINIMUM_PROVEN_RECOVERY",
        "ESCALATE",
    ]
    assert policy["families"]["math"]["coverage_state"] == "PROVEN"
    assert policy["families"]["unknown"]["coverage_state"] == "UNTESTED"
    assert policy["compounds"]["math_logic"]["observed_compound_frontier"] == 2

    raw = route_task(policy, ["math"], 4)
    assert raw["action"] == "RAW"
    assert raw["reasoning_effort"] == "low"
    assert raw["evidence_basis"] == "RELIABLE_FLOOR"

    high = route_task(policy, ["math"], 5)
    assert high["action"] == "RAW"
    assert high["reasoning_effort"] == "high"
    assert high["evidence_basis"] == "HIGH_EFFORT_EXTENSION"

    recovery = route_task(policy, ["math"], 6)
    assert recovery["action"] == "RECOVERY"
    assert recovery["recovery_level"] == "R3"
    assert recovery["evidence_basis"] == "PROVEN_RECOVERY"

    beyond = route_task(policy, ["math"], 7)
    assert beyond["action"] == "ESCALATE"
    assert beyond["reason"] == "NO_PROVEN_ROUTE"

    unknown = route_task(policy, ["unknown"], 1)
    assert unknown["action"] == "ESCALATE"
    assert unknown["reason"] == "CAPABILITY_NOT_PROVEN"

    unmeasured_composition = route_task(policy, ["math", "logic"], 2)
    assert unmeasured_composition["action"] == "ESCALATE"
    assert unmeasured_composition["reason"] == "COMPOSITION_UNMEASURED"

    compound_raw = route_task(policy, ["math", "logic"], 2, compound_id="math_logic")
    assert compound_raw["action"] == "RAW"
    assert compound_raw["reasoning_effort"] == "medium"
    assert compound_raw["evidence_basis"] == "MEASURED_COMPOUND_FRONTIER"
    assert compound_raw["composition_penalty"] == -1

    compound_beyond = route_task(policy, ["math", "logic"], 3, compound_id="math_logic")
    assert compound_beyond["action"] == "ESCALATE"
    assert compound_beyond["reason"] == "COMPOUND_FRONTIER_EXCEEDED"

    mismatch = route_task(policy, ["math"], 2, compound_id="math_logic")
    assert mismatch["action"] == "ESCALATE"
    assert mismatch["reason"] == "COMPOUND_CAPABILITY_MISMATCH"


def test_partial_boundary_never_becomes_a_raw_policy_route():
    from compute_cost.operating_policy import build_operating_policy, route_task

    frontier = _frontier("math", 4, 5)
    frontier["levels"][0]["valid_count"] = 1
    frontier["levels"][0]["observation_count"] = 1
    frontier["levels"][1]["valid_count"] = 1
    frontier["levels"][1]["observation_count"] = 1
    policy = build_operating_policy(
        "gpt-oss:20b",
        {"families": {"math": frontier}},
        {
            "families": {
                "math": {
                    "cheapest_proven_raw_config": {"reasoning_effort": "low"},
                    "high_effort_extension_to": 5,
                    "minimum_proven_recovery": {"difficulty_level": 6, "level": "R3"},
                }
            }
        },
        boundary_repeats=3,
    )

    assert policy["families"]["math"]["coverage_state"] == "PARTIAL"
    decision = route_task(policy, ["math"], 1)
    assert decision["action"] == "ESCALATE"
    assert decision["reason"] == "CAPABILITY_NOT_PROVEN"
