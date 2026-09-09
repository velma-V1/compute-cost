def _frontier(family_id, floor, failure, *, repeats=3):
    levels = []
    if floor is not None:
        levels.append({
            "level": floor,
            "observation_count": repeats,
            "valid_count": repeats,
            "invalid_count": 0,
            "pass_count": repeats,
            "fail_count": 0,
            "pass_rate": 1.0,
            "label": "reliable",
        })
    if failure is not None:
        levels.append({
            "level": failure,
            "observation_count": repeats,
            "valid_count": repeats,
            "invalid_count": 0,
            "pass_count": 0,
            "fail_count": repeats,
            "pass_rate": 0.0,
            "label": "failure",
        })
    return {
        "family_id": family_id,
        "levels": levels,
        "reliable_floor": floor,
        "first_failure_level": failure,
        "transition_bracket": (
            None if floor is None or failure is None else {
                "lower_level": floor,
                "lower_label": "reliable",
                "upper_level": failure,
                "upper_label": "failure",
            }
        ),
        "coverage": {"tested_levels": [row["level"] for row in levels], "tested_count": len(levels)},
    }


def test_capability_profile_separates_proven_boundaries_gaps_and_intervention_effects():
    from compute_cost.capability_profile import build_capability_profile

    frontiers = {
        "taxonomy_version": "capability-taxonomy-v1",
        "families": {
            "math": _frontier("math", 4, 5),
            "logic": _frontier("logic", 3, 4),
            "unknown": _frontier("unknown", None, None),
        },
    }
    coverage = {
        "families": {
            "math": {"state": "PROVEN"},
            "logic": {"state": "PROVEN"},
            "unknown": {"state": "UNTESTED"},
        }
    }
    value = {
        "families": {
            "math": {
                "cheapest_proven_raw_config": {"reasoning_effort": "low", "median_wall_clock_s": 1.0},
                "high_effort_extension_to": 5,
                "minimum_proven_recovery": {"difficulty_level": 6, "level": "R3"},
                "robustness": "ROBUST",
                "compound_risks": [{"compound_id": "math_logic", "composition_penalty": -1}],
            },
            "logic": {
                "cheapest_proven_raw_config": {"reasoning_effort": "medium", "median_wall_clock_s": 2.0},
                "high_effort_extension_to": None,
                "minimum_proven_recovery": None,
                "robustness": "FRAGILE",
                "compound_risks": [{"compound_id": "math_logic", "composition_penalty": -1}],
            },
            "unknown": {
                "cheapest_proven_raw_config": {"reasoning_effort": "medium", "median_wall_clock_s": None},
                "high_effort_extension_to": None,
                "minimum_proven_recovery": None,
                "robustness": None,
                "compound_risks": [],
            },
        }
    }
    atlas = {
        "failures": [
            {
                "family_id": "math",
                "difficulty_level": 5,
                "failure_origin": "MODEL_FAILURE",
                "failure_signature": {
                    "subtype": "arithmetic_error",
                    "inference_kind": "FAMILY_LOCAL_SIGNATURE",
                    "causal_claim": False,
                },
                "valid_for_capability": True,
            },
            {
                "family_id": "logic",
                "difficulty_level": 4,
                "failure_origin": "MODEL_FAILURE",
                "failure_signature": {
                    "subtype": "logic_error",
                    "inference_kind": "FAMILY_LOCAL_SIGNATURE",
                    "causal_claim": False,
                },
                "valid_for_capability": True,
            },
        ]
    }
    policy = {
        "families": {
            "math": {"coverage_state": "PROVEN"},
            "logic": {"coverage_state": "PROVEN"},
            "unknown": {"coverage_state": "UNTESTED"},
        }
    }

    profile = build_capability_profile(
        "gpt-oss:20b",
        frontiers,
        coverage,
        value,
        atlas,
        policy,
    )

    assert profile["schema_version"] == 1
    assert profile["model"] == "gpt-oss:20b"
    assert profile["measurement_policy"]["scalar_capability_score"] is False
    assert profile["summary"] == {
        "families_total": 3,
        "proven": 2,
        "partial": 0,
        "uncertain": 0,
        "untested": 1,
    }

    math = profile["families"]["math"]
    assert math["evidence_state"] == "PROVEN"
    assert math["raw_reliable_through"] == 4
    assert math["first_raw_failure"] == 5
    assert math["cheapest_proven_reasoning_effort"] == "low"
    assert math["high_effort_extension_to"] == 5
    assert math["minimum_proven_recovery"] == {"difficulty_level": 6, "level": "R3"}
    assert math["observed_failure_signatures"] == ["arithmetic_error"]
    assert math["signals"] == [
        "RAW_FRONTIER_BOUNDARY",
        "EFFORT_SENSITIVE",
        "RECOVERY_EXTENDS_CAPABILITY",
        "COMPOSITION_SENSITIVE",
    ]

    logic = profile["families"]["logic"]
    assert logic["signals"] == [
        "RAW_FRONTIER_BOUNDARY",
        "ROBUSTNESS_RISK",
        "COMPOSITION_SENSITIVE",
    ]
    assert logic["observed_failure_signatures"] == ["logic_error"]

    unknown = profile["families"]["unknown"]
    assert unknown["evidence_state"] == "UNTESTED"
    assert unknown["signals"] == ["EVIDENCE_GAP"]
    assert unknown["raw_reliable_through"] is None
    assert unknown["observed_failure_signatures"] == []


def test_capability_profile_does_not_turn_non_model_failures_into_model_weaknesses():
    from compute_cost.capability_profile import build_capability_profile

    profile = build_capability_profile(
        "gpt-oss:20b",
        {"families": {"math": _frontier("math", 4, 5)}},
        {"families": {"math": {"state": "PROVEN"}}},
        {"families": {"math": {"cheapest_proven_raw_config": {"reasoning_effort": "medium"}}}},
        {
            "failures": [
                {
                    "family_id": "math",
                    "difficulty_level": 5,
                    "failure_origin": "INFRA_FAILURE",
                    "failure_signature": {"subtype": "unresolved", "causal_claim": False},
                    "valid_for_capability": False,
                }
            ]
        },
        {"families": {"math": {"coverage_state": "PROVEN"}}},
    )

    math = profile["families"]["math"]
    assert math["observed_failure_signatures"] == []
    assert math["invalid_or_non_model_failure_count"] == 1
    assert "MODEL_FAILURE_OBSERVED" not in math["signals"]
