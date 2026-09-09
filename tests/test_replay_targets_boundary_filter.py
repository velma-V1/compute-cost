def test_boundary_registry_entries_do_not_count_as_unmatched_failure_targets():
    from compute_cost.replay_targets import build_replay_targets

    frontiers = {"families": {"math": {"first_failure_level": 5}}}
    failure_atlas = {
        "failures": [
            {
                "experiment_id": "math-L5-fail",
                "family_id": "math",
                "difficulty_level": 5,
                "result_class": "ANSWER_WRONG",
                "failure_origin": "MODEL_FAILURE",
                "valid_for_capability": True,
                "recovery_level": None,
                "failure_signature": {"subtype": "arithmetic_error"},
            }
        ]
    }
    replay_index = [
        {
            "replay_id": "math-L5-fail",
            "category": "failures",
            "path": "replay/failures/math-L5-fail.json",
            "family_id": "math",
            "difficulty_level": 5,
            "result_class": "ANSWER_WRONG",
            "valid_for_capability": True,
            "recovery_level": None,
        },
        {
            "replay_id": "math-L4-pass--boundary",
            "category": "boundaries",
            "path": "replay/boundaries/math-L4-pass--boundary.json",
            "source_experiment_id": "math-L4-pass",
            "family_id": "math",
            "difficulty_level": 4,
            "result_class": "ANSWER_CORRECT",
            "valid_for_capability": True,
            "recovery_level": None,
            "boundary_role": "lower_reliable",
        },
    ]

    result = build_replay_targets("gpt-oss:20b", frontiers, failure_atlas, replay_index)

    assert result["summary"]["registry_entries"] == 2
    assert result["summary"]["matched_entries"] == 1
    assert result["summary"]["unmatched_registry_entries"] == 0
    assert result["summary"]["selected_groups"] == 1
    assert result["targets"][0]["reason"] == "RAW_BOUNDARY_RETEST"
    assert result["targets"][0]["primary_replay_id"] == "math-L5-fail"
