def test_raw_boundary_target_attaches_exact_lower_and_upper_boundary_replays():
    from compute_cost.replay_targets import build_replay_targets

    frontiers = {
        "families": {
            "math": {
                "first_failure_level": 5,
                "transition_bracket": {
                    "lower_level": 4,
                    "lower_label": "reliable",
                    "upper_level": 5,
                    "upper_label": "failure",
                },
            }
        }
    }
    atlas = {
        "failures": [
            {
                "experiment_id": "math-fail-a",
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
            "replay_id": "math-fail-a",
            "category": "failures",
            "path": "replay/failures/math-fail-a.json",
            "family_id": "math",
            "difficulty_level": 5,
            "result_class": "ANSWER_WRONG",
            "valid_for_capability": True,
            "recovery_level": None,
        },
        {
            "replay_id": "math-pass-a--boundary",
            "category": "boundaries",
            "path": "replay/boundaries/math-pass-a--boundary.json",
            "source_experiment_id": "math-pass-a",
            "family_id": "math",
            "difficulty_level": 4,
            "result_class": "ANSWER_CORRECT",
            "valid_for_capability": True,
            "boundary_role": "lower_reliable",
        },
        {
            "replay_id": "math-fail-a--boundary",
            "category": "boundaries",
            "path": "replay/boundaries/math-fail-a--boundary.json",
            "source_experiment_id": "math-fail-a",
            "family_id": "math",
            "difficulty_level": 5,
            "result_class": "ANSWER_WRONG",
            "valid_for_capability": True,
            "boundary_role": "upper_transition",
        },
    ]

    result = build_replay_targets("gpt-oss:20b", frontiers, atlas, replay_index)

    assert result["summary"]["registry_entries"] == 3
    assert result["summary"]["matched_entries"] == 3
    assert result["summary"]["unmatched_registry_entries"] == 0
    assert result["summary"]["boundary_entries_attached"] == 2

    target = result["targets"][0]
    assert target["reason"] == "RAW_BOUNDARY_RETEST"
    assert target["boundary_replays"] == {
        "lower_reliable": [
            {
                "replay_id": "math-pass-a--boundary",
                "path": "replay/boundaries/math-pass-a--boundary.json",
                "source_experiment_id": "math-pass-a",
                "difficulty_level": 4,
            }
        ],
        "upper_transition": [
            {
                "replay_id": "math-fail-a--boundary",
                "path": "replay/boundaries/math-fail-a--boundary.json",
                "source_experiment_id": "math-fail-a",
                "difficulty_level": 5,
            }
        ],
    }
