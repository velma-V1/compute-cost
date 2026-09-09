def test_replay_targets_group_replicates_and_rank_decision_value():
    from compute_cost.replay_targets import build_replay_targets

    frontiers = {
        "families": {
            "math": {"first_failure_level": 5},
            "logic": {"first_failure_level": 4},
        }
    }
    atlas = {
        "failures": [
            {
                "experiment_id": "math-b1",
                "family_id": "math",
                "difficulty_level": 5,
                "result_class": "ANSWER_WRONG",
                "failure_origin": "MODEL_FAILURE",
                "valid_for_capability": True,
                "recovery_level": None,
                "failure_signature": {"subtype": "arithmetic_error"},
            },
            {
                "experiment_id": "math-b2",
                "family_id": "math",
                "difficulty_level": 5,
                "result_class": "ANSWER_WRONG",
                "failure_origin": "MODEL_FAILURE",
                "valid_for_capability": True,
                "recovery_level": None,
                "failure_signature": {"subtype": "arithmetic_error"},
            },
            {
                "experiment_id": "math-r1",
                "family_id": "math",
                "difficulty_level": 6,
                "result_class": "ANSWER_WRONG",
                "failure_origin": "MODEL_FAILURE",
                "valid_for_capability": True,
                "recovery_level": "R3",
                "failure_signature": {"subtype": "arithmetic_error"},
            },
            {
                "experiment_id": "logic-m1",
                "family_id": "logic",
                "difficulty_level": 6,
                "result_class": "ANSWER_WRONG",
                "failure_origin": "MODEL_FAILURE",
                "valid_for_capability": True,
                "recovery_level": None,
                "failure_signature": {"subtype": "logic_error"},
            },
            {
                "experiment_id": "math-a1",
                "family_id": "math",
                "difficulty_level": 7,
                "result_class": "RUNTIME_FAILURE",
                "failure_origin": "INFRA_FAILURE",
                "valid_for_capability": False,
                "recovery_level": None,
                "failure_signature": {"subtype": "unresolved"},
            },
        ]
    }
    replay_index = [
        {
            "replay_id": "math-b1",
            "category": "failures",
            "path": "replay/failures/math-b1.json",
            "family_id": "math",
            "difficulty_level": 5,
            "result_class": "ANSWER_WRONG",
            "valid_for_capability": True,
            "recovery_level": None,
        },
        {
            "replay_id": "math-b2",
            "category": "failures",
            "path": "replay/failures/math-b2.json",
            "family_id": "math",
            "difficulty_level": 5,
            "result_class": "ANSWER_WRONG",
            "valid_for_capability": True,
            "recovery_level": None,
        },
        {
            "replay_id": "math-r1",
            "category": "recoveries",
            "path": "replay/recoveries/math-r1.json",
            "family_id": "math",
            "difficulty_level": 6,
            "result_class": "ANSWER_WRONG",
            "valid_for_capability": True,
            "recovery_level": "R3",
        },
        {
            "replay_id": "logic-m1",
            "category": "failures",
            "path": "replay/failures/logic-m1.json",
            "family_id": "logic",
            "difficulty_level": 6,
            "result_class": "ANSWER_WRONG",
            "valid_for_capability": True,
            "recovery_level": None,
        },
        {
            "replay_id": "math-a1",
            "category": "anomalies",
            "path": "replay/anomalies/math-a1.json",
            "family_id": "math",
            "difficulty_level": 7,
            "result_class": "RUNTIME_FAILURE",
            "valid_for_capability": False,
            "recovery_level": None,
        },
        {
            "replay_id": "orphan",
            "category": "anomalies",
            "path": "replay/anomalies/orphan.json",
            "family_id": "math",
            "difficulty_level": 8,
            "result_class": "CAPTURE_GAP",
            "valid_for_capability": False,
            "recovery_level": None,
        },
    ]

    result = build_replay_targets("gpt-oss:20b", frontiers, atlas, replay_index)

    assert result["schema_version"] == 1
    assert result["model"] == "gpt-oss:20b"
    assert result["selection_policy"]["full_registry_is_source_of_truth"] is True
    assert result["selection_policy"]["executes_model_calls"] is False
    assert result["summary"] == {
        "registry_entries": 6,
        "matched_entries": 5,
        "selected_groups": 4,
        "raw_boundary_groups": 1,
        "recovery_limit_groups": 1,
        "other_model_failure_groups": 1,
        "evidence_integrity_groups": 1,
        "unmatched_registry_entries": 1,
    }

    boundary, recovery, other, anomaly = result["targets"]
    assert boundary == {
        "priority": 1,
        "reason": "RAW_BOUNDARY_RETEST",
        "family_id": "math",
        "difficulty_level": 5,
        "result_class": "ANSWER_WRONG",
        "recovery_level": None,
        "failure_origin": "MODEL_FAILURE",
        "failure_signature": "arithmetic_error",
        "primary_replay_id": "math-b1",
        "primary_path": "replay/failures/math-b1.json",
        "supporting_replay_ids": ["math-b2"],
        "evidence_count": 2,
    }
    assert recovery["priority"] == 2
    assert recovery["reason"] == "RECOVERY_LIMIT_RETEST"
    assert recovery["primary_replay_id"] == "math-r1"
    assert recovery["recovery_level"] == "R3"
    assert other["priority"] == 3
    assert other["reason"] == "MODEL_FAILURE_RETEST"
    assert other["primary_replay_id"] == "logic-m1"
    assert anomaly["priority"] == 4
    assert anomaly["reason"] == "EVIDENCE_INTEGRITY_REPRODUCTION"
    assert anomaly["failure_origin"] == "INFRA_FAILURE"
    assert anomaly["failure_signature"] is None
