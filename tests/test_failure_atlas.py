def row(result_class, *, family="formal_logic_deduction", experiment_id="exp-1", valid=True):
    return {
        "experiment": {
            "experiment_id": experiment_id,
            "task_id": family,
            "task_family": family,
            "difficulty_level": 6,
            "reasoning_effort": "medium",
            "recovery_level": None,
        },
        "classification": {
            "result_class": result_class,
            "valid_for_capability": valid,
        },
        "score": 0.0,
        "status": "FAIL",
        "evidence_key": experiment_id,
        "evidence_refs": {"response": f"raw/{experiment_id}.json"},
    }


def test_failure_origin_separates_model_evaluator_infra_timeout_and_invalid_fixture():
    from compute_cost.failure_atlas import classify_failure_origin

    assert classify_failure_origin("ANSWER_WRONG") == "MODEL_FAILURE"
    assert classify_failure_origin("TOOL_FAILURE") == "MODEL_FAILURE"
    assert classify_failure_origin("SCORER_DEFECT") == "EVALUATOR_FAILURE"
    assert classify_failure_origin("RUNTIME_FAILURE") == "INFRA_FAILURE"
    assert classify_failure_origin("RESOURCE_LIMIT") == "INFRA_FAILURE"
    assert classify_failure_origin("CAPTURE_GAP") == "INFRA_FAILURE"
    assert classify_failure_origin("TIMEOUT") == "TIMEOUT"
    assert classify_failure_origin("TEST_DEFECT") == "INVALID_FIXTURE"
    assert classify_failure_origin("ANSWER_CORRECT") is None


def test_failure_subtype_uses_family_signature_without_overclaiming_causality():
    from compute_cost.failure_atlas import infer_failure_signature

    arithmetic = infer_failure_signature(
        "arithmetic_numerical_reasoning",
        "ANSWER_WRONG",
    )
    assert arithmetic == {
        "subtype": "arithmetic_error",
        "inference_kind": "FAMILY_LOCAL_SIGNATURE",
        "causal_claim": False,
    }

    stale = infer_failure_signature(
        "updated_obsolete_state_rejection",
        "ANSWER_WRONG",
    )
    assert stale["subtype"] == "stale_state_use"
    assert stale["causal_claim"] is False

    tool = infer_failure_signature("tool_argument_correctness", "TOOL_FAILURE")
    assert tool["subtype"] == "tool_argument_error"
    assert tool["causal_claim"] is False


def test_failure_subtype_stays_unresolved_when_evidence_does_not_support_a_specific_cause():
    from compute_cost.failure_atlas import infer_failure_signature

    signature = infer_failure_signature("spatial_reasoning", "ANSWER_WRONG")
    assert signature == {
        "subtype": "unresolved",
        "inference_kind": "INSUFFICIENT_EVIDENCE",
        "causal_claim": False,
    }


def test_failure_atlas_preserves_lineage_and_keeps_invalid_experiments_out_of_model_failure_counts():
    from compute_cost.failure_atlas import build_failure_atlas

    rows = [
        row("ANSWER_WRONG", family="formal_logic_deduction", experiment_id="exp-model"),
        row("SCORER_DEFECT", family="formal_logic_deduction", experiment_id="exp-scorer", valid=False),
        row("RUNTIME_FAILURE", family="formal_logic_deduction", experiment_id="exp-runtime", valid=False),
        row("ANSWER_CORRECT", family="formal_logic_deduction", experiment_id="exp-pass"),
    ]

    atlas = build_failure_atlas("gpt-oss:20b", rows)

    assert atlas["schema_version"] == 1
    assert atlas["model"] == "gpt-oss:20b"
    assert atlas["taxonomy"]["answer_wrong_subtypes"] == [
        "knowledge_gap",
        "logic_error",
        "arithmetic_error",
        "assumption_error",
        "premature_commit",
        "instruction_loss",
        "distractor_capture",
        "state_loss",
        "stale_state_use",
        "hallucinated_fact",
        "tool_selection_error",
        "tool_argument_error",
        "verification_failure",
        "cascading_error",
        "unresolved",
    ]
    assert [item["experiment_id"] for item in atlas["failures"]] == [
        "exp-model", "exp-scorer", "exp-runtime"
    ]
    assert atlas["failures"][0]["failure_origin"] == "MODEL_FAILURE"
    assert atlas["failures"][0]["failure_signature"]["subtype"] == "logic_error"
    assert atlas["failures"][0]["evidence_refs"] == {"response": "raw/exp-model.json"}
    assert atlas["summary"]["by_origin"] == {
        "EVALUATOR_FAILURE": 1,
        "INFRA_FAILURE": 1,
        "MODEL_FAILURE": 1,
    }
    assert atlas["summary"]["model_failures"] == 1
    assert atlas["summary"]["invalid_experiments"] == 2
