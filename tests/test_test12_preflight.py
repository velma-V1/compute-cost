from compute_cost.test12_campaign import fresh_model_source
from compute_cost.test12_preflight import (
    APPLICABILITY_STATES,
    EFFECT_STATES,
    build_test12_cell_budget_plan,
)


def _cases():
    families = [
        "instruction_following_constraint_stacking",
        "strict_structured_output",
        "extraction_transformation",
        "arithmetic_numerical_reasoning",
        "algebra_quantitative_reasoning",
        "formal_logic_deduction",
        "causal_counterfactual_reasoning",
        "temporal_reasoning",
        "spatial_reasoning",
        "planning_optimization",
        "coding_generation",
        "code_comprehension",
        "debugging_root_cause_diagnosis",
        "refactoring_under_constraints",
        "test_generation_verification",
        "tool_selection",
        "tool_argument_correctness",
        "multi_tool_sequencing",
        "tool_error_recovery",
        "ambiguity_detection",
        "missing_information_handling",
        "uncertainty_calibration",
        "hallucination_resistance",
        "context_retrieval",
        "context_reasoning",
        "lost_in_middle_resistance",
        "distractor_noise_resistance",
        "contradictory_information_handling",
        "multi_turn_state_tracking",
        "updated_obsolete_state_rejection",
        "memory_compression_summary_fidelity",
        "decomposition",
        "self_correction",
        "verification_critique",
        "meta_reasoning",
        "prompt_instruction_conflict_handling",
        "format_robustness",
        "adversarial_wording_robustness",
        "sibling_transfer_generalization",
        "composite_agent_tasks",
    ]
    return [
        {
            "id": f"case-{index}",
            "category": family,
            "family_id": family,
            "difficulty_level": 0,
            "prompt": family,
        }
        for index, family in enumerate(families)
    ]


def test_cell_budget_plan_freezes_schema_and_forces_subset_arithmetic():
    source = fresh_model_source(_cases())
    report = {
        "analysis_type": "ZERO_MODEL_CALL_RUNTIME_FAMILY_CLASSIFIER_CONFUSION",
        "top1_accuracy": 0.8,
        "families": {
            case["family_id"]: {"accuracy": 0.8}
            for case in _cases()
        },
    }
    plan = build_test12_cell_budget_plan(source, report)

    assert plan["family_count"] == 40
    assert plan["semantic_mechanism_count"] >= 50
    assert plan["potential_cell_count"] == (
        plan["family_count"] * plan["semantic_mechanism_count"]
    )
    assert set(plan["applicability_counts"]) == set(APPLICABILITY_STATES)
    assert plan["applicability_counts"]["structural_no"] > 0
    assert plan["applicability_counts"]["untested"] == 0
    assert plan["applicability_counts"]["unbuilt"] == 0
    assert plan["untested_applicability_cell_count"] == 0
    assert plan["implementation_status_is_static_zero_call_audit"] is True
    assert plan["built_semantic_mechanism_count"] == plan["semantic_mechanism_count"]
    assert plan["unbuilt_semantic_mechanism_count"] == 0
    assert plan["untested_applicability_excluded_from_campaign_calls"] is True
    assert plan["schema_contract"]["effect_states"] == list(EFFECT_STATES)
    assert plan["schema_contract"]["null_verified_distinct_from_null_censored"] is True
    assert plan["schema_contract"]["conditional_requires_populated_condition_predicate"] is True
    assert plan["schema_contract"]["harm_population_required"] is True
    assert plan["schema_contract"]["cost_and_effect_must_share_observation_set"] is True
    assert plan["schema_contract"]["cost_and_effect_must_share_operating_point"] is True
    assert plan["scheduler_contract"]["proof_replication_owner"] == "TEST2"
    assert plan["preferred_selection_scenario"] == "decision_complete_estimate"

    scenario = plan["capacity_scenarios"]["decision_complete_estimate"]
    assert scenario["valid_effect_observations_per_cell"] == 3
    assert scenario["harm_sentinel_observations_per_cell"] == 1
    assert 0.0 < scenario["selected_cell_fraction"] < 1.0


def test_preflight_cells_start_unknown_and_cost_is_not_faked():
    plan = build_test12_cell_budget_plan(fresh_model_source(_cases()))
    measured = next(
        cell for cell in plan["cells"]
        if cell["applicability"] == "yes"
    )
    assert measured["effect"] == "unknown"
    assert measured["conditions"]["predicate"] is None
    assert measured["cost"]["status"] == "UNMEASURED"
    assert measured["cost"]["effect_observation_binding"] is None
    assert measured["harm"]["population"] is None
    assert measured["composition"]["status"] == "unknown"



def test_static_implementation_audit_turns_missing_delivery_into_unbuilt():
    from compute_cost.test12_campaign import mechanism_implementation_status

    unsupported = mechanism_implementation_status({
        "id":"FUTURE",
        "category":"FUTURE",
        "mode":"not-implemented",
    })
    assert unsupported["status"] == "UNBUILT"
    assert unsupported["basis"] == "INTERVENTION_MODE_HAS_NO_EXECUTION_BRANCH"

    malformed = mechanism_implementation_status({
        "id":"BROKEN-RETRY",
        "category":"RETRY_RECOVERY",
        "mode":"retry",
        "aux_instruction":"diagnose",
    })
    assert malformed["status"] == "UNBUILT"
    assert malformed["basis"] == "INTERVENTION_REQUIRED_FIELDS_MISSING"
