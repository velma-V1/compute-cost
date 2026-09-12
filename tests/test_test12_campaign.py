from __future__ import annotations

from compute_cost.cli import build_parser
from compute_cost.config import load_config
from compute_cost.test12_campaign import (
    ACTIVE_SECONDS,
    COLLECTION_HARD_SECONDS,
    CONTROL_GRAMMAR,
    CORE_INTERVENTIONS,
    IMPROVEMENT_SURFACE,
    PROMPT_PRIMITIVES,
    TEST2_CAPABILITY_FAMILIES,
    FAMILY_CONTROL_SURFACES,
    build_intervention_bank,
    build_test12_plan,
    capability_frontier_cover,
    fresh_model_source,
    generate_prompt_control_candidates,
    validate_test12_plan,
)
from compute_cost.test12_toollab import (
    TOOL_HARNESS_POLICIES,
    execute_tool,
    parse_action,
    score_final,
)
from compute_cost.test12_tuning import (
    TUNING_HARD_SECONDS,
    build_tuning_plan,
    validate_tuning_plan,
)


def _cases(n: int = 800):
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
            "id": f"synthetic-{i:04d}",
            "category": families[i % len(families)],
            "difficulty_level": i % 11,
            "prompt": f"Synthetic task {i}",
            "scorer": "exact",
            "expected": "ok",
        }
        for i in range(n)
    ]


def test_collection_plan_is_full_campaign_under_14_hour_two_run_contract():
    cases = _cases()
    plan = build_test12_plan(cases)
    validate_test12_plan(plan)
    assert plan["wall_clock_seconds"] == 6 * 60 * 60 + 15 * 60
    assert plan["active_model_seconds"] == 6 * 60 * 60
    assert sum(row["seconds"] for row in plan["phases"]) == ACTIVE_SECONDS
    assert plan["allowed_partitions"] == ["DISCOVERY"]
    assert plan["reserved_for_tuning"] == ["VALIDATION"]
    assert plan["prohibited_partitions"] == ["TEST2_BLIND", "TEST3_PROTECTED"]
    assert plan["generated_prompt_control_count"] >= 200
    assert plan["control_search_contract"] == (
        "EVERY_DECLARED_CONTROL_CANDIDATE_GETS_MINIMUM_COVERAGE_BEFORE_REPLICATION_DEPTH_IS_ADAPTIVE"
    )
    assert plan["adaptive_allocation"]["coverage_floor_first"] is True
    assert plan["adaptive_allocation"]["successive_halving"] is True
    assert plan["adaptive_allocation"]["oracle_routing_prohibited"] is True
    assert set(IMPROVEMENT_SURFACE).issubset(set(plan["improvement_surface"]))


def test_prompt_injection_grammar_covers_every_declared_level_for_every_primitive():
    rows = generate_prompt_control_candidates()
    assert len(rows) >= 200
    by_primitive = {}
    for row in rows:
        by_primitive.setdefault(row["primitive_id"], []).append(row)

    assert set(by_primitive) == {row["id"] for row in PROMPT_PRIMITIVES}
    for primitive_id, variants in by_primitive.items():
        placements = {row["placement"] for row in variants}
        representations = {row["representation"] for row in variants}
        doses = {row["dose"] for row in variants}
        recurrence = {row["recurrence"] for row in variants}
        assert {"system", "prefix", "middle", "suffix", "post_candidate"} <= placements, primitive_id
        assert {"prose", "bullets", "schema"} <= representations, primitive_id
        assert {0.5, 1.0, 2.0} <= doses, primitive_id
        assert {1, 2, 3} <= recurrence, primitive_id


def test_fresh_model_bank_requires_no_prior_model_specific_run_and_is_broad():
    cases = _cases()
    source = fresh_model_source(cases)
    bank = build_intervention_bank(source)
    assert source["run_id"] == "FRESH-MODEL"
    assert source["source_integrity"]["verification_mode"] == "FRESH_MODEL_NO_PRIOR_EVIDENCE"
    assert len(bank) >= len(CORE_INTERVENTIONS) + 200
    categories = {row["category"] for row in bank}
    for category in {
        "PROMPT_CONTROL",
        "REASONING_MODE",
        "GENERATION_BUDGET",
        "CONTEXT_WINDOW",
        "PLANNING",
        "VERIFICATION",
        "CRITIQUE",
        "RETRY_RECOVERY",
        "STATE_TRACKING",
        "MEMORY",
        "CONTEXT_SELECTION_COMPRESSION",
        "TOOL_POLICY",
        "DELEGATION",
        "ENSEMBLE_CONSENSUS",
        "ADAPTIVE_ROUTING",
        "STOP_ESCALATE_POLICY",
        "COMPOSITION_LAYERING",
        "COMPUTE_COST_ROUTING",
    }:
        assert category in categories


def test_real_tool_lab_executes_and_returns_errors_for_bad_arguments():
    state = {}
    assert parse_action('{"tool":"calculator","arguments":{"expression":"17*23"}}') == {
        "tool": "calculator",
        "arguments": {"expression": "17*23"},
    }
    result = execute_tool("calculator", {"expression": "17*23"}, state)
    assert result == {"ok": True, "result": 391}

    bad = execute_tool("get_user", {"id": 42}, state)
    assert bad["ok"] is False
    assert bad["error"] == "ARGUMENT_TYPE"

    good = execute_tool("get_user", {"id": "42"}, state)
    assert good["ok"] is True
    assert good["result"]["name"] == "Mira"

    execute_tool("set_state", {"key": "mode", "value": "safe"}, state)
    readback = execute_tool("get_state", {"key": "mode"}, state)
    assert readback == {"ok": True, "result": "safe"}
    assert score_final("SAFE", "safe") is True
    assert len(TOOL_HARNESS_POLICIES) >= 5


def test_tuning_run_uses_validation_only_and_combined_hard_ceiling_is_12h30m():
    cases = _cases()
    plan = build_tuning_plan(cases, collection_run="collection-run")
    validate_tuning_plan(plan)
    assert plan["allowed_partitions"] == ["VALIDATION"]
    assert set(plan["prohibited_partitions"]) == {"DISCOVERY", "TEST2_BLIND", "TEST3_PROTECTED"}
    assert plan["wall_clock_seconds"] == 6 * 60 * 60 + 15 * 60
    assert plan["total_two_run_hard_ceiling_seconds"] == COLLECTION_HARD_SECONDS + TUNING_HARD_SECONDS
    assert plan["total_two_run_hard_ceiling_seconds"] == int(12.5 * 60 * 60)
    assert plan["total_two_run_hard_ceiling_seconds"] < 14 * 60 * 60


def test_default_config_and_cli_expose_collection_and_tuning_runs():
    config = load_config()
    assert "test12_campaign" in config
    assert "test12_tuning" in config

    parser = build_parser()
    collect = parser.parse_args(["gpt20b-test1.2", "--dry-run"])
    assert collect.command == "gpt20b-test1.2"
    assert collect.dry_run is True
    assert collect.seed_run is None

    tune = parser.parse_args([
        "gpt20b-test1.2-tune",
        "--collection-run",
        "test1.2-example",
        "--dry-run",
    ])
    assert tune.command == "gpt20b-test1.2-tune"
    assert tune.collection_run == "test1.2-example"
    assert tune.dry_run is True


def test_control_grammar_declares_full_finite_search_surface():
    assert len(CONTROL_GRAMMAR["prompt_primitives"]) >= 16
    assert set(CONTROL_GRAMMAR["placements"]) == {
        "system", "prefix", "middle", "suffix", "post_candidate"
    }
    assert set(CONTROL_GRAMMAR["representations"]) == {"prose", "bullets", "schema"}
    assert set(CONTROL_GRAMMAR["doses"]) == {0.5, 1.0, 2.0}
    assert set(CONTROL_GRAMMAR["recurrence_counts"]) == {1, 2, 3}
    assert len(CONTROL_GRAMMAR["retry_policies"]) >= 6
    assert len(CONTROL_GRAMMAR["multi_call_topologies"]) >= 8
    assert len(CONTROL_GRAMMAR["compute_controls"]) >= 13



def test_capability_frontier_cover_spans_easy_middle_hard_and_boundary_levels():
    cases = []
    for family_index in range(3):
        for level in range(11):
            cases.append({
                "id": f"fam{family_index}-l{level}",
                "category": f"family_{family_index}",
                "difficulty_level": level,
                "prompt": "x",
            })
    selected = capability_frontier_cover(cases)
    by_family = {}
    for row in selected:
        by_family.setdefault(row["category"], []).append(row["difficulty_level"])

    assert set(by_family) == {"family_0", "family_1", "family_2"}
    for levels in by_family.values():
        assert set(levels) == {0, 2, 5, 8, 10}



def test_test12_freezes_all_40_test2_capability_families():
    assert len(TEST2_CAPABILITY_FAMILIES) == 40
    assert set(TEST2_CAPABILITY_FAMILIES) == {
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
    }
    assert len(FAMILY_CONTROL_SURFACES) >= 10


def test_collection_plan_fails_if_any_test2_capability_family_is_missing():
    cases = [
        case for case in _cases()
        if case["category"] != "tool_error_recovery"
    ]
    plan = build_test12_plan(cases)
    assert plan["missing_capability_families"] == ["tool_error_recovery"]
    try:
        validate_test12_plan(plan)
    except ValueError as exc:
        assert "capability family contract incomplete" in str(exc)
    else:
        raise AssertionError("Test 1.2 accepted a suite missing a required capability family")


def test_collection_plan_contains_all_capability_family_manufacturing_contracts():
    plan = build_test12_plan(_cases())
    assert plan["required_capability_family_count"] == 40
    assert set(plan["required_capability_families"]) == set(TEST2_CAPABILITY_FAMILIES)
    assert plan["missing_capability_families"] == []
    assert set(plan["family_control_surfaces"]) == set(FAMILY_CONTROL_SURFACES)
    assert "capability-family-coverage.json" in plan["required_outputs"]
    assert "capability-building-block-manufacturing-map.json" in plan["required_outputs"]
