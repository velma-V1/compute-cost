from __future__ import annotations

from compute_cost.test12_campaign import REQUIRED_OUTPUTS
from compute_cost.test12_model_manufacturing import (
    ZERO_CLOCK_MODEL_BUILDING_PRODUCTS,
    build_capability_curriculum,
    build_cross_family_transfer_graph,
    build_harness_to_weight_distillation,
    build_reliability_weighted_distillation,
    build_long_horizon_training_mix,
    build_preference_quality_index,
    build_failure_credit_assignment,
    build_calibration_verify_supervision,
    build_pareto_targets,
    build_router_supervision,
    build_stability_anchors,
    build_weighted_preference_pairs,
    build_zero_clock_model_manufacturing,
)


def _rows():
    base_cost = {
        "prompt_tokens_observed": 100,
        "output_tokens_observed": 20,
        "wall_seconds": 2.0,
    }
    fast_cost = {
        "prompt_tokens_observed": 90,
        "output_tokens_observed": 18,
        "wall_seconds": 1.5,
    }
    slow_cost = {
        "prompt_tokens_observed": 130,
        "output_tokens_observed": 30,
        "wall_seconds": 3.0,
    }

    rows = [
        {
            "partition": "DISCOVERY",
            "family_id": "family_a",
            "fixture_id": "a-pass",
            "difficulty_level": 2,
            "intervention_id": "CONTROL",
            "score": 1.0,
            "response_text": "A-good",
            "cost": dict(base_cost),
        },
        {
            "partition": "DISCOVERY",
            "family_id": "family_a",
            "fixture_id": "a-pass",
            "difficulty_level": 2,
            "intervention_id": "VERIFY",
            "intervention_category": "VERIFICATION",
            "model_calls_per_application": 2,
            "score": 0.0,
            "control_score": 1.0,
            "delta": -1.0,
            "task_text": "Task A pass",
            "control_response_text": "A-good",
            "treatment_response_text": "A-bad",
            "cost": dict(slow_cost),
            "control_cost": dict(base_cost),
        },
        {
            "partition": "DISCOVERY",
            "family_id": "family_a",
            "fixture_id": "a-fail",
            "difficulty_level": 7,
            "intervention_id": "CONTROL",
            "score": 0.0,
            "response_text": "A-wrong",
            "cost": dict(base_cost),
        },
        {
            "partition": "DISCOVERY",
            "family_id": "family_a",
            "fixture_id": "a-fail",
            "difficulty_level": 7,
            "intervention_id": "PLAN",
            "intervention_category": "PLANNING",
            "model_calls_per_application": 1,
            "score": 1.0,
            "control_score": 0.0,
            "delta": 1.0,
            "task_text": "Task A hard",
            "control_response_text": "A-wrong",
            "treatment_response_text": "A-fixed",
            "cost": dict(fast_cost),
            "control_cost": dict(base_cost),
        },
        {
            "partition": "DISCOVERY",
            "family_id": "family_b",
            "fixture_id": "b-pass",
            "difficulty_level": 4,
            "intervention_id": "CONTROL",
            "score": 1.0,
            "response_text": "B-good",
            "cost": dict(base_cost),
        },
        {
            "partition": "DISCOVERY",
            "family_id": "family_b",
            "fixture_id": "b-pass",
            "difficulty_level": 4,
            "intervention_id": "PLAN",
            "intervention_category": "PLANNING",
            "model_calls_per_application": 1,
            "score": 1.0,
            "control_score": 1.0,
            "delta": 0.0,
            "task_text": "Task B pass",
            "control_response_text": "B-good",
            "treatment_response_text": "B-good-short",
            "cost": dict(fast_cost),
            "control_cost": dict(base_cost),
        },
        {
            "partition": "DISCOVERY",
            "family_id": "family_b",
            "fixture_id": "b-fail",
            "difficulty_level": 8,
            "intervention_id": "CONTROL",
            "score": 0.0,
            "response_text": "B-wrong",
            "cost": dict(base_cost),
        },
        {
            "partition": "DISCOVERY",
            "family_id": "family_b",
            "fixture_id": "b-fail",
            "difficulty_level": 8,
            "intervention_id": "PLAN",
            "intervention_category": "PLANNING",
            "model_calls_per_application": 1,
            "score": 1.0,
            "control_score": 0.0,
            "delta": 1.0,
            "task_text": "Task B hard",
            "control_response_text": "B-wrong",
            "treatment_response_text": "B-fixed",
            "cost": dict(fast_cost),
            "control_cost": dict(base_cost),
        },
        {
            "partition": "DISCOVERY",
            "family_id": "family_b",
            "fixture_id": "b-fail",
            "difficulty_level": 8,
            "intervention_id": "VERIFY",
            "intervention_category": "VERIFICATION",
            "model_calls_per_application": 2,
            "score": 0.0,
            "control_score": 0.0,
            "delta": 0.0,
            "task_text": "Task B hard",
            "control_response_text": "B-wrong",
            "treatment_response_text": "B-still-wrong",
            "cost": dict(slow_cost),
            "control_cost": dict(base_cost),
        },
    ]
    return rows


def test_zero_clock_refinery_declares_twelve_training_products_and_no_new_inference():
    result = build_zero_clock_model_manufacturing(
        _rows(),
        ["family_a", "family_b"],
    )
    assert set(result["products"]) == set(ZERO_CLOCK_MODEL_BUILDING_PRODUCTS)
    assert len(result["products"]) == 12
    assert result["zero_model_calls_added"] is True
    assert result["zero_active_test_seconds_added"] is True


def test_harness_rescues_become_raw_task_distillation_targets():
    corpus = build_harness_to_weight_distillation(_rows())
    by_fixture = {row["fixture_id"]: row for row in corpus}
    assert by_fixture["a-fail"]["input"] == "Task A hard"
    assert by_fixture["a-fail"]["target"] == "A-fixed"
    assert by_fixture["a-fail"]["source_intervention_id"] == "PLAN"
    assert all("CONTROL:" not in row["input"] for row in corpus)


def test_regressions_and_rescues_become_weighted_same_task_preferences():
    pairs = build_weighted_preference_pairs(_rows())
    regression = next(
        row for row in pairs
        if row["fixture_id"] == "a-pass"
        and row["source_intervention_id"] == "VERIFY"
    )
    assert regression["chosen"] == "A-good"
    assert regression["rejected"] == "A-bad"
    assert regression["preference_reason"] == "QUALITY_REGRESSION_NEGATIVE"
    assert regression["same_task_hard_negative"] is True

    rescue = next(
        row for row in pairs
        if row["fixture_id"] == "a-fail"
        and row["source_intervention_id"] == "PLAN"
    )
    assert rescue["chosen"] == "A-fixed"
    assert rescue["rejected"] == "A-wrong"
    assert rescue["sample_weight"] > 1.0


def test_equal_quality_lower_compute_becomes_efficiency_preference():
    pairs = build_weighted_preference_pairs(_rows())
    row = next(
        item for item in pairs
        if item["fixture_id"] == "b-pass"
        and item["source_intervention_id"] == "PLAN"
    )
    assert row["preference_reason"] == "EQUAL_QUALITY_LOWER_COMPUTE"
    assert row["chosen"] == "B-good-short"
    assert row["rejected"] == "B-good"


def test_curriculum_prioritizes_measured_weakness_and_keeps_all_families():
    curriculum = build_capability_curriculum(
        _rows(),
        ["family_a", "family_b"],
    )
    assert set(curriculum["families"]) == {"family_a", "family_b"}
    assert abs(
        sum(
            row["recommended_training_mix_weight"]
            for row in curriculum["families"].values()
        )
        - 1.0
    ) < 1e-9
    assert curriculum["families"]["family_a"]["hard_fail_rate"] > 0.0
    assert curriculum["priority_order"]


def test_router_supervision_uses_direct_for_harm_and_control_for_rescue():
    rows = build_router_supervision(_rows())
    by_fixture = {row["fixture_id"]: row for row in rows}
    assert by_fixture["a-pass"]["target_action"] == "DIRECT"
    assert by_fixture["a-fail"]["target_action"] == "PLAN"
    assert by_fixture["a-fail"]["target_category"] == "PLANNING"


def test_stability_anchors_preserve_correct_base_behavior_after_observed_regression():
    anchors = build_stability_anchors(_rows())
    row = next(item for item in anchors if item["fixture_id"] == "a-pass")
    assert row["target"] == "A-good"
    assert "VERIFY" in row["observed_regressing_controls"]
    assert row["sample_weight"] > 1.0


def test_cross_family_transfer_graph_identifies_generalizers_and_specialists():
    graph = build_cross_family_transfer_graph(_rows())
    assert graph["controls"]["PLAN"]["transfer_class"] in {
        "SPARSE_POSITIVE",
        "GENERALIZER",
    }
    assert graph["controls"]["VERIFY"]["transfer_class"] in {
        "GLOBAL_VETO_CANDIDATE",
        "NULL_OR_UNRESOLVED",
        "CONDITIONAL_SPECIALIST",
    }
    assert graph["edges"]


def test_pareto_targets_preserve_quality_then_minimize_compute():
    targets = build_pareto_targets(_rows())
    by_fixture = {row["fixture_id"]: row for row in targets}
    assert by_fixture["a-fail"]["target"] == "A-fixed"
    assert by_fixture["a-fail"]["quality_score"] == 1.0
    assert by_fixture["b-pass"]["target"] == "B-good-short"
    assert by_fixture["b-pass"]["calls"] == 1.0



def test_zero_clock_training_artifacts_are_hard_required_outputs():
    expected = {
        "harness-to-weight-distillation-corpus.jsonl",
        "weighted-preference-corpus.jsonl",
        "capability-curriculum.json",
        "router-supervision-corpus.jsonl",
        "stability-anchor-corpus.jsonl",
        "cross-family-transfer-graph.json",
        "pareto-training-targets.jsonl",
        "reliability-weighted-distillation-corpus.jsonl",
        "long-horizon-training-mix.json",
        "preference-quality-index.jsonl",
        "failure-credit-assignment-corpus.jsonl",
        "calibration-verify-supervision-corpus.jsonl",
        "zero-clock-model-manufacturing-map.json",
    }
    assert expected <= set(REQUIRED_OUTPUTS)



def test_reliability_weighted_distillation_downweights_one_off_teacher_targets():
    rows = _rows()
    reliable = build_reliability_weighted_distillation(rows)
    assert reliable
    by_fixture = {
        fixture: row
        for row in reliable
        for fixture in row["fixture_ids"]
    }
    target = by_fixture["a-fail"]
    assert 0.0 <= target["teacher_reliability"] <= 1.0
    assert target["independent_support_count"] >= 1
    assert target["sample_weight"] > 0.0
    assert "supporting_interventions" in target
    assert "train_ready" in target


def test_long_horizon_mix_preserves_coverage_and_prevents_single_family_monopoly():
    rows = _rows()
    curriculum = build_capability_curriculum(
        rows,
        ["family_a", "family_b"],
    )
    anchors = build_stability_anchors(rows)
    transfer = build_cross_family_transfer_graph(rows)
    mix = build_long_horizon_training_mix(
        curriculum,
        anchors,
        transfer,
    )
    weights = [
        row["recommended_long_horizon_mix_weight"]
        for row in mix["families"].values()
    ]
    assert set(mix["families"]) == {"family_a", "family_b"}
    assert abs(sum(weights) - 1.0) < 1e-9
    assert all(weight > 0.0 for weight in weights)
    assert max(weights) < 0.85


def test_preference_quality_index_keeps_model_specific_evidence_strength():
    rows = _rows()
    preferences = build_weighted_preference_pairs(rows)
    quality = build_preference_quality_index(preferences, rows)
    assert len(quality) == len(preferences)
    assert all(
        0.0 <= row["preference_quality_score"] <= 1.0
        for row in quality
    )
    regression = next(
        row for row in quality
        if row["fixture_id"] == "a-pass"
        and row["source_intervention_id"] == "VERIFY"
    )
    assert regression["preference_reason"] == "QUALITY_REGRESSION_NEGATIVE"
    assert "outcome_sign_consistency" in regression
    assert "train_ready" in regression


def test_failure_credit_assignment_separates_model_weakness_from_control_harm():
    credit = build_failure_credit_assignment(_rows())
    by_fixture = {row["fixture_id"]: row for row in credit}
    assert (
        by_fixture["a-fail"]["failure_owner"]
        == "BASE_MODEL_BEHAVIOR_RESCUABLE_BY_CONTROL"
    )
    assert by_fixture["a-fail"]["validated_repair_action"] == "PLAN"
    assert (
        by_fixture["a-pass"]["failure_owner"]
        == "CONTROL_NEGATIVE_TRANSFER"
    )
    assert by_fixture["a-pass"]["validated_repair_action"] == "DIRECT"


def test_calibration_verify_supervision_teaches_when_to_trust_or_escalate():
    labels = build_calibration_verify_supervision(_rows())
    by_fixture = {row["fixture_id"]: row for row in labels}
    assert by_fixture["a-pass"]["target_action"] == "TRUST_DIRECT"
    assert by_fixture["a-fail"]["target_action"] in {
        "ESCALATE_TO_VALIDATED_CONTROL",
        "VERIFY",
    }
    assert by_fixture["a-fail"]["observed_direct_pass_rate"] < 1.0


def test_zero_clock_training_artifacts_include_five_new_quality_layers():
    expected = {
        "reliability-weighted-distillation-corpus.jsonl",
        "long-horizon-training-mix.json",
        "preference-quality-index.jsonl",
        "failure-credit-assignment-corpus.jsonl",
        "calibration-verify-supervision-corpus.jsonl",
    }
    assert expected <= set(REQUIRED_OUTPUTS)
