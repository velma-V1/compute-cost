import json
from pathlib import Path

from compute_cost.capability_suite import build_ladder_coverage, validate_capability_suite
from compute_cost.scoring import score_case


CALIBRATED_FAMILIES = {
    "instruction_following_constraint_stacking",
    "debugging_root_cause_diagnosis",
    "test_generation_verification",
    "multi_tool_sequencing",
    "tool_error_recovery",
    "missing_information_handling",
    "hallucination_resistance",
    "decomposition",
    "self_correction",
    "meta_reasoning",
    "sibling_transfer_generalization",
    "composite_agent_tasks",
}

PRIMARY_DIMENSIONS = {
    "instruction_following_constraint_stacking": "constraint_count",
    "debugging_root_cause_diagnosis": "fault_distance",
    "test_generation_verification": "requirement_count",
    "multi_tool_sequencing": "dependency_depth",
    "tool_error_recovery": "recovery_steps",
    "missing_information_handling": "missing_fields",
    "hallucination_resistance": "pressure_to_answer",
    "decomposition": "subproblem_count",
    "self_correction": "repair_depth",
    "meta_reasoning": "strategy_choices",
    "sibling_transfer_generalization": "distribution_shift",
    "composite_agent_tasks": "capability_count",
}


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_materialized_suite_replaces_metadata_only_ladders_with_semantic_escalation():
    from compute_cost.gpt_oss_calibration import CALIBRATION_VERSION, materialize_gpt_oss_suite

    taxonomy = _load("benchmarks/capability-taxonomy-v1.json")
    seed = _load("benchmarks/gpt-oss-20b-capability-v1.json")
    first = materialize_gpt_oss_suite(seed, taxonomy)
    second = materialize_gpt_oss_suite(seed, taxonomy)

    assert first == second
    assert first["ladder_calibration_version"] == CALIBRATION_VERSION
    assert len(first["cases"]) == 440
    validate_capability_suite(first, taxonomy)
    assert build_ladder_coverage(first, taxonomy)["complete"] is True

    by_family = {}
    for case in first["cases"]:
        by_family.setdefault(case["capability_map"]["family_id"], {})[case["difficulty_level"]] = case

    for family_id in CALIBRATED_FAMILIES:
        levels = by_family[family_id]
        assert set(levels) == set(range(11))
        primary = PRIMARY_DIMENSIONS[family_id]
        low = levels[0]
        high = levels[10]
        assert low["fixture_calibration"]["calibration_version"] == CALIBRATION_VERSION
        assert high["fixture_calibration"]["calibration_version"] == CALIBRATION_VERSION
        assert low["prompt"] != high["prompt"]
        assert (
            high["capability_map"]["difficulty"]["dimensions"][primary]
            > low["capability_map"]["difficulty"]["dimensions"][primary]
        ), family_id


def test_every_semantically_calibrated_oracle_passes_the_real_scorer():
    from compute_cost.gpt_oss_calibration import materialize_gpt_oss_suite

    taxonomy = _load("benchmarks/capability-taxonomy-v1.json")
    seed = _load("benchmarks/gpt-oss-20b-capability-v1.json")
    suite = materialize_gpt_oss_suite(seed, taxonomy)

    calibrated = [
        case for case in suite["cases"]
        if case["capability_map"]["family_id"] in CALIBRATED_FAMILIES
    ]
    assert len(calibrated) == len(CALIBRATED_FAMILIES) * 11
    for case in calibrated:
        scored = score_case(case, str(case["oracle_response"]))
        assert scored["status"] == "SCORED", case["id"]
        assert scored["score"] == 1.0, (case["id"], scored)


def test_boundary_tiers_add_distinct_adversarial_or_pathological_pressure():
    from compute_cost.gpt_oss_calibration import materialize_gpt_oss_suite

    taxonomy = _load("benchmarks/capability-taxonomy-v1.json")
    seed = _load("benchmarks/gpt-oss-20b-capability-v1.json")
    suite = materialize_gpt_oss_suite(seed, taxonomy)

    for case in suite["cases"]:
        family_id = case["capability_map"]["family_id"]
        if family_id not in CALIBRATED_FAMILIES or case["difficulty_level"] < 8:
            continue
        pressure = case["fixture_calibration"]["pressure"]
        assert pressure in {"ADVERSARIAL", "PATHOLOGICAL", "BOUNDARY_BREAKING"}
        expected = {
            8: "ADVERSARIAL",
            9: "PATHOLOGICAL",
            10: "BOUNDARY_BREAKING",
        }[case["difficulty_level"]]
        assert pressure == expected
