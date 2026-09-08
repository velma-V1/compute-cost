import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest


EXPECTED_QWEN_SHA256 = "8d7cd2eadaa3c105a491f234200f57c6332d019938104619d11088330c855618"
EXPECTED_FAMILIES = {
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
    "debugging_root_cause",
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
    "memory_compression_summarization_fidelity",
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
ALLOWED_COVERAGE = {"PROVEN", "PARTIAL", "UNCERTAIN", "UNTESTED"}
ALLOWED_SCORERS = {
    "exact",
    "numeric",
    "json",
    "extraction_set",
    "tool_call",
    "ambiguity",
    "context_retrieval",
    "python_function",
}


def _load_required_json(path: str) -> dict:
    file_path = Path(path)
    assert file_path.is_file(), f"required capability artifact missing: {path}"
    value = json.loads(file_path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_validator():
    spec = importlib.util.find_spec("compute_cost.capability_suite")
    assert spec is not None, "compute_cost.capability_suite is not implemented"
    module = importlib.import_module("compute_cost.capability_suite")
    assert hasattr(module, "validate_capability_suite")
    return module.validate_capability_suite


def test_qwen_characterization_template_is_byte_for_byte_unchanged():
    payload = Path("benchmarks/qwen-characterization-v1.json").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_QWEN_SHA256


def test_gpt_oss_taxonomy_and_breadth_suite_cover_all_40_families():
    taxonomy = _load_required_json("benchmarks/capability-taxonomy-v1.json")
    suite = _load_required_json("benchmarks/gpt-oss-20b-capability-v1.json")

    assert taxonomy["taxonomy_version"] == "capability-taxonomy-v1"
    assert suite["benchmark_version"] == "gpt-oss-20b-capability-v1"
    assert suite["schema_version"] == "capability-suite-v1"
    assert suite["taxonomy_version"] == taxonomy["taxonomy_version"]

    families = taxonomy["families"]
    family_ids = [family["id"] for family in families]
    assert len(family_ids) == 40
    assert len(family_ids) == len(set(family_ids))
    assert set(family_ids) == EXPECTED_FAMILIES

    for family in families:
        assert family["name"]
        assert family["rubric_version"]
        assert isinstance(family["difficulty_dimensions"], list)
        assert family["difficulty_dimensions"]
        assert all(isinstance(item, str) and item for item in family["difficulty_dimensions"])

    assert set(suite["coverage"]) == EXPECTED_FAMILIES
    assert set(suite["coverage"].values()) <= ALLOWED_COVERAGE

    case_ids = [case["id"] for case in suite["cases"]]
    assert len(case_ids) == len(set(case_ids))
    case_family_ids = {case["capability_map"]["family_id"] for case in suite["cases"]}
    assert case_family_ids == EXPECTED_FAMILIES


def test_each_gpt_oss_case_preserves_legacy_runner_fields_and_adds_capability_metadata():
    taxonomy = _load_required_json("benchmarks/capability-taxonomy-v1.json")
    suite = _load_required_json("benchmarks/gpt-oss-20b-capability-v1.json")
    taxonomy_ids = {family["id"] for family in taxonomy["families"]}

    for case in suite["cases"]:
        for field in ("id", "category", "difficulty_level", "prompt", "scorer", "timeout_s"):
            assert field in case, f"{case.get('id')} missing legacy field {field}"
        assert case["id"]
        assert case["category"]
        assert case["prompt"]
        assert case["scorer"] in ALLOWED_SCORERS
        assert isinstance(case["difficulty_level"], int)
        assert 0 <= case["difficulty_level"] <= 10
        assert isinstance(case["timeout_s"], (int, float)) and case["timeout_s"] > 0

        meta = case["capability_map"]
        assert meta["taxonomy_version"] == "capability-taxonomy-v1"
        assert meta["family_id"] in taxonomy_ids
        assert meta["rubric_version"]
        assert meta["difficulty"]["level"] == case["difficulty_level"]
        assert isinstance(meta["difficulty"]["dimensions"], dict)
        assert meta["difficulty"]["dimensions"]
        assert isinstance(meta["capabilities_required"], list)
        assert meta["capabilities_required"]
        assert set(meta["capabilities_required"]) <= taxonomy_ids
        assert isinstance(meta["recovery_eligible"], bool)
        assert isinstance(meta["robustness_eligible"], bool)


def test_validator_accepts_the_committed_capability_suite():
    validate_capability_suite = _load_validator()
    taxonomy = _load_required_json("benchmarks/capability-taxonomy-v1.json")
    suite = _load_required_json("benchmarks/gpt-oss-20b-capability-v1.json")
    assert validate_capability_suite(suite, taxonomy) is None


def test_validator_rejects_unknown_family_reference():
    validate_capability_suite = _load_validator()
    taxonomy = _load_required_json("benchmarks/capability-taxonomy-v1.json")
    suite = _load_required_json("benchmarks/gpt-oss-20b-capability-v1.json")
    broken = copy.deepcopy(suite)
    broken["cases"][0]["capability_map"]["family_id"] = "not_a_real_family"
    with pytest.raises(ValueError, match="unknown family"):
        validate_capability_suite(broken, taxonomy)


def test_validator_rejects_mismatched_nested_difficulty():
    validate_capability_suite = _load_validator()
    taxonomy = _load_required_json("benchmarks/capability-taxonomy-v1.json")
    suite = _load_required_json("benchmarks/gpt-oss-20b-capability-v1.json")
    broken = copy.deepcopy(suite)
    broken["cases"][0]["capability_map"]["difficulty"]["level"] += 1
    with pytest.raises(ValueError, match="difficulty level mismatch"):
        validate_capability_suite(broken, taxonomy)
