from types import SimpleNamespace

import pytest

from compute_cost.call_ledger import CallBudgetExceeded, CallLedger
from compute_cost.comparison_matrix import FIXED_LEVELS, native_reasoning_conditions, planned_fixed_core
from compute_cost.score_vector import build_score_vector


def test_call_ledger_blocks_701st_call_before_authorization():
    ledger = CallLedger(limit=700)
    for _ in range(700):
        ledger.authorize("fixed_capability", family="f")
    assert ledger.snapshot()["calls_used"] == 700
    assert ledger.snapshot()["remaining_calls"] == 0
    with pytest.raises(CallBudgetExceeded):
        ledger.authorize("diagnostic", family="f")
    snap = ledger.snapshot()
    assert snap["calls_used"] == 700
    assert snap["rejected_call_attempts"] == 1


def test_native_reasoning_conditions_are_real_and_standardized():
    gpt = native_reasoning_conditions("gpt-oss:20b")
    assert [(x.role, x.request_value) for x in gpt] == [
        ("BASELINE_MINIMAL", "low"),
        ("ENHANCED", "medium"),
        ("MAX_NATIVE", "high"),
    ]
    qwen = native_reasoning_conditions("qwen3.5:35b-a3b-q4_K_M")
    assert [(x.role, x.request_value) for x in qwen] == [
        ("BASELINE_MINIMAL", False),
        ("ENHANCED", True),
    ]


def test_fixed_core_call_math_matches_approved_contract():
    assert FIXED_LEVELS == (2, 5, 8, 10)
    gpt = planned_fixed_core("gpt-oss:20b", family_count=40, scenario_count=6, turns=8)
    assert gpt == {"capability": 480, "autonomous": 144, "total": 624, "reserve": 76, "limit": 700}
    qwen = planned_fixed_core("qwen3.5:35b-a3b-q4_K_M", family_count=40, scenario_count=6, turns=8)
    assert qwen["capability"] == 320
    assert qwen["autonomous"] == 96
    assert qwen["total"] == 416
    assert qwen["reserve"] == 284


def test_score_vector_separates_semantic_from_format_contract():
    case = {"id": "decomp", "scorer": "json", "expected": {"steps": ["A1", "A2", "A3"]}}
    scoring = {
        "score": 0.0,
        "status": "SCORED",
        "checks": [{"name": "valid_json", "pass": False}],
        "evidence": {"raw_response": "```json\n{\"steps\":[\"A1\",\"A2\",\"A3\"]}\n```"},
    }
    classification = {"result_class": "FORMAT_FAILURE", "valid_for_capability": True}
    generation = {
        "normalized": {"text": "```json\n{\"steps\":[\"A1\",\"A2\",\"A3\"]}\n```", "thinking": ""},
        "metrics": {"prompt_eval_count": 100, "eval_count": 20, "prompt_eval_duration_ns": 100_000_000, "eval_duration_ns": 200_000_000},
        "timing": {"client_latency_ns": 350_000_000},
    }
    vector = build_score_vector(case, scoring, classification, generation, telemetry=[])
    assert vector["semantic_correctness"] == 100.0
    assert vector["contract_format_compliance"] == 0.0
    assert vector["runtime_validity"] == 100.0
    assert vector["prompt_tokens_per_second"] == 1000.0
    assert vector["generation_tokens_per_second"] == 100.0


def test_score_vector_uses_explicit_non_applicable_and_unavailable_states():
    vector = build_score_vector(
        {"id": "plain", "scorer": "exact", "expected": "OK"},
        {"score": 1.0, "status": "SCORED", "checks": [{"name": "exact_match", "pass": True}], "evidence": {"raw_response": "OK"}},
        {"result_class": "ANSWER_CORRECT", "valid_for_capability": True},
        {"normalized": {"text": "OK", "thinking": ""}, "metrics": {}, "timing": {}},
        telemetry=[],
    )
    assert vector["semantic_correctness"] == 100.0
    assert vector["tool_procedure_compliance"] == "NOT_APPLICABLE"
    assert vector["state_checkpoint_accuracy"] == "NOT_APPLICABLE"
    assert vector["gpu_energy_wh"] == "UNAVAILABLE"


def test_attempt_dossier_policy_persists_successful_non_retry_calls():
    from compute_cost.attempt_dossier import should_persist_attempt_dossier

    spec = SimpleNamespace(parent_experiment_id=None, changed_variable="baseline")
    assert should_persist_attempt_dossier(spec, {"result_class": "ANSWER_CORRECT"}) is True
