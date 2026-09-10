from pathlib import Path

from compute_cost.attempt_dossier import build_attempt_dossier
from compute_cost.experiments import ExperimentSpec


def _spec(*, parent=None, changed="baseline", budget=256):
    return ExperimentSpec(
        experiment_id="exp-2",
        parent_experiment_id=parent,
        task_id="task-x",
        task_family="formal_logic_deduction",
        difficulty_level=7,
        hypothesis="locate failure boundary",
        changed_variable=changed,
        thinking_mode=True,
        reasoning_effort="low",
        generation_budget=budget,
        context_request=8192,
        temperature=0.0,
        seed=42,
        prompt_variant="base",
        recovery_level=None,
    )


def test_dossier_captures_every_observable_layer_and_marks_hidden_reasoning_unobservable():
    case = {
        "id": "logic-L7",
        "category": "formal_logic_deduction",
        "difficulty_level": 7,
        "prompt": "Question text",
        "scorer": "exact",
        "expected": "NO",
        "tags": ["logic", "boundary"],
    }
    invocation = {
        "model": "gpt-oss:20b",
        "messages": [
            {"role": "system", "content": "SYSTEM STATE"},
            {"role": "user", "content": "Question text"},
        ],
        "options": {"num_predict": 256, "temperature": 0.0, "seed": 42},
        "stream": True,
        "request_fields": {"think": "low"},
    }
    generation = {
        "ok": True,
        "normalized": {
            "text": "YES",
            "thinking": "I considered A then B",
            "tool_calls": [],
            "done_reason": "stop",
        },
        "metrics": {
            "prompt_eval_count": 100,
            "prompt_eval_duration": 2_000_000_000,
            "eval_count": 50,
            "eval_duration": 1_000_000_000,
        },
        "timing": {"client_latency_ns": 4_000_000_000},
        "phase_metrics": {"thinking_tokens": 20, "answer_tokens": 30},
        "stream_events": [{"sequence": 0, "parsed": {"message": {"thinking": "I considered"}}}],
        "raw_response_b64": "e30=",
        "request": {"body_b64": "e30="},
    }
    scoring = {
        "score": 0.0,
        "status": "SCORED",
        "checks": [{"name": "exact_match", "pass": False, "expected": "NO", "actual": "YES"}],
        "evidence": {"raw_response": "YES"},
    }
    classification = {
        "result_class": "ANSWER_WRONG",
        "basis": "deterministic scorer failed",
        "valid_for_capability": True,
    }
    refs = {
        "request_id": "req-1",
        "request": {"path": "raw/runtime/requests/req-1.bin"},
        "response": {"path": "raw/runtime/responses/req-1.bin"},
        "streams": [{"path": "raw/runtime/streams/req-1/000000.bin"}],
    }

    dossier = build_attempt_dossier(
        case=case,
        spec=_spec(),
        invocation=invocation,
        generation=generation,
        scoring=scoring,
        classification=classification,
        evidence_refs=refs,
        telemetry_before=[{"gpu_util": 50}],
        telemetry_after=[{"gpu_util": 60}],
    )

    assert dossier["question"]["family"] == "formal_logic_deduction"
    assert dossier["question"]["difficulty_level"] == 7
    assert dossier["model_visible"]["messages"] == invocation["messages"]
    assert dossier["model_visible"]["exact_serialized_request_ref"] == refs["request"]
    assert dossier["model_output"]["final_answer"] == "YES"
    assert dossier["model_output"]["exposed_thinking"] == "I considered A then B"
    assert dossier["model_output"]["hidden_internal_reasoning"] == "UNOBSERVABLE"
    assert dossier["model_output"]["raw_response_ref"] == refs["response"]
    assert dossier["model_output"]["raw_stream_refs"] == refs["streams"]
    assert dossier["evaluation"]["wrong_checks"][0]["name"] == "exact_match"
    assert dossier["evaluation"]["right_checks"] == []
    assert dossier["performance"]["prompt_tokens_per_second"] == 50.0
    assert dossier["performance"]["generation_tokens_per_second"] == 50.0
    assert dossier["performance"]["client_latency_seconds"] == 4.0
    assert dossier["telemetry"]["before"] == [{"gpu_util": 50}]
    assert dossier["telemetry"]["after"] == [{"gpu_util": 60}]


def test_retry_dossier_preserves_parent_and_only_changed_variable():
    dossier = build_attempt_dossier(
        case={"id": "x", "category": "coding_generation", "difficulty_level": 6, "prompt": "p", "scorer": "exact", "expected": "x"},
        spec=_spec(parent="exp-1", changed="generation_budget", budget=512),
        invocation={"model": "m", "messages": [{"role": "user", "content": "p"}], "options": {"num_predict": 512}, "request_fields": {}},
        generation={"ok": True, "normalized": {"text": "x", "thinking": "", "tool_calls": [], "done_reason": "stop"}, "metrics": {}, "timing": {}, "phase_metrics": {}},
        scoring={"score": 1.0, "status": "SCORED", "checks": [{"name": "exact_match", "pass": True}]},
        classification={"result_class": "ANSWER_CORRECT", "basis": "passed", "valid_for_capability": True},
        evidence_refs={"request_id": "r"},
        telemetry_before=[],
        telemetry_after=[],
    )
    assert dossier["lineage"]["is_retry"] is True
    assert dossier["lineage"]["parent_experiment_id"] == "exp-1"
    assert dossier["lineage"]["changed_variable"] == "generation_budget"
    assert dossier["evaluation"]["right_checks"][0]["name"] == "exact_match"
