from compute_cost.classification import classify_result


def generation(*, text="", thinking="", done_reason="stop", ok=True, error=None):
    return {
        "ok": ok,
        "normalized": {"text": text, "thinking": thinking, "done_reason": done_reason},
        "metrics": {"eval_count": 16},
        "error": error,
    }


def test_thinking_only_length_stop_is_not_semantic_failure():
    result = classify_result(
        {"scorer": "exact"},
        generation(thinking="Thinking Process: analyze request", done_reason="length"),
        {"status": "SCORED", "score": 0.0, "checks": []},
    )
    assert result["result_class"] == "THINK_TRUNCATED"
    assert result["basis"] == "length stop with exposed thinking and no final content"
    assert result["valid_for_capability"] is False


def test_scorer_defect_is_quarantined():
    result = classify_result(
        {"scorer": "json"},
        generation(text="{}"),
        {"status": "SCORER_ERROR", "score": None, "checks": []},
    )
    assert result["result_class"] == "SCORER_DEFECT"
    assert result["valid_for_capability"] is False


def test_invalid_json_and_correct_answer_classify_separately():
    malformed = classify_result(
        {"scorer": "json"},
        generation(text="not-json"),
        {"status": "SCORED", "score": 0.0, "checks": [{"name": "valid_json", "pass": False}]},
    )
    correct = classify_result(
        {"scorer": "exact"},
        generation(text="BLUE", thinking="plan", done_reason="length"),
        {"status": "SCORED", "score": 1.0, "checks": []},
    )
    assert malformed["result_class"] == "FORMAT_FAILURE"
    assert malformed["valid_for_capability"] is True
    assert correct["result_class"] == "ANSWER_CORRECT"
    assert correct["valid_for_capability"] is True


def test_runtime_timeout_and_resource_exhaustion_are_not_capability_results():
    timeout = classify_result(
        {"scorer": "exact"},
        generation(ok=False, error={"type": "TimeoutError", "message": "request timeout"}),
        {"status": "RUNTIME_ERROR", "score": 0.0, "checks": []},
    )
    oom = classify_result(
        {"scorer": "exact"},
        generation(ok=False, error={"type": "RuntimeError", "message": "out of memory"}),
        {"status": "RUNTIME_ERROR", "score": 0.0, "checks": []},
    )
    assert timeout["result_class"] == "TIMEOUT"
    assert timeout["valid_for_capability"] is False
    assert oom["result_class"] == "RESOURCE_LIMIT"
    assert oom["valid_for_capability"] is False
