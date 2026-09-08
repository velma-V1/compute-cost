from compute_cost.scoring import score_case


def case(scorer, expected=None, **extra):
    value = {"id": "x", "category": "test", "scorer": scorer, "expected": expected}
    value.update(extra)
    return value


def test_exact_and_numeric_scoring_are_deterministic():
    assert score_case(case("exact", "BLUE"), "  BLUE\n")["score"] == 1.0
    assert score_case(case("numeric", 42, tolerance=0.0), "The answer is 42.")["score"] == 1.0
    assert score_case(case("numeric", 42, tolerance=0.0), "41")["score"] == 0.0


def test_json_schema_scorer_records_every_check():
    result = score_case(
        case("json", {"name": "Ada", "count": 3}, required=["name", "count"]),
        '{"name":"Ada","count":3,"future":true}',
    )
    assert result["score"] == 1.0
    assert result["evidence"]["parsed"]["future"] is True
    assert len(result["checks"]) >= 4


def test_extraction_set_requires_exact_requested_set():
    c = case("extraction_set", ["red", "blue"])
    assert score_case(c, '["blue", "red"]')["score"] == 1.0
    assert score_case(c, '["blue", "red", "green"]')["score"] == 0.0


def test_tool_call_shape_and_ambiguity_decision():
    tool = case("tool_call", {"tool": "lookup", "arguments": {"id": 7}})
    assert score_case(tool, '{"tool":"lookup","arguments":{"id":7}}')["score"] == 1.0
    assert score_case(case("ambiguity", "CLARIFY"), "CLARIFY: which file?")["score"] == 1.0
    assert score_case(case("ambiguity", "CLARIFY"), "ANSWER: guessed")["score"] == 0.0


def test_context_retrieval_checks_planted_fact():
    assert score_case(case("context_retrieval", "PURPLE-ORBIT-731"), "PURPLE-ORBIT-731")["score"] == 1.0


def test_coding_scorer_executes_bounded_function_tests_and_records_all_subprocess_evidence():
    c = case(
        "python_function",
        None,
        function_name="add_one",
        tests="assert add_one(1) == 2\nassert add_one(-1) == 0\n",
        timeout_s=5,
    )
    response = "```python\ndef add_one(x):\n    return x + 1\n```"
    result = score_case(c, response)

    assert result["score"] == 1.0
    sub = result["evidence"]["subprocess"]
    assert sub["returncode"] == 0
    assert sub["stdout_b64"] is not None
    assert sub["stderr_b64"] is not None
    assert sub["duration_ns"] > 0
    assert result["evidence"]["executed_source_sha256"]
    assert result["evidence"]["test_source_sha256"]


def test_coding_scorer_rejects_imports_without_execution():
    c = case("python_function", None, function_name="f", tests="assert f() == 1", timeout_s=5)
    result = score_case(c, "def f():\n    import os\n    return 1")
    assert result["score"] == 0.0
    assert result["evidence"]["safety_rejection"]
    assert "subprocess" not in result["evidence"]


def test_unknown_scorer_is_benchmark_error_not_model_failure():
    result = score_case(case("does-not-exist"), "anything")
    assert result["score"] is None
    assert result["status"] == "SCORER_ERROR"
    assert result["error"]["type"] == "UnknownScorer"
