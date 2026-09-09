import json
from pathlib import Path


def test_base_v1_has_unique_complete_cases_across_all_required_categories():
    suite = json.loads(Path("benchmarks/base-v1.json").read_text(encoding="utf-8"))
    cases = suite["cases"]
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))

    required_categories = {
        "instruction_following",
        "structured_output",
        "extraction",
        "reasoning_math",
        "coding",
        "tool_call",
        "ambiguity_handling",
        "long_context_retrieval",
    }
    assert {case["category"] for case in cases} == required_categories

    for case in cases:
        assert case["id"]
        assert case["prompt"]
        assert case["scorer"]
        assert isinstance(case["timeout_s"], (int, float)) and case["timeout_s"] > 0
        assert isinstance(case["max_output_tokens"], int) and case["max_output_tokens"] > 0


def test_each_base_category_has_multiple_cases_to_reduce_single_prompt_noise():
    suite = json.loads(Path("benchmarks/base-v1.json").read_text(encoding="utf-8"))
    counts = {}
    for case in suite["cases"]:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    assert all(count >= 2 for count in counts.values())


def test_qwen_characterization_v1_is_small_deterministic_and_budget_free():
    suite = json.loads(Path("benchmarks/qwen-characterization-v1.json").read_text(encoding="utf-8"))
    assert suite["benchmark_version"] == "qwen-characterization-v1"
    cases = suite["cases"]
    assert [case["id"] for case in cases] == [
        "char-if-001",
        "char-json-001",
        "char-math-001",
    ]
    assert [case["difficulty_level"] for case in cases] == [1, 2, 3]
    assert all("max_output_tokens" not in case for case in cases)
    assert all(case["timeout_s"] == 120 for case in cases)
