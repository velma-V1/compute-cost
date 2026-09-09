from compute_cost.runner import build_context_case


def test_context_prompt_scales_materially_with_requested_window():
    small = build_context_case(4096, timeout_s=120)
    medium = build_context_case(8192, timeout_s=120)
    large = build_context_case(32768, timeout_s=120)

    small_len = len(small["prompt"])
    medium_len = len(medium["prompt"])
    large_len = len(large["prompt"])

    assert medium_len > small_len * 1.8
    assert large_len > medium_len * 3.5


def test_context_case_distributes_three_unique_keys_across_payload():
    case = build_context_case(8192, timeout_s=120)

    assert case["scorer"] == "context_retrieval"
    assert len(case["planted_keys"]) == 3
    assert case["expected"] == "|".join(case["planted_keys"])
    for key in case["planted_keys"]:
        assert case["prompt"].count(key) == 1
    assert case["approx_target_tokens"] == 8192
    assert case["payload_chars"] == len(case["prompt"])
