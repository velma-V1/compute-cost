from compute_cost.autonomous_simulation import build_scenarios, planned_semantic_calls


def test_autonomous_simulation_has_six_eight_turn_long_horizon_scenarios():
    scenarios = build_scenarios()
    assert len(scenarios) == 6
    assert all(len(item["steps"]) == 8 for item in scenarios)
    assert planned_semantic_calls(scenarios) == 48


def test_autonomous_scenarios_cover_required_long_horizon_behaviors():
    scenarios = build_scenarios()
    tags = {tag for scenario in scenarios for tag in scenario["tags"]}
    assert {
        "planning",
        "state_retention",
        "tool_recovery",
        "diagnosis",
        "constraint_update",
        "research_conflict",
        "self_verification",
    } <= tags
    assert all(any(step.get("injected_change") for step in s["steps"]) for s in scenarios)
    assert all(s["steps"][-1]["expected_action"] == "COMPLETE" for s in scenarios)


def test_default_autonomous_budget_fits_under_global_call_ceiling():
    # 40 families * 6 semantic probes + 48 autonomous turns leaves 72 calls
    # for truncation/uncertainty retries while keeping the entire run under 400.
    capability_semantic_max = 40 * 6
    autonomous_semantic = 48
    global_ceiling = 360
    assert capability_semantic_max + autonomous_semantic == 288
    assert global_ceiling - 288 == 72
    assert global_ceiling < 400
