from compute_cost.adaptive import AdaptiveBudgetController, BudgetObservation


def obs(budget, result):
    return BudgetObservation(budget=budget, result_class=result)


def controller():
    return AdaptiveBudgetController(
        initial_budget=256,
        min_budget=32,
        max_budget=2048,
        granularity=32,
        boundary_repeats=3,
    )


def test_brackets_pass_fail_boundary():
    c = controller()
    assert c.next([obs(256, "ANSWER_CORRECT")]).budget == 128
    assert c.next([obs(256, "ANSWER_CORRECT"), obs(128, "ANSWER_WRONG")]).budget == 192
    assert c.next([obs(256, "ANSWER_CORRECT"), obs(128, "ANSWER_WRONG"), obs(192, "ANSWER_CORRECT")]).budget == 160


def test_truncation_scales_until_evidence_changes():
    decision = controller().next([obs(256, "THINK_TRUNCATED")])
    assert decision.action == "PROBE"
    assert decision.budget == 512
    assert decision.reason == "truncation justifies more generation headroom"
    assert controller().next([obs(256, "NO_FINAL_ANSWER")]).action == "STOP"


def test_semantic_failure_gets_one_diagnostic_larger_budget_not_an_open_loop():
    first = controller().next([obs(256, "ANSWER_WRONG")])
    assert first.action == "PROBE"
    assert first.budget == 512
    assert first.reason == "single diagnostic budget escalation for semantic failure"

    repeated = controller().next([obs(256, "ANSWER_WRONG"), obs(512, "ANSWER_WRONG")])
    assert repeated.action == "STOP"
    assert repeated.reason == "semantic failure persisted after diagnostic budget escalation"


def test_diagnostic_semantic_escalation_can_reveal_a_pass_then_search_boundary():
    c = controller()
    rows = [obs(256, "ANSWER_WRONG"), obs(512, "ANSWER_CORRECT")]
    decision = c.next(rows)
    assert decision.action == "PROBE"
    assert decision.budget == 384


def test_invalid_observation_stops_branch():
    decision = controller().next([obs(256, "SCORER_DEFECT")])
    assert decision.action == "STOP"
    assert decision.reason == "invalid experiment observation"


def test_minimum_pass_is_replicated():
    c = controller()
    rows = [obs(128, "ANSWER_WRONG"), obs(160, "ANSWER_WRONG"), obs(192, "ANSWER_CORRECT")]
    assert c.next(rows).action == "REPLICATE"
    rows += [obs(192, "ANSWER_CORRECT"), obs(192, "ANSWER_CORRECT")]
    decision = c.next(rows)
    assert decision.action == "STOP"
    assert decision.reason == "minimum passing boundary reproduced"
