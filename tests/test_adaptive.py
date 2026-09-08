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


def test_only_evidenced_truncation_scales_up():
    assert controller().next([obs(256, "THINK_TRUNCATED")]).budget == 512
    assert controller().next([obs(256, "ANSWER_WRONG")]).action == "STOP"
    assert controller().next([obs(256, "NO_FINAL_ANSWER")]).action == "STOP"


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
