from compute_cost.adaptive import (
    AdaptiveBudgetController,
    AdaptiveDifficultyController,
    BudgetObservation,
    DifficultyObservation,
)


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


def dobs(level, passed, valid=True):
    return DifficultyObservation(
        level=level,
        passed=passed,
        valid_for_capability=valid,
    )


def difficulty_controller():
    return AdaptiveDifficultyController(
        min_level=0,
        max_level=10,
        anchor_level=1,
        jump=3,
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


def test_difficulty_search_starts_at_family_anchor():
    decision = difficulty_controller().next([])
    assert decision.action == "PROBE"
    assert decision.level == 1
    assert decision.reason == "establish family difficulty anchor"


def test_difficulty_search_jumps_up_after_passes():
    c = difficulty_controller()
    first = c.next([dobs(1, True)])
    assert (first.action, first.level) == ("PROBE", 4)
    second = c.next([dobs(1, True), dobs(4, True)])
    assert (second.action, second.level) == ("PROBE", 7)


def test_difficulty_search_bisects_pass_fail_bracket():
    c = difficulty_controller()
    rows = [dobs(1, True), dobs(4, True), dobs(7, False)]
    decision = c.next(rows)
    assert (decision.action, decision.level) == ("PROBE", 5)
    rows.append(dobs(5, True))
    decision = c.next(rows)
    assert (decision.action, decision.level) == ("PROBE", 6)


def test_adjacent_difficulty_transition_replicates_both_sides_then_stops():
    c = difficulty_controller()
    rows = [dobs(5, True), dobs(6, False)]

    decision = c.next(rows)
    assert (decision.action, decision.level) == ("REPLICATE", 5)
    rows.extend([dobs(5, True), dobs(5, True)])

    decision = c.next(rows)
    assert (decision.action, decision.level) == ("REPLICATE", 6)
    rows.extend([dobs(6, False), dobs(6, False)])

    decision = c.next(rows)
    assert decision.action == "STOP"
    assert decision.level is None
    assert decision.reason == "adjacent difficulty boundary reproduced"


def test_invalid_difficulty_observation_retries_same_level_without_counting_failure():
    c = difficulty_controller()
    decision = c.next([dobs(4, None, valid=False)])
    assert (decision.action, decision.level) == ("REPLICATE", 4)
    assert decision.reason == "retry invalid difficulty observation"


def test_difficulty_ceiling_pass_is_reproduced_before_stop():
    c = difficulty_controller()
    rows = [dobs(1, True), dobs(4, True), dobs(7, True), dobs(10, True)]
    decision = c.next(rows)
    assert (decision.action, decision.level) == ("REPLICATE", 10)
    rows.extend([dobs(10, True), dobs(10, True)])
    decision = c.next(rows)
    assert decision.action == "STOP"
    assert decision.reason == "maximum difficulty pass reproduced"


def test_difficulty_anchor_failure_searches_down_to_floor_and_reproduces_it():
    c = difficulty_controller()
    decision = c.next([dobs(1, False)])
    assert (decision.action, decision.level) == ("PROBE", 0)
    rows = [dobs(1, False), dobs(0, False)]
    decision = c.next(rows)
    assert (decision.action, decision.level) == ("REPLICATE", 0)
    rows.extend([dobs(0, False), dobs(0, False)])
    decision = c.next(rows)
    assert decision.action == "STOP"
    assert decision.reason == "minimum difficulty failure reproduced"
