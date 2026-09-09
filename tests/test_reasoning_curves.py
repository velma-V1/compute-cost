from compute_cost.experiments import ExperimentSpec


def base_row(level, result_class, *, experiment_id, effort="medium", duration_ns=100, eval_count=10):
    spec = ExperimentSpec(
        experiment_id=experiment_id,
        parent_experiment_id=None,
        task_id="math",
        task_family="math",
        difficulty_level=level,
        hypothesis="base frontier",
        changed_variable="baseline",
        thinking_mode=True,
        generation_budget=256,
        context_request=None,
        temperature=0.0,
        seed=42,
        prompt_variant="base",
        recovery_level=None,
        reasoning_effort=effort,
    )
    return {
        "experiment": spec.to_dict(),
        "classification": {
            "result_class": result_class,
            "valid_for_capability": True,
        },
        "score": 1.0 if result_class == "ANSWER_CORRECT" else 0.0,
        "metrics": {"total_duration_ns": duration_ns, "eval_count": eval_count},
        "phase_metrics": {"thinking_chars": 20, "thinking_span_ns": 40},
    }


def test_reasoning_target_selection_is_dominance_aware():
    from compute_cost.reasoning_curves import select_reasoning_targets

    frontiers = {
        "families": {
            "math": {"reliable_floor": 4, "first_failure_level": 5},
            "all_pass": {"reliable_floor": 10, "first_failure_level": None},
            "all_fail": {"reliable_floor": None, "first_failure_level": 0},
            "unknown": {"reliable_floor": None, "first_failure_level": None},
        }
    }
    assert select_reasoning_targets(frontiers) == {
        "all_fail": [{"effort": "high", "level": 0, "purpose": "frontier_extension"}],
        "all_pass": [{"effort": "low", "level": 10, "purpose": "cost_reduction"}],
        "math": [
            {"effort": "low", "level": 4, "purpose": "cost_reduction"},
            {"effort": "high", "level": 5, "purpose": "frontier_extension"},
        ],
        "unknown": [],
    }


def test_reasoning_curve_execution_replicates_only_selected_frontier_targets(monkeypatch):
    import compute_cost.reasoning_curves as curves

    calls = []

    class Store:
        def __init__(self):
            self.rows = []

        def append_jsonl(self, path, row):
            self.rows.append((path, row))

    class Runner:
        def __init__(self):
            self.store = Store()
            self.config = {"reasoning_curves": {"enabled": True, "repeats": 2}}
            self.progress = None

        @staticmethod
        def _utc():
            return "x"

    cases = [
        {"id": "math-L4", "family_id": "math", "category": "math", "difficulty_level": 4},
        {"id": "math-L5", "family_id": "math", "category": "math", "difficulty_level": 5},
    ]
    base_rows = [
        base_row(4, "ANSWER_CORRECT", experiment_id="base-L4"),
        base_row(5, "ANSWER_WRONG", experiment_id="base-L5"),
    ]
    frontiers = {"families": {"math": {"reliable_floor": 4, "first_failure_level": 5}}}

    def execute(runner, case, spec, *, parent=None):
        calls.append((spec.reasoning_effort, spec.difficulty_level, spec.changed_variable, parent.experiment_id))
        return {
            "experiment": spec.to_dict(),
            "classification": {"result_class": "ANSWER_CORRECT", "valid_for_capability": True},
            "score": 1.0,
            "metrics": {"total_duration_ns": 50 if spec.reasoning_effort == "low" else 150, "eval_count": 8},
            "phase_metrics": {"thinking_chars": 10, "thinking_span_ns": 20},
        }

    monkeypatch.setattr(curves, "execute_experiment", execute)
    rows = curves.run_reasoning_curves(
        Runner(),
        cases,
        base_rows,
        frontiers,
        sequence_start=2,
    )

    assert [(effort, level) for effort, level, _, _ in calls] == [
        ("low", 4), ("low", 4), ("high", 5), ("high", 5)
    ]
    assert calls[0][2:] == ("reasoning_effort", "base-L4")
    assert calls[1][2] == "replication"
    assert calls[2][2:] == ("reasoning_effort", "base-L5")
    assert calls[3][2] == "replication"
    assert len(rows) == 4


def test_reasoning_curve_summary_reports_cheapest_effort_and_demonstrated_extension():
    from compute_cost.reasoning_curves import build_reasoning_curves

    base_rows = [
        base_row(4, "ANSWER_CORRECT", experiment_id="m4-a", duration_ns=100),
        base_row(4, "ANSWER_CORRECT", experiment_id="m4-b", duration_ns=100),
        base_row(5, "ANSWER_WRONG", experiment_id="m5-a", duration_ns=110),
        base_row(5, "ANSWER_WRONG", experiment_id="m5-b", duration_ns=110),
    ]
    effort_rows = []
    for i in range(2):
        effort_rows.append(base_row(4, "ANSWER_CORRECT", experiment_id=f"l4-{i}", effort="low", duration_ns=60))
        effort_rows.append(base_row(5, "ANSWER_CORRECT", experiment_id=f"h5-{i}", effort="high", duration_ns=180))

    summary = build_reasoning_curves(
        "gpt-oss:20b",
        {"families": {"math": {"reliable_floor": 4, "first_failure_level": 5}}},
        base_rows,
        effort_rows,
        repeats=2,
        reliable_threshold=0.90,
    )

    assert summary["schema_version"] == 1
    assert summary["measurement_policy"]["gpt_oss_think_control"] == ["low", "medium", "high"]
    assert summary["measurement_policy"]["thinking_disabled"] is False
    assert summary["measurement_policy"]["generation_budget_is_thinking_budget"] is False
    math = summary["families"]["math"]
    assert math["minimum_reliable_effort_at_baseline_floor"] == "low"
    assert math["demonstrated_high_effort_extension_to"] == 5
    assert math["baseline_medium_frontier"] == {"reliable_floor": 4, "first_failure_level": 5}
    assert math["efforts"]["low"]["levels"]["4"]["pass_rate"] == 1.0
    assert math["efforts"]["high"]["levels"]["5"]["pass_rate"] == 1.0
    assert math["efforts"]["low"]["levels"]["4"]["average_total_duration_ns"] == 60.0
