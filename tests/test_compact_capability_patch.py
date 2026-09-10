from __future__ import annotations

from pathlib import Path


def _case(level: int = 2, family: str = "math") -> dict:
    return {
        "id": f"{family}-L{level}",
        "family_id": family,
        "category": family,
        "difficulty_level": level,
        "difficulty": {
            "level": level,
            "rubric_version": f"{family}-v1",
            "dimensions": {"load": level},
        },
        "prompt": f"{family} level {level}",
        "scorer": "exact",
        "expected": "OK",
        "timeout_s": 120,
        "capabilities_required": [family],
        "recovery_eligible": True,
        "robustness_eligible": True,
        "compound": False,
        "tags": [],
    }


class _Store:
    def __init__(self):
        self.rows: dict[str, list[dict]] = {}
        self.jsons: dict[str, dict] = {}

    def append_jsonl(self, path, row):
        self.rows.setdefault(path, []).append(row)

    def write_json(self, path, value, **kwargs):
        self.jsons[path] = value
        return {"sha256": "x"}


class _Runner:
    def __init__(self):
        self.store = _Store()
        self.model = "qwen3.5:35b-a3b-q4_K_M"
        self.suite = {
            "benchmark_version": "test",
            "taxonomy_version": "capability-taxonomy-v1",
        }
        self.config = {
            "capability_campaign": {
                "anchor_level": 2,
                "jump": 3,
                "boundary_repeats": 1,
                "max_experiments_per_family": 5,
                "thinking_mode": False,
                "reasoning_effort": None,
                "generation_budget": 256,
                "reliable_threshold": 0.90,
                "unstable_threshold": 0.40,
            },
            "characterization": {"max_generation_budget": 2048},
            "reasoning_curves": {"enabled": False, "repeats": 1},
            "recovery_lab": {"enabled": False, "repeats": 1, "max_level": "R7"},
            "robustness_lab": {
                "enabled": False,
                "repeats": 1,
                "max_perturbations_per_family": 1,
                "max_attempts_per_perturbation": 1,
            },
            "compound_lab": {
                "enabled": False,
                "jump": 2,
                "boundary_repeats": 1,
                "max_experiments_per_compound": 2,
                "max_compounds": 1,
            },
        }
        self.progress = None

    @staticmethod
    def _utc():
        return "x"


def _fake_executor_for_budgets(calls):
    def execute(runner, fixture, spec, *, parent=None):
        calls.append(
            {
                "family": spec.task_family,
                "budget": spec.generation_budget,
                "changed_variable": spec.changed_variable,
                "parent": None if parent is None else parent.experiment_id,
            }
        )
        if spec.generation_budget == 256:
            result_class, valid = "THINK_TRUNCATED", False
        elif spec.generation_budget == 512:
            result_class, valid = "ANSWER_TRUNCATED", False
        else:
            result_class, valid = "ANSWER_CORRECT", True
        return {
            "experiment": spec.to_dict(),
            "classification": {
                "result_class": result_class,
                "valid_for_capability": valid,
            },
            "score": 1.0 if valid else 0.0,
            "status": "PASS" if valid else "FAIL",
            "metrics": {"eval_count": spec.generation_budget},
            "timing": {"client_latency_ns": 1_000_000_000},
            "phase_metrics": {},
            "evidence_refs": {},
        }

    return execute


def test_truncation_retries_same_fixture_at_next_budget_and_resets_for_next_family(monkeypatch):
    import compute_cost.capability_campaign as campaign

    calls = []
    monkeypatch.setattr(campaign, "execute_experiment", _fake_executor_for_budgets(calls))
    runner = _Runner()

    rows_a, seq = campaign.run_family_frontier(runner, "math", {2: _case(2, "math")})
    rows_b, _ = campaign.run_family_frontier(
        runner,
        "logic",
        {2: _case(2, "logic")},
        sequence_start=seq,
    )

    assert [row["budget"] for row in calls] == [256, 512, 1024, 256, 512, 1024]
    assert [row["family"] for row in calls] == ["math", "math", "math", "logic", "logic", "logic"]
    assert calls[0]["changed_variable"] == "baseline"
    assert calls[1]["changed_variable"] == "generation_budget"
    assert calls[2]["changed_variable"] == "generation_budget"
    assert len(rows_a) == 3
    assert len(rows_b) == 3


def test_wrong_answer_never_raises_token_budget(monkeypatch):
    import compute_cost.capability_campaign as campaign

    calls = []

    def execute(runner, fixture, spec, *, parent=None):
        calls.append(spec.generation_budget)
        return {
            "experiment": spec.to_dict(),
            "classification": {
                "result_class": "ANSWER_WRONG",
                "valid_for_capability": True,
            },
            "score": 0.0,
            "status": "FAIL",
            "metrics": {"eval_count": 8},
            "timing": {},
            "phase_metrics": {},
            "evidence_refs": {},
        }

    monkeypatch.setattr(campaign, "execute_experiment", execute)
    campaign.run_family_frontier(_Runner(), "math", {2: _case()})

    assert calls == [256]


def test_model_reasoning_control_is_model_specific():
    from compute_cost.capability_campaign import resolve_model_reasoning_control

    cfg = {"thinking_mode": False, "reasoning_effort": None}
    assert resolve_model_reasoning_control("qwen3.5:35b-a3b-q4_K_M", cfg) == (False, None)
    assert resolve_model_reasoning_control("devstral-small-2:24b-instruct-2512-q8_0", cfg) == (False, None)
    assert resolve_model_reasoning_control("gpt-oss:20b", cfg) == (True, "low")


def test_disabled_secondary_labs_make_zero_calls(monkeypatch):
    import compute_cost.capability_campaign as campaign

    runner = _Runner()
    family = "math"
    base_row = {
        "experiment": {
            "experiment_id": "base",
            "task_id": family,
            "task_family": family,
            "difficulty_level": 2,
            "reasoning_effort": None,
            "generation_budget": 256,
        },
        "classification": {
            "result_class": "ANSWER_CORRECT",
            "valid_for_capability": True,
        },
        "score": 1.0,
        "status": "PASS",
    }
    monkeypatch.setattr(
        campaign,
        "run_family_frontier",
        lambda runner, family_id, ladder, sequence_start=0: ([base_row], sequence_start + 1),
    )
    monkeypatch.setattr(
        campaign,
        "_baseline_frontiers",
        lambda runner, rows: {
            "taxonomy_version": "capability-taxonomy-v1",
            "families": {family: {"levels": [], "reliable_floor": 2}},
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled lab was executed")

    monkeypatch.setattr(campaign, "run_reasoning_curves", forbidden)
    monkeypatch.setattr(campaign, "run_recovery_lab", forbidden)
    monkeypatch.setattr(campaign, "run_robustness_lab", forbidden)
    monkeypatch.setattr(campaign, "run_compound_lab", forbidden)

    rows = campaign.run_capability_campaign(runner, [_case()])
    assert rows == [base_row]


def test_compact_scorecard_exposes_numeric_family_score_levels_failures_and_budget_curve(tmp_path: Path):
    from compute_cost.compact_scorecard import build_compact_scorecard

    rows = [
        {
            "experiment": {
                "experiment_id": "a",
                "task_family": "math",
                "difficulty_level": 2,
                "generation_budget": 256,
                "thinking_mode": False,
                "reasoning_effort": None,
            },
            "classification": {
                "result_class": "THINK_TRUNCATED",
                "valid_for_capability": False,
            },
            "score": 0.0,
            "metrics": {"eval_count": 256},
            "timing": {"client_latency_ns": 2_000_000_000},
        },
        {
            "experiment": {
                "experiment_id": "b",
                "task_family": "math",
                "difficulty_level": 2,
                "generation_budget": 512,
                "thinking_mode": False,
                "reasoning_effort": None,
            },
            "classification": {
                "result_class": "ANSWER_CORRECT",
                "valid_for_capability": True,
            },
            "score": 1.0,
            "metrics": {"eval_count": 42},
            "timing": {"client_latency_ns": 1_000_000_000},
        },
        {
            "experiment": {
                "experiment_id": "c",
                "task_family": "math",
                "difficulty_level": 5,
                "generation_budget": 256,
                "thinking_mode": False,
                "reasoning_effort": None,
            },
            "classification": {
                "result_class": "ANSWER_WRONG",
                "valid_for_capability": True,
            },
            "score": 0.0,
            "metrics": {"eval_count": 15},
            "timing": {"client_latency_ns": 1_500_000_000},
        },
    ]
    frontiers = {
        "taxonomy_version": "capability-taxonomy-v1",
        "families": {
            "math": {
                "levels": [
                    {
                        "level": 2,
                        "observation_count": 2,
                        "valid_count": 1,
                        "invalid_count": 1,
                        "pass_count": 1,
                        "fail_count": 0,
                        "pass_rate": 1.0,
                        "label": "reliable",
                    },
                    {
                        "level": 5,
                        "observation_count": 1,
                        "valid_count": 1,
                        "invalid_count": 0,
                        "pass_count": 0,
                        "fail_count": 1,
                        "pass_rate": 0.0,
                        "label": "failure",
                    },
                ],
                "reliable_floor": 2,
                "first_failure_level": 5,
                "coverage": {"tested_levels": [2, 5], "untested_levels": [0, 1, 3, 4, 6, 7, 8, 9, 10]},
            }
        },
    }

    scorecard = build_compact_scorecard("fake", rows, frontiers)
    math = scorecard["families"]["math"]

    assert math["score_percent"] == 50.0
    assert math["valid_observations"] == 2
    assert math["invalid_observations"] == 1
    assert math["correct"] == 1
    assert math["wrong"] == 1
    assert math["levels"]["2"]["score_percent"] == 100.0
    assert math["levels"]["5"]["score_percent"] == 0.0
    assert math["result_classes"] == {
        "ANSWER_CORRECT": 1,
        "ANSWER_WRONG": 1,
        "THINK_TRUNCATED": 1,
    }
    assert math["token_budget_curve"]["256"]["attempts"] == 2
    assert math["token_budget_curve"]["512"]["correct"] == 1
    assert math["minimum_observed_correct_budget"] == 512
    assert math["reliable_floor"] == 2
    assert math["first_failure_level"] == 5
    assert scorecard["overall"]["score_percent"] == 50.0
