from __future__ import annotations

import json
from pathlib import Path

import pytest

from compute_cost.config import load_config
from compute_cost.runner import BenchmarkRunner
from compute_cost.test1_campaign import ACTIVE_SECONDS, HARD_SECONDS, partition_cases
from compute_cost.test11_campaign import (
    NEW_SEED_INGREDIENTS,
    Test11Campaign,
    _placebo_recipe,
    _residual_failure_ownership,
    _run_recipe_reserve,
    binary_effect_summary,
    build_ingredient_bank,
    build_recipe_variants,
    build_test11_plan,
    synthetic_test1_source,
    validate_test11_plan,
)


def _cases(count: int = 440) -> list[dict]:
    return [
        {
            "id": f"fixture-{index:04d}",
            "family_id": f"family-{index % 10}",
            "category": f"family-{index % 10}",
            "difficulty_level": index % 11,
            "prompt": f"Return {index}",
            "scorer": "exact",
            "expected": str(index),
            "timeout_s": 120,
        }
        for index in range(count)
    ]


def test_test11_plan_is_exactly_seven_hours_and_uses_recipe_sink():
    plan = build_test11_plan(_cases(), test1_run="test1-source")
    validate_test11_plan(plan)

    assert plan["wall_clock_seconds"] == HARD_SECONDS == 7 * 3600
    assert plan["active_model_seconds"] == ACTIVE_SECONDS == 6 * 3600 + 50 * 60
    assert sum(row["seconds"] for row in plan["phases"]) == ACTIVE_SECONDS
    assert plan["unused_time_sink"] == "MORE_RECIPE_TESTS"
    assert set(plan["prohibited_partitions"]) == {"TEST2_BLIND", "TEST3_PROTECTED"}
    assert plan["seed_ingredient_count"] >= 80

def test_test11_plan_contains_decision_useful_outputs_not_frontier_fantasy():
    plan = build_test11_plan(_cases(), test1_run="test1-source")
    outputs = set(plan["required_outputs"])

    assert "prompt-overhead-placebo-map.json" in outputs
    assert "capability-value-per-cost.json" in outputs
    assert "hard-case-leverage-map.json" in outputs
    assert "residual-failure-ownership.json" in outputs
    assert "fine-tuning-readiness-map.json" in outputs
    assert "frontier-gap-map.json" not in outputs
    assert all("frontier" not in name.lower() for name in outputs)


def test_placebo_recipe_is_explicitly_nonsemantic_and_length_matched():
    recipe = _placebo_recipe(256, placement="system")

    assert recipe["mode"] == "length_placebo"
    assert recipe["target_chars"] == 256
    assert len(recipe["placebo_text"]) == 256
    assert recipe["placement"] == "system"



def test_test11_expands_beyond_original_ingredient_bank():
    cases = _cases()
    source = synthetic_test1_source(cases)
    source["failures"] = [
        {
            "family_id": "tool_argument_correctness",
            "classification": {"result_class": "TOOL_FAILURE"},
        },
        {
            "family_id": "coding_generation",
            "classification": {"result_class": "THINK_TRUNCATED"},
        },
    ]
    bank = build_ingredient_bank(source)
    ids = {row["id"] for row in bank}

    assert len(NEW_SEED_INGREDIENTS) >= 64
    assert len(bank) > 80
    assert any(value.startswith("DYN-FAM-") for value in ids)
    assert any(value.startswith("DYN-CLS-") for value in ids)


def test_binary_effect_promotes_conditional_rescue_without_global_median_gain():
    cfg = load_config()["test11_campaign"]
    rows = []
    # Four baseline failures: two rescued, two unchanged.
    for index in range(4):
        rows.append({
            "kind": "ingredient_screen",
            "control_score": 0.0,
            "score": 1.0 if index < 2 else 0.0,
            "classification": {"result_class": "ANSWER_CORRECT" if index < 2 else "ANSWER_WRONG"},
        })
    # Twelve passing sentinels: no regressions.
    for _ in range(12):
        rows.append({
            "kind": "ingredient_screen",
            "control_score": 1.0,
            "score": 1.0,
            "classification": {"result_class": "ANSWER_CORRECT"},
        })

    summary = binary_effect_summary(rows, cfg)

    assert summary["rescue_rate"] == 0.5
    assert summary["capability_regression_rate"] == 0.0
    assert summary["classification"] in {
        "PROMISING_CONDITIONAL_RESCUE",
        "STRONG_CONDITIONAL_RESCUE",
    }


def test_truncation_is_not_classified_as_capability_harm():
    cfg = load_config()["test11_campaign"]
    rows = [
        {
            "kind": "ingredient_screen",
            "control_score": 1.0,
            "score": 0.0,
            "classification": {"result_class": "THINK_TRUNCATED"},
        }
    ]
    rows.extend([
        {
            "kind": "ingredient_screen",
            "control_score": 1.0,
            "score": 1.0,
            "classification": {"result_class": "ANSWER_CORRECT"},
        }
        for _ in range(11)
    ])

    summary = binary_effect_summary(rows, cfg)

    assert summary["truncation_regressions"] == 1
    assert summary["capability_regressions"] == 0
    assert summary["classification"] == "TRUNCATION_SENSITIVE"


def test_recipe_variants_cover_order_recurrence_knockout_and_factorial_dimensions():
    cfg = load_config()["test11_campaign"]
    recipes = build_recipe_variants(
        ["ING-017", "ING-018", "ING-019", "ING-020", "ING-021", "ING-022", "ING-023", "ING-024"],
        cfg,
    )

    modes = {recipe["mode"] for recipe in recipes}
    placements = {
        step["placement"]
        for recipe in recipes
        for step in recipe["steps"]
    }
    representations = {
        step["representation"]
        for recipe in recipes
        for step in recipe["steps"]
    }
    doses = {
        float(step["dose"])
        for recipe in recipes
        for step in recipe["steps"]
    }

    assert {"single", "ordered_recurrence", "triple", "dense", "knockout", "double_knockout"}.issubset(modes)
    assert {"system", "suffix", "prefix", "middle"}.issubset(placements)
    assert {"prose", "bullets", "schema"}.issubset(representations)
    assert {0.5, 1.0, 2.0}.issubset(doses)
    assert len(recipes) > 1000


class _ReserveCampaign:
    def __init__(self):
        self.calls = 0
        self.cfg = {"seeds": [42], "generation_budgets": [256]}
        self.bank_list = [{"id": "ING-017"}]
        self.partitions = {
            "DISCOVERY": _cases(1),
            "VALIDATION": [],
        }

    def can_start(self, deadline):
        return self.calls < 7

    def treatment(self, case, deadline, *, phase, recipe, budget, seed, kind):
        self.calls += 1
        return {"score": 1.0}


def test_recipe_reserve_repeats_instead_of_exhausting_queue_early():
    campaign = _ReserveCampaign()
    recipe = {
        "recipe_id": "REC",
        "mode": "single",
        "steps": [
            {
                "ingredient_id": "ING-017",
                "dose": 0.5,
                "representation": "prose",
                "placement": "system",
            }
        ],
    }
    attempted: set[str] = set()

    _run_recipe_reserve(campaign, 999999.0, [recipe], "reserve", attempted)

    assert campaign.calls == 7
    assert attempted == {"fixture-0000"}


def test_fine_tuning_requires_three_independent_unresolved_fixtures():
    cases = _cases(20)
    config = load_config()

    class _Runner:
        def __init__(self):
            self.config = config
            self.store = None
        def _utc(self):
            return "2026-09-11T00:00:00Z"

    campaign = Test11Campaign(_Runner(), cases, synthetic_test1_source(cases))
    target_cases = cases[:3]
    campaign.rows = []
    for case in target_cases:
        fixture_id = case["id"]
        family = "reasoning_family"
        campaign.rows.append({
            "kind": "control",
            "fixture_id": fixture_id,
            "family_id": family,
            "difficulty_level": 6,
            "score": 0.0,
            "control_score": 0.0,
            "generation_budget": 256,
            "classification": {"result_class": "ANSWER_WRONG"},
        })
        campaign.rows.append({
            "kind": "ingredient_screen",
            "fixture_id": fixture_id,
            "family_id": family,
            "difficulty_level": 6,
            "score": 0.0,
            "control_score": 0.0,
            "generation_budget": 256,
            "classification": {"result_class": "ANSWER_WRONG"},
        })

    ownership, readiness = _residual_failure_ownership(
        campaign,
        atlas={"REC": {"classification": "NO_RESCUE_SIGNAL"}},
        operators={"OP": {"classification": "NO_RESCUE_SIGNAL"}},
        capability_value={},
    )

    candidates = readiness["candidates"]
    assert len(candidates) == 1
    candidate = next(iter(candidates.values()))
    assert candidate["independent_unresolved_fixture_count"] == 3
    assert candidate["qualification"]["recurrent"] is True
    assert candidate["qualification"]["independent"] is True


def test_two_independent_failures_do_not_qualify_for_fine_tuning():
    cases = _cases(20)
    config = load_config()

    class _Runner:
        def __init__(self):
            self.config = config
            self.store = None
        def _utc(self):
            return "2026-09-11T00:00:00Z"

    campaign = Test11Campaign(_Runner(), cases, synthetic_test1_source(cases))
    campaign.rows = []
    for case in cases[:2]:
        campaign.rows.extend([
            {
                "kind": "control",
                "fixture_id": case["id"],
                "family_id": "reasoning_family",
                "difficulty_level": 6,
                "score": 0.0,
                "control_score": 0.0,
                "generation_budget": 256,
                "classification": {"result_class": "ANSWER_WRONG"},
            },
            {
                "kind": "ingredient_screen",
                "fixture_id": case["id"],
                "family_id": "reasoning_family",
                "difficulty_level": 6,
                "score": 0.0,
                "control_score": 0.0,
                "generation_budget": 256,
                "classification": {"result_class": "ANSWER_WRONG"},
            },
        ])

    _, readiness = _residual_failure_ownership(
        campaign,
        atlas={"REC": {"classification": "NO_RESCUE_SIGNAL"}},
        operators={"OP": {"classification": "NO_RESCUE_SIGNAL"}},
        capability_value={},
    )

    assert readiness["candidates"] == {}


class _Runtime:
    def __init__(self):
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("Test-1.1 dry run must not invoke the model")


class _Progress:
    def __init__(self, total: int):
        self.total_tasks = total
        self.done = 0

    def start(self, task: str):
        return None

    def start_live(self):
        return None

    def stop_live(self, newline: bool = True):
        return None

    def begin_task(self, task: str):
        return None

    def complete_task(self, task: str):
        self.done += 1

    def snapshot(self):
        return {"done": self.done, "total": self.total_tasks}

    def render(self):
        return ""


def test_test11_dry_run_uses_zero_calls_and_reports_expanded_bank(tmp_path: Path):
    runtime = _Runtime()
    config = load_config()
    config["telemetry"]["background"] = False
    runner = BenchmarkRunner(
        runtime,
        config,
        {"benchmark_version": "test11-dry-run", "cases": _cases()},
        results_root=tmp_path,
        hardware_collector=lambda: {},
        progress_factory=lambda total: _Progress(total),
    )

    run_dir = runner.gpt20b_test11("gpt-oss:20b", dry_run=True)

    assert runtime.calls == 0
    validation = json.loads((run_dir / "test1.1-dry-run-validation.json").read_text(encoding="utf-8"))
    assert validation["expanded_ingredient_count"] >= 80
    assert validation["unused_time_sink"] == "MORE_RECIPE_TESTS"
    assert validation["protected_partitions_exposed"] is False


def test_test11_refuses_protected_partitions(tmp_path: Path):
    cases = _cases()
    config = load_config()
    runner = BenchmarkRunner(
        _Runtime(),
        config,
        {"benchmark_version": "test11-protection", "cases": cases},
        results_root=tmp_path,
        hardware_collector=lambda: {},
        progress_factory=lambda total: _Progress(total),
    )
    runner.model = "gpt-oss:20b"
    runner.store = None
    campaign = Test11Campaign(runner, cases, synthetic_test1_source(cases))
    protected = partition_cases(cases)["TEST3_PROTECTED"][0]

    with pytest.raises(ValueError, match="protected partition"):
        campaign.assert_allowed(protected)


class _PreflightRuntime:
    @staticmethod
    def _exchange(parsed):
        return {
            "request": {"body_b64": ""},
            "raw_response_b64": "",
            "stream_events": [],
            "parsed": parsed,
        }

    def version(self):
        return self._exchange({"version": "test"})

    def list_models(self):
        return self._exchange({"models": [{"name": "gpt-oss:20b", "size": 1}]})

    def model_available_in(self, tags, model):
        return True

    def model_info(self, model):
        return self._exchange({"model": model})

    def generate(self, *args, **kwargs):
        raise AssertionError("campaign stub should prevent model invocation")


def test_test11_active_clock_starts_after_source_preflight(tmp_path: Path, monkeypatch):
    import compute_cost.runner as runner_module

    cases = _cases()
    config = load_config()
    config["telemetry"]["background"] = False
    clock = {"value": 10.0}
    observed = {}

    def fake_load_test1_source(results_root, run_id, source_cases):
        # Simulate a long source-integrity preflight before the active window.
        clock["value"] = 1234.5
        return synthetic_test1_source(source_cases)

    def fake_run_test11_campaign(runner, source_cases, *, test1_run, started_monotonic):
        observed["started_monotonic"] = started_monotonic
        observed["test1_run"] = test1_run
        return []

    monkeypatch.setattr(runner_module, "load_test1_source", fake_load_test1_source)
    monkeypatch.setattr(runner_module, "run_test11_campaign", fake_run_test11_campaign)
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: clock["value"])

    runner = BenchmarkRunner(
        _PreflightRuntime(),
        config,
        {"benchmark_version": "test11-clock", "cases": cases},
        results_root=tmp_path,
        hardware_collector=lambda: {},
        progress_factory=lambda total: _Progress(total),
    )

    run_dir = runner.gpt20b_test11(
        "gpt-oss:20b",
        test1_run="source-test1",
        dry_run=False,
    )

    assert observed["started_monotonic"] == 1234.5
    assert observed["test1_run"] == "source-test1"

    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_types = [row["type"] for row in events]
    assert event_types.index("PREFLIGHT_COMPLETE") < event_types.index("TEST11_ACTIVE_WINDOW_START")
    assert event_types.index("TEST11_ACTIVE_WINDOW_START") < event_types.index("TEST11_COMPLETE")
