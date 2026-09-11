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
