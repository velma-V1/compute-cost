from __future__ import annotations

import json
from pathlib import Path

import pytest

import compute_cost.test2_campaign as test2_module

from compute_cost.config import load_config
from compute_cost.evidence import EvidenceStore
from compute_cost.runner import BenchmarkRunner
from compute_cost.test2_campaign import (
    ACTIVE_SECONDS,
    HARD_SECONDS,
    REQUIRED_OUTPUTS,
    Test2Campaign,
    _effect_map,
    _final_recipe_registry,
    build_test2_plan,
    load_test1_handoff,
    source_recipes,
    synthetic_test1_handoff,
    validate_test2_plan,
)
from compute_cost.test1_campaign import partition_cases


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


def test_test2_plan_is_exactly_seven_hours_and_complete():
    plan = build_test2_plan(_cases(), test1_run="test1-source")
    validate_test2_plan(plan)

    assert plan["wall_clock_seconds"] == HARD_SECONDS == 7 * 3600
    assert plan["active_model_seconds"] == ACTIVE_SECONDS == 6 * 3600 + 50 * 60
    assert sum(row["seconds"] for row in plan["phases"]) == ACTIVE_SECONDS
    assert plan["prohibited_partitions"] == ["TEST3_PROTECTED"]
    assert set(REQUIRED_OUTPUTS).issubset(set(plan["required_outputs"]))
    assert all(plan["finalization_contract"].values())


def test_synthetic_handoff_provides_build_time_contract_without_test1_results():
    cases = _cases()
    handoff = synthetic_test1_handoff(cases)
    recipes = source_recipes(handoff)

    assert handoff["synthetic"] is True
    assert recipes
    assert all(recipe["ingredient_ids"] for recipe in recipes)
    assert set(handoff["partitions"]) == {
        "DISCOVERY",
        "VALIDATION",
        "TEST2_BLIND",
        "TEST3_PROTECTED",
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _materialize_test1_handoff(root: Path, run_id: str, cases: list[dict]) -> Path:
    store = EvidenceStore(root, run_id)
    run = store.run_dir
    partitions = partition_cases(cases)
    partition_payload = {
        "schema_version": 1,
        "partitions": {
            name: [case["id"] for case in rows]
            for name, rows in partitions.items()
        },
    }
    store.write_json("test1-plan.json", {"schema_version": 1}, producer="test", stage="test")
    store.write_json("fixture-partitions.json", partition_payload, producer="test", stage="test")
    store.write_json("noise-model.json", {"global_noise_sigma": 0.1, "families": {}}, producer="test", stage="test")
    store.write_json("ingredient-registry.json", synthetic_test1_handoff(cases)["ingredient_registry"], producer="test", stage="test")
    store.write_json("pair-interaction-graph.json", {"edges": {}}, producer="test", stage="test")
    store.write_json("directional-order-graph.json", {"edges": {}}, producer="test", stage="test")
    store.write_json("higher-order-candidate-queue.json", {"candidates": []}, producer="test", stage="test")
    store.write_json("failure-registry.json", {"failures": []}, producer="test", stage="test")
    store.write_json("negative-effect-registry.json", {"effects": []}, producer="test", stage="test")
    store.write_json("uncertainty-ledger.json", {"unknowns": []}, producer="test", stage="test")
    store.write_json("test2-priority-queue.json", synthetic_test1_handoff(cases)["priority_queue"], producer="test", stage="test")
    store.finalize_manifest(metadata={"mode": "test"})
    return run


def test_test1_handoff_rejects_partition_drift(tmp_path: Path):
    cases = _cases()
    run = _materialize_test1_handoff(tmp_path, "source", cases)
    payload = json.loads((run / "fixture-partitions.json").read_text(encoding="utf-8"))
    moved = payload["partitions"]["TEST3_PROTECTED"].pop()
    payload["partitions"]["DISCOVERY"].append(moved)
    # Rebuild a valid manifest around the intentionally drifted partition so
    # this test reaches the semantic partition-drift guard rather than failing
    # earlier on integrity.
    store = EvidenceStore(tmp_path, "source")
    record = store.write_json("fixture-partitions.json", payload, producer="test", stage="test")
    assert record["sha256"]
    store.finalize_manifest(metadata={"mode": "test-drift"})

    with pytest.raises(ValueError, match="partition drift"):
        load_test1_handoff(tmp_path, "source", cases)


class _Runtime:
    def __init__(self):
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("Test-2 dry run must not invoke the model")


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


def test_runner_dry_run_uses_zero_calls_and_materializes_finalization_contract(tmp_path: Path):
    runtime = _Runtime()
    config = load_config()
    config["telemetry"]["background"] = False
    runner = BenchmarkRunner(
        runtime,
        config,
        {"benchmark_version": "test2-dry-run", "cases": _cases()},
        results_root=tmp_path,
        hardware_collector=lambda: {},
        progress_factory=lambda total: _Progress(total),
    )

    run_dir = runner.gpt20b_test2("gpt-oss:20b", dry_run=True)

    assert runtime.calls == 0
    assert (run_dir / "test2-plan.json").is_file()
    assert (run_dir / "synthetic-test1-handoff-template.json").is_file()
    validation = json.loads((run_dir / "test2-dry-run-validation.json").read_text(encoding="utf-8"))
    assert validation["test3_protected_exposed"] is False
    assert validation["finalization_contract_complete"] is True
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row["type"] == "TEST2_DRY_RUN_COMPLETE" for row in events)
    assert not any(row["type"] == "RUN_FAILED" for row in events)


def test_campaign_refuses_test3_protected_fixture(tmp_path: Path):
    cases = _cases()
    config = load_config()
    runner = BenchmarkRunner(
        _Runtime(),
        config,
        {"benchmark_version": "test2-protection", "cases": cases},
        results_root=tmp_path,
        hardware_collector=lambda: {},
        progress_factory=lambda total: _Progress(total),
    )
    runner.model = "gpt-oss:20b"
    handoff = synthetic_test1_handoff(cases)
    campaign = Test2Campaign(runner, cases, handoff)
    protected = partition_cases(cases)["TEST3_PROTECTED"][0]

    with pytest.raises(ValueError, match="Test-3 protected"):
        campaign._assert_partition_allowed(protected)



def test_test2_invalid_delta_persists_as_null_not_fake_zero():
    class Store:
        @staticmethod
        def append_jsonl(*args, **kwargs):
            return None

    class Runner:
        store = Store()

        @staticmethod
        def _utc():
            return "2026-09-14T00:00:00Z"

    campaign = Test2Campaign.__new__(Test2Campaign)
    campaign.runner = Runner()
    campaign.rows = []
    campaign.cfg = {"generation_budget": 512}
    campaign._partition = lambda case: "VALIDATION"

    row = campaign._record(
        {
            "id": "case-a",
            "category": "arithmetic_numerical_reasoning",
            "difficulty_level": 6,
        },
        {
            "score": None,
            "classification": {
                "result_class": "THINK_TRUNCATED",
                "valid_for_capability": False,
            },
            "experiment": {
                "experiment_id": "exp-a",
                "generation_budget": 512,
            },
        },
        phase="recurrence_higher_order",
        kind="recurrence",
        recipe={
            "ingredient_ids": ["ING-001"],
            "dose": 1.0,
            "representation": "prose",
            "placement": "prefix",
        },
        baseline_score=1.0,
        baseline_valid=True,
        baseline_budget=512,
        source_key="x",
    )
    assert row["delta_valid"] is False
    assert row["delta"] is None
    assert row["censored_for_capability"] is True
    assert row["censoring_class"] == "CONTROL_EXCEEDS_BASELINE_BUDGET"


def test_test2_effect_map_reports_informative_censoring_instead_of_null():
    recipe = {
        "ingredient_ids": ["ING-001"],
        "dose": 1.0,
        "representation": "prose",
        "placement": "prefix",
    }
    rows = [
        {
            "kind": "recurrence",
            "recipe": recipe,
            "delta_valid": False,
            "delta": None,
            "censored_for_capability": True,
        }
        for _ in range(4)
    ] + [{
        "kind": "recurrence",
        "recipe": recipe,
        "delta_valid": True,
        "delta": 1.0,
        "censored_for_capability": False,
    }]
    effects = _effect_map(
        rows,
        noise_sigma=0.1,
        bootstrap_samples=20,
        max_censoring_rate=0.20,
        key_fn=lambda row: "CTRL",
    )
    summary = effects["CTRL"]
    assert summary["raw_n"] == 5
    assert summary["valid_n"] == 1
    assert summary["censored_n"] == 4
    assert summary["censoring_rate"] == 0.8
    assert summary["classification"] == "CENSORING_DOMINATED"
    assert summary["requires_own_budget_cost_probe"] is True


def test_final_recipe_cannot_be_shipping_verified_without_harm_evidence():
    recipe = {
        "ingredient_ids": ["ING-001"],
        "dose": 1.0,
        "representation": "prose",
        "placement": "prefix",
    }
    key = "ING-001"
    knockouts = {
        key: {
            "minimal_recipe": recipe,
            "required_ingredients": ["ING-001"],
            "removable_ingredients": [],
        }
    }
    blind = {
        "effects": {
            key: {
                "classification": "STRONG",
                "normalized_effect": 1.0,
            }
        }
    }

    missing = _final_recipe_registry(knockouts, blind, {})
    assert missing[0]["verified_for_shipping"] is False

    safe = _final_recipe_registry(
        knockouts,
        blind,
        {
            key: {
                "harm_evidence_sufficient": True,
                "harm_safe": True,
                "break_rate": 0.0,
            }
        },
    )
    assert safe[0]["verified_for_shipping"] is True


def test_test2_control_does_not_mutate_resolved_family_budget(monkeypatch):
    class Store:
        @staticmethod
        def append_jsonl(*args, **kwargs):
            return None

    class Runner:
        def __init__(self):
            self.config = load_config()
            self.progress = None
            self.store = Store()
            self.model = "gpt-oss:20b"

        @staticmethod
        def _utc():
            return "2026-09-14T00:00:00Z"

    cases = _cases()
    runner = Runner()
    handoff = synthetic_test1_handoff(cases)
    campaign = Test2Campaign(runner, cases, handoff)
    case = campaign.partitions["VALIDATION"][0]
    family = case["category"]
    campaign.cfg["generation_budget_by_family"] = {family: 1024}
    before = dict(campaign.cfg["generation_budget_by_family"])

    def fake_execute(runner, case, spec, parent=None, messages_override=None):
        return {
            "score": 1.0,
            "classification": {
                "result_class": "ANSWER_CORRECT",
                "valid_for_capability": True,
            },
            "experiment": {
                "experiment_id": spec.experiment_id,
                "generation_budget": spec.generation_budget,
            },
            "timing": {},
            "evidence_refs": {},
        }

    monkeypatch.setattr(test2_module, "execute_experiment", fake_execute)
    result = campaign.control(
        case,
        campaign.clock() + 100.0,
        phase="unit",
        force=True,
    )

    assert result == 1.0
    assert campaign.current_baseline_budgets[case["id"]] == 1024
    assert campaign.cfg["generation_budget_by_family"] == before
