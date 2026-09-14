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
    case = partition_cases(cases)["VALIDATION"][0]
    family = case["category"]
    handoff["generation_budget_by_family"] = {family: 1024}
    campaign = Test2Campaign(runner, cases, handoff)
    before = dict(campaign.generation_budget_by_family)

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
    assert campaign.generation_budget_by_family == before
    assert (campaign.cfg.get("generation_budget_by_family") or {}) == {}


def test_test12_exact_source_recipes_preserve_semantic_hash():
    intervention = {
        "id": "CTRL-EXACT",
        "category": "PROMPT_CONTROL",
        "mode": "single",
        "instruction": "Preserve this exact instruction.",
    }
    semantic_hash = test2_module._intervention_fingerprint(intervention)
    handoff = {
        "handoff_mode": "TEST12_EXACT",
        "test12_exact_controls": [{
            "intervention_id": "CTRL-EXACT",
            "exact_test12_intervention": intervention,
            "discovery_semantic_hash": semantic_hash,
            "proof_semantic_hash": semantic_hash,
            "ingredient_ids": ["TEST12:CTRL-EXACT"],
        }],
    }

    recipes = source_recipes(handoff)
    assert len(recipes) == 1
    assert recipes[0]["exact_test12_intervention"] == intervention
    assert recipes[0]["proof_semantic_hash"] == semantic_hash

    mutated = json.loads(json.dumps(handoff))
    mutated["test12_exact_controls"][0]["exact_test12_intervention"]["instruction"] = "Changed"
    with pytest.raises(ValueError, match="semantic drift"):
        source_recipes(mutated)


def test_exact_policy_finalization_requires_blind_and_harm_proof():
    intervention = {
        "id": "CTRL-EXACT",
        "category": "PROMPT_CONTROL",
        "mode": "single",
        "instruction": "Exact",
    }
    semantic_hash = test2_module._intervention_fingerprint(intervention)
    recipe = {
        "intervention_id": "CTRL-EXACT",
        "exact_test12_intervention": intervention,
        "discovery_semantic_hash": semantic_hash,
        "proof_semantic_hash": semantic_hash,
        "ingredient_ids": ["TEST12:CTRL-EXACT"],
    }

    class Campaign:
        recipes = [recipe]

    policy = {
        "policy_id": "STATIC-CTRL-EXACT",
        "mode": "static",
        "intervention_id": "CTRL-EXACT",
        "intervention": intervention,
    }
    policy_lock = test2_module.hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    blind_key = "POLICY-" + policy_lock[:16]
    blind = {
        "locked_policy_proved_exactly": True,
        "locked_policy": policy,
        "policy_lock_sha256": policy_lock,
        "effects": {
            blind_key: {
                "classification": "PROMISING",
                "normalized_effect": 0.2,
            }
        },
    }
    harm = {
        test2_module._recipe_key(recipe): {
            "harm_evidence_sufficient": True,
            "harm_safe": True,
            "break_rate_wilson90": [0.0, 0.04],
        }
    }

    rows = _final_recipe_registry({}, blind, harm, campaign=Campaign())
    assert len(rows) == 1
    assert rows[0]["verified_for_shipping"] is True
    assert rows[0]["recipe"]["exact_locked_policy"] == policy
    assert rows[0]["semantic_translation_used"] is False

    harm[test2_module._recipe_key(recipe)]["harm_safe"] = False
    blocked = _final_recipe_registry({}, blind, harm, campaign=Campaign())
    assert blocked[0]["verified_for_shipping"] is False


def test_stopping_rule_2_uses_fresh_blind_cost_exchange():
    class Runner:
        model = "gpt-oss:20b"
        results_root = Path("__missing__")

    class Campaign:
        runner = Runner()
        cfg = {
            "harm_max_break_rate": 0.05,
            "policy_cost_ratio_ceiling": 1.25,
            "minimum_accuracy_advantage_when_over_cost_ceiling": 0.02,
        }
        harm_evidence = {}

    blind = {
        "effects": {
            "POLICY-x": {
                "normalized_effect": 0.1,
            }
        },
        "policy_cost_exchange": {
            "mean_policy_cost_ratio": 2.0,
            "accuracy_advantage": 0.0,
        },
    }
    limits = {"phenotypes": {}}

    result = test2_module._build_harness_stopping_rules(Campaign(), blind, limits)
    by_id = {row["id"]: row for row in result["rules"]}
    assert by_id["STOP-2"]["triggered"] is True
    assert result["stop_shipping_new_controls"] is True


def test_holdout_partition_is_consumed_once_across_runs(tmp_path):
    fixtures = [
        {"id": "blind-1"},
        {"id": "blind-2"},
    ]

    class Store:
        def __init__(self, root, run_id):
            self.run_id = run_id
            self.run_dir = root / run_id
            self.run_dir.mkdir()

        def write_json(self, name, payload, **kwargs):
            (self.run_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    class Runner:
        def __init__(self, root, run_id):
            self.results_root = root
            self.store = Store(root, run_id)

    class Campaign:
        def __init__(self, root, run_id):
            self.runner = Runner(root, run_id)
            self.cfg = {"holdout_max_cross_run_acceptance_uses": 1}

    first = Campaign(tmp_path, "cycle-1")
    claim = test2_module._claim_holdout_partition(first, "TEST2_BLIND", fixtures)
    assert claim["status"] == "CONSUMED_ON_EXPOSURE"
    assert claim["retire_permanently_after_cycle"] is True

    second = Campaign(tmp_path, "cycle-2")
    with pytest.raises(ValueError, match="already been consumed"):
        test2_module._claim_holdout_partition(second, "TEST2_BLIND", fixtures)



def test_mixed_invalid_failure_evidence_cannot_enter_finetuning():
    cases = [
        {
            "id": f"mix-{i}",
            "category": "reasoning",
            "family_id": "reasoning",
            "difficulty_level": 5,
            "prompt": f"q{i}",
            "scorer": "exact",
            "expected": "x",
        }
        for i in range(4)
    ]

    class Campaign:
        case_by_id = {case["id"]: case for case in cases}
        handoff = {"failures": {"failures": []}, "run_id": "source"}
        cfg = {"fine_tuning_min_independent_failures": 3}

        @staticmethod
        def _partition(case):
            return "VALIDATION"

    Campaign.rows = [
        {
            "fixture_id": case["id"],
            "family_id": "reasoning",
            "partition": "VALIDATION",
            "experiment_id": f"exp-{i}",
            "classification": {
                "result_class": "ANSWER_WRONG",
                "valid_for_capability": i < 3,
            },
            "valid_for_capability": i < 3,
        }
        for i, case in enumerate(cases)
    ]

    limits, queue, dataset = test2_module._build_model_limit_and_finetuning(
        Campaign(),
        {"matrix": {}},
        {"harm": {"boundary_class": "NEUTRAL"}},
    )
    phenotype = next(iter(limits["phenotypes"].values()))
    assert phenotype["valid_independent_fixture_count"] == 3
    assert phenotype["invalid_fixture_count"] == 1
    assert phenotype["owner"] == "SYSTEM_DISAMBIGUATION_REQUIRED"
    assert queue == []
    assert dataset == []



def test_effect_map_uses_fixture_as_unit_of_independence():
    recipe = {
        "ingredient_ids": ["ING-001"],
        "dose": 1.0,
        "representation": "prose",
        "placement": "prefix",
    }
    rows = []
    for seed in range(10):
        rows.append({
            "kind": "recurrence",
            "fixture_id": "same-fixture",
            "seed": seed,
            "recipe": recipe,
            "delta_valid": True,
            "delta": 1.0,
            "censored_for_capability": False,
        })
    rows.append({
        "kind": "recurrence",
        "fixture_id": "other-fixture",
        "seed": 42,
        "recipe": recipe,
        "delta_valid": True,
        "delta": -1.0,
        "censored_for_capability": False,
    })

    summary = _effect_map(
        rows,
        noise_sigma=0.1,
        bootstrap_samples=40,
        key_fn=lambda row: "CTRL",
    )["CTRL"]

    assert summary["raw_valid_n"] == 11
    assert summary["independent_fixture_n"] == 2
    assert summary["unit_of_independence"] == "fixture"
    assert summary["fixture_ids"] == ["other-fixture", "same-fixture"]


def test_effect_map_applies_bh_fdr_to_positive_promotions():
    recipe = {
        "ingredient_ids": ["ING-001"],
        "dose": 1.0,
        "representation": "prose",
        "placement": "prefix",
    }
    rows = []
    # Three independent all-positive fixtures are suggestive but the exact
    # one-sided sign-test p=0.125 cannot survive q<=0.10.
    for index in range(3):
        rows.append({
            "kind": "recurrence",
            "fixture_id": f"fixture-{index}",
            "recipe": recipe,
            "delta_valid": True,
            "delta": 1.0,
            "censored_for_capability": False,
        })

    summary = _effect_map(
        rows,
        noise_sigma=0.1,
        bootstrap_samples=40,
        fdr_level=0.10,
        key_fn=lambda row: "CTRL",
    )["CTRL"]

    assert summary["positive_sign_test_p_value"] == pytest.approx(0.125)
    assert summary["bh_fdr_q_value"] == pytest.approx(0.125)
    assert summary["classification"] == "UNCERTAIN_MULTIPLICITY"
    assert summary["classification_before_multiplicity"] in {"STRONG", "PROMISING"}



def test_exact_test12_handoff_preserves_failure_provenance(tmp_path: Path):
    cases = _cases()
    partitions = partition_cases(cases)
    source_case = partitions["DISCOVERY"][0]
    fixture_id = source_case["id"]
    family = source_case["category"]

    collection_id = "collection-source"
    collection = EvidenceStore(tmp_path, collection_id)
    intervention = {
        "id": "CTRL-EXACT",
        "category": "PROMPT_CONTROL",
        "mode": "single",
        "instruction": "Exact control",
    }
    collection.write_json(
        "full-control-candidate-registry.json",
        {"candidates": [intervention]},
        producer="test",
        stage="test",
    )
    collection.write_json(
        "runtime-characterization-profile.json",
        {
            "gate_passed": True,
            "profile_sha256": "profile-sha",
            "resolved_generation_budget_by_family": {family: 512},
        },
        producer="test",
        stage="test",
    )
    collection.write_json(
        "test1.2-handoff.json",
        {"schema_version": 1},
        producer="test",
        stage="test",
    )
    collection.write_json(
        "test1.2-opportunity-discovery-map.json",
        {"unresolved_failed_fixtures": [fixture_id]},
        producer="test",
        stage="test",
    )
    collection.write_json(
        "control-redundancy-map.json",
        {
            "schema_version": 1,
            "clusters": [],
            "representative_intervention_ids": [],
            "alternate_intervention_ids": [],
        },
        producer="test",
        stage="test",
    )
    collection.append_jsonl(
        "test1.2-observations.jsonl",
        {
            "fixture_id": fixture_id,
            "family_id": family,
            "difficulty_level": source_case["difficulty_level"],
            "partition": "DISCOVERY",
            "experiment_id": "baseline-exp",
            "intervention_id": "CONTROL",
            "score": 0.0,
            "valid_for_capability": True,
            "classification": {
                "result_class": "ANSWER_WRONG",
                "valid_for_capability": True,
            },
        },
    )
    collection.finalize_manifest(metadata={"mode": "test"})

    tuning_id = "tuning-source"
    tuning = EvidenceStore(tmp_path, tuning_id)
    tuning.write_json(
        "test1.2-terminal-handoff.json",
        {
            "state": "TEST1.2_PROVISIONAL_COMPILER_COMPLETE",
            "test2_blind_reserved_and_unexposed": True,
            "winner_lock_sha256": "winner-lock",
        },
        producer="test",
        stage="test",
    )
    tuning.write_json(
        "inverted-model-integration-package.json",
        {
            "release_authorized": False,
            "collection_run": collection_id,
        },
        producer="test",
        stage="test",
    )
    tuning.write_json(
        "compiled-harness-policy.json",
        {
            "winner_policy": {
                "policy_id": "STATIC-CTRL-EXACT",
                "mode": "static",
                "intervention_id": "CTRL-EXACT",
            }
        },
        producer="test",
        stage="test",
    )
    tuning.finalize_manifest(metadata={"mode": "test"})

    handoff = load_test1_handoff(tmp_path, tuning_id, cases)
    failures = handoff["failures"]["failures"]
    assert len(failures) == 1
    failure = failures[0]
    assert failure["fixture_id"] == fixture_id
    assert failure["family_id"] == family
    assert failure["difficulty_level"] == source_case["difficulty_level"]
    assert failure["classification"]["result_class"] == "ANSWER_WRONG"
    assert failure["valid_for_capability"] is True
    assert failure["source_experiment_id"] == "baseline-exp"
    assert failure["source_evidence_kind"] == "CAPABILITY_VALID_UNRESOLVED_BASELINE_FAILURE"



def test_test2_blind_failure_never_enters_finetuning_dataset():
    cases = [
        {
            "id": "valid-a",
            "category": "reasoning",
            "family_id": "reasoning",
            "difficulty_level": 5,
            "prompt": "a",
            "scorer": "exact",
            "expected": "x",
        },
        {
            "id": "valid-b",
            "category": "reasoning",
            "family_id": "reasoning",
            "difficulty_level": 5,
            "prompt": "b",
            "scorer": "exact",
            "expected": "x",
        },
        {
            "id": "blind-c",
            "category": "reasoning",
            "family_id": "reasoning",
            "difficulty_level": 5,
            "prompt": "c",
            "scorer": "exact",
            "expected": "x",
        },
    ]

    class Campaign:
        case_by_id = {case["id"]: case for case in cases}
        handoff = {"failures": {"failures": []}, "run_id": "source"}
        cfg = {"fine_tuning_min_independent_failures": 3}

        @staticmethod
        def _partition(case):
            return "TEST2_BLIND" if case["id"] == "blind-c" else "VALIDATION"

    Campaign.rows = [
        {
            "fixture_id": case["id"],
            "family_id": "reasoning",
            "partition": Campaign._partition(case),
            "experiment_id": f"exp-{case['id']}",
            "classification": {
                "result_class": "ANSWER_WRONG",
                "valid_for_capability": True,
            },
            "valid_for_capability": True,
        }
        for case in cases
    ]

    limits, queue, dataset = test2_module._build_model_limit_and_finetuning(
        Campaign(),
        {"matrix": {}},
        {"harm": {"boundary_class": "NEUTRAL"}},
    )

    assert queue == []
    assert dataset == []
    assert all(
        "blind-c" not in row.get("fixture_ids", [])
        for row in limits["phenotypes"].values()
    )
