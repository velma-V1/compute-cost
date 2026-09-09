def case():
    return {
        "id": "formal_logic_deduction-L5",
        "family_id": "formal_logic_deduction",
        "category": "formal_logic_deduction",
        "difficulty_level": 5,
        "prompt": "logic task",
        "scorer": "exact",
        "expected": "OK",
        "robustness_eligible": True,
    }


def experiment(experiment_id, *, recovery_level=None, prompt_variant="base"):
    return {
        "experiment_id": experiment_id,
        "parent_experiment_id": None,
        "task_id": "formal_logic_deduction",
        "task_family": "formal_logic_deduction",
        "difficulty_level": 5,
        "hypothesis": "test",
        "changed_variable": "baseline",
        "thinking_mode": True,
        "reasoning_effort": "medium",
        "generation_budget": 256,
        "context_request": None,
        "temperature": 0.0,
        "seed": 42,
        "prompt_variant": prompt_variant,
        "recovery_level": recovery_level,
    }


def row(experiment_id, result_class, *, recovery_level=None, prompt_variant="base"):
    return {
        "experiment": experiment(
            experiment_id,
            recovery_level=recovery_level,
            prompt_variant=prompt_variant,
        ),
        "classification": {
            "result_class": result_class,
            "valid_for_capability": True,
        },
        "score": 1.0 if result_class == "ANSWER_CORRECT" else 0.0,
        "status": "PASS" if result_class == "ANSWER_CORRECT" else "FAIL",
    }


class Store:
    def __init__(self):
        self.jsons = {}
        self.rows = {}

    def write_json(self, path, value, **kwargs):
        self.jsons[path] = value

    def append_jsonl(self, path, value):
        self.rows.setdefault(path, []).append(value)


class Runner:
    def __init__(self):
        self.model = "gpt-oss:20b"
        self.store = Store()
        self.suite = {"taxonomy_version": "capability-taxonomy-v1"}
        self.config = {
            "capability_campaign": {
                "reliable_threshold": 0.90,
                "unstable_threshold": 0.40,
            },
            "reasoning_curves": {"enabled": False, "repeats": 2},
            "recovery_lab": {"enabled": True, "repeats": 3, "max_level": "R7"},
            "robustness_lab": {
                "enabled": True,
                "repeats": 2,
                "max_perturbations_per_family": 2,
                "max_attempts_per_perturbation": 4,
            },
        }


def test_campaign_runs_robustness_after_recovery_and_rebuilds_final_failure_atlas(monkeypatch):
    import compute_cost.capability_campaign as campaign

    base_row = row("base-fail", "ANSWER_WRONG")
    recovery_row = row(
        "r4-pass",
        "ANSWER_CORRECT",
        recovery_level="R4",
        prompt_variant="recovery-r4",
    )
    robustness_row = row(
        "robust-fail",
        "ANSWER_WRONG",
        prompt_variant="recovery-r4+robust-prompt_wording",
    )
    frontiers = {
        "families": {
            "formal_logic_deduction": {
                "reliable_floor": 4,
                "first_failure_level": 5,
            }
        }
    }
    recovery_map = {
        "schema_version": 1,
        "families": {
            "formal_logic_deduction": {
                "difficulty_level": 5,
                "minimum_successful_recovery": "R4",
            }
        },
    }
    captured = {}

    monkeypatch.setattr(
        campaign,
        "run_family_frontier",
        lambda runner, family_id, ladder, sequence_start=0: ([base_row], 1),
    )
    monkeypatch.setattr(campaign, "_baseline_frontiers", lambda runner, rows: frontiers)

    def fake_recovery(
        runner,
        cases,
        base_rows,
        reasoning_rows,
        frontier_arg,
        failure_atlas,
        *,
        sequence_start=0,
    ):
        return [recovery_row], recovery_map, sequence_start + 1

    def fake_robustness(
        runner,
        cases,
        base_rows,
        reasoning_rows,
        recovery_rows,
        frontier_arg,
        recovery_map_arg,
        *,
        sequence_start=0,
    ):
        captured["base_rows"] = list(base_rows)
        captured["reasoning_rows"] = list(reasoning_rows)
        captured["recovery_rows"] = list(recovery_rows)
        captured["frontiers"] = frontier_arg
        captured["recovery_map"] = recovery_map_arg
        captured["sequence_start"] = sequence_start
        return [robustness_row], {
            "schema_version": 1,
            "families": {"formal_logic_deduction": {"robustness": "FRAGILE"}},
        }, sequence_start + 1

    monkeypatch.setattr(campaign, "run_recovery_lab", fake_recovery)
    monkeypatch.setattr(campaign, "run_robustness_lab", fake_robustness, raising=False)

    runner = Runner()
    rows = campaign.run_capability_campaign(runner, [case()])

    assert captured["base_rows"] == [base_row]
    assert captured["reasoning_rows"] == []
    assert captured["recovery_rows"] == [recovery_row]
    assert captured["frontiers"] is frontiers
    assert captured["recovery_map"] is recovery_map
    assert captured["sequence_start"] == 2
    assert runner.store.jsons["recovery-map.json"] is recovery_map
    assert runner.store.jsons["robustness-map.json"]["families"]["formal_logic_deduction"]["robustness"] == "FRAGILE"
    assert [item["experiment"]["experiment_id"] for item in rows] == [
        "base-fail",
        "r4-pass",
        "robust-fail",
    ]
    assert runner.store.jsons["failure-atlas.json"]["summary"]["model_failures"] == 2
