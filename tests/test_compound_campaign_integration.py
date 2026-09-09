def case():
    return {
        "id": "formal_logic_deduction-L4",
        "family_id": "formal_logic_deduction",
        "category": "formal_logic_deduction",
        "difficulty_level": 4,
        "prompt": "logic task",
        "scorer": "exact",
        "expected": "OK",
    }


def row(experiment_id, result_class, *, family="formal_logic_deduction", level=4):
    return {
        "experiment": {
            "experiment_id": experiment_id,
            "parent_experiment_id": None,
            "task_id": family,
            "task_family": family,
            "difficulty_level": level,
            "hypothesis": "test",
            "changed_variable": "baseline",
            "thinking_mode": True,
            "reasoning_effort": "medium",
            "generation_budget": 256,
            "context_request": None,
            "temperature": 0.0,
            "seed": 42,
            "prompt_variant": "base",
            "recovery_level": None,
        },
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
            "compound_lab": {
                "enabled": True,
                "jump": 2,
                "boundary_repeats": 3,
                "max_experiments_per_compound": 16,
                "max_compounds": 6,
            },
        }


def test_campaign_runs_compound_after_prior_phases_and_rebuilds_final_failure_atlas(monkeypatch):
    import compute_cost.capability_campaign as campaign

    base_row = row("base-pass", "ANSWER_CORRECT")
    compound_row = row(
        "compound-fail",
        "ANSWER_WRONG",
        family="extract_calculate_json",
        level=5,
    )
    frontiers = {
        "families": {
            "extraction_transformation": {"reliable_floor": 7},
            "arithmetic_numerical_reasoning": {"reliable_floor": 7},
            "strict_structured_output": {"reliable_floor": 7},
        }
    }
    compound_map = {
        "schema_version": 1,
        "compounds": {
            "extract_calculate_json": {
                "expected_component_frontier": 7,
                "observed_compound_frontier": 4,
                "composition_penalty": -3,
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

    def fake_compound(runner, frontier_arg, *, sequence_start=0):
        captured["frontiers"] = frontier_arg
        captured["sequence_start"] = sequence_start
        return [compound_row], compound_map, sequence_start + 1

    monkeypatch.setattr(campaign, "run_compound_lab", fake_compound, raising=False)

    runner = Runner()
    rows = campaign.run_capability_campaign(runner, [case()])

    assert captured["frontiers"] is frontiers
    assert captured["sequence_start"] == 1
    assert runner.store.jsons["compound-map.json"] is compound_map
    assert [item["experiment"]["experiment_id"] for item in rows] == [
        "base-pass",
        "compound-fail",
    ]
    assert runner.store.jsons["failure-atlas.json"]["summary"]["model_failures"] == 1
