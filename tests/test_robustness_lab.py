from compute_cost.experiments import ExperimentSpec


def spec(
    experiment_id,
    *,
    family="formal_logic_deduction",
    level=4,
    effort="medium",
    seed=42,
    prompt_variant="base",
    recovery_level=None,
):
    return ExperimentSpec(
        experiment_id=experiment_id,
        parent_experiment_id=None,
        task_id=family,
        task_family=family,
        difficulty_level=level,
        hypothesis="evidence",
        changed_variable="baseline",
        thinking_mode=True,
        reasoning_effort=effort,
        generation_budget=256,
        context_request=None,
        temperature=0.0,
        seed=seed,
        prompt_variant=prompt_variant,
        recovery_level=recovery_level,
    )


def row(experiment, result_class="ANSWER_CORRECT"):
    return {
        "experiment": experiment.to_dict(),
        "classification": {
            "result_class": result_class,
            "valid_for_capability": True,
        },
        "score": 1.0 if result_class == "ANSWER_CORRECT" else 0.0,
        "status": "PASS" if result_class == "ANSWER_CORRECT" else "FAIL",
        "metrics": {"total_duration_ns": 100, "eval_count": 10},
    }


def fixture(family="formal_logic_deduction", level=4):
    return {
        "id": f"{family}-L{level}",
        "family_id": family,
        "category": family,
        "difficulty_level": level,
        "prompt": "Use the supplied premises and return exactly the requested symbol.",
        "scorer": "exact",
        "expected": "EXPECTED_SECRET",
        "robustness_eligible": True,
    }


def frontiers():
    return {
        "families": {
            "formal_logic_deduction": {
                "reliable_floor": 4,
                "first_failure_level": 5,
            },
            "strict_structured_output": {
                "reliable_floor": 4,
                "first_failure_level": 5,
            },
        }
    }


def test_robustness_target_prefers_proven_recovery_over_raw_floor():
    from compute_cost.robustness_lab import select_robustness_targets

    recovery_map = {
        "families": {
            "formal_logic_deduction": {
                "difficulty_level": 5,
                "minimum_successful_recovery": "R4",
            },
            "strict_structured_output": {
                "difficulty_level": 5,
                "minimum_successful_recovery": None,
            },
        }
    }
    selected = select_robustness_targets(frontiers(), recovery_map)

    assert selected["formal_logic_deduction"] == {
        "level": 5,
        "source": "RECOVERY_R4",
        "recovery_level": "R4",
    }
    assert selected["strict_structured_output"] == {
        "level": 4,
        "source": "BASELINE_RELIABLE_FLOOR",
        "recovery_level": None,
    }


def test_perturbation_plan_is_family_local_bounded_and_oracle_free():
    from compute_cost.robustness_lab import perturbation_plan, perturbation_suffix

    assert perturbation_plan("strict_structured_output", max_perturbations=2) == [
        "prompt_wording",
        "format_pressure",
    ]
    assert perturbation_plan("distractor_noise_resistance", max_perturbations=2) == [
        "prompt_wording",
        "distractor_noise",
    ]
    assert perturbation_plan("formal_logic_deduction", max_perturbations=2) == [
        "prompt_wording",
        "seed",
    ]

    for kind in ("prompt_wording", "format_pressure", "distractor_noise"):
        suffix = perturbation_suffix(kind)
        assert "EXPECTED_SECRET" not in suffix
        assert "answer" in suffix.lower() or "task" in suffix.lower() or "context" in suffix.lower()


class Store:
    def __init__(self):
        self.rows = {}

    def append_jsonl(self, path, value):
        self.rows.setdefault(path, []).append(value)


class Runner:
    def __init__(self):
        self.store = Store()
        self.model = "gpt-oss:20b"
        self.progress = None
        self.config = {
            "capability_campaign": {"reliable_threshold": 0.90},
            "robustness_lab": {
                "enabled": True,
                "repeats": 2,
                "max_perturbations_per_family": 2,
                "max_attempts_per_perturbation": 4,
            },
        }

    def _utc(self):
        return "x"


def test_robustness_branches_each_dimension_from_same_target_and_stops_valid_failure_early(monkeypatch):
    import compute_cost.robustness_lab as robustness

    executed = []
    outcomes = {
        "prompt_wording": ["ANSWER_WRONG"],
        "seed": ["ANSWER_CORRECT", "ANSWER_CORRECT"],
    }

    def execute(runner, case, experiment, *, parent=None):
        if experiment.changed_variable == "seed":
            kind = "seed"
        elif "prompt_wording" in experiment.prompt_variant:
            kind = "prompt_wording"
        else:
            kind = "seed"
        result_class = outcomes[kind].pop(0)
        executed.append({
            "kind": kind,
            "parent": parent.experiment_id,
            "changed_variable": experiment.changed_variable,
            "prompt_variant": experiment.prompt_variant,
            "seed": experiment.seed,
            "prompt": case["prompt"],
        })
        return row(experiment, result_class)

    monkeypatch.setattr(robustness, "execute_experiment", execute)
    base = [row(spec("base-pass"))]

    generated, robustness_map, _ = robustness.run_robustness_lab(
        Runner(),
        [fixture()],
        base,
        [],
        [],
        {"families": {"formal_logic_deduction": {"reliable_floor": 4, "first_failure_level": 5}}},
        {"families": {}},
        sequence_start=10,
    )

    assert [item["kind"] for item in executed] == ["prompt_wording", "seed", "seed"]
    assert executed[0]["parent"] == "base-pass"
    assert executed[0]["changed_variable"] == "prompt_variant"
    assert executed[1]["parent"] == "base-pass"
    assert executed[1]["changed_variable"] == "seed"
    assert executed[2]["changed_variable"] == "replication"
    assert executed[1]["seed"] != 42
    assert "EXPECTED_SECRET" not in executed[0]["prompt"]
    assert len(generated) == 3

    family = robustness_map["families"]["formal_logic_deduction"]
    assert family["target"]["source"] == "BASELINE_RELIABLE_FLOOR"
    assert [item["perturbation"] for item in family["perturbations"]] == [
        "prompt_wording",
        "seed",
    ]
    assert family["perturbations"][0]["status"] == "FAILED"
    assert family["perturbations"][1]["status"] == "SUCCESS"
    assert family["robustness"] == "FRAGILE"


def test_robustness_of_prompt_recovery_uses_recovered_parent_and_reconstructs_recovery_instruction(monkeypatch):
    import compute_cost.robustness_lab as robustness

    executed = []

    def execute(runner, case, experiment, *, parent=None):
        executed.append({
            "parent": parent.experiment_id,
            "changed_variable": experiment.changed_variable,
            "prompt": case["prompt"],
        })
        return row(experiment, "ANSWER_WRONG")

    monkeypatch.setattr(robustness, "execute_experiment", execute)
    recovery_parent = spec(
        "r4-pass",
        level=5,
        prompt_variant="recovery-r4",
        recovery_level="R4",
    )
    recovery_rows = [row(recovery_parent)]
    recovery_map = {
        "families": {
            "formal_logic_deduction": {
                "difficulty_level": 5,
                "minimum_successful_recovery": "R4",
                "steps": [
                    {
                        "level": "R4",
                        "status": "SUCCESS",
                        "intervention": {
                            "kind": "prompt_variant",
                            "suffix": "A prior attempt had the non-causal failure signature `logic_error`. Re-check it.",
                        },
                    }
                ],
            }
        }
    }

    generated, robustness_map, _ = robustness.run_robustness_lab(
        Runner(),
        [fixture(level=5)],
        [],
        [],
        recovery_rows,
        {"families": {"formal_logic_deduction": {"reliable_floor": 4, "first_failure_level": 5}}},
        recovery_map,
        sequence_start=20,
    )

    assert len(generated) == 2
    assert all(item["parent"] == "r4-pass" for item in executed)
    assert "logic_error" in executed[0]["prompt"]
    assert robustness_map["families"]["formal_logic_deduction"]["target"] == {
        "level": 5,
        "source": "RECOVERY_R4",
        "recovery_level": "R4",
    }
