from compute_cost.experiments import ExperimentSpec


def _spec(experiment_id: str, *, effort: str = "medium", recovery_level=None) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=experiment_id,
        parent_experiment_id=None,
        task_id="formal_logic_deduction",
        task_family="formal_logic_deduction",
        difficulty_level=5,
        hypothesis="baseline",
        changed_variable="baseline",
        thinking_mode=True,
        reasoning_effort=effort,
        generation_budget=256,
        context_request=None,
        temperature=0.0,
        seed=42,
        prompt_variant="base",
        recovery_level=recovery_level,
    )


def _row(experiment_id: str, result_class: str, *, valid: bool, effort: str = "medium") -> dict:
    return {
        "experiment": _spec(experiment_id, effort=effort).to_dict(),
        "classification": {
            "result_class": result_class,
            "valid_for_capability": valid,
        },
        "score": 1.0 if result_class == "ANSWER_CORRECT" else 0.0,
        "status": "PASS" if result_class == "ANSWER_CORRECT" else "FAIL",
    }


def _fixture() -> dict:
    return {
        "id": "logic-L5",
        "family_id": "formal_logic_deduction",
        "category": "formal_logic_deduction",
        "difficulty_level": 5,
        "prompt": "If A implies B and A is true, return exactly B.",
        "scorer": "exact",
        "expected": "B",
    }


def _frontier() -> dict:
    return {
        "schema_version": 1,
        "families": {
            "formal_logic_deduction": {
                "reliable_floor": 4,
                "first_failure_level": 5,
                "levels": [
                    {"level": 4, "label": "reliable", "valid_count": 3, "pass_count": 3, "fail_count": 0},
                    {"level": 5, "label": "failure", "valid_count": 3, "pass_count": 0, "fail_count": 3},
                ],
            }
        },
    }


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
            "capability_campaign": {
                "generation_budget": 256,
                "reliable_threshold": 0.90,
                "boundary_repeats": 3,
            },
            "recovery_lab": {
                "enabled": True,
                "repeats": 3,
                "max_level": "R7",
                "max_attempts_per_candidate": 4,
            },
        }

    def _utc(self):
        return "x"


def test_recovery_stops_inconclusive_candidate_at_attempt_cap_and_marks_uncertain(monkeypatch):
    import compute_cost.recovery_lab as recovery

    executed = []

    def execute(runner, case, experiment, *, parent=None):
        if experiment.recovery_level != "R3":
            raise AssertionError(f"unexpected recovery escalation to {experiment.recovery_level}")
        executed.append(experiment.experiment_id)
        if len(executed) > 4:
            raise AssertionError("recovery exceeded max_attempts_per_candidate")
        if len(executed) == 1:
            result_class, valid = "ANSWER_CORRECT", True
        else:
            result_class, valid = "RUNTIME_FAILURE", False
        return {
            "experiment": experiment.to_dict(),
            "classification": {
                "result_class": result_class,
                "valid_for_capability": valid,
            },
            "score": 1.0 if result_class == "ANSWER_CORRECT" else 0.0,
            "status": "PASS" if result_class == "ANSWER_CORRECT" else "RUNTIME_ERROR",
        }

    monkeypatch.setattr(recovery, "execute_experiment", execute)
    base = [_row(f"base-{i}", "ANSWER_WRONG", valid=True) for i in range(3)]
    high_fail = [_row(f"high-{i}", "ANSWER_WRONG", valid=True, effort="high") for i in range(3)]

    generated, recovery_map, _ = recovery.run_recovery_lab(
        Runner(),
        [_fixture()],
        base,
        high_fail,
        _frontier(),
        failure_atlas={
            "failures": [
                {
                    "family_id": "formal_logic_deduction",
                    "difficulty_level": 5,
                    "failure_signature": {"subtype": "logic_error", "causal_claim": False},
                }
            ]
        },
    )

    assert len(generated) == 4
    assert len(executed) == 4
    family = recovery_map["families"]["formal_logic_deduction"]
    assert [step["level"] for step in family["steps"]] == ["R0", "R1", "R2", "R3"]
    assert family["steps"][-1]["status"] == "UNCERTAIN"
    assert family["minimum_successful_recovery"] is None
    assert family["recovered"] is False
