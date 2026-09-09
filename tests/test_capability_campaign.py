import importlib
import importlib.util


def _module():
    spec = importlib.util.find_spec("compute_cost.capability_campaign")
    assert spec is not None, "compute_cost.capability_campaign is not implemented"
    return importlib.import_module("compute_cost.capability_campaign")


def case(level: int, family: str = "math") -> dict:
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


class Store:
    def __init__(self):
        self.rows = {}
        self.jsons = {}

    def append_jsonl(self, path, row):
        self.rows.setdefault(path, []).append(row)

    def write_json(self, path, value, **kwargs):
        self.jsons[path] = value


class Runner:
    def __init__(self):
        self.store = Store()
        self.model = "fake"
        self.suite = {
            "benchmark_version": "test",
            "taxonomy_version": "capability-taxonomy-v1",
        }
        self.config = {
            "capability_campaign": {
                "anchor_level": 1,
                "jump": 3,
                "boundary_repeats": 2,
                "max_experiments_per_family": 12,
                "thinking_mode": True,
                "generation_budget": 256,
                "reliable_threshold": 0.90,
                "unstable_threshold": 0.40,
            }
        }
        self.progress = None

    def _utc(self):
        return "x"


def fake_executor(outcomes, calls):
    def execute(runner, fixture, spec, *, parent=None):
        level = int(fixture["difficulty_level"])
        calls.append({
            "level": level,
            "changed_variable": spec.changed_variable,
            "parent": None if parent is None else parent.experiment_id,
            "experiment_id": spec.experiment_id,
        })
        values = outcomes[level]
        if isinstance(values, list):
            result_class, valid = values.pop(0)
        else:
            result_class, valid = values
        return {
            "experiment": spec.to_dict(),
            "classification": {
                "result_class": result_class,
                "valid_for_capability": valid,
            },
            "score": 1.0 if result_class == "ANSWER_CORRECT" else 0.0,
            "status": "PASS" if result_class == "ANSWER_CORRECT" else "FAIL",
        }
    return execute


def test_family_frontier_jumps_bisects_then_replicates_adjacent_boundary(monkeypatch):
    module = _module()
    calls = []
    outcomes = {
        1: ("ANSWER_CORRECT", True),
        4: ("ANSWER_CORRECT", True),
        5: ("ANSWER_WRONG", True),
        7: ("ANSWER_WRONG", True),
    }
    monkeypatch.setattr(module, "execute_experiment", fake_executor(outcomes, calls))
    ladder = {level: case(level) for level in (1, 4, 5, 7)}

    rows, sequence = module.run_family_frontier(Runner(), "math", ladder)

    assert [item["level"] for item in calls] == [1, 4, 7, 5, 4, 5]
    assert [item["changed_variable"] for item in calls] == [
        "baseline",
        "difficulty_level",
        "difficulty_level",
        "difficulty_level",
        "replication",
        "replication",
    ]
    assert len(rows) == 6
    assert sequence == 6


def test_invalid_observation_retries_same_declared_fixture_without_counting_as_failure(monkeypatch):
    module = _module()
    calls = []
    outcomes = {
        1: [
            ("RUNTIME_FAILURE", False),
            ("ANSWER_CORRECT", True),
        ],
    }
    runner = Runner()
    monkeypatch.setattr(module, "execute_experiment", fake_executor(outcomes, calls))

    rows, _ = module.run_family_frontier(runner, "math", {1: case(1)})

    assert [item["level"] for item in calls] == [1, 1]
    observations = runner.store.rows["capability-observations.jsonl"]
    assert observations[0]["valid_for_capability"] is False
    assert observations[0]["passed"] is None
    assert observations[1]["valid_for_capability"] is True
    assert observations[1]["passed"] is True
    assert len(rows) == 2


def test_sparse_ladder_records_missing_fixture_coverage_instead_of_inventing_level(monkeypatch):
    module = _module()
    calls = []
    runner = Runner()
    outcomes = {2: ("ANSWER_CORRECT", True)}
    monkeypatch.setattr(module, "execute_experiment", fake_executor(outcomes, calls))

    rows, _ = module.run_family_frontier(runner, "math", {2: case(2)})

    assert [item["level"] for item in calls] == [2]
    events = runner.store.rows["capability-events.jsonl"]
    assert events[-1]["event"] == "family_stop"
    assert events[-1]["reason"] == "MISSING_FIXTURE_COVERAGE"
    assert events[-1]["requested_level"] == 5
    assert len(rows) == 1


def test_campaign_writes_failure_atlas_from_executed_experiments(monkeypatch):
    module = _module()
    calls = []
    runner = Runner()
    family = "formal_logic_deduction"
    outcomes = {1: ("ANSWER_WRONG", True)}
    monkeypatch.setattr(module, "execute_experiment", fake_executor(outcomes, calls))

    rows = module.run_capability_campaign(runner, [case(1, family)])

    assert len(rows) == 1
    assert "failure-atlas.json" in runner.store.jsons
    atlas = runner.store.jsons["failure-atlas.json"]
    assert atlas["model"] == "fake"
    assert atlas["summary"]["model_failures"] == 1
    assert atlas["failures"][0]["failure_signature"]["subtype"] == "logic_error"


def test_campaign_runs_recovery_after_reasoning_and_includes_recovery_in_final_atlas(monkeypatch):
    module = _module()
    calls = []
    runner = Runner()
    runner.config["reasoning_curves"] = {"enabled": False, "repeats": 3}
    runner.config["recovery_lab"] = {"enabled": True, "repeats": 3, "max_level": "R7"}
    family = "formal_logic_deduction"
    outcomes = {1: ("ANSWER_WRONG", True)}
    monkeypatch.setattr(module, "execute_experiment", fake_executor(outcomes, calls))

    captured = {}
    recovery_row = {
        "experiment": {
            "experiment_id": "recovery-exp",
            "parent_experiment_id": "base-exp",
            "task_id": family,
            "task_family": family,
            "difficulty_level": 1,
            "hypothesis": "recovery",
            "changed_variable": "prompt_variant",
            "thinking_mode": True,
            "reasoning_effort": "medium",
            "generation_budget": 256,
            "context_request": None,
            "temperature": 0.0,
            "seed": 42,
            "prompt_variant": "recovery-r4",
            "recovery_level": "R4",
        },
        "classification": {"result_class": "ANSWER_WRONG", "valid_for_capability": True},
        "score": 0.0,
        "status": "FAIL",
    }

    def fake_recovery(
        runner_arg,
        cases_arg,
        base_rows,
        reasoning_rows,
        frontiers,
        failure_atlas,
        *,
        sequence_start=0,
    ):
        captured["base_rows"] = list(base_rows)
        captured["reasoning_rows"] = list(reasoning_rows)
        captured["frontiers"] = frontiers
        captured["failure_atlas"] = failure_atlas
        captured["sequence_start"] = sequence_start
        return [recovery_row], {"schema_version": 1, "families": {family: {}}}, sequence_start + 1

    monkeypatch.setattr(module, "run_recovery_lab", fake_recovery, raising=False)

    rows = module.run_capability_campaign(runner, [case(1, family)])

    assert captured["base_rows"]
    assert captured["reasoning_rows"] == []
    assert captured["frontiers"]["families"][family]["first_failure_level"] == 1
    assert captured["failure_atlas"]["summary"]["model_failures"] == 1
    assert "recovery-map.json" in runner.store.jsons
    assert rows[-1]["experiment"]["experiment_id"] == "recovery-exp"
    assert runner.store.jsons["failure-atlas.json"]["summary"]["model_failures"] == 2
