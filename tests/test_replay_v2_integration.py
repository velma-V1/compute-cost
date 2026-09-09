from compute_cost.characterization import _write_replay_v2
from compute_cost.experiments import ExperimentSpec


class Store:
    def __init__(self):
        self.jsons = {}
        self.jsonl = {}

    def write_json(self, path, value, **kwargs):
        self.jsons[path] = value
        return {"path": path, "sha256": f"sha:{path}", "bytes": 1}

    def append_jsonl(self, path, value):
        self.jsonl.setdefault(path, []).append(value)
        return {"path": path, "sha256": f"sha:{path}", "bytes": 1}


class SnapshotRunner:
    def __init__(self):
        self.store = Store()
        self.suite = {"benchmark_version": "cap-v1"}
        self.model = "gpt-oss:20b"
        self.config = {"capability_campaign": {"reasoning_effort": "medium"}}
        self._recent_telemetry = [{"gpu": {"power": 10}}]

    def _utc(self):
        return "x"


def spec(*, recovery_level=None):
    return ExperimentSpec(
        experiment_id="exp-1",
        parent_experiment_id="exp-parent",
        task_id="formal_logic_deduction",
        task_family="formal_logic_deduction",
        difficulty_level=6,
        hypothesis="boundary failure",
        changed_variable="difficulty_level",
        thinking_mode=True,
        reasoning_effort="medium",
        generation_budget=256,
        context_request=None,
        temperature=0.0,
        seed=42,
        prompt_variant="base",
        recovery_level=recovery_level,
    )


def test_v2_failure_snapshot_gets_canonical_category_legacy_alias_and_index():
    runner = SnapshotRunner()
    _write_replay_v2(
        runner,
        case={"id": "logic-L6", "prompt": "p"},
        spec=spec(),
        invocation={"messages": [{"role": "user", "content": "p"}], "options": {}},
        generation={"normalized": {"text": "wrong"}},
        scoring={"score": 0.0},
        classification={"result_class": "ANSWER_WRONG", "valid_for_capability": True},
    )

    assert "replay/failures/exp-1.json" in runner.store.jsons
    assert "replay/exp-1.json" in runner.store.jsons
    index = runner.store.jsonl["replay/index.jsonl"][0]
    assert index["replay_id"] == "exp-1"
    assert index["category"] == "failures"
    assert index["path"] == "replay/failures/exp-1.json"
    assert index["compatibility_path"] == "replay/exp-1.json"
    assert index["canonical_sha256"] == "sha:replay/failures/exp-1.json"
    assert index["family_id"] == "formal_logic_deduction"
    assert index["difficulty_level"] == 6
    assert index["result_class"] == "ANSWER_WRONG"


def test_v2_recovery_and_anomaly_snapshots_route_to_their_own_buckets():
    recovery = SnapshotRunner()
    _write_replay_v2(
        recovery,
        case={"id": "logic-L6", "prompt": "p"},
        spec=spec(recovery_level="R4"),
        invocation={}, generation={}, scoring={},
        classification={"result_class": "ANSWER_WRONG", "valid_for_capability": True},
    )
    assert "replay/recoveries/exp-1.json" in recovery.store.jsons

    anomaly = SnapshotRunner()
    _write_replay_v2(
        anomaly,
        case={"id": "logic-L6", "prompt": "p"},
        spec=spec(),
        invocation={}, generation={}, scoring={},
        classification={"result_class": "THINK_TRUNCATED", "valid_for_capability": False},
    )
    assert "replay/anomalies/exp-1.json" in anomaly.store.jsons
