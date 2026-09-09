import base64
import json
from pathlib import Path

from compute_cost.characterization import _write_replay_v2
from compute_cost.config import load_config
from compute_cost.experiments import ExperimentSpec
from compute_cost.runner_core import BenchmarkRunner


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


def envelope(text="OK"):
    raw = json.dumps({"message": {"role": "assistant", "content": text}, "done": True}).encode()
    return {
        "ok": True,
        "http_status": 200,
        "request": {
            "body_b64": base64.b64encode(b"{}").decode(),
            "body_text": "{}",
            "started_monotonic_ns": 1,
        },
        "stream_events": [{
            "sequence": 0,
            "received_monotonic_ns": 2,
            "received_at_utc": "x",
            "raw_b64": base64.b64encode(raw).decode(),
            "raw_text": raw.decode(),
            "parsed": json.loads(raw),
            "parse_error": None,
        }],
        "raw_response_b64": base64.b64encode(raw).decode(),
        "normalized": {"text": text, "thinking": "", "tool_calls": [], "done": True},
        "metrics": {},
        "timing": {},
    }


class Runtime:
    def __init__(self):
        self.calls = []

    def generate(self, model, messages, options, *, stream=True, request_fields=None):
        self.calls.append((model, messages, options, request_fields))
        return envelope()


class Telemetry:
    def sample(self):
        return {"timestamp_utc": "x", "monotonic_ns": 1, "host": {}, "gpu": {}}


def test_replay_case_executes_indexed_snapshot_without_root_alias(tmp_path: Path):
    source = tmp_path / "source-run"
    target = source / "replay" / "failures" / "exp-indexed.json"
    target.parent.mkdir(parents=True)
    snapshot = {
        "schema_version": 2,
        "model": "fake",
        "invocation": {
            "messages": [{"role": "user", "content": "exact prompt"}],
            "options": {"temperature": 0.0, "seed": 42, "num_predict": 256},
            "request_fields": {"think": "medium"},
        },
    }
    target.write_text(json.dumps(snapshot), encoding="utf-8")
    (source / "replay" / "index.jsonl").write_text(
        json.dumps({
            "replay_id": "exp-indexed",
            "category": "failures",
            "path": "replay/failures/exp-indexed.json",
        }) + "\n",
        encoding="utf-8",
    )

    runtime = Runtime()
    cfg = load_config()
    cfg["telemetry"]["background"] = False
    runner = BenchmarkRunner(
        runtime,
        cfg,
        {"benchmark_version": "x", "cases": []},
        results_root=tmp_path,
        telemetry=Telemetry(),
    )
    replay_dir = runner.replay_case("source-run", "exp-indexed")

    assert runtime.calls == [(
        "fake",
        [{"role": "user", "content": "exact prompt"}],
        {"temperature": 0.0, "seed": 42, "num_predict": 256},
        {"think": "medium"},
    )]
    assert json.loads((replay_dir / "source-replay.json").read_text())["schema_version"] == 2
