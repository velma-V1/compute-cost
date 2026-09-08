import base64
import json
from pathlib import Path

from compute_cost.config import load_config
from compute_cost.evidence import EvidenceStore
from compute_cost.runner import BenchmarkRunner


def envelope(*, text="", thinking="reasoned"):
    payload = {
        "message": {"role": "assistant", "content": text, "thinking": thinking},
        "done": True,
        "done_reason": "stop",
        "eval_count": 8,
        "eval_duration": 1,
        "prompt_eval_count": 3,
        "prompt_eval_duration": 1,
        "total_duration": 2,
        "load_duration": 0,
    }
    raw = (json.dumps(payload) + "\n").encode()
    return {
        "ok": True,
        "http_status": 200,
        "request": {
            "body_b64": base64.b64encode(b'{"preserved":true}').decode(),
            "body_text": '{"preserved":true}',
            "started_monotonic_ns": 1,
        },
        "stream_events": [{
            "sequence": 0,
            "received_monotonic_ns": 2,
            "received_at_utc": "x",
            "raw_b64": base64.b64encode(raw).decode(),
            "raw_text": raw.decode(),
            "parsed": payload,
            "parse_error": None,
        }],
        "raw_response_b64": base64.b64encode(raw).decode(),
        "normalized": {
            "text": text,
            "thinking": thinking,
            "tool_calls": [],
            "done": True,
            "done_reason": "stop",
        },
        "metrics": {
            "eval_count": 8,
            "eval_duration_ns": 1,
            "prompt_eval_count": 3,
            "prompt_eval_duration_ns": 1,
            "total_duration_ns": 2,
            "load_duration_ns": 0,
        },
        "timing": {"first_event_latency_ns": 1, "last_event_latency_ns": 1},
        "phase_metrics": {
            "measurement_kind": "MEASURED",
            "thinking_chunks": 1,
            "answer_chunks": 1,
            "thinking_chars": len(thinking),
            "answer_chars": len(text),
            "time_to_first_thinking_ns": 1,
            "time_to_first_answer_ns": 1,
            "thinking_span_ns": 0,
            "answer_span_ns": 0,
        },
    }


class Runtime:
    def __init__(self):
        self.calls = []

    def version(self):
        return {**envelope(text=""), "parsed": {"version": "test"}}

    def list_models(self):
        return {**envelope(text=""), "parsed": {"models": [{"name": "fake", "size": 1234}]}}

    def model_available_in(self, tags, model):
        return True

    def model_info(self, model):
        return {**envelope(text=""), "parsed": {"details": {"parameter_size": "1B"}}}

    def pull(self, model):
        return {**envelope(text=""), "parsed_events": [{"status": "success"}]}

    def generate(self, model, messages, options, *, stream=True, request_fields=None):
        prompt = messages[-1]["content"]
        level = int(prompt.split("L")[-1])
        self.calls.append({
            "level": level,
            "options": dict(options),
            "request_fields": dict(request_fields or {}),
        })
        return envelope(text="OK" if level <= 4 else "WRONG")


class Telemetry:
    def sample(self):
        return {
            "timestamp_utc": "x",
            "monotonic_ns": 1,
            "host": {"memory_percent": 1.0},
            "gpu": {"availability": "unavailable"},
        }


class FakeProgress:
    def __init__(self, total_tasks):
        self.total_tasks = total_tasks
        self.done = 0

    def start(self, task): pass
    def start_live(self): pass
    def begin_task(self, task): pass
    def complete_task(self, next_task=None): self.done += 1
    def snapshot(self):
        total = max(1, self.total_tasks)
        return {
            "done": self.done,
            "total": self.total_tasks,
            "left": self.total_tasks - self.done,
            "percent": self.done / total * 100,
            "current_task": "x",
            "elapsed_s": self.done,
            "eta_s": 0.0,
            "finish": None,
        }
    def stop_live(self, *, newline=True): pass


def fixture(level):
    return {
        "id": f"math-L{level}",
        "category": "math",
        "family_id": "math",
        "taxonomy_version": "capability-taxonomy-v1",
        "difficulty_level": level,
        "difficulty": {
            "level": level,
            "rubric_version": "math-v1",
            "dimensions": {"steps": level},
        },
        "prompt": f"Solve synthetic capability L{level}",
        "scorer": "exact",
        "scorer_version": "1",
        "expected": "OK",
        "timeout_s": 120,
        "capabilities_required": ["math"],
        "recovery_eligible": True,
        "robustness_eligible": True,
        "compound": False,
        "tags": [],
    }


def suite():
    return {
        "benchmark_version": "capability-runner-test",
        "schema_version": "capability-suite-v1",
        "taxonomy_version": "capability-taxonomy-v1",
        "coverage": {"math": "PARTIAL", "logic": "UNTESTED"},
        "cases": [fixture(level) for level in (1, 4, 5, 7)],
    }


def config():
    cfg = load_config()
    cfg["telemetry"]["background"] = False
    cfg["capability_campaign"] = {
        "anchor_level": 1,
        "jump": 3,
        "boundary_repeats": 2,
        "max_experiments_per_family": 12,
        "thinking_mode": True,
        "generation_budget": 256,
        "reliable_threshold": 0.90,
        "unstable_threshold": 0.40,
    }
    return cfg


def test_capability_runner_executes_adaptive_levels_and_writes_frontier_artifacts(tmp_path: Path):
    runtime = Runtime()
    runner = BenchmarkRunner(
        runtime,
        config(),
        suite(),
        results_root=tmp_path,
        telemetry=Telemetry(),
        progress_factory=FakeProgress,
    )

    run_dir = runner.capability_characterize("fake")

    assert [call["level"] for call in runtime.calls] == [1, 4, 7, 5, 4, 5]
    assert all(call["request_fields"] == {"think": True} for call in runtime.calls)
    assert all(call["options"]["num_predict"] == 256 for call in runtime.calls)

    frontiers = json.loads((run_dir / "capability-frontiers.json").read_text())
    assert frontiers["schema_version"] == 1
    assert set(frontiers["families"]) == {"logic", "math"}
    assert frontiers["families"]["math"]["reliable_floor"] == 4
    assert frontiers["families"]["math"]["first_failure_level"] == 5
    assert frontiers["families"]["logic"]["coverage"]["tested_count"] == 0

    ledger = json.loads((run_dir / "coverage-ledger.json").read_text())
    assert ledger["schema_version"] == 1
    assert ledger["families"]["math"]["state"] == "PROVEN"
    assert ledger["families"]["logic"]["state"] == "UNTESTED"
    assert set(item["state"] for item in ledger["families"].values()) <= {
        "PROVEN", "PARTIAL", "UNCERTAIN", "UNTESTED"
    }

    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert any(row["event"] == "CAPABILITY_CHARACTERIZATION_COMPLETE" for row in events)
    assert EvidenceStore(tmp_path, run_dir.name).verify_manifest() == []

    progress = [json.loads(line) for line in (run_dir / "progress.jsonl").read_text().splitlines()]
    adjustments = [row for row in progress if row["event"] == "plan_adjusted"]
    assert len(adjustments) == 5
    assert all(row["reason"] == "adaptive experiment added" for row in adjustments)
    assert progress[-1]["event"] == "complete"
    assert progress[-1]["progress"]["percent"] == 100.0
