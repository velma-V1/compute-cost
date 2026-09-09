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
        "eval_duration": 10,
        "prompt_eval_count": 3,
        "prompt_eval_duration": 5,
        "total_duration": 100,
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
        "stream_events": [],
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
            "eval_duration_ns": 10,
            "prompt_eval_count": 3,
            "prompt_eval_duration_ns": 5,
            "total_duration_ns": 100,
            "load_duration_ns": 0,
        },
        "timing": {"first_event_latency_ns": 1, "last_event_latency_ns": 100},
        "phase_metrics": {
            "measurement_kind": "MEASURED",
            "thinking_chunks": 1,
            "answer_chunks": 1,
            "thinking_chars": len(thinking),
            "answer_chars": len(text),
            "time_to_first_thinking_ns": 1,
            "time_to_first_answer_ns": 50,
            "thinking_span_ns": 40,
            "answer_span_ns": 20,
        },
    }


class Runtime:
    def __init__(self):
        self.calls = []

    def version(self):
        return {**envelope(), "parsed": {"version": "test"}}

    def list_models(self):
        return {**envelope(), "parsed": {"models": [{"name": "gpt-oss:20b", "size": 1234}]}}

    def model_available_in(self, tags, model):
        return True

    def model_info(self, model):
        return {**envelope(), "parsed": {"details": {"parameter_size": "20B"}}}

    def generate(self, model, messages, options, *, stream=True, request_fields=None):
        prompt = messages[-1]["content"]
        level = int(prompt.rsplit("L", 1)[1])
        effort = (request_fields or {}).get("think")
        self.calls.append((level, effort))
        passed = level <= 4 or (effort == "high" and level == 5)
        return envelope(text="OK" if passed else "WRONG")


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
    def stop_live(self, *, newline=True): pass
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


def fixture(level):
    return {
        "id": f"math-L{level}",
        "category": "math",
        "family_id": "math",
        "taxonomy_version": "capability-taxonomy-v1",
        "difficulty_level": level,
        "difficulty": {"level": level, "rubric_version": "math-v1", "dimensions": {"steps": level}},
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


def config():
    cfg = load_config()
    cfg["telemetry"]["background"] = False
    cfg["capability_campaign"].update({
        "anchor_level": 1,
        "jump": 3,
        "boundary_repeats": 2,
        "max_experiments_per_family": 12,
        "reasoning_effort": "medium",
        "generation_budget": 256,
    })
    cfg["reasoning_curves"] = {"enabled": True, "repeats": 2}
    cfg["recovery_lab"]["enabled"] = False
    cfg["robustness_lab"]["enabled"] = False
    return cfg


def suite():
    return {
        "benchmark_version": "reasoning-runner-test",
        "schema_version": "capability-suite-v1",
        "taxonomy_version": "capability-taxonomy-v1",
        "coverage": {"math": "PARTIAL"},
        "cases": [fixture(level) for level in (1, 4, 5, 7)],
    }


def test_capability_runner_writes_frontier_local_reasoning_curves_without_mutating_baseline(tmp_path: Path):
    runtime = Runtime()
    runner = BenchmarkRunner(
        runtime,
        config(),
        suite(),
        results_root=tmp_path,
        telemetry=Telemetry(),
        progress_factory=FakeProgress,
    )

    run_dir = runner.capability_characterize("gpt-oss:20b")

    assert runtime.calls[:6] == [
        (1, "medium"), (4, "medium"), (7, "medium"),
        (5, "medium"), (4, "medium"), (5, "medium"),
    ]
    assert runtime.calls[6:] == [(4, "low"), (4, "low"), (5, "high"), (5, "high")]

    frontiers = json.loads((run_dir / "capability-frontiers.json").read_text())
    assert frontiers["families"]["math"]["reliable_floor"] == 4
    assert frontiers["families"]["math"]["first_failure_level"] == 5

    curves = json.loads((run_dir / "reasoning-curves.json").read_text())
    math = curves["families"]["math"]
    assert math["baseline_medium_frontier"] == {"reliable_floor": 4, "first_failure_level": 5}
    assert math["minimum_reliable_effort_at_baseline_floor"] == "low"
    assert math["demonstrated_high_effort_extension_to"] == 5

    observations = [
        json.loads(line)
        for line in (run_dir / "reasoning-observations.jsonl").read_text().splitlines()
    ]
    assert [(row["level"], row["effort"]) for row in observations] == [
        (4, "low"), (4, "low"), (5, "high"), (5, "high")
    ]
    assert EvidenceStore(tmp_path, run_dir.name).verify_manifest() == []
