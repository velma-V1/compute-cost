import base64
import json
from pathlib import Path

from compute_cost.config import load_config
from compute_cost.evidence import EvidenceStore
from compute_cost.runner import BenchmarkRunner


def _envelope(text="OK"):
    raw = (json.dumps({"message": {"role": "assistant", "content": text}, "done": True}) + "\n").encode()
    return {
        "ok": True,
        "http_status": 200,
        "request": {
            "body_b64": base64.b64encode(b'{"request":true}').decode(),
            "body_text": '{"request":true}',
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
        self.calls = 0

    def generate(self, model, messages, options, *, stream=True, request_fields=None):
        self.calls += 1
        return _envelope()


class Telemetry:
    def sample(self):
        return {
            "timestamp_utc": "x",
            "monotonic_ns": 1,
            "host": {"memory_percent": 1.0},
            "gpu": {"availability": "unavailable"},
        }


def test_third_generation_is_blocked_by_two_call_run_budget(tmp_path: Path):
    runtime = Runtime()
    config = load_config(overrides={"limits.max_model_calls_per_run": 2})
    config["telemetry"]["background"] = False
    runner = BenchmarkRunner(
        runtime,
        config,
        {"benchmark_version": "budget-test", "cases": []},
        results_root=tmp_path,
        telemetry=Telemetry(),
    )
    runner.model = "fake"
    runner.store = EvidenceStore(tmp_path, "run")

    results = []
    for index in range(3):
        generation, _, _ = runner._invoke_generation(
            stage="characterize",
            case_id=f"case-{index}",
            messages=[{"role": "user", "content": "return OK"}],
            options={"num_predict": 8},
            request_fields={"think": "medium"},
        )
        results.append(generation)

    assert runtime.calls == 2
    assert results[0]["ok"] is True
    assert results[1]["ok"] is True
    assert results[2]["ok"] is False
    assert results[2]["error"]["type"] == "MODEL_CALL_BUDGET_EXHAUSTED"
    assert results[2]["error"]["limit"] == 2
    assert results[2]["error"]["completed_calls"] == 2

    events_path = runner.store.run_dir / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    exhausted = [row for row in events if row["type"] == "MODEL_CALL_BUDGET_EXHAUSTED"]
    assert len(exhausted) == 1
    assert exhausted[0]["limit"] == 2
    assert exhausted[0]["completed_calls"] == 2
