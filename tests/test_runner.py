import base64
import json
from pathlib import Path

from compute_cost.config import load_config
from compute_cost.evidence import EvidenceStore
from compute_cost.runner import BenchmarkRunner


def envelope(text="READY", *, load_ns=10, eval_count=2, eval_ns=20, prompt_count=3, prompt_ns=30):
    raw_request = b'{"preserved":"request"}'
    raw = json.dumps({"message": {"role": "assistant", "content": text}, "done": True, "future_field": 99}).encode()
    return {
        "ok": True,
        "http_status": 200,
        "request": {"body_b64": base64.b64encode(raw_request).decode(), "body_text": raw_request.decode(), "started_monotonic_ns": 1},
        "stream_events": [{"sequence": 0, "received_monotonic_ns": 2, "received_at_utc": "x", "raw_b64": base64.b64encode(raw).decode(), "raw_text": raw.decode(), "parsed": json.loads(raw), "parse_error": None}],
        "raw_response_b64": base64.b64encode(raw).decode(),
        "normalized": {"text": text, "thinking": "", "tool_calls": [], "done": True, "done_reason": "stop"},
        "metrics": {"load_duration_ns": load_ns, "eval_count": eval_count, "eval_duration_ns": eval_ns, "prompt_eval_count": prompt_count, "prompt_eval_duration_ns": prompt_ns, "total_duration_ns": load_ns + eval_ns + prompt_ns},
        "timing": {"first_event_latency_ns": 1, "last_event_latency_ns": 2},
    }


class FakeRuntime:
    def __init__(self, fail_base=False):
        self.fail_base = fail_base
        self.calls = []

    def version(self):
        return {**envelope(), "parsed": {"version": "test"}}

    def list_models(self):
        return {**envelope(), "parsed": {"models": [{"name": "fake", "size": 1234, "future": True}]}}

    def model_info(self, model):
        return {**envelope(), "parsed": {"details": {"parameter_size": "1B"}, "model_info": {"context_length": 32768}, "future": {"x": 1}}}

    def model_available_in(self, tags, model):
        return True

    def is_model_available(self, model):
        return True

    def unload(self, model):
        return {**envelope(""), "parsed": {"done": True}}

    def pull(self, model):
        return {**envelope(""), "parsed_events": [{"status": "success"}]}

    def generate(self, model, messages, options, *, stream=True, request_fields=None):
        prompt = messages[-1]["content"]
        self.calls.append({"model": model, "messages": messages, "options": options, "stream": stream, "request_fields": request_fields})
        if "CONTEXT_SWEEP target=8192" in prompt:
            return envelope("WRONG")
        if "CONTEXT_SWEEP" in prompt:
            facts = []
            for label in ("FACT_A=", "FACT_B=", "FACT_C="):
                facts.append(prompt.split(label, 1)[1].split()[0])
            return envelope("|".join(facts))
        if "PING" in prompt:
            return envelope("PING")
        if "BASE-OK" in prompt:
            return envelope("WRONG" if self.fail_base else "BASE-OK")
        return envelope("READY")


class FakeTelemetry:
    def __init__(self):
        self.n = 0

    def sample(self):
        self.n += 1
        raw = f"gpu-sample-{self.n}\n".encode()
        return {
            "timestamp_utc": f"t{self.n}",
            "monotonic_ns": self.n * 1_000_000_000,
            "host": {"memory_total_bytes": 1000, "memory_used_bytes": 100, "memory_percent": 10.0},
            "process": {"pid": 1, "rss_bytes": 10},
            "gpu": {"availability": "available", "devices": [{"memory_total_mib": 1000.0, "memory_used_mib": 100.0, "power_draw_w": 50.0}]},
            "total_gpu_power_w": 50.0,
            "nvidia_collector": {"returncode": 0, "stdout_b64": base64.b64encode(raw).decode(), "stderr_b64": "", "stdout_text": raw.decode(), "stderr_text": "", "error": None},
        }


def suite():
    return {
        "benchmark_version": "test-v1",
        "cases": [{"id": "base-1", "category": "instruction_following", "prompt": "Reply exactly BASE-OK", "scorer": "exact", "expected": "BASE-OK", "timeout_s": 10, "max_output_tokens": 16}],
    }


def config():
    value = load_config()
    value["telemetry"]["background"] = False
    value["limits"]["context_schedule"] = [4096, 8192]
    value["limits"]["consecutive_context_failures"] = 1
    value["limits"]["sustained_iterations"] = 2
    return value


def test_onboarding_retains_raw_runtime_telemetry_cases_reports_and_integrity(tmp_path: Path):
    runner = BenchmarkRunner(FakeRuntime(), config(), suite(), results_root=tmp_path, telemetry=FakeTelemetry())
    run_dir = runner.onboard("fake")

    required = ["resolved-config.json", "benchmark-snapshot.json", "hardware.json", "runtime.json", "events.jsonl", "telemetry.jsonl", "cases.jsonl", "summary.json", "report.md", "manifest.json"]
    for name in required:
        assert (run_dir / name).exists(), name
    assert list((run_dir / "raw/runtime/requests").glob("*"))
    assert list((run_dir / "raw/runtime/responses").glob("*"))
    assert list((run_dir / "raw/runtime/streams").rglob("*.*"))
    assert list((run_dir / "raw/telemetry/nvidia").glob("*stdout*"))
    assert EvidenceStore(tmp_path, run_dir.name).verify_manifest() == []

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["context_boundary"]["last_success"] == 4096
    assert summary["context_boundary"]["first_stop"] == 8192
    rows = [json.loads(line) for line in (run_dir / "cases.jsonl").read_text().splitlines()]
    assert len([r for r in rows if r.get("stage") == "sustained"]) == 2


def test_failed_case_becomes_exact_replay_snapshot(tmp_path: Path):
    runner = BenchmarkRunner(FakeRuntime(fail_base=True), config(), suite(), results_root=tmp_path, telemetry=FakeTelemetry())
    run_dir = runner.onboard("fake")

    replay = json.loads((run_dir / "replay/base-1.json").read_text())
    assert replay["case"]["id"] == "base-1"
    assert replay["invocation"]["messages"][-1]["content"] == "Reply exactly BASE-OK"
    assert replay["generation"]["normalized"]["text"] == "WRONG"
    assert replay["scoring"]["score"] == 0.0
    assert replay["telemetry_before_failure"]
    assert replay["resolved_config"]["limits"]["context_schedule"] == [4096, 8192]
