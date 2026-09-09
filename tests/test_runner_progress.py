import base64
import json
from pathlib import Path

from compute_cost.config import load_config
from compute_cost.runner import BenchmarkRunner


def envelope(text="READY"):
    raw = json.dumps({"message": {"role": "assistant", "content": text}, "done": True}).encode()
    return {
        "ok": True,
        "http_status": 200,
        "request": {"body_b64": base64.b64encode(b"{}").decode()},
        "stream_events": [{"sequence": 0, "raw_b64": base64.b64encode(raw).decode(), "parsed": json.loads(raw)}],
        "raw_response_b64": base64.b64encode(raw).decode(),
        "normalized": {"text": text, "thinking": "", "tool_calls": [], "done": True},
        "metrics": {"eval_count": 1, "eval_duration_ns": 1, "prompt_eval_count": 1, "prompt_eval_duration_ns": 1},
        "timing": {},
    }


class Runtime:
    def version(self): return {**envelope(), "parsed": {"version": "test"}}
    def list_models(self): return {**envelope(), "parsed": {"models": [{"name": "fake", "size": 1}]}}
    def model_available_in(self, tags, model): return True
    def model_info(self, model): return {**envelope(), "parsed": {"model_info": {"context_length": 4096}}}
    def unload(self, model): return envelope("")
    def generate(self, model, messages, options, *, stream=True, request_fields=None):
        prompt = messages[-1]["content"]
        if "CONTEXT_SWEEP" in prompt:
            keys = [prompt.split(label, 1)[1].split()[0] for label in ("FACT_A=", "FACT_B=", "FACT_C=")]
            return envelope("|".join(keys))
        if "PING" in prompt: return envelope("PING")
        if "BASE" in prompt: return envelope("BASE")
        return envelope("READY")


class Telemetry:
    def sample(self):
        return {"timestamp_utc": "x", "monotonic_ns": 1, "host": {"memory_percent": 1.0}, "gpu": {"availability": "unavailable"}}


class FakeProgress:
    def __init__(self, total_tasks):
        self.total_tasks = total_tasks
        self.done = 0
        self.calls = []

    def start(self, task): self.calls.append(("start", task))
    def start_live(self): self.calls.append(("start_live", None))
    def begin_task(self, task): self.calls.append(("begin", task))
    def complete_task(self, next_task=None):
        self.done += 1
        self.calls.append(("complete", next_task))
    def snapshot(self):
        return {"done": self.done, "total": self.total_tasks, "left": self.total_tasks - self.done, "percent": self.done / self.total_tasks * 100, "current_task": "x", "elapsed_s": self.done, "eta_s": 0.0, "finish": None}
    def stop_live(self, *, newline=True): self.calls.append(("stop_live", newline))


def test_onboard_progress_covers_full_independent_run_and_is_retained(tmp_path: Path):
    cfg = load_config()
    cfg["telemetry"]["background"] = False
    cfg["limits"]["context_schedule"] = [4096]
    cfg["limits"]["sustained_iterations"] = 1
    suite = {"benchmark_version": "p1", "cases": [{"id": "base", "category": "instruction_following", "prompt": "BASE", "scorer": "exact", "expected": "BASE", "timeout_s": 10, "max_output_tokens": 8}]}
    made = []

    def factory(total_tasks):
        progress = FakeProgress(total_tasks)
        made.append(progress)
        return progress

    runner = BenchmarkRunner(Runtime(), cfg, suite, results_root=tmp_path, telemetry=Telemetry(), progress_factory=factory)
    run_dir = runner.onboard("fake")

    progress = made[0]
    assert progress.total_tasks == 7  # preflight, cold, warm, base, context, sustained, finalize
    assert progress.done == 7
    assert ("start_live", None) in progress.calls
    assert ("stop_live", True) in progress.calls
    assert (run_dir / "progress.jsonl").exists()
    rows = [json.loads(line) for line in (run_dir / "progress.jsonl").read_text().splitlines()]
    assert rows[0]["event"] == "start"
    assert rows[-1]["event"] == "complete"
    assert rows[-1]["progress"]["percent"] == 100.0
