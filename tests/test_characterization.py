import base64
import json
from pathlib import Path

from compute_cost.characterization import build_task_profile
from compute_cost.config import load_config
from compute_cost.evidence import EvidenceStore
from compute_cost.runner import BenchmarkRunner


def envelope(*, text="", thinking="", done_reason="stop", eval_count=8):
    payload = {
        "message": {"role": "assistant", "content": text, "thinking": thinking},
        "done": True,
        "done_reason": done_reason,
        "eval_count": eval_count,
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
        "stream_events": [
            {
                "sequence": 0,
                "received_monotonic_ns": 2,
                "received_at_utc": "x",
                "raw_b64": base64.b64encode(raw).decode(),
                "raw_text": raw.decode(),
                "parsed": payload,
                "parse_error": None,
            }
        ],
        "raw_response_b64": base64.b64encode(raw).decode(),
        "normalized": {
            "text": text,
            "thinking": thinking,
            "tool_calls": [],
            "done": True,
            "done_reason": done_reason,
        },
        "metrics": {
            "eval_count": eval_count,
            "eval_duration_ns": 1,
            "prompt_eval_count": 3,
            "prompt_eval_duration_ns": 1,
            "total_duration_ns": 2,
            "load_duration_ns": 0,
        },
        "timing": {"first_event_latency_ns": 1, "last_event_latency_ns": 1},
        "phase_metrics": {
            "measurement_kind": "MEASURED",
            "thinking_chunks": 1 if thinking else 0,
            "answer_chunks": 1 if text else 0,
            "thinking_chars": len(thinking),
            "answer_chars": len(text),
            "time_to_first_thinking_ns": 1 if thinking else None,
            "time_to_first_answer_ns": 1 if text else None,
            "thinking_span_ns": 0 if thinking else None,
            "answer_span_ns": 0 if text else None,
        },
    }


class AdaptiveRuntime:
    def __init__(self):
        self.calls = []

    def version(self):
        return {**envelope(text=""), "parsed": {"version": "test"}}

    def list_models(self):
        return {**envelope(text=""), "parsed": {"models": [{"name": "fake", "size": 1234}]}}

    def model_available_in(self, tags, model):
        return True

    def model_info(self, model):
        return {**envelope(text=""), "parsed": {"details": {"parameter_size": "1B"}, "model_info": {"context_length": 32768}}}

    def pull(self, model):
        return {**envelope(text=""), "parsed_events": [{"status": "success"}]}

    def generate(self, model, messages, options, *, stream=True, request_fields=None):
        call = {
            "model": model,
            "messages": messages,
            "options": dict(options),
            "request_fields": dict(request_fields or {}),
        }
        self.calls.append(call)
        think = bool((request_fields or {}).get("think"))
        budget = int(options["num_predict"])
        if not think:
            return envelope(text="WRONG", eval_count=min(budget, 8))
        if budget < 192:
            return envelope(
                text="",
                thinking="Thinking Process:\n\n1. Analyze the Request",
                done_reason="length",
                eval_count=budget,
            )
        return envelope(text="BLUE", thinking="reasoned", eval_count=min(budget, 24))


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
        self.calls = []

    def start(self, task): self.calls.append(("start", task))
    def start_live(self): self.calls.append(("start_live", None))
    def begin_task(self, task): self.calls.append(("begin", task))
    def complete_task(self, next_task=None):
        self.done += 1
        self.calls.append(("complete", next_task))
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
    def stop_live(self, *, newline=True): self.calls.append(("stop_live", newline))


def suite():
    return {
        "benchmark_version": "qwen-characterization-test",
        "cases": [
            {
                "id": "char-if-001",
                "category": "instruction_following",
                "difficulty_level": 1,
                "prompt": "Reply with exactly: BLUE",
                "scorer": "exact",
                "expected": "BLUE",
                "timeout_s": 120,
            }
        ],
    }


def config():
    cfg = load_config()
    cfg["telemetry"]["background"] = False
    return cfg


def test_characterize_brackets_reproduces_and_retains_exact_evidence(tmp_path: Path):
    made = []
    runtime = AdaptiveRuntime()

    def factory(total):
        progress = FakeProgress(total)
        made.append(progress)
        return progress

    runner = BenchmarkRunner(
        runtime,
        config(),
        suite(),
        results_root=tmp_path,
        telemetry=Telemetry(),
        progress_factory=factory,
    )
    run_dir = runner.characterize("fake")

    rows = [json.loads(line) for line in (run_dir / "experiments.jsonl").read_text().splitlines()]
    assert rows[0]["experiment"]["thinking_mode"] is False
    assert rows[1]["experiment"]["thinking_mode"] is True
    assert rows[0]["experiment"]["generation_budget"] == 256
    assert rows[1]["experiment"]["generation_budget"] == 256
    assert any(row["classification"]["result_class"] == "THINK_TRUNCATED" for row in rows)
    assert all(
        row["classification"]["valid_for_capability"] is False
        for row in rows
        if row["classification"]["result_class"] == "THINK_TRUNCATED"
    )
    passes_192 = [
        row for row in rows
        if row["experiment"]["generation_budget"] == 192
        and row["classification"]["result_class"] == "ANSWER_CORRECT"
    ]
    assert len(passes_192) == 3
    assert len({row["evidence_key"] for row in rows}) == len(rows)
    assert any(row["experiment"]["recovery_level"] == "R1" for row in rows)

    assert runtime.calls[0]["request_fields"] == {"think": False}
    assert runtime.calls[1]["request_fields"] == {"think": True}
    assert [call["options"]["num_predict"] for call in runtime.calls] == [256, 256, 128, 192, 160, 192, 192]

    summary = json.loads((run_dir / "characterization-summary.json").read_text())
    profile = summary["tasks"][0]
    assert profile["minimum_reproduced_pass_budget"] == {"value": 192, "kind": "DERIVED"}
    assert profile["transition_bracket"] == {"lower_fail": 160, "upper_pass": 192, "kind": "DERIVED"}
    assert profile["think_off"]["result_class"] == "ANSWER_WRONG"
    report = (run_dir / "characterization-report.md").read_text()
    assert "MEASURED" in report
    assert "DERIVED" in report
    assert "fabricated per-phase token counts" in report

    assert (run_dir / "replay").exists()
    assert EvidenceStore(tmp_path, run_dir.name).verify_manifest() == []

    progress_rows = [json.loads(line) for line in (run_dir / "progress.jsonl").read_text().splitlines()]
    adjustments = [row for row in progress_rows if row["event"] == "plan_adjusted"]
    assert adjustments
    assert all(row["reason"] == "adaptive experiment added" for row in adjustments)
    assert progress_rows[-1]["event"] == "complete"
    assert progress_rows[-1]["progress"]["percent"] == 100.0
    assert made[0].done == made[0].total_tasks


def test_test1_failure_shape_is_think_truncation_not_capability_failure(tmp_path: Path):
    runtime = AdaptiveRuntime()
    runner = BenchmarkRunner(
        runtime,
        config(),
        suite(),
        results_root=tmp_path,
        telemetry=Telemetry(),
        progress_factory=FakeProgress,
    )
    run_dir = runner.characterize("fake")
    rows = [json.loads(line) for line in (run_dir / "experiments.jsonl").read_text().splitlines()]
    row = next(row for row in rows if row["experiment"]["generation_budget"] == 128)
    assert row["classification"]["result_class"] == "THINK_TRUNCATED"
    assert row["classification"]["valid_for_capability"] is False
    replay = json.loads((run_dir / "replay" / f'{row["experiment"]["experiment_id"]}.json').read_text())
    assert replay["schema_version"] == 2
    assert replay["classification"]["result_class"] == "THINK_TRUNCATED"
    assert replay["generation"]["normalized"]["thinking"]
    assert replay["invocation"]["request_fields"] == {"think": True}


def test_harness_invalid_rows_are_retained_but_cannot_move_capability_boundary():
    def row(budget, result_class, *, valid, thinking=True):
        return {
            "experiment": {
                "task_id": "char-if-001",
                "thinking_mode": thinking,
                "generation_budget": budget,
            },
            "classification": {
                "result_class": result_class,
                "valid_for_capability": valid,
            },
            "score": 1.0 if result_class == "ANSWER_CORRECT" else None,
        }

    rows = [
        row(256, "ANSWER_WRONG", valid=True, thinking=False),
        row(160, "THINK_TRUNCATED", valid=False),
        row(176, "SCORER_DEFECT", valid=False),
        row(192, "ANSWER_CORRECT", valid=True),
        row(192, "ANSWER_CORRECT", valid=True),
        row(192, "ANSWER_CORRECT", valid=True),
    ]
    profile = build_task_profile("char-if-001", rows, boundary_repeats=3)
    assert profile["minimum_reproduced_pass_budget"] == {"value": 192, "kind": "DERIVED"}
    assert profile["transition_bracket"] == {"lower_fail": 160, "upper_pass": 192, "kind": "DERIVED"}
    assert any(r["classification"]["result_class"] == "SCORER_DEFECT" for r in profile["behavioral_observations"])
    assert all(r["classification"]["result_class"] != "SCORER_DEFECT" for r in profile["capability_observations"])
