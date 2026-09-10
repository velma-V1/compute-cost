import base64
import json
import re
from pathlib import Path

from compute_cost.autonomous_simulation import build_scenarios
from compute_cost.config import load_config
from compute_cost.runner import BenchmarkRunner


def _exchange(text: str, request_payload: dict):
    payload = {
        "message": {"role": "assistant", "content": text, "thinking": ""},
        "done": True,
        "done_reason": "stop",
        "eval_count": 12,
        "eval_duration": 120_000_000,
        "prompt_eval_count": 20,
        "prompt_eval_duration": 100_000_000,
        "total_duration": 250_000_000,
        "load_duration": 0,
    }
    raw = (json.dumps(payload) + "\n").encode()
    request_raw = json.dumps(request_payload).encode()
    return {
        "ok": True,
        "http_status": 200,
        "request": {
            "body_b64": base64.b64encode(request_raw).decode(),
            "body_text": request_raw.decode(),
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
            "thinking": "",
            "tool_calls": [],
            "done": True,
            "done_reason": "stop",
        },
        "metrics": {
            "eval_count": 12,
            "eval_duration_ns": 120_000_000,
            "prompt_eval_count": 20,
            "prompt_eval_duration_ns": 100_000_000,
            "total_duration_ns": 250_000_000,
            "load_duration_ns": 0,
        },
        "timing": {"first_event_latency_ns": 1, "last_event_latency_ns": 1},
        "phase_metrics": {
            "measurement_kind": "MEASURED",
            "thinking_chunks": 0,
            "answer_chunks": 1,
            "thinking_chars": 0,
            "answer_chars": len(text),
            "time_to_first_thinking_ns": None,
            "time_to_first_answer_ns": 1,
            "thinking_span_ns": 0,
            "answer_span_ns": 0,
        },
    }


class Runtime:
    def __init__(self):
        self.calls = []
        self.scenario = build_scenarios()[0]

    def version(self):
        return {**_exchange("", {}), "parsed": {"version": "test"}}

    def list_models(self):
        return {**_exchange("", {}), "parsed": {"models": [{"name": "gpt-oss:20b", "size": 1234}]}}

    def model_available_in(self, tags, model):
        return True

    def model_info(self, model):
        return {**_exchange("", {}), "parsed": {"details": {"parameter_size": "20B"}}}

    def pull(self, model):
        return {**_exchange("", {}), "parsed_events": [{"status": "success"}]}

    def generate(self, model, messages, options, *, stream=True, request_fields=None):
        self.calls.append({
            "model": model,
            "messages": messages,
            "options": dict(options),
            "request_fields": dict(request_fields or {}),
        })
        last = str(messages[-1]["content"])
        if last.startswith("TURN "):
            match = re.search(r"TURN (\d+)/", last)
            assert match is not None
            step = int(match.group(1))
            expected = self.scenario["steps"][step - 1]
            text = json.dumps({
                "action": expected["expected_action"],
                "checkpoint": expected["expected_checkpoint"],
                "rationale": "follow current evidence and preserve the authoritative state",
                "state": {"checkpoint": expected["expected_checkpoint"], "goal": "preserved"},
            })
        else:
            text = "OK"
        request_payload = {
            "model": model,
            "messages": messages,
            "options": options,
            "stream": stream,
            **(request_fields or {}),
        }
        return _exchange(text, request_payload)


class Telemetry:
    def sample(self):
        return {
            "timestamp_utc": "x",
            "monotonic_ns": 1,
            "host": {"memory_percent": 1.0},
            "gpu": {"availability": "unavailable"},
        }


class Progress:
    def __init__(self, total_tasks):
        self.total_tasks = total_tasks
        self.done = 0

    def start(self, task): pass
    def start_live(self): pass
    def begin_task(self, task): pass
    def complete_task(self, next_task=None): self.done += 1
    def snapshot(self):
        return {"done": self.done, "total": self.total_tasks, "left": max(0, self.total_tasks-self.done), "percent": 0.0, "current_task": "x", "elapsed_s": 0.0, "eta_s": 0.0, "finish": None}
    def stop_live(self, *, newline=True): pass


def _suite():
    family = "acceptance_family"
    return {
        "benchmark_version": "full-comparability-acceptance",
        "schema_version": "capability-suite-v1",
        "taxonomy_version": "capability-taxonomy-v1",
        "coverage": {family: "FULL"},
        "cases": [
            {
                "id": f"{family}-L{level}",
                "family_id": family,
                "category": family,
                "difficulty_level": level,
                "difficulty": {"level": level, "rubric_version": "acceptance-v1", "dimensions": {"load": level}},
                "prompt": f"Capability fixture L{level}; reply exactly OK",
                "scorer": "exact",
                "scorer_version": "1",
                "expected": "OK",
                "timeout_s": 120,
                "capabilities_required": [family],
                "recovery_eligible": True,
                "robustness_eligible": True,
                "compound": False,
                "tags": [],
            }
            for level in (2, 5, 8, 10)
        ],
    }


def test_public_runner_emits_reconciled_full_comparability_artifacts(tmp_path: Path):
    runtime = Runtime()
    cfg = load_config()
    cfg["telemetry"]["background"] = False
    cfg["autonomous_simulation"]["scenario_count"] = 1
    cfg["autonomous_simulation"]["steps_per_scenario"] = 2
    runner = BenchmarkRunner(
        runtime,
        cfg,
        _suite(),
        results_root=tmp_path,
        telemetry=Telemetry(),
        hardware_collector=lambda: {"platform": "test"},
        progress_factory=Progress,
    )

    run_dir = runner.capability_characterize("gpt-oss:20b")

    assert len(runtime.calls) == 18  # 4 fixed tasks*3 efforts + 1 scenario*2 turns*3 efforts
    required = {
        "scorecard.json", "scorecard.md", "task-scorecard.json", "task-scorecard.md",
        "reasoning-comparison.json", "reasoning-comparison.md", "retry-deltas.json", "retry-deltas.md",
        "comparison-cells.json", "call-ledger.json", "model-call-request-index.jsonl",
        "failure-atlas.json", "autonomous-simulation.json", "run-integrity.json",
        "fixed-core-plan.json", "diagnostic-reserve.json",
    }
    assert required.issubset({path.name for path in run_dir.iterdir()})
    assert (run_dir / "attempt-dossiers" / "index.jsonl").is_file()

    integrity = json.loads((run_dir / "run-integrity.json").read_text(encoding="utf-8"))
    assert integrity["ok"] is True
    assert integrity["counts"]["ledger_calls_used"] == 18
    assert integrity["counts"]["runtime_requests"] == 18
    assert integrity["counts"]["scored_dossiers"] == 18

    scorecard = json.loads((run_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["model_summary"]["official_attempts"] == 12
    assert scorecard["model_summary"]["autonomous_tested"] is True
    assert set(scorecard["reasoning_comparison"]["conditions"]) == {
        "BASELINE_MINIMAL", "ENHANCED", "MAX_NATIVE"
    }


def test_missing_fixed_fixture_releases_unused_protected_slot(monkeypatch):
    from compute_cost.full_campaign import run_fixed_capability_matrix
    import compute_cost.capability_campaign as campaign

    class Store:
        def append_jsonl(self, *args, **kwargs): pass

    class Runner:
        model = "gpt-oss:20b"
        store = Store()
        progress = None
        config = {
            "capability_campaign": {"generation_budget": 256},
            "characterization": {"max_generation_budget": 2048},
            "limits": {"max_model_calls_per_run": 700},
        }
        _mandatory_fixed_remaining = 12

    def fake_execute(runner, case, spec, *, parent=None):
        return {
            "experiment": spec.to_dict(),
            "classification": {"result_class": "ANSWER_CORRECT", "valid_for_capability": True},
            "score": 1.0,
            "status": "SCORED",
            "metrics": {}, "timing": {}, "phase_metrics": {}, "evidence_refs": {},
        }

    monkeypatch.setattr(campaign, "execute_experiment", fake_execute)
    cases = [
        {"id": f"f-L{level}", "family_id": "f", "category": "f", "difficulty_level": level, "prompt": "x", "scorer": "exact", "expected": "OK"}
        for level in (2, 5, 8)  # L10 intentionally absent: 3 native GPT cells are missing.
    ]
    rows, _ = run_fixed_capability_matrix(Runner(), cases)
    assert len(rows) == 9
    assert Runner._mandatory_fixed_remaining == 0
