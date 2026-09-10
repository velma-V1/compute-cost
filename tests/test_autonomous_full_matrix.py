import json

from compute_cost.autonomous_matrix import run_autonomous_matrix, semantic_action_equivalent
from compute_cost.autonomous_simulation import build_scenarios


class Store:
    def __init__(self):
        self.rows = []
        self.json = {}
        self.run_id = "fake-run"

    def append_jsonl(self, path, row):
        self.rows.append((path, row))

    def write_json(self, path, value, **kwargs):
        self.json[path] = value
        return {"sha256": f"sha-{len(self.json)}"}


class Runner:
    def __init__(self, model):
        self.model = model
        self.store = Store()
        self._recent_telemetry = []
        self.config = {
            "autonomous_simulation": {
                "enabled": True,
                "scenario_count": 6,
                "steps_per_scenario": 8,
                "generation_budget": 512,
                "max_generation_budget": 2048,
            },
            "limits": {"request_timeout_s": 120, "max_model_calls_per_run": 700},
        }
        self._expected = {
            (scenario["id"], i): (step["expected_action"], step["expected_checkpoint"])
            for scenario in build_scenarios()
            for i, step in enumerate(scenario["steps"], start=1)
        }

    def _generation_options(self, case, options):
        return options

    def _invoke_generation(self, *, stage, case_id, messages, options, request_fields=None):
        assert stage == "autonomous-simulation"
        marker = case_id.split("--", 1)[0]
        scenario_step = marker.removeprefix("auto-").rsplit("-s", 1)
        scenario = scenario_step[0]
        step = int(scenario_step[1][:2])
        action, checkpoint = self._expected[(scenario, step)]
        text = json.dumps({
            "action": action,
            "checkpoint": checkpoint,
            "rationale": "follow evidence and preserve state",
            "state": {"checkpoint": checkpoint},
        })
        generation = {
            "ok": True,
            "http_status": 200,
            "normalized": {"text": text, "thinking": "", "tool_calls": [], "done_reason": "stop"},
            "metrics": {
                "prompt_eval_count": 100,
                "eval_count": 20,
                "prompt_eval_duration_ns": 100_000_000,
                "eval_duration_ns": 200_000_000,
            },
            "timing": {"client_latency_ns": 350_000_000},
            "phase_metrics": {},
        }
        invocation = {
            "model": self.model,
            "messages": messages,
            "options": options,
            "stream": True,
            "request_fields": request_fields or {},
        }
        refs = {"request_id": case_id, "request": {}, "response": {}, "streams": []}
        return generation, invocation, refs

    def _persist_scoring(self, case_id, stage, scoring):
        self.store.write_json(f"raw/scoring/{stage}-{case_id}.json", scoring)

    def _write_replay(self, **kwargs):
        raise AssertionError("all fake autonomous turns should pass")


def test_semantic_action_equivalence_is_declared_before_run():
    assert semantic_action_equivalent("repo_repair", 6, "RETEST", "TEST") is True
    assert semantic_action_equivalent("repo_repair", 6, "RETEST", "COMPLETE") is False
    assert semantic_action_equivalent("diagnostic_root_cause", 4, "ELIMINATE", "TEST") is True


def test_gpt_autonomous_matrix_scores_all_144_turn_cells():
    runner = Runner("gpt-oss:20b")
    rows, summary, sequence = run_autonomous_matrix(runner)
    assert len(rows) == 144
    assert sequence == 144
    assert summary["semantic_turns"] == 144
    assert set(summary["reasoning_conditions"]) == {"BASELINE_MINIMAL", "ENHANCED", "MAX_NATIVE"}
    assert len({row["evidence_key"] for row in rows}) == 144
    assert all("score_vector" in row for row in rows)
    assert all(row["score_vector"]["state_checkpoint_accuracy"] == 100.0 for row in rows)
    dossier_rows = [row for path, row in runner.store.rows if path == "attempt-dossiers/index.jsonl"]
    assert len(dossier_rows) == 144


def test_qwen_autonomous_matrix_scores_all_96_turn_cells():
    runner = Runner("qwen3.5:35b-a3b-q4_K_M")
    rows, summary, _ = run_autonomous_matrix(runner)
    assert len(rows) == 96
    assert summary["semantic_turns"] == 96
    assert set(summary["reasoning_conditions"]) == {"BASELINE_MINIMAL", "ENHANCED"}
    request_values = {row["comparison"]["native_reasoning_value"] for row in rows}
    assert request_values == {False, True}


def test_semantic_equivalent_action_keeps_semantic_credit_but_loses_exact_contract():
    runner = Runner("qwen3.5:35b-a3b-q4_K_M")
    original = runner._invoke_generation

    def equivalent_turn(**kwargs):
        generation, invocation, refs = original(**kwargs)
        if kwargs["case_id"].startswith("auto-repo_repair-s06"):
            payload = json.loads(generation["normalized"]["text"])
            payload["action"] = "TEST"
            generation["normalized"]["text"] = json.dumps(payload)
        return generation, invocation, refs

    runner._invoke_generation = equivalent_turn
    rows, _, _ = run_autonomous_matrix(runner)
    target = next(row for row in rows if row["evidence_key"].startswith("auto-repo_repair-s06--baseline_minimal"))
    assert target["score_vector"]["semantic_correctness"] == 100.0
    assert target["score_vector"]["decision_quality"] == 100.0
    assert target["score_vector"]["exact_action_contract_compliance"] == 0.0
