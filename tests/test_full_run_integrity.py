import json
from pathlib import Path
from types import SimpleNamespace

from compute_cost.run_integrity import reconcile_run_integrity
from compute_cost.full_run import run_full_comparability_campaign


class Store:
    def __init__(self, root: Path):
        self.run_dir = root
        self.rows = []
        self.json = {}
        self.raw = {}

    def append_jsonl(self, path, row):
        self.rows.append((path, row))

    def write_json(self, path, value, **kwargs):
        del kwargs
        self.json[path] = value
        target = self.run_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value), encoding="utf-8")
        return {"sha256": "x"}

    def write_raw(self, path, value, **kwargs):
        del kwargs
        self.raw[path] = value
        target = self.run_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
        return {"sha256": "x"}


class Runner:
    def __init__(self, root: Path, model="gpt-oss:20b"):
        self.model = model
        self.store = Store(root)
        self.config = {
            "limits": {"max_model_calls_per_run": 700},
            "capability_campaign": {"generation_budget": 256},
            "characterization": {"max_generation_budget": 2048},
            "autonomous_simulation": {"enabled": True, "scenario_count": 6, "steps_per_scenario": 8},
        }


def _cases(families=2):
    rows = []
    for family_index in range(families):
        family = f"f{family_index}"
        for level in (2, 5, 8, 10):
            rows.append({
                "id": f"{family}-L{level}",
                "family_id": family,
                "category": family,
                "difficulty_level": level,
                "prompt": "x",
                "scorer": "exact",
                "expected": "OK",
            })
    return rows


def test_full_run_executes_fixed_capability_before_autonomy_and_protects_core(monkeypatch, tmp_path):
    runner = Runner(tmp_path)
    order = []

    def fixed(runner_arg, cases, *, sequence_start=0):
        assert runner_arg._mandatory_fixed_remaining == 30  # 2*4*3 + 6*1*? protected below by actual plan helper config turns=8 => 168? overwritten test assertion after setup
        order.append("capability")
        runner_arg._mandatory_fixed_remaining = 144
        return [{"experiment": {"task_family": "f0"}, "comparison": {"scope": "fixed_core"}}], 24

    def autonomy(runner_arg, *, sequence_start=0):
        assert sequence_start == 24
        order.append("autonomy")
        runner_arg._mandatory_fixed_remaining = 0
        return [{"experiment": {"task_family": "autonomous_simulation"}, "comparison": {"scope": "fixed_autonomous"}}], {"schema_version": 2}, 168

    import compute_cost.full_run as module
    monkeypatch.setattr(module, "run_fixed_capability_matrix", fixed)
    monkeypatch.setattr(module, "run_autonomous_matrix", autonomy)
    monkeypatch.setattr(module, "planned_fixed_core", lambda *args, **kwargs: {"capability": 24, "autonomous": 6, "total": 30, "reserve": 670, "limit": 700})

    rows = run_full_comparability_campaign(runner, _cases())
    assert order == ["capability", "autonomy"]
    assert len(rows) == 2
    assert runner._mandatory_fixed_remaining == 0
    assert "comparison-cells.json" in runner.store.json
    assert "autonomous-simulation.json" in runner.store.json


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_integrity_reconciles_runtime_requests_ledger_dossiers_and_failure_atlas(tmp_path):
    requests = tmp_path / "raw" / "runtime" / "requests"
    requests.mkdir(parents=True)
    (requests / "r1.bin").write_bytes(b"1")
    (requests / "r2.bin").write_bytes(b"2")
    _write_json(tmp_path / "call-ledger.json", {"calls_used": 2, "hard_ceiling": 700})
    index = tmp_path / "attempt-dossiers" / "index.jsonl"
    index.parent.mkdir(parents=True)
    index.write_text(
        json.dumps({"experiment_id": "a", "result_class": "ANSWER_CORRECT"}) + "\n" +
        json.dumps({"experiment_id": "b", "result_class": "ANSWER_WRONG"}) + "\n",
        encoding="utf-8",
    )
    _write_json(tmp_path / "failure-atlas.json", {
        "summary": {"failure_count": 1},
        "failures": [{"experiment_id": "b", "result_class": "ANSWER_WRONG"}],
    })
    result = reconcile_run_integrity(tmp_path)
    assert result["ok"] is True
    assert result["counts"] == {
        "ledger_calls_used": 2,
        "runtime_requests": 2,
        "scored_dossiers": 2,
        "failure_dossiers": 1,
        "failure_atlas_entries": 1,
    }
    assert result["issues"] == []


def test_integrity_mismatch_is_explicit_failure(tmp_path):
    requests = tmp_path / "raw" / "runtime" / "requests"
    requests.mkdir(parents=True)
    (requests / "r1.bin").write_bytes(b"1")
    _write_json(tmp_path / "call-ledger.json", {"calls_used": 2, "hard_ceiling": 700})
    index = tmp_path / "attempt-dossiers" / "index.jsonl"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({"experiment_id": "a", "result_class": "ANSWER_CORRECT"}) + "\n", encoding="utf-8")
    _write_json(tmp_path / "failure-atlas.json", {"summary": {"failure_count": 0}, "failures": []})
    result = reconcile_run_integrity(tmp_path)
    assert result["ok"] is False
    assert "LEDGER_RUNTIME_REQUEST_COUNT_MISMATCH" in result["issues"]
    assert "LEDGER_DOSSIER_COUNT_MISMATCH" in result["issues"]
