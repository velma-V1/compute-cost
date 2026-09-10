from types import SimpleNamespace

from compute_cost.comparison_matrix import fixed_comparison_cells
from compute_cost.full_campaign import run_fixed_capability_matrix


class Store:
    def __init__(self):
        self.rows = []

    def append_jsonl(self, path, row):
        self.rows.append((path, row))


class Runner:
    def __init__(self, model):
        self.model = model
        self.store = Store()
        self.progress = None
        self.config = {
            "capability_campaign": {"generation_budget": 256},
            "characterization": {"max_generation_budget": 2048},
            "limits": {"max_model_calls_per_run": 700},
        }


def cases():
    result = []
    for level in (2, 5, 8, 10):
        result.append({
            "id": f"family-a-L{level}",
            "family_id": "family_a",
            "category": "family_a",
            "difficulty_level": level,
            "prompt": f"L{level}",
            "scorer": "exact",
            "expected": "OK",
        })
    return result


def test_fixed_cells_use_identical_task_ids_between_models():
    gpt = fixed_comparison_cells(cases(), "gpt-oss:20b")
    qwen = fixed_comparison_cells(cases(), "qwen3.5:35b-a3b-q4_K_M")
    gpt_ids = {(x["family_id"], x["difficulty_level"], x["task_id"]) for x in gpt}
    qwen_ids = {(x["family_id"], x["difficulty_level"], x["task_id"]) for x in qwen}
    assert gpt_ids == qwen_ids


def test_fixed_matrix_executes_every_supported_reasoning_cell(monkeypatch):
    calls = []

    def fake_execute(runner, case, spec, *, parent=None, **kwargs):
        del runner, parent, kwargs
        calls.append((case["id"], spec.difficulty_level, spec.reasoning_effort, spec.thinking_mode))
        return {
            "experiment": spec.to_dict(),
            "classification": {"result_class": "ANSWER_CORRECT", "valid_for_capability": True},
            "score": 1.0,
            "status": "SCORED",
            "metrics": {"eval_count": 1},
            "timing": {"client_latency_ns": 1_000_000},
            "evidence_refs": {"request_id": spec.experiment_id},
        }

    import compute_cost.capability_campaign as campaign
    monkeypatch.setattr(campaign, "execute_experiment", fake_execute)

    runner = Runner("gpt-oss:20b")
    rows, sequence = run_fixed_capability_matrix(runner, cases(), sequence_start=0)
    assert len(rows) == 12
    assert sequence == 12
    assert {(level, effort) for _, level, effort, _ in calls} == {
        (level, effort) for level in (2, 5, 8, 10) for effort in ("low", "medium", "high")
    }
    assert all(row["comparison"]["scope"] == "fixed_core" for row in rows)
    assert all(row["comparison"]["reasoning_role"] in {"BASELINE_MINIMAL", "ENHANCED", "MAX_NATIVE"} for row in rows)


def test_qwen_fixed_matrix_uses_only_false_and_true(monkeypatch):
    calls = []

    def fake_execute(runner, case, spec, *, parent=None, **kwargs):
        del runner, parent, kwargs
        calls.append((case["id"], spec.thinking_mode, spec.reasoning_effort))
        return {
            "experiment": spec.to_dict(),
            "classification": {"result_class": "ANSWER_CORRECT", "valid_for_capability": True},
            "score": 1.0,
            "status": "SCORED",
            "metrics": {},
            "timing": {},
            "evidence_refs": {"request_id": spec.experiment_id},
        }

    import compute_cost.capability_campaign as campaign
    monkeypatch.setattr(campaign, "execute_experiment", fake_execute)

    runner = Runner("qwen3.5:35b-a3b-q4_K_M")
    rows, _ = run_fixed_capability_matrix(runner, cases())
    assert len(rows) == 8
    assert {thinking for _, thinking, _ in calls} == {False, True}
    assert {effort for _, _, effort in calls} == {None}
