import json
from pathlib import Path

from compute_cost.report import build_summary, compare_runs, render_report


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_run(root: Path, name: str, score: float = 1.0):
    run = root / name
    run.mkdir()
    (run / "resolved-config.json").write_text(json.dumps({"cost": {"electricity_configured": True, "electricity_per_kwh": 0.20}}), encoding="utf-8")
    (run / "hardware.json").write_text(json.dumps({"memory": {"total_bytes": 64 * 1024**3}}), encoding="utf-8")
    (run / "runtime.json").write_text(json.dumps({"model": "fake", "model_size_bytes": 10_000}), encoding="utf-8")
    write_jsonl(run / "events.jsonl", [
        {"type": "RUN_START", "monotonic_ns": 0},
        {"type": "RUN_END", "monotonic_ns": 10_000_000_000},
    ])
    write_jsonl(run / "telemetry.jsonl", [
        {"monotonic_ns": 0, "total_gpu_power_w": 100.0, "host": {"memory_used_bytes": 8 * 1024**3}, "gpu": {"availability": "available", "devices": [{"memory_used_mib": 4096.0}] }},
        {"monotonic_ns": 3_600_000_000_000, "total_gpu_power_w": 200.0, "host": {"memory_used_bytes": 12 * 1024**3}, "gpu": {"availability": "available", "devices": [{"memory_used_mib": 6144.0}] }},
    ])
    write_jsonl(run / "cases.jsonl", [
        {"case_id": "a", "category": "instruction_following", "stage": "base", "score": score, "status": "SCORED", "timing": {"client_latency_ns": 2_000_000_000}, "metrics": {"eval_count": 20, "eval_duration_ns": 1_000_000_000}},
        {"case_id": "b", "category": "reasoning_math", "stage": "base", "score": 1.0, "status": "SCORED", "timing": {"client_latency_ns": 3_000_000_000}, "metrics": {"eval_count": 30, "eval_duration_ns": 1_500_000_000}},
        {"case_id": "ctx8", "category": "long_context_retrieval", "stage": "context", "score": 1.0, "target_context": 8192, "status": "SCORED"},
        {"case_id": "ctx16", "category": "long_context_retrieval", "stage": "context", "score": 0.0, "target_context": 16384, "status": "SCORED", "stop_boundary": True},
        {"case_id": "s1", "category": "sustained", "stage": "sustained", "score": 1.0, "iteration": 0, "metrics": {"eval_count": 10, "eval_duration_ns": 1_000_000_000}},
        {"case_id": "s2", "category": "sustained", "stage": "sustained", "score": 1.0, "iteration": 1, "metrics": {"eval_count": 10, "eval_duration_ns": 2_000_000_000}},
    ])
    return run


def test_summary_separates_measured_derived_and_cost_metrics(tmp_path: Path):
    run = make_run(tmp_path, "r1")
    summary = build_summary(run)

    assert summary["capability"]["aggregate"] == 1.0
    assert summary["resources"]["peak_ram_bytes"]["classification"] == "measured"
    assert summary["resources"]["peak_vram_mib"]["value"] == 6144.0
    assert summary["energy"]["watt_hours"]["value"] == 150.0
    assert summary["energy"]["electricity_cost"]["value"] == 0.03
    assert summary["energy"]["electricity_cost"]["classification"] == "derived"
    assert summary["context_boundary"]["last_success"] == 8192
    assert summary["context_boundary"]["first_stop"] == 16384
    assert summary["sustained"]["throughput_change_percent"] == -50.0
    assert summary["storage"]["evidence_bytes"]["value"] > 0


def test_report_contains_decision_relevant_sections(tmp_path: Path):
    report = render_report(build_summary(make_run(tmp_path, "r1")))
    assert "Capability" in report
    assert "Compute / Resource Cost" in report
    assert "Context Boundary" in report
    assert "Sustained Load" in report
    assert "Evidence" in report


def test_compare_uses_existing_runs_without_model_execution(tmp_path: Path):
    a = make_run(tmp_path, "a", score=1.0)
    b = make_run(tmp_path, "b", score=0.0)
    comparison = compare_runs([a, b])
    assert len(comparison["runs"]) == 2
    assert comparison["runs"][0]["capability"]["aggregate"] > comparison["runs"][1]["capability"]["aggregate"]
