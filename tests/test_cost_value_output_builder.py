import json
from pathlib import Path


def test_build_cost_value_outputs_uses_complete_run_telemetry_and_phase_maps(tmp_path: Path):
    from compute_cost.runner import _build_cost_value_outputs

    telemetry = [
        {
            "monotonic_ns": 100,
            "host": {"memory_used_bytes": 1000, "cpu_percent": 10.0},
            "gpu": {
                "availability": "available",
                "devices": [
                    {
                        "memory_used_mib": 4000.0,
                        "utilization_gpu_percent": 50.0,
                    }
                ],
            },
            "total_gpu_power_w": 100.0,
            "runtime_processes": [{"io": {"read_bytes": 100, "write_bytes": 200}}],
        },
        {
            "monotonic_ns": 1_000_000_100,
            "host": {"memory_used_bytes": 1200, "cpu_percent": 30.0},
            "gpu": {
                "availability": "available",
                "devices": [
                    {
                        "memory_used_mib": 4500.0,
                        "utilization_gpu_percent": 70.0,
                    }
                ],
            },
            "total_gpu_power_w": 120.0,
            "runtime_processes": [{"io": {"read_bytes": 300, "write_bytes": 500}}],
        },
    ]
    (tmp_path / "telemetry.jsonl").write_text(
        "\n".join(json.dumps(row) for row in telemetry) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "reasoning-curves.json").write_text(
        json.dumps(
            {
                "families": {
                    "math": {
                        "minimum_reliable_effort_at_baseline_floor": "low",
                        "demonstrated_high_effort_extension_to": 5,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "recovery-map.json").write_text(
        json.dumps({"families": {"math": {"difficulty_level": 5, "minimum_successful_recovery": "R3"}}}),
        encoding="utf-8",
    )
    (tmp_path / "robustness-map.json").write_text(
        json.dumps({"families": {"math": {"robustness": "ROBUST"}}}),
        encoding="utf-8",
    )
    (tmp_path / "compound-map.json").write_text(
        json.dumps(
            {
                "compounds": {
                    "math_json": {
                        "capabilities_required": ["math"],
                        "composition_penalty": -1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    rows = [
        {
            "experiment": {
                "experiment_id": "math-medium",
                "task_family": "math",
                "task_id": "math",
                "difficulty_level": 4,
                "reasoning_effort": "medium",
                "prompt_variant": "base",
                "recovery_level": None,
            },
            "classification": {"result_class": "ANSWER_CORRECT", "valid_for_capability": True},
            "metrics": {
                "eval_count": 100,
                "prompt_eval_count": 20,
                "eval_duration_ns": 1_000_000_000,
                "prompt_eval_duration_ns": 100_000_000,
                "total_duration_ns": 1_100_000_000,
                "load_duration_ns": 0,
            },
            "timing": {
                "client_started_monotonic_ns": 100,
                "client_ended_monotonic_ns": 1_000_000_100,
                "client_latency_ns": 1_000_000_000,
            },
            "phase_metrics": {"time_to_first_answer_ns": 50_000_000, "thinking_span_ns": 300_000_000},
        },
        {
            "experiment": {
                "experiment_id": "math-low",
                "task_family": "math",
                "task_id": "math",
                "difficulty_level": 4,
                "reasoning_effort": "low",
                "prompt_variant": "base",
                "recovery_level": None,
            },
            "classification": {"result_class": "ANSWER_CORRECT", "valid_for_capability": True},
            "metrics": {
                "eval_count": 60,
                "prompt_eval_count": 20,
                "eval_duration_ns": 600_000_000,
                "prompt_eval_duration_ns": 100_000_000,
                "total_duration_ns": 700_000_000,
                "load_duration_ns": 0,
            },
            "timing": {
                "client_started_monotonic_ns": 100,
                "client_ended_monotonic_ns": 1_000_000_100,
                "client_latency_ns": 700_000_000,
            },
            "phase_metrics": {"time_to_first_answer_ns": 40_000_000, "thinking_span_ns": 150_000_000},
        },
    ]
    frontiers = {
        "families": {
            "math": {
                "reliable_floor": 4,
                "first_failure_level": 5,
            }
        }
    }

    cost_map, value_map = _build_cost_value_outputs(
        "gpt-oss:20b",
        rows,
        frontiers,
        tmp_path,
    )

    measured = cost_map["experiments"]["math-medium"]["metrics"]
    assert measured["ram_peak_bytes"]["value"] == 1200.0
    assert measured["vram_peak_mib"]["value"] == 4500.0
    assert measured["gpu_energy_wh"]["measurement_kind"] == "DERIVED"
    assert measured["gpu_energy_wh"]["value"] > 0

    family = value_map["families"]["math"]
    assert family["cheapest_proven_raw_config"]["reasoning_effort"] == "low"
    assert family["high_effort_extension_to"] == 5
    assert family["minimum_proven_recovery"] == {"difficulty_level": 5, "level": "R3"}
    assert family["robustness"] == "ROBUST"
    assert family["compound_risks"] == [{"compound_id": "math_json", "composition_penalty": -1}]
