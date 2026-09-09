import math


def row(
    experiment_id="math-medium",
    *,
    family="arithmetic_numerical_reasoning",
    level=4,
    effort="medium",
    wall_s=2.0,
    generated=50,
    result="ANSWER_CORRECT",
    recovery_level=None,
    prompt_variant="base",
):
    start = 1_000_000_000
    end = start + int(wall_s * 1_000_000_000)
    return {
        "experiment": {
            "experiment_id": experiment_id,
            "task_family": family,
            "task_id": family,
            "difficulty_level": level,
            "thinking_mode": True,
            "reasoning_effort": effort,
            "generation_budget": 256,
            "prompt_variant": prompt_variant,
            "recovery_level": recovery_level,
        },
        "classification": {
            "result_class": result,
            "valid_for_capability": True,
        },
        "metrics": {
            "total_duration_ns": int(wall_s * 1_000_000_000),
            "load_duration_ns": 200_000_000,
            "prompt_eval_count": 100,
            "prompt_eval_duration_ns": 500_000_000,
            "eval_count": generated,
            "eval_duration_ns": 1_000_000_000,
        },
        "timing": {
            "client_started_monotonic_ns": start,
            "client_ended_monotonic_ns": end,
            "client_latency_ns": end - start,
        },
        "phase_metrics": {
            "time_to_first_answer_ns": 800_000_000,
            "thinking_span_ns": 400_000_000,
        },
    }


def telemetry(t_s, *, ram, vram, cpu, gpu_util, power, read_bytes, write_bytes):
    return {
        "monotonic_ns": int(t_s * 1_000_000_000),
        "host": {"memory_used_bytes": ram, "cpu_percent": cpu},
        "gpu": {
            "availability": "available",
            "devices": [
                {
                    "memory_used_mib": vram,
                    "utilization_gpu_percent": gpu_util,
                    "power_draw_w": power,
                }
            ],
        },
        "total_gpu_power_w": power,
        "runtime_processes": [
            {
                "pid": 10,
                "io": {"read_bytes": read_bytes, "write_bytes": write_bytes},
            }
        ],
    }


def test_cost_map_joins_telemetry_by_monotonic_window_and_preserves_measurement_kind():
    from compute_cost.cost_value import build_cost_map

    samples = [
        telemetry(0.5, ram=5, vram=500, cpu=5, gpu_util=5, power=50, read_bytes=100, write_bytes=20),
        telemetry(1.0, ram=10, vram=1000, cpu=20, gpu_util=10, power=100, read_bytes=1000, write_bytes=200),
        telemetry(2.0, ram=20, vram=2000, cpu=40, gpu_util=50, power=200, read_bytes=3000, write_bytes=500),
        telemetry(3.0, ram=15, vram=1500, cpu=30, gpu_util=30, power=100, read_bytes=5000, write_bytes=1000),
        telemetry(3.5, ram=99, vram=9999, cpu=99, gpu_util=99, power=999, read_bytes=9999, write_bytes=9999),
    ]

    result = build_cost_map("gpt-oss:20b", [row()], samples)
    exp = result["experiments"]["math-medium"]
    metrics = exp["metrics"]

    assert exp["telemetry_sample_count"] == 3
    assert metrics["wall_clock_s"] == {"value": 2.0, "measurement_kind": "MEASURED", "unit": "s"}
    assert metrics["prompt_tokens"]["value"] == 100
    assert metrics["generated_tokens"]["value"] == 50
    assert metrics["prompt_tokens_per_s"]["value"] == 200.0
    assert metrics["generation_tokens_per_s"]["value"] == 50.0
    assert metrics["ram_average_bytes"]["value"] == 15.0
    assert metrics["ram_peak_bytes"]["value"] == 20.0
    assert metrics["vram_average_mib"]["value"] == 1500.0
    assert metrics["vram_peak_mib"]["value"] == 2000.0
    assert metrics["cpu_average_percent"]["value"] == 30.0
    assert metrics["gpu_utilization_average_percent"]["value"] == 30.0
    assert math.isclose(metrics["gpu_energy_wh"]["value"], 1 / 12, rel_tol=1e-9)
    assert metrics["gpu_energy_wh"]["measurement_kind"] == "DERIVED"
    assert metrics["runtime_disk_read_bytes"]["value"] == 4000.0
    assert metrics["runtime_disk_write_bytes"]["value"] == 800.0
    assert metrics["successful_capabilities_per_second"]["value"] == 0.5
    assert metrics["successful_capabilities_per_1k_generated_tokens"]["value"] == 20.0
    assert math.isclose(metrics["successful_capabilities_per_wh"]["value"], 12.0, rel_tol=1e-9)


def test_cost_map_marks_energy_and_resource_metrics_unavailable_instead_of_estimating():
    from compute_cost.cost_value import build_cost_map

    result = build_cost_map("gpt-oss:20b", [row()], [])
    metrics = result["experiments"]["math-medium"]["metrics"]

    for key in (
        "ram_average_bytes",
        "ram_peak_bytes",
        "vram_average_mib",
        "vram_peak_mib",
        "gpu_energy_wh",
        "runtime_disk_read_bytes",
    ):
        assert metrics[key]["value"] is None
        assert metrics[key]["measurement_kind"] == "UNAVAILABLE"
    assert metrics["successful_capabilities_per_wh"]["measurement_kind"] == "UNAVAILABLE"


def test_value_map_uses_only_proven_reasoning_and_recovery_evidence_for_routing_economics():
    from compute_cost.cost_value import build_cost_map, build_value_map

    rows = [
        row("medium-1", wall_s=2.0),
        row("medium-2", wall_s=2.2),
        row("low-1", effort="low", wall_s=1.0),
        row("low-2", effort="low", wall_s=1.2),
        row("high-1", level=5, effort="high", wall_s=3.0),
        row("high-2", level=5, effort="high", wall_s=3.2),
        row("r4-1", level=5, wall_s=2.5, recovery_level="R4", prompt_variant="recovery-r4"),
    ]
    cost_map = build_cost_map("gpt-oss:20b", rows, [])
    frontiers = {
        "families": {
            "arithmetic_numerical_reasoning": {
                "reliable_floor": 4,
                "first_failure_level": 5,
            }
        }
    }
    reasoning = {
        "families": {
            "arithmetic_numerical_reasoning": {
                "minimum_reliable_effort_at_baseline_floor": "low",
                "demonstrated_high_effort_extension_to": 5,
            }
        }
    }
    recovery = {
        "families": {
            "arithmetic_numerical_reasoning": {
                "difficulty_level": 5,
                "minimum_successful_recovery": "R4",
            }
        }
    }
    robustness = {
        "families": {
            "arithmetic_numerical_reasoning": {"robustness": "ROBUST"}
        }
    }
    compound = {
        "compounds": {
            "extract_calculate_json": {
                "capabilities_required": [
                    "extraction_transformation",
                    "arithmetic_numerical_reasoning",
                    "strict_structured_output",
                ],
                "composition_penalty": -2,
            }
        }
    }

    value = build_value_map(
        "gpt-oss:20b",
        frontiers,
        cost_map,
        reasoning_curves=reasoning,
        recovery_map=recovery,
        robustness_map=robustness,
        compound_map=compound,
    )
    family = value["families"]["arithmetic_numerical_reasoning"]

    assert family["baseline_medium_frontier"] == {"reliable_floor": 4, "first_failure_level": 5}
    assert family["cheapest_proven_raw_config"]["reasoning_effort"] == "low"
    assert math.isclose(family["cheapest_proven_raw_config"]["median_wall_clock_s"], 1.1)
    assert math.isclose(family["baseline_medium_cost"]["median_wall_clock_s"], 2.1)
    assert math.isclose(family["raw_cost_savings_vs_medium"]["seconds"], 1.0)
    assert math.isclose(family["raw_cost_savings_vs_medium"]["fraction"], 1.0 / 2.1)
    assert family["high_effort_extension_to"] == 5
    assert family["minimum_proven_recovery"] == {"difficulty_level": 5, "level": "R4"}
    assert family["robustness"] == "ROBUST"
    assert family["compound_risks"] == [
        {"compound_id": "extract_calculate_json", "composition_penalty": -2}
    ]
