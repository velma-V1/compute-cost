"""Evidence-based compute cost and capability-value synthesis.

This module is deliberately read-only with respect to model execution.  It turns
retained experiment rows and telemetry into measured/derived cost metrics, then
combines those costs with already-proven frontier, reasoning, recovery,
robustness, and compound evidence.  Missing measurements remain unavailable;
they are never silently estimated or converted to zero.
"""

from __future__ import annotations

from statistics import median
from typing import Any, Iterable


MEASURED = "MEASURED"
DERIVED = "DERIVED"
UNAVAILABLE = "UNAVAILABLE"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _metric(value: Any, kind: str, unit: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"value": value, "measurement_kind": kind}
    if unit is not None:
        result["unit"] = unit
    return result


def _sum_or_unknown(values: list[float | None], *, integer: bool = False) -> int | float | None:
    if not values:
        return 0
    if any(value is None for value in values):
        return None
    total = sum(float(value) for value in values if value is not None)
    return int(total) if integer else total


def rollup_experiment_cost(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Strictly sum experiment counters while preserving missing-measurement state."""
    materialized = list(rows)
    expected = len(materialized)
    fields: dict[str, list[float | None]] = {
        "generated_tokens": [],
        "prompt_tokens": [],
        "generation_seconds": [],
        "total_runtime_seconds": [],
        "client_latency_seconds": [],
    }

    for row in materialized:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        timing = row.get("timing") if isinstance(row.get("timing"), dict) else {}
        fields["generated_tokens"].append(_number(metrics.get("eval_count")))
        fields["prompt_tokens"].append(_number(metrics.get("prompt_eval_count")))
        eval_ns = _number(metrics.get("eval_duration_ns"))
        total_ns = _number(metrics.get("total_duration_ns"))
        latency_ns = _number(timing.get("client_latency_ns"))
        fields["generation_seconds"].append(None if eval_ns is None else eval_ns / 1_000_000_000)
        fields["total_runtime_seconds"].append(None if total_ns is None else total_ns / 1_000_000_000)
        fields["client_latency_seconds"].append(None if latency_ns is None else latency_ns / 1_000_000_000)

    coverage = {
        name: {"observed": sum(value is not None for value in values), "expected": expected}
        for name, values in fields.items()
    }
    return {
        "model_calls": expected,
        "generated_tokens": _sum_or_unknown(fields["generated_tokens"], integer=True),
        "prompt_tokens": _sum_or_unknown(fields["prompt_tokens"], integer=True),
        "generation_seconds": _sum_or_unknown(fields["generation_seconds"]),
        "total_runtime_seconds": _sum_or_unknown(fields["total_runtime_seconds"]),
        "client_latency_seconds": _sum_or_unknown(fields["client_latency_seconds"]),
        "coverage": coverage,
    }


def _window(row: dict[str, Any]) -> tuple[int, int] | None:
    timing = row.get("timing") if isinstance(row.get("timing"), dict) else {}
    start = timing.get("client_started_monotonic_ns")
    end = timing.get("client_ended_monotonic_ns")
    if isinstance(start, bool) or not isinstance(start, int):
        return None
    if isinstance(end, bool) or not isinstance(end, int) or end < start:
        return None
    return start, end


def _telemetry_window(row: dict[str, Any], telemetry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    window = _window(row)
    if window is None:
        return []
    start, end = window
    selected: list[dict[str, Any]] = []
    for sample in telemetry:
        timestamp = sample.get("monotonic_ns")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            continue
        if start <= timestamp <= end:
            selected.append(sample)
    return sorted(selected, key=lambda sample: int(sample["monotonic_ns"]))


def _host_values(samples: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        host = sample.get("host") if isinstance(sample.get("host"), dict) else {}
        value = _number(host.get(key))
        if value is not None:
            values.append(value)
    return values


def _gpu_totals(samples: list[dict[str, Any]], key: str) -> list[float]:
    totals: list[float] = []
    for sample in samples:
        gpu = sample.get("gpu") if isinstance(sample.get("gpu"), dict) else {}
        if gpu.get("availability") != "available":
            continue
        values: list[float] = []
        for device in gpu.get("devices") or []:
            if not isinstance(device, dict):
                continue
            value = _number(device.get(key))
            if value is not None:
                values.append(value)
        if values:
            totals.append(sum(values))
    return totals


def _gpu_device_average(samples: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        gpu = sample.get("gpu") if isinstance(sample.get("gpu"), dict) else {}
        if gpu.get("availability") != "available":
            continue
        device_values = [
            value
            for device in gpu.get("devices") or []
            if isinstance(device, dict)
            for value in [_number(device.get(key))]
            if value is not None
        ]
        if device_values:
            values.append(sum(device_values) / len(device_values))
    return values


def _runtime_counter(sample: dict[str, Any], key: str) -> float | None:
    values: list[float] = []
    for process in sample.get("runtime_processes") or []:
        if not isinstance(process, dict):
            continue
        io = process.get("io") if isinstance(process.get("io"), dict) else {}
        value = _number(io.get(key))
        if value is not None:
            values.append(value)
    return None if not values else sum(values)


def _counter_delta(samples: list[dict[str, Any]], key: str) -> float | None:
    observed = [value for sample in samples for value in [_runtime_counter(sample, key)] if value is not None]
    if len(observed) < 2:
        return None
    return max(0.0, observed[-1] - observed[0])


def _integrate_power_wh(samples: list[dict[str, Any]]) -> float | None:
    points: list[tuple[int, float]] = []
    for sample in samples:
        timestamp = sample.get("monotonic_ns")
        power = _number(sample.get("total_gpu_power_w"))
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or power is None:
            continue
        points.append((timestamp, power))
    points.sort()
    if len(points) < 2:
        return None
    watt_seconds = 0.0
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        if t1 <= t0:
            continue
        seconds = (t1 - t0) / 1_000_000_000
        watt_seconds += ((p0 + p1) / 2.0) * seconds
    return watt_seconds / 3600.0


def _mean_metric(values: list[float], unit: str) -> dict[str, Any]:
    if not values:
        return _metric(None, UNAVAILABLE, unit)
    return _metric(sum(values) / len(values), MEASURED, unit)


def _peak_metric(values: list[float], unit: str) -> dict[str, Any]:
    if not values:
        return _metric(None, UNAVAILABLE, unit)
    return _metric(max(values), MEASURED, unit)


def _ratio(numerator: float | int | None, denominator: float | int | None, unit: str) -> dict[str, Any]:
    if numerator is None or denominator is None or float(denominator) <= 0:
        return _metric(None, UNAVAILABLE, unit)
    return _metric(float(numerator) / float(denominator), DERIVED, unit)


def _experiment_cost(row: dict[str, Any], telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    spec = row.get("experiment") if isinstance(row.get("experiment"), dict) else {}
    classification = row.get("classification") if isinstance(row.get("classification"), dict) else {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    timing = row.get("timing") if isinstance(row.get("timing"), dict) else {}
    phase = row.get("phase_metrics") if isinstance(row.get("phase_metrics"), dict) else {}
    samples = _telemetry_window(row, telemetry)

    latency_ns = _number(timing.get("client_latency_ns"))
    if latency_ns is None:
        window = _window(row)
        if window is not None:
            latency_ns = float(window[1] - window[0])
    wall_s = None if latency_ns is None else latency_ns / 1_000_000_000

    prompt_tokens = _number(metrics.get("prompt_eval_count"))
    generated_tokens = _number(metrics.get("eval_count"))
    prompt_ns = _number(metrics.get("prompt_eval_duration_ns"))
    eval_ns = _number(metrics.get("eval_duration_ns"))
    total_ns = _number(metrics.get("total_duration_ns"))
    load_ns = _number(metrics.get("load_duration_ns"))
    ttfa_ns = _number(phase.get("time_to_first_answer_ns"))
    thinking_ns = _number(phase.get("thinking_span_ns"))

    ram = _host_values(samples, "memory_used_bytes")
    cpu = _host_values(samples, "cpu_percent")
    vram = _gpu_totals(samples, "memory_used_mib")
    gpu_util = _gpu_device_average(samples, "utilization_gpu_percent")
    energy_wh = _integrate_power_wh(samples)
    disk_read = _counter_delta(samples, "read_bytes")
    disk_write = _counter_delta(samples, "write_bytes")

    valid = classification.get("valid_for_capability") is True
    success = 1.0 if valid and classification.get("result_class") == "ANSWER_CORRECT" else 0.0

    metric_map: dict[str, Any] = {
        "wall_clock_s": _metric(wall_s, MEASURED if wall_s is not None else UNAVAILABLE, "s"),
        "total_runtime_s": _metric(None if total_ns is None else total_ns / 1e9, MEASURED if total_ns is not None else UNAVAILABLE, "s"),
        "load_s": _metric(None if load_ns is None else load_ns / 1e9, MEASURED if load_ns is not None else UNAVAILABLE, "s"),
        "prompt_tokens": _metric(None if prompt_tokens is None else int(prompt_tokens), MEASURED if prompt_tokens is not None else UNAVAILABLE, "tokens"),
        "generated_tokens": _metric(None if generated_tokens is None else int(generated_tokens), MEASURED if generated_tokens is not None else UNAVAILABLE, "tokens"),
        "prompt_tokens_per_s": _ratio(prompt_tokens, None if prompt_ns is None else prompt_ns / 1e9, "tokens/s"),
        "generation_tokens_per_s": _ratio(generated_tokens, None if eval_ns is None else eval_ns / 1e9, "tokens/s"),
        "time_to_first_answer_s": _metric(None if ttfa_ns is None else ttfa_ns / 1e9, MEASURED if ttfa_ns is not None else UNAVAILABLE, "s"),
        "thinking_span_s": _metric(None if thinking_ns is None else thinking_ns / 1e9, MEASURED if thinking_ns is not None else UNAVAILABLE, "s"),
        "ram_average_bytes": _mean_metric(ram, "bytes"),
        "ram_peak_bytes": _peak_metric(ram, "bytes"),
        "vram_average_mib": _mean_metric(vram, "MiB"),
        "vram_peak_mib": _peak_metric(vram, "MiB"),
        "cpu_average_percent": _mean_metric(cpu, "percent"),
        "gpu_utilization_average_percent": _mean_metric(gpu_util, "percent"),
        "gpu_energy_wh": _metric(energy_wh, DERIVED if energy_wh is not None else UNAVAILABLE, "Wh"),
        "runtime_disk_read_bytes": _metric(disk_read, DERIVED if disk_read is not None else UNAVAILABLE, "bytes"),
        "runtime_disk_write_bytes": _metric(disk_write, DERIVED if disk_write is not None else UNAVAILABLE, "bytes"),
        "successful_capabilities_per_second": _ratio(success, wall_s, "success/s"),
        "successful_capabilities_per_1k_generated_tokens": _ratio(success, None if generated_tokens is None else generated_tokens / 1000.0, "success/1k tokens"),
        "successful_capabilities_per_wh": _ratio(success, energy_wh, "success/Wh"),
    }
    return {
        "experiment_id": spec.get("experiment_id"),
        "family_id": spec.get("task_family") or spec.get("task_id"),
        "difficulty_level": spec.get("difficulty_level"),
        "reasoning_effort": spec.get("reasoning_effort") or "medium",
        "recovery_level": spec.get("recovery_level"),
        "prompt_variant": spec.get("prompt_variant") or "base",
        "valid_for_capability": valid,
        "successful": bool(success),
        "telemetry_sample_count": len(samples),
        "metrics": metric_map,
    }


def build_cost_map(
    model: str,
    rows: Iterable[dict[str, Any]],
    telemetry: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Join each experiment to telemetry inside its monotonic execution window."""
    materialized_rows = list(rows)
    samples = list(telemetry)
    experiments: dict[str, Any] = {}
    for index, row in enumerate(materialized_rows):
        spec = row.get("experiment") if isinstance(row.get("experiment"), dict) else {}
        experiment_id = spec.get("experiment_id")
        key = str(experiment_id) if experiment_id is not None else f"unknown-{index:06d}"
        if key in experiments:
            raise ValueError(f"duplicate experiment_id in cost map: {key}")
        experiments[key] = _experiment_cost(row, samples)
    return {
        "schema_version": 1,
        "model": model,
        "measurement_policy": {
            "experiment_cost": "MONOTONIC_EXECUTION_WINDOW",
            "missing_measurements": "UNAVAILABLE_NOT_ESTIMATED",
            "energy_integration": "TRAPEZOIDAL_TIMESTAMPED_GPU_POWER",
        },
        "experiments": experiments,
        "rollup": rollup_experiment_cost(materialized_rows),
    }


def _median_wall(
    cost_map: dict[str, Any],
    *,
    family_id: str,
    level: int,
    effort: str,
) -> float | None:
    values: list[float] = []
    for experiment in (cost_map.get("experiments") or {}).values():
        if not isinstance(experiment, dict):
            continue
        if experiment.get("family_id") != family_id or experiment.get("difficulty_level") != level:
            continue
        if experiment.get("reasoning_effort") != effort or experiment.get("recovery_level") is not None:
            continue
        prompt_variant = str(experiment.get("prompt_variant") or "base")
        if prompt_variant.startswith("recovery-") or "robust" in prompt_variant:
            continue
        metric = (experiment.get("metrics") or {}).get("wall_clock_s") or {}
        value = _number(metric.get("value")) if isinstance(metric, dict) else None
        if value is not None:
            values.append(value)
    return None if not values else float(median(values))


def _compound_risks(compound_map: dict[str, Any], family_id: str) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    for compound_id, compound in sorted((compound_map.get("compounds") or {}).items()):
        if not isinstance(compound, dict):
            continue
        if family_id not in (compound.get("capabilities_required") or []):
            continue
        risks.append(
            {
                "compound_id": str(compound_id),
                "composition_penalty": compound.get("composition_penalty"),
            }
        )
    return risks


def build_value_map(
    model: str,
    frontiers: dict[str, Any],
    cost_map: dict[str, Any],
    *,
    reasoning_curves: dict[str, Any] | None = None,
    recovery_map: dict[str, Any] | None = None,
    robustness_map: dict[str, Any] | None = None,
    compound_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine measured cost with demonstrated capability effects; never invent a scalar score."""
    reasoning_families = ((reasoning_curves or {}).get("families") or {})
    recovery_families = ((recovery_map or {}).get("families") or {})
    robustness_families = ((robustness_map or {}).get("families") or {})
    compounds = compound_map or {}
    families: dict[str, Any] = {}

    for family_id, frontier in sorted((frontiers.get("families") or {}).items()):
        if not isinstance(frontier, dict):
            continue
        floor = frontier.get("reliable_floor")
        failure = frontier.get("first_failure_level")
        reasoning = reasoning_families.get(family_id) if isinstance(reasoning_families.get(family_id), dict) else {}
        minimum_effort = reasoning.get("minimum_reliable_effort_at_baseline_floor") or "medium"
        high_extension = reasoning.get("demonstrated_high_effort_extension_to")

        baseline_wall = None
        chosen_wall = None
        if isinstance(floor, int) and not isinstance(floor, bool):
            baseline_wall = _median_wall(cost_map, family_id=family_id, level=floor, effort="medium")
            chosen_wall = _median_wall(cost_map, family_id=family_id, level=floor, effort=str(minimum_effort))

        recovery = recovery_families.get(family_id) if isinstance(recovery_families.get(family_id), dict) else {}
        recovery_level = recovery.get("minimum_successful_recovery")
        recovery_difficulty = recovery.get("difficulty_level")
        robustness = robustness_families.get(family_id) if isinstance(robustness_families.get(family_id), dict) else {}
        robustness_value = robustness.get("robustness")
        if robustness_value is None:
            robustness_value = robustness.get("status")

        savings_seconds = None
        savings_fraction = None
        if baseline_wall is not None and chosen_wall is not None:
            savings_seconds = baseline_wall - chosen_wall
            if baseline_wall > 0:
                savings_fraction = savings_seconds / baseline_wall

        families[str(family_id)] = {
            "baseline_medium_frontier": {
                "reliable_floor": floor,
                "first_failure_level": failure,
            },
            "baseline_medium_cost": {"median_wall_clock_s": baseline_wall},
            "cheapest_proven_raw_config": {
                "reasoning_effort": minimum_effort,
                "median_wall_clock_s": chosen_wall,
            },
            "raw_cost_savings_vs_medium": {
                "seconds": savings_seconds,
                "fraction": savings_fraction,
            },
            "high_effort_extension_to": high_extension,
            "minimum_proven_recovery": (
                None
                if recovery_level is None
                else {"difficulty_level": recovery_difficulty, "level": recovery_level}
            ),
            "robustness": robustness_value,
            "compound_risks": _compound_risks(compounds, str(family_id)),
        }

    return {
        "schema_version": 1,
        "model": model,
        "measurement_policy": {
            "cost_source": "cost-map measured and derived evidence",
            "capability_source": "proven frontier/reasoning/recovery/robustness/compound evidence",
            "scalar_value_score": False,
        },
        "families": families,
    }
