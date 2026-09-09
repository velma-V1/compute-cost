"""Derive decision-oriented reports strictly from retained run evidence."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .telemetry import integrate_power_wh


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _metric(value: Any, classification: str, unit: str | None = None, **extra: Any) -> dict[str, Any]:
    result = {"value": value, "classification": classification}
    if unit is not None:
        result["unit"] = unit
    result.update(extra)
    return result


def _throughput(record: dict[str, Any]) -> float | None:
    metrics = record.get("metrics") or {}
    count = metrics.get("eval_count")
    duration = metrics.get("eval_duration_ns")
    if isinstance(count, (int, float)) and isinstance(duration, (int, float)) and duration > 0:
        return float(count) / (float(duration) / 1_000_000_000)
    return None


def _ns_seconds(value: Any) -> float | None:
    return float(value) / 1_000_000_000 if isinstance(value, (int, float)) else None


def _first_stage(cases: list[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    return next((row for row in cases if row.get("stage") == stage), None)


def _sum_runtime_field(sample: dict[str, Any], field: str) -> float | None:
    values = []
    for proc in sample.get("runtime_processes", []) or []:
        if isinstance(proc, dict) and isinstance(proc.get(field), (int, float)):
            values.append(float(proc[field]))
    return sum(values) if values else None


def _sum_runtime_io(sample: dict[str, Any], field: str) -> float | None:
    values = []
    for proc in sample.get("runtime_processes", []) or []:
        io = proc.get("io") if isinstance(proc, dict) else None
        if isinstance(io, dict) and isinstance(io.get(field), (int, float)):
            values.append(float(io[field]))
    return sum(values) if values else None


def _counter_delta(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return max(0.0, values[-1] - values[0])


def build_summary(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    config = _read_json(run / "resolved-config.json", {})
    runtime = _read_json(run / "runtime.json", {})
    hardware = _read_json(run / "hardware.json", {})
    cases = _read_jsonl(run / "cases.jsonl")
    telemetry = _read_jsonl(run / "telemetry.jsonl")
    events = _read_jsonl(run / "events.jsonl")

    base = [r for r in cases if r.get("stage") == "base" and isinstance(r.get("score"), (int, float))]
    categories: dict[str, list[float]] = {}
    for record in base:
        categories.setdefault(str(record.get("category", "unknown")), []).append(float(record["score"]))
    category_scores = {name: mean(values) for name, values in sorted(categories.items()) if values}
    aggregate = mean(category_scores.values()) if category_scores else None

    cold = _first_stage(cases, "cold") or {}
    warm = _first_stage(cases, "warmup") or {}
    cold_timing = cold.get("timing") or {}
    warm_timing = warm.get("timing") or {}
    cold_metrics = cold.get("metrics") or {}
    warm_metrics = warm.get("metrics") or {}

    ram_values = [
        float(sample.get("host", {}).get("memory_used_bytes"))
        for sample in telemetry
        if isinstance(sample.get("host", {}).get("memory_used_bytes"), (int, float))
    ]
    vram_values: list[float] = []
    gpu_util_values: list[float] = []
    gpu_temp_values: list[float] = []
    runtime_rss_values: list[float] = []
    runtime_cpu_values: list[float] = []
    runtime_read_values: list[float] = []
    runtime_write_values: list[float] = []
    for sample in telemetry:
        rss = _sum_runtime_field(sample, "rss_bytes")
        cpu = _sum_runtime_field(sample, "cpu_percent")
        read_bytes = _sum_runtime_io(sample, "read_bytes")
        write_bytes = _sum_runtime_io(sample, "write_bytes")
        if rss is not None:
            runtime_rss_values.append(rss)
        if cpu is not None:
            runtime_cpu_values.append(cpu)
        if read_bytes is not None:
            runtime_read_values.append(read_bytes)
        if write_bytes is not None:
            runtime_write_values.append(write_bytes)

        gpu = sample.get("gpu") or {}
        if isinstance(gpu, dict) and gpu.get("availability") == "available":
            devices = gpu.get("devices") or []
            vram_values.append(sum(float(d.get("memory_used_mib", 0) or 0) for d in devices if isinstance(d, dict)))
            for device in devices:
                if not isinstance(device, dict):
                    continue
                if isinstance(device.get("utilization_gpu_percent"), (int, float)):
                    gpu_util_values.append(float(device["utilization_gpu_percent"]))
                if isinstance(device.get("temperature_c"), (int, float)):
                    gpu_temp_values.append(float(device["temperature_c"]))

    peak_ram = max(ram_values) if ram_values else None
    peak_vram = max(vram_values) if vram_values else None
    peak_runtime_rss = max(runtime_rss_values) if runtime_rss_values else None
    peak_runtime_cpu = max(runtime_cpu_values) if runtime_cpu_values else None
    runtime_read_delta = _counter_delta(runtime_read_values)
    runtime_write_delta = _counter_delta(runtime_write_values)
    peak_gpu_util = max(gpu_util_values) if gpu_util_values else None
    peak_gpu_temp = max(gpu_temp_values) if gpu_temp_values else None
    peak_gpu_power = max(
        (float(s["total_gpu_power_w"]) for s in telemetry if isinstance(s.get("total_gpu_power_w"), (int, float))),
        default=None,
    )

    energy_wh = integrate_power_wh(telemetry)
    cost_cfg = config.get("cost") or {}
    electricity_configured = bool(cost_cfg.get("electricity_configured", False))
    electricity_rate = cost_cfg.get("electricity_per_kwh")
    electricity_cost = None
    if energy_wh is not None and electricity_configured and isinstance(electricity_rate, (int, float)):
        electricity_cost = (energy_wh / 1000.0) * float(electricity_rate)

    context = sorted(
        [r for r in cases if r.get("stage") == "context" and isinstance(r.get("target_context"), int)],
        key=lambda r: r["target_context"],
    )
    successes = [r["target_context"] for r in context if r.get("score") == 1.0]
    explicit_stops = [
        r["target_context"]
        for r in context
        if r.get("stop_boundary") or r.get("status") in {"FAILED", "STOPPED", "RUNTIME_ERROR"}
    ]
    context_boundary = {
        "last_success": max(successes) if successes else None,
        "first_stop": min(explicit_stops) if explicit_stops else None,
    }

    sustained = sorted(
        [r for r in cases if r.get("stage") == "sustained"],
        key=lambda r: r.get("iteration", 0),
    )
    sustained_rates = [rate for rate in (_throughput(r) for r in sustained) if rate is not None]
    throughput_change = None
    if len(sustained_rates) >= 2 and sustained_rates[0] != 0:
        throughput_change = ((sustained_rates[-1] - sustained_rates[0]) / sustained_rates[0]) * 100.0

    base_seconds = sum(
        float((r.get("timing") or {}).get("client_latency_ns", 0) or 0) / 1_000_000_000 for r in base
    )
    successful = sum(1 for r in base if float(r.get("score", 0)) >= 1.0)
    generated_tokens = sum(
        int((r.get("metrics") or {}).get("eval_count", 0) or 0)
        for r in base
        if isinstance((r.get("metrics") or {}).get("eval_count", 0), (int, float))
    )
    generation_ns = sum(
        float((r.get("metrics") or {}).get("eval_duration_ns", 0) or 0)
        for r in base
        if isinstance((r.get("metrics") or {}).get("eval_duration_ns", 0), (int, float))
    )
    prompt_tokens = sum(
        int((r.get("metrics") or {}).get("prompt_eval_count", 0) or 0)
        for r in base
        if isinstance((r.get("metrics") or {}).get("prompt_eval_count", 0), (int, float))
    )
    prompt_ns = sum(
        float((r.get("metrics") or {}).get("prompt_eval_duration_ns", 0) or 0)
        for r in base
        if isinstance((r.get("metrics") or {}).get("prompt_eval_duration_ns", 0), (int, float))
    )
    generation_tps = None if generation_ns <= 0 else generated_tokens / (generation_ns / 1_000_000_000)
    prompt_tps = None if prompt_ns <= 0 else prompt_tokens / (prompt_ns / 1_000_000_000)

    peak_vram_gib = None if peak_vram is None else peak_vram / 1024.0
    efficiency = {
        "capability_per_second": None if aggregate is None or base_seconds <= 0 else aggregate / base_seconds,
        "capability_per_peak_vram_gib": None if aggregate is None or not peak_vram_gib else aggregate / peak_vram_gib,
        "capability_per_wh": None if aggregate is None or not energy_wh else aggregate / energy_wh,
        "successful_cases_per_minute": None if base_seconds <= 0 else successful / (base_seconds / 60.0),
    }

    run_start = next((e.get("monotonic_ns") for e in events if e.get("type") == "RUN_START"), None)
    run_end = next((e.get("monotonic_ns") for e in reversed(events) if e.get("type") == "RUN_END"), None)
    run_seconds = None
    if isinstance(run_start, int) and isinstance(run_end, int) and run_end >= run_start:
        run_seconds = (run_end - run_start) / 1_000_000_000

    evidence_bytes = 0
    for path in run.rglob("*"):
        if path.is_file() and path.name not in {"manifest.json"}:
            try:
                evidence_bytes += path.stat().st_size
            except OSError:
                pass

    model_size = runtime.get("model_size_bytes")
    pull = runtime.get("pull") if isinstance(runtime.get("pull"), dict) else {}
    summary: dict[str, Any] = {
        "run_id": run.name,
        "model": runtime.get("model"),
        "onboarding": {
            "model_was_local": runtime.get("was_local_before_run"),
            "pull_observed": bool(pull.get("observed", False)),
            "pull_seconds": _metric(_ns_seconds((pull.get("timing") or {}).get("last_event_latency_ns")), "measured", "s"),
            "cold_load_seconds": _metric(_ns_seconds(cold_metrics.get("load_duration_ns")), "measured", "s"),
            "cold_first_event_seconds": _metric(_ns_seconds(cold_timing.get("first_event_latency_ns")), "measured", "s"),
            "cold_client_latency_seconds": _metric(_ns_seconds(cold_timing.get("client_latency_ns")), "measured", "s"),
            "warm_load_seconds": _metric(_ns_seconds(warm_metrics.get("load_duration_ns")), "measured", "s"),
            "warm_first_event_seconds": _metric(_ns_seconds(warm_timing.get("first_event_latency_ns")), "measured", "s"),
            "warm_client_latency_seconds": _metric(_ns_seconds(warm_timing.get("client_latency_ns")), "measured", "s"),
        },
        "capability": {
            "aggregate": aggregate,
            "categories": category_scores,
            "base_cases": len(base),
            "successful_cases": successful,
        },
        "throughput": {
            "generated_tokens": generated_tokens,
            "generation_tokens_per_second": generation_tps,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_per_second": prompt_tps,
        },
        "timing": {
            "run_seconds": _metric(run_seconds, "measured", "s"),
            "base_case_seconds": _metric(base_seconds, "derived", "s"),
        },
        "resources": {
            "peak_ram_bytes": _metric(peak_ram, "measured", "bytes", scope="host"),
            "peak_vram_mib": _metric(peak_vram, "measured", "MiB", scope="all sampled NVIDIA GPUs"),
            "peak_runtime_rss_bytes": _metric(peak_runtime_rss, "measured", "bytes", scope="identified Ollama processes"),
            "peak_runtime_cpu_percent": _metric(peak_runtime_cpu, "measured", "percent", scope="identified Ollama processes"),
            "runtime_read_bytes_delta": _metric(runtime_read_delta, "derived", "bytes", source="runtime process cumulative I/O counters"),
            "runtime_write_bytes_delta": _metric(runtime_write_delta, "derived", "bytes", source="runtime process cumulative I/O counters"),
            "peak_gpu_utilization_percent": _metric(peak_gpu_util, "measured", "percent"),
            "peak_gpu_temperature_c": _metric(peak_gpu_temp, "measured", "C"),
        },
        "energy": {
            "watt_hours": _metric(energy_wh, "derived", "Wh", source="timestamped total sampled GPU power"),
            "peak_gpu_power_w": _metric(peak_gpu_power, "measured", "W"),
            "electricity_rate": _metric(electricity_rate if electricity_configured else None, "estimated", "currency/kWh"),
            "electricity_cost": _metric(electricity_cost, "derived", "currency"),
        },
        "storage": {
            "model_size_bytes": _metric(model_size, "measured", "bytes"),
            "evidence_bytes": _metric(evidence_bytes, "measured", "bytes", scope="pre-manifest/pre-self-referential summary size"),
        },
        "context_boundary": context_boundary,
        "sustained": {
            "iterations": len(sustained),
            "initial_tokens_per_second": sustained_rates[0] if sustained_rates else None,
            "final_tokens_per_second": sustained_rates[-1] if sustained_rates else None,
            "throughput_change_percent": throughput_change,
        },
        "efficiency": efficiency,
        "hardware": hardware,
    }
    return summary


def render_report(summary: dict[str, Any]) -> str:
    onboarding = summary.get("onboarding", {})
    capability = summary.get("capability", {})
    throughput = summary.get("throughput", {})
    resources = summary.get("resources", {})
    energy = summary.get("energy", {})
    storage = summary.get("storage", {})
    context = summary.get("context_boundary", {})
    sustained = summary.get("sustained", {})
    efficiency = summary.get("efficiency", {})

    category_lines = "\n".join(
        f"- {name}: {value:.3f}" for name, value in capability.get("categories", {}).items()
    ) or "- unavailable"
    return f"""# Compute Cost Report — {summary.get('run_id')}

## Onboarding Cost

- Model already local: {onboarding.get('model_was_local')}
- Pull observed: {onboarding.get('pull_observed')}
- Pull duration: {onboarding.get('pull_seconds', {}).get('value')} s
- Cold load: {onboarding.get('cold_load_seconds', {}).get('value')} s
- Cold first observable event: {onboarding.get('cold_first_event_seconds', {}).get('value')} s
- Cold client latency: {onboarding.get('cold_client_latency_seconds', {}).get('value')} s
- Warm first observable event: {onboarding.get('warm_first_event_seconds', {}).get('value')} s
- Warm client latency: {onboarding.get('warm_client_latency_seconds', {}).get('value')} s

## Capability

- Aggregate baseline: {capability.get('aggregate')}
- Successful base cases: {capability.get('successful_cases')} / {capability.get('base_cases')}
{category_lines}

## Throughput

- Generated tokens: {throughput.get('generated_tokens')}
- Generation throughput: {throughput.get('generation_tokens_per_second')} tok/s
- Prompt tokens: {throughput.get('prompt_tokens')}
- Prompt processing throughput: {throughput.get('prompt_tokens_per_second')} tok/s

## Compute / Resource Cost

- Peak host RAM: {resources.get('peak_ram_bytes', {}).get('value')} bytes [measured]
- Peak Ollama-process RSS: {resources.get('peak_runtime_rss_bytes', {}).get('value')} bytes [measured]
- Peak Ollama-process CPU: {resources.get('peak_runtime_cpu_percent', {}).get('value')}% [measured]
- Ollama read delta: {resources.get('runtime_read_bytes_delta', {}).get('value')} bytes [derived]
- Ollama write delta: {resources.get('runtime_write_bytes_delta', {}).get('value')} bytes [derived]
- Peak VRAM: {resources.get('peak_vram_mib', {}).get('value')} MiB [measured]
- Peak GPU utilization: {resources.get('peak_gpu_utilization_percent', {}).get('value')}% [measured]
- Peak GPU temperature: {resources.get('peak_gpu_temperature_c', {}).get('value')} C [measured]
- Peak GPU power: {energy.get('peak_gpu_power_w', {}).get('value')} W [measured]
- Energy: {energy.get('watt_hours', {}).get('value')} Wh [derived from measured samples]
- Electricity cost: {energy.get('electricity_cost', {}).get('value')} [derived; unavailable unless rate configured]
- Model storage: {storage.get('model_size_bytes', {}).get('value')} bytes [measured when runtime exposes it]

## Context Boundary

- Last successful target: {context.get('last_success')}
- First failed/stopped target: {context.get('first_stop')}

## Sustained Load

- Iterations: {sustained.get('iterations')}
- Initial generation throughput: {sustained.get('initial_tokens_per_second')} tok/s
- Final generation throughput: {sustained.get('final_tokens_per_second')} tok/s
- Throughput change: {sustained.get('throughput_change_percent')}%

## Usable Efficiency

- Capability / second: {efficiency.get('capability_per_second')}
- Capability / peak VRAM GiB: {efficiency.get('capability_per_peak_vram_gib')}
- Capability / Wh: {efficiency.get('capability_per_wh')}
- Successful cases / minute: {efficiency.get('successful_cases_per_minute')}

## Evidence

- Evidence bytes retained: {storage.get('evidence_bytes', {}).get('value')}
- Raw evidence is authoritative; this report is a derived view.
"""


def compare_runs(run_dirs: Iterable[str | Path]) -> dict[str, Any]:
    summaries = [build_summary(path) for path in run_dirs]
    return {
        "runs": summaries,
        "comparison_fields": [
            "onboarding",
            "capability.aggregate",
            "throughput",
            "timing.base_case_seconds",
            "resources",
            "energy",
            "storage.model_size_bytes",
            "context_boundary.last_success",
            "sustained.throughput_change_percent",
            "efficiency",
        ],
    }
