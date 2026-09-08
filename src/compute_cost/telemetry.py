"""Best-effort telemetry that preserves every raw collector result."""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable

import psutil

from .hardware import run_command_capture
from .schema import unavailable


NVIDIA_FIELDS = [
    "index",
    "uuid",
    "name",
    "driver_version",
    "utilization.gpu",
    "memory.total",
    "memory.used",
    "memory.free",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.gr",
    "clocks.sm",
    "clocks.mem",
    "pstate",
    "fan.speed",
]

NVIDIA_QUERY = "--query-gpu=" + ",".join(NVIDIA_FIELDS)


def _number(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]", "NA"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_nvidia_csv(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        cols = [part.strip() for part in raw_line.split(",")]
        padded = cols + [""] * max(0, len(NVIDIA_FIELDS) - len(cols))
        rows.append(
            {
                "index": int(_number(padded[0]) or 0),
                "uuid": padded[1],
                "name": padded[2],
                "driver_version": padded[3],
                "utilization_gpu_percent": _number(padded[4]),
                "memory_total_mib": _number(padded[5]),
                "memory_used_mib": _number(padded[6]),
                "memory_free_mib": _number(padded[7]),
                "temperature_c": _number(padded[8]),
                "power_draw_w": _number(padded[9]),
                "power_limit_w": _number(padded[10]),
                "graphics_clock_mhz": _number(padded[11]),
                "sm_clock_mhz": _number(padded[12]),
                "memory_clock_mhz": _number(padded[13]),
                "pstate": padded[14],
                "fan_percent": _number(padded[15]),
                "raw_line": raw_line.strip(),
                "raw_columns": cols,
                "extra_columns": cols[len(NVIDIA_FIELDS) :],
            }
        )
    return rows


class TelemetrySampler:
    def __init__(
        self,
        *,
        command_runner: Callable[..., dict[str, Any]] = run_command_capture,
        nvidia_timeout_s: float = 5.0,
    ) -> None:
        self.command_runner = command_runner
        self.nvidia_timeout_s = nvidia_timeout_s
        self.process = psutil.Process(os.getpid())

    def sample(self) -> dict[str, Any]:
        wall = datetime.now(timezone.utc).isoformat()
        mono = time.monotonic_ns()
        vm = psutil.virtual_memory()
        freq = psutil.cpu_freq()
        proc_mem = self.process.memory_info()

        collector = self.command_runner(
            ["nvidia-smi", NVIDIA_QUERY, "--format=csv,noheader,nounits"],
            timeout=self.nvidia_timeout_s,
        )

        if collector.get("error") or collector.get("returncode") not in (0, None):
            detail = collector.get("error") or {
                "type": "CollectorExit",
                "message": f"nvidia-smi exited {collector.get('returncode')}",
            }
            gpu: Any = unavailable(str(detail.get("message", detail)), collector="nvidia-smi")
            total_power = None
        elif collector.get("returncode") is None:
            detail = collector.get("error") or {"message": "nvidia-smi unavailable"}
            gpu = unavailable(str(detail.get("message", detail)), collector="nvidia-smi")
            total_power = None
        else:
            parsed = parse_nvidia_csv(collector.get("stdout_text", ""))
            gpu = {"availability": "available", "devices": parsed}
            powers = [row["power_draw_w"] for row in parsed if row.get("power_draw_w") is not None]
            total_power = sum(powers) if powers else None

        return {
            "timestamp_utc": wall,
            "monotonic_ns": mono,
            "host": {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "cpu_frequency_mhz": None if freq is None else freq.current,
                "memory_total_bytes": vm.total,
                "memory_available_bytes": vm.available,
                "memory_used_bytes": vm.used,
                "memory_percent": vm.percent,
            },
            "process": {
                "pid": self.process.pid,
                "cpu_percent": self.process.cpu_percent(interval=None),
                "rss_bytes": proc_mem.rss,
                "vms_bytes": proc_mem.vms,
            },
            "gpu": gpu,
            "total_gpu_power_w": total_power,
            "nvidia_collector": collector,
        }


def integrate_power_wh(samples: list[dict[str, Any]]) -> float | None:
    points: list[tuple[int, float]] = []
    for sample in samples:
        power = sample.get("total_gpu_power_w")
        mono = sample.get("monotonic_ns")
        if isinstance(power, (int, float)) and math.isfinite(float(power)) and isinstance(mono, int):
            points.append((mono, float(power)))
    if len(points) < 2:
        return None
    points.sort(key=lambda item: item[0])
    watt_seconds = 0.0
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        seconds = (t1 - t0) / 1_000_000_000
        if seconds < 0:
            continue
        watt_seconds += ((p0 + p1) / 2.0) * seconds
    return watt_seconds / 3600.0
