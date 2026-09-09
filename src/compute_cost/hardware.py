"""Host preflight and exact subprocess capture."""

from __future__ import annotations

import base64
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import psutil


def run_command_capture(argv: Sequence[str], timeout: float) -> dict[str, Any]:
    """Run a read-only collector command while preserving exact stdout/stderr bytes."""
    started_wall = datetime.now(timezone.utc).isoformat()
    started_ns = time.monotonic_ns()
    try:
        completed = subprocess.run(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        ended_ns = time.monotonic_ns()
        return {
            "command": list(argv),
            "started_at_utc": started_wall,
            "duration_ns": ended_ns - started_ns,
            "returncode": completed.returncode,
            "stdout_b64": base64.b64encode(completed.stdout).decode("ascii"),
            "stderr_b64": base64.b64encode(completed.stderr).decode("ascii"),
            "stdout_text": completed.stdout.decode("utf-8", errors="replace"),
            "stderr_text": completed.stderr.decode("utf-8", errors="replace"),
            "error": None,
        }
    except subprocess.TimeoutExpired as exc:
        ended_ns = time.monotonic_ns()
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        if isinstance(stdout, str):
            stdout = stdout.encode("utf-8", errors="replace")
        if isinstance(stderr, str):
            stderr = stderr.encode("utf-8", errors="replace")
        return {
            "command": list(argv),
            "started_at_utc": started_wall,
            "duration_ns": ended_ns - started_ns,
            "returncode": None,
            "stdout_b64": base64.b64encode(stdout).decode("ascii"),
            "stderr_b64": base64.b64encode(stderr).decode("ascii"),
            "stdout_text": stdout.decode("utf-8", errors="replace"),
            "stderr_text": stderr.decode("utf-8", errors="replace"),
            "error": {"type": "TimeoutExpired", "message": str(exc)},
        }
    except OSError as exc:
        ended_ns = time.monotonic_ns()
        return {
            "command": list(argv),
            "started_at_utc": started_wall,
            "duration_ns": ended_ns - started_ns,
            "returncode": None,
            "stdout_b64": "",
            "stderr_b64": "",
            "stdout_text": "",
            "stderr_text": "",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def collect_hardware_snapshot() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()
    disk = psutil.disk_usage(Path.cwd().anchor or os.getcwd())
    uname = platform.uname()
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "node": uname.node,
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "cpu": {
            "processor": uname.processor or platform.processor(),
            "logical_count": psutil.cpu_count(logical=True),
            "physical_count": psutil.cpu_count(logical=False),
            "frequency_mhz": None if cpu_freq is None else cpu_freq.current,
        },
        "memory": {
            "total_bytes": vm.total,
            "available_bytes": vm.available,
            "used_bytes": vm.used,
            "percent": vm.percent,
        },
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "percent": disk.percent,
        },
    }
