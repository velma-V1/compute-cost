import base64
import sys

from compute_cost.hardware import collect_hardware_snapshot, run_command_capture


def test_command_capture_preserves_exact_stdout_stderr_bytes():
    code = 'import sys; sys.stdout.buffer.write(b"out\\x00"); sys.stderr.buffer.write(b"err\\xff")'
    result = run_command_capture([sys.executable, "-c", code], timeout=10)

    assert base64.b64decode(result["stdout_b64"]) == b"out\x00"
    assert base64.b64decode(result["stderr_b64"]) == b"err\xff"
    assert result["returncode"] == 0
    assert result["duration_ns"] > 0
    assert result["command"][-2:] == ["-c", code]


def test_hardware_snapshot_has_host_cpu_memory_and_python_sections():
    snapshot = collect_hardware_snapshot()
    assert snapshot["host"]["system"]
    assert snapshot["python"]["version"]
    assert snapshot["cpu"]["logical_count"] >= 1
    assert snapshot["memory"]["total_bytes"] > 0
    assert "available_bytes" in snapshot["memory"]
