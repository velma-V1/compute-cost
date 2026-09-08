import math

from compute_cost.telemetry import TelemetrySampler, integrate_power_wh, parse_nvidia_csv


def test_parse_nvidia_csv_keeps_raw_line_and_normalizes_numeric_fields():
    text = "0, GPU-abc, NVIDIA Test, 555.1, 87, 12288, 4096, 8192, 67, 150.5, 200.0, 1800, 1700, 7000, P2, 45\n"
    rows = parse_nvidia_csv(text)

    assert len(rows) == 1
    gpu = rows[0]
    assert gpu["index"] == 0
    assert gpu["uuid"] == "GPU-abc"
    assert gpu["utilization_gpu_percent"] == 87.0
    assert gpu["memory_used_mib"] == 4096.0
    assert gpu["power_draw_w"] == 150.5
    assert gpu["raw_line"] == text.strip()


def test_parse_nvidia_csv_preserves_extra_columns_instead_of_dropping_them():
    text = "0, GPU-abc, GPU, 555, 1, 10, 2, 8, 50, 20, 30, 100, 100, 100, P8, 0, future-field\n"
    gpu = parse_nvidia_csv(text)[0]
    assert gpu["extra_columns"] == ["future-field"]


def test_sampler_records_raw_collector_failure_as_unavailable():
    def fake_runner(argv, timeout):
        return {
            "command": list(argv),
            "returncode": None,
            "stdout_b64": "",
            "stderr_b64": "",
            "stdout_text": "",
            "stderr_text": "",
            "duration_ns": 10,
            "error": {"type": "FileNotFoundError", "message": "nvidia-smi missing"},
        }

    sample = TelemetrySampler(command_runner=fake_runner).sample()

    assert sample["gpu"]["availability"] == "unavailable"
    assert sample["nvidia_collector"]["error"]["type"] == "FileNotFoundError"
    assert sample["host"]["memory_total_bytes"] > 0
    assert isinstance(sample["runtime_processes"], list)


def test_runtime_process_capture_retains_cpu_memory_io_and_identity_fields():
    class FakeMemory:
        rss = 123
        vms = 456

    class FakeIO:
        read_count = 1
        write_count = 2
        read_bytes = 3
        write_bytes = 4
        other_count = 5
        other_bytes = 6

    class FakeTimes:
        user = 1.25
        system = 0.5

    class FakeProcess:
        pid = 77
        info = {"pid": 77, "name": "ollama.exe", "exe": "C:/Ollama/ollama.exe", "cmdline": ["ollama", "serve"]}

        def cpu_percent(self, interval=None):
            return 12.5

        def memory_info(self):
            return FakeMemory()

        def io_counters(self):
            return FakeIO()

        def cpu_times(self):
            return FakeTimes()

        def num_threads(self):
            return 9

        def num_handles(self):
            return 11

    def fake_process_iter(attrs):
        assert "name" in attrs
        return [FakeProcess()]

    def fake_runner(argv, timeout):
        return {"command": list(argv), "returncode": None, "stdout_b64": "", "stderr_b64": "", "stdout_text": "", "stderr_text": "", "duration_ns": 1, "error": {"type": "FileNotFoundError", "message": "no gpu"}}

    sample = TelemetrySampler(command_runner=fake_runner, process_iter=fake_process_iter).sample()
    proc = sample["runtime_processes"][0]
    assert proc["pid"] == 77
    assert proc["name"] == "ollama.exe"
    assert proc["rss_bytes"] == 123
    assert proc["io"]["read_bytes"] == 3
    assert proc["cpu_times"]["user_s"] == 1.25
    assert proc["num_threads"] == 9
    assert proc["num_handles"] == 11


def test_integrate_power_uses_trapezoidal_rule():
    hour_ns = 3_600_000_000_000
    samples = [
        {"monotonic_ns": 0, "total_gpu_power_w": 100.0},
        {"monotonic_ns": hour_ns, "total_gpu_power_w": 200.0},
    ]

    assert math.isclose(integrate_power_wh(samples), 150.0, rel_tol=1e-9)


def test_integrate_power_is_unavailable_without_two_numeric_samples():
    assert integrate_power_wh([]) is None
    assert integrate_power_wh([{"monotonic_ns": 0, "total_gpu_power_w": None}]) is None
