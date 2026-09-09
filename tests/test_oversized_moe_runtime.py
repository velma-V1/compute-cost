import json

from compute_cost.runtimes.oversized_moe import HttpExchange, OversizedMoEAdapter


class FakeTransport:
    def __init__(self):
        self.calls = []

    def exchange(self, method, url, body, timeout, stream):
        self.calls.append((method, url, body, timeout, stream))
        now = 1_000_000
        if url.endswith("/health"):
            raw = b'{"status":"ok"}'
            return HttpExchange(200, {"content-type": "application/json"}, [{
                "raw": raw,
                "received_monotonic_ns": now,
                "received_at_utc": "2026-09-09T00:00:00+00:00",
            }])
        if url.endswith("/v1/chat/completions"):
            raw = json.dumps({
                "choices": [{
                    "message": {"role": "assistant", "content": "BLUE"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
                "timings": {"prompt_per_second": 2.0, "predicted_per_second": 1.25},
            }).encode("utf-8")
            return HttpExchange(200, {"content-type": "application/json"}, [{
                "raw": raw,
                "received_monotonic_ns": now,
                "received_at_utc": "2026-09-09T00:00:00+00:00",
            }])
        raise AssertionError(f"unexpected URL: {url}")


def test_oversized_adapter_reports_single_served_model_when_health_is_ok():
    transport = FakeTransport()
    adapter = OversizedMoEAdapter(
        "http://127.0.0.1:8080",
        timeout_s=600,
        served_model="qwen3-next-80b-a3b-instruct-q4_k_m",
        transport=transport,
        monotonic_ns=lambda: 0,
    )

    tags = adapter.list_models()

    assert tags["ok"] is True
    assert tags["parsed"] == {
        "models": [{"name": "qwen3-next-80b-a3b-instruct-q4_k_m", "model": "qwen3-next-80b-a3b-instruct-q4_k_m"}]
    }
    assert adapter.model_available_in(tags, "qwen3-next-80b-a3b-instruct-q4_k_m") is True
    assert adapter.is_model_available("qwen3-next-80b-a3b-instruct-q4_k_m") is True


def test_oversized_adapter_translates_generation_options_and_normalizes_openai_response():
    transport = FakeTransport()
    adapter = OversizedMoEAdapter(
        "http://127.0.0.1:8080",
        timeout_s=600,
        served_model="qwen3-next-80b-a3b-instruct-q4_k_m",
        transport=transport,
        monotonic_ns=lambda: 0,
    )

    result = adapter.generate(
        "qwen3-next-80b-a3b-instruct-q4_k_m",
        [{"role": "user", "content": "Reply exactly BLUE"}],
        {"num_predict": 32, "temperature": 0.0, "seed": 42},
        stream=True,
        request_fields={"think": "medium"},
    )

    assert result["ok"] is True
    assert result["normalized"]["text"] == "BLUE"
    assert result["normalized"]["thinking"] == ""
    assert result["normalized"]["tool_calls"] == []
    assert result["metrics"]["prompt_eval_count"] == 7
    assert result["metrics"]["eval_count"] == 1

    method, url, body, timeout, stream = transport.calls[-1]
    payload = json.loads(body.decode("utf-8"))
    assert method == "POST"
    assert url.endswith("/v1/chat/completions")
    assert timeout == 600
    assert stream is False
    assert payload["messages"] == [{"role": "user", "content": "Reply exactly BLUE"}]
    assert payload["max_tokens"] == 32
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 42
    assert "think" not in payload
