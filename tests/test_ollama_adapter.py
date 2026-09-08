import base64
import json

from compute_cost.runtimes.ollama import HttpExchange, OllamaAdapter


class FakeTransport:
    def __init__(self, exchanges):
        self.exchanges = list(exchanges)
        self.calls = []

    def exchange(self, method, url, body, timeout, stream):
        self.calls.append({"method": method, "url": url, "body": body, "timeout": timeout, "stream": stream})
        return self.exchanges.pop(0)


def chunk(raw: bytes, mono: int):
    return {"raw": raw, "received_monotonic_ns": mono, "received_at_utc": f"t-{mono}"}


def test_generate_preserves_exact_request_stream_chunks_unknown_fields_and_thinking():
    events = [
        b'{"model":"fake","message":{"role":"assistant","thinking":"plan","content":"Hi"},"done":false}\n',
        b'{"model":"fake","message":{"role":"assistant","content":" there","tool_calls":[{"function":{"name":"lookup","arguments":{"x":1}}}]},"done":true,"done_reason":"stop","total_duration":100,"load_duration":10,"prompt_eval_count":4,"prompt_eval_duration":20,"eval_count":2,"eval_duration":30,"future_metric":{"x":9}}\n',
    ]
    transport = FakeTransport([
        HttpExchange(200, {"content-type": "application/x-ndjson"}, [chunk(events[0], 110), chunk(events[1], 130)])
    ])
    adapter = OllamaAdapter("http://127.0.0.1:11434", transport=transport, monotonic_ns=lambda: 100)

    result = adapter.generate(
        "fake",
        [{"role": "user", "content": "hello"}],
        {"temperature": 0, "seed": 42},
        stream=True,
        request_fields={"think": True, "logprobs": True},
    )

    sent = json.loads(base64.b64decode(result["request"]["body_b64"]))
    assert sent["model"] == "fake"
    assert sent["messages"] == [{"role": "user", "content": "hello"}]
    assert sent["options"]["seed"] == 42
    assert sent["think"] is True
    assert result["raw_response_b64"] == base64.b64encode(b"".join(events)).decode("ascii")
    assert [base64.b64decode(e["raw_b64"]) for e in result["stream_events"]] == events
    assert result["stream_events"][1]["parsed"]["future_metric"] == {"x": 9}
    assert result["normalized"]["text"] == "Hi there"
    assert result["normalized"]["thinking"] == "plan"
    assert result["normalized"]["tool_calls"][0]["function"]["name"] == "lookup"
    assert result["metrics"]["load_duration_ns"] == 10
    assert result["metrics"]["eval_count"] == 2
    assert result["timing"]["first_event_latency_ns"] == 10


def test_http_error_retains_raw_error_body_and_status():
    raw = b'{"error":"model failed","future_error_field":17}'
    transport = FakeTransport([HttpExchange(500, {"content-type": "application/json"}, [chunk(raw, 200)])])
    adapter = OllamaAdapter("http://127.0.0.1:11434", transport=transport, monotonic_ns=lambda: 100)

    result = adapter.generate("fake", [{"role": "user", "content": "x"}], {}, stream=False)

    assert result["ok"] is False
    assert result["http_status"] == 500
    assert base64.b64decode(result["raw_response_b64"]) == raw
    assert result["error"]["raw_parsed"]["future_error_field"] == 17


def test_tags_and_show_keep_entire_raw_payload():
    tags_raw = b'{"models":[{"name":"fake:latest","size":123,"future":true}]}'
    show_raw = b'{"details":{"parameter_size":"9B"},"model_info":{"context_length":32768},"future":{"a":1}}'
    transport = FakeTransport([
        HttpExchange(200, {}, [chunk(tags_raw, 1)]),
        HttpExchange(200, {}, [chunk(show_raw, 2)]),
    ])
    adapter = OllamaAdapter("http://127.0.0.1:11434", transport=transport)

    tags = adapter.list_models()
    info = adapter.model_info("fake:latest")

    assert tags["parsed"]["models"][0]["future"] is True
    assert info["parsed"]["future"] == {"a": 1}
    assert adapter.model_available_in(tags, "fake:latest") is True
