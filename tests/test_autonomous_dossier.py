import base64
import json

from compute_cost.autonomous_dossier import invocation_from_exchange


def test_autonomous_dossier_recovers_exact_growing_messages_and_request_fields():
    payload = {
        "model": "gpt-oss:20b",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "scenario"},
            {"role": "assistant", "content": "prior state"},
            {"role": "user", "content": "turn 5 update"},
        ],
        "options": {"num_predict": 1024, "temperature": 0.0, "seed": 42},
        "stream": True,
        "think": "low",
    }
    exchange = {
        "request": {
            "body_b64": base64.b64encode(json.dumps(payload).encode()).decode()
        }
    }

    invocation = invocation_from_exchange(exchange)
    assert invocation["model"] == "gpt-oss:20b"
    assert invocation["messages"] == payload["messages"]
    assert invocation["options"] == payload["options"]
    assert invocation["stream"] is True
    assert invocation["request_fields"] == {"think": "low"}
