"""Minimal lossless adapter for an already-running oversized-moe OpenAI-compatible server."""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable

from .ollama import HttpExchange, UrllibTransport


class OversizedMoEAdapter:
    """Adapt one persistent oversized-moe/llama.cpp server to the benchmark runtime protocol."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8080",
        *,
        timeout_s: float = 600.0,
        served_model: str = "qwen3-next-80b-a3b-instruct-q4_k_m",
        transport: Any | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.served_model = served_model
        self.transport = transport or UrllibTransport()
        self.monotonic_ns = monotonic_ns

    @staticmethod
    def _encode(payload: dict[str, Any] | None) -> bytes:
        if payload is None:
            return b""
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _safe_parse(raw: bytes) -> tuple[Any | None, str | None]:
        if not raw:
            return None, None
        try:
            return json.loads(raw.decode("utf-8")), None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        body = self._encode(payload)
        started_ns = self.monotonic_ns()
        started_wall = datetime.now(timezone.utc).isoformat()
        exchange = self.transport.exchange(method, f"{self.endpoint}{path}", body, self.timeout_s, stream)

        stream_events: list[dict[str, Any]] = []
        raw_parts: list[bytes] = []
        for sequence, item in enumerate(exchange.chunks):
            raw = item.get("raw", b"")
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            raw_parts.append(raw)
            parsed, parse_error = self._safe_parse(raw.strip())
            stream_events.append(
                {
                    "sequence": sequence,
                    "received_monotonic_ns": item.get("received_monotonic_ns"),
                    "received_at_utc": item.get("received_at_utc"),
                    "raw_b64": base64.b64encode(raw).decode("ascii"),
                    "raw_text": raw.decode("utf-8", errors="replace"),
                    "parsed": parsed,
                    "parse_error": parse_error,
                }
            )

        raw_response = b"".join(raw_parts)
        first_received = next(
            (event["received_monotonic_ns"] for event in stream_events if isinstance(event.get("received_monotonic_ns"), int)),
            None,
        )
        last_received = next(
            (event["received_monotonic_ns"] for event in reversed(stream_events) if isinstance(event.get("received_monotonic_ns"), int)),
            None,
        )
        envelope: dict[str, Any] = {
            "ok": 200 <= exchange.status < 300 and exchange.error is None,
            "http_status": exchange.status,
            "response_headers": dict(exchange.headers),
            "request": {
                "method": method,
                "url": f"{self.endpoint}{path}",
                "body_b64": base64.b64encode(body).decode("ascii"),
                "body_text": body.decode("utf-8", errors="replace"),
                "started_at_utc": started_wall,
                "started_monotonic_ns": started_ns,
                "timeout_s": self.timeout_s,
                "stream": stream,
            },
            "stream_events": stream_events,
            "raw_response_b64": base64.b64encode(raw_response).decode("ascii"),
            "timing": {
                "first_event_latency_ns": None if first_received is None else first_received - started_ns,
                "last_event_latency_ns": None if last_received is None else last_received - started_ns,
            },
        }
        if exchange.error is not None:
            envelope["error"] = exchange.error
        elif not envelope["ok"]:
            parsed, parse_error = self._safe_parse(raw_response.strip())
            envelope["error"] = {
                "type": "HTTP_ERROR",
                "status": exchange.status,
                "raw_parsed": parsed,
                "parse_error": parse_error,
            }
        return envelope

    @staticmethod
    def _single_parsed(envelope: dict[str, Any]) -> Any | None:
        parsed = [event.get("parsed") for event in envelope.get("stream_events", []) if event.get("parsed") is not None]
        return None if not parsed else parsed[-1]

    def health(self) -> dict[str, Any]:
        envelope = self._request("GET", "/health")
        envelope["parsed"] = self._single_parsed(envelope)
        return envelope

    def version(self) -> dict[str, Any]:
        envelope = self.health()
        health = envelope.get("parsed")
        envelope["parsed"] = {
            "runtime": "oversized-moe",
            "endpoint": self.endpoint,
            "served_model": self.served_model,
            "health": health,
        }
        return envelope

    def list_models(self) -> dict[str, Any]:
        envelope = self.health()
        envelope["parsed"] = {
            "models": ([{"name": self.served_model, "model": self.served_model}] if envelope.get("ok") else [])
        }
        return envelope

    def model_info(self, model: str) -> dict[str, Any]:
        envelope = self.health()
        health = envelope.get("parsed")
        envelope["parsed"] = {
            "model": model,
            "served_model": self.served_model,
            "backend": "oversized-moe",
            "endpoint": self.endpoint,
            "health": health,
        }
        return envelope

    @staticmethod
    def model_available_in(tags: dict[str, Any], model: str) -> bool:
        parsed = tags.get("parsed") or {}
        for item in parsed.get("models", []) if isinstance(parsed, dict) else []:
            if item.get("name") == model or item.get("model") == model:
                return True
        return False

    def is_model_available(self, model: str) -> bool:
        return self.model_available_in(self.list_models(), model)

    def generate(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        *,
        stream: bool = True,
        request_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del stream, request_fields
        payload: dict[str, Any] = {"messages": messages, "stream": False}
        if isinstance(options.get("num_predict"), int):
            payload["max_tokens"] = int(options["num_predict"])
        if isinstance(options.get("temperature"), (int, float)):
            payload["temperature"] = float(options["temperature"])
        if isinstance(options.get("seed"), int):
            payload["seed"] = int(options["seed"])
        if isinstance(options.get("top_p"), (int, float)):
            payload["top_p"] = float(options["top_p"])

        envelope = self._request("POST", "/v1/chat/completions", payload, stream=False)
        parsed = self._single_parsed(envelope)
        if not envelope.get("ok") or not isinstance(parsed, dict):
            envelope.setdefault("normalized", {"text": "", "thinking": "", "tool_calls": []})
            envelope.setdefault("metrics", {})
            return envelope

        choices = parsed.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content") if isinstance(message.get("content"), str) else ""
        tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
        usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
        timings = parsed.get("timings") if isinstance(parsed.get("timings"), dict) else {}

        prompt_ms = timings.get("prompt_ms")
        predicted_ms = timings.get("predicted_ms")
        prompt_ns = int(float(prompt_ms) * 1_000_000) if isinstance(prompt_ms, (int, float)) else None
        eval_ns = int(float(predicted_ms) * 1_000_000) if isinstance(predicted_ms, (int, float)) else None
        total_ns = prompt_ns + eval_ns if prompt_ns is not None and eval_ns is not None else None

        envelope["normalized"] = {
            "text": content,
            "thinking": "",
            "tool_calls": tool_calls,
            "done": True,
            "done_reason": choice.get("finish_reason"),
        }
        envelope["metrics"] = {
            "total_duration_ns": total_ns,
            "load_duration_ns": None,
            "prompt_eval_count": usage.get("prompt_tokens"),
            "prompt_eval_duration_ns": prompt_ns,
            "eval_count": usage.get("completion_tokens"),
            "eval_duration_ns": eval_ns,
            "prompt_tokens_per_second": timings.get("prompt_per_second"),
            "eval_tokens_per_second": timings.get("predicted_per_second"),
        }
        envelope["phase_metrics"] = {
            "measurement_kind": "MEASURED",
            "thinking_chunks": 0,
            "answer_chunks": 1 if content else 0,
            "thinking_chars": 0,
            "answer_chars": len(content),
            "time_to_first_thinking_ns": None,
            "time_to_first_answer_ns": envelope.get("timing", {}).get("first_event_latency_ns"),
            "thinking_span_ns": None,
            "answer_span_ns": 0 if content else None,
        }
        return envelope

    def unload(self, model: str) -> dict[str, Any]:
        return {
            "ok": True,
            "parsed": {"model": model, "status": "server-managed"},
            "request": {},
            "stream_events": [],
            "raw_response_b64": "",
            "timing": {},
        }

    def pull(self, model: str) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {"type": "UNSUPPORTED", "message": "oversized-moe models are not pulled by compute-cost", "model": model},
            "request": {},
            "stream_events": [],
            "raw_response_b64": "",
            "timing": {},
        }


__all__ = ["HttpExchange", "OversizedMoEAdapter"]
