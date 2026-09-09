"""Lossless Ollama HTTP adapter.

The adapter retains the exact serialized request body and every response chunk
before producing normalized fields. Unknown API fields therefore survive
schema evolution and can be analyzed later without rerunning a model.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class HttpExchange:
    status: int
    headers: dict[str, str]
    chunks: list[dict[str, Any]]
    error: dict[str, Any] | None = None


class UrllibTransport:
    def exchange(self, method: str, url: str, body: bytes, timeout: float, stream: bool) -> HttpExchange:
        headers = {"Content-Type": "application/json"}
        request = Request(url, data=body if method != "GET" else None, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout) as response:
                response_headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
                chunks: list[dict[str, Any]] = []
                if stream:
                    for raw in response:
                        chunks.append(self._chunk(raw))
                else:
                    chunks.append(self._chunk(response.read()))
                return HttpExchange(int(response.status), response_headers, chunks)
        except HTTPError as exc:
            raw = exc.read()
            response_headers = {str(k).lower(): str(v) for k, v in exc.headers.items()} if exc.headers else {}
            return HttpExchange(int(exc.code), response_headers, [self._chunk(raw)])
        except (URLError, OSError) as exc:
            return HttpExchange(0, {}, [], error={"type": type(exc).__name__, "message": str(exc)})

    @staticmethod
    def _chunk(raw: bytes) -> dict[str, Any]:
        return {
            "raw": raw,
            "received_monotonic_ns": time.monotonic_ns(),
            "received_at_utc": datetime.now(timezone.utc).isoformat(),
        }


class OllamaAdapter:
    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:11434",
        *,
        timeout_s: float = 120.0,
        transport: Any | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s
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
        parsed = [e.get("parsed") for e in envelope.get("stream_events", []) if e.get("parsed") is not None]
        if not parsed:
            return None
        return parsed[-1]

    @staticmethod
    def _phase_metrics(stream_events: list[dict[str, Any]], started_ns: int) -> dict[str, Any]:
        thinking_events: list[tuple[int, str]] = []
        answer_events: list[tuple[int, str]] = []
        for event in stream_events:
            received = event.get("received_monotonic_ns")
            parsed = event.get("parsed")
            if not isinstance(received, int) or not isinstance(parsed, dict):
                continue
            message = parsed.get("message")
            if not isinstance(message, dict):
                continue
            thinking = message.get("thinking")
            content = message.get("content")
            if isinstance(thinking, str) and thinking:
                thinking_events.append((received, thinking))
            if isinstance(content, str) and content:
                answer_events.append((received, content))

        def latency(items: list[tuple[int, str]]) -> int | None:
            return None if not items else items[0][0] - started_ns

        def span(items: list[tuple[int, str]]) -> int | None:
            return None if not items else items[-1][0] - items[0][0]

        return {
            "measurement_kind": "MEASURED",
            "thinking_chunks": len(thinking_events),
            "answer_chunks": len(answer_events),
            "thinking_chars": sum(len(text) for _, text in thinking_events),
            "answer_chars": sum(len(text) for _, text in answer_events),
            "time_to_first_thinking_ns": latency(thinking_events),
            "time_to_first_answer_ns": latency(answer_events),
            "thinking_span_ns": span(thinking_events),
            "answer_span_ns": span(answer_events),
        }

    def health(self) -> dict[str, Any]:
        evidence = self.version()
        return {"ok": evidence.get("ok", False), "evidence": evidence}

    def version(self) -> dict[str, Any]:
        envelope = self._request("GET", "/api/version")
        envelope["parsed"] = self._single_parsed(envelope)
        return envelope

    def list_models(self) -> dict[str, Any]:
        envelope = self._request("GET", "/api/tags")
        envelope["parsed"] = self._single_parsed(envelope)
        return envelope

    def model_info(self, model: str) -> dict[str, Any]:
        envelope = self._request("POST", "/api/show", {"model": model, "verbose": True})
        envelope["parsed"] = self._single_parsed(envelope)
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
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }
        if request_fields:
            payload.update(request_fields)
        envelope = self._request("POST", "/api/chat", payload, stream=stream)
        if not envelope["ok"]:
            return envelope

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[Any] = []
        final: dict[str, Any] = {}
        for event in envelope["stream_events"]:
            parsed = event.get("parsed")
            if not isinstance(parsed, dict):
                continue
            final = parsed
            message = parsed.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                thinking = message.get("thinking")
                if isinstance(content, str):
                    text_parts.append(content)
                if isinstance(thinking, str):
                    thinking_parts.append(thinking)
                calls = message.get("tool_calls")
                if isinstance(calls, list):
                    tool_calls.extend(calls)

        envelope["normalized"] = {
            "text": "".join(text_parts),
            "thinking": "".join(thinking_parts),
            "tool_calls": tool_calls,
            "done": final.get("done"),
            "done_reason": final.get("done_reason"),
        }
        envelope["metrics"] = {
            "total_duration_ns": final.get("total_duration"),
            "load_duration_ns": final.get("load_duration"),
            "prompt_eval_count": final.get("prompt_eval_count"),
            "prompt_eval_duration_ns": final.get("prompt_eval_duration"),
            "eval_count": final.get("eval_count"),
            "eval_duration_ns": final.get("eval_duration"),
        }
        envelope["phase_metrics"] = self._phase_metrics(
            envelope["stream_events"],
            int(envelope["request"]["started_monotonic_ns"]),
        )
        return envelope

    def unload(self, model: str) -> dict[str, Any]:
        envelope = self._request(
            "POST",
            "/api/generate",
            {"model": model, "keep_alive": 0, "stream": False},
            stream=False,
        )
        envelope["parsed"] = self._single_parsed(envelope)
        return envelope

    def pull(self, model: str) -> dict[str, Any]:
        envelope = self._request(
            "POST",
            "/api/pull",
            {"model": model, "stream": True},
            stream=True,
        )
        envelope["parsed_events"] = [
            event["parsed"] for event in envelope["stream_events"] if event.get("parsed") is not None
        ]
        return envelope
