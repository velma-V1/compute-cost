"""Benchmark lifecycle runner with lossless evidence-first persistence."""

from __future__ import annotations

import base64
import copy
import json
import threading
import time
import traceback
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .evidence import EvidenceStore
from .hardware import collect_hardware_snapshot
from .report import build_summary, render_report
from .scoring import score_case
from .telemetry import TelemetrySampler


def _repeat_to_length(seed: str, length: int) -> str:
    if length <= 0:
        return ""
    repeated = (seed * ((length // len(seed)) + 2))[:length]
    return repeated


def build_context_case(target_context: int, *, timeout_s: float) -> dict[str, Any]:
    """Build a deterministic payload that materially scales with the requested context.

    `target_context` is the runtime window requested through Ollama's `num_ctx`, not an
    asserted token count. The generated text intentionally uses roughly 80% of that
    window under common English tokenizers so there is headroom for instructions and
    output. The runtime-reported prompt token count remains the authoritative measure.
    """

    target = max(256, int(target_context))
    planted_keys = [
        f"CTX-{target}-A-{(target * 7919 + 17) % 100000:05d}",
        f"CTX-{target}-B-{(target * 1543 + 29) % 100000:05d}",
        f"CTX-{target}-C-{(target * 3571 + 43) % 100000:05d}",
    ]
    expected = "|".join(planted_keys)

    # Approximate only. We deliberately preserve the runtime's observed prompt token
    # count in run evidence rather than pretending characters map exactly to tokens.
    target_chars = max(1024, int(target * 3.2))
    seed = " cedar orbit copper river maple engine violet stone amber circuit valley north "

    instruction = (
        f"CONTEXT_SWEEP target={target}. Three retrieval keys are hidden in the payload. "
        f"Return them in A|B|C order with no other text. "
    )
    marker_templates = [
        f"\nFACT_A={planted_keys[0]}\n",
        f"\nFACT_B={planted_keys[1]}\n",
        f"\nFACT_C={planted_keys[2]}\n",
    ]
    fixed = len(instruction) + sum(len(marker) for marker in marker_templates)
    filler_budget = max(256, target_chars - fixed)
    quarter = filler_budget // 4
    segments = [
        _repeat_to_length(seed, quarter),
        _repeat_to_length(seed[::-1], quarter),
        _repeat_to_length(seed, quarter),
        _repeat_to_length(seed[::-1], filler_budget - (quarter * 3)),
    ]
    prompt = (
        instruction
        + segments[0]
        + marker_templates[0]
        + segments[1]
        + marker_templates[1]
        + segments[2]
        + marker_templates[2]
        + segments[3]
    )

    return {
        "id": f"context-{target}",
        "category": "long_context_retrieval",
        "prompt": prompt,
        "scorer": "context_retrieval",
        "expected": expected,
        "planted_keys": planted_keys,
        "timeout_s": timeout_s,
        "max_output_tokens": 48,
        "approx_target_tokens": target,
        "payload_chars": len(prompt),
    }


class BenchmarkRunner:
    def __init__(
        self,
        runtime: Any,
        config: dict[str, Any],
        suite: dict[str, Any],
        *,
        results_root: str | Path = "results",
        telemetry: Any | None = None,
        hardware_collector: Callable[[], dict[str, Any]] = collect_hardware_snapshot,
    ) -> None:
        self.runtime = runtime
        self.config = copy.deepcopy(config)
        self.suite = copy.deepcopy(suite)
        self.results_root = Path(results_root)
        self.telemetry = telemetry or TelemetrySampler()
        self.hardware_collector = hardware_collector
        self.model: str | None = None
        self.store: EvidenceStore | None = None
        self._write_lock = threading.RLock()
        self._telemetry_stop = threading.Event()
        self._telemetry_thread: threading.Thread | None = None
        self._telemetry_seq = 0
        self._request_seq = 0
        self._recent_telemetry: deque[dict[str, Any]] = deque(maxlen=32)

    @staticmethod
    def _utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        assert self.store is not None
        event = {
            "type": event_type,
            "timestamp_utc": self._utc(),
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        with self._write_lock:
            self.store.append_jsonl("events.jsonl", event)
        return event

    @staticmethod
    def _decode_b64(value: Any) -> bytes:
        if not isinstance(value, str) or not value:
            return b""
        try:
            return base64.b64decode(value)
        except Exception:
            return b""

    def _persist_telemetry_raw(self, sample: dict[str, Any], seq: int) -> None:
        assert self.store is not None
        collector = sample.get("nvidia_collector")
        if not isinstance(collector, dict):
            return
        prefix = f"raw/telemetry/nvidia/{seq:06d}"
        self.store.write_json(f"{prefix}-collector.json", collector, producer="telemetry", stage="sample")
        stdout = self._decode_b64(collector.get("stdout_b64"))
        stderr = self._decode_b64(collector.get("stderr_b64"))
        self.store.write_raw(f"{prefix}-stdout.bin", stdout, producer="nvidia-smi", stage="sample")
        self.store.write_raw(f"{prefix}-stderr.bin", stderr, producer="nvidia-smi", stage="sample")

    def _sample(self, stage: str, case_id: str | None = None) -> dict[str, Any] | None:
        assert self.store is not None
        try:
            sample = self.telemetry.sample()
            sample = copy.deepcopy(sample)
            sample["stage"] = stage
            if case_id is not None:
                sample["case_id"] = case_id
            with self._write_lock:
                seq = self._telemetry_seq
                self._telemetry_seq += 1
                self.store.append_jsonl("telemetry.jsonl", sample)
                self._persist_telemetry_raw(sample, seq)
                self._recent_telemetry.append(copy.deepcopy(sample))
            return sample
        except Exception as exc:
            with self._write_lock:
                self.store.record_capture_gap(
                    channel="telemetry",
                    collector=type(self.telemetry).__name__,
                    error=f"{type(exc).__name__}: {exc}",
                    affected=f"{stage}:{case_id or '-'}",
                    continued=True,
                )
            return None

    def _telemetry_loop(self) -> None:
        interval = float(self.config.get("telemetry", {}).get("interval_s", 0.25))
        while not self._telemetry_stop.wait(max(interval, 0.01)):
            self._sample("background")

    def _start_telemetry(self) -> None:
        self._telemetry_stop.clear()
        self._sample("run_start")
        if bool(self.config.get("telemetry", {}).get("background", True)):
            self._telemetry_thread = threading.Thread(
                target=self._telemetry_loop,
                name="compute-cost-telemetry",
                daemon=True,
            )
            self._telemetry_thread.start()

    def _stop_telemetry(self) -> None:
        self._telemetry_stop.set()
        if self._telemetry_thread is not None:
            self._telemetry_thread.join(
                timeout=float(self.config.get("telemetry", {}).get("interval_s", 0.25)) + 2.0
            )
            self._telemetry_thread = None
        self._sample("run_end")

    def _next_request_id(self, stage: str, case_id: str | None = None) -> str:
        self._request_seq += 1
        safe = (case_id or "request").replace("/", "-").replace("\\", "-")
        return f"{self._request_seq:06d}-{stage}-{safe}"

    def _persist_exchange(
        self,
        exchange: dict[str, Any],
        *,
        request_id: str,
        stage: str,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        assert self.store is not None
        refs: dict[str, Any] = {"request_id": request_id}
        request = exchange.get("request") or {}
        request_raw = self._decode_b64(request.get("body_b64"))
        response_raw = self._decode_b64(exchange.get("raw_response_b64"))
        with self._write_lock:
            refs["request"] = self.store.write_raw(
                f"raw/runtime/requests/{request_id}.bin",
                request_raw,
                producer="runtime-adapter",
                stage=stage,
                case_id=case_id,
            )
            refs["response"] = self.store.write_raw(
                f"raw/runtime/responses/{request_id}.bin",
                response_raw,
                producer="runtime-adapter",
                stage=stage,
                case_id=case_id,
            )
            self.store.write_json(
                f"raw/runtime/exchanges/{request_id}.json",
                exchange,
                producer="runtime-adapter",
                stage=stage,
                case_id=case_id,
            )
            stream_refs = []
            for event in exchange.get("stream_events", []) or []:
                if not isinstance(event, dict):
                    continue
                seq = int(event.get("sequence", len(stream_refs)))
                raw = self._decode_b64(event.get("raw_b64"))
                stream_refs.append(
                    self.store.write_raw(
                        f"raw/runtime/streams/{request_id}/{seq:06d}.bin",
                        raw,
                        producer="runtime-stream",
                        stage=stage,
                        case_id=case_id,
                    )
                )
            refs["streams"] = stream_refs
            if exchange.get("error") is not None:
                refs["error"] = self.store.write_json(
                    f"raw/runtime/errors/{request_id}.json",
                    exchange.get("error"),
                    producer="runtime-adapter",
                    stage=stage,
                    case_id=case_id,
                )
        return refs

    def _persist_control_exchange(self, stage: str, exchange: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_request_id(stage)
        return self._persist_exchange(exchange, request_id=request_id, stage=stage)

    def _invoke_generation(
        self,
        *,
        stage: str,
        case_id: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        request_fields: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        assert self.model is not None
        invocation = {
            "model": self.model,
            "messages": copy.deepcopy(messages),
            "options": copy.deepcopy(options),
            "stream": True,
            "request_fields": copy.deepcopy(request_fields or {}),
        }
        self._sample(stage, case_id)
        started_ns = time.monotonic_ns()
        try:
            generation = self.runtime.generate(
                self.model,
                messages,
                options,
                stream=True,
                request_fields=request_fields,
            )
        except Exception as exc:
            generation = {
                "ok": False,
                "http_status": 0,
                "request": {},
                "stream_events": [],
                "raw_response_b64": "",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "normalized": {"text": "", "thinking": "", "tool_calls": []},
                "metrics": {},
                "timing": {},
            }
        ended_ns = time.monotonic_ns()
        generation = copy.deepcopy(generation)
        generation.setdefault("timing", {})["client_started_monotonic_ns"] = started_ns
        generation["timing"]["client_ended_monotonic_ns"] = ended_ns
        generation["timing"]["client_latency_ns"] = ended_ns - started_ns
        request_id = self._next_request_id(stage, case_id)
        refs = self._persist_exchange(
            generation,
            request_id=request_id,
            stage=stage,
            case_id=case_id,
        )
        self._sample(stage, case_id)
        return generation, invocation, refs

    def _persist_scoring(self, case_id: str, stage: str, scoring: dict[str, Any]) -> None:
        assert self.store is not None
        safe = case_id.replace("/", "-").replace("\\", "-")
        with self._write_lock:
            self.store.write_json(
                f"raw/scoring/{stage}-{safe}.json",
                scoring,
                producer="scorer",
                stage=stage,
                case_id=case_id,
            )
            sub = (scoring.get("evidence") or {}).get("subprocess")
            if isinstance(sub, dict):
                self.store.write_raw(
                    f"raw/scoring/{stage}-{safe}-stdout.bin",
                    self._decode_b64(sub.get("stdout_b64")),
                    producer="coding-scorer",
                    stage=stage,
                    case_id=case_id,
                )
                self.store.write_raw(
                    f"raw/scoring/{stage}-{safe}-stderr.bin",
                    self._decode_b64(sub.get("stderr_b64")),
                    producer="coding-scorer",
                    stage=stage,
                    case_id=case_id,
                )

    def _write_replay(
        self,
        *,
        case: dict[str, Any],
        invocation: dict[str, Any],
        generation: dict[str, Any],
        scoring: dict[str, Any],
        stage: str,
        replay_name: str | None = None,
    ) -> None:
        assert self.store is not None
        name = replay_name or str(case["id"])
        safe = name.replace("/", "-").replace("\\", "-")
        snapshot = {
            "schema_version": 1,
            "created_at_utc": self._utc(),
            "benchmark_version": self.suite.get("benchmark_version"),
            "stage": stage,
            "model": self.model,
            "case": copy.deepcopy(case),
            "invocation": copy.deepcopy(invocation),
            "generation": copy.deepcopy(generation),
            "scoring": copy.deepcopy(scoring),
            "telemetry_before_failure": list(copy.deepcopy(self._recent_telemetry)),
            "resolved_config": copy.deepcopy(self.config),
        }
        with self._write_lock:
            self.store.write_json(
                f"replay/{safe}.json",
                snapshot,
                producer="runner",
                stage=stage,
                case_id=str(case.get("id")),
            )

    def _generation_options(self, case: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        generation = self.config.get("generation", {})
        options: dict[str, Any] = {
            "temperature": generation.get("temperature", 0.0),
            "seed": generation.get("seed", 42),
            "num_predict": case.get("max_output_tokens", generation.get("max_tokens", 256)),
        }
        if overrides:
            options.update(overrides)
        return options

    def _execute_case(
        self,
        case: dict[str, Any],
        *,
        stage: str,
        target_context: int | None = None,
        iteration: int | None = None,
        option_overrides: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        messages = [{"role": "user", "content": str(case["prompt"])}]
        options = self._generation_options(case, option_overrides)
        generation, invocation, refs = self._invoke_generation(
            stage=stage,
            case_id=str(case["id"]),
            messages=messages,
            options=options,
        )
        if generation.get("ok", False):
            response_text = str((generation.get("normalized") or {}).get("text", ""))
            scoring = score_case(case, response_text)
        else:
            scoring = {
                "score": 0.0,
                "status": "RUNTIME_ERROR",
                "checks": [],
                "evidence": {"raw_response": ""},
                "error": copy.deepcopy(generation.get("error") or {"type": "RUNTIME_ERROR"}),
            }
        self._persist_scoring(str(case["id"]), stage, scoring)
        record: dict[str, Any] = {
            "case_id": case["id"],
            "category": case.get("category"),
            "stage": stage,
            "status": scoring.get("status"),
            "score": scoring.get("score"),
            "timing": copy.deepcopy(generation.get("timing") or {}),
            "metrics": copy.deepcopy(generation.get("metrics") or {}),
            "runtime_ok": bool(generation.get("ok", False)),
            "evidence_refs": refs,
            "scorer": case.get("scorer"),
        }
        if target_context is not None:
            record["target_context"] = target_context
        if iteration is not None:
            record["iteration"] = iteration
        return record, invocation, generation, scoring

    def run_case(
        self,
        case: dict[str, Any],
        *,
        stage: str = "base",
        target_context: int | None = None,
        iteration: int | None = None,
        option_overrides: dict[str, Any] | None = None,
        append: bool = True,
    ) -> dict[str, Any]:
        assert self.store is not None
        record, invocation, generation, scoring = self._execute_case(
            case,
            stage=stage,
            target_context=target_context,
            iteration=iteration,
            option_overrides=option_overrides,
        )
        if append:
            with self._write_lock:
                self.store.append_jsonl("cases.jsonl", record)
        if scoring.get("score") != 1.0 or scoring.get("status") == "SCORER_ERROR":
            replay_name = str(case["id"])
            if stage == "context" and target_context is not None:
                replay_name = f"{case['id']}-ctx-{target_context}"
            elif stage == "sustained" and iteration is not None:
                replay_name = f"{case['id']}-iter-{iteration}"
            self._write_replay(
                case=case,
                invocation=invocation,
                generation=generation,
                scoring=scoring,
                stage=stage,
                replay_name=replay_name,
            )
        return record

    def _resource_stop_reason(self) -> str | None:
        if not self._recent_telemetry:
            return None
        sample = self._recent_telemetry[-1]
        limits = self.config.get("limits", {})
        memory_percent = (sample.get("host") or {}).get("memory_percent")
        host_limit = limits.get("host_ram_percent")
        if isinstance(memory_percent, (int, float)) and isinstance(host_limit, (int, float)):
            if float(memory_percent) >= float(host_limit):
                return "HOST_RAM_LIMIT"
        gpu = sample.get("gpu") or {}
        if isinstance(gpu, dict) and gpu.get("availability") == "available":
            for device in gpu.get("devices", []) or []:
                total = device.get("memory_total_mib") if isinstance(device, dict) else None
                used = device.get("memory_used_mib") if isinstance(device, dict) else None
                vram_limit = limits.get("vram_percent")
                if all(isinstance(v, (int, float)) for v in (total, used, vram_limit)) and float(total) > 0:
                    if (float(used) / float(total)) * 100.0 >= float(vram_limit):
                        return "VRAM_LIMIT"
        return None

    def run_context_sweep(self) -> list[dict[str, Any]]:
        assert self.store is not None
        schedule = [int(v) for v in self.config.get("limits", {}).get("context_schedule", [])]
        threshold = max(1, int(self.config.get("limits", {}).get("consecutive_context_failures", 1)))
        failures = 0
        records: list[dict[str, Any]] = []
        for target in schedule:
            case = build_context_case(
                target,
                timeout_s=float(self.config.get("limits", {}).get("request_timeout_s", 120)),
            )
            record, invocation, generation, scoring = self._execute_case(
                case,
                stage="context",
                target_context=target,
                option_overrides={"num_ctx": target},
            )
            failed = scoring.get("score") != 1.0 or not generation.get("ok", False)
            failures = failures + 1 if failed else 0
            stop_reason = self._resource_stop_reason()
            if not generation.get("ok", False):
                stop_reason = stop_reason or "RUNTIME_ERROR"
            if failures >= threshold:
                stop_reason = stop_reason or "CONSECUTIVE_CORRECTNESS_FAILURES"
            if stop_reason:
                record["stop_boundary"] = True
                record["stop_reason"] = stop_reason
            with self._write_lock:
                self.store.append_jsonl("cases.jsonl", record)
            if failed:
                self._write_replay(
                    case=case,
                    invocation=invocation,
                    generation=generation,
                    scoring=scoring,
                    stage="context",
                    replay_name=f"{case['id']}-ctx-{target}",
                )
            records.append(record)
            if stop_reason:
                break
        return records

    def run_sustained_load(self) -> list[dict[str, Any]]:
        assert self.store is not None
        iterations = max(0, int(self.config.get("limits", {}).get("sustained_iterations", 0)))
        records: list[dict[str, Any]] = []
        for iteration in range(iterations):
            case = {
                "id": f"sustained-{iteration}",
                "category": "sustained",
                "prompt": "PING — reply exactly PING",
                "scorer": "exact",
                "expected": "PING",
                "timeout_s": self.config.get("limits", {}).get("request_timeout_s", 120),
                "max_output_tokens": 16,
            }
            record, invocation, generation, scoring = self._execute_case(
                case,
                stage="sustained",
                iteration=iteration,
            )
            stop_reason = self._resource_stop_reason()
            if not generation.get("ok", False):
                stop_reason = stop_reason or "RUNTIME_ERROR"
            if stop_reason:
                record["stop_boundary"] = True
                record["stop_reason"] = stop_reason
            with self._write_lock:
                self.store.append_jsonl("cases.jsonl", record)
            if scoring.get("score") != 1.0:
                self._write_replay(
                    case=case,
                    invocation=invocation,
                    generation=generation,
                    scoring=scoring,
                    stage="sustained",
                    replay_name=f"{case['id']}-iter-{iteration}",
                )
            records.append(record)
            if stop_reason:
                break
        return records

    @staticmethod
    def _find_model_size(tags: dict[str, Any], model: str) -> int | None:
        parsed = tags.get("parsed") or {}
        if not isinstance(parsed, dict):
            return None
        for item in parsed.get("models", []) or []:
            if isinstance(item, dict) and (item.get("name") == model or item.get("model") == model):
                size = item.get("size")
                return int(size) if isinstance(size, (int, float)) else None
        return None

    def onboard(self, model: str, *, pull: bool = False) -> Path:
        self.model = model
        run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.store = EvidenceStore(self.results_root, run_id)
        store = self.store
        store.write_json("resolved-config.json", self.config, producer="runner", stage="preflight")
        store.write_json("benchmark-snapshot.json", self.suite, producer="runner", stage="preflight")
        self._event("RUN_START", model=model, benchmark_version=self.suite.get("benchmark_version"))
        self._start_telemetry()

        try:
            hardware = self.hardware_collector()
            store.write_json("hardware.json", hardware, producer="hardware", stage="preflight")

            version = self.runtime.version()
            version_refs = self._persist_control_exchange("runtime-version", version)
            tags = self.runtime.list_models()
            tags_refs = self._persist_control_exchange("model-list", tags)
            available = self.runtime.model_available_in(tags, model) if hasattr(self.runtime, "model_available_in") else self.runtime.is_model_available(model)

            pull_refs = None
            if not available and pull:
                pull_result = self.runtime.pull(model)
                pull_refs = self._persist_control_exchange("model-pull", pull_result)
                tags = self.runtime.list_models()
                tags_refs = self._persist_control_exchange("model-list-after-pull", tags)
                available = self.runtime.model_available_in(tags, model) if hasattr(self.runtime, "model_available_in") else self.runtime.is_model_available(model)

            if not available:
                runtime_snapshot = {
                    "model": model,
                    "available": False,
                    "version": version.get("parsed"),
                    "model_size_bytes": None,
                    "evidence_refs": {"version": version_refs, "tags": tags_refs, "pull": pull_refs},
                }
                store.write_json("runtime.json", runtime_snapshot, producer="runner", stage="preflight")
                self._event("RUN_FAILED", failure="MODEL_NOT_FOUND", model=model)
                return self._finalize_run()

            info = self.runtime.model_info(model)
            info_refs = self._persist_control_exchange("model-info", info)
            runtime_snapshot = {
                "model": model,
                "available": True,
                "version": version.get("parsed"),
                "model_size_bytes": self._find_model_size(tags, model),
                "model_info": info.get("parsed"),
                "evidence_refs": {"version": version_refs, "tags": tags_refs, "show": info_refs, "pull": pull_refs},
            }
            store.write_json("runtime.json", runtime_snapshot, producer="runner", stage="preflight")
            self._event("PREFLIGHT_COMPLETE", model=model)

            unload = self.runtime.unload(model)
            self._persist_control_exchange("cold-unload", unload)
            cold_case = {
                "id": "cold-start",
                "category": "onboarding",
                "prompt": "Reply exactly READY",
                "scorer": "exact",
                "expected": "READY",
                "timeout_s": self.config.get("limits", {}).get("request_timeout_s", 120),
                "max_output_tokens": 16,
            }
            self.run_case(cold_case, stage="cold")

            warm_case = {
                **cold_case,
                "id": "warmup",
            }
            self.run_case(warm_case, stage="warmup")

            for case in self.suite.get("cases", []) or []:
                self.run_case(case, stage="base")

            self.run_context_sweep()
            self.run_sustained_load()
            self._event("BENCHMARK_COMPLETE", model=model)
            return self._finalize_run()
        except Exception as exc:
            store.write_json(
                "raw/runner-failure.json",
                {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                producer="runner",
                stage="fatal",
            )
            self._event("RUN_FAILED", failure="BENCHMARK_ERROR", error_type=type(exc).__name__, error=str(exc))
            return self._finalize_run()

    def _finalize_run(self) -> Path:
        assert self.store is not None
        self._stop_telemetry()
        self._event("RUN_END", model=self.model)
        summary = build_summary(self.store.run_dir)
        self.store.write_json("summary.json", summary, producer="report", stage="report")
        self.store.write_raw(
            "report.md",
            render_report(summary),
            producer="report",
            stage="report",
            media_type="text/markdown",
        )
        self.store.finalize_manifest(
            metadata={
                "model": self.model,
                "benchmark_version": self.suite.get("benchmark_version"),
            }
        )
        return self.store.run_dir

    def replay_case(self, source_run_id: str, case_id: str) -> Path:
        source = self.results_root / source_run_id / "replay" / f"{case_id}.json"
        snapshot = json.loads(source.read_text(encoding="utf-8"))
        model = str(snapshot["model"])
        invocation = snapshot["invocation"]
        replay_run_id = f"replay-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        previous_store = self.store
        previous_model = self.model
        try:
            self.store = EvidenceStore(self.results_root, replay_run_id)
            self.model = model
            self.store.write_json("source-replay.json", snapshot, producer="runner", stage="replay")
            self._event("RUN_START", model=model, replay_of=f"{source_run_id}:{case_id}")
            generation, _, _ = self._invoke_generation(
                stage="replay",
                case_id=case_id,
                messages=invocation["messages"],
                options=invocation["options"],
                request_fields=invocation.get("request_fields") or None,
            )
            self.store.write_json("replay-result.json", generation, producer="runner", stage="replay")
            self._event("RUN_END", model=model)
            self.store.finalize_manifest(metadata={"replay_of": f"{source_run_id}:{case_id}", "model": model})
            return self.store.run_dir
        finally:
            self.store = previous_store
            self.model = previous_model