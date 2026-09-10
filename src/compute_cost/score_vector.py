"""Multidimensional scoring derived from retained deterministic evidence."""

from __future__ import annotations

import json
import re
from typing import Any

NOT_APPLICABLE = "NOT_APPLICABLE"
UNAVAILABLE = "UNAVAILABLE"


def _pct(checks: list[dict[str, Any]]) -> float | str:
    applicable = [c for c in checks if c.get("pass") in (True, False)]
    if not applicable:
        return NOT_APPLICABLE
    return round(100.0 * sum(c.get("pass") is True for c in applicable) / len(applicable), 3)


def _ns_seconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) / 1_000_000_000.0


def _rate(count: Any, duration_ns: Any) -> float | str:
    if isinstance(count, bool) or not isinstance(count, (int, float)):
        return UNAVAILABLE
    seconds = _ns_seconds(duration_ns)
    if seconds is None or seconds <= 0:
        return UNAVAILABLE
    return round(float(count) / seconds, 6)


def _strip_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def _semantic_correct(case: dict[str, Any], scoring: dict[str, Any], classification: dict[str, Any], text: str) -> float:
    if classification.get("result_class") == "ANSWER_CORRECT":
        return 100.0
    if classification.get("valid_for_capability") is not True:
        return 0.0
    scorer = case.get("scorer")
    expected = case.get("expected")
    cleaned = _strip_fence(text)
    try:
        if scorer == "json":
            parsed = json.loads(cleaned)
            if isinstance(expected, dict) and isinstance(parsed, dict):
                return 100.0 if all(parsed.get(k) == v for k, v in expected.items()) else 0.0
        if scorer == "exact":
            return 100.0 if cleaned == str(expected).strip() else 0.0
        if scorer == "numeric":
            match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
            if match is not None and float(match.group(0)) == float(expected):
                return 100.0
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
    checks = list(scoring.get("checks") or [])
    semantic = [c for c in checks if not str(c.get("name", "")).startswith(("valid_json", "required:"))]
    value = _pct(semantic)
    return float(value) if isinstance(value, (int, float)) else 0.0


def build_score_vector(
    case: dict[str, Any],
    scoring: dict[str, Any],
    classification: dict[str, Any],
    generation: dict[str, Any],
    *,
    telemetry: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    checks = list(scoring.get("checks") or [])
    normalized = generation.get("normalized") if isinstance(generation.get("normalized"), dict) else {}
    text = str(normalized.get("text") or scoring.get("evidence", {}).get("raw_response") or "")
    metrics = generation.get("metrics") if isinstance(generation.get("metrics"), dict) else {}
    timing = generation.get("timing") if isinstance(generation.get("timing"), dict) else {}

    format_checks = [c for c in checks if str(c.get("name", "")).startswith(("valid_json", "required:", "safe_executable_subset"))]
    tool_checks = [c for c in checks if "tool" in str(c.get("name", "")).lower() or str(c.get("name", "")).startswith(("value:tool_sequence", "value:arguments"))]
    state_checks = [c for c in checks if "checkpoint" in str(c.get("name", "")).lower() or "state" in str(c.get("name", "")).lower()]
    decision_checks = [c for c in checks if "action" in str(c.get("name", "")).lower() or str(c.get("name", "")) == "decision"]
    verification_checks = [c for c in checks if "verif" in str(c.get("name", "")).lower() or "executable_tests" in str(c.get("name", ""))]

    prompt_tokens = metrics.get("prompt_eval_count", metrics.get("prompt_tokens"))
    generated_tokens = metrics.get("eval_count", metrics.get("generated_tokens"))
    prompt_tps = _rate(prompt_tokens, metrics.get("prompt_eval_duration_ns", metrics.get("prompt_eval_duration")))
    generation_tps = _rate(generated_tokens, metrics.get("eval_duration_ns", metrics.get("eval_duration")))
    latency = _ns_seconds(timing.get("client_latency_ns"))

    gpu_energy: Any = UNAVAILABLE
    ram_peak: Any = UNAVAILABLE
    vram_peak: Any = UNAVAILABLE
    for sample in telemetry or []:
        if isinstance(sample.get("memory"), dict):
            value = sample["memory"].get("used_bytes")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                ram_peak = max(float(value), 0.0 if ram_peak == UNAVAILABLE else float(ram_peak))
        gpus = sample.get("gpus")
        if isinstance(gpus, list):
            for gpu in gpus:
                if not isinstance(gpu, dict):
                    continue
                value = gpu.get("memory_used_mib")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    vram_peak = max(float(value), 0.0 if vram_peak == UNAVAILABLE else float(vram_peak))

    semantic = _semantic_correct(case, scoring, classification, text)
    runtime_validity = 100.0 if generation.get("ok", True) and classification.get("valid_for_capability") is True else 0.0
    total_tokens = (
        int(prompt_tokens) + int(generated_tokens)
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and isinstance(generated_tokens, int) and not isinstance(generated_tokens, bool)
        else UNAVAILABLE
    )
    value_per_second = UNAVAILABLE if latency is None or latency <= 0 else round(semantic / latency, 6)
    value_per_generated_token = (
        UNAVAILABLE
        if not isinstance(generated_tokens, int) or isinstance(generated_tokens, bool) or generated_tokens <= 0
        else round(semantic / generated_tokens, 6)
    )

    return {
        "semantic_correctness": semantic,
        "contract_format_compliance": _pct(format_checks),
        "tool_procedure_compliance": _pct(tool_checks),
        "constraint_compliance": _pct([c for c in checks if "constraint" in str(c.get("name", "")).lower()]),
        "state_checkpoint_accuracy": _pct(state_checks),
        "decision_quality": _pct(decision_checks),
        "recovery_quality": _pct([c for c in checks if "recover" in str(c.get("name", "")).lower()]),
        "verification_quality": _pct(verification_checks),
        "evidence_use_quality": _pct([c for c in checks if "evidence" in str(c.get("name", "")).lower()]),
        "goal_preservation": _pct([c for c in checks if "goal" in str(c.get("name", "")).lower()]),
        "reasoning_condition_status": "SUPPORTED",
        "runtime_validity": runtime_validity,
        "truncation_status": "TRUNCATED" if classification.get("result_class") in {"THINK_TRUNCATED", "ANSWER_TRUNCATED"} else "COMPLETE",
        "latency_seconds": UNAVAILABLE if latency is None else latency,
        "prompt_tokens_per_second": prompt_tps,
        "generation_tokens_per_second": generation_tps,
        "prompt_tokens": prompt_tokens if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) else UNAVAILABLE,
        "generated_tokens": generated_tokens if isinstance(generated_tokens, int) and not isinstance(generated_tokens, bool) else UNAVAILABLE,
        "total_tokens": total_tokens,
        "ram_peak_bytes": ram_peak,
        "vram_peak_mib": vram_peak,
        "gpu_energy_wh": gpu_energy,
        "wall_clock_seconds": UNAVAILABLE if latency is None else latency,
        "value_per_second": value_per_second,
        "value_per_generated_token": value_per_generated_token,
        "value_per_wh": UNAVAILABLE,
    }
