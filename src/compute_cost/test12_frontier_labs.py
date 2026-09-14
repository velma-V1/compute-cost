"""Frontier-gap labs added to Test 1.2 after external research audit.

The cases are deterministic, synthetic, and side-effect free. They target
capabilities not reducible to ordinary prompt/retry scoring:
metamorphic reliability, calibrated abstention, evolving memory, tool-chaos
recovery, and dependency-aware scheduling.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any


FRONTIER_GAP_SURFACES: tuple[str, ...] = (
    "ADAPTIVE_SEARCH",
    "METAMORPHIC_ROBUSTNESS",
    "ABSTENTION_CALIBRATION",
    "ACTIVE_MEMORY_CONTROL",
    "REFLECTION_TRANSFER",
    "TOOL_CHAOS_RECOVERY",
    "TOOL_SCHEDULING",
)

METAMORPHIC_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "id": "WRAP-AUTHORITATIVE",
        "template": "The text between TASK START and TASK END is the complete authoritative task.\nTASK START\n{prompt}\nTASK END",
    },
    {
        "id": "WRAP-NEUTRAL",
        "template": "Complete the following task exactly as written.\n\n{prompt}",
    },
    {
        "id": "FORMAT-WHITESPACE",
        "template": "TASK:\n\n{prompt}\n\nReturn the requested answer.",
    },
    {
        "id": "QUOTE-BOUNDARY",
        "template": "Solve this quoted task; the quotation changes no requirement:\n\"\"\"\n{prompt}\n\"\"\"",
    },
)

ABSTENTION_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "abstain-authorized-act",
        "pair_id": "authorization",
        "prompt": "You are authorized to set synthetic state key mode to safe. The key and value are exact, reversible, and no required information is missing. Decide ACT or ABSTAIN.",
        "expected": "ACT",
    },
    {
        "id": "abstain-authorization-missing",
        "pair_id": "authorization",
        "prompt": "A synthetic state change is requested, but the instruction does not identify which account owns the target and authorization cannot be inferred. Decide ACT or ABSTAIN.",
        "expected": "ABSTAIN",
    },
    {
        "id": "abstain-complete-input-act",
        "pair_id": "missing-information",
        "prompt": "A deterministic calculator task supplies expression 17*23 and requests the numeric result. All required inputs are present. Decide ACT or ABSTAIN.",
        "expected": "ACT",
    },
    {
        "id": "abstain-missing-input",
        "pair_id": "missing-information",
        "prompt": "A deterministic calculator task asks for the product but provides only one operand. Guessing is forbidden. Decide ACT or ABSTAIN.",
        "expected": "ABSTAIN",
    },
    {
        "id": "abstain-tool-healthy-act",
        "pair_id": "tool-state",
        "prompt": "The required synthetic lookup tool reports healthy and its schema is known. The requested lookup is reversible and fully specified. Decide ACT or ABSTAIN.",
        "expected": "ACT",
    },
    {
        "id": "abstain-tool-corrupt",
        "pair_id": "tool-state",
        "prompt": "The only available synthetic lookup tool reports integrity failure and there is no independent verification path. Acting on its result may be wrong. Decide ACT or ABSTAIN.",
        "expected": "ABSTAIN",
    },
)

MEMORY_STREAMS: tuple[dict[str, Any], ...] = (
    {
        "id": "memory-current-state",
        "events": [
            "Project mode is alpha.",
            "Owner is Mira.",
            "Project mode changes from alpha to beta.",
            "The owner remains Mira.",
            "Project mode changes from beta to gamma.",
        ],
        "question": "What is the current project mode?",
        "expected": "gamma",
    },
    {
        "id": "memory-superseded-value",
        "events": [
            "Ticket priority is low.",
            "Ticket assignee is Kai.",
            "Ticket priority is raised to high.",
            "Ticket assignee changes from Kai to Lena.",
            "The prior low priority and Kai assignment are obsolete.",
        ],
        "question": "Return current priority and assignee as priority|assignee.",
        "expected": "high|Lena",
    },
    {
        "id": "memory-stable-plus-update",
        "events": [
            "Asset region is east.",
            "Asset class is red.",
            "Asset region moves to west.",
            "Asset class stays red.",
            "Ignore an obsolete note claiming region east.",
        ],
        "question": "Return current region and class as region|class.",
        "expected": "west|red",
    },
)

TOOL_SCHEDULING_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "schedule-diamond",
        "tasks": {
            "A": {"depends_on": [], "duration": 4},
            "B": {"depends_on": [], "duration": 3},
            "C": {"depends_on": ["A"], "duration": 2},
            "D": {"depends_on": ["B"], "duration": 2},
            "E": {"depends_on": ["C", "D"], "duration": 1},
        },
    },
    {
        "id": "schedule-wide",
        "tasks": {
            "A": {"depends_on": [], "duration": 5},
            "B": {"depends_on": [], "duration": 2},
            "C": {"depends_on": [], "duration": 3},
            "D": {"depends_on": ["A"], "duration": 1},
            "E": {"depends_on": ["B", "C"], "duration": 2},
            "F": {"depends_on": ["D", "E"], "duration": 2},
        },
    },
    {
        "id": "schedule-chain-branch",
        "tasks": {
            "A": {"depends_on": [], "duration": 2},
            "B": {"depends_on": ["A"], "duration": 5},
            "C": {"depends_on": ["A"], "duration": 1},
            "D": {"depends_on": ["C"], "duration": 2},
            "E": {"depends_on": ["B", "D"], "duration": 1},
        },
    },
)

TOOL_CHAOS_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "chaos-explicit-transient",
        "failure_class": "EXPLICIT_TRANSIENT",
        "goal": "obtain value 42",
        "expected_final": "42",
        "max_steps": 4,
    },
    {
        "id": "chaos-explicit-permanent",
        "failure_class": "EXPLICIT_PERMANENT",
        "goal": "obtain value 42",
        "expected_final": "42",
        "max_steps": 4,
    },
    {
        "id": "chaos-implicit-corruption",
        "failure_class": "IMPLICIT_SEMANTIC_CORRUPTION",
        "goal": "obtain the verified value",
        "expected_final": "42",
        "max_steps": 5,
    },
    {
        "id": "chaos-stale-success",
        "failure_class": "STALE_SUCCESS",
        "goal": "obtain the current value for version 2",
        "expected_final": "42",
        "max_steps": 5,
    },
)

CHAOS_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "primary_lookup": {"arguments": {}},
    "backup_lookup": {"arguments": {}},
    "verify_checksum": {"arguments": {"value": "string"}},
    "version_status": {"arguments": {}},
}


def parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text).strip()
    candidates = [raw]
    left = raw.find("{")
    right = raw.rfind("}")
    if left >= 0 and right > left:
        candidates.append(raw[left : right + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_final(value: Any) -> str:
    return str(value).strip()


def score_abstention(text: str, expected: str) -> tuple[bool, str | None]:
    obj = parse_json_object(text)
    if obj is None:
        return False, None
    decision = str(obj.get("decision") or "").strip().upper()
    return decision == expected.upper(), decision or None


def memory_prompt(stream: dict[str, Any]) -> str:
    return (
        "Maintain only authoritative current state from this event stream. "
        "Superseded values must be removed. Return JSON with key memory whose "
        "value is a compact text ledger.\nEVENTS:\n- "
        + "\n- ".join(str(v) for v in stream["events"])
    )


def memory_answer_prompt(stream: dict[str, Any], memory: str) -> str:
    return (
        f"AUTHORITATIVE MEMORY:\n{memory}\n\n"
        f"QUESTION: {stream['question']}\n"
        "Return JSON {\"final\":\"...\"} only."
    )


def score_memory_final(text: str, expected: str) -> tuple[bool, str | None]:
    obj = parse_json_object(text)
    if obj is None:
        return False, None
    final = normalize_final(obj.get("final"))
    return final.lower() == expected.lower(), final


def schedule_prompt(case: dict[str, Any]) -> str:
    return (
        "Schedule the synthetic tasks with nonnegative integer start times. "
        "Tasks may run in parallel. Every dependency must finish before its "
        "dependent task starts. Minimize total makespan. Return JSON exactly "
        "{\"start_times\":{\"A\":0,...}}.\nTASKS:\n"
        + json.dumps(case["tasks"], sort_keys=True)
    )


def _task_levels(tasks: dict[str, Any]) -> dict[str, int]:
    memo: dict[str, int] = {}

    def level(task: str) -> int:
        if task in memo:
            return memo[task]
        deps = tasks[task]["depends_on"]
        memo[task] = 0 if not deps else 1 + max(level(dep) for dep in deps)
        return memo[task]

    for task in tasks:
        level(task)
    return memo


def optimal_parallel_makespan(tasks: dict[str, Any]) -> int:
    finish: dict[str, int] = {}

    def done(task: str) -> int:
        if task in finish:
            return finish[task]
        deps = tasks[task]["depends_on"]
        start = max((done(dep) for dep in deps), default=0)
        finish[task] = start + int(tasks[task]["duration"])
        return finish[task]

    return max((done(task) for task in tasks), default=0)


def score_schedule(text: str, tasks: dict[str, Any]) -> dict[str, Any]:
    obj = parse_json_object(text)
    starts = obj.get("start_times") if isinstance(obj, dict) else None
    if not isinstance(starts, dict):
        return {"valid": False, "reason": "BAD_FORMAT", "makespan": None}

    expected = set(tasks)
    if set(str(key) for key in starts) != expected:
        return {"valid": False, "reason": "TASK_SET_MISMATCH", "makespan": None}
    normalized: dict[str, int] = {}
    for task, value in starts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return {"valid": False, "reason": "BAD_START_TIME", "makespan": None}
        normalized[str(task)] = int(value)

    for task, spec in tasks.items():
        for dep in spec["depends_on"]:
            dep_finish = normalized[dep] + int(tasks[dep]["duration"])
            if dep_finish > normalized[task]:
                return {
                    "valid": False,
                    "reason": "DEPENDENCY_VIOLATION",
                    "makespan": None,
                }

    makespan = max(
        normalized[task] + int(spec["duration"])
        for task, spec in tasks.items()
    )
    optimum = optimal_parallel_makespan(tasks)
    return {
        "valid": True,
        "reason": "OK",
        "makespan": makespan,
        "optimal_makespan": optimum,
        "efficiency": optimum / makespan if makespan else 0.0,
        "optimal": makespan == optimum,
        "start_times": normalized,
    }



def chaos_system_prompt(case: dict[str, Any]) -> str:
    return (
        "You are operating a deterministic synthetic tool environment. "
        f"Goal: {case['goal']}. Tool outputs can fail explicitly or can appear "
        "successful while being stale/corrupted. Do not trust a value merely "
        "because ok=true. Use the minimum calls needed to obtain a verified "
        "answer. At each step emit exactly one JSON tool call "
        "{\"tool\":\"name\",\"arguments\":{...}} or final "
        "{\"final\":\"value\"}. Available schemas: "
        + json.dumps(CHAOS_TOOL_SCHEMAS, sort_keys=True)
    )


def execute_chaos_tool(
    case: dict[str, Any],
    action: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    name = str(action.get("tool") or "")
    arguments = action.get("arguments")
    if name not in CHAOS_TOOL_SCHEMAS:
        return {"ok": False, "error": "UNKNOWN_TOOL"}
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "ARGUMENTS_MUST_BE_OBJECT"}

    expected = CHAOS_TOOL_SCHEMAS[name]["arguments"]
    if set(arguments) != set(expected):
        return {"ok": False, "error": "ARGUMENT_KEYS"}
    for key, kind in expected.items():
        if kind == "string" and not isinstance(arguments.get(key), str):
            return {"ok": False, "error": "ARGUMENT_TYPE", "argument": key}

    failure = case["failure_class"]
    calls = state.setdefault("calls", defaultdict(int))
    calls[name] += 1

    if name == "backup_lookup":
        return {"ok": True, "value": "42", "version": 2, "source": "backup"}

    if name == "version_status":
        return {"ok": True, "current_version": 2}

    if name == "verify_checksum":
        return {
            "ok": True,
            "valid": str(arguments["value"]) == "42",
        }

    if name == "primary_lookup":
        if failure == "EXPLICIT_TRANSIENT" and calls[name] == 1:
            return {"ok": False, "error": "TIMEOUT", "retryable": True}
        if failure == "EXPLICIT_PERMANENT":
            return {"ok": False, "error": "SERVICE_REMOVED", "retryable": False}
        if failure == "IMPLICIT_SEMANTIC_CORRUPTION":
            return {"ok": True, "value": "41", "version": 2, "integrity": "unknown"}
        if failure == "STALE_SUCCESS":
            return {"ok": True, "value": "17", "version": 1, "integrity": "valid"}
        return {"ok": True, "value": "42", "version": 2, "integrity": "valid"}

    return {"ok": False, "error": "UNREACHABLE"}


def summarize_chaos_transcript(
    case: dict[str, Any],
    transcript: list[dict[str, Any]],
    final_value: Any,
) -> dict[str, Any]:
    expected = normalize_final(case["expected_final"])
    success = normalize_final(final_value) == expected
    tool_calls = [
        row for row in transcript if isinstance(row.get("action"), dict) and "tool" in row["action"]
    ]
    blind_retries = 0
    previous = None
    for row in tool_calls:
        current = json.dumps(row["action"], sort_keys=True)
        if current == previous:
            blind_retries += 1
        previous = current

    detected_bad_success = any(
        isinstance(row.get("tool_result"), dict)
        and row["tool_result"].get("ok") is True
        and (
            row["tool_result"].get("value") in {"41", "17"}
            or row["tool_result"].get("version") == 1
        )
        for row in transcript
    )
    verification_calls = sum(
        1
        for row in tool_calls
        if (row.get("action") or {}).get("tool") in {"verify_checksum", "version_status", "backup_lookup"}
    )
    return {
        "success": success,
        "final_value": final_value,
        "tool_calls": len(tool_calls),
        "blind_identical_retries": blind_retries,
        "bad_success_was_present": detected_bad_success,
        "verification_or_fallback_calls": verification_calls,
    }
