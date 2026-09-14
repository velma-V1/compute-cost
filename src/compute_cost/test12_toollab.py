"""Deterministic synthetic tool lab for Test 1.2.

This is real harness execution against safe in-process synthetic tools. It does
not claim native provider function-calling or access to external systems.
"""

from __future__ import annotations

import ast
import json
import math
import operator
from typing import Any


TOOL_MICROCASES: tuple[dict[str, Any], ...] = (
    {
        "id":"tool-live-calc",
        "prompt":"Compute 17*23 using the calculator tool and return the numeric result.",
        "expected_final":"391",
        "max_steps":3,
    },
    {
        "id":"tool-live-user",
        "prompt":"Retrieve user id 42 with get_user and return only the user's name. The schema requires id:string.",
        "expected_final":"Mira",
        "max_steps":3,
    },
    {
        "id":"tool-live-sequence",
        "prompt":"Find the current synthetic temperature in Paris with weather, then calculate 15% of it. Return only the number.",
        "expected_final":"3",
        "max_steps":4,
    },
    {
        "id":"tool-live-state",
        "prompt":"Set state key mode to safe, read it back, and return only the stored value.",
        "expected_final":"safe",
        "max_steps":4,
    },
)

TOOL_SCHEMAS = {
    "calculator":{"arguments":{"expression":"string"}},
    "get_user":{"arguments":{"id":"string"}},
    "weather":{"arguments":{"city":"string"}},
    "set_state":{"arguments":{"key":"string","value":"string"}},
    "get_state":{"arguments":{"key":"string"}},
}

TOOL_HARNESS_POLICIES: tuple[dict[str, Any], ...] = (
    {"id":"TOOL-LIVE-DIRECT","instruction":"Use the available tools as needed. Emit exactly one JSON tool call or one JSON final object at each step."},
    {"id":"TOOL-LIVE-SCHEMA","instruction":"Before emitting a call, check the tool name and exact argument names/types against the schema. Emit only JSON."},
    {"id":"TOOL-LIVE-DEPENDENCY","instruction":"Choose the minimum legal dependency order. Never call a downstream tool before its required upstream result exists. Emit only JSON."},
    {"id":"TOOL-LIVE-VERIFY","instruction":"Before every tool call and final answer, verify tool choice, argument types, dependencies, and the requested postcondition. Emit only JSON."},
    {"id":"TOOL-LIVE-RECOVERY","instruction":"If a tool returns an error, preserve valid state, diagnose the exact call defect, change the minimum necessary field, and retry. Emit only JSON."},
)


def tool_system_prompt(policy: dict[str, Any]) -> str:
    return (
        "You are operating a deterministic synthetic tool harness. "
        + str(policy["instruction"])
        + "\nAvailable tools and schemas:\n"
        + json.dumps(TOOL_SCHEMAS, sort_keys=True)
        + '\nTool call format: {"tool":"name","arguments":{...}}'
        + '\nFinal format: {"final":"value"}'
    )


def parse_action(text: str) -> dict[str, Any] | None:
    raw=str(text).strip()
    candidates=[raw]
    left=raw.find("{")
    right=raw.rfind("}")
    if left>=0 and right>left:
        candidates.append(raw[left:right+1])
    for candidate in candidates:
        try:
            value=json.loads(candidate)
        except Exception:
            continue
        if isinstance(value,dict) and ("tool" in value or "final" in value):
            return value
    return None


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_number(node: ast.AST) -> float:
    if isinstance(node,ast.Expression):
        return _safe_number(node.body)
    if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)) and not isinstance(node.value,bool):
        return float(node.value)
    if isinstance(node,ast.BinOp) and type(node.op) in _BINOPS:
        return float(_BINOPS[type(node.op)](_safe_number(node.left),_safe_number(node.right)))
    if isinstance(node,ast.UnaryOp) and type(node.op) in _UNARY:
        return float(_UNARY[type(node.op)](_safe_number(node.operand)))
    raise ValueError("unsupported arithmetic expression")


def execute_tool(name: str, arguments: dict[str, Any], state: dict[str, str]) -> dict[str, Any]:
    if name not in TOOL_SCHEMAS:
        return {"ok":False,"error":"UNKNOWN_TOOL"}
    expected=(TOOL_SCHEMAS[name].get("arguments") or {})
    if set(arguments) != set(expected):
        return {"ok":False,"error":"ARGUMENT_KEYS","expected":sorted(expected),"received":sorted(arguments)}
    for key,kind in expected.items():
        if kind=="string" and not isinstance(arguments.get(key),str):
            return {"ok":False,"error":"ARGUMENT_TYPE","argument":key,"expected":"string"}

    if name=="calculator":
        try:
            value=_safe_number(ast.parse(arguments["expression"],mode="eval"))
        except Exception as exc:
            return {"ok":False,"error":"CALCULATOR_ERROR","detail":type(exc).__name__}
        if math.isfinite(value) and value.is_integer():
            value=int(value)
        return {"ok":True,"result":value}
    if name=="get_user":
        if arguments["id"]!="42":
            return {"ok":False,"error":"USER_NOT_FOUND"}
        return {"ok":True,"result":{"id":"42","name":"Mira"}}
    if name=="weather":
        table={"Paris":20,"Atlanta":24,"Tokyo":18}
        city=arguments["city"]
        if city not in table:
            return {"ok":False,"error":"CITY_NOT_FOUND"}
        return {"ok":True,"result":{"city":city,"temperature_c":table[city]}}
    if name=="set_state":
        state[arguments["key"]]=arguments["value"]
        return {"ok":True,"result":{"stored":True}}
    if name=="get_state":
        key=arguments["key"]
        if key not in state:
            return {"ok":False,"error":"STATE_MISSING"}
        return {"ok":True,"result":state[key]}
    return {"ok":False,"error":"UNREACHABLE"}


def score_final(expected: str, actual: Any) -> bool:
    return str(actual).strip().lower() == str(expected).strip().lower()
