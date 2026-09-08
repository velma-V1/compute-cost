"""Deterministic base-v1 benchmark scorers."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from .hardware import run_command_capture


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip() + "\n"


def _validate_candidate(source: str, function_name: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"
    forbidden_names = {"open", "exec", "eval", "compile", "__import__", "os", "sys", "subprocess", "socket", "pathlib", "shutil"}
    found_function = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "imports are not allowed in base-v1 coding cases"
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            found_function = True
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            return f"forbidden name: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"dunder attribute is not allowed: {node.attr}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
            return f"forbidden call: {node.func.id}"
    if not found_function:
        return f"required function not found: {function_name}"
    return None


def _python_function(case: dict[str, Any], response: str) -> dict[str, Any]:
    source = _extract_code(response)
    function_name = str(case["function_name"])
    tests = str(case["tests"])
    rejection = _validate_candidate(source, function_name)
    evidence: dict[str, Any] = {
        "raw_response": response,
        "extracted_source": source,
        "executed_source_sha256": _sha(source),
        "test_source": tests,
        "test_source_sha256": _sha(tests),
        "safety_rejection": rejection,
    }
    if rejection:
        return {"score": 0.0, "status": "SCORED", "checks": [{"name": "safe_executable_subset", "pass": False}], "evidence": evidence}

    combined = source + "\n" + tests + "\n"
    with tempfile.TemporaryDirectory(prefix="compute-cost-case-") as temp:
        path = Path(temp) / "case.py"
        path.write_text(combined, encoding="utf-8")
        invocation_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        sub = run_command_capture(
            [sys.executable, "-I", "-S", str(path)],
            timeout=float(case.get("timeout_s", 5)),
        )
        evidence["invocation_sha256"] = invocation_sha
        evidence["subprocess"] = sub
        passed = sub.get("returncode") == 0 and sub.get("error") is None
        return {
            "score": 1.0 if passed else 0.0,
            "status": "SCORED",
            "checks": [{"name": "executable_tests", "pass": passed}],
            "evidence": evidence,
        }


def score_case(case: dict[str, Any], response: str, workspace: Path | None = None) -> dict[str, Any]:
    del workspace
    scorer = case.get("scorer")
    base_evidence: dict[str, Any] = {"raw_response": response}
    try:
        if scorer == "exact":
            expected = str(case.get("expected", "")).strip()
            actual = response.strip()
            passed = actual == expected
            return {"score": float(passed), "status": "SCORED", "checks": [{"name": "exact_match", "pass": passed, "expected": expected, "actual": actual}], "evidence": base_evidence}

        if scorer == "numeric":
            match = re.search(r"[-+]?\d+(?:\.\d+)?", response)
            actual = float(match.group(0)) if match else None
            expected = float(case["expected"])
            tolerance = float(case.get("tolerance", 0.0))
            passed = actual is not None and abs(actual - expected) <= tolerance
            return {"score": float(passed), "status": "SCORED", "checks": [{"name": "numeric_match", "pass": passed, "expected": expected, "actual": actual, "tolerance": tolerance}], "evidence": base_evidence}

        if scorer == "json":
            parsed = json.loads(response)
            expected = case.get("expected", {})
            required = case.get("required", list(expected.keys()) if isinstance(expected, dict) else [])
            checks = [{"name": "valid_json", "pass": True}]
            for key in required:
                checks.append({"name": f"required:{key}", "pass": isinstance(parsed, dict) and key in parsed})
            if isinstance(expected, dict):
                for key, value in expected.items():
                    checks.append({"name": f"value:{key}", "pass": isinstance(parsed, dict) and parsed.get(key) == value, "expected": value, "actual": parsed.get(key) if isinstance(parsed, dict) else None})
            passed = all(item["pass"] for item in checks)
            return {"score": float(passed), "status": "SCORED", "checks": checks, "evidence": {**base_evidence, "parsed": parsed}}

        if scorer == "extraction_set":
            parsed = json.loads(response)
            actual = set(parsed) if isinstance(parsed, list) and all(isinstance(v, str) for v in parsed) else set()
            expected = set(str(v) for v in case.get("expected", []))
            passed = actual == expected
            return {"score": float(passed), "status": "SCORED", "checks": [{"name": "exact_set", "pass": passed, "expected": sorted(expected), "actual": sorted(actual)}], "evidence": {**base_evidence, "parsed": parsed}}

        if scorer == "tool_call":
            parsed = json.loads(response)
            expected = case.get("expected")
            passed = parsed == expected
            return {"score": float(passed), "status": "SCORED", "checks": [{"name": "tool_call_exact", "pass": passed, "expected": expected, "actual": parsed}], "evidence": {**base_evidence, "parsed": parsed}}

        if scorer == "ambiguity":
            expected = str(case.get("expected", "")).upper()
            actual = response.strip().upper()
            passed = actual == expected or actual.startswith(expected + ":")
            return {"score": float(passed), "status": "SCORED", "checks": [{"name": "decision", "pass": passed, "expected": expected, "actual": actual}], "evidence": base_evidence}

        if scorer == "context_retrieval":
            expected = str(case.get("expected", ""))
            passed = expected in response
            return {"score": float(passed), "status": "SCORED", "checks": [{"name": "planted_fact", "pass": passed, "expected": expected}], "evidence": base_evidence}

        if scorer == "python_function":
            return _python_function(case, response)

        return {
            "score": None,
            "status": "SCORER_ERROR",
            "checks": [],
            "evidence": base_evidence,
            "error": {"type": "UnknownScorer", "message": f"unknown scorer: {scorer}"},
        }
    except Exception as exc:  # scorer defects are benchmark errors, never model failures
        return {
            "score": None,
            "status": "SCORER_ERROR",
            "checks": [],
            "evidence": base_evidence,
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        }
