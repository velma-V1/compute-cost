"""Foundational GPT-OSS runtime and role-specialization labs for Test 1.2.

These probes answer upstream questions before higher-level harness discovery is
allowed to dominate the clock.  Collection discovers the operating semantics;
Run 2/Test 2 owns recurrence and proof.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from typing import Any

from .scoring import score_case
from .test1_campaign import _family, _fixture_id


FOUNDATION_QUESTIONS: tuple[dict[str, Any], ...] = (
    {"id":1,"group":"runtime","critical":True,"text":"Does Ollama honor think:false for gpt-oss, error, or substitute thinking?"},
    {"id":2,"group":"runtime","critical":True,"text":"Does native format=json produce empty content with populated thinking, by task family?"},
    {"id":3,"group":"runtime","critical":True,"text":"Is thinking always separated from content or does reasoning leak into content?"},
    {"id":4,"group":"runtime","critical":True,"text":"Does done_reason identify thinking-phase versus answer-phase exhaustion?"},
    {"id":5,"group":"runtime","critical":True,"text":"Does num_predict cap thinking plus answer or answer only?"},
    {"id":6,"group":"runtime","critical":True,"text":"What does eval_count include and can thinking/answer tokens be separated natively?"},
    {"id":7,"group":"sampling","critical":False,"text":"Does explicit top_p=1.0 outperform the runtime/default call path?"},
    {"id":8,"group":"sampling","critical":False,"text":"How does temperature=0.0 versus 1.0 move capability by family?"},
    {"id":9,"group":"runtime","critical":True,"text":"Does the harness preserve/drop prior thinking correctly across final answers and tool loops?"},
    {"id":10,"group":"runtime","critical":True,"text":"Does a tool call terminate cleanly or require harness stop handling?"},
    {"id":11,"group":"budget","critical":True,"text":"Minimum reproduced passing generation budget per family and reasoning effort."},
    {"id":12,"group":"budget","critical":True,"text":"Observable thinking/answer share at the passing budget and with rising difficulty."},
    {"id":13,"group":"budget","critical":True,"text":"Sharpness of THINK_TRUNCATED to PASS transition."},
    {"id":14,"group":"budget","critical":False,"text":"Boundary continuity across difficulty."},
    {"id":15,"group":"budget","critical":False,"text":"Accuracy plateau at 2x/4x/10x minimum budget."},
    {"id":16,"group":"budget","critical":False,"text":"Accuracy drop at excess budget / overthink corruption."},
    {"id":17,"group":"budget","critical":False,"text":"Families where low effort beats high."},
    {"id":18,"group":"budget","critical":False,"text":"Accuracy-per-token Pareto effort by family."},
    {"id":19,"group":"budget","critical":False,"text":"Analysis-loop rate by category."},
    {"id":20,"group":"budget","critical":False,"text":"Early observable predictor for doomed generation."},
    {"id":21,"group":"output","critical":False,"text":"First-attempt parse rate."},
    {"id":22,"group":"output","critical":False,"text":"Developer JSON instruction versus runtime format=json."},
    {"id":23,"group":"output","critical":False,"text":"Schema/grammar effect on empty-content failures."},
    {"id":24,"group":"output","critical":False,"text":"Cheapest output contract meeting parse target."},
    {"id":25,"group":"output","critical":False,"text":"Parse-retry convergence."},
    {"id":26,"group":"output","critical":False,"text":"Wrong-answer versus unparseable-answer classification separation."},
    {"id":27,"group":"context","critical":False,"text":"Real usable context before retrieval degradation."},
    {"id":28,"group":"context","critical":False,"text":"Context ceiling versus reasoning effort."},
    {"id":29,"group":"context","critical":False,"text":"Observable truncation behavior when num_ctx is exceeded."},
    {"id":30,"group":"context","critical":False,"text":"Retrieval rate versus planted-key position."},
    {"id":31,"group":"context","critical":False,"text":"VRAM/latency cost per context doubling and hardware knee."},
    {"id":32,"group":"role","critical":True,"text":"Auditor versus executor accuracy on matched tasks."},
    {"id":33,"group":"role","critical":True,"text":"Auditor versus executor budget separation."},
    {"id":34,"group":"role","critical":True,"text":"Auditor false-accept versus false-reject asymmetry."},
    {"id":35,"group":"role","critical":False,"text":"Audit accuracy across candidate-quality sweep."},
    {"id":36,"group":"role","critical":False,"text":"Audit candidate alone versus candidate plus reasoning."},
    {"id":37,"group":"role","critical":False,"text":"Second audit-pass verdict stability."},
    {"id":38,"group":"role","critical":True,"text":"Low-effort versus high-effort auditor accuracy/cost."},
    {"id":39,"group":"reliability","critical":False,"text":"Seed spread at temperature=1.0 and effect uncertainty."},
    {"id":40,"group":"reliability","critical":False,"text":"Cold versus warm capability, not only latency."},
    {"id":41,"group":"reliability","critical":False,"text":"Sustained-load throughput or accuracy degradation."},
    {"id":42,"group":"reliability","critical":False,"text":"Full-campaign wall-clock and Wh cost."},
    {"id":43,"group":"trust","critical":False,"text":"Can candidate text steer auditor verdict?"},
    {"id":44,"group":"trust","critical":False,"text":"Does audit rationale diverge from verdict?"},
    {"id":45,"group":"trust","critical":False,"text":"Can thinking/analysis reach any scored output path?"},
)

CRITICAL_FOUNDATION_IDS = frozenset(
    row["id"] for row in FOUNDATION_QUESTIONS if row["critical"]
)


def _json_ok(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def _generation_record(
    *,
    probe_id: str,
    question_ids: list[int],
    family_id: str,
    generation: dict[str, Any],
    request: dict[str, Any],
    expected: Any = None,
    scorer: str | None = None,
    score: float | None = None,
) -> dict[str, Any]:
    normalized = generation.get("normalized") or {}
    metrics = generation.get("metrics") or {}
    phases = generation.get("phase_metrics") or {}
    content = str(normalized.get("text") or "")
    thinking = str(normalized.get("thinking") or "")
    done_reason = normalized.get("done_reason")
    eval_count = metrics.get("eval_count")
    requested_budget = (request.get("options") or {}).get("num_predict")
    return {
        "schema_version": 1,
        "probe_id": probe_id,
        "question_ids": list(question_ids),
        "family_id": family_id,
        "request": copy.deepcopy(request),
        "ok": bool(generation.get("ok")),
        "http_status": generation.get("http_status"),
        "content": content,
        "thinking": thinking,
        "content_chars": len(content),
        "thinking_chars": len(thinking),
        "content_empty": not bool(content.strip()),
        "thinking_present": bool(thinking.strip()),
        "thinking_markup_in_content": any(
            marker in content.lower()
            for marker in ("<think>", "</think>", "<|channel|>analysis", "<|channel|>thinking")
        ),
        "tool_call_count": len(normalized.get("tool_calls") or []),
        "tool_calls": copy.deepcopy(normalized.get("tool_calls") or []),
        "done": normalized.get("done"),
        "done_reason": done_reason,
        "eval_count": eval_count,
        "requested_num_predict": requested_budget,
        "eval_hit_budget": (
            isinstance(eval_count, int)
            and isinstance(requested_budget, int)
            and int(eval_count) >= int(requested_budget)
        ),
        "json_parse_ok": _json_ok(content) if content.strip() else False,
        "score": score,
        "scorer": scorer,
        "expected": copy.deepcopy(expected),
        "phase_metrics": copy.deepcopy(phases),
        "native_thinking_answer_token_split_available": False,
        "native_split_reason": (
            "Ollama exposes aggregate eval_count and separate text fields/chunk timing, "
            "but no per-field token counts through this interface."
        ),
    }


def _invoke_probe(
    campaign: Any,
    deadline: float,
    *,
    probe_id: str,
    question_ids: list[int],
    family_id: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    request_fields: dict[str, Any] | None = None,
    case: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not campaign.can_start(deadline) or not campaign._has_runway(deadline, 1):
        return None
    label = f"test1.2 foundation {probe_id}"
    campaign._progress(label, True)
    try:
        generation, invocation, _ = campaign.runner._invoke_generation(
            stage="test1.2-foundation",
            case_id=probe_id,
            messages=messages,
            options=options,
            request_fields=request_fields,
        )
    finally:
        campaign._progress(label, False)

    score = None
    scorer = None
    expected = None
    if case is not None and generation.get("ok"):
        scorer = str(case.get("scorer") or "")
        expected = case.get("expected")
        scoring = score_case(case, str((generation.get("normalized") or {}).get("text") or ""))
        raw_score = scoring.get("score")
        if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
            score = float(raw_score)

    row = _generation_record(
        probe_id=probe_id,
        question_ids=question_ids,
        family_id=family_id,
        generation=generation,
        request=invocation,
        expected=expected,
        scorer=scorer,
        score=score,
    )
    campaign.runner.store.append_jsonl(
        "test1.2-foundation-observations.jsonl",
        row,
    )
    return row


def _representative_cases(campaign: Any) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in campaign.partitions["DISCOVERY"]:
        by_family[_family(case)].append(case)
    result = []
    for family in sorted(by_family):
        pool = sorted(
            by_family[family],
            key=lambda case: (
                -int(case.get("difficulty_level") or 0),
                _fixture_id(case),
            ),
        )
        result.append(pool[0])
    return result


def run_runtime_semantics_gate(campaign: Any, deadline: float) -> dict[str, Any]:
    """Answer interface questions 1-6, 9-10 before normal Collection search."""
    rows: list[dict[str, Any]] = []
    base_options = {
        "num_predict": 96,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 42,
    }
    exact_messages = [{"role":"user","content":"Reply with exactly: OK"}]

    for name, request_fields in (
        ("think-false", {"think": False}),
        ("think-omitted", {}),
        ("think-low", {"think": "low"}),
        ("think-high", {"think": "high"}),
    ):
        row = _invoke_probe(
            campaign,
            deadline,
            probe_id=f"runtime-{name}",
            question_ids=[1,3,4,5,6],
            family_id="RUNTIME_SEMANTICS",
            messages=exact_messages,
            options=base_options,
            request_fields=request_fields,
        )
        if row is not None:
            rows.append(row)

    # Deliberately force length stops to determine whether done_reason alone
    # identifies thinking-phase versus answer-phase exhaustion.
    for budget in (8, 16, 32):
        row = _invoke_probe(
            campaign,
            deadline,
            probe_id=f"runtime-budget-cliff-{budget}",
            question_ids=[4,5,6,13],
            family_id="RUNTIME_SEMANTICS",
            messages=[{"role":"user","content":"Compute 987654321 * 123456789. Return only the integer."}],
            options={**base_options, "num_predict": budget},
            request_fields={"think": "high"},
        )
        if row is not None:
            rows.append(row)

    # Measure native format=json behavior once per capability family.
    for case in _representative_cases(campaign):
        if not campaign.can_start(deadline):
            break
        prompt = (
            str(case.get("prompt") or "")
            + "\nReturn a JSON object with exactly one string field named answer."
        )
        row = _invoke_probe(
            campaign,
            deadline,
            probe_id=f"runtime-format-json-{_fixture_id(case)}",
            question_ids=[2,3,6,21,22,23],
            family_id=_family(case),
            messages=[{"role":"user","content":prompt}],
            options=base_options,
            request_fields={"think":"medium","format":"json"},
        )
        if row is not None:
            rows.append(row)

    # Real Ollama tool-call boundary. This checks parser/termination semantics;
    # deterministic synthetic tool correctness remains covered elsewhere.
    tool_schema = [{
        "type":"function",
        "function":{
            "name":"add",
            "description":"Add two integers.",
            "parameters":{
                "type":"object",
                "required":["a","b"],
                "properties":{
                    "a":{"type":"integer"},
                    "b":{"type":"integer"},
                },
            },
        },
    }]
    tool_row = _invoke_probe(
        campaign,
        deadline,
        probe_id="runtime-real-tool-call",
        question_ids=[9,10],
        family_id="tool_selection",
        messages=[{"role":"user","content":"Use the add tool to add 17 and 23. Do not compute it yourself."}],
        options=base_options,
        request_fields={"think":"medium","tools":tool_schema},
    )
    if tool_row is not None:
        rows.append(tool_row)

        calls = tool_row.get("tool_calls") or []
        if calls and campaign.can_start(deadline):
            assistant = {
                "role":"assistant",
                "thinking":tool_row.get("thinking") or "",
                "content":tool_row.get("content") or "",
                "tool_calls":copy.deepcopy(calls),
            }
            follow = _invoke_probe(
                campaign,
                deadline,
                probe_id="runtime-real-tool-followup",
                question_ids=[9,10],
                family_id="tool_error_recovery",
                messages=[
                    {"role":"user","content":"Use the add tool to add 17 and 23. Do not compute it yourself."},
                    assistant,
                    {"role":"tool","tool_name":"add","content":"40"},
                ],
                options=base_options,
                request_fields={"think":"medium","tools":tool_schema},
            )
            if follow is not None:
                rows.append(follow)

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("family_id") or "UNKNOWN")].append(row)

    format_rates = {}
    for family, values in by_family.items():
        format_rows = [row for row in values if str(row.get("probe_id","")).startswith("runtime-format-json-")]
        if not format_rows:
            continue
        format_rates[family] = {
            "n": len(format_rows),
            "empty_content_with_thinking_rate": (
                sum(1 for row in format_rows if row["content_empty"] and row["thinking_present"])
                / len(format_rows)
            ),
            "json_parse_rate": sum(1 for row in format_rows if row["json_parse_ok"]) / len(format_rows),
            "thinking_leak_rate": sum(1 for row in format_rows if row["thinking_markup_in_content"]) / len(format_rows),
        }

    false_row = next((row for row in rows if row["probe_id"] == "runtime-think-false"), None)
    if false_row is None:
        think_false_behavior = "UNMEASURED"
    elif not false_row["ok"]:
        think_false_behavior = "ERROR"
    elif false_row["thinking_present"]:
        think_false_behavior = "ACCEPTED_BUT_THINKING_PRESENT"
    else:
        think_false_behavior = "ACCEPTED_NO_EXPOSED_THINKING"

    length_rows = [row for row in rows if str(row.get("probe_id","")).startswith("runtime-budget-cliff-")]
    stage_ambiguity = any(
        row.get("done_reason") == "length"
        and row.get("content_empty")
        and row.get("thinking_present")
        for row in length_rows
    )
    return {
        "schema_version":1,
        "questions_answered":[1,2,3,4,5,6,9,10],
        "think_false_behavior":think_false_behavior,
        "format_json_by_family":format_rates,
        "thinking_markup_leak_count":sum(1 for row in rows if row["thinking_markup_in_content"]),
        "done_reason_stage_ambiguity_observed":stage_ambiguity,
        "num_predict_total_generation_cap_evidence":[
            {
                "probe_id":row["probe_id"],
                "requested_num_predict":row["requested_num_predict"],
                "eval_count":row["eval_count"],
                "eval_hit_budget":row["eval_hit_budget"],
                "content_empty":row["content_empty"],
                "thinking_present":row["thinking_present"],
                "done_reason":row["done_reason"],
            }
            for row in length_rows
        ],
        "eval_count_semantics":{
            "native_value":"aggregate generated token count",
            "native_thinking_answer_split_available":False,
            "observable_proxies":["thinking_chars","answer_chars","thinking_chunks","answer_chunks","phase timing"],
        },
        "tool_call_probe":next(
            (row for row in rows if row["probe_id"] == "runtime-real-tool-call"),
            None,
        ),
        "tool_followup_probe":next(
            (row for row in rows if row["probe_id"] == "runtime-real-tool-followup"),
            None,
        ),
        "observation_count":len(rows),
    }



def _round_up_budget(value: float, ladder: list[int]) -> int:
    for budget in sorted(int(v) for v in ladder):
        if budget >= value:
            return budget
    return max(int(v) for v in ladder)


def run_runtime_budget_characterization(
    campaign: Any,
    deadline: float,
    *,
    replicates: int = 3,
    safety_factor: float = 1.5,
) -> dict[str, Any]:
    """Replicated Stage-0 family budget calibration.

    The operating point is not the first lucky valid completion. A family must
    produce k/k capability-valid final answers at a budget, then the deployed
    baseline is moved up by a safety factor and rounded to the tested ladder.
    No campaign config is mutated here.
    """
    ladder = sorted(
        {
            int(campaign.cfg.get("base_generation_budget") or 256),
            *[int(v) for v in campaign.cfg.get("generation_budgets", [256, 512, 1024, 2048])],
        }
    )
    seeds = list(campaign.cfg.get("seeds") or [42, 43, 44])[:replicates]
    if len(seeds) < replicates:
        base = seeds[-1] if seeds else 42
        seeds.extend(base + i + 1 for i in range(replicates - len(seeds)))

    families: dict[str, Any] = {}
    unresolved: list[str] = []
    cases = _representative_cases(campaign)

    for case in cases:
        if not campaign.can_start(deadline):
            break
        family = _family(case)
        levels: dict[str, Any] = {}
        reproducible_boundary: int | None = None
        reproducible_pass_boundary: int | None = None

        for budget in ladder:
            if not campaign.can_start(deadline):
                break
            obs: list[dict[str, Any]] = []
            for seed in seeds:
                if not campaign.can_start(deadline):
                    break
                row = _invoke_probe(
                    campaign,
                    deadline,
                    probe_id=f"stage0-budget-{family}-{budget}-s{seed}",
                    question_ids=[11,12,13,14,15,16,17,18,19,20],
                    family_id=family,
                    messages=[{"role":"user","content":str(case.get("prompt") or "")}],
                    options={
                        "num_predict":int(budget),
                        "temperature":1.0,
                        "top_p":1.0,
                        "seed":int(seed),
                    },
                    request_fields={"think":"medium"},
                    case=case,
                )
                if row is not None:
                    row["capability_valid_final_answer"] = bool(
                        row.get("ok")
                        and not row.get("content_empty")
                        and row.get("done_reason") != "length"
                    )
                    obs.append(row)
            valid = [row for row in obs if row.get("capability_valid_final_answer") is True]
            passed = [row for row in valid if float(row.get("score") or 0.0) >= 1.0]
            levels[str(budget)] = {
                "attempts":len(obs),
                "valid_final_answers":len(valid),
                "valid_rate":(len(valid)/len(obs)) if obs else None,
                "passes":len(passed),
                "pass_rate_among_valid":(len(passed)/len(valid)) if valid else None,
                "seeds":[int(row.get("request",{}).get("options",{}).get("seed") or 0) for row in obs],
                "done_reasons":sorted({str(row.get("done_reason")) for row in obs}),
                "mean_eval_count":(
                    sum(int(row["eval_count"]) for row in obs if isinstance(row.get("eval_count"), int))
                    / max(1, sum(1 for row in obs if isinstance(row.get("eval_count"), int)))
                ),
            }
            if len(obs) == replicates and len(valid) == replicates and reproducible_boundary is None:
                reproducible_boundary = int(budget)
            if len(obs) == replicates and len(passed) == replicates and reproducible_pass_boundary is None:
                reproducible_pass_boundary = int(budget)
            if reproducible_boundary is not None:
                # No need to spend discovery clock proving larger raw boundaries.
                break

        if reproducible_boundary is None:
            unresolved.append(family)
            safe_budget = max(ladder)
            basis = "NO_REPRODUCIBLE_VALID_BOUNDARY"
        else:
            safe_budget = _round_up_budget(
                float(reproducible_boundary) * float(safety_factor),
                ladder,
            )
            basis = "K_OF_K_VALID_BOUNDARY_WITH_SAFETY_FACTOR"

        families[family] = {
            "fixture_id":_fixture_id(case),
            "difficulty_level":int(case.get("difficulty_level") or 0),
            "replicates_required":int(replicates),
            "reasoning_effort":"medium",
            "temperature":1.0,
            "top_p":1.0,
            "tested_budget_ladder":list(ladder),
            "levels":levels,
            "minimum_reproducibly_valid_budget":reproducible_boundary,
            "minimum_reproducibly_passing_budget":reproducible_pass_boundary,
            "resolved_safe_baseline_budget":int(safe_budget),
            "safety_factor":float(safety_factor),
            "resolution_basis":basis,
        }

    expected_families = sorted({_family(case) for case in cases})
    missing = sorted(set(expected_families) - set(families))
    unresolved = sorted(set(unresolved) | set(missing))
    resolved = {
        family:int(payload["resolved_safe_baseline_budget"])
        for family, payload in families.items()
        if family not in unresolved
    }
    return {
        "schema_version":1,
        "stage":"STAGE0_RUNTIME_CHARACTERIZATION",
        "questions_answered":[11,12,13,14,15,16,17,18,19,20],
        "replicates_required":int(replicates),
        "safety_factor":float(safety_factor),
        "budget_ladder":list(ladder),
        "families":families,
        "resolved_generation_budget_by_family":resolved,
        "unresolved_families":unresolved,
        "all_families_reproducibly_valid":not bool(unresolved),
        "config_mutated_during_characterization":False,
    }


def build_runtime_characterization_profile(
    campaign: Any,
    runtime_semantics: dict[str, Any],
    budget_characterization: dict[str, Any],
    role_specialization: dict[str, Any],
) -> dict[str, Any]:
    runtime_snapshot = {}
    runtime_path = getattr(getattr(campaign.runner, "store", None), "run_dir", None)
    if runtime_path is not None:
        path = runtime_path / "runtime.json"
        if path.is_file():
            try:
                runtime_snapshot = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                runtime_snapshot = {}

    critical_runtime = {1,2,3,4,5,6,9,10}
    runtime_answered = set(runtime_semantics.get("questions_answered") or [])
    role_answered = set(role_specialization.get("questions_answered") or [])
    budget_answered = set(budget_characterization.get("questions_answered") or [])

    gate_reasons = []
    if not critical_runtime.issubset(runtime_answered):
        gate_reasons.append("RUNTIME_SEMANTICS_INCOMPLETE")
    if not budget_characterization.get("all_families_reproducibly_valid"):
        gate_reasons.append("FAMILY_BUDGET_CALIBRATION_INCOMPLETE")
    if not {32,33,34,38}.issubset(role_answered):
        gate_reasons.append("ROLE_SPECIALIZATION_INCOMPLETE")

    identity = {
        "model":getattr(campaign.runner, "model", None),
        "runtime_version":runtime_snapshot.get("version"),
        "model_size_bytes":runtime_snapshot.get("model_size_bytes"),
        "model_info":runtime_snapshot.get("model_info"),
    }
    payload = {
        "schema_version":1,
        "stage":"STAGE0_RUNTIME_CHARACTERIZATION",
        "identity":identity,
        "runtime_semantics":copy.deepcopy(runtime_semantics),
        "budget_characterization":copy.deepcopy(budget_characterization),
        "role_specialization":copy.deepcopy(role_specialization),
        "resolved_generation_budget_by_family":copy.deepcopy(
            budget_characterization.get("resolved_generation_budget_by_family") or {}
        ),
        "gate_passed":not bool(gate_reasons),
        "gate_failures":gate_reasons,
        "capability_claims_allowed":not bool(gate_reasons),
        "profile_scope":"EXACT_MODEL_RUNTIME_QUANT_CONFIGURATION",
    }
    stable = json.dumps(payload, sort_keys=True, separators=(",",":"), default=str)
    payload["profile_sha256"] = hashlib.sha256(stable.encode("utf-8")).hexdigest()
    return payload

def _audit_case(case: dict[str, Any], candidate: str) -> dict[str, Any]:
    return {
        "prompt": (
            "Audit the candidate against the task. Return exactly ACCEPT if it is fully correct, "
            "otherwise return exactly REJECT.\n\nTASK:\n"
            + str(case.get("prompt") or "")
            + "\n\nCANDIDATE:\n"
            + candidate
        ),
        "scorer":"exact",
    }


def run_role_specialization_lab(campaign: Any, deadline: float) -> dict[str, Any]:
    """Paired executor/auditor lab for questions 32-34 and 38."""
    rows: list[dict[str, Any]] = []
    cases = _representative_cases(campaign)
    # Spread across all families when time allows; the deadline remains binding.
    for case in cases:
        if not campaign.can_start(deadline):
            break
        family = _family(case)
        options = {"num_predict":256,"temperature":1.0,"top_p":1.0,"seed":42}
        executor = _invoke_probe(
            campaign,
            deadline,
            probe_id=f"role-executor-{_fixture_id(case)}",
            question_ids=[32,33],
            family_id=family,
            messages=[{"role":"user","content":str(case.get("prompt") or "")}],
            options=options,
            request_fields={"think":"medium"},
            case=case,
        )
        if executor is None:
            break
        rows.append(executor)
        executor_correct = float(executor.get("score") or 0.0) >= 1.0
        expected_verdict = "ACCEPT" if executor_correct else "REJECT"
        audit_case = _audit_case(case, str(executor.get("content") or ""))
        audit_case["expected"] = expected_verdict
        audit_case["id"] = f"audit-{_fixture_id(case)}"
        audit_case["category"] = family

        for effort in ("low","high"):
            if not campaign.can_start(deadline):
                break
            audit = _invoke_probe(
                campaign,
                deadline,
                probe_id=f"role-auditor-{effort}-{_fixture_id(case)}",
                question_ids=[32,33,34,38],
                family_id=family,
                messages=[{"role":"user","content":audit_case["prompt"]}],
                options=options,
                request_fields={"think":effort},
                case=audit_case,
            )
            if audit is not None:
                audit["candidate_was_correct"] = executor_correct
                audit["expected_verdict"] = expected_verdict
                rows.append(audit)
                campaign.runner.store.append_jsonl(
                    "test1.2-role-specialization-observations.jsonl",
                    audit,
                )

    executor_rows=[row for row in rows if str(row.get("probe_id","")).startswith("role-executor-")]
    low=[row for row in rows if str(row.get("probe_id","")).startswith("role-auditor-low-")]
    high=[row for row in rows if str(row.get("probe_id","")).startswith("role-auditor-high-")]

    def accuracy(values: list[dict[str, Any]]) -> float | None:
        scored=[row for row in values if row.get("score") is not None]
        return None if not scored else sum(float(row["score"]) for row in scored)/len(scored)

    def mean_eval(values: list[dict[str, Any]]) -> float | None:
        counts=[int(row["eval_count"]) for row in values if isinstance(row.get("eval_count"), int)]
        return None if not counts else sum(counts)/len(counts)

    false_accept=0
    false_reject=0
    for row in [*low,*high]:
        verdict=str(row.get("content") or "").strip().upper()
        correct=bool(row.get("candidate_was_correct"))
        if not correct and verdict=="ACCEPT":
            false_accept += 1
        if correct and verdict=="REJECT":
            false_reject += 1

    return {
        "schema_version":1,
        "questions_answered":[32,33,34,38],
        "matched_family_count":len({str(row.get("family_id")) for row in executor_rows}),
        "executor_accuracy":accuracy(executor_rows),
        "auditor_low_accuracy":accuracy(low),
        "auditor_high_accuracy":accuracy(high),
        "executor_mean_eval_count":mean_eval(executor_rows),
        "auditor_low_mean_eval_count":mean_eval(low),
        "auditor_high_mean_eval_count":mean_eval(high),
        "false_accepts":false_accept,
        "false_rejects":false_reject,
        "observation_count":len(rows),
        "proof_owner":"RUN2_TEST2",
    }


def foundation_question_ledger(
    runtime_map: dict[str, Any] | None,
    role_map: dict[str, Any] | None,
) -> dict[str, Any]:
    answered = set((runtime_map or {}).get("questions_answered") or [])
    answered.update((role_map or {}).get("questions_answered") or [])
    # Existing Test 1.2 owners for later groups are explicitly declared rather
    # than incorrectly marked as answered by these foundation probes.
    owners = {
        **{i:"runtime_semantics_gate" for i in (1,2,3,4,5,6,9,10)},
        **{i:"fractional_compute_surface" for i in range(11,21)},
        **{i:"output_contract_and_existing_format_families" for i in range(21,27)},
        **{i:"context/frontier labs" for i in range(27,32)},
        **{i:"role_specialization_gate" for i in (32,33,34,38)},
        **{i:"role/trust follow-on discovery" for i in (35,36,37,43,44,45)},
        **{i:"reliability/cost evidence" for i in range(39,43)},
        7:"fractional_compute_surface",
        8:"fractional_compute_surface",
    }
    rows=[]
    for question in FOUNDATION_QUESTIONS:
        qid=int(question["id"])
        rows.append({
            **copy.deepcopy(question),
            "status":"MEASURED_IN_FOUNDATION" if qid in answered else "ASSIGNED_TO_LATER_PHASE",
            "owner":owners.get(qid,"UNASSIGNED"),
        })
    return {
        "schema_version":1,
        "questions":rows,
        "foundation_answered_ids":sorted(answered),
        "critical_foundation_ids":sorted(CRITICAL_FOUNDATION_IDS),
        "missing_critical_foundation_ids":sorted(CRITICAL_FOUNDATION_IDS-answered),
    }
