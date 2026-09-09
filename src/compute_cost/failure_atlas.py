"""Evidence-first failure origin, signature, and atlas derivation.

Failure signatures are deliberately not represented as causal diagnoses.  A
family-local signature records the type of task that failed and can seed later
replay/recovery experiments; unsupported cognitive root causes remain
``unresolved`` until stronger evidence exists.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


ANSWER_WRONG_SUBTYPES = [
    "knowledge_gap",
    "logic_error",
    "arithmetic_error",
    "assumption_error",
    "premature_commit",
    "instruction_loss",
    "distractor_capture",
    "state_loss",
    "stale_state_use",
    "hallucinated_fact",
    "tool_selection_error",
    "tool_argument_error",
    "verification_failure",
    "cascading_error",
    "unresolved",
]

MODEL_RESULT_CLASSES = {
    "ANSWER_WRONG",
    "FORMAT_FAILURE",
    "TOOL_FAILURE",
    "CONTEXT_FAILURE",
    "NO_FINAL_ANSWER",
    "THINK_TRUNCATED",
    "ANSWER_TRUNCATED",
    "REASONING_LOOP",
    "OVERTHINK_CORRUPTION",
}
EVALUATOR_RESULT_CLASSES = {"SCORER_DEFECT"}
INFRA_RESULT_CLASSES = {"RUNTIME_FAILURE", "RESOURCE_LIMIT", "CAPTURE_GAP"}
INVALID_FIXTURE_RESULT_CLASSES = {"TEST_DEFECT"}
NON_FAILURE_RESULT_CLASSES = {"ANSWER_CORRECT", "SELF_CORRECTED"}


FAMILY_SIGNATURES = {
    "formal_logic_deduction": "logic_error",
    "arithmetic_numerical_reasoning": "arithmetic_error",
    "algebra_quantitative_reasoning": "arithmetic_error",
    "ambiguity_detection": "assumption_error",
    "missing_information_handling": "assumption_error",
    "instruction_following_constraint_stacking": "instruction_loss",
    "prompt_instruction_conflict_handling": "instruction_loss",
    "distractor_noise_resistance": "distractor_capture",
    "lost_in_middle_resistance": "distractor_capture",
    "multi_turn_state_tracking": "state_loss",
    "memory_compression_summary_fidelity": "state_loss",
    "updated_obsolete_state_rejection": "stale_state_use",
    "hallucination_resistance": "hallucinated_fact",
    "tool_selection": "tool_selection_error",
    "tool_argument_correctness": "tool_argument_error",
    "verification_critique": "verification_failure",
    "test_generation_verification": "verification_failure",
    "composite_agent_tasks": "cascading_error",
}


def classify_failure_origin(result_class: str) -> str | None:
    """Map behavioral result class to experiment failure origin."""
    value = str(result_class)
    if value in NON_FAILURE_RESULT_CLASSES:
        return None
    if value == "TIMEOUT":
        return "TIMEOUT"
    if value in EVALUATOR_RESULT_CLASSES:
        return "EVALUATOR_FAILURE"
    if value in INFRA_RESULT_CLASSES:
        return "INFRA_FAILURE"
    if value in INVALID_FIXTURE_RESULT_CLASSES:
        return "INVALID_FIXTURE"
    if value in MODEL_RESULT_CLASSES:
        return "MODEL_FAILURE"
    return "INFRA_FAILURE"


def infer_failure_signature(family_id: str, result_class: str) -> dict[str, Any]:
    """Return a conservative task-family signature, never a causal diagnosis."""
    origin = classify_failure_origin(result_class)
    if origin != "MODEL_FAILURE":
        return {
            "subtype": "unresolved",
            "inference_kind": "NOT_MODEL_CAPABILITY_EVIDENCE",
            "causal_claim": False,
        }

    subtype = FAMILY_SIGNATURES.get(str(family_id))
    if subtype is None:
        return {
            "subtype": "unresolved",
            "inference_kind": "INSUFFICIENT_EVIDENCE",
            "causal_claim": False,
        }
    return {
        "subtype": subtype,
        "inference_kind": "FAMILY_LOCAL_SIGNATURE",
        "causal_claim": False,
    }


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def build_failure_atlas(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a lossless failure index from experiment rows without inventing causes."""
    failures: list[dict[str, Any]] = []
    by_origin: Counter[str] = Counter()
    by_result: Counter[str] = Counter()
    by_subtype: Counter[str] = Counter()
    model_failures = 0
    invalid_experiments = 0

    for row in rows:
        classification = row.get("classification") or {}
        result_class = str(classification.get("result_class") or "")
        origin = classify_failure_origin(result_class)
        if origin is None:
            continue

        experiment = row.get("experiment") or {}
        family_id = str(
            experiment.get("task_family")
            or experiment.get("task_id")
            or "unknown"
        )
        valid_for_capability = classification.get("valid_for_capability") is True
        signature = infer_failure_signature(family_id, result_class)

        entry = {
            "experiment_id": experiment.get("experiment_id"),
            "parent_experiment_id": experiment.get("parent_experiment_id"),
            "family_id": family_id,
            "difficulty_level": experiment.get("difficulty_level"),
            "reasoning_effort": experiment.get("reasoning_effort"),
            "recovery_level": experiment.get("recovery_level"),
            "result_class": result_class,
            "failure_origin": origin,
            "failure_signature": signature,
            "valid_for_capability": valid_for_capability,
            "score": row.get("score"),
            "status": row.get("status"),
            "evidence_key": row.get("evidence_key"),
            "evidence_refs": row.get("evidence_refs"),
        }
        failures.append(entry)
        by_origin[origin] += 1
        by_result[result_class] += 1
        by_subtype[str(signature["subtype"])] += 1
        if origin == "MODEL_FAILURE" and valid_for_capability:
            model_failures += 1
        if not valid_for_capability:
            invalid_experiments += 1

    return {
        "schema_version": 1,
        "model": model,
        "measurement_policy": {
            "failure_origin": "DERIVED_FROM_RESULT_CLASS",
            "failure_signature": "FAMILY_LOCAL_SIGNATURE_OR_UNRESOLVED",
            "causal_root_cause_claimed": False,
            "invalid_experiments_are_model_failures": False,
        },
        "taxonomy": {
            "origins": [
                "MODEL_FAILURE",
                "EVALUATOR_FAILURE",
                "TOOL_FAILURE",
                "INFRA_FAILURE",
                "TIMEOUT",
                "INVALID_FIXTURE",
            ],
            "answer_wrong_subtypes": list(ANSWER_WRONG_SUBTYPES),
        },
        "summary": {
            "failure_count": len(failures),
            "model_failures": model_failures,
            "invalid_experiments": invalid_experiments,
            "by_origin": _counter_dict(by_origin),
            "by_result_class": _counter_dict(by_result),
            "by_subtype": _counter_dict(by_subtype),
        },
        "failures": failures,
    }
