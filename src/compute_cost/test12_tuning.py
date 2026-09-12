"""Test 1.2 tuning/compile run.

Consumes a completed Test 1.2 collection run and compiles a model-specific
adaptive harness in <= 6h15m. VALIDATION is used for tuning and policy lock;
TEST2_BLIND and TEST3_PROTECTED are then used exactly once for immutable final
acceptance. The two Test 1.2 runs are the complete model-onboarding decision
for Inverted; no later characterization test is required.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

from .evidence import EvidenceStore
from .test1_campaign import _balanced_cases, _family, _fixture_id, partition_cases
from .test12_campaign import (
    COLLECTION_HARD_SECONDS,
    FRONTIER_GAP_SURFACES,
    SECOND_GAP_SURFACES,
    TEST2_CAPABILITY_FAMILIES,
    Test12Campaign,
    _cost_value_frontier,
    _group_summary,
    _intervention_fingerprint,
    _rank_mechanisms,
    build_intervention_bank,
    fresh_model_source,
    partition_test12_cases,
)

TUNING_HARD_SECONDS = (6 * 60 * 60) + (15 * 60)
TUNING_ACTIVE_SECONDS = 6 * 60 * 60

TUNING_PHASES = (
    ("validation_baseline", 25 * 60),
    ("candidate_harness_screen", 65 * 60),
    ("successive_halving", 75 * 60),
    ("routing_and_boundary_tuning", 60 * 60),
    ("residual_failure_replay", 45 * 60),
    ("final_validation_lock", 30 * 60),
    ("test2_blind_acceptance", 30 * 60),
    ("test3_protected_acceptance", 30 * 60),
)

REQUIRED_COLLECTION_FILES = (
    "test1.2-observations.jsonl",
    "full-control-candidate-registry.json",
    "control-grammar-coverage.json",
    "mechanism-coverage-ledger.json",
    "capability-family-coverage.json",
    "capability-building-block-manufacturing-map.json",
    "capability-improvement-dossiers.json",
    "family-value-completeness.json",
    "control-response-tensor.json",
    "frontier-shift-map.json",
    "compute-quality-elasticity-map.json",
    "negative-effect-exploitation-map.json",
    "contrastive-negative-corpus.jsonl",
    "observation-value-index.jsonl",
    "adaptive-search-map.json",
    "metamorphic-reliability-map.json",
    "abstention-calibration-map.json",
    "active-memory-evolution-map.json",
    "reflection-transfer-map.json",
    "tool-chaos-recovery-map.json",
    "tool-scheduling-map.json",
    "frontier-gap-value-map.json",
    "authority-separation-map.json",
    "reward-hacking-resistance-map.json",
    "clarification-value-map.json",
    "governance-compaction-map.json",
    "belief-state-map.json",
    "semantic-transaction-map.json",
    "dynamic-replanning-map.json",
    "second-frontier-gap-value-map.json",
    "harness-to-weight-distillation-corpus.jsonl",
    "weighted-preference-corpus.jsonl",
    "capability-curriculum.json",
    "router-supervision-corpus.jsonl",
    "stability-anchor-corpus.jsonl",
    "cross-family-transfer-graph.json",
    "pareto-training-targets.jsonl",
    "zero-clock-model-manufacturing-map.json",
    "reliability-weighted-distillation-corpus.jsonl",
    "long-horizon-training-mix.json",
    "preference-quality-index.jsonl",
    "failure-credit-assignment-corpus.jsonl",
    "calibration-verify-supervision-corpus.jsonl",
    "cost-value-frontier-1.2.json",
    "activation-boundary-map.json",
    "negative-transfer-map-1.2.json",
    "fine-tuning-readiness-map-1.2.json",
    "real-tool-execution-map.json",
    "tuning-example-corpus.jsonl",
    "harness-policy-blueprint.json",
    "test1.2-handoff.json",
)

REQUIRED_TUNING_OUTPUTS = (
    "test1.2-tuning-plan.json",
    "candidate-harness-registry.json",
    "test1.2-tuning-observations.jsonl",
    "successive-halving-ledger.json",
    "compiled-harness-policy.json",
    "compiled-harness-validation.json",
    "do-not-use-registry.json",
    "fine-tuning-training-corpus.jsonl",
    "fine-tuning-qualification.json",
    "model-harness-card.json",
    "test1.2-final-acceptance.json",
    "integration-capability-contract.json",
    "inverted-model-integration-package.json",
    "test1.2-terminal-handoff.json",
)

DEFAULT_TUNING_CONFIG = {
    "expected_calls": 4200,
    "safety_call_cap": 10000,
    "screen_candidates": 24,
    "screen_cases": 16,
    "halving_cases": [24, 48, 96],
    "final_candidates": 4,
    "final_repeats": 2,
    "minimum_validation_families": 40,
    "max_capability_regression_rate": 0.05,
    "minimum_positive_value": 0.0,
    "minimum_acceptance_pass_rate": 0.67,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_collection(results_root: Path, run_id: str) -> dict[str, Any]:
    run_dir = results_root / run_id
    if not run_dir.is_dir():
        raise ValueError(f"collection run does not exist: {run_id}")
    problems = EvidenceStore(results_root, run_id).verify_manifest_paths(REQUIRED_COLLECTION_FILES)
    if problems:
        raise ValueError(f"collection consumed-artifact verification failed: {problems}")
    coverage = _read_json(run_dir / "control-grammar-coverage.json")
    if not coverage.get("all_declared_candidates_tested"):
        raise ValueError("collection did not exercise every declared control candidate")
    family_coverage = _read_json(run_dir / "capability-family-coverage.json")
    if int(family_coverage.get("required_family_count", 0)) != 40:
        raise ValueError("collection capability-family contract is not the required 40-family Test-2 set")
    if not family_coverage.get("all_families_manufacturing_ready"):
        raise ValueError(
            "collection is not manufacturing-ready for all capability families: "
            + ", ".join(family_coverage.get("not_manufacturing_ready") or [])
        )
    manufacturing_map = _read_json(run_dir / "capability-building-block-manufacturing-map.json")
    if int(manufacturing_map.get("family_count", 0)) != 40:
        raise ValueError("collection manufacturing map does not contain all 40 capability families")
    value_completeness = _read_json(run_dir / "family-value-completeness.json")
    if not value_completeness.get("all_families_critical_value_ready"):
        raise ValueError(
            "collection lacks critical improvement information for capability families: "
            + ", ".join(value_completeness.get("incomplete_families") or [])
        )
    improvement_dossiers = _read_json(run_dir / "capability-improvement-dossiers.json")
    negative_exploitation = _read_json(run_dir / "negative-effect-exploitation-map.json")
    frontier_shift = _read_json(run_dir / "frontier-shift-map.json")
    compute_elasticity = _read_json(run_dir / "compute-quality-elasticity-map.json")
    frontier_gap_maps = {
        "adaptive_search": _read_json(run_dir / "adaptive-search-map.json"),
        "metamorphic": _read_json(run_dir / "metamorphic-reliability-map.json"),
        "abstention": _read_json(run_dir / "abstention-calibration-map.json"),
        "active_memory": _read_json(run_dir / "active-memory-evolution-map.json"),
        "reflection_transfer": _read_json(run_dir / "reflection-transfer-map.json"),
        "tool_chaos": _read_json(run_dir / "tool-chaos-recovery-map.json"),
        "tool_scheduling": _read_json(run_dir / "tool-scheduling-map.json"),
        "index": _read_json(run_dir / "frontier-gap-value-map.json"),
    }
    second_gap_maps = {
        "authority": _read_json(run_dir / "authority-separation-map.json"),
        "reward_hacking": _read_json(run_dir / "reward-hacking-resistance-map.json"),
        "clarification": _read_json(run_dir / "clarification-value-map.json"),
        "governance_compaction": _read_json(run_dir / "governance-compaction-map.json"),
        "belief_state": _read_json(run_dir / "belief-state-map.json"),
        "semantic_transactions": _read_json(run_dir / "semantic-transaction-map.json"),
        "dynamic_replanning": _read_json(run_dir / "dynamic-replanning-map.json"),
        "index": _read_json(run_dir / "second-frontier-gap-value-map.json"),
    }
    if set((frontier_gap_maps["index"] or {}).get("surfaces") or []) != set(FRONTIER_GAP_SURFACES):
        raise ValueError("collection frontier-gap surface contract drifted")
    expected_labs = {
        "adaptive_search",
        "metamorphic",
        "abstention",
        "active_memory",
        "reflection_transfer",
        "tool_chaos",
        "tool_scheduling",
    }
    completed_labs = set((frontier_gap_maps["index"] or {}).get("completed_labs") or [])
    if completed_labs != expected_labs:
        raise ValueError(
            "collection did not complete every frontier-gap lab: "
            + ", ".join(sorted(expected_labs - completed_labs))
        )
    if any(
        int((frontier_gap_maps[name] or {}).get("schema_version", 0)) != 1
        for name in expected_labs
    ):
        raise ValueError("one or more frontier-gap maps are missing measured output")
    if set((second_gap_maps["index"] or {}).get("surfaces") or []) != set(SECOND_GAP_SURFACES):
        raise ValueError("collection second-gap surface contract drifted")
    expected_second_labs = {
        "authority",
        "reward_hacking",
        "clarification",
        "governance_compaction",
        "belief_state",
        "semantic_transactions",
        "dynamic_replanning",
    }
    completed_second_labs = set((second_gap_maps["index"] or {}).get("completed_labs") or [])
    if completed_second_labs != expected_second_labs:
        raise ValueError(
            "collection did not complete every second frontier-gap lab: "
            + ", ".join(sorted(expected_second_labs - completed_second_labs))
        )
    if any(
        int((second_gap_maps[name] or {}).get("schema_version", 0)) != 1
        or int((second_gap_maps[name] or {}).get("n", 0)) <= 0
        for name in expected_second_labs
    ):
        raise ValueError("one or more second-gap maps are missing measured output")
    zero_clock_model = _read_json(run_dir / "zero-clock-model-manufacturing-map.json")
    if zero_clock_model.get("zero_model_calls_added") is not True:
        raise ValueError("zero-clock model refinery must add zero model calls")
    if zero_clock_model.get("zero_active_test_seconds_added") is not True:
        raise ValueError("zero-clock model refinery must add zero active-test seconds")
    required_zero_clock_products = {
        "HARNESS_TO_WEIGHT_DISTILLATION",
        "WEIGHTED_HARD_NEGATIVE_PREFERENCES",
        "CAPABILITY_CURRICULUM",
        "ROUTER_ACTIVATION_SUPERVISION",
        "STABILITY_ANCHORS",
        "CROSS_FAMILY_TRANSFER_GRAPH",
        "PARETO_EFFICIENCY_TARGETS",
        "RELIABILITY_WEIGHTED_DISTILLATION",
        "LONG_HORIZON_BALANCED_TRAINING_MIX",
        "PREFERENCE_QUALITY_FILTER",
        "FAILURE_CREDIT_ASSIGNMENT",
        "CALIBRATION_VERIFY_SUPERVISION",
    }
    if set(zero_clock_model.get("products") or []) != required_zero_clock_products:
        raise ValueError("zero-clock model-building product contract drifted")
    zero_clock_assets = {
        "reliability_weighted_distillation": _read_jsonl(run_dir / "reliability-weighted-distillation-corpus.jsonl"),
        "long_horizon_training_mix": _read_json(run_dir / "long-horizon-training-mix.json"),
        "preference_quality": _read_jsonl(run_dir / "preference-quality-index.jsonl"),
        "failure_credit": _read_jsonl(run_dir / "failure-credit-assignment-corpus.jsonl"),
        "calibration_verify": _read_jsonl(run_dir / "calibration-verify-supervision-corpus.jsonl"),
    }
    registry = _read_json(run_dir / "full-control-candidate-registry.json")
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "registry": registry,
        "coverage": coverage,
        "family_coverage": family_coverage,
        "manufacturing_map": manufacturing_map,
        "value_completeness": value_completeness,
        "improvement_dossiers": improvement_dossiers,
        "negative_exploitation": negative_exploitation,
        "frontier_shift": frontier_shift,
        "compute_elasticity": compute_elasticity,
        "frontier_gap_maps": frontier_gap_maps,
        "second_gap_maps": second_gap_maps,
        "zero_clock_model": zero_clock_model,
        "zero_clock_assets": zero_clock_assets,
        "frontier": _read_json(run_dir / "cost-value-frontier-1.2.json"),
        "activation": _read_json(run_dir / "activation-boundary-map.json"),
        "negative": _read_json(run_dir / "negative-transfer-map-1.2.json"),
        "fine_tuning": _read_json(run_dir / "fine-tuning-readiness-map-1.2.json"),
        "real_tool": _read_json(run_dir / "real-tool-execution-map.json"),
        "corpus": _read_jsonl(run_dir / "tuning-example-corpus.jsonl"),
        "blueprint": _read_json(run_dir / "harness-policy-blueprint.json"),
        "handoff": _read_json(run_dir / "test1.2-handoff.json"),
    }


def build_tuning_plan(cases: list[dict[str, Any]], *, collection_run: str) -> dict[str, Any]:
    parts = partition_test12_cases(cases)
    return {
        "schema_version": 1,
        "campaign": "model-harness-compiler-test1.2-tuning",
        "collection_run": collection_run,
        "wall_clock_seconds": TUNING_HARD_SECONDS,
        "active_model_seconds": TUNING_ACTIVE_SECONDS,
        "phases": [{"name": n, "seconds": s} for n, s in TUNING_PHASES],
        "allowed_partitions": ["VALIDATION", "TEST2_BLIND", "TEST3_PROTECTED"],
        "prohibited_partitions": ["DISCOVERY"],
        "phase_partition_policy": {
            "validation_baseline": "VALIDATION",
            "candidate_harness_screen": "VALIDATION",
            "successive_halving": "VALIDATION",
            "routing_and_boundary_tuning": "VALIDATION",
            "residual_failure_replay": "VALIDATION",
            "final_validation_lock": "VALIDATION",
            "test2_blind_acceptance": "TEST2_BLIND",
            "test3_protected_acceptance": "TEST3_PROTECTED",
        },
        "partition_counts": {name: len(rows) for name, rows in parts.items()},
        "objective": "finish model onboarding inside the two-run ceiling: optimize on VALIDATION, freeze the winner, perform immutable blind/protected acceptance, and emit the final Inverted integration package",
        "required_capability_families": list(TEST2_CAPABILITY_FAMILIES),
        "required_capability_family_count": len(TEST2_CAPABILITY_FAMILIES),
        "observed_validation_families": sorted({_family(case) for case in parts["VALIDATION"]}),
        "missing_validation_families": sorted(
            set(TEST2_CAPABILITY_FAMILIES) - {_family(case) for case in parts["VALIDATION"]}
        ),
        "per_family_non_regression_required": True,
        "blind_acceptance_is_tuning_input": False,
        "protected_acceptance_is_tuning_input": False,
        "winner_locked_before_holdouts": True,
        "no_additional_characterization_test_required": True,
        "terminal_decisions": [
            "FULL_INVERTED_INTEGRATION",
            "CONSTRAINED_CAPABILITY_SCOPED_INTEGRATION",
            "REJECT_MODEL_ADDITION",
        ],
        "required_outputs": list(REQUIRED_TUNING_OUTPUTS),
        "total_two_run_hard_ceiling_seconds": TUNING_HARD_SECONDS + COLLECTION_HARD_SECONDS,
    }


def validate_tuning_plan(plan: dict[str, Any]) -> None:
    if int(plan["wall_clock_seconds"]) != TUNING_HARD_SECONDS:
        raise ValueError("tuning hard ceiling must be 6h15m")
    if sum(int(row["seconds"]) for row in plan["phases"]) != TUNING_ACTIVE_SECONDS:
        raise ValueError("tuning active phases must total six hours")
    if plan["allowed_partitions"] != ["VALIDATION", "TEST2_BLIND", "TEST3_PROTECTED"]:
        raise ValueError("tuning/acceptance must use VALIDATION then TEST2_BLIND then TEST3_PROTECTED")
    if set(plan["prohibited_partitions"]) != {"DISCOVERY"}:
        raise ValueError("tuning/acceptance may never reopen DISCOVERY")
    expected_phase_partitions = {
        "validation_baseline": "VALIDATION",
        "candidate_harness_screen": "VALIDATION",
        "successive_halving": "VALIDATION",
        "routing_and_boundary_tuning": "VALIDATION",
        "residual_failure_replay": "VALIDATION",
        "final_validation_lock": "VALIDATION",
        "test2_blind_acceptance": "TEST2_BLIND",
        "test3_protected_acceptance": "TEST3_PROTECTED",
    }
    if plan.get("phase_partition_policy") != expected_phase_partitions:
        raise ValueError("tuning/acceptance phase partition policy drifted")
    if int(plan["total_two_run_hard_ceiling_seconds"]) >= 14 * 60 * 60:
        raise ValueError("two-run model-to-harness compiler exceeds 14-hour target")
    if int(plan.get("required_capability_family_count", 0)) != 40:
        raise ValueError("tuning must validate all 40 capability families")
    if plan.get("missing_validation_families"):
        raise ValueError(
            "VALIDATION is missing required capability families: "
            + ", ".join(plan["missing_validation_families"])
        )
    if plan.get("per_family_non_regression_required") is not True:
        raise ValueError("tuning must enforce per-family non-regression")
    if plan.get("winner_locked_before_holdouts") is not True:
        raise ValueError("winner must be frozen before blind/protected acceptance")
    if plan.get("blind_acceptance_is_tuning_input") is not False or plan.get("protected_acceptance_is_tuning_input") is not False:
        raise ValueError("blind/protected acceptance may never tune or select the policy")
    if plan.get("no_additional_characterization_test_required") is not True:
        raise ValueError("Test 1.2 must be terminal for model onboarding")


def _candidate_registry(collection: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    by_id = {
        str(row["id"]): copy.deepcopy(row)
        for row in (collection.get("registry") or {}).get("candidates", [])
        if isinstance(row, dict) and row.get("id")
    }
    ranked = []
    for row in (collection.get("frontier") or {}).get("ranked", []):
        ident = str(row.get("intervention_id") or "")
        if ident not in by_id:
            continue
        if row.get("classification") == "CAPABILITY_HARM":
            continue
        ranked.append((float(row.get("net_value", 0.0)), float(row.get("value_per_call",0.0)), ident))
    ranked.sort(reverse=True)
    result = []
    for _, __, ident in ranked[:limit]:
        item = by_id[ident]
        item["collection_rank_source"] = next(
            (row for row in (collection.get("frontier") or {}).get("ranked", []) if row.get("intervention_id") == ident),
            {},
        )
        result.append(item)
    return result


def _route_map(candidates: list[dict[str, Any]]) -> dict[str, str]:
    preferences = {
        "TOOL": {"TOOL_POLICY","VERIFICATION"},
        "STATE": {"STATE_TRACKING","MEMORY"},
        "EVIDENCE": {"PROMPT_CONTROL","CONTEXT_SELECTION_COMPRESSION"},
        "FORMAT": {"PROMPT_CONTROL","VERIFICATION"},
        "PLAN": {"PLANNING","DELEGATION"},
        "VERIFY": {"VERIFICATION","CRITIQUE","RETRY_RECOVERY"},
    }
    result = {"DIRECT": "DIRECT"}
    for route, cats in preferences.items():
        found = next((row for row in candidates if row.get("category") in cats), None)
        result[route] = str(found["id"]) if found else "DIRECT"
    return result


def _compile_frontier_gap_policy(collection: dict[str, Any]) -> dict[str, Any]:
    maps = collection.get("frontier_gap_maps") or {}
    adaptive = maps.get("adaptive_search") or {}
    metamorphic = maps.get("metamorphic") or {}
    abstention = maps.get("abstention") or {}
    memory = maps.get("active_memory") or {}
    reflection = maps.get("reflection_transfer") or {}
    chaos = maps.get("tool_chaos") or {}
    scheduling = maps.get("tool_scheduling") or {}

    return {
        "schema_version": 1,
        "adaptive_search": {
            "enabled": int(adaptive.get("rescues", 0)) > int(adaptive.get("regressions", 0)),
            "rescues": adaptive.get("rescues", 0),
            "regressions": adaptive.get("regressions", 0),
            "activation": "HARD_OR_FAILED_TASK_ONLY",
        },
        "metamorphic_robustness": {
            "regressed_cases": int(metamorphic.get("regressed", 0)),
            "family_count": int(metamorphic.get("family_count", 0)),
            "require_wrapper_regression_sentinels": int(metamorphic.get("regressed", 0)) > 0,
        },
        "abstention": {
            "paired_accuracy": float(abstention.get("paired_accuracy", 0.0)),
            "false_act": int(abstention.get("false_act", 0)),
            "false_abstain": int(abstention.get("false_abstain", 0)),
            "require_pre_action_guard": int(abstention.get("false_act", 0)) > 0,
            "require_evidence_gather_before_abstain": int(abstention.get("false_abstain", 0)) > 0,
        },
        "active_memory": {
            "enabled": float(memory.get("active_accuracy", 0.0)) >= float(memory.get("direct_accuracy", 0.0)),
            "direct_accuracy": float(memory.get("direct_accuracy", 0.0)),
            "active_accuracy": float(memory.get("active_accuracy", 0.0)),
            "activation": "EVOLVING_OR_SUPERSEDED_STATE",
        },
        "reflection_transfer": {
            "enabled": int(reflection.get("sibling_rescues", 0)) > int(reflection.get("negative_transfer", 0)),
            "sibling_rescues": int(reflection.get("sibling_rescues", 0)),
            "negative_transfer": int(reflection.get("negative_transfer", 0)),
            "activation": "MATCHED_FAILURE_PHENOTYPE_ONLY",
        },
        "tool_chaos": {
            "success_rate": float(chaos.get("success_rate", 0.0)),
            "implicit_failure_success_rate": float(chaos.get("implicit_failure_success_rate", 0.0)),
            "force_verify_successful_tool_results": float(chaos.get("implicit_failure_success_rate", 0.0)) < 1.0,
            "ban_identical_blind_retry": int(chaos.get("blind_identical_retries", 0)) > 0,
        },
        "tool_scheduling": {
            "valid_rate": float(scheduling.get("valid_rate", 0.0)),
            "optimal_rate": float(scheduling.get("optimal_rate", 0.0)),
            "mean_efficiency": float(scheduling.get("mean_efficiency", 0.0)),
            "parallel_scheduler_enabled": (
                float(scheduling.get("valid_rate", 0.0)) >= 0.8
                and float(scheduling.get("mean_efficiency", 0.0)) >= 0.8
            ),
            "fallback": "DEPENDENCY_ORDERED_SEQUENTIAL",
        },
    }


def _compile_second_gap_policy(collection: dict[str, Any]) -> dict[str, Any]:
    maps = collection.get("second_gap_maps") or {}
    authority = maps.get("authority") or {}
    reward = maps.get("reward_hacking") or {}
    clarification = maps.get("clarification") or {}
    compaction = maps.get("governance_compaction") or {}
    belief = maps.get("belief_state") or {}
    txn = maps.get("semantic_transactions") or {}
    replan = maps.get("dynamic_replanning") or {}

    return {
        "schema_version": 1,
        "authority_separation": {
            "accuracy": float(authority.get("accuracy", 0.0)),
            "force_metadata_authorization_gate": int(authority.get("unsafe_authority_accepts", 0)) > 0,
            "preserve_approved_changes": int(authority.get("approved_change_overblocks", 0)) > 0,
            "untrusted_text_never_grants_authority": True,
        },
        "reward_hacking": {
            "accuracy": float(reward.get("accuracy", 0.0)),
            "shortcut_exploits": int(reward.get("shortcut_exploits", 0)),
            "legitimate_optimization_overblocks": int(reward.get("legitimate_optimization_overblocks", 0)),
            "protect_evaluator_and_verification_path": int(reward.get("shortcut_exploits", 0)) > 0,
        },
        "clarification": {
            "accuracy": float(clarification.get("accuracy", 0.0)),
            "under_clarification": int(clarification.get("under_clarification", 0)),
            "over_clarification": int(clarification.get("over_clarification", 0)),
            "use_value_of_information_gate": (
                int(clarification.get("under_clarification", 0)) > 0
                or int(clarification.get("over_clarification", 0)) > 0
            ),
        },
        "governance_compaction": {
            "checkpoint_preservation_rate": float(compaction.get("checkpoint_preservation_rate", 0.0)),
            "resume_accuracy": float(compaction.get("resume_accuracy", 0.0)),
            "pin_governance_constraints": (
                float(compaction.get("checkpoint_preservation_rate", 0.0)) < 1.0
                or int(compaction.get("governance_decay_events", 0)) > 0
            ),
        },
        "belief_state": {
            "accuracy": float(belief.get("accuracy", 0.0)),
            "premature_commitments": int(belief.get("premature_commitments", 0)),
            "explicit_belief_state_required": int(belief.get("premature_commitments", 0)) > 0,
        },
        "semantic_transactions": {
            "accuracy": float(txn.get("accuracy", 0.0)),
            "unsafe_commits_or_duplicates": int(txn.get("unsafe_commits_or_duplicates", 0)),
            "stage_validate_commit_boundary": True,
            "idempotency_guard_required": int(txn.get("unsafe_commits_or_duplicates", 0)) > 0,
        },
        "dynamic_replanning": {
            "accuracy": float(replan.get("accuracy", 0.0)),
            "failed_replans": int(replan.get("failed_replans", 0)),
            "unnecessary_replans": int(replan.get("unnecessary_replans", 0)),
            "invalidate_plan_on_cost_or_availability_change": True,
        },
    }


def _policy_candidates(collection: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    base = _candidate_registry(collection, limit)
    policies = [{"policy_id":"DIRECT","mode":"direct","intervention_id":None}]
    for row in base:
        policies.append({
            "policy_id":"STATIC-"+str(row["id"]),
            "mode":"static",
            "intervention_id":str(row["id"]),
            "intervention":copy.deepcopy(row),
        })
    route_map = _route_map(base)
    policies.append({
        "policy_id":"ROUTER-COMPILED",
        "mode":"router",
        "route_map":route_map,
    })
    policies.append({
        "policy_id":"RISK-GATED-COMPILED",
        "mode":"risk_gate",
        "fallback_intervention_id": next(
            (str(row["id"]) for row in base if row.get("category") in {"VERIFICATION","CRITIQUE","RETRY_RECOVERY"}),
            None,
        ),
    })
    return policies


def _score_policy_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n":0,
            "mean_score":0.0,
            "pass_rate":0.0,
            "mean_control_score":0.0,
            "mean_delta":0.0,
            "regression_rate":1.0,
            "mean_calls":0.0,
            "net_value":-999.0,
        }
    scores=[float(row.get("score") or 0.0) for row in rows]
    control_scores=[float(row.get("control_score") or 0.0) for row in rows]
    deltas=[float(row.get("delta") or 0.0) for row in rows]
    regressions=sum(1 for value in deltas if value < 0)
    calls=[float(row.get("model_calls") or 0.0) for row in rows]
    mean_delta=mean(deltas)
    regression_rate=regressions/len(rows)
    mean_calls=mean(calls)
    return {
        "n":len(rows),
        "mean_score":mean(scores),
        "pass_rate":sum(1 for value in scores if value >= 1.0)/len(scores),
        "mean_control_score":mean(control_scores),
        "mean_delta":mean_delta,
        "median_delta":median(deltas),
        "wins":sum(1 for value in deltas if value>0),
        "losses":regressions,
        "regression_rate":regression_rate,
        "mean_calls":mean_calls,
        "net_value":mean_delta - 0.08*mean_calls - 2.0*regression_rate,
        "families":sorted({str(row.get("family_id")) for row in rows}),
    }


def _score_policies_by_family(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row.get("policy_id")), str(row.get("family_id")))
        ].append(row)
    result: dict[str, dict[str, Any]] = defaultdict(dict)
    for (policy_id, family_id), values in grouped.items():
        result[policy_id][family_id] = _score_policy_rows(values)
    return dict(result)


def _top_family_safe_policies(
    registry: list[dict[str, Any]],
    global_scores: dict[str, Any],
    family_scores: dict[str, dict[str, Any]],
    *,
    keep: int,
    max_family_regression_rate: float,
) -> list[dict[str, Any]]:
    required = set(TEST2_CAPABILITY_FAMILIES)

    def rank(policy: dict[str, Any]) -> tuple[Any, ...]:
        policy_id = str(policy["policy_id"])
        per_family = family_scores.get(policy_id) or {}
        missing = required - set(per_family)
        regressions = [
            payload
            for family, payload in per_family.items()
            if family in required
            and (
                float(payload.get("mean_delta", 0.0)) < 0.0
                or float(payload.get("regression_rate", 0.0))
                > max_family_regression_rate
            )
        ]
        safe = not missing and not regressions
        minimum_delta = min(
            (
                float(payload.get("mean_delta", 0.0))
                for family, payload in per_family.items()
                if family in required
            ),
            default=-999.0,
        )
        improved_families = sum(
            1
            for family, payload in per_family.items()
            if family in required and float(payload.get("mean_delta", 0.0)) > 0
        )
        global_net = float(
            (global_scores.get(policy_id) or {}).get("net_value", -999.0)
        )
        return (
            1 if safe else 0,
            minimum_delta,
            improved_families,
            global_net,
        )

    return sorted(registry, key=rank, reverse=True)[: max(1, keep)]


class TuningRun:
    def __init__(self, runner: Any, cases: list[dict[str, Any]], collection: dict[str, Any], *, clock: Callable[[],float]=time.monotonic, started: float|None=None):
        self.runner=runner
        self.cases=cases
        self.collection=collection
        self.clock=clock
        self.start=clock() if started is None else float(started)
        self.active_end=self.start+TUNING_ACTIVE_SECONDS
        self.parts=partition_test12_cases(cases)
        self.validation=self.parts["VALIDATION"]
        self.test2_blind=self.parts["TEST2_BLIND"]
        self.test3_protected=self.parts["TEST3_PROTECTED"]
        self.cfg={**DEFAULT_TUNING_CONFIG, **copy.deepcopy((runner.config.get("test12_tuning") or {}))}
        source=fresh_model_source(cases)
        source["baselines"]={}
        self.campaign=Test12Campaign(runner,cases,source,clock=clock,started_monotonic=self.start)
        self.campaign.allowed_partitions={"VALIDATION"}
        # Replace campaign bank with exact collection candidates so no mechanism
        # definition drifts between collection and tuning.
        collected=[
            copy.deepcopy(row)
            for row in (collection.get("registry") or {}).get("candidates",[])
            if isinstance(row,dict) and row.get("id")
        ]
        self.campaign.interventions=collected
        self.campaign.intervention_by_id={str(row["id"]):row for row in collected}
        self.rows=[]
        self.router_cache: dict[tuple[str, int], tuple[str, dict[str, Any] | None]] = {}
        self.treatment_cache: dict[tuple[str, int, str], dict[str, Any]] = {}
        self.reuse_counters = {
            "router_decisions_reused": 0,
            "treatment_trials_reused": 0,
        }

    def can_start(self, deadline: float) -> bool:
        return self.clock() < min(deadline,self.active_end)

    def baseline(self, case: dict[str,Any], deadline: float, seed: int=42) -> dict[str,Any]|None:
        return self.campaign.control(case,deadline,seed=seed,force=False)

    def _router_choice(self, case: dict[str,Any], deadline: float, seed: int) -> tuple[str,dict[str,Any]|None]:
        cache_key = (_fixture_id(case), int(seed))
        if cache_key in self.router_cache:
            self.reuse_counters["router_decisions_reused"] += 1
            choice, aux = self.router_cache[cache_key]
            return choice, copy.deepcopy(aux)
        aux=self.campaign._aux(
            case,deadline,stage="compiled-router",
            messages=[{"role":"user","content":str(case["prompt"])+"\n\nClassify with exactly one token: TOOL, STATE, EVIDENCE, FORMAT, PLAN, VERIFY, DIRECT."}],
            intervention={"id":"COMPILED-ROUTER","reasoning_effort":None},
            seed=seed,call_index=1,
        )
        if aux is None:
            return "DIRECT",None
        choice = self.campaign._router_choice(str(aux.get("text") or ""))
        self.router_cache[cache_key] = (choice, copy.deepcopy(aux))
        return choice,aux

    def _treatment_trial(
        self,
        selected: dict[str, Any],
        case: dict[str, Any],
        deadline: float,
        *,
        seed: int,
    ) -> dict[str, Any] | None:
        cache_key = (
            _fixture_id(case),
            int(seed),
            _intervention_fingerprint(selected),
        )
        if cache_key in self.treatment_cache:
            self.reuse_counters["treatment_trials_reused"] += 1
            return copy.deepcopy(self.treatment_cache[cache_key])
        trial = self.campaign.treatment(
            case,
            deadline,
            phase="test1.2_tuning",
            intervention=selected,
            seed=seed,
        )
        if trial is not None:
            self.treatment_cache[cache_key] = copy.deepcopy(trial)
        return trial

    def run_policy(self, policy: dict[str,Any], case: dict[str,Any], deadline: float, *, seed: int) -> dict[str,Any]|None:
        control=self.baseline(case,deadline,seed)
        if control is None:
            return None
        control_score=float(control.get("score") or 0.0)
        policy_id=str(policy["policy_id"])
        if policy["mode"]=="direct":
            row={
                "schema_version":1,"policy_id":policy_id,"fixture_id":_fixture_id(case),
                "family_id":_family(case),"seed":seed,"control_score":control_score,
                "score":control_score,"delta":0.0,"model_calls":0,"route":"DIRECT",
                "selected_intervention_id":None,
            }
        else:
            router_aux=None
            selected=None
            route=None
            if policy["mode"]=="static":
                selected=policy["intervention"]
            elif policy["mode"]=="router":
                route,router_aux=self._router_choice(case,deadline,seed)
                ident=(policy.get("route_map") or {}).get(route,"DIRECT")
                selected=self.campaign.intervention_by_id.get(str(ident)) if ident!="DIRECT" else None
            elif policy["mode"]=="risk_gate":
                route,router_aux=self._router_choice(case,deadline,seed)
                ident=policy.get("fallback_intervention_id") if route!="DIRECT" else None
                selected=self.campaign.intervention_by_id.get(str(ident)) if ident else None

            if selected is None:
                score=control_score
                calls=1 if router_aux else 0
                row={
                    "schema_version":1,"policy_id":policy_id,"fixture_id":_fixture_id(case),
                    "family_id":_family(case),"seed":seed,"control_score":control_score,
                    "score":score,"delta":0.0,"model_calls":calls,"route":route or "DIRECT",
                    "selected_intervention_id":None,
                }
            else:
                trial=self._treatment_trial(selected,case,deadline,seed=seed)
                if trial is None:
                    return None
                calls=int(trial.get("model_calls_per_application") or 0)+(1 if router_aux else 0)
                row={
                    "schema_version":1,"policy_id":policy_id,"fixture_id":_fixture_id(case),
                    "family_id":_family(case),"seed":seed,"control_score":control_score,
                    "score":float(trial.get("score") or 0.0),
                    "delta":float(trial.get("score") or 0.0)-control_score,
                    "model_calls":calls,"route":route,
                    "selected_intervention_id":selected.get("id"),
                    "trial":trial,
                }
        row["partition"] = self.campaign.partition_name(case)
        row["evidence_reuse_counters"] = copy.deepcopy(self.reuse_counters)
        self.rows.append(row)
        self.runner.store.append_jsonl("test1.2-tuning-observations.jsonl",row)
        return row


def _balanced_validation(run: TuningRun, n: int) -> list[dict[str,Any]]:
    return _balanced_cases(run.validation,min(n,len(run.validation)))


def _balanced_partition(cases: list[dict[str, Any]], n: int | None = None) -> list[dict[str, Any]]:
    limit = len(cases) if n is None else min(int(n), len(cases))
    return _balanced_cases(cases, limit)


def _evaluate(run: TuningRun, policies: list[dict[str,Any]], cases: list[dict[str,Any]], deadline: float, *, seeds: list[int]) -> dict[str,Any]:
    start=len(run.rows)
    for seed in seeds:
        for policy in policies:
            for case in cases:
                if not run.can_start(deadline):
                    break
                run.run_policy(policy,case,deadline,seed=seed)
    rows=run.rows[start:]
    grouped=defaultdict(list)
    for row in rows:
        grouped[str(row["policy_id"])].append(row)
    return {key:_score_policy_rows(values) for key,values in grouped.items()}


def _evaluate_acceptance(
    run: TuningRun,
    winner: dict[str, Any],
    cases: list[dict[str, Any]],
    deadline: float,
    *,
    primary_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Breadth first, then spend remaining holdout time on extra evidence."""
    start = len(run.rows)
    primary = _balanced_partition(
        cases,
        min(len(cases), max(40, 2 * len(TEST2_CAPABILITY_FAMILIES))),
    )
    _evaluate(run, [winner], primary, deadline, seeds=[primary_seed])

    if run.can_start(deadline):
        used = {_fixture_id(case) for case in primary}
        remaining = [case for case in cases if _fixture_id(case) not in used]
        if remaining:
            _evaluate(run, [winner], _balanced_partition(remaining), deadline, seeds=[primary_seed])

    if run.can_start(deadline):
        # If breadth is complete early, buy independent repeat evidence rather
        # than leave acceptance minutes idle.
        _evaluate(run, [winner], primary, deadline, seeds=[primary_seed + 100])

    rows = run.rows[start:]
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["policy_id"])].append(row)
    return {
        key: _score_policy_rows(values)
        for key, values in grouped.items()
    }, rows


def _top_policies(registry: list[dict[str,Any]], scores: dict[str,Any], keep: int) -> list[dict[str,Any]]:
    ranked=sorted(
        registry,
        key=lambda p: float((scores.get(str(p["policy_id"])) or {}).get("net_value",-999)),
        reverse=True,
    )
    target=max(1,keep)
    selected=ranked[:target]
    direct=next((p for p in registry if str(p.get("policy_id"))=="DIRECT"),None)
    if direct is not None and all(str(p.get("policy_id"))!="DIRECT" for p in selected):
        if target == 1:
            selected=[direct]
        else:
            selected=selected[: target-1] + [direct]
    return selected


def _fine_tuning_records(run: TuningRun, winner: dict[str,Any]) -> list[dict[str,Any]]:
    winner_id=str(winner["policy_id"])
    rows=[
        row for row in run.rows
        if row.get("policy_id")==winner_id and row.get("partition")=="VALIDATION"
    ]
    result=[]
    for row in rows:
        if float(row.get("control_score",0.0))>=1.0 or float(row.get("score",0.0))>=1.0:
            continue
        trial=row.get("trial") or {}
        result.append({
            "schema_version":1,
            "fixture_id":row.get("fixture_id"),
            "family_id":row.get("family_id"),
            "policy_id":winner_id,
            "selected_intervention_id":row.get("selected_intervention_id"),
            "failure_class":((trial.get("classification") or {}).get("result_class") if isinstance(trial,dict) else None),
            "task_text":trial.get("task_text") if isinstance(trial,dict) else None,
            "failed_response_text":trial.get("treatment_response_text") if isinstance(trial,dict) else None,
            "qualification_state":"RESIDUAL_AFTER_COMPILED_HARNESS",
        })
    return result


def _acceptance_pass(
    summary: dict[str, Any],
    max_regression_rate: float,
    minimum_pass_rate: float,
) -> bool:
    return (
        int(summary.get("n", 0)) > 0
        and float(summary.get("pass_rate", 0.0)) >= float(minimum_pass_rate)
        and float(summary.get("mean_delta", 0.0)) >= 0.0
        and float(summary.get("regression_rate", 1.0)) <= float(max_regression_rate)
    )


def _family_is_safe(
    payload: dict[str, Any],
    max_regression_rate: float,
    minimum_pass_rate: float,
) -> bool:
    return (
        int(payload.get("n", 0)) > 0
        and float(payload.get("pass_rate", 0.0)) >= float(minimum_pass_rate)
        and float(payload.get("mean_delta", 0.0)) >= 0.0
        and float(payload.get("regression_rate", 1.0)) <= float(max_regression_rate)
    )


def _policy_lock_hash(policy: dict[str, Any]) -> str:
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_test12_tuning(runner: Any, cases: list[dict[str,Any]], *, collection_run: str, clock: Callable[[],float]=time.monotonic, started_monotonic: float|None=None) -> list[dict[str,Any]]:
    assert runner.store is not None
    collection=load_collection(Path(runner.results_root),collection_run)
    plan=build_tuning_plan(cases,collection_run=collection_run)
    validate_tuning_plan(plan)
    runner.store.write_json("test1.2-tuning-plan.json",plan,producer="test1.2-tuning",stage="preflight")

    run=TuningRun(runner,cases,collection,clock=clock,started=started_monotonic)
    policies=_policy_candidates(collection,int(run.cfg["screen_candidates"]))
    runner.store.write_json("candidate-harness-registry.json",{"schema_version":1,"policies":policies},producer="test1.2-tuning",stage="preflight")

    phase_start=run.start
    ledger=[]
    current=policies
    aggregate_scores={}
    final_confirmation_family_scores={}
    final_confirmation_rows=[]
    winner_locked: dict[str, Any] | None = None
    winner_lock_hash: str | None = None
    blind_rows: list[dict[str, Any]] = []
    protected_rows: list[dict[str, Any]] = []
    blind_scores: dict[str, Any] = {}
    protected_scores: dict[str, Any] = {}

    for phase_name,seconds in TUNING_PHASES:
        deadline=min(run.active_end,phase_start+seconds)
        partition = plan["phase_partition_policy"][phase_name]
        run.campaign.allowed_partitions={partition}

        if phase_name=="validation_baseline":
            cases0=_balanced_validation(run,min(len(run.validation),128))
            for case in cases0:
                if not run.can_start(deadline): break
                run.baseline(case,deadline,seed=42)
            scores={}
        elif phase_name=="candidate_harness_screen":
            scores=_evaluate(run,current,_balanced_validation(run,int(run.cfg["screen_cases"])),deadline,seeds=[42])
            current=_top_policies(current,scores,max(8,len(current)//2))
        elif phase_name=="successive_halving":
            scores={}
            for n in run.cfg["halving_cases"]:
                if not run.can_start(deadline): break
                stage=_evaluate(run,current,_balanced_validation(run,int(n)),deadline,seeds=[42])
                scores.update(stage)
                current=_top_policies(current,stage,max(int(run.cfg["final_candidates"]),len(current)//2))
        elif phase_name=="routing_and_boundary_tuning":
            scores=_evaluate(run,current,_balanced_validation(run,min(96,len(run.validation))),deadline,seeds=[42,43])
            current=_top_policies(current,scores,int(run.cfg["final_candidates"]))
        elif phase_name=="residual_failure_replay":
            scores=_evaluate(run,current,_balanced_validation(run,len(run.validation)),deadline,seeds=[43])
        elif phase_name=="final_validation_lock":
            final_start=len(run.rows)
            scores=_evaluate(
                run,
                current,
                _balanced_validation(run,len(run.validation)),
                deadline,
                seeds=[42,44][:int(run.cfg["final_repeats"])],
            )
            final_confirmation_rows=run.rows[final_start:]
            final_confirmation_family_scores=_score_policies_by_family(
                final_confirmation_rows
            )
            current=_top_family_safe_policies(
                current,
                scores,
                final_confirmation_family_scores,
                keep=1,
                max_family_regression_rate=float(
                    run.cfg["max_capability_regression_rate"]
                ),
            )
            winner_locked=copy.deepcopy(
                current[0] if current else {"policy_id":"DIRECT","mode":"direct"}
            )
            winner_lock_hash=_policy_lock_hash(winner_locked)
            current=[winner_locked]
        elif phase_name=="test2_blind_acceptance":
            if winner_locked is None:
                winner_locked=copy.deepcopy(
                    current[0] if current else {"policy_id":"DIRECT","mode":"direct"}
                )
                winner_lock_hash=_policy_lock_hash(winner_locked)
                current=[winner_locked]
            scores, blind_rows = _evaluate_acceptance(
                run,
                winner_locked,
                run.test2_blind,
                deadline,
                primary_seed=45,
            )
            blind_scores=copy.deepcopy(scores)
        elif phase_name=="test3_protected_acceptance":
            if winner_locked is None:
                winner_locked=copy.deepcopy(
                    current[0] if current else {"policy_id":"DIRECT","mode":"direct"}
                )
                winner_lock_hash=_policy_lock_hash(winner_locked)
                current=[winner_locked]
            scores, protected_rows = _evaluate_acceptance(
                run,
                winner_locked,
                run.test3_protected,
                deadline,
                primary_seed=46,
            )
            protected_scores=copy.deepcopy(scores)
        else:
            raise ValueError(f"unknown Test 1.2 tuning phase: {phase_name}")

        aggregate_scores.update(scores)
        ledger.append({
            "phase":phase_name,
            "partition":partition,
            "remaining_policy_ids":[p["policy_id"] for p in current],
            "winner_lock_hash":winner_lock_hash,
            "scores":scores,
        })
        phase_start=deadline
        if not run.can_start(run.active_end): break

    winner=winner_locked or (current[0] if current else {"policy_id":"DIRECT","mode":"direct"})
    winner_rows=[
        row for row in run.rows
        if row.get("policy_id")==winner["policy_id"]
        and row.get("partition")=="VALIDATION"
    ]
    winner_summary=_score_policy_rows(winner_rows)
    winner_family_validation=(
        final_confirmation_family_scores.get(str(winner["policy_id"])) or {}
    )
    if not winner_family_validation:
        winner_family_validation=(
            _score_policies_by_family(winner_rows).get(str(winner["policy_id"])) or {}
        )
    blind_summary=_score_policy_rows([
        row for row in blind_rows if row.get("policy_id")==winner["policy_id"]
    ])
    protected_summary=_score_policy_rows([
        row for row in protected_rows if row.get("policy_id")==winner["policy_id"]
    ])
    acceptance_rows=[
        row for row in [*blind_rows,*protected_rows]
        if row.get("policy_id")==winner["policy_id"]
    ]
    acceptance_family_validation=(
        _score_policies_by_family(acceptance_rows).get(str(winner["policy_id"])) or {}
    )
    missing_winner_families=sorted(
        set(TEST2_CAPABILITY_FAMILIES) - set(winner_family_validation)
    )
    regressing_winner_families=sorted(
        family
        for family, payload in winner_family_validation.items()
        if family in TEST2_CAPABILITY_FAMILIES
        and (
            float(payload.get("mean_delta",0.0)) < 0.0
            or float(payload.get("regression_rate",0.0))
            > float(run.cfg["max_capability_regression_rate"])
        )
    )
    family_safe=not missing_winner_families and not regressing_winner_families
    max_regression=float(run.cfg["max_capability_regression_rate"])
    minimum_pass_rate=float(run.cfg["minimum_acceptance_pass_rate"])
    blind_pass=_acceptance_pass(blind_summary,max_regression,minimum_pass_rate)
    protected_pass=_acceptance_pass(protected_summary,max_regression,minimum_pass_rate)

    certified_families=[]
    for family in TEST2_CAPABILITY_FAMILIES:
        validation_payload=winner_family_validation.get(family)
        if not validation_payload or not _family_is_safe(validation_payload,max_regression,minimum_pass_rate):
            continue
        holdout_payload=acceptance_family_validation.get(family)
        if holdout_payload and not _family_is_safe(holdout_payload,max_regression,minimum_pass_rate):
            continue
        certified_families.append(family)
    blocked_families=sorted(set(TEST2_CAPABILITY_FAMILIES)-set(certified_families))

    if family_safe and blind_pass and protected_pass:
        terminal_decision="FULL_INVERTED_INTEGRATION"
    elif certified_families and int(protected_summary.get("n",0))>0:
        terminal_decision="CONSTRAINED_CAPABILITY_SCOPED_INTEGRATION"
    else:
        terminal_decision="REJECT_MODEL_ADDITION"

    route_map=winner.get("route_map") or {}
    do_not_use=[
        key for key,value in ((collection.get("negative") or {}).get("effects") or {}).items()
        if int(value.get("capability_regressions",0))>0 or value.get("classification")=="CAPABILITY_HARM"
    ]
    tool_effects=(collection.get("real_tool") or {}).get("effects") or {}
    tool_execution_policy=None
    if tool_effects:
        tool_execution_policy=max(
            tool_effects,
            key=lambda key: (
                float((tool_effects.get(key) or {}).get("success_rate",0.0)),
                -float((tool_effects.get(key) or {}).get("mean_model_calls",999.0)),
            ),
        )
    frontier_gap_policy=_compile_frontier_gap_policy(collection)
    second_gap_policy=_compile_second_gap_policy(collection)
    compiled={
        "schema_version":1,
        "model":getattr(runner,"model",None),
        "collection_run":collection_run,
        "winner_policy":winner,
        "winner_validation":winner_summary,
        "winner_family_validation":winner_family_validation,
        "all_40_families_non_regressing":family_safe,
        "missing_validation_families":missing_winner_families,
        "regressing_validation_families":regressing_winner_families,
        "route_map":route_map,
        "do_not_use":sorted(do_not_use),
        "negative_effect_exploitation":copy.deepcopy(
            collection.get("negative_exploitation") or {}
        ),
        "capability_advancement_paths":{
            family: copy.deepcopy(payload.get("advancement_paths") or [])
            for family, payload in (
                (collection.get("improvement_dossiers") or {}).get("families") or {}
            ).items()
        },
        "tool_execution_policy":tool_execution_policy,
        "frontier_gap_policy":frontier_gap_policy,
        "second_gap_policy":second_gap_policy,
        "zero_clock_model_manufacturing":copy.deepcopy(
            collection.get("zero_clock_model") or {}
        ),
        "direct_default_when_unmatched":True,
        "oracle_routing_prohibited":True,
        "hard_ceiling_total_seconds":TUNING_HARD_SECONDS+COLLECTION_HARD_SECONDS,
        "winner_locked_before_holdouts":True,
        "winner_lock_sha256":winner_lock_hash,
        "blind_acceptance_is_tuning_input":False,
        "protected_acceptance_is_tuning_input":False,
        "terminal_decision":terminal_decision,
        "certified_capability_families":certified_families,
        "blocked_capability_families":blocked_families,
    }
    runner.store.write_json("successive-halving-ledger.json",{"schema_version":1,"stages":ledger},producer="test1.2-tuning",stage="report")
    runner.store.write_json("compiled-harness-policy.json",compiled,producer="test1.2-tuning",stage="report")
    runner.store.write_json("compiled-harness-validation.json",{
        "schema_version":1,
        "winner":winner_summary,
        "winner_by_family":winner_family_validation,
        "all_40_families_non_regressing":family_safe,
        "missing_families":missing_winner_families,
        "regressing_families":regressing_winner_families,
        "all_policy_scores":aggregate_scores,
        "winner_lock_sha256":winner_lock_hash,
        "test2_blind_acceptance":blind_summary,
        "test3_protected_acceptance":protected_summary,
        "acceptance_family_validation":acceptance_family_validation,
        "terminal_decision":terminal_decision,
    },producer="test1.2-tuning",stage="report")
    runner.store.write_json("do-not-use-registry.json",{"schema_version":1,"keys":sorted(do_not_use)},producer="test1.2-tuning",stage="report")

    ft=_fine_tuning_records(run,winner)
    for row in ft:
        runner.store.append_jsonl("fine-tuning-training-corpus.jsonl",row)
    by_family=defaultdict(int)
    for row in ft: by_family[str(row.get("family_id"))]+=1
    qualification={
        "schema_version":1,
        "residual_examples":len(ft),
        "recurrent_residual_families":{k:v for k,v in by_family.items() if v>=3},
        "weight_tuning_recommended":bool(any(v>=3 for v in by_family.values())),
        "onboarding_dependency":False,
        "rule":"weight tuning is an optional future model-development path for recurrent residual failures; it is not required to complete this model's Inverted onboarding",
    }
    runner.store.write_json("fine-tuning-qualification.json",qualification,producer="test1.2-tuning",stage="report")

    final_acceptance={
        "schema_version":1,
        "onboarding_complete":True,
        "winner_policy_id":winner.get("policy_id"),
        "winner_lock_sha256":winner_lock_hash,
        "winner_locked_before_holdouts":True,
        "holdouts_used_for_tuning_or_selection":False,
        "validation_family_safe":family_safe,
        "minimum_acceptance_pass_rate":minimum_pass_rate,
        "maximum_capability_regression_rate":max_regression,
        "test2_blind":{
            "summary":blind_summary,
            "passed":blind_pass,
            "policy_scores":blind_scores,
        },
        "test3_protected":{
            "summary":protected_summary,
            "passed":protected_pass,
            "policy_scores":protected_scores,
        },
        "terminal_decision":terminal_decision,
        "certified_capability_families":certified_families,
        "blocked_capability_families":blocked_families,
        "no_additional_characterization_test_required":True,
        "rerun_only_if":[
            "run_invalid_or_corrupted",
            "runtime_or_evidence_failure_prevented_terminal_acceptance",
            "model_weights_or_runtime_behavior_changed_materially",
        ],
    }
    runner.store.write_json(
        "test1.2-final-acceptance.json",
        final_acceptance,
        producer="test1.2-tuning",
        stage="terminal-acceptance",
    )

    capability_contract={
        "schema_version":1,
        "model":getattr(runner,"model",None),
        "terminal_decision":terminal_decision,
        "mode":(
            "ALL_CERTIFIED_CAPABILITIES"
            if terminal_decision=="FULL_INVERTED_INTEGRATION"
            else "CAPABILITY_ALLOWLIST"
            if terminal_decision=="CONSTRAINED_CAPABILITY_SCOPED_INTEGRATION"
            else "DISABLED"
        ),
        "allowed_capability_families":(
            list(TEST2_CAPABILITY_FAMILIES)
            if terminal_decision=="FULL_INVERTED_INTEGRATION"
            else certified_families
            if terminal_decision=="CONSTRAINED_CAPABILITY_SCOPED_INTEGRATION"
            else []
        ),
        "blocked_capability_families":(
            []
            if terminal_decision=="FULL_INVERTED_INTEGRATION"
            else blocked_families
            if terminal_decision=="CONSTRAINED_CAPABILITY_SCOPED_INTEGRATION"
            else list(TEST2_CAPABILITY_FAMILIES)
        ),
        "unmatched_task_policy":"DIRECT_OR_STRONGER_MODEL_FALLBACK",
        "scope_enforcement_required":terminal_decision!="FULL_INVERTED_INTEGRATION",
        "winner_lock_sha256":winner_lock_hash,
        "acceptance_artifact":"test1.2-final-acceptance.json",
    }
    runner.store.write_json(
        "integration-capability-contract.json",
        capability_contract,
        producer="test1.2-tuning",
        stage="terminal-acceptance",
    )

    integration_package={
        "schema_version":1,
        "package_type":"INVERTED_MODEL_ONBOARDING_TERMINAL_PACKAGE",
        "model":getattr(runner,"model",None),
        "onboarding_complete":True,
        "terminal_decision":terminal_decision,
        "collection_run":collection_run,
        "winner_policy":winner,
        "winner_lock_sha256":winner_lock_hash,
        "compiled_harness_policy":"compiled-harness-policy.json",
        "capability_contract":"integration-capability-contract.json",
        "final_acceptance":"test1.2-final-acceptance.json",
        "do_not_use_registry":"do-not-use-registry.json",
        "route_map":route_map,
        "tool_execution_policy":tool_execution_policy,
        "frontier_gap_policy":frontier_gap_policy,
        "second_gap_policy":second_gap_policy,
        "evidence_reuse":copy.deepcopy(run.reuse_counters),
        "no_additional_characterization_test_required":True,
        "next_action":(
            "INSTALL_MODEL_AND_COMPILED_POLICY_IN_INVERTED"
            if terminal_decision=="FULL_INVERTED_INTEGRATION"
            else "INSTALL_MODEL_WITH_CAPABILITY_ALLOWLIST_AND_FALLBACKS"
            if terminal_decision=="CONSTRAINED_CAPABILITY_SCOPED_INTEGRATION"
            else "DO_NOT_ADD_MODEL_TO_INVERTED"
        ),
        "optional_future_weight_improvement":{
            "is_onboarding_dependency":False,
            "qualification":"fine-tuning-qualification.json",
            "training_corpus":"fine-tuning-training-corpus.jsonl",
            "note":"changing model weights creates a new model version and therefore a new onboarding event",
        },
        "total_two_run_hard_ceiling_seconds":TUNING_HARD_SECONDS+COLLECTION_HARD_SECONDS,
    }
    runner.store.write_json(
        "inverted-model-integration-package.json",
        integration_package,
        producer="test1.2-tuning",
        stage="terminal-acceptance",
    )

    terminal_handoff={
        "schema_version":1,
        "state":"TEST1.2_MODEL_ONBOARDING_COMPLETE",
        "terminal_decision":terminal_decision,
        "integration_package":"inverted-model-integration-package.json",
        "no_test2_followup_required":True,
        "no_test3_followup_required":True,
        "no_additional_characterization_test_required":True,
        "next_action":integration_package["next_action"],
        "rerun_only_if":final_acceptance["rerun_only_if"],
    }
    runner.store.write_json(
        "test1.2-terminal-handoff.json",
        terminal_handoff,
        producer="test1.2-tuning",
        stage="terminal-acceptance",
    )

    runner.store.write_json("model-harness-card.json",{
        "schema_version":1,
        "model":getattr(runner,"model",None),
        "collection_run":collection_run,
        "compiled_policy":"compiled-harness-policy.json",
        "validated_policy_summary":winner_summary,
        "validated_family_summaries":winner_family_validation,
        "test2_blind_acceptance":blind_summary,
        "test3_protected_acceptance":protected_summary,
        "frontier_gap_policy":frontier_gap_policy,
        "second_gap_policy":second_gap_policy,
        "zero_clock_model_manufacturing":copy.deepcopy(
            collection.get("zero_clock_model") or {}
        ),
        "all_40_families_non_regressing":family_safe,
        "release_status":terminal_decision,
        "certified_capability_families":certified_families,
        "blocked_capability_families":blocked_families,
        "fine_tuning_qualification":qualification,
        "zero_clock_training_assets":{
            "product_count":len(collection.get("zero_clock_model",{}).get("products") or []),
            "manifest":"zero-clock-model-manufacturing-map.json",
            "recommended_primary_assets":[
                "reliability-weighted-distillation-corpus.jsonl",
                "preference-quality-index.jsonl",
                "long-horizon-training-mix.json",
                "stability-anchor-corpus.jsonl",
                "failure-credit-assignment-corpus.jsonl",
                "calibration-verify-supervision-corpus.jsonl",
            ],
        },
        "blind_partitions_touched":True,
        "holdouts_used_for_tuning_or_selection":False,
        "onboarding_complete":True,
        "no_additional_characterization_test_required":True,
        "integration_package":"inverted-model-integration-package.json",
        "total_two_run_hard_ceiling_hours":(TUNING_HARD_SECONDS+COLLECTION_HARD_SECONDS)/3600.0,
    },producer="test1.2-tuning",stage="report")
    return run.rows
