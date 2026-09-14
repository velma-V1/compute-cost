"""Test 1.2 validation-only tuning/compile run.

Consumes a completed Test 1.2 collection run and compiles a frozen provisional
model-specific harness in <= 6h15m. Only VALIDATION may be exposed here.
TEST2_BLIND is owned by Test 2 proof and TEST3_PROTECTED is owned by final
release acceptance. This stage may rank and lock a candidate, but may not ship,
certify, or consume future holdouts.
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
    DEFAULT_TEST12_CONFIG,
    FAMILY_CONTROL_SURFACES,
    FRONTIER_GAP_SURFACES,
    SECOND_GAP_SURFACES,
    TEST2_CAPABILITY_FAMILIES,
    Test12Campaign,
    _cost_value_frontier,
    _group_summary,
    _intervention_fingerprint,
    _rank_mechanisms,
    mechanism_summary,
    build_intervention_bank,
    fresh_model_source,
    load_test12_recovery,
    _read_recovery_checkpoint,
    partition_test12_cases,
)

from .test12_value import (
    build_control_response_tensor,
    build_family_value_dossiers,
    build_value_completeness,
    build_frontier_shift_map,
    build_compute_quality_elasticity,
    build_negative_effect_exploitation,
)

from .test12_model_manufacturing import build_zero_clock_model_manufacturing
TUNING_HARD_SECONDS = (6 * 60 * 60) + (15 * 60)
TUNING_ACTIVE_SECONDS = 6 * 60 * 60

TUNING_PHASES = (
    ("validation_baseline", 30 * 60),
    ("candidate_harness_screen", 75 * 60),
    ("successive_halving", 90 * 60),
    ("routing_and_boundary_tuning", 70 * 60),
    ("residual_failure_replay", 55 * 60),
    ("final_validation_lock", 40 * 60),
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
    "test1.2-tuning-recovery-checkpoint.json",
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


def _tuning_row_hash(row: dict[str, Any]) -> str:
    stable = {key: value for key, value in row.items() if key != "tuning_observation_sha256"}
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_tuning_recovery(run_dir: Path) -> dict[str, Any]:
    checkpoint_path = run_dir / "test1.2-tuning-recovery-checkpoint.json"
    checkpoint, checkpoint_issues = _read_recovery_checkpoint(checkpoint_path)
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = list(checkpoint_issues)
    path = run_dir / "test1.2-tuning-observations.jsonl"
    if path.is_file():
        for line_number, raw in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                issues.append({
                    "line": line_number,
                    "problem": "MALFORMED_TUNING_ATOMIC_RECORD",
                    "detail": str(exc),
                })
                continue
            if not isinstance(row, dict):
                issues.append({"line": line_number, "problem": "NON_OBJECT_TUNING_RECORD"})
                continue
            expected = row.get("tuning_observation_sha256")
            if expected and expected != _tuning_row_hash(row):
                issues.append({
                    "line": line_number,
                    "problem": "TUNING_RECORD_HASH_MISMATCH",
                    "policy_id": row.get("policy_id"),
                    "fixture_id": row.get("fixture_id"),
                    "seed": row.get("seed"),
                })
                continue
            rows.append(row)
    campaign_recovery = load_test12_recovery(run_dir)

    sanitized_campaign_rows, campaign_sanitization = _sanitize_collection_observations(
        list(campaign_recovery.get("rows") or [])
    )
    campaign_recovery["rows"] = sanitized_campaign_rows

    controls = {
        (str(row.get("fixture_id") or ""), int(row.get("seed") or 0)): row
        for row in sanitized_campaign_rows
        if row.get("intervention_id") == "CONTROL"
    }
    sanitized_policy_rows: list[dict[str, Any]] = []
    quarantined_policy_rows: list[dict[str, Any]] = []
    for raw in rows:
        row = copy.deepcopy(raw)
        if (
            not sanitized_campaign_rows
            and "delta_valid" not in row
            and "valid_for_capability" not in row
            and "control_valid_for_capability" not in row
        ):
            # Legacy/synthetic recovery records may predate campaign-level
            # evidence capture entirely. Preserve them rather than inventing
            # invalidity. Real runs with campaign evidence are reconstructed.
            sanitized_policy_rows.append(row)
            continue

        key = (str(row.get("fixture_id") or ""), int(row.get("seed") or 0))
        control = controls.get(key)
        control_valid = bool(control and control.get("valid_for_capability"))
        trial = row.get("trial") if isinstance(row.get("trial"), dict) else None
        if trial is not None:
            classification = trial.get("classification") or {}
            treatment_valid = (
                trial.get("valid_for_capability") is True
                or classification.get("valid_for_capability") is True
            )
        else:
            treatment_valid = control_valid

        row["control_valid_for_capability"] = control_valid
        row["valid_for_capability"] = bool(treatment_valid)
        row["delta_valid"] = bool(control_valid and treatment_valid)
        row["raw_delta_before_validity_filter"] = row.get("delta")
        if row["delta_valid"]:
            row["delta"] = float(row.get("score") or 0.0) - float(
                row.get("control_score") or 0.0
            )
            sanitized_policy_rows.append(row)
        else:
            row["delta"] = None
            row["recovery_quarantined_invalid"] = True
            quarantined_policy_rows.append(row)

    reserved_holdout_rows = [
        row for row in sanitized_policy_rows
        if row.get("partition") in {"TEST2_BLIND", "TEST3_PROTECTED"}
    ]
    legacy_holdout_exposure = {
        "TEST2_BLIND": any(
            row.get("partition") == "TEST2_BLIND"
            for row in reserved_holdout_rows
        ),
        "TEST3_PROTECTED": any(
            row.get("partition") == "TEST3_PROTECTED"
            for row in reserved_holdout_rows
        ),
    }
    checkpoint["legacy_holdout_exposure"] = copy.deepcopy(legacy_holdout_exposure)
    if reserved_holdout_rows:
        for row in reserved_holdout_rows:
            row["recovery_quarantined_holdout_exposure"] = True
        quarantined_policy_rows.extend(reserved_holdout_rows)
    rows = [
        row for row in sanitized_policy_rows
        if row.get("partition") in {None, "", "VALIDATION"}
    ]
    checkpoint["completed_phases"] = [
        phase for phase in (checkpoint.get("completed_phases") or [])
        if phase not in {"test2_blind_acceptance", "test3_protected_acceptance"}
    ]
    if checkpoint.get("current_phase") in {
        "test2_blind_acceptance",
        "test3_protected_acceptance",
    }:
        checkpoint["current_phase"] = None
        checkpoint["current_phase_elapsed_seconds"] = 0.0

    if quarantined_policy_rows and not checkpoint.get("winner_lock_sha256"):
        # Evidence remains on disk, but phase completion derived from contaminated
        # policy rows is reopened. Elapsed budget and physical-call counters are
        # intentionally not reset.
        checkpoint["completed_phases"] = []
        checkpoint["current_phase"] = None
        checkpoint["current_policy_ids"] = []

    atomic_checkpoint = campaign_recovery.get("checkpoint") or {}
    if float(atomic_checkpoint.get("active_seconds_used") or 0.0) > float(
        checkpoint.get("active_seconds_used") or 0.0
    ):
        checkpoint["active_seconds_used"] = float(
            atomic_checkpoint.get("active_seconds_used") or 0.0
        )
    checkpoint["physical_model_calls_used"] = max(
        int(checkpoint.get("physical_model_calls_used") or 0),
        int(atomic_checkpoint.get("physical_model_calls_used") or 0),
    )
    return {
        "checkpoint": checkpoint,
        "rows": rows,
        "issues": issues,
        "campaign_recovery": campaign_recovery,
        "campaign_sanitization": campaign_sanitization,
        "quarantined_policy_rows": quarantined_policy_rows,
        "quarantined_policy_observation_count": len(quarantined_policy_rows),
        "legacy_holdout_exposure": legacy_holdout_exposure,
    }




def _sanitize_collection_observations(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a non-mutating capability-valid overlay over raw Collection rows."""
    controls: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in rows:
        if raw.get("intervention_id") != "CONTROL":
            continue
        row = copy.deepcopy(raw)
        classification = row.get("classification") or {}
        valid = classification.get("valid_for_capability") is True
        row["valid_for_capability"] = bool(valid)
        row["delta_valid"] = bool(valid)
        controls[(str(row.get("fixture_id") or ""), int(row.get("seed") or 0))] = row

    sanitized: list[dict[str, Any]] = []
    invalid_controls = 0
    invalid_treatments = 0
    invalid_pairs = 0
    invalid_classes: dict[str, int] = defaultdict(int)
    for raw in rows:
        row = copy.deepcopy(raw)
        classification = row.get("classification") or {}
        own_valid = classification.get("valid_for_capability") is True
        row["valid_for_capability"] = bool(own_valid)
        if row.get("intervention_id") == "CONTROL":
            row["delta_valid"] = bool(own_valid)
            if not own_valid:
                invalid_controls += 1
                invalid_classes[str(classification.get("result_class") or "UNKNOWN")] += 1
            sanitized.append(row)
            continue

        control = controls.get(
            (str(row.get("fixture_id") or ""), int(row.get("seed") or 0))
        )
        control_valid = bool(control and control.get("valid_for_capability"))
        treatment_budget = int(
            row.get("generation_budget")
            or DEFAULT_TEST12_CONFIG["base_generation_budget"]
        )
        control_budget = int(
            (control or {}).get("generation_budget")
            or DEFAULT_TEST12_CONFIG["base_generation_budget"]
        )
        budget_comparison_valid = (
            row.get("intervention_category") == "GENERATION_BUDGET"
            or treatment_budget == control_budget
        )
        row["control_valid_for_capability"] = control_valid
        row["control_generation_budget"] = control_budget
        row["budget_comparison_valid"] = bool(budget_comparison_valid)
        row["delta_valid"] = bool(
            own_valid and control_valid and budget_comparison_valid
        )
        row["raw_delta_before_validity_filter"] = row.get("delta")
        if row["delta_valid"]:
            row["delta"] = float(row.get("score") or 0.0) - float(
                row.get("control_score") or 0.0
            )
        else:
            row["delta"] = None
            invalid_pairs += 1
            if not own_valid:
                invalid_treatments += 1
                invalid_classes[str(classification.get("result_class") or "UNKNOWN")] += 1
            if not control_valid:
                invalid_classes["INVALID_BASELINE_CONTROL"] += 1
        sanitized.append(row)

    report = {
        "schema_version": 1,
        "raw_observation_count": len(rows),
        "sanitized_observation_count": len(sanitized),
        "invalid_control_observations": invalid_controls,
        "invalid_treatment_observations": invalid_treatments,
        "invalid_capability_pairs": invalid_pairs,
        "invalid_classes": dict(sorted(invalid_classes.items())),
        "policy": "CAPABILITY_DELTA_REQUIRES_VALID_CONTROL_AND_VALID_TREATMENT",
    }
    return sanitized, report




def _derive_generation_budget_calibration(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Turn truncation-heavy Collection evidence into a safe baseline budget map.

    Only CONTROL and pure GENERATION_BUDGET rows are used so prompt/controller
    effects cannot masquerade as budget effects. Capability validity measures
    whether a final answer was produced; correctness is reported separately.
    """
    by_family_budget: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.get("partition") != "DISCOVERY":
            continue
        if (
            row.get("intervention_id") != "CONTROL"
            and row.get("intervention_category") != "GENERATION_BUDGET"
        ):
            continue
        budget = int(row.get("generation_budget") or 0)
        if budget <= 0:
            continue
        by_family_budget[str(row.get("family_id") or "")][budget].append(row)

    configured = sorted(
        {int(DEFAULT_TEST12_CONFIG["base_generation_budget"])}
        | {int(value) for value in DEFAULT_TEST12_CONFIG["generation_budgets"]}
    )
    maximum_budget = max(configured)
    families: dict[str, Any] = {}
    recommended: dict[str, int] = {}

    for family in TEST2_CAPABILITY_FAMILIES:
        budget_rows = by_family_budget.get(family) or {}
        levels: dict[str, Any] = {}
        qualifying: list[int] = []
        any_valid: list[int] = []
        any_pass: list[int] = []
        for budget in configured:
            values = list(budget_rows.get(budget) or [])
            valid = [
                row for row in values
                if row.get("valid_for_capability") is True
            ]
            passes = [
                row for row in valid
                if float(row.get("score") or 0.0) >= 1.0
            ]
            valid_rate = len(valid) / len(values) if values else None
            pass_rate = len(passes) / len(valid) if valid else None
            levels[str(budget)] = {
                "n": len(values),
                "valid_final_answers": len(valid),
                "valid_final_answer_rate": valid_rate,
                "passes_among_valid": len(passes),
                "pass_rate_among_valid": pass_rate,
            }
            if valid:
                any_valid.append(budget)
            if passes:
                any_pass.append(budget)
            if len(valid) >= 2 and valid_rate is not None and valid_rate >= 0.80:
                qualifying.append(budget)

        if qualifying:
            safe_budget = min(qualifying)
            basis = "AT_LEAST_2_VALID_AND_80_PERCENT_VALID"
        elif any_valid:
            # Sparse Collection evidence: use the largest observed budget that
            # actually produced a final answer rather than extrapolating a small
            # budget from one lucky sample.
            safe_budget = max(any_valid)
            basis = "SPARSE_EVIDENCE_MAX_OBSERVED_VALID_BUDGET"
        else:
            safe_budget = maximum_budget
            basis = "NO_VALID_FINAL_ANSWER_OBSERVED_USE_MAX_TESTED_BUDGET"

        recommended[family] = int(safe_budget)
        families[family] = {
            "levels": levels,
            "minimum_observed_valid_budget": min(any_valid) if any_valid else None,
            "minimum_observed_passing_budget": min(any_pass) if any_pass else None,
            "recommended_safe_baseline_budget": int(safe_budget),
            "recommendation_basis": basis,
        }

    return {
        "schema_version": 1,
        "purpose": "CAPABILITY_VALID_BASELINE_NOT_ACCURACY_INFLATION",
        "families": families,
        "recommended_safe_baseline_budget_by_family": recommended,
        "maximum_tested_budget": maximum_budget,
        "invalid_outputs_are_not_capability_failures": True,
    }

def _rebuild_collection_analytics(
    rows: list[dict[str, Any]],
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Recompute contaminated Collection summaries from raw evidence only."""
    candidates = [
        copy.deepcopy(row)
        for row in (registry.get("candidates") or [])
        if isinstance(row, dict)
    ]
    by_id = {
        str(row.get("id")): row
        for row in candidates
        if row.get("id")
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ident = str(row.get("intervention_id") or "")
        if ident and ident != "CONTROL":
            grouped[ident].append(row)

    ranked = []
    for ident, values in grouped.items():
        summary = mechanism_summary(values, DEFAULT_TEST12_CONFIG)
        candidate = by_id.get(ident) or {}
        ranked.append({
            "intervention_id": ident,
            "category": candidate.get("category"),
            **copy.deepcopy(summary),
        })
    ranked.sort(
        key=lambda row: (
            float(row.get("net_value", 0.0)),
            float(row.get("value_per_call", 0.0)),
        ),
        reverse=True,
    )
    frontier = {
        "schema_version": 1,
        "sanitized_from_raw_collection": True,
        "objective": "valid capability rescue minus valid capability regression and cost",
        "ranked": ranked,
        "pareto_candidates": [
            row for row in ranked
            if float(row.get("raw_value", 0.0)) > 0
            and row.get("classification") != "CAPABILITY_HARM"
        ][:25],
    }

    tensor = build_control_response_tensor(
        rows,
        candidates,
        TEST2_CAPABILITY_FAMILIES,
    )
    dossiers = build_family_value_dossiers(
        rows,
        candidates,
        TEST2_CAPABILITY_FAMILIES,
        FAMILY_CONTROL_SURFACES,
    )
    completeness = build_value_completeness(dossiers)
    frontier_shift = build_frontier_shift_map(
        rows,
        tensor,
        TEST2_CAPABILITY_FAMILIES,
    )
    compute_elasticity = build_compute_quality_elasticity(
        tensor,
        TEST2_CAPABILITY_FAMILIES,
    )
    negative_exploitation = build_negative_effect_exploitation(
        tensor,
        TEST2_CAPABILITY_FAMILIES,
    )
    return {
        "frontier": frontier,
        "tensor": tensor,
        "dossiers": dossiers,
        "value_completeness": completeness,
        "frontier_shift": frontier_shift,
        "compute_elasticity": compute_elasticity,
        "negative_exploitation": negative_exploitation,
    }

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
    collection_value_gap_families = sorted(
        str(value)
        for value in (value_completeness.get("incomplete_families") or [])
    )
    collection_value_gaps = {
        family: copy.deepcopy(
            ((value_completeness.get("families") or {}).get(family) or {})
        )
        for family in collection_value_gap_families
    }
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
    opportunity_path = run_dir / "test1.2-opportunity-discovery-map.json"
    if opportunity_path.is_file():
        opportunity_discovery = _read_json(opportunity_path)
        if opportunity_discovery.get("collection_role") != "OPPORTUNITY_DISCOVERY":
            raise ValueError("collection opportunity-discovery contract drifted")
        if opportunity_discovery.get("proof_owner") != "RUN2_TEST2":
            raise ValueError("collection incorrectly claims proof ownership")
    else:
        # Backward compatibility for completed pre-opportunity-first Test 1.2
        # collections. Their evidence remains valid; Run 2 still owns proof.
        opportunity_discovery = {
            "schema_version": 1,
            "collection_role": "LEGACY_BROAD_COLLECTION",
            "proof_owner": "RUN2_TEST2",
            "legacy_collection_without_opportunity_map": True,
        }
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

    # The original Collection reports are immutable evidence artifacts, but
    # pre-validity-fix runs may have derived capability deltas from invalid
    # generations. Rebuild an analytical overlay directly from raw evidence.
    raw_collection_rows = _read_jsonl(run_dir / "test1.2-observations.jsonl")
    sanitized_collection_rows, collection_sanitization = (
        _sanitize_collection_observations(raw_collection_rows)
    )
    rebuilt = _rebuild_collection_analytics(
        sanitized_collection_rows,
        registry,
    )
    value_completeness = rebuilt["value_completeness"]
    collection_value_gap_families = sorted(
        str(value)
        for value in (value_completeness.get("incomplete_families") or [])
    )
    collection_value_gaps = {
        family: copy.deepcopy(
            ((value_completeness.get("families") or {}).get(family) or {})
        )
        for family in collection_value_gap_families
    }
    improvement_dossiers = rebuilt["dossiers"]
    negative_exploitation = rebuilt["negative_exploitation"]
    frontier_shift = rebuilt["frontier_shift"]
    compute_elasticity = rebuilt["compute_elasticity"]
    sanitized_frontier = rebuilt["frontier"]
    budget_calibration = _derive_generation_budget_calibration(
        sanitized_collection_rows
    )
    sanitized_zero_clock_model = build_zero_clock_model_manufacturing(
        sanitized_collection_rows,
        TEST2_CAPABILITY_FAMILIES,
    )
    sanitized_zero_clock_assets = {
        "reliability_weighted_distillation": copy.deepcopy(
            sanitized_zero_clock_model.get("reliable_distillation") or []
        ),
        "long_horizon_training_mix": copy.deepcopy(
            sanitized_zero_clock_model.get("long_horizon_mix") or {}
        ),
        "preference_quality": copy.deepcopy(
            sanitized_zero_clock_model.get("preference_quality") or []
        ),
        "failure_credit": copy.deepcopy(
            sanitized_zero_clock_model.get("failure_credit") or []
        ),
        "calibration_verify": copy.deepcopy(
            sanitized_zero_clock_model.get("calibration") or []
        ),
    }

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "registry": registry,
        "collection_sanitization": collection_sanitization,
        "generation_budget_calibration": budget_calibration,
        "sanitized_collection_observations": sanitized_collection_rows,
        "coverage": coverage,
        "family_coverage": family_coverage,
        "manufacturing_map": manufacturing_map,
        "value_completeness": value_completeness,
        "collection_value_gap_families": collection_value_gap_families,
        "collection_value_gaps": collection_value_gaps,
        "improvement_dossiers": improvement_dossiers,
        "negative_exploitation": negative_exploitation,
        "frontier_shift": frontier_shift,
        "compute_elasticity": compute_elasticity,
        "frontier_gap_maps": frontier_gap_maps,
        "second_gap_maps": second_gap_maps,
        "zero_clock_model": sanitized_zero_clock_model,
        "legacy_zero_clock_model": zero_clock_model,
        "opportunity_discovery": opportunity_discovery,
        "zero_clock_assets": sanitized_zero_clock_assets,
        "legacy_zero_clock_assets": zero_clock_assets,
        "frontier": sanitized_frontier,
        "legacy_frontier": _read_json(run_dir / "cost-value-frontier-1.2.json"),
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
        "allowed_partitions": ["VALIDATION"],
        "prohibited_partitions": ["DISCOVERY", "TEST2_BLIND", "TEST3_PROTECTED"],
        "phase_partition_policy": {
            "validation_baseline": "VALIDATION",
            "candidate_harness_screen": "VALIDATION",
            "successive_halving": "VALIDATION",
            "routing_and_boundary_tuning": "VALIDATION",
            "residual_failure_replay": "VALIDATION",
            "final_validation_lock": "VALIDATION",
        },
        "partition_counts": {name: len(rows) for name, rows in parts.items()},
        "objective": "optimize on VALIDATION only, freeze an exact provisional winner, preserve all blind/protected holdouts, and hand the locked candidate to Test 2 for proof",
        "required_capability_families": list(TEST2_CAPABILITY_FAMILIES),
        "required_capability_family_count": len(TEST2_CAPABILITY_FAMILIES),
        "observed_validation_families": sorted({_family(case) for case in parts["VALIDATION"]}),
        "missing_validation_families": sorted(
            set(TEST2_CAPABILITY_FAMILIES) - {_family(case) for case in parts["VALIDATION"]}
        ),
        "per_family_non_regression_required": True,
        "blind_acceptance_is_tuning_input": False,
        "protected_acceptance_is_tuning_input": False,
        "test2_blind_exposed": False,
        "test3_protected_exposed": False,
        "winner_locked_before_test2": True,
        "test2_proof_required": True,
        "full_rerun_recovery_prohibited": True,
        "same_run_id_resume_required": True,
        "atomic_evidence_salvage_required": True,
        "winner_lock_must_survive_resume": True,
        "terminal_decisions": [
            "PROVISIONAL_READY_FOR_TEST2",
            "PROVISIONAL_CONSTRAINED_FOR_TEST2",
            "REJECT_BEFORE_TEST2",
        ],
        "required_outputs": list(REQUIRED_TUNING_OUTPUTS),
        "total_two_run_hard_ceiling_seconds": TUNING_HARD_SECONDS + COLLECTION_HARD_SECONDS,
    }


def validate_tuning_plan(plan: dict[str, Any]) -> None:
    if int(plan["wall_clock_seconds"]) != TUNING_HARD_SECONDS:
        raise ValueError("tuning hard ceiling must be 6h15m")
    if sum(int(row["seconds"]) for row in plan["phases"]) != TUNING_ACTIVE_SECONDS:
        raise ValueError("tuning active phases must total six hours")
    if plan["allowed_partitions"] != ["VALIDATION"]:
        raise ValueError("Test 1.2 tuning may expose VALIDATION only")
    if set(plan["prohibited_partitions"]) != {"DISCOVERY","TEST2_BLIND","TEST3_PROTECTED"}:
        raise ValueError("Test 1.2 tuning must preserve discovery and all holdouts")
    expected_phase_partitions = {
        "validation_baseline": "VALIDATION",
        "candidate_harness_screen": "VALIDATION",
        "successive_halving": "VALIDATION",
        "routing_and_boundary_tuning": "VALIDATION",
        "residual_failure_replay": "VALIDATION",
        "final_validation_lock": "VALIDATION",
    }
    if plan.get("phase_partition_policy") != expected_phase_partitions:
        raise ValueError("validation-only tuning phase partition policy drifted")
    if int(plan["total_two_run_hard_ceiling_seconds"]) >= 14 * 60 * 60:
        raise ValueError("two-run discovery+compiler exceeds 14-hour target")
    if int(plan.get("required_capability_family_count", 0)) != 40:
        raise ValueError("tuning must validate all 40 capability families")
    if plan.get("missing_validation_families"):
        raise ValueError(
            "VALIDATION is missing required capability families: "
            + ", ".join(plan["missing_validation_families"])
        )
    if plan.get("per_family_non_regression_required") is not True:
        raise ValueError("tuning must enforce per-family non-regression")
    if plan.get("test2_blind_exposed") is not False or plan.get("test3_protected_exposed") is not False:
        raise ValueError("Test 1.2 tuning may not consume blind/protected holdouts")
    if plan.get("test2_proof_required") is not True:
        raise ValueError("Test 2 proof must remain mandatory")
    if plan.get("full_rerun_recovery_prohibited") is not True:
        raise ValueError("Test 1.2 recovery may not require a full rerun")
    if plan.get("winner_lock_must_survive_resume") is not True:
        raise ValueError("winner lock must survive tuning recovery")

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
        discovery_rank = {
            "NEW_RESCUE_OPPORTUNITY": 3,
            "NEGATIVE_BOUNDARY_OPPORTUNITY": 1,
            "NO_OBSERVED_OPPORTUNITY": 0,
        }.get(str(row.get("discovery_status") or ""), 0)
        ranked.append((
            discovery_rank,
            int(row.get("rescues") or 0),
            float(row.get("net_value", 0.0)),
            float(row.get("value_per_call", 0.0)),
            ident,
        ))
    ranked.sort(reverse=True)
    result = []
    for _, __, ___, ____, ident in ranked[:limit]:
        item = by_id[ident]
        item["collection_rank_source"] = next(
            (row for row in (collection.get("frontier") or {}).get("ranked", []) if row.get("intervention_id") == ident),
            {},
        )
        item["verification_owner"] = "RUN2_TEST2"
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
    valid_rows=[
        row for row in rows
        if row.get("delta_valid") is True
        or (
            "delta_valid" not in row
            and row.get("valid_for_capability") is not False
            and row.get("control_valid_for_capability") is not False
        )
    ]
    invalid_count=len(rows)-len(valid_rows)
    if not valid_rows:
        return {
            "n":0,
            "raw_n":len(rows),
            "invalid_observations_excluded":invalid_count,
            "mean_score":0.0,
            "pass_rate":0.0,
            "mean_control_score":0.0,
            "mean_delta":0.0,
            "regression_rate":1.0,
            "mean_calls":0.0,
            "net_value":-999.0,
        }
    scores=[float(row.get("score") or 0.0) for row in valid_rows]
    control_scores=[float(row.get("control_score") or 0.0) for row in valid_rows]
    deltas=[float(row["delta"]) for row in valid_rows if row.get("delta") is not None]
    regressions=sum(1 for value in deltas if value < 0)
    calls=[float(row.get("model_calls") or 0.0) for row in valid_rows]
    mean_delta=mean(deltas)
    regression_rate=regressions/len(valid_rows)
    mean_calls=mean(calls)
    return {
        "n":len(valid_rows),
        "raw_n":len(rows),
        "invalid_observations_excluded":invalid_count,
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
        "families":sorted({str(row.get("family_id")) for row in valid_rows}),
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
    def __init__(
        self,
        runner: Any,
        cases: list[dict[str, Any]],
        collection: dict[str, Any],
        *,
        clock: Callable[[],float]=time.monotonic,
        started: float|None=None,
        resume_state: dict[str, Any] | None = None,
    ):
        self.runner=runner
        self.cases=cases
        self.collection=collection
        self.clock=clock
        self.resume_state=copy.deepcopy(resume_state or {})
        checkpoint=self.resume_state.get("checkpoint") or {}
        elapsed_before_resume=max(0.0,float(checkpoint.get("active_seconds_used") or 0.0))
        self.start=(clock()-elapsed_before_resume) if self.resume_state else (clock() if started is None else float(started))
        self.active_end=self.start+TUNING_ACTIVE_SECONDS
        self.parts=partition_test12_cases(cases)
        self.validation=self.parts["VALIDATION"]
        self.test2_blind=self.parts["TEST2_BLIND"]
        self.test3_protected=self.parts["TEST3_PROTECTED"]
        self.cfg={**DEFAULT_TUNING_CONFIG, **copy.deepcopy((runner.config.get("test12_tuning") or {}))}
        source=fresh_model_source(cases)
        source["baselines"]={}
        self.campaign=Test12Campaign(
            runner,
            cases,
            source,
            clock=clock,
            started_monotonic=self.start,
            resume_state=self.resume_state.get("campaign_recovery"),
        )
        self.campaign.allowed_partitions={"VALIDATION"}
        budget_map = (
            (collection.get("generation_budget_calibration") or {})
            .get("recommended_safe_baseline_budget_by_family")
            or {}
        )
        self.campaign.baseline_generation_budget_by_family = {
            str(family): int(budget)
            for family, budget in budget_map.items()
        }
        # Recovered baselines were measured under the legacy operating
        # budget. Keep only controls measured at the newly calibrated family
        # budget; mismatched controls remain in raw evidence but are not reusable.
        for cache in (self.campaign.controls, self.campaign.invalid_controls):
            for key, recovered_control in list(cache.items()):
                fixture_id, _seed = key
                case = self.campaign.case_by_id.get(fixture_id)
                if case is None:
                    continue
                recommended = int(
                    self.campaign.baseline_generation_budget_by_family.get(
                        _family(case),
                        self.campaign.cfg["base_generation_budget"],
                    )
                )
                observed = int(
                    recovered_control.get("generation_budget")
                    or self.campaign.cfg["base_generation_budget"]
                )
                if observed != recommended:
                    cache.pop(key, None)

        # Recovered non-budget treatments must match the calibrated family
        # operating budget before they can suppress or satisfy a future trial.
        for recovered in list(self.campaign.rows):
            intervention_id = str(recovered.get("intervention_id") or "")
            if intervention_id in {"", "CONTROL"}:
                continue
            intervention = self.campaign.intervention_by_id.get(intervention_id)
            if intervention is None or intervention.get("category") == "GENERATION_BUDGET":
                continue
            fixture_id = str(recovered.get("fixture_id") or "")
            case = self.campaign.case_by_id.get(fixture_id)
            if case is None:
                continue
            recommended = int(
                self.campaign.baseline_generation_budget_by_family.get(
                    _family(case),
                    self.campaign.cfg["base_generation_budget"],
                )
            )
            observed = int(
                recovered.get("generation_budget")
                or self.campaign.cfg["base_generation_budget"]
            )
            if observed != recommended:
                seed = int(recovered.get("seed") or 0)
                self.campaign.completed_treatment_ids.discard(
                    (fixture_id, seed, intervention_id)
                )
                self.campaign.completed_treatment_signatures.discard(
                    self.campaign._trial_signature(case, intervention, seed)
                )

        # Replace campaign bank with exact collection candidates so no mechanism
        # definition drifts between collection and tuning.
        collected=[
            copy.deepcopy(row)
            for row in (collection.get("registry") or {}).get("candidates",[])
            if isinstance(row,dict) and row.get("id")
        ]
        self.campaign.interventions=collected
        self.campaign.intervention_by_id={str(row["id"]):row for row in collected}
        self.rows=copy.deepcopy(self.resume_state.get("rows") or [])
        self.router_cache: dict[tuple[str, int], tuple[str, dict[str, Any] | None]] = {}
        self.treatment_cache: dict[tuple[str, int, str], dict[str, Any]] = {}
        self.policy_observation_index: dict[tuple[str, str, int, str], dict[str, Any]] = {}
        self.reuse_counters = {
            "router_decisions_reused": int((checkpoint.get("reuse_counters") or {}).get("router_decisions_reused",0)),
            "treatment_trials_reused": int((checkpoint.get("reuse_counters") or {}).get("treatment_trials_reused",0)),
            "policy_observations_reused": int((checkpoint.get("reuse_counters") or {}).get("policy_observations_reused",0)),
        }
        self.completed_phases=set(str(x) for x in (checkpoint.get("completed_phases") or []))
        self.current_phase: str|None=None
        self.current_phase_started: float|None=None
        self.current_phase_elapsed_base=0.0
        self.current_policy_ids=[str(x) for x in (checkpoint.get("current_policy_ids") or [])]
        self.winner_locked=copy.deepcopy(checkpoint.get("winner_locked"))
        self.winner_lock_hash=checkpoint.get("winner_lock_sha256")
        self.phase_ledger=copy.deepcopy(checkpoint.get("phase_ledger") or [])
        self._restore_tuning_evidence()

    def _restore_tuning_evidence(self) -> None:
        for row in self.campaign.rows:
            intervention_id=str(row.get("intervention_id") or "")
            if intervention_id in {"", "CONTROL"}:
                continue
            intervention=self.campaign.intervention_by_id.get(intervention_id)
            if intervention is None:
                continue
            fixture_id=str(row.get("fixture_id") or "")
            case=self.campaign.case_by_id.get(fixture_id)
            if case is None:
                continue
            recommended=int(
                self.campaign.baseline_generation_budget_by_family.get(
                    _family(case),
                    self.campaign.cfg["base_generation_budget"],
                )
            )
            observed=int(
                row.get("generation_budget")
                or self.campaign.cfg["base_generation_budget"]
            )
            if (
                intervention.get("category")!="GENERATION_BUDGET"
                and observed!=recommended
            ):
                continue
            key=(
                fixture_id,
                int(row.get("seed") or 0),
                _intervention_fingerprint(intervention),
            )
            self.treatment_cache[key]=copy.deepcopy(row)

        valid_policy_rows=[]
        for row in self.rows:
            fixture_id=str(row.get("fixture_id") or "")
            case=self.campaign.case_by_id.get(fixture_id)
            if case is None:
                continue
            recommended=int(
                self.campaign.baseline_generation_budget_by_family.get(
                    _family(case),
                    self.campaign.cfg["base_generation_budget"],
                )
            )
            baseline_budget=int(
                row.get("baseline_generation_budget")
                or row.get("control_generation_budget")
                or self.campaign.cfg["base_generation_budget"]
            )
            selected_id=str(row.get("selected_intervention_id") or "")
            selected=self.campaign.intervention_by_id.get(selected_id)
            if (
                baseline_budget!=recommended
                and not (
                    selected is not None
                    and selected.get("category")=="GENERATION_BUDGET"
                )
            ):
                continue
            valid_policy_rows.append(row)
        self.rows=valid_policy_rows

        for row in self.rows:
            key=(
                str(row.get("policy_id") or ""),
                str(row.get("fixture_id") or ""),
                int(row.get("seed") or 0),
                str(row.get("partition") or ""),
            )
            self.policy_observation_index[key]=copy.deepcopy(row)
            route=str(row.get("route") or "")
            if route:
                self.router_cache[
                    (str(row.get("fixture_id") or ""), int(row.get("seed") or 0))
                ]=(route,{"recovered_router_evidence":True})

    def _phase_elapsed_seconds(self) -> float:
        if self.current_phase_started is None:
            return float(self.current_phase_elapsed_base)
        return float(self.current_phase_elapsed_base)+max(
            0.0,self.clock()-self.current_phase_started
        )

    def _write_checkpoint(self, *, state: str="ACTIVE") -> None:
        store=self.runner.store
        if store is None:
            return
        writer=getattr(store,"write_json_atomic",None)
        if writer is None:
            writer=getattr(store,"write_json",None)
        if writer is None:
            return
        run_id=getattr(store,"run_id",None)
        physical_calls=int(getattr(self.runner,"_model_call_counts",{}).get(run_id,0)) if run_id else 0
        payload={
            "schema_version":1,
            "recovery_policy":"NO_FULL_RERUN_ATOMIC_RESUME",
            "state":state,
            "run_id":run_id,
            "model":getattr(self.runner,"model",None),
            "collection_run":self.collection.get("run_id"),
            "active_seconds_used":min(
                float(TUNING_ACTIVE_SECONDS),
                max(0.0,self.clock()-self.start),
            ),
            "active_seconds_remaining":max(
                0.0,
                float(TUNING_ACTIVE_SECONDS)-max(0.0,self.clock()-self.start),
            ),
            "physical_model_calls_used":physical_calls,
            "completed_policy_observations":len(self.rows),
            "completed_phases":sorted(self.completed_phases),
            "current_phase":self.current_phase,
            "current_phase_elapsed_seconds":self._phase_elapsed_seconds(),
            "current_policy_ids":list(self.current_policy_ids),
            "winner_locked":copy.deepcopy(self.winner_locked),
            "winner_lock_sha256":self.winner_lock_hash,
            "phase_ledger":copy.deepcopy(self.phase_ledger),
            "reuse_counters":copy.deepcopy(self.reuse_counters),
            "full_rerun_allowed":False,
        }
        writer(
            "test1.2-tuning-recovery-checkpoint.json",
            payload,
            producer="test1.2-tuning",
            stage="recovery-checkpoint",
        )

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
        partition=self.campaign.partition_name(case)
        observation_key=(str(policy["policy_id"]),_fixture_id(case),int(seed),partition)
        if observation_key in self.policy_observation_index:
            self.reuse_counters["policy_observations_reused"] += 1
            return copy.deepcopy(self.policy_observation_index[observation_key])
        control=self.baseline(case,deadline,seed)
        if control is None:
            return None
        control_score=float(control.get("score") or 0.0)
        control_valid=bool(control.get("valid_for_capability")) or (
            (control.get("classification") or {}).get("valid_for_capability") is True
        )
        if not control_valid:
            # Runtime/capture failures are evidence, not policy outcomes. Do not
            # spend router or treatment calls behind an invalid baseline.
            return None
        baseline_budget=int(
            control.get("generation_budget")
            or self.campaign.cfg["base_generation_budget"]
        )
        policy_id=str(policy["policy_id"])
        if policy["mode"]=="direct":
            row={
                "schema_version":1,"policy_id":policy_id,"fixture_id":_fixture_id(case),
                "family_id":_family(case),"seed":seed,"control_score":control_score,
                "score":control_score,"delta":0.0,"model_calls":0,"route":"DIRECT",
                "selected_intervention_id":None,
                "valid_for_capability":control_valid,
                "control_valid_for_capability":control_valid,
                "baseline_generation_budget":baseline_budget,
                "treatment_generation_budget":baseline_budget,
                "budget_comparison_valid":True,
                "delta_valid":control_valid,
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
                    "valid_for_capability":control_valid,
                    "control_valid_for_capability":control_valid,
                    "baseline_generation_budget":baseline_budget,
                    "treatment_generation_budget":baseline_budget,
                    "budget_comparison_valid":True,
                    "delta_valid":control_valid,
                }
            else:
                trial=self._treatment_trial(selected,case,deadline,seed=seed)
                if trial is None:
                    return None
                calls=int(trial.get("model_calls_per_application") or 0)+(1 if router_aux else 0)
                treatment_budget=int(
                    trial.get("generation_budget")
                    or self.campaign.cfg["base_generation_budget"]
                )
                budget_comparison_valid=(
                    selected.get("category")=="GENERATION_BUDGET"
                    or treatment_budget==baseline_budget
                )
                trial_delta_valid=bool(
                    trial.get("delta_valid") is True
                    and budget_comparison_valid
                )
                row={
                    "schema_version":1,"policy_id":policy_id,"fixture_id":_fixture_id(case),
                    "family_id":_family(case),"seed":seed,"control_score":control_score,
                    "score":float(trial.get("score") or 0.0),
                    "delta":(
                        float(trial.get("score") or 0.0)-control_score
                        if trial_delta_valid
                        else None
                    ),
                    "model_calls":calls,"route":route,
                    "valid_for_capability":bool(trial.get("valid_for_capability")),
                    "control_valid_for_capability":control_valid,
                    "baseline_generation_budget":baseline_budget,
                    "treatment_generation_budget":treatment_budget,
                    "budget_comparison_valid":budget_comparison_valid,
                    "delta_valid":trial_delta_valid,
                    "selected_intervention_id":selected.get("id"),
                    "trial":trial,
                }
        row["partition"] = self.campaign.partition_name(case)
        row["evidence_reuse_counters"] = copy.deepcopy(self.reuse_counters)
        row["tuning_observation_sha256"] = _tuning_row_hash(row)
        self.rows.append(row)
        self.policy_observation_index[observation_key]=copy.deepcopy(row)
        self.runner.store.append_jsonl("test1.2-tuning-observations.jsonl",row)
        self._write_checkpoint()
        return row


def _balanced_validation(run: TuningRun, n: int) -> list[dict[str,Any]]:
    return _balanced_cases(run.validation,min(n,len(run.validation)))


def _priority_validation(run: TuningRun, n: int) -> list[dict[str, Any]]:
    """Spend existing validation budget on unresolved Collection gaps first."""
    limit = min(int(n), len(run.validation))
    gap_families = set(run.collection.get("collection_value_gap_families") or [])
    if not gap_families:
        return _balanced_validation(run, limit)

    gaps = [case for case in run.validation if _family(case) in gap_families]
    rest = [case for case in run.validation if _family(case) not in gap_families]
    # Within unresolved families, buy the hardest evidence first while still
    # balancing across families. This directly targets hard-case/frontier gaps.
    gaps = sorted(
        gaps,
        key=lambda case: (
            -int(case.get("difficulty_level") or 0),
            _family(case),
            _fixture_id(case),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen_per_family: dict[str, int] = defaultdict(int)
    while gaps and len(selected) < limit:
        gaps.sort(
            key=lambda case: (
                seen_per_family[_family(case)],
                -int(case.get("difficulty_level") or 0),
                _family(case),
                _fixture_id(case),
            )
        )
        case = gaps.pop(0)
        selected.append(case)
        seen_per_family[_family(case)] += 1

    if len(selected) < limit:
        selected.extend(
            _balanced_cases(rest, min(limit - len(selected), len(rest)))
        )
    return selected[:limit]


def _balanced_partition(cases: list[dict[str, Any]], n: int | None = None) -> list[dict[str, Any]]:
    limit = len(cases) if n is None else min(int(n), len(cases))
    return _balanced_cases(cases, limit)


def _evaluate(run: TuningRun, policies: list[dict[str,Any]], cases: list[dict[str,Any]], deadline: float, *, seeds: list[int]) -> dict[str,Any]:
    requested_keys=set()
    for seed in seeds:
        for policy in policies:
            for case in cases:
                partition=run.campaign.partition_name(case)
                key=(str(policy["policy_id"]),_fixture_id(case),int(seed),partition)
                requested_keys.add(key)
                if key in run.policy_observation_index:
                    run.reuse_counters["policy_observations_reused"] += 1
                    continue
                if not run.can_start(deadline):
                    break
                run.run_policy(policy,case,deadline,seed=seed)
    grouped=defaultdict(list)
    for key in requested_keys:
        row=run.policy_observation_index.get(key)
        if row is not None:
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
    partition = run.campaign.partition_name(cases[0]) if cases else "UNKNOWN"
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

    rows = [
        row for row in run.rows
        if str(row.get("policy_id")) == str(winner["policy_id"])
        and str(row.get("partition")) == partition
        and int(row.get("seed") or 0) in {int(primary_seed), int(primary_seed + 100)}
    ]
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


def run_test12_tuning(
    runner: Any,
    cases: list[dict[str,Any]],
    *,
    collection_run: str,
    clock: Callable[[],float]=time.monotonic,
    started_monotonic: float|None=None,
    resume_state: dict[str,Any]|None=None,
) -> list[dict[str,Any]]:
    assert runner.store is not None
    collection=load_collection(Path(runner.results_root),collection_run)
    plan=build_tuning_plan(cases,collection_run=collection_run)
    validate_tuning_plan(plan)
    runner.store.write_json("test1.2-tuning-plan.json",plan,producer="test1.2-tuning",stage="preflight")

    run=TuningRun(
        runner,
        cases,
        collection,
        clock=clock,
        started=started_monotonic,
        resume_state=resume_state,
    )
    policies=_policy_candidates(collection,int(run.cfg["screen_candidates"]))
    runner.store.write_json("candidate-harness-registry.json",{"schema_version":1,"policies":policies},producer="test1.2-tuning",stage="preflight")

    checkpoint=(resume_state or {}).get("checkpoint") or {}
    legacy_holdout_exposure = copy.deepcopy(
        (resume_state or {}).get("legacy_holdout_exposure")
        or checkpoint.get("legacy_holdout_exposure")
        or {"TEST2_BLIND":False,"TEST3_PROTECTED":False}
    )
    test2_blind_previously_exposed = bool(
        legacy_holdout_exposure.get("TEST2_BLIND")
    )
    test3_protected_previously_exposed = bool(
        legacy_holdout_exposure.get("TEST3_PROTECTED")
    )
    policy_by_id={str(row["policy_id"]):row for row in policies}
    current=[
        policy_by_id[value]
        for value in run.current_policy_ids
        if value in policy_by_id
    ] or policies
    ledger=copy.deepcopy(run.phase_ledger)
    aggregate_scores={}
    for item in ledger:
        for key,value in (item.get("scores") or {}).items():
            aggregate_scores[str(key)]=copy.deepcopy(value)
    final_confirmation_family_scores={}
    final_confirmation_rows=[]
    winner_locked: dict[str, Any] | None = copy.deepcopy(run.winner_locked)
    winner_lock_hash: str | None = run.winner_lock_hash
    resume_phase=str(checkpoint.get("current_phase") or "")
    resume_phase_elapsed=float(checkpoint.get("current_phase_elapsed_seconds") or 0.0)
    for phase_name,seconds in TUNING_PHASES:
        if phase_name in run.completed_phases:
            continue
        phase_elapsed_before=resume_phase_elapsed if phase_name==resume_phase else 0.0
        actual_started=run.clock()
        run.current_phase=phase_name
        run.current_phase_started=actual_started
        run.current_phase_elapsed_base=phase_elapsed_before
        remaining_phase_seconds=max(0.0,float(seconds)-phase_elapsed_before)
        deadline=min(run.active_end,actual_started+remaining_phase_seconds)
        partition = plan["phase_partition_policy"][phase_name]
        run.campaign.allowed_partitions={partition}
        run.current_policy_ids=[str(p["policy_id"]) for p in current]
        run._write_checkpoint()

        if phase_name=="validation_baseline":
            cases0=_priority_validation(run,min(len(run.validation),128))
            for case in cases0:
                if not run.can_start(deadline): break
                run.baseline(case,deadline,seed=42)
            scores={}
        elif phase_name=="candidate_harness_screen":
            scores=_evaluate(run,current,_priority_validation(run,int(run.cfg["screen_cases"])),deadline,seeds=[42])
            current=_top_policies(current,scores,max(8,len(current)//2))
        elif phase_name=="successive_halving":
            scores={}
            for n in run.cfg["halving_cases"]:
                if not run.can_start(deadline): break
                stage=_evaluate(run,current,_priority_validation(run,int(n)),deadline,seeds=[42])
                scores.update(stage)
                current=_top_policies(current,stage,max(int(run.cfg["final_candidates"]),len(current)//2))
        elif phase_name=="routing_and_boundary_tuning":
            scores=_evaluate(run,current,_priority_validation(run,min(96,len(run.validation))),deadline,seeds=[42,43])
            current=_top_policies(current,scores,int(run.cfg["final_candidates"]))
        elif phase_name=="residual_failure_replay":
            scores=_evaluate(run,current,_priority_validation(run,len(run.validation)),deadline,seeds=[43])
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
            run.winner_locked=copy.deepcopy(winner_locked)
            run.winner_lock_hash=winner_lock_hash
            current=[winner_locked]
            run.current_policy_ids=[str(winner_locked["policy_id"])]
            run._write_checkpoint()
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
        run.phase_ledger=copy.deepcopy(ledger)
        run.current_policy_ids=[str(p["policy_id"]) for p in current]
        run.completed_phases.add(phase_name)
        run.current_phase=None
        run.current_phase_started=None
        run.current_phase_elapsed_base=0.0
        run._write_checkpoint()
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
    blind_summary={"status":"NOT_EXPOSED_RESERVED_FOR_TEST2","n":0}
    protected_summary={"status":"NOT_EXPOSED_RESERVED_FOR_FINAL_ACCEPTANCE","n":0}
    blind_scores={}
    protected_scores={}
    acceptance_family_validation={}
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
    collection_value_gap_families=sorted(
        str(value)
        for value in (collection.get("collection_value_gap_families") or [])
    )
    collection_value_gaps=copy.deepcopy(collection.get("collection_value_gaps") or {})
    resolved_collection_value_gap_families=[]
    unresolved_collection_value_gap_families=[]
    for family in collection_value_gap_families:
        payload=winner_family_validation.get(family)
        if payload and _family_is_safe(payload,max_regression,minimum_pass_rate):
            resolved_collection_value_gap_families.append(family)
        else:
            unresolved_collection_value_gap_families.append(family)

    provisionally_validated_families=[]
    for family in TEST2_CAPABILITY_FAMILIES:
        validation_payload=winner_family_validation.get(family)
        if validation_payload and _family_is_safe(
            validation_payload,
            max_regression,
            minimum_pass_rate,
        ):
            provisionally_validated_families.append(family)
    blocked_families=sorted(
        set(TEST2_CAPABILITY_FAMILIES)-set(provisionally_validated_families)
    )
    all_40_families_competent=(
        len(provisionally_validated_families)==len(TEST2_CAPABILITY_FAMILIES)
    )

    if family_safe and all_40_families_competent:
        terminal_decision="PROVISIONAL_READY_FOR_TEST2"
    elif provisionally_validated_families:
        terminal_decision="PROVISIONAL_CONSTRAINED_FOR_TEST2"
    else:
        terminal_decision="REJECT_BEFORE_TEST2"

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
        "collection_opportunity_discovery":copy.deepcopy(
            collection.get("opportunity_discovery") or {}
        ),
        "direct_default_when_unmatched":True,
        "oracle_routing_prohibited":True,
        "hard_ceiling_total_seconds":TUNING_HARD_SECONDS+COLLECTION_HARD_SECONDS,
        "winner_locked_before_test2":True,
        "test2_blind_exposed":test2_blind_previously_exposed,
        "test3_protected_exposed":test3_protected_previously_exposed,
        "winner_lock_sha256":winner_lock_hash,
        "blind_acceptance_is_tuning_input":False,
        "protected_acceptance_is_tuning_input":False,
        "terminal_decision":terminal_decision,
        "provisionally_validated_capability_families":provisionally_validated_families,
        "blocked_capability_families":blocked_families,
        "collection_value_gap_families":collection_value_gap_families,
        "resolved_collection_value_gap_families":resolved_collection_value_gap_families,
        "unresolved_collection_value_gap_families":unresolved_collection_value_gap_families,
        "collection_value_gaps":collection_value_gaps,
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
        "collection_value_gap_families":collection_value_gap_families,
        "resolved_collection_value_gap_families":resolved_collection_value_gap_families,
        "unresolved_collection_value_gap_families":unresolved_collection_value_gap_families,
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
        "rule":"these are provisional validation-derived weight candidates only; Test 2 must establish persistent model-owned failures before weight tuning is authorized",
    }
    runner.store.write_json("fine-tuning-qualification.json",qualification,producer="test1.2-tuning",stage="report")

    provisional_result={
        "schema_version":1,
        "artifact_role":"PROVISIONAL_PRE_TEST2_VALIDATION_RESULT",
        "onboarding_complete":False,
        "release_authorized":False,
        "winner_policy_id":winner.get("policy_id"),
        "winner_lock_sha256":winner_lock_hash,
        "winner_locked_before_test2":True,
        "holdouts_used_for_tuning_or_selection":False,
        "test2_blind_exposed":test2_blind_previously_exposed,
        "test3_protected_exposed":test3_protected_previously_exposed,
        "validation_family_safe":family_safe,
        "all_40_families_validation_competent":all_40_families_competent,
        "minimum_validation_pass_rate":minimum_pass_rate,
        "maximum_capability_regression_rate":max_regression,
        "test2_blind":{
            "status":"RESERVED_UNTOUCHED_FOR_TEST2_PROOF",
            "summary":blind_summary,
        },
        "test3_protected":{
            "status":"RESERVED_UNTOUCHED_FOR_FINAL_RELEASE_ACCEPTANCE",
            "summary":protected_summary,
        },
        "terminal_decision":terminal_decision,
        "provisionally_validated_capability_families":provisionally_validated_families,
        "blocked_capability_families":blocked_families,
        "collection_value_gap_families":collection_value_gap_families,
        "resolved_collection_value_gap_families":resolved_collection_value_gap_families,
        "unresolved_collection_value_gap_families":unresolved_collection_value_gap_families,
        "collection_value_gaps_are_validation_priorities_not_release_evidence":True,
        "collection_role":"OPPORTUNITY_DISCOVERY",
        "compiler_role":"VALIDATION_ONLY_POLICY_LOCK",
        "proof_owner":"RUN2_TEST2",
        "final_release_owner":"FRESH_PROTECTED_ACCEPTANCE",
        "next_action":(
            "RUN_TEST2_PROOF_STAGE"
            if terminal_decision!="REJECT_BEFORE_TEST2"
            else "DO_NOT_ADVANCE_TO_TEST2_WITH_THIS_CANDIDATE"
        ),
        "recovery_policy":{
            "full_rerun_allowed":False,
            "interruption_action":"RESUME_SAME_RUN_ID",
            "corruption_action":"QUARANTINE_DAMAGED_ATOMIC_RECORD_AND_REPLAY_ONLY_MISSING_SLICE",
            "preserve_valid_evidence":True,
            "preserve_elapsed_time_budget":True,
            "preserve_physical_call_budget":True,
            "preserve_winner_lock":True,
        },
        "new_onboarding_event_only_if":[
            "model_weights_changed_materially",
            "runtime_behavior_changed_materially",
            "benchmark_contract_changed_materially",
        ],
    }
    runner.store.write_json(
        "test1.2-final-acceptance.json",
        provisional_result,
        producer="test1.2-tuning",
        stage="provisional-compiler",
    )

    capability_contract={
        "schema_version":1,
        "model":getattr(runner,"model",None),
        "status":"PROVISIONAL_PENDING_TEST2_PROOF",
        "terminal_decision":terminal_decision,
        "deployable":False,
        "mode":(
            "PROVISIONAL_TEST2_CANDIDATE"
            if terminal_decision!="REJECT_BEFORE_TEST2"
            else "DISABLED"
        ),
        "provisionally_validated_capability_families":(
            provisionally_validated_families
            if terminal_decision!="REJECT_BEFORE_TEST2"
            else []
        ),
        "blocked_capability_families":blocked_families,
        "unmatched_task_policy":"DIRECT_OR_STRONGER_MODEL_FALLBACK",
        "winner_lock_sha256":winner_lock_hash,
        "validation_artifact":"test1.2-final-acceptance.json",
        "test2_proof_required":True,
        "fresh_protected_release_acceptance_required":True,
    }
    runner.store.write_json(
        "integration-capability-contract.json",
        capability_contract,
        producer="test1.2-tuning",
        stage="provisional-compiler",
    )

    integration_package={
        "schema_version":1,
        "package_type":"INVERTED_MODEL_ONBOARDING_PROVISIONAL_PACKAGE",
        "model":getattr(runner,"model",None),
        "onboarding_complete":False,
        "release_authorized":False,
        "terminal_decision":terminal_decision,
        "collection_run":collection_run,
        "winner_policy":winner,
        "winner_lock_sha256":winner_lock_hash,
        "compiled_harness_policy":"compiled-harness-policy.json",
        "capability_contract":"integration-capability-contract.json",
        "validation_result":"test1.2-final-acceptance.json",
        "do_not_use_registry":"do-not-use-registry.json",
        "route_map":route_map,
        "tool_execution_policy":tool_execution_policy,
        "frontier_gap_policy":frontier_gap_policy,
        "second_gap_policy":second_gap_policy,
        "evidence_reuse":copy.deepcopy(run.reuse_counters),
        "collection_value_gap_families":collection_value_gap_families,
        "resolved_collection_value_gap_families":resolved_collection_value_gap_families,
        "unresolved_collection_value_gap_families":unresolved_collection_value_gap_families,
        "test2_blind_exposed":test2_blind_previously_exposed,
        "test3_protected_exposed":test3_protected_previously_exposed,
        "next_action":(
            "RUN_TEST2_PROOF_STAGE"
            if terminal_decision!="REJECT_BEFORE_TEST2"
            else "DO_NOT_ADVANCE_TO_TEST2_WITH_THIS_CANDIDATE"
        ),
        "optional_future_weight_improvement":{
            "qualification":"fine-tuning-qualification.json",
            "training_corpus":"fine-tuning-training-corpus.jsonl",
            "note":"weight tuning is considered only after Test 2 establishes the persistent capability floor or the harness ceiling is reached",
        },
        "total_two_run_hard_ceiling_seconds":TUNING_HARD_SECONDS+COLLECTION_HARD_SECONDS,
    }
    runner.store.write_json(
        "inverted-model-integration-package.json",
        integration_package,
        producer="test1.2-tuning",
        stage="provisional-compiler",
    )

    terminal_handoff={
        "schema_version":1,
        "state":"TEST1.2_PROVISIONAL_COMPILER_COMPLETE",
        "terminal_decision":terminal_decision,
        "integration_package":"inverted-model-integration-package.json",
        "winner_lock_sha256":winner_lock_hash,
        "test2_proof_stage_required":True,
        "test2_role":"RECURRENCE_ROBUSTNESS_NEGATIVE_TRANSFER_DISTILLATION_PROOF",
        "test2_blind_reserved_and_unexposed":not test2_blind_previously_exposed,
        "test3_protected_reserved_and_unexposed":not test3_protected_previously_exposed,
        "legacy_holdout_exposure":copy.deepcopy(legacy_holdout_exposure),
        "fresh_protected_final_acceptance_required":True,
        "next_action":(
            "GENERATE_FRESH_TEST2_BLIND_BEFORE_TEST2"
            if test2_blind_previously_exposed
            else "RUN_TEST2_PROOF_STAGE"
            if terminal_decision!="REJECT_BEFORE_TEST2"
            else "STOP_CANDIDATE"
        ),
        "recovery_policy":provisional_result["recovery_policy"],
        "new_onboarding_event_only_if":provisional_result["new_onboarding_event_only_if"],
    }
    runner.store.write_json(
        "test1.2-terminal-handoff.json",
        terminal_handoff,
        producer="test1.2-tuning",
        stage="provisional-compiler",
    )

    runner.store.write_json("model-harness-card.json",{
        "schema_version":1,
        "model":getattr(runner,"model",None),
        "collection_run":collection_run,
        "compiled_policy":"compiled-harness-policy.json",
        "validated_policy_summary":winner_summary,
        "validated_family_summaries":winner_family_validation,
        "test2_blind_acceptance":{"status":"UNTOUCHED_RESERVED_FOR_TEST2"},
        "test3_protected_acceptance":{"status":"UNTOUCHED_RESERVED_FOR_FINAL_RELEASE"},
        "frontier_gap_policy":frontier_gap_policy,
        "second_gap_policy":second_gap_policy,
        "zero_clock_model_manufacturing":copy.deepcopy(
            collection.get("zero_clock_model") or {}
        ),
        "all_40_families_non_regressing_on_validation":family_safe,
        "release_status":"PENDING_TEST2_PROOF",
        "provisional_decision":terminal_decision,
        "provisionally_validated_capability_families":provisionally_validated_families,
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
        "blind_partitions_touched":bool(
            test2_blind_previously_exposed
            or test3_protected_previously_exposed
        ),
        "legacy_holdout_exposure":copy.deepcopy(legacy_holdout_exposure),
        "holdouts_used_for_tuning_or_selection":False,
        "onboarding_complete":False,
        "test2_proof_stage_required":True,
        "fresh_protected_release_acceptance_required":True,
        "integration_package":"inverted-model-integration-package.json",
        "total_two_run_hard_ceiling_hours":(TUNING_HARD_SECONDS+COLLECTION_HARD_SECONDS)/3600.0,
    },producer="test1.2-tuning",stage="report")
    run._write_checkpoint(state="PROVISIONAL_COMPILER_COMPLETE")
    return run.rows
