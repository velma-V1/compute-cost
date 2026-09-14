"""Seven-hour GPT-20B Test-2 break/recover/distill/finalize campaign.

Test 2 consumes the machine-readable Test-1 handoff, attacks higher-order and
recurrence behavior, learns failure recovery, measures negative transfer and rare
boundary failures, distills minimal recipes, performs blind confirmation, and
emits the complete Inverted finalization + Test-3 handoff contract.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable

from .characterization import execute_experiment
from .experiments import ExperimentSpec, make_experiment_id
from .evidence import EvidenceStore
from .test11_campaign import TRUNCATION_CLASSES
from .test12_campaign import (
    Test12Campaign,
    _intervention_fingerprint,
    fresh_model_source,
    partition_test12_cases,
)

CENSORING_CLASSES = frozenset(set(TRUNCATION_CLASSES) | {"NO_FINAL_ANSWER"})

from .test1_campaign import (
    ACTIVE_SECONDS,
    HARD_SECONDS,
    EPSILON_NOISE,
    INGREDIENTS,
    _balanced_cases,
    _family,
    _fixture_id,
    build_treatment_messages,
    effect_summary,
    partition_cases,
)

CALL_START_CUTOFF_SECONDS = ACTIVE_SECONDS

PHASES = (
    ("recurrence_higher_order", 85 * 60),
    ("failure_recovery", 105 * 60),
    ("negative_transfer", 55 * 60),
    ("censoring_cost_tradeoff", 30 * 60),
    ("purple_unicorn", 45 * 60),
    ("knockout_distillation", 55 * 60),
    ("blind_confirmation", 35 * 60),
)

REQUIRED_TEST1_FILES = (
    "test1-plan.json",
    "fixture-partitions.json",
    "noise-model.json",
    "ingredient-registry.json",
    "pair-interaction-graph.json",
    "directional-order-graph.json",
    "higher-order-candidate-queue.json",
    "failure-registry.json",
    "negative-effect-registry.json",
    "uncertainty-ledger.json",
    "test2-priority-queue.json",
)

REQUIRED_OUTPUTS = (
    "recurrence-map.json",
    "proof-value-scheduler.json",
    "higher-order-hypergraph.json",
    "failure-phenotype-registry.json",
    "failure-recovery-matrix.json",
    "negative-transfer-boundaries.json",
    "harm-sentinel-evidence.json",
    "censoring-cost-tradeoff.json",
    "purple-unicorn-registry.json",
    "knockout-registry.json",
    "minimal-recipe-registry.json",
    "blind-confirmation-results.json",
    "model-limit-registry.json",
    "remaining-unknowns.json",
    "fine-tuning-candidate-queue.json",
    "test3-dataset-manifest.json",
    "test3-protected-manifest.json",
    "inverted-finalization-contract.json",
    "final-acceptance-manifest.json",
    "rollback-configuration.json",
    "exact-model-configuration.json",
    "latency-resource-envelope.json",
    "holdout-consumption-ledger.json",
    "holdout-replenishment-plan.json",
    "harness-stopping-rules.json",
    "training-asset-yield.json",
)

DEFAULT_TEST2_CONFIG: dict[str, Any] = {
    "expected_calls": 4100,
    "safety_call_cap": 10000,
    "generation_budget": 256,
    "generation_budget_by_family": {},
    "thinking_mode": False,
    "reasoning_effort": None,
    "bootstrap_samples": 500,
    "recurrence_min_independent_fixtures": 4,
    "recurrence_min_distinct_seeds": 2,
    "recurrence_target_valid_observations": 6,
    "recurrence_max_valid_observations": 12,
    "top_recipes": 12,
    "max_recovery_recipes": 8,
    "negative_transfer_recipes": 8,
    "blind_recipes": 4,
    "control_interval": 12,
    "fine_tuning_min_independent_failures": 3,
    "general_recovery_threshold": 0.80,
    "partial_recovery_threshold": 0.60,
    "acceptance_latency_ratio": 1.25,
    "max_effect_censoring_rate": 0.20,
    "censoring_cost_probe_multipliers": [2, 4],
    "censoring_cost_probe_max_budget": 4096,
    "harm_sentinel_min_per_control": 16,
    "harm_min_distinct_families": 4,
    "harm_max_break_rate": 0.05,
    "holdout_max_cross_run_acceptance_uses": 1,
    "policy_cost_ratio_ceiling": 1.25,
    "minimum_accuracy_advantage_when_over_cost_ceiling": 0.02,
}


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_TEST2_CONFIG)
    incoming = config.get("test2_campaign")
    if isinstance(incoming, dict):
        merged.update(copy.deepcopy(incoming))
    return merged


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            value = json.loads(raw)
            if isinstance(value, dict):
                result.append(value)
    return result



def _wilson90(successes: int, total: int, z: float = 1.6448536269514722) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _fixture_set_fingerprint(cases: Iterable[dict[str, Any]]) -> str:
    fixture_ids = sorted({_fixture_id(case) for case in cases})
    payload = json.dumps(fixture_ids, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prior_holdout_claims(
    results_root: Path,
    *,
    partition: str,
    fingerprint: str,
    current_run_id: str,
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    if not results_root.is_dir():
        return claims
    for run_dir in results_root.iterdir():
        if not run_dir.is_dir() or run_dir.name == current_run_id:
            continue
        path = run_dir / "holdout-consumption-ledger.json"
        if not path.is_file():
            continue
        try:
            payload = _read_json(path)
        except Exception:
            continue
        for claim in payload.get("claims") or []:
            if (
                claim.get("partition") == partition
                and claim.get("partition_fingerprint") == fingerprint
            ):
                claims.append(copy.deepcopy(claim))
    return claims


def _claim_holdout_partition(
    campaign: "Test2Campaign",
    partition: str,
    fixtures: list[dict[str, Any]],
) -> dict[str, Any]:
    store = campaign.runner.store
    run_id = str(getattr(store, "run_id", "") or store.run_dir.name)
    fingerprint = _fixture_set_fingerprint(fixtures)
    fixture_ids = sorted({_fixture_id(case) for case in fixtures})
    ledger_path = store.run_dir / "holdout-consumption-ledger.json"

    if ledger_path.is_file():
        payload = _read_json(ledger_path)
        for claim in payload.get("claims") or []:
            if (
                claim.get("partition") == partition
                and claim.get("partition_fingerprint") == fingerprint
                and claim.get("cycle_run_id") == run_id
            ):
                return copy.deepcopy(claim)
    else:
        payload = {"schema_version":1, "claims":[]}

    prior = _prior_holdout_claims(
        Path(campaign.runner.results_root),
        partition=partition,
        fingerprint=fingerprint,
        current_run_id=run_id,
    )
    max_uses = int(campaign.cfg["holdout_max_cross_run_acceptance_uses"])
    if len(prior) >= max_uses:
        raise ValueError(
            f"{partition} holdout has already been consumed by a prior acceptance cycle; "
            "generate a fresh partition before continuing"
        )

    # A new cycle must not recycle exact fixtures from prior claimed holdouts,
    # even if the full partition fingerprint changed.
    prior_fixture_ids: set[str] = set()
    root = Path(campaign.runner.results_root)
    if root.is_dir():
        for run_dir in root.iterdir():
            if not run_dir.is_dir() or run_dir.name == run_id:
                continue
            path = run_dir / "holdout-consumption-ledger.json"
            if not path.is_file():
                continue
            try:
                prior_payload = _read_json(path)
            except Exception:
                continue
            for old in prior_payload.get("claims") or []:
                if old.get("partition") == partition:
                    prior_fixture_ids.update(str(v) for v in old.get("fixture_ids") or [])
    overlap = sorted(set(fixture_ids) & prior_fixture_ids)
    if overlap:
        raise ValueError(
            f"{partition} contains {len(overlap)} fixtures already consumed by prior cycles; "
            "cycle N acceptance must use fixtures absent from earlier acceptance partitions"
        )

    claim = {
        "partition":partition,
        "partition_fingerprint":fingerprint,
        "fixture_ids":fixture_ids,
        "fixture_count":len(fixture_ids),
        "cycle_run_id":run_id,
        "cross_run_acceptance_use_number":len(prior) + 1,
        "max_cross_run_acceptance_uses":max_uses,
        "status":"CONSUMED_ON_EXPOSURE",
        "retire_permanently_after_cycle":True,
        "same_run_resume_allowed":True,
        "next_cycle_requirement":"GENERATE_NEW_FIXTURES_FROM_NEW_FIELD_FAILURE_PHENOTYPES_WITH_ZERO_FIXTURE_OVERLAP",
    }
    payload.setdefault("claims", []).append(copy.deepcopy(claim))
    writer = getattr(store, "write_json_atomic", None) or store.write_json
    writer(
        "holdout-consumption-ledger.json",
        payload,
        producer="test2",
        stage="holdout-governance",
    )
    return claim


def _capability_floor_signature(limits: dict[str, Any]) -> dict[str, Any]:
    phenotypes = limits.get("phenotypes") or {}
    floor = sorted(
        key
        for key, row in phenotypes.items()
        if row.get("owner") in {
            "MODEL_CAPABILITY_LIMIT",
            "FINE_TUNING_CANDIDATE",
        }
    )
    payload = json.dumps(floor, sort_keys=True, separators=(",", ":"))
    return {
        "phenotype_ids":floor,
        "count":len(floor),
        "sha256":hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _previous_cycle_floor(campaign: "Test2Campaign") -> dict[str, Any] | None:
    root = Path(campaign.runner.results_root)
    store = getattr(campaign.runner, "store", None)
    current = str(getattr(store, "run_id", "") or "")
    model = str(campaign.runner.model)
    candidates: list[tuple[float, dict[str, Any]]] = []
    if not root.is_dir():
        return None
    for run_dir in root.iterdir():
        if not run_dir.is_dir() or run_dir.name == current:
            continue
        contract_path = run_dir / "inverted-finalization-contract.json"
        limits_path = run_dir / "model-limit-registry.json"
        if not contract_path.is_file() or not limits_path.is_file():
            continue
        try:
            contract = _read_json(contract_path)
            if str(contract.get("base_model")) != model:
                continue
            limits = _read_json(limits_path)
            candidates.append((limits_path.stat().st_mtime, _capability_floor_signature(limits)))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _build_harness_stopping_rules(
    campaign: "Test2Campaign",
    blind: dict[str, Any],
    limits: dict[str, Any],
) -> dict[str, Any]:
    blind_effects = list((blind.get("effects") or {}).values())
    positive_values = [
        float(row.get("normalized_effect"))
        for row in blind_effects
        if isinstance(row.get("normalized_effect"), (int, float))
    ]
    best_fresh_holdout_value = max(positive_values) if positive_values else None
    rule1_triggered = (
        best_fresh_holdout_value is not None
        and best_fresh_holdout_value <= 0.0
    )

    harm_ceiling = float(campaign.cfg["harm_max_break_rate"])
    harmful_ci = [
        key for key, row in campaign.harm_evidence.items()
        if isinstance((row.get("break_rate_wilson90") or [None, None])[1], (int, float))
        and float(row["break_rate_wilson90"][1]) > harm_ceiling
    ]
    rule3_triggered = bool(harmful_ci)

    current_floor = _capability_floor_signature(limits)
    previous_floor = _previous_cycle_floor(campaign)
    rule4_triggered = bool(
        previous_floor
        and current_floor["sha256"] == previous_floor.get("sha256")
    )

    return {
        "schema_version":1,
        "rules":[
            {
                "id":"STOP-1",
                "rule":"MARGINAL_NET_VALUE_NEXT_CONTROL_LE_ZERO_ON_FRESH_HOLDOUT",
                "best_fresh_holdout_normalized_effect":best_fresh_holdout_value,
                "triggered":rule1_triggered,
            },
            {
                "id":"STOP-2",
                "rule":"POLICY_COST_EXCEEDS_KX_DIRECT_AND_ACCURACY_ADVANTAGE_BELOW_PREREGISTERED_EXCHANGE_RATE",
                "cost_ratio_ceiling":float(campaign.cfg["policy_cost_ratio_ceiling"]),
                "minimum_accuracy_advantage":float(
                    campaign.cfg["minimum_accuracy_advantage_when_over_cost_ceiling"]
                ),
                "observed":copy.deepcopy(blind.get("policy_cost_exchange") or {}),
                "triggered":bool(
                    isinstance((blind.get("policy_cost_exchange") or {}).get("mean_policy_cost_ratio"), (int, float))
                    and float((blind.get("policy_cost_exchange") or {})["mean_policy_cost_ratio"])
                    > float(campaign.cfg["policy_cost_ratio_ceiling"])
                    and isinstance((blind.get("policy_cost_exchange") or {}).get("accuracy_advantage"), (int, float))
                    and float((blind.get("policy_cost_exchange") or {})["accuracy_advantage"])
                    < float(campaign.cfg["minimum_accuracy_advantage_when_over_cost_ceiling"])
                ),
                "evaluation_owner":"FRESH_TEST2_BLIND_WITH_FROZEN_DIRECT_BASELINE",
            },
            {
                "id":"STOP-3",
                "rule":"HARM_CI_UPPER_EXCEEDS_PREREGISTERED_BREAK_RATE_CEILING",
                "harm_ci":"WILSON_90",
                "break_rate_ceiling":harm_ceiling,
                "controls_over_ceiling":harmful_ci,
                "triggered":rule3_triggered,
            },
            {
                "id":"STOP-4",
                "rule":"CAPABILITY_FLOOR_UNCHANGED_ACROSS_TWO_CYCLES",
                "current_floor":current_floor,
                "previous_cycle_floor":previous_floor,
                "triggered":rule4_triggered,
            },
        ],
        "harness_ceiling_reached":rule4_triggered,
        "next_strategy":(
            "MOVE_TO_WEIGHTS"
            if rule4_triggered
            else "CONTINUE_HARNESS_ONLY_WHILE_FRESH_HOLDOUT_NET_VALUE_REMAINS_POSITIVE"
        ),
        "stop_shipping_new_controls":bool(
            rule1_triggered
            or rule3_triggered
            or bool(
                isinstance((blind.get("policy_cost_exchange") or {}).get("mean_policy_cost_ratio"), (int, float))
                and float((blind.get("policy_cost_exchange") or {})["mean_policy_cost_ratio"])
                > float(campaign.cfg["policy_cost_ratio_ceiling"])
                and isinstance((blind.get("policy_cost_exchange") or {}).get("accuracy_advantage"), (int, float))
                and float((blind.get("policy_cost_exchange") or {})["accuracy_advantage"])
                < float(campaign.cfg["minimum_accuracy_advantage_when_over_cost_ceiling"])
            )
        ),
    }


def build_test2_plan(cases: list[dict[str, Any]], *, test1_run: str | None = None) -> dict[str, Any]:
    partitions = partition_cases(cases)
    return {
        "schema_version": 1,
        "campaign": "gpt20b-test2-break-recover-distill-finalize",
        "source_test1_run": test1_run,
        "wall_clock_seconds": HARD_SECONDS,
        "active_model_seconds": ACTIVE_SECONDS,
        "call_start_cutoff_seconds": CALL_START_CUTOFF_SECONDS,
        "phases": [{"name": name, "seconds": seconds} for name, seconds in PHASES],
        "partition_counts": {name: len(rows) for name, rows in partitions.items()},
        "allowed_partitions": ["DISCOVERY", "VALIDATION", "TEST2_BLIND"],
        "prohibited_partitions": ["TEST3_PROTECTED"],
        "required_test1_files": list(REQUIRED_TEST1_FILES),
        "required_outputs": list(REQUIRED_OUTPUTS),
        "stage_role": "DEDICATED_PROOF_RECURRENCE_ROBUSTNESS_DISTILLATION",
        "test12_exact_control_execution_required": True,
        "runtime_characterization_profile_required_for_test12": True,
        "in_run_budget_calibration_prohibited": True,
        "legacy_recipe_translation_to_test12_prohibited": True,
        "finalization_contract": {
            "system_recipe": True,
            "recovery_policy": True,
            "tool_boundaries": True,
            "ownership_rules": True,
            "routing_and_retry": True,
            "context_memory_policy": True,
            "negative_transfer_boundaries": True,
            "minimal_recipe": True,
            "exact_model_config": True,
            "latency_resource_envelope": True,
            "rollback": True,
            "acceptance_thresholds": True,
            "test3_handoff": True,
        },
    }


def validate_test2_plan(plan: dict[str, Any]) -> None:
    if plan["wall_clock_seconds"] != 7 * 3600:
        raise ValueError("Test 2 wall clock must be exactly seven hours")
    if sum(int(row["seconds"]) for row in plan["phases"]) != ACTIVE_SECONDS:
        raise ValueError("Test 2 active phases must total exactly 6h50m")
    counts = plan["partition_counts"]
    for name in ("DISCOVERY", "VALIDATION", "TEST2_BLIND", "TEST3_PROTECTED"):
        if int(counts.get(name, 0)) <= 0:
            raise ValueError(f"{name} partition is empty")
    if plan["prohibited_partitions"] != ["TEST3_PROTECTED"]:
        raise ValueError("Test 2 may never expose Test-3 protected fixtures")
    missing = [name for name in REQUIRED_OUTPUTS if name not in plan["required_outputs"]]
    if missing:
        raise ValueError(f"Test 2 output contract incomplete: {missing}")


def synthetic_test1_handoff(cases: list[dict[str, Any]]) -> dict[str, Any]:
    partitions = partition_cases(cases)
    ingredients = []
    for item in INGREDIENTS[:6]:
        ingredients.append({
            **copy.deepcopy(item),
            "discovery": {
                "classification": "PROMISING",
                "n": 16,
                "median_delta": 0.1,
                "normalized_effect": 1.0,
                "treatment": {
                    "ingredient_ids": [item["id"]],
                    "dose": 1.0,
                    "representation": "prose",
                    "placement": "prefix",
                },
            },
        })
    queue = []
    for item in ingredients[:4]:
        queue.append({
            "key": f"single:{item['id']}",
            "classification": "PROMISING",
            "n": 16,
            "median_delta": 0.1,
            "normalized_effect": 1.0,
            "next_action": "GENERALIZATION_TEST",
            "treatment": item["discovery"]["treatment"],
        })
    return {
        "run_id": "SYNTHETIC-TEST1-HANDOFF",
        "synthetic": True,
        "plan": build_test2_plan(cases, test1_run="SYNTHETIC-TEST1-HANDOFF"),
        "partitions": {
            name: [_fixture_id(case) for case in rows]
            for name, rows in partitions.items()
        },
        "noise_model": {"global_noise_sigma": EPSILON_NOISE, "families": {}},
        "ingredient_registry": {"ingredients": ingredients},
        "priority_queue": {"queue": queue},
        "higher_order": {"candidates": []},
        "failures": {"failures": []},
        "negative_effects": {"effects": []},
        "uncertainty": {"unknowns": []},
        "pair_graph": {"edges": {}},
        "direction_graph": {"edges": {}},
        "source_observations": [],
        "baselines": {},
    }


def load_test1_handoff(
    results_root: Path,
    run_id: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    run_dir = results_root / run_id
    if not run_dir.is_dir():
        raise ValueError(f"Test-1 run does not exist: {run_id}")

    if (run_dir / "test1.2-terminal-handoff.json").is_file():
        manifest_problems = EvidenceStore(results_root, run_id).verify_manifest()
        if manifest_problems:
            raise ValueError(
                f"Test 1.2 provisional evidence manifest verification failed: {manifest_problems}"
            )
        terminal = _read_json(run_dir / "test1.2-terminal-handoff.json")
        package = _read_json(run_dir / "inverted-model-integration-package.json")
        compiled = _read_json(run_dir / "compiled-harness-policy.json")
        if terminal.get("state") != "TEST1.2_PROVISIONAL_COMPILER_COMPLETE":
            raise ValueError("Test 2 requires a completed provisional Test 1.2 compiler handoff")
        if package.get("release_authorized") is not False:
            raise ValueError("Test 1.2 package must be provisional and non-deployable before Test 2")
        if terminal.get("test2_blind_reserved_and_unexposed") is not True:
            raise ValueError("Test 2 blind holdout was already exposed before Test 2")

        collection_run = str(package.get("collection_run") or "")
        collection_dir = results_root / collection_run
        if not collection_run or not collection_dir.is_dir():
            raise ValueError("Test 1.2 provisional package does not resolve to a Collection run")
        collection_problems = EvidenceStore(results_root, collection_run).verify_manifest()
        if collection_problems:
            raise ValueError(
                f"Test 1.2 Collection manifest verification failed: {collection_problems}"
            )
        required = [
            "full-control-candidate-registry.json",
            "runtime-characterization-profile.json",
            "test1.2-handoff.json",
            "test1.2-opportunity-discovery-map.json",
            "control-redundancy-map.json",
            "test1.2-observations.jsonl",
        ]
        missing = [name for name in required if not (collection_dir / name).is_file()]
        if missing:
            raise ValueError(f"Test 1.2 exact handoff incomplete; missing: {missing}")

        runtime_profile = _read_json(collection_dir / "runtime-characterization-profile.json")
        if runtime_profile.get("gate_passed") is not True:
            raise ValueError("Test 2 refuses a Test 1.2 source whose Stage 0 runtime gate did not pass")

        registry = _read_json(collection_dir / "full-control-candidate-registry.json")
        candidates = [
            copy.deepcopy(row)
            for row in registry.get("candidates") or []
            if isinstance(row, dict) and row.get("id")
        ]
        candidate_by_id = {str(row["id"]): row for row in candidates}
        redundancy_map = _read_json(collection_dir / "control-redundancy-map.json")
        representative_by_member: dict[str, str] = {}
        redundancy_representatives: list[str] = []
        redundancy_alternates: list[str] = []
        for cluster in redundancy_map.get("clusters") or []:
            representative = str(cluster.get("representative_intervention_id") or "")
            if representative and representative in candidate_by_id:
                if representative not in redundancy_representatives:
                    redundancy_representatives.append(representative)
            for member in cluster.get("member_intervention_ids") or []:
                ident = str(member or "")
                if ident and representative:
                    representative_by_member[ident] = representative
            for alternate in cluster.get("alternate_intervention_ids") or []:
                ident = str(alternate or "")
                if ident and ident in candidate_by_id and ident not in redundancy_alternates:
                    redundancy_alternates.append(ident)

        policy_registry_path = run_dir / "candidate-harness-registry.json"
        policies = (
            _read_json(policy_registry_path).get("policies") or []
            if policy_registry_path.is_file()
            else []
        )
        winner = copy.deepcopy(compiled.get("winner_policy") or {})
        selected_ids: list[str] = []
        required_policy_control_ids: list[str] = []

        def add_required_id(value: Any) -> None:
            ident = str(value or "")
            if ident and ident != "None" and ident in candidate_by_id and ident not in required_policy_control_ids:
                required_policy_control_ids.append(ident)

        def add_id(value: Any) -> None:
            ident = str(value or "")
            if ident and ident != "None" and ident in candidate_by_id and ident not in selected_ids:
                selected_ids.append(ident)

        add_required_id(winner.get("intervention_id"))
        add_required_id(winner.get("fallback_intervention_id"))
        for ident in (winner.get("route_map") or {}).values():
            add_required_id(ident)
        # Required controls reachable from the frozen winner are never
        # deduplicated away. They are proof obligations regardless of cluster.
        for ident in required_policy_control_ids:
            add_id(ident)

        # Buy the broadest independent mechanism evidence first. Redundancy
        # clustering changes proof order only; it never deletes an alternate.
        for ident in redundancy_representatives:
            if len(selected_ids) >= 12:
                break
            add_id(ident)

        # Preserve non-redundant compiler candidates next.
        for policy in policies:
            if len(selected_ids) >= 12:
                break
            ident = str(policy.get("intervention_id") or "")
            representative = representative_by_member.get(ident)
            if representative and ident != representative and ident not in required_policy_control_ids:
                continue
            add_id(ident)

        # Alternates remain available when capacity remains. They are fallback
        # proof candidates if their representative fails, censors, or harms.
        for ident in redundancy_alternates:
            if len(selected_ids) >= 12:
                break
            add_id(ident)

        if not selected_ids:
            for row in candidates[:12]:
                add_id(row.get("id"))

        exact_controls = []
        for ident in selected_ids:
            intervention = copy.deepcopy(candidate_by_id[ident])
            semantic_hash = _intervention_fingerprint(intervention)
            exact_controls.append({
                "source":"TEST1.2_EXACT_CONTROL",
                "source_key":ident,
                "classification":"PROVISIONAL_TEST1.2",
                "normalized_effect":0.0,
                "exact_test12_intervention":intervention,
                "intervention_id":ident,
                "semantic_hash":semantic_hash,
                "discovery_semantic_hash":semantic_hash,
                "proof_semantic_hash":semantic_hash,
                "ingredient_ids":[f"TEST12:{ident}"],
                "dose":1.0,
                "representation":"exact",
                "placement":"semantic",
            })

        opportunity = _read_json(collection_dir / "test1.2-opportunity-discovery-map.json")
        unresolved_ids = [
            str(value)
            for value in (opportunity.get("unresolved_failed_fixtures") or [])
        ]
        collection_observations = _read_jsonl(
            collection_dir / "test1.2-observations.jsonl"
        )
        valid_failure_baseline_by_fixture: dict[str, dict[str, Any]] = {}
        for row in collection_observations:
            fixture_id = str(row.get("fixture_id") or "")
            if fixture_id not in unresolved_ids:
                continue
            classification = row.get("classification") or {}
            capability_valid = bool(
                row.get("valid_for_capability") is True
                or classification.get("valid_for_capability") is True
            )
            if (
                row.get("intervention_id") == "CONTROL"
                and capability_valid
                and float(row.get("score") or 0.0) < 1.0
            ):
                valid_failure_baseline_by_fixture[fixture_id] = row

        missing_failure_provenance = sorted(
            set(unresolved_ids) - set(valid_failure_baseline_by_fixture)
        )
        if missing_failure_provenance:
            raise ValueError(
                "Test 1.2 unresolved failure lacks valid baseline provenance: "
                + ", ".join(missing_failure_provenance)
            )

        failures = {
            "failures":[
                {
                    "fixture_id":fixture_id,
                    "family_id":valid_failure_baseline_by_fixture[fixture_id].get("family_id"),
                    "difficulty_level":valid_failure_baseline_by_fixture[fixture_id].get("difficulty_level"),
                    "classification":copy.deepcopy(
                        valid_failure_baseline_by_fixture[fixture_id].get("classification") or {}
                    ),
                    "valid_for_capability":True,
                    "partition":valid_failure_baseline_by_fixture[fixture_id].get("partition") or "DISCOVERY",
                    "source_experiment_id":valid_failure_baseline_by_fixture[fixture_id].get("experiment_id"),
                    "source_collection_run":collection_run,
                    "source_evidence_kind":"CAPABILITY_VALID_UNRESOLVED_BASELINE_FAILURE",
                }
                for fixture_id in unresolved_ids
            ]
        }
        return {
            "run_id":run_id,
            "run_dir":str(run_dir),
            "collection_run":collection_run,
            "collection_dir":str(collection_dir),
            "synthetic":False,
            "handoff_mode":"TEST12_EXACT",
            "test12_exact_controls":exact_controls,
            "winner_policy":winner,
            "winner_lock_sha256":terminal.get("winner_lock_sha256"),
            "required_policy_control_ids":required_policy_control_ids,
            "control_redundancy_map":copy.deepcopy(redundancy_map),
            "redundancy_proof_policy":"WINNER_REQUIRED_THEN_CLUSTER_REPRESENTATIVES_THEN_NONREDUNDANT_THEN_ALTERNATES",
            "runtime_profile":runtime_profile,
            "runtime_profile_sha256":runtime_profile.get("profile_sha256"),
            "generation_budget_by_family":copy.deepcopy(
                runtime_profile.get("resolved_generation_budget_by_family") or {}
            ),
            "noise_model":{"global_noise_sigma":EPSILON_NOISE,"families":{}},
            "failures":failures,
            "negative_effects":{"effects":[]},
            "uncertainty":{"unknowns":[]},
            "priority_queue":{"queue":[]},
            "higher_order":{"candidates":[]},
            "ingredient_registry":{"ingredients":[]},
            "pair_graph":{"edges":{}},
            "direction_graph":{"edges":{}},
            "source_observations":[],
            "baselines":{},
            "semantic_identity_contract":{
                "translation_allowed":False,
                "exact_executor":"Test12Campaign.treatment",
                "every_proof_semantic_hash_must_equal_discovery_semantic_hash":True,
            },
        }

    if (run_dir / "test1.2-handoff.json").is_file():
        raise ValueError(
            "Pass the completed Test 1.2 provisional tuning run to Test 2, not the raw Collection run."
        )
    manifest_problems = EvidenceStore(results_root, run_id).verify_manifest()
    if manifest_problems:
        raise ValueError(f"Test-1 evidence manifest verification failed: {manifest_problems}")
    missing = [name for name in REQUIRED_TEST1_FILES if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"Test-1 handoff incomplete; missing: {missing}")

    partitions_payload = _read_json(run_dir / "fixture-partitions.json")
    source_partitions = partitions_payload.get("partitions") or {}
    computed = partition_cases(cases)
    computed_ids = {
        name: {_fixture_id(case) for case in rows}
        for name, rows in computed.items()
    }
    source_ids = {
        name: set(values or [])
        for name, values in source_partitions.items()
    }
    for name in ("DISCOVERY", "VALIDATION", "TEST2_BLIND", "TEST3_PROTECTED"):
        if source_ids.get(name, set()) != computed_ids.get(name, set()):
            raise ValueError(f"fixture partition drift detected for {name}")

    observations = _read_jsonl(run_dir / "test1-observations.jsonl")
    baselines: dict[str, float] = {}
    for row in observations:
        fixture_id = row.get("fixture_id")
        baseline = row.get("baseline_score")
        if (
            isinstance(fixture_id, str)
            and isinstance(baseline, (int, float))
            and not isinstance(baseline, bool)
        ):
            baselines[fixture_id] = float(baseline)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "synthetic": False,
        "test1_plan": _read_json(run_dir / "test1-plan.json"),
        "partitions": source_partitions,
        "noise_model": _read_json(run_dir / "noise-model.json"),
        "ingredient_registry": _read_json(run_dir / "ingredient-registry.json"),
        "priority_queue": _read_json(run_dir / "test2-priority-queue.json"),
        "higher_order": _read_json(run_dir / "higher-order-candidate-queue.json"),
        "failures": _read_json(run_dir / "failure-registry.json"),
        "negative_effects": _read_json(run_dir / "negative-effect-registry.json"),
        "uncertainty": _read_json(run_dir / "uncertainty-ledger.json"),
        "pair_graph": _read_json(run_dir / "pair-interaction-graph.json"),
        "direction_graph": _read_json(run_dir / "directional-order-graph.json"),
        "source_observations": observations,
        "baselines": baselines,
    }


def _treatment_from_summary(value: dict[str, Any]) -> dict[str, Any] | None:
    treatment = value.get("treatment")
    if not isinstance(treatment, dict):
        return None
    ids = treatment.get("ingredient_ids")
    if not isinstance(ids, list) or not ids:
        return None
    valid = {item["id"] for item in INGREDIENTS}
    if any(str(item) not in valid for item in ids):
        return None
    return {
        "ingredient_ids": [str(item) for item in ids],
        "dose": float(treatment.get("dose", 1.0)),
        "representation": str(treatment.get("representation", "prose")),
        "placement": str(treatment.get("placement", "prefix")),
    }


def source_recipes(handoff: dict[str, Any], *, limit: int = 12) -> list[dict[str, Any]]:
    if handoff.get("handoff_mode") == "TEST12_EXACT":
        recipes = [copy.deepcopy(row) for row in handoff.get("test12_exact_controls") or []]
        for recipe in recipes:
            intervention = recipe.get("exact_test12_intervention") or {}
            current_hash = _intervention_fingerprint(intervention)
            if current_hash != recipe.get("discovery_semantic_hash"):
                raise ValueError(
                    f"Test 1.2 control semantic drift before Test 2: {recipe.get('intervention_id')}"
                )
            recipe["proof_semantic_hash"] = current_hash
        return recipes[:limit]

    candidates: list[dict[str, Any]] = []
    for row in (handoff.get("priority_queue") or {}).get("queue", []) or []:
        treatment = _treatment_from_summary(row)
        if treatment is None:
            continue
        candidates.append({
            "source": "test2-priority-queue",
            "source_key": row.get("key"),
            "classification": row.get("classification"),
            "normalized_effect": float(row.get("normalized_effect", 0.0)),
            **treatment,
        })
    for row in (handoff.get("higher_order") or {}).get("candidates", []) or []:
        treatment = _treatment_from_summary(row)
        if treatment is None:
            continue
        candidates.append({
            "source": "higher-order-candidate-queue",
            "source_key": row.get("key"),
            "classification": row.get("classification"),
            "normalized_effect": float(row.get("normalized_effect", 0.0)),
            **treatment,
        })
    for row in (handoff.get("ingredient_registry") or {}).get("ingredients", []) or []:
        discovery = row.get("discovery") or {}
        if discovery.get("classification") not in {"STRONG", "PROMISING", "UNCERTAIN"}:
            continue
        treatment = _treatment_from_summary(discovery)
        if treatment is None:
            continue
        candidates.append({
            "source": "ingredient-registry",
            "source_key": row.get("id"),
            "classification": discovery.get("classification"),
            "normalized_effect": float(discovery.get("normalized_effect", 0.0)),
            **treatment,
        })

    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    rank = {"STRONG": 3, "PROMISING": 2, "UNCERTAIN": 1}
    candidates.sort(
        key=lambda row: (
            rank.get(str(row.get("classification")), 0),
            float(row.get("normalized_effect", 0.0)),
            len(row.get("ingredient_ids") or []),
        ),
        reverse=True,
    )
    for row in candidates:
        key = (
            tuple(row["ingredient_ids"]),
            row["dose"],
            row["representation"],
            row["placement"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
        if len(unique) >= limit:
            break
    if not unique:
        for ingredient in INGREDIENTS[: min(limit, 6)]:
            unique.append({
                "source": "fallback",
                "source_key": ingredient["id"],
                "classification": "UNRESOLVED",
                "normalized_effect": 0.0,
                "ingredient_ids": [ingredient["id"]],
                "dose": 1.0,
                "representation": "prose",
                "placement": "prefix",
            })
    return unique


def _phenotype_id(row: dict[str, Any]) -> str:
    classification = row.get("classification") or {}
    family = str(row.get("family_id") or "unknown")
    result_class = str(classification.get("result_class") or "UNKNOWN")
    digest = hashlib.sha256(f"{family}|{result_class}".encode("utf-8")).hexdigest()[:12]
    return f"FAIL-{digest}"


def _case_index(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_fixture_id(case): case for case in cases}


def _siblings(case: dict[str, Any], cases: list[dict[str, Any]], *, limit: int = 6) -> list[dict[str, Any]]:
    family = _family(case)
    level = int(case.get("difficulty_level", 0))
    peers = [
        row for row in cases
        if _family(row) == family and _fixture_id(row) != _fixture_id(case)
    ]
    peers.sort(
        key=lambda row: (
            abs(int(row.get("difficulty_level", 0)) - level),
            int(row.get("difficulty_level", 0)),
            _fixture_id(row),
        )
    )
    return peers[:limit]


def _safe_percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = fraction * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _timing_seconds(row: dict[str, Any]) -> float | None:
    timing = row.get("timing") or {}
    for key in ("elapsed_s", "total_s", "duration_s", "wall_s"):
        value = timing.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return float(value)
    for key in ("elapsed_ns", "total_ns", "duration_ns"):
        value = timing.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return float(value) / 1_000_000_000.0
    return None


def _spec(
    *,
    sequence: int,
    case: dict[str, Any],
    label: str,
    cfg: dict[str, Any],
    baseline: bool,
    generation_budget: int | None = None,
) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=make_experiment_id(sequence, _fixture_id(case), label),
        parent_experiment_id=None,
        task_id=_fixture_id(case),
        task_family=_family(case),
        difficulty_level=int(case.get("difficulty_level", 0)),
        hypothesis="Test-2 controlled recovery/finalization experiment",
        changed_variable="baseline" if baseline else "prompt_variant",
        thinking_mode=bool(cfg["thinking_mode"]),
        reasoning_effort=cfg.get("reasoning_effort"),
        generation_budget=int(
            generation_budget
            if generation_budget is not None
            else (cfg.get("generation_budget_by_family") or {}).get(
                _family(case),
                cfg["generation_budget"],
            )
        ),
        context_request=None,
        temperature=0.0,
        seed=42,
        prompt_variant="base" if baseline else label,
        recovery_level=None,
    )


class Test2Campaign:
    __test__ = False

    def __init__(
        self,
        runner: Any,
        cases: list[dict[str, Any]],
        handoff: dict[str, Any],
        *,
        clock: Callable[[], float] = time.monotonic,
        started_monotonic: float | None = None,
    ) -> None:
        self.runner = runner
        self.cases = cases
        self.handoff = handoff
        self.cfg = _cfg(runner.config)
        self.clock = clock
        self.start = clock() if started_monotonic is None else float(started_monotonic)
        self.active_end = self.start + ACTIVE_SECONDS
        self.call_start_cutoff = self.start + CALL_START_CUTOFF_SECONDS
        self.partitions = partition_cases(cases)
        self.case_by_id = _case_index(cases)
        self.sequence = 0
        self.rows: list[dict[str, Any]] = []
        self.current_baselines: dict[str, float] = dict(handoff.get("baselines") or {})
        self.current_baseline_validity: dict[str, bool] = {
            str(key): True for key in self.current_baselines
        }
        self.current_baseline_budgets: dict[str, int] = {}
        inherited_budgets = handoff.get("generation_budget_by_family") or {}
        self.generation_budget_by_family = {
            str(family): int(value)
            for family, value in inherited_budgets.items()
        }
        self.noise_sigma = max(
            EPSILON_NOISE,
            float((handoff.get("noise_model") or {}).get("global_noise_sigma", EPSILON_NOISE)),
        )
        self.treatment_calls = 0
        self.harm_evidence: dict[str, Any] = {}
        self.proof_scheduler_audit: dict[str, Any] = {}
        self.recipes = source_recipes(handoff, limit=int(self.cfg["top_recipes"]))
        self.exact_test12: Test12Campaign | None = None
        if handoff.get("handoff_mode") == "TEST12_EXACT":
            source = fresh_model_source(cases)
            source["baselines"] = {}
            exact = Test12Campaign(
                runner,
                cases,
                source,
                clock=clock,
                started_monotonic=self.start,
            )
            exact.active_end = self.active_end
            exact.call_start_cutoff = self.call_start_cutoff
            exact.baseline_generation_budget_by_family = copy.deepcopy(
                self.generation_budget_by_family
            )
            exact_controls = [
                copy.deepcopy(row["exact_test12_intervention"])
                for row in self.recipes
                if row.get("exact_test12_intervention")
            ]
            exact.interventions = exact_controls
            exact.intervention_by_id = {
                str(row["id"]): row for row in exact_controls if row.get("id")
            }
            exact.allowed_partitions = {"DISCOVERY", "VALIDATION"}
            self.exact_test12 = exact

    def can_start(self, deadline: float) -> bool:
        return self.clock() < min(deadline, self.active_end, self.call_start_cutoff)

    def _partition(self, case: dict[str, Any]) -> str:
        fixture_id = _fixture_id(case)
        for name, rows in self.partitions.items():
            if any(_fixture_id(row) == fixture_id for row in rows):
                return name
        return "UNKNOWN"

    def _assert_partition_allowed(self, case: dict[str, Any], *, blind: bool = False) -> None:
        partition = self._partition(case)
        if partition == "TEST3_PROTECTED":
            raise ValueError("Test 2 attempted to expose a Test-3 protected fixture")
        if blind and partition != "TEST2_BLIND":
            raise ValueError("blind confirmation may only use TEST2_BLIND")
        if not blind and partition == "TEST2_BLIND":
            raise ValueError("TEST2_BLIND fixture exposed before blind confirmation")

    def _progress(self, label: str, begin: bool) -> None:
        progress = getattr(self.runner, "progress", None)
        if progress is None:
            return
        if begin:
            if int(getattr(progress, "done", 0)) >= int(getattr(progress, "total_tasks", 0)) - 1:
                old = int(progress.total_tasks)
                progress.total_tasks = old + 128
                self.runner._record_progress(
                    "plan_adjusted",
                    label,
                    old_total=old,
                    new_total=int(progress.total_tasks),
                    reason="wall-clock Test-2 capacity extended",
                )
            self.runner._progress_begin(label)
        else:
            self.runner._progress_complete(label)

    def control(self, case: dict[str, Any], deadline: float, *, phase: str, blind: bool = False, force: bool = False) -> float | None:
        self._assert_partition_allowed(case, blind=blind)
        fixture_id = _fixture_id(case)
        if self.exact_test12 is not None:
            self.exact_test12.allowed_partitions = (
                {"TEST2_BLIND"} if blind else {"DISCOVERY", "VALIDATION"}
            )
            seed = 48 if blind else 42
            record = self.exact_test12.control(
                case,
                deadline,
                seed=seed,
                force=force,
            )
            if record is None:
                return None
            valid = bool(record.get("valid_for_capability"))
            score = float(record.get("score") or 0.0)
            budget = int(
                record.get("generation_budget")
                or self.generation_budget_by_family.get(
                    _family(case), self.cfg["generation_budget"]
                )
            )
            self.current_baseline_validity[fixture_id] = valid
            self.current_baseline_budgets[fixture_id] = budget
            if valid:
                self.current_baselines[fixture_id] = score
            else:
                self.current_baselines.pop(fixture_id, None)
            if force or not any(
                row.get("kind") == "control"
                and row.get("fixture_id") == fixture_id
                and row.get("phase") == phase
                for row in self.rows
            ):
                out = {
                    "schema_version":1,
                    "timestamp_utc":self.runner._utc(),
                    "phase":phase,
                    "kind":"control",
                    "fixture_id":fixture_id,
                    "family_id":_family(case),
                    "difficulty_level":int(case.get("difficulty_level",0)),
                    "partition":self._partition(case),
                    "experiment_id":record.get("experiment_id"),
                    "classification":copy.deepcopy(record.get("classification") or {}),
                    "valid_for_capability":valid,
                    "baseline_valid_for_capability":valid,
                    "baseline_generation_budget":budget,
                    "treatment_generation_budget":budget,
                    "budget_comparison_valid":True,
                    "delta_valid":False,
                    "score":score,
                    "baseline_score":score,
                    "delta":None,
                    "recipe":{"exact_test12_control":"CONTROL"},
                    "source_key":None,
                    "timing":{"wall_s":float((record.get("timing") or {}).get("client_latency_ns") or 0)/1_000_000_000.0},
                    "evidence_refs":{},
                }
                self.rows.append(out)
                self.runner.store.append_jsonl("test2-observations.jsonl", out)
            return score if valid else None

        if not force and fixture_id in self.current_baselines:
            return self.current_baselines[fixture_id]
        if not self.can_start(deadline):
            return None

        family = _family(case)
        budget = int(
            self.generation_budget_by_family.get(
                family,
                self.cfg["generation_budget"],
            )
        )
        self.sequence += 1
        spec = _spec(
            sequence=self.sequence,
            case=case,
            label=f"{phase}-control-b{budget}",
            cfg=self.cfg,
            baseline=True,
            generation_budget=budget,
        )
        label = f"test2 {phase} control {fixture_id} b{budget}"
        self._progress(label, True)
        try:
            row = execute_experiment(self.runner, case, spec, parent=None)
        finally:
            self._progress(label, False)

        score = row.get("score")
        valid = (row.get("classification") or {}).get("valid_for_capability") is True
        numeric = (
            float(score)
            if valid and isinstance(score, (int, float)) and not isinstance(score, bool)
            else 0.0
        )
        self.current_baseline_validity[fixture_id] = bool(valid)
        self.current_baseline_budgets[fixture_id] = int(budget)
        if valid:
            self.current_baselines[fixture_id] = numeric
        else:
            self.current_baselines.pop(fixture_id, None)

        self._record(
            case,
            row,
            phase=phase,
            kind="control",
            recipe={
                "ingredient_ids": [],
                "dose": 0.0,
                "representation": "none",
                "placement": "none",
            },
            baseline_score=numeric,
            baseline_valid=bool(valid),
            baseline_budget=int(budget),
            source_key=None,
        )
        return numeric if valid else None

    def treatment(
        self,
        case: dict[str, Any],
        deadline: float,
        *,
        phase: str,
        kind: str,
        recipe: dict[str, Any],
        label: str,
        source_key: str | None = None,
        blind: bool = False,
        generation_budget_override: int | None = None,
    ) -> dict[str, Any] | None:
        self._assert_partition_allowed(case, blind=blind)
        if not self.can_start(deadline):
            return None

        if self.exact_test12 is not None:
            exact_recipe = recipe.get("exact_test12_intervention")
            if not isinstance(exact_recipe, dict):
                return None
            discovery_hash = str(recipe.get("discovery_semantic_hash") or "")
            current_hash = _intervention_fingerprint(exact_recipe)
            if discovery_hash and current_hash != discovery_hash:
                raise ValueError(
                    f"Test 2 semantic drift for exact control {exact_recipe.get('id')}"
                )
            proof_seed = int(
                recipe.get("_proof_seed")
                or {
                    "recurrence_higher_order":42,
                    "failure_recovery":45,
                    "negative_transfer":46,
                    "purple_unicorn":47,
                    "knockout_distillation":49,
                    "blind_confirmation":50,
                    "censoring_cost_tradeoff":51,
                }.get(phase, 42)
            )
            intervention = copy.deepcopy(exact_recipe)
            own_budget_probe = generation_budget_override is not None
            if own_budget_probe:
                intervention["generation_budget"] = int(generation_budget_override)
            proof_hash = _intervention_fingerprint(intervention)
            exact_semantic_match = proof_hash == current_hash

            self.exact_test12.allowed_partitions = (
                {"TEST2_BLIND"} if blind else {"DISCOVERY", "VALIDATION"}
            )
            exact_row = self.exact_test12.treatment(
                case,
                deadline,
                phase=f"test2-{phase}",
                intervention=intervention,
                seed=proof_seed,
            )
            if exact_row is None:
                return None

            classification = copy.deepcopy(exact_row.get("classification") or {})
            valid = bool(exact_row.get("valid_for_capability"))
            delta_valid = exact_row.get("delta_valid") is True and not own_budget_probe
            result_class = str(classification.get("result_class") or "")
            censored = bool(
                not valid
                and result_class in CENSORING_CLASSES
            )
            record = {
                "schema_version":1,
                "timestamp_utc":self.runner._utc(),
                "phase":phase,
                "kind":kind,
                "fixture_id":_fixture_id(case),
                "family_id":_family(case),
                "difficulty_level":int(case.get("difficulty_level",0)),
                "partition":self._partition(case),
                "experiment_id":exact_row.get("experiment_id"),
                "classification":classification,
                "valid_for_capability":valid,
                "baseline_valid_for_capability":bool(
                    exact_row.get("control_valid_for_capability")
                ),
                "baseline_generation_budget":int(
                    exact_row.get("control_generation_budget")
                    or self.generation_budget_by_family.get(
                        _family(case), self.cfg["generation_budget"]
                    )
                ),
                "treatment_generation_budget":int(
                    exact_row.get("generation_budget")
                    or self.cfg["generation_budget"]
                ),
                "budget_comparison_valid":bool(
                    exact_row.get("budget_comparison_valid")
                ),
                "delta_valid":bool(delta_valid),
                "score":float(exact_row.get("score") or 0.0),
                "baseline_score":float(exact_row.get("control_score") or 0.0),
                "delta":(
                    float(exact_row["delta"])
                    if delta_valid and exact_row.get("delta") is not None
                    else None
                ),
                "recipe":copy.deepcopy(recipe),
                "source_key":source_key,
                "proof_seed":proof_seed,
                "semantic_hash_discovery":current_hash,
                "semantic_hash_proof":proof_hash,
                "semantic_hash_match":exact_semantic_match,
                "own_budget_cost_probe":own_budget_probe,
                "censored_for_capability":censored,
                "censoring_class":(
                    "CONTROL_EXCEEDS_BASELINE_BUDGET" if censored else None
                ),
                "timing":{"wall_s":float((exact_row.get("cost") or {}).get("wall_seconds") or 0.0)},
                "evidence_refs":copy.deepcopy(exact_row.get("evidence_refs") or {}),
            }
            self.rows.append(record)
            self.runner.store.append_jsonl("test2-observations.jsonl", record)
            self.treatment_calls += 1
            return record

        fixture_id = _fixture_id(case)
        baseline = self.current_baselines.get(fixture_id)
        if baseline is None:
            baseline = self.control(case, deadline, phase=phase, blind=blind)
        elif self.treatment_calls and self.treatment_calls % int(self.cfg["control_interval"]) == 0:
            refreshed = self.control(case, deadline, phase=phase, blind=blind, force=True)
            if refreshed is not None:
                baseline = refreshed
        if (
            baseline is None
            or not self.current_baseline_validity.get(fixture_id, False)
            or not self.can_start(deadline)
        ):
            return None

        ids = [str(value) for value in recipe.get("ingredient_ids") or []]
        if not ids:
            return None
        self.sequence += 1
        baseline_budget = int(
            self.current_baseline_budgets.get(
                fixture_id,
                self.generation_budget_by_family.get(
                    _family(case),
                    self.cfg["generation_budget"],
                ),
            )
        )
        treatment_budget = int(
            generation_budget_override
            if generation_budget_override is not None
            else baseline_budget
        )
        spec = _spec(
            sequence=self.sequence,
            case=case,
            label=label,
            cfg=self.cfg,
            baseline=False,
            generation_budget=treatment_budget,
        )
        messages = build_treatment_messages(
            case,
            ids,
            dose=float(recipe.get("dose", 1.0)),
            representation=str(recipe.get("representation", "prose")),
            placement=str(recipe.get("placement", "prefix")),
        )
        progress_label = f"test2 {phase} {fixture_id} {label}"
        self._progress(progress_label, True)
        try:
            row = execute_experiment(
                self.runner,
                case,
                spec,
                parent=None,
                messages_override=messages,
            )
        finally:
            self._progress(progress_label, False)
        self.treatment_calls += 1
        return self._record(
            case,
            row,
            phase=phase,
            kind=kind,
            recipe=recipe,
            baseline_score=float(baseline),
            baseline_valid=True,
            baseline_budget=baseline_budget,
            source_key=source_key,
        )

    def _record(
        self,
        case: dict[str, Any],
        row: dict[str, Any],
        *,
        phase: str,
        kind: str,
        recipe: dict[str, Any],
        baseline_score: float,
        baseline_valid: bool,
        baseline_budget: int,
        source_key: str | None,
    ) -> dict[str, Any]:
        score = row.get("score")
        valid = (row.get("classification") or {}).get("valid_for_capability") is True
        numeric = (
            float(score)
            if valid and isinstance(score, (int, float)) and not isinstance(score, bool)
            else 0.0
        )
        treatment_budget = int(
            (row.get("experiment") or {}).get("generation_budget")
            or self.cfg["generation_budget"]
        )
        budget_valid = treatment_budget == int(baseline_budget)
        result_class = str((row.get("classification") or {}).get("result_class") or "")
        censored_for_capability = bool(
            baseline_valid
            and budget_valid
            and not valid
            and result_class in CENSORING_CLASSES
        )
        delta_valid = bool(valid and baseline_valid and budget_valid)
        record = {
            "schema_version": 1,
            "timestamp_utc": self.runner._utc(),
            "phase": phase,
            "kind": kind,
            "fixture_id": _fixture_id(case),
            "family_id": _family(case),
            "difficulty_level": int(case.get("difficulty_level", 0)),
            "partition": self._partition(case),
            "experiment_id": (row.get("experiment") or {}).get("experiment_id"),
            "classification": copy.deepcopy(row.get("classification") or {}),
            "valid_for_capability": bool(valid),
            "baseline_valid_for_capability": bool(baseline_valid),
            "baseline_generation_budget": int(baseline_budget),
            "treatment_generation_budget": treatment_budget,
            "budget_comparison_valid": bool(budget_valid),
            "delta_valid": bool(delta_valid),
            "censored_for_capability": bool(censored_for_capability),
            "censoring_class": (
                "CONTROL_EXCEEDS_BASELINE_BUDGET"
                if censored_for_capability
                else None
            ),
            "score": numeric,
            "baseline_score": baseline_score,
            "delta": (numeric - baseline_score) if delta_valid else None,
            "recipe": copy.deepcopy(recipe),
            "source_key": source_key,
            "timing": copy.deepcopy(row.get("timing") or {}),
            "evidence_refs": copy.deepcopy(row.get("evidence_refs") or {}),
        }
        self.rows.append(record)
        self.runner.store.append_jsonl("test2-observations.jsonl", record)
        return record


def _positive_sign_test_p(deltas: list[float]) -> float:
    """Exact one-sided sign test after fixture clustering; ties carry no vote."""
    nonzero = [value for value in deltas if value != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    wins = sum(1 for value in nonzero if value > 0.0)
    return min(
        1.0,
        sum(math.comb(n, k) for k in range(wins, n + 1)) / (2.0 ** n),
    )


def _bh_qvalues(p_values: dict[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg FDR q-values, monotone over ranked p-values."""
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    m = len(ordered)
    adjusted = [0.0] * m
    running = 1.0
    for index in range(m - 1, -1, -1):
        rank = index + 1
        value = min(1.0, ordered[index][1] * m / rank)
        running = min(running, value)
        adjusted[index] = running
    return {
        key: adjusted[index]
        for index, (key, _p) in enumerate(ordered)
    }


def _effect_map(
    rows: Iterable[dict[str, Any]],
    *,
    noise_sigma: float,
    bootstrap_samples: int,
    key_fn: Callable[[dict[str, Any]], str],
    max_censoring_rate: float = 0.20,
    fdr_level: float = 0.10,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    samples: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("kind") == "control":
            continue
        key = key_fn(row)
        grouped[key].append(row)
        samples.setdefault(key, copy.deepcopy(row.get("recipe") or {}))

    result: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {}
    for key, values in grouped.items():
        valid = [
            row for row in values
            if row.get("delta_valid") is True and row.get("delta") is not None
        ]
        censored = [
            row for row in values
            if row.get("censored_for_capability") is True
        ]

        by_fixture: dict[str, list[float]] = defaultdict(list)
        for index, row in enumerate(valid):
            fixture_id = str(
                row.get("fixture_id")
                or row.get("experiment_id")
                or f"__row_{index}"
            )
            by_fixture[fixture_id].append(float(row["delta"]))
        fixture_deltas = [
            float(median(values_for_fixture))
            for _, values_for_fixture in sorted(by_fixture.items())
        ]

        if fixture_deltas:
            summary = effect_summary(
                fixture_deltas,
                noise_sigma,
                bootstrap_samples=bootstrap_samples,
                seed_key=f"test2:fixture-cluster:{key}",
            )
        else:
            summary = {
                "n":0,
                "median_delta":None,
                "normalized_effect":0.0,
                "classification":"NO_VALID_EFFECT",
            }

        censoring_rate = len(censored) / len(values) if values else 0.0
        p_value = _positive_sign_test_p(fixture_deltas)
        p_values[key] = p_value
        summary["raw_n"] = len(values)
        summary["raw_valid_n"] = len(valid)
        summary["valid_n"] = len(valid)
        summary["independent_fixture_n"] = len(fixture_deltas)
        summary["unit_of_independence"] = "fixture"
        summary["fixture_ids"] = sorted(by_fixture)
        summary["censored_n"] = len(censored)
        summary["censoring_rate"] = censoring_rate
        summary["requires_own_budget_cost_probe"] = bool(censored)
        summary["positive_sign_test_p_value"] = p_value
        if censoring_rate > float(max_censoring_rate):
            summary["classification_before_censoring"] = summary.get("classification")
            summary["classification"] = "CENSORING_DOMINATED"
        summary["recipe"] = samples[key]
        result[key] = summary

    q_values = _bh_qvalues(p_values)
    for key, summary in result.items():
        q_value = q_values.get(key, 1.0)
        summary["bh_fdr_q_value"] = q_value
        summary["bh_fdr_level"] = float(fdr_level)
        summary["multiple_comparisons_method"] = "BENJAMINI_HOCHBERG"
        if (
            summary.get("classification") in {"STRONG", "PROMISING"}
            and q_value > float(fdr_level)
        ):
            summary["classification_before_multiplicity"] = summary["classification"]
            summary["classification"] = "UNCERTAIN_MULTIPLICITY"
    return result


def _recipe_key(recipe: dict[str, Any]) -> str:
    if recipe.get("exact_locked_policy") is not None:
        payload = json.dumps(
            recipe["exact_locked_policy"],
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return "POLICY-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    if recipe.get("exact_test12_intervention") is not None:
        semantic_hash = str(
            recipe.get("discovery_semantic_hash")
            or _intervention_fingerprint(recipe["exact_test12_intervention"])
        )
        return "TEST12-" + semantic_hash[:16]
    ids = [str(value) for value in recipe.get("ingredient_ids") or []]
    return "->".join(ids)


def _recurrence_variants(recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if recipes and any(recipe.get("exact_test12_intervention") is not None for recipe in recipes):
        result = []
        for recipe in recipes:
            if recipe.get("exact_test12_intervention") is None:
                continue
            for seed in (42, 43, 44):
                variant = copy.deepcopy(recipe)
                variant["_proof_seed"] = seed
                result.append(variant)
        return result

    result: list[dict[str, Any]] = []
    for recipe in recipes:
        ids = list(recipe["ingredient_ids"])
        base = {
            "dose": recipe.get("dose", 1.0),
            "representation": recipe.get("representation", "prose"),
            "placement": recipe.get("placement", "prefix"),
        }
        variants: list[list[str]] = [ids]
        if len(ids) == 1:
            a = ids[0]
            variants.extend([[a, a], [a, a, a]])
        else:
            a, b = ids[0], ids[1]
            variants.extend([[a, b, a], [a, b, a, b], [b, a, b]])
        for variant in variants:
            result.append({**base, "ingredient_ids": variant})
    strong_ids = []
    for recipe in recipes:
        for ingredient in recipe["ingredient_ids"]:
            if ingredient not in strong_ids:
                strong_ids.append(ingredient)
    for a, b, c in itertools.combinations(strong_ids[:6], 3):
        for variant in ([a, b, c], [a, c, b], [b, a, c]):
            result.append({
                "ingredient_ids": list(variant),
                "dose": 1.0,
                "representation": "prose",
                "placement": "prefix",
            })
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for recipe in result:
        key = (
            tuple(recipe["ingredient_ids"]),
            recipe["dose"],
            recipe["representation"],
            recipe["placement"],
        )
        unique.setdefault(key, recipe)
    return list(unique.values())


def _recurrence_candidate_status(
    campaign: Test2Campaign,
    recipe: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    key = _recipe_key(recipe)
    values = [
        row for row in rows
        if _recipe_key(row.get("recipe") or {}) == key
    ]
    valid = [
        row for row in values
        if row.get("delta_valid") is True and row.get("delta") is not None
    ]
    fixtures = sorted({str(row.get("fixture_id")) for row in valid})
    seeds = sorted({
        int(
            row.get("proof_seed")
            or (row.get("recipe") or {}).get("_proof_seed")
            or 42
        )
        for row in valid
    })
    censored = [
        row for row in values
        if row.get("censored_for_capability") is True
    ]
    deltas = [float(row["delta"]) for row in valid]

    min_fixtures = int(campaign.cfg["recurrence_min_independent_fixtures"])
    min_seeds = int(campaign.cfg["recurrence_min_distinct_seeds"])
    target_valid = int(campaign.cfg["recurrence_target_valid_observations"])
    max_valid = int(campaign.cfg["recurrence_max_valid_observations"])

    coverage_ready = bool(
        len(fixtures) >= min_fixtures
        and len(seeds) >= min_seeds
        and len(valid) >= target_valid
    )
    all_positive = bool(deltas) and all(value > 0.0 for value in deltas)
    all_nonpositive = bool(deltas) and all(value <= 0.0 for value in deltas)
    mixed = bool(deltas) and not all_positive and not all_nonpositive

    if coverage_ready and all_positive:
        settled = True
        reason = "CONSISTENT_POSITIVE_MINIMUM_PROOF_MET"
    elif coverage_ready and all_nonpositive:
        settled = True
        reason = "CONSISTENT_NONPOSITIVE_MINIMUM_PROOF_MET"
    elif len(valid) >= max_valid:
        settled = True
        reason = "MAX_PROOF_REACHED_WITH_MIXED_EFFECT"
    else:
        settled = False
        reason = "MORE_PROOF_VALUE_AVAILABLE"

    required_ids = {
        str(value)
        for value in campaign.handoff.get("required_policy_control_ids") or []
    }
    ident = str(recipe.get("intervention_id") or "")
    required = ident in required_ids

    fixture_debt = max(0, min_fixtures - len(fixtures))
    seed_debt = max(0, min_seeds - len(seeds))
    valid_debt = max(0, target_valid - len(valid))
    censor_rate = len(censored) / len(values) if values else 0.0
    priority = (
        (1000.0 if required else 0.0)
        + 100.0 * fixture_debt
        + 40.0 * seed_debt
        + 10.0 * valid_debt
        + 25.0 * censor_rate
        + (30.0 if mixed else 0.0)
    )
    if settled:
        priority = -1.0

    return {
        "recipe_key":key,
        "intervention_id":ident or None,
        "required_by_frozen_policy":required,
        "attempts":len(values),
        "valid_observations":len(valid),
        "independent_fixture_count":len(fixtures),
        "distinct_seed_count":len(seeds),
        "fixture_ids":fixtures,
        "proof_seeds":seeds,
        "censored_observations":len(censored),
        "censoring_rate":censor_rate,
        "positive_observations":sum(1 for value in deltas if value > 0.0),
        "negative_observations":sum(1 for value in deltas if value < 0.0),
        "null_observations":sum(1 for value in deltas if value == 0.0),
        "coverage_ready":coverage_ready,
        "settled":settled,
        "settled_reason":reason,
        "proof_priority":priority,
    }


def _next_recurrence_task(
    campaign: Test2Campaign,
    recipe: dict[str, Any],
    fixtures: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], int] | None:
    key = _recipe_key(recipe)
    existing = [
        row for row in rows
        if _recipe_key(row.get("recipe") or {}) == key
    ]
    attempted = {
        (
            str(row.get("fixture_id") or ""),
            int(
                row.get("proof_seed")
                or (row.get("recipe") or {}).get("_proof_seed")
                or 42
            ),
        )
        for row in existing
    }
    valid_fixture_ids = {
        str(row.get("fixture_id") or "")
        for row in existing
        if row.get("delta_valid") is True
        and row.get("delta") is not None
    }
    valid_seeds = {
        int(
            row.get("proof_seed")
            or (row.get("recipe") or {}).get("_proof_seed")
            or 42
        )
        for row in existing
        if row.get("delta_valid") is True
        and row.get("delta") is not None
    }
    seed_order = sorted(
        (42, 43, 44),
        key=lambda seed: (seed in valid_seeds, seed),
    )
    fixture_order = sorted(
        fixtures,
        key=lambda case: (
            _fixture_id(case) in valid_fixture_ids,
            -int(case.get("difficulty_level") or 0),
            _family(case),
            _fixture_id(case),
        ),
    )
    for case in fixture_order:
        for seed in seed_order:
            if (_fixture_id(case), seed) not in attempted:
                return case, seed
    return None


def phase_recurrence(campaign: Test2Campaign, deadline: float) -> dict[str, dict[str, Any]]:
    fixtures = _balanced_cases(
        campaign.partitions["VALIDATION"],
        len(campaign.partitions["VALIDATION"]),
    )
    if not fixtures:
        fixtures = _balanced_cases(
            campaign.partitions["DISCOVERY"],
            len(campaign.partitions["DISCOVERY"]),
        )

    exact_mode = bool(
        campaign.recipes
        and any(
            recipe.get("exact_test12_intervention") is not None
            for recipe in campaign.recipes
        )
    )
    if not exact_mode:
        variants = _recurrence_variants(campaign.recipes)
        cursor = 0
        while variants and fixtures and campaign.can_start(deadline):
            recipe = variants[cursor % len(variants)]
            case = fixtures[(cursor // len(variants)) % len(fixtures)]
            campaign.treatment(
                case,
                deadline,
                phase="recurrence_higher_order",
                kind="recurrence",
                recipe=recipe,
                label="recurrence-" + _recipe_key(recipe),
                source_key="test1-promoted",
            )
            cursor += 1
            if cursor >= len(variants) * len(fixtures):
                break
    else:
        recipes = [
            copy.deepcopy(recipe)
            for recipe in campaign.recipes
            if recipe.get("exact_test12_intervention") is not None
        ]
        exhausted: set[str] = set()
        while recipes and fixtures and campaign.can_start(deadline):
            phase_rows = [
                row for row in campaign.rows
                if row.get("phase") == "recurrence_higher_order"
            ]
            statuses = {
                _recipe_key(recipe): _recurrence_candidate_status(
                    campaign, recipe, phase_rows
                )
                for recipe in recipes
            }
            available = [
                recipe for recipe in recipes
                if not statuses[_recipe_key(recipe)]["settled"]
                and _recipe_key(recipe) not in exhausted
            ]
            if not available:
                break
            available.sort(
                key=lambda recipe: (
                    float(statuses[_recipe_key(recipe)]["proof_priority"]),
                    -int(statuses[_recipe_key(recipe)]["attempts"]),
                    _recipe_key(recipe),
                ),
                reverse=True,
            )
            recipe = copy.deepcopy(available[0])
            task = _next_recurrence_task(
                campaign, recipe, fixtures, phase_rows
            )
            if task is None:
                exhausted.add(_recipe_key(recipe))
                continue
            case, seed = task
            recipe["_proof_seed"] = int(seed)
            observation = campaign.treatment(
                case,
                deadline,
                phase="recurrence_higher_order",
                kind="recurrence",
                recipe=recipe,
                label="recurrence-" + _recipe_key(recipe),
                source_key="proof-value-scheduler",
            )
            if observation is None:
                exhausted.add(_recipe_key(recipe))

        final_rows = [
            row for row in campaign.rows
            if row.get("phase") == "recurrence_higher_order"
        ]
        final_statuses = {
            _recipe_key(recipe): _recurrence_candidate_status(
                campaign, recipe, final_rows
            )
            for recipe in recipes
        }
        campaign.proof_scheduler_audit = {
            "schema_version":1,
            "phase":"recurrence_higher_order",
            "mode":"ADAPTIVE_PROOF_VALUE",
            "objective":"spend fixed proof clock on independent fixture/seed debt and unresolved decisions",
            "candidate_count":len(recipes),
            "settled_candidate_count":sum(
                1 for row in final_statuses.values()
                if row.get("settled") is True
            ),
            "exhausted_candidate_keys":sorted(exhausted),
            "minimum_independent_fixtures":int(
                campaign.cfg["recurrence_min_independent_fixtures"]
            ),
            "minimum_distinct_seeds":int(
                campaign.cfg["recurrence_min_distinct_seeds"]
            ),
            "target_valid_observations":int(
                campaign.cfg["recurrence_target_valid_observations"]
            ),
            "maximum_valid_observations":int(
                campaign.cfg["recurrence_max_valid_observations"]
            ),
            "candidates":final_statuses,
        }

    effects = _effect_map(
        [row for row in campaign.rows if row["phase"] == "recurrence_higher_order"],
        noise_sigma=campaign.noise_sigma,
        bootstrap_samples=int(campaign.cfg["bootstrap_samples"]),
        max_censoring_rate=float(campaign.cfg["max_effect_censoring_rate"]),
        key_fn=lambda row: _recipe_key(row["recipe"]),
    )
    for key, summary in effects.items():
        status = (campaign.proof_scheduler_audit.get("candidates") or {}).get(key)
        if status:
            summary["proof_scheduler_status"] = copy.deepcopy(status)
    return effects


def _recovery_candidates(campaign: Test2Campaign, case: dict[str, Any]) -> list[dict[str, Any]]:
    if campaign.handoff.get("handoff_mode") == "TEST12_EXACT":
        result = []
        for index, recipe in enumerate(campaign.recipes[: int(campaign.cfg["max_recovery_recipes"])]):
            exact = copy.deepcopy(recipe)
            exact["_proof_seed"] = 45 + (index % 3)
            result.append(exact)
        return result

    family = _family(case)
    result: list[dict[str, Any]] = []
    for recipe in campaign.recipes[:4]:
        result.append(copy.deepcopy(recipe))
        combined = copy.deepcopy(recipe)
        combined["ingredient_ids"] = list(recipe["ingredient_ids"]) + ["ING-003"]
        result.append(combined)

    domain = []
    if "tool" in family:
        domain.append("ING-011")
    if "arithmetic" in family or "algebra" in family or "quantitative" in family:
        domain.append("ING-012")
    if "logic" in family or "deduction" in family:
        domain.append("ING-013")
    if "code" in family or "debug" in family:
        domain.append("ING-014")
    if "context" in family or "state" in family or "contradict" in family:
        domain.extend(["ING-005", "ING-006"])
    if not domain:
        domain.extend(["ING-001", "ING-003"])

    result.extend([
        {
            "ingredient_ids": domain,
            "dose": 1.0,
            "representation": "prose",
            "placement": "prefix",
        },
        {
            "ingredient_ids": ["ING-001", "ING-003", *domain],
            "dose": 1.0,
            "representation": "bullets",
            "placement": "prefix",
        },
    ])
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for recipe in result:
        ids = tuple(recipe.get("ingredient_ids") or [])
        if ids:
            unique.setdefault(ids, recipe)
    return list(unique.values())[: int(campaign.cfg["max_recovery_recipes"])]


def phase_failure_recovery(campaign: Test2Campaign, deadline: float) -> dict[str, Any]:
    source_failures = list((campaign.handoff.get("failures") or {}).get("failures", []) or [])
    eligible = [
        row for row in source_failures
        if row.get("partition") in {None, "DISCOVERY", "VALIDATION"}
        and isinstance(row.get("fixture_id"), str)
        and row.get("fixture_id") in campaign.case_by_id
    ]
    if not eligible:
        # Treat current unresolved/negative Test-1 branches as recovery targets.
        eligible = [
            {
                "fixture_id": _fixture_id(case),
                "family_id": _family(case),
                "classification": {"result_class": "UNRESOLVED"},
                "partition": "VALIDATION",
            }
            for case in _balanced_cases(campaign.partitions["VALIDATION"], 24)
        ]

    attempts: list[dict[str, Any]] = []
    cursor = 0
    while eligible and campaign.can_start(deadline):
        failure = eligible[cursor % len(eligible)]
        case = campaign.case_by_id[str(failure["fixture_id"])]
        sibling_cases = [case, *_siblings(case, campaign.partitions["DISCOVERY"] + campaign.partitions["VALIDATION"], limit=5)]
        recipes = _recovery_candidates(campaign, case)
        recipe = recipes[(cursor // len(eligible)) % len(recipes)]
        sibling = sibling_cases[(cursor // max(1, len(eligible) * len(recipes))) % len(sibling_cases)]
        observation = campaign.treatment(
            sibling,
            deadline,
            phase="failure_recovery",
            kind="recovery",
            recipe=recipe,
            label="recovery-" + _recipe_key(recipe),
            source_key=str(failure.get("experiment_id") or failure.get("fixture_id")),
        )
        if observation is not None:
            observation["source_failure_phenotype"] = _phenotype_id(failure)
            attempts.append(observation)
        cursor += 1
        budget = len(eligible) * max(1, len(recipes)) * max(1, len(sibling_cases))
        if cursor >= budget:
            break

    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        by_key[(str(row.get("source_failure_phenotype")), _recipe_key(row["recipe"]))].append(row)

    matrix: dict[str, Any] = {}
    phenotype_registry: dict[str, Any] = {}
    partial = float(campaign.cfg["partial_recovery_threshold"])
    general = float(campaign.cfg["general_recovery_threshold"])
    for (phenotype, recipe_key), rows in by_key.items():
        valid_rows = [
            row for row in rows
            if (row.get("classification") or {}).get("valid_for_capability") is True
        ]
        success = [
            row for row in valid_rows
            if (row.get("classification") or {}).get("result_class") == "ANSWER_CORRECT"
        ]
        rate = len(success) / len(valid_rows) if valid_rows else 0.0
        if rate >= general and len(valid_rows) >= 3:
            status = "GENERAL_RECOVERY"
        elif rate >= partial and len(valid_rows) >= 2:
            status = "PARTIAL_RECOVERY"
        elif success:
            status = "FIXTURE_PATCH"
        else:
            status = "FAILED_RECOVERY"
        key = f"{phenotype}|{recipe_key}"
        matrix[key] = {
            "phenotype_id": phenotype,
            "recipe_key": recipe_key,
            "recipe": copy.deepcopy(rows[0]["recipe"]),
            "attempts": len(rows),
            "valid_attempts": len(valid_rows),
            "successful_attempts": len(success),
            "success_rate": rate,
            "status": status,
            "fixture_ids": sorted({_fixture_id(campaign.case_by_id[row["fixture_id"]]) for row in rows}),
            "experiment_ids": [
                str(row.get("experiment_id"))
                for row in rows
                if row.get("experiment_id")
            ],
            "source_failure_refs": sorted({
                str(row.get("source_key"))
                for row in rows
                if row.get("source_key")
            }),
        }
        family = rows[0].get("family_id")
        phenotype_registry.setdefault(phenotype, {
            "phenotype_id": phenotype,
            "family_id": family,
            "result_classes": sorted({
                str((row.get("classification") or {}).get("result_class"))
                for row in rows
            }),
            "attempt_count": 0,
            "recovery_keys": [],
        })
        phenotype_registry[phenotype]["attempt_count"] += len(rows)
        phenotype_registry[phenotype]["recovery_keys"].append(key)

    return {
        "phenotypes": phenotype_registry,
        "matrix": matrix,
    }


def phase_negative_transfer(
    campaign: Test2Campaign,
    deadline: float,
    recurrence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Adversarial harm-seeking on independently passing baseline sentinels."""
    ranked = sorted(
        recurrence.items(),
        key=lambda item: (
            item[1].get("classification") == "STRONG",
            item[1].get("classification") == "PROMISING",
            float(item[1].get("normalized_effect", 0.0)),
        ),
        reverse=True,
    )
    recipes = [
        copy.deepcopy(summary["recipe"])
        for _, summary in ranked
        if summary.get("classification") in {"STRONG", "PROMISING", "UNCERTAIN"}
    ][: int(campaign.cfg["negative_transfer_recipes"])]

    if campaign.handoff.get("handoff_mode") == "TEST12_EXACT":
        required_ids = [
            str(value)
            for value in campaign.handoff.get("required_policy_control_ids") or []
        ]
        required = [
            copy.deepcopy(recipe)
            for ident in required_ids
            for recipe in campaign.recipes
            if str(recipe.get("intervention_id") or "") == ident
        ]
        seen = {_recipe_key(recipe) for recipe in required}
        exploratory = [
            recipe for recipe in recipes
            if _recipe_key(recipe) not in seen
        ]
        recipes = [*required, *exploratory]
    if not recipes:
        recipes = [copy.deepcopy(row) for row in campaign.recipes[:4]]

    # Build a deterministic baseline-pass sentinel pool. Failing baselines are
    # not informative for "did the control break something that already worked?"
    validation = _balanced_cases(
        campaign.partitions["VALIDATION"],
        len(campaign.partitions["VALIDATION"]),
    )
    sentinels: list[dict[str, Any]] = []
    for case in validation:
        if not campaign.can_start(deadline):
            break
        baseline = campaign.control(
            case,
            deadline,
            phase="negative_transfer_sentinel",
        )
        if baseline is not None and baseline >= 1.0:
            sentinels.append(case)

    min_trials = int(campaign.cfg["harm_sentinel_min_per_control"])
    min_families = int(campaign.cfg["harm_min_distinct_families"])
    max_break_rate = float(campaign.cfg["harm_max_break_rate"])

    for recipe in recipes:
        if not campaign.can_start(deadline):
            break
        key = _recipe_key(recipe)
        valid_rows: list[dict[str, Any]] = []
        attempted_rows: list[dict[str, Any]] = []
        for case in sentinels:
            if not campaign.can_start(deadline):
                break
            row = campaign.treatment(
                case,
                deadline,
                phase="negative_transfer",
                kind="harm_probe",
                recipe=recipe,
                label="harm-" + key,
                source_key="baseline-pass-sentinel",
            )
            if row is None:
                continue
            attempted_rows.append(row)
            if row.get("delta_valid") is True and row.get("delta") is not None:
                valid_rows.append(row)

            family_count = len({str(r.get("family_id")) for r in valid_rows})
            if len(valid_rows) >= min_trials and family_count >= min_families:
                break

        breaks = [
            row for row in valid_rows
            if float(row["delta"]) < 0.0
        ]
        families = sorted({str(row.get("family_id")) for row in valid_rows})
        censoring = [
            row for row in attempted_rows
            if row.get("censored_for_capability") is True
        ]
        break_rate = len(breaks) / len(valid_rows) if valid_rows else None
        break_ci = _wilson90(len(breaks), len(valid_rows))
        sufficient = bool(
            len(valid_rows) >= min_trials
            and len(families) >= min_families
        )
        harm_safe = bool(
            sufficient
            and break_rate is not None
            and break_ci[1] <= max_break_rate
        )
        harm_ci_definitively_bad = bool(
            sufficient and break_ci[0] > max_break_rate
        )
        campaign.harm_evidence[key] = {
            "recipe_key":key,
            "recipe":copy.deepcopy(recipe),
            "attempts":len(attempted_rows),
            "valid_baseline_pass_sentinels":len(valid_rows),
            "distinct_families":families,
            "distinct_family_count":len(families),
            "breaks":len(breaks),
            "break_rate":break_rate,
            "break_rate_wilson90":break_ci,
            "harm_ci_upper_bound":break_ci[1],
            "censored_observations":len(censoring),
            "censoring_rate":(
                len(censoring)/len(attempted_rows)
                if attempted_rows else None
            ),
            "minimum_valid_sentinels_required":min_trials,
            "minimum_distinct_families_required":min_families,
            "maximum_break_rate":max_break_rate,
            "harm_evidence_sufficient":sufficient,
            "harm_safe":harm_safe,
            "verification_status":(
                "HARM_SAFE"
                if harm_safe
                else "HARMFUL"
                if harm_ci_definitively_bad
                else "INSUFFICIENT_HARM_EVIDENCE"
            ),
        }

    effects = _effect_map(
        [row for row in campaign.rows if row["phase"] == "negative_transfer"],
        noise_sigma=campaign.noise_sigma,
        bootstrap_samples=int(campaign.cfg["bootstrap_samples"]),
        max_censoring_rate=float(campaign.cfg["max_effect_censoring_rate"]),
        key_fn=lambda row: f"{_recipe_key(row['recipe'])}|family={row['family_id']}",
    )
    boundaries: dict[str, Any] = {}
    for key, summary in effects.items():
        if summary["classification"] == "HARMFUL":
            label = "NEGATIVE_TRANSFER"
        elif summary["classification"] in {"STRONG", "PROMISING"}:
            label = "DOMAIN_POSITIVE"
        elif summary["classification"] == "CENSORING_DOMINATED":
            label = "CENSORING_DOMINATED"
        elif summary["classification"] == "NULL":
            label = "NEUTRAL"
        else:
            label = "CONTEXT_DEPENDENT"
        boundaries[key] = {**copy.deepcopy(summary), "boundary_class": label}
    return boundaries



def phase_censoring_cost_tradeoff(
    campaign: Test2Campaign,
    deadline: float,
    recurrence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Measure censored controls at their own compute cost without calling it a matched-budget rescue."""
    censored_keys = {
        key
        for key, summary in recurrence.items()
        if summary.get("classification") == "CENSORING_DOMINATED"
        or float(summary.get("censoring_rate") or 0.0) > 0.0
    }
    if not censored_keys:
        return {
            "schema_version":1,
            "tested_controls":0,
            "findings":{},
            "policy":"OWN_BUDGET_RESULTS_ARE_CAPABILITY_PLUS_COST_NOT_MATCHED_BUDGET_DELTAS",
        }

    recipe_by_key = {
        _recipe_key(summary["recipe"]): copy.deepcopy(summary["recipe"])
        for summary in recurrence.values()
        if summary.get("recipe")
    }
    multipliers = [
        int(value)
        for value in campaign.cfg.get("censoring_cost_probe_multipliers", [2, 4])
    ]
    max_budget = int(campaign.cfg.get("censoring_cost_probe_max_budget", 4096))
    findings: dict[str, Any] = {}

    for key in sorted(censored_keys):
        if not campaign.can_start(deadline):
            break
        recipe = recipe_by_key.get(key)
        if recipe is None:
            # Recurrence keys may contain composition syntax; use summary recipe directly.
            summary = recurrence.get(key) or {}
            recipe = copy.deepcopy(summary.get("recipe") or {})
        if not recipe:
            continue

        source_rows = [
            row for row in campaign.rows
            if row.get("recipe")
            and _recipe_key(row["recipe"]) == _recipe_key(recipe)
            and row.get("censored_for_capability") is True
        ]
        by_fixture = {}
        for row in source_rows:
            by_fixture.setdefault(str(row.get("fixture_id")), row)

        control_results = []
        for fixture_id, censored in list(by_fixture.items())[:8]:
            if not campaign.can_start(deadline):
                break
            case = campaign.case_by_id.get(fixture_id)
            if case is None:
                continue
            baseline_budget = int(
                censored.get("baseline_generation_budget")
                or campaign.current_baseline_budgets.get(fixture_id)
                or campaign.generation_budget_by_family.get(
                    _family(case),
                    campaign.cfg["generation_budget"],
                )
            )
            baseline_score = float(censored.get("baseline_score") or 0.0)
            attempts = []
            for multiplier in multipliers:
                if not campaign.can_start(deadline):
                    break
                budget = min(max_budget, max(baseline_budget + 1, baseline_budget * multiplier))
                row = campaign.treatment(
                    case,
                    deadline,
                    phase="censoring_cost_tradeoff",
                    kind="own_budget_cost_probe",
                    recipe=recipe,
                    label=f"own-budget-{_recipe_key(recipe)}-b{budget}",
                    source_key="censoring-debt",
                    generation_budget_override=budget,
                )
                if row is None:
                    continue
                valid = row.get("valid_for_capability") is True
                attempts.append({
                    "generation_budget":budget,
                    "valid_for_capability":valid,
                    "score":row.get("score") if valid else None,
                    "baseline_score":baseline_score,
                    "matched_budget_delta":None,
                    "cost_ratio_vs_baseline":(
                        float(budget) / float(baseline_budget)
                        if baseline_budget > 0 else None
                    ),
                    "experiment_id":row.get("experiment_id"),
                })
                if valid:
                    break
            control_results.append({
                "fixture_id":fixture_id,
                "family_id":_family(case),
                "baseline_budget":baseline_budget,
                "baseline_score":baseline_score,
                "attempts":attempts,
                "minimum_valid_own_budget":next(
                    (
                        int(row["generation_budget"])
                        for row in attempts
                        if row.get("valid_for_capability") is True
                    ),
                    None,
                ),
            })

        valid_own = [
            attempt
            for result in control_results
            for attempt in result["attempts"]
            if attempt.get("valid_for_capability") is True
        ]
        capability_gains = [
            attempt
            for result in control_results
            for attempt in result["attempts"]
            if attempt.get("valid_for_capability") is True
            and float(attempt.get("score") or 0.0) > float(result["baseline_score"])
        ]
        budget_resolved = bool(valid_own)
        findings[_recipe_key(recipe)] = {
            "recipe":copy.deepcopy(recipe),
            "censored_source_observations":len(source_rows),
            "fixture_results":control_results,
            "valid_own_budget_results":len(valid_own),
            "capability_gains_with_extra_compute":len(capability_gains),
            "primary_cause":(
                "INSUFFICIENT_TOKEN_BUDGET"
                if budget_resolved
                else "UNRESOLVED_CENSORING"
            ),
            "budget_only_causal_probe":True,
            "finding_class":(
                "CAPABILITY_PLUS_COST_OPPORTUNITY"
                if capability_gains
                else "OWN_BUDGET_VALID_NO_CAPABILITY_GAIN"
                if valid_own
                else "UNRESOLVED_CENSORING"
            ),
            "matched_budget_rescue_claim_allowed":False,
        }

    return {
        "schema_version":1,
        "tested_controls":len(findings),
        "findings":findings,
        "policy":"OWN_BUDGET_RESULTS_ARE_CAPABILITY_PLUS_COST_NOT_MATCHED_BUDGET_DELTAS",
    }


def phase_purple_unicorn(
    campaign: Test2Campaign,
    deadline: float,
    recurrence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_unknowns = list((campaign.handoff.get("uncertainty") or {}).get("unknowns", []) or [])
    source_negative = list((campaign.handoff.get("negative_effects") or {}).get("effects", []) or [])
    target_families: list[str] = []
    for row in source_unknowns + source_negative:
        key = str(row.get("key") or "")
        for case in campaign.partitions["VALIDATION"]:
            family = _family(case)
            if family in key and family not in target_families:
                target_families.append(family)

    candidates = [
        case for case in campaign.partitions["VALIDATION"]
        if not target_families or _family(case) in target_families
    ]
    candidates.sort(
        key=lambda case: (
            -int(case.get("difficulty_level", 0)),
            _family(case),
            _fixture_id(case),
        )
    )
    recipes = []
    for summary in recurrence.values():
        if summary.get("classification") in {"STRONG", "PROMISING", "UNCERTAIN"}:
            recipes.append(copy.deepcopy(summary["recipe"]))
    recipes = recipes[:8] or [copy.deepcopy(row) for row in campaign.recipes[:4]]

    records: dict[str, Any] = {}
    cursor = 0
    while candidates and recipes and campaign.can_start(deadline):
        case = candidates[(cursor // len(recipes)) % len(candidates)]
        recipe = copy.deepcopy(recipes[cursor % len(recipes)])
        if recipe.get("exact_test12_intervention") is not None:
            recipe["_proof_seed"] = 47 + (cursor % 3)
        else:
            mode = cursor % 3
            if mode == 1:
                recipe["ingredient_ids"] = list(reversed(recipe["ingredient_ids"]))
            elif mode == 2 and recipe["ingredient_ids"]:
                recipe["ingredient_ids"] = list(recipe["ingredient_ids"]) + [recipe["ingredient_ids"][0]]
        observation = campaign.treatment(
            case,
            deadline,
            phase="purple_unicorn",
            kind="unicorn_probe",
            recipe=recipe,
            label="unicorn-" + _recipe_key(recipe),
            source_key="boundary-search",
        )
        if observation is not None:
            result_class = str((observation.get("classification") or {}).get("result_class"))
            if observation.get("delta_valid") is not True:
                continue
            if result_class != "ANSWER_CORRECT" or float(observation["delta"]) < (-0.5 * campaign.noise_sigma):
                key = f"{_family(case)}|{_fixture_id(case)}|{_recipe_key(recipe)}"
                records[key] = {
                    "family_id": _family(case),
                    "fixture_id": _fixture_id(case),
                    "difficulty_level": int(case.get("difficulty_level", 0)),
                    "recipe": recipe,
                    "result_class": result_class,
                    "delta": observation.get("delta"),
                    "experiment_id": observation.get("experiment_id"),
                    "status": "REPRODUCIBLE_CANDIDATE",
                    "next_action": "MAP_LOCAL_FAILURE_REGION",
                }
        cursor += 1
        if cursor >= len(candidates) * len(recipes) * 3:
            break
    return records


def phase_knockout(
    campaign: Test2Campaign,
    deadline: float,
    recurrence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if campaign.handoff.get("handoff_mode") == "TEST12_EXACT":
        records: dict[str, Any] = {}
        ranked_exact = sorted(
            recurrence.items(),
            key=lambda item: float(item[1].get("normalized_effect", 0.0)),
            reverse=True,
        )
        for key, summary in ranked_exact[:8]:
            recipe = copy.deepcopy(summary.get("recipe") or {})
            if not recipe or recipe.get("exact_test12_intervention") is None:
                continue
            records[key] = {
                "source_recipe":copy.deepcopy(recipe),
                "full_median_delta":summary.get("median_delta"),
                "required_ingredients":[],
                "removable_ingredients":[],
                "minimal_recipe":copy.deepcopy(recipe),
                "observations":{"FULL":int(summary.get("n") or 0)},
                "distillation_status":"EXACT_CONTROL_ATOMIC_OR_ALREADY_COMPOSED",
                "semantic_mutation_prohibited":True,
            }
        return records

    ranked = sorted(
        recurrence.items(),
        key=lambda item: float(item[1].get("normalized_effect", 0.0)),
        reverse=True,
    )
    recipes = [
        copy.deepcopy(summary["recipe"])
        for _, summary in ranked
        if len((summary.get("recipe") or {}).get("ingredient_ids") or []) >= 2
        and summary.get("classification") in {"STRONG", "PROMISING", "UNCERTAIN"}
    ][:8]
    if not recipes:
        recipes = [
            {
                "ingredient_ids": ["ING-001", "ING-003"],
                "dose": 1.0,
                "representation": "prose",
                "placement": "prefix",
            }
        ]

    fixtures = _balanced_cases(campaign.partitions["VALIDATION"], len(campaign.partitions["VALIDATION"]))
    records: dict[str, Any] = {}
    for recipe in recipes:
        if not campaign.can_start(deadline):
            break
        full_ids = list(recipe["ingredient_ids"])
        variants = [("FULL", full_ids)]
        for index, ingredient in enumerate(full_ids):
            reduced = full_ids[:index] + full_ids[index + 1 :]
            if reduced:
                variants.append((f"WITHOUT-{ingredient}", reduced))

        observed: dict[str, list[float]] = defaultdict(list)
        cursor = 0
        while fixtures and variants and campaign.can_start(deadline) and cursor < len(variants) * min(16, len(fixtures)):
            label, ids = variants[cursor % len(variants)]
            case = fixtures[(cursor // len(variants)) % min(16, len(fixtures))]
            variant = {
                **copy.deepcopy(recipe),
                "ingredient_ids": ids,
            }
            row = campaign.treatment(
                case,
                deadline,
                phase="knockout_distillation",
                kind="knockout",
                recipe=variant,
                label=f"knockout-{label}-" + _recipe_key(variant),
                source_key=_recipe_key(recipe),
            )
            if row is not None and row.get("delta_valid") is True and row.get("delta") is not None:
                observed[label].append(float(row["delta"]))
            cursor += 1

        full_center = float(median(observed.get("FULL", [0.0])))
        required: list[str] = []
        removable: list[str] = []
        for label, values in observed.items():
            if label == "FULL" or not values:
                continue
            ingredient = label.removeprefix("WITHOUT-")
            center = float(median(values))
            degradation = full_center - center
            if degradation > 0.5 * campaign.noise_sigma:
                required.append(ingredient)
            else:
                removable.append(ingredient)

        minimal_ids = [item for item in full_ids if item not in removable]
        if not minimal_ids:
            minimal_ids = [full_ids[0]]
        key = _recipe_key(recipe)
        records[key] = {
            "source_recipe": recipe,
            "full_median_delta": full_center,
            "required_ingredients": required,
            "removable_ingredients": removable,
            "minimal_recipe": {
                **copy.deepcopy(recipe),
                "ingredient_ids": minimal_ids,
            },
            "observations": {name: len(values) for name, values in observed.items()},
        }
    return records


def phase_blind(
    campaign: Test2Campaign,
    deadline: float,
    knockouts: dict[str, Any],
    recurrence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    recipes = [
        copy.deepcopy(row["minimal_recipe"])
        for row in knockouts.values()
        if row.get("minimal_recipe")
    ]
    if not recipes:
        ranked = sorted(
            recurrence.values(),
            key=lambda row: float(row.get("normalized_effect", 0.0)),
            reverse=True,
        )
        recipes = [copy.deepcopy(row["recipe"]) for row in ranked if row.get("recipe")]
    recipes = recipes[: int(campaign.cfg["blind_recipes"])]
    if not recipes:
        recipes = [copy.deepcopy(campaign.recipes[0])]

    fixtures = _balanced_cases(campaign.partitions["TEST2_BLIND"], len(campaign.partitions["TEST2_BLIND"]))
    holdout_claim = _claim_holdout_partition(campaign, "TEST2_BLIND", fixtures)

    if campaign.handoff.get("handoff_mode") == "TEST12_EXACT":
        from .test12_tuning import TuningRun, _policy_lock_hash, load_collection

        winner = copy.deepcopy(campaign.handoff.get("winner_policy") or {})
        expected_lock = str(campaign.handoff.get("winner_lock_sha256") or "")
        observed_lock = _policy_lock_hash(winner)
        if expected_lock and observed_lock != expected_lock:
            raise ValueError("frozen Test 1.2 policy hash changed before Test 2 blind proof")

        collection = load_collection(
            Path(campaign.runner.results_root),
            str(campaign.handoff["collection_run"]),
        )
        proof = TuningRun(
            campaign.runner,
            campaign.cases,
            collection,
            clock=campaign.clock,
            started=campaign.start,
        )
        proof.active_end = campaign.active_end
        proof.campaign.active_end = campaign.active_end
        proof.campaign.call_start_cutoff = campaign.call_start_cutoff
        proof.campaign.allowed_partitions = {"TEST2_BLIND"}

        exact_rows: list[dict[str, Any]] = []
        recipe = {
            "exact_locked_policy":copy.deepcopy(winner),
            "policy_lock_sha256":observed_lock,
            "ingredient_ids":[f"POLICY:{winner.get('policy_id','UNKNOWN')}"],
            "dose":1.0,
            "representation":"exact-policy",
            "placement":"semantic",
        }
        for case in fixtures:
            if not campaign.can_start(deadline):
                break
            row = proof.run_policy(winner, case, deadline, seed=50)
            if row is None:
                continue
            converted = {
                "schema_version":1,
                "timestamp_utc":campaign.runner._utc(),
                "phase":"blind_confirmation",
                "kind":"blind",
                "fixture_id":row.get("fixture_id"),
                "family_id":row.get("family_id"),
                "difficulty_level":int(case.get("difficulty_level",0)),
                "partition":"TEST2_BLIND",
                "experiment_id":row.get("tuning_observation_sha256"),
                "classification":{
                    "result_class":(
                        "ANSWER_CORRECT"
                        if float(row.get("score") or 0.0) >= 1.0
                        else "ANSWER_WRONG"
                    ),
                    "valid_for_capability":bool(row.get("valid_for_capability")),
                },
                "valid_for_capability":bool(row.get("valid_for_capability")),
                "baseline_valid_for_capability":bool(row.get("control_valid_for_capability")),
                "baseline_generation_budget":row.get("baseline_generation_budget"),
                "treatment_generation_budget":row.get("treatment_generation_budget"),
                "budget_comparison_valid":bool(row.get("budget_comparison_valid")),
                "delta_valid":bool(row.get("delta_valid")),
                "score":float(row.get("score") or 0.0),
                "baseline_score":float(row.get("control_score") or 0.0),
                "delta":row.get("delta") if row.get("delta_valid") is True else None,
                "recipe":copy.deepcopy(recipe),
                "source_key":"FROZEN_TEST1.2_POLICY",
                "policy_id":winner.get("policy_id"),
                "policy_lock_sha256":observed_lock,
                "semantic_hash_match":True,
                "model_calls":row.get("model_calls"),
                "timing":{},
                "evidence_refs":{},
            }
            campaign.rows.append(converted)
            campaign.runner.store.append_jsonl("test2-observations.jsonl", converted)
            exact_rows.append(converted)

        effects = _effect_map(
            exact_rows,
            noise_sigma=campaign.noise_sigma,
            bootstrap_samples=int(campaign.cfg["bootstrap_samples"]),
            max_censoring_rate=float(campaign.cfg["max_effect_censoring_rate"]),
            key_fn=lambda row: _recipe_key(row["recipe"]),
        )
        valid_exact = [row for row in exact_rows if row.get("delta_valid") is True]
        direct_pass_rate = (
            sum(1 for row in valid_exact if float(row.get("baseline_score") or 0.0) >= 1.0)
            / len(valid_exact)
            if valid_exact else None
        )
        policy_pass_rate = (
            sum(1 for row in valid_exact if float(row.get("score") or 0.0) >= 1.0)
            / len(valid_exact)
            if valid_exact else None
        )
        mean_policy_cost_ratio = (
            mean([1.0 + float(row.get("model_calls") or 0.0) for row in valid_exact])
            if valid_exact else None
        )
        accuracy_advantage = (
            policy_pass_rate - direct_pass_rate
            if policy_pass_rate is not None and direct_pass_rate is not None
            else None
        )
        return {
            "recipes_frozen_before_phase":True,
            "locked_policy_proved_exactly":True,
            "policy_lock_sha256":observed_lock,
            "locked_policy":copy.deepcopy(winner),
            "fixture_count":len(fixtures),
            "effects":effects,
            "observations":len(exact_rows),
            "holdout_claim":holdout_claim,
            "partition_retired_after_this_cycle":True,
            "policy_cost_exchange":{
                "direct_execution_cost_ratio":1.0,
                "mean_policy_cost_ratio":mean_policy_cost_ratio,
                "direct_pass_rate":direct_pass_rate,
                "policy_pass_rate":policy_pass_rate,
                "accuracy_advantage":accuracy_advantage,
            },
        }

    rows_before = len(campaign.rows)
    cursor = 0
    while recipes and fixtures and campaign.can_start(deadline):
        recipe = recipes[cursor % len(recipes)]
        case = fixtures[(cursor // len(recipes)) % len(fixtures)]
        campaign.treatment(
            case,
            deadline,
            phase="blind_confirmation",
            kind="blind",
            recipe=recipe,
            label="blind-" + _recipe_key(recipe),
            source_key="frozen-minimal-recipe",
            blind=True,
        )
        cursor += 1
        if cursor >= len(recipes) * len(fixtures):
            break

    blind_rows = campaign.rows[rows_before:]
    effects = _effect_map(
        blind_rows,
        noise_sigma=campaign.noise_sigma,
        bootstrap_samples=int(campaign.cfg["bootstrap_samples"]),
        max_censoring_rate=float(campaign.cfg["max_effect_censoring_rate"]),
        key_fn=lambda row: _recipe_key(row["recipe"]),
    )
    return {
        "recipes_frozen_before_phase": True,
        "fixture_count": len(fixtures),
        "effects": effects,
        "observations": len(blind_rows),
        "holdout_claim":holdout_claim,
        "partition_retired_after_this_cycle":True,
    }



def _row_capability_valid(row: dict[str, Any]) -> bool:
    if row.get("valid_for_capability") is True:
        return True
    classification = row.get("classification") or {}
    return classification.get("valid_for_capability") is True


def _build_model_limit_and_finetuning(
    campaign: Test2Campaign,
    recovery: dict[str, Any],
    negative_transfer: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    matrix = recovery.get("matrix") or {}
    recovery_by_phenotype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matrix.values():
        recovery_by_phenotype[str(row.get("phenotype_id"))].append(row)

    failures: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # Start with every Test-1 failure so a source phenotype cannot disappear
    # merely because Test 2 successfully recovers it.
    for source_row in (campaign.handoff.get("failures") or {}).get("failures", []) or []:
        fixture_id = source_row.get("fixture_id")
        if not isinstance(fixture_id, str) or fixture_id not in campaign.case_by_id:
            continue
        source_copy = copy.deepcopy(source_row)
        source_copy.setdefault("family_id", _family(campaign.case_by_id[fixture_id]))
        source_copy.setdefault("partition", campaign._partition(campaign.case_by_id[fixture_id]))
        source_copy["source_run"] = campaign.handoff.get("run_id")
        failures[_phenotype_id(source_copy)].append(source_copy)

    for row in campaign.rows:
        if row.get("partition") not in {"DISCOVERY", "VALIDATION"}:
            continue
        classification = row.get("classification") or {}
        if classification.get("result_class") == "ANSWER_CORRECT":
            continue
        failures[_phenotype_id(row)].append(row)

    min_failures = int(campaign.cfg["fine_tuning_min_independent_failures"])
    limits: dict[str, Any] = {}
    finetune: list[dict[str, Any]] = []
    dataset: list[dict[str, Any]] = []
    negative_transfer_available = bool(negative_transfer)

    for phenotype, rows in sorted(failures.items()):
        fixture_ids = sorted({str(row["fixture_id"]) for row in rows})
        family_ids = sorted({str(row["family_id"]) for row in rows})
        result_classes = sorted({
            str((row.get("classification") or {}).get("result_class"))
            for row in rows
        })
        valid_rows = [row for row in rows if _row_capability_valid(row)]
        invalid_rows = [row for row in rows if not _row_capability_valid(row)]
        valid_fixture_ids = sorted({
            str(row["fixture_id"])
            for row in valid_rows
            if row.get("fixture_id")
        })
        invalid_fixture_ids = sorted({
            str(row["fixture_id"])
            for row in invalid_rows
            if row.get("fixture_id")
        })
        recoveries = recovery_by_phenotype.get(phenotype, [])
        general = [row for row in recoveries if row.get("status") == "GENERAL_RECOVERY"]
        best_general = max(
            general,
            key=lambda row: float(row.get("success_rate", 0.0)),
            default=None,
        )
        valid_cases = [
            campaign.case_by_id.get(fixture_id)
            for fixture_id in valid_fixture_ids
        ]
        observable_target = bool(valid_cases) and all(
            case is not None and case.get("scorer") is not None and "expected" in case
            for case in valid_cases
        )
        levels = [
            int(case.get("difficulty_level", 0))
            for case in valid_cases
            if case is not None
        ]

        if invalid_rows and not valid_rows:
            toolish = any("tool" in family.lower() for family in family_ids)
            owner = "TOOL_SOLVABLE" if toolish else "SYSTEM_SOLVABLE"
        elif invalid_rows and valid_rows:
            owner = "SYSTEM_DISAMBIGUATION_REQUIRED"
        elif not observable_target:
            owner = "DATA_SOLVABLE"
        elif best_general is not None:
            recipe_len = len((best_general.get("recipe") or {}).get("ingredient_ids") or [])
            owner = "PROMPT_SOLVABLE" if recipe_len <= 1 else "RECIPE_SOLVABLE"
        elif len(valid_fixture_ids) >= min_failures:
            # Recurrent residual failures at the extreme edge are retained as a
            # measured base-model capability limit rather than automatically
            # converted into training data. Lower/mid-level recurrent failures
            # are the scientifically useful Test-3 fine-tuning candidates.
            owner = (
                "MODEL_CAPABILITY_LIMIT"
                if levels and min(levels) >= 9
                else "FINE_TUNING_CANDIDATE"
            )
        else:
            owner = "INSUFFICIENT_EVIDENCE"

        entry = {
            "phenotype_id": phenotype,
            "owner": owner,
            "independent_fixture_count": len(valid_fixture_ids),
            "valid_independent_fixture_count": len(valid_fixture_ids),
            "invalid_fixture_count": len(invalid_fixture_ids),
            "fixture_ids": fixture_ids,
            "valid_fixture_ids": valid_fixture_ids,
            "invalid_fixture_ids": invalid_fixture_ids,
            "family_ids": family_ids,
            "difficulty_levels": sorted(levels),
            "result_classes": result_classes,
            "experiment_ids": [
                str(row.get("experiment_id"))
                for row in rows
                if row.get("experiment_id")
            ],
            "recovery_evidence": copy.deepcopy(recoveries),
            "observable_target": observable_target,
            "negative_transfer_evidence_available": negative_transfer_available,
            "source_test1_run": campaign.handoff.get("run_id"),
        }
        limits[phenotype] = entry

        if owner == "FINE_TUNING_CANDIDATE":
            package = {
                **copy.deepcopy(entry),
                "qualification": {
                    "recurrent": True,
                    "independent": len(valid_fixture_ids) >= min_failures,
                    "model_owned": True,
                    "prior_generalization_evidence": len(valid_fixture_ids) >= min_failures,
                    "cheaper_prompt_recipe_owner_resolved": best_general is None,
                    "tool_system_owner_resolved": not invalid_rows,
                    "all_source_failures_capability_valid": not invalid_rows,
                    "observable_target": observable_target,
                    "protected_fixtures_excluded": True,
                    "negative_transfer_evidence_available": negative_transfer_available,
                    "leakage_safe_partitioning": True,
                },
                "next_action": "TEST3_FINE_TUNING_QUALIFICATION",
            }
            finetune.append(package)
            for fixture_id in valid_fixture_ids:
                case = campaign.case_by_id.get(fixture_id)
                if case is None or campaign._partition(case) not in {"DISCOVERY", "VALIDATION"}:
                    continue
                dataset.append({
                    "phenotype_id": phenotype,
                    "fixture_id": fixture_id,
                    "family_id": _family(case),
                    "difficulty_level": int(case.get("difficulty_level", 0)),
                    "prompt": case.get("prompt"),
                    "expected": copy.deepcopy(case.get("expected")),
                    "scorer": case.get("scorer"),
                    "partition": campaign._partition(case),
                    "train_eligible": True,
                    "source_experiment_ids": [
                        str(row.get("experiment_id"))
                        for row in valid_rows
                        if row.get("fixture_id") == fixture_id and row.get("experiment_id")
                    ],
                    "source_evidence_capability_valid": True,
                })
    return {"phenotypes": limits}, finetune, dataset


def _latency_envelope(campaign: Test2Campaign) -> dict[str, Any]:
    values = [
        value
        for value in (_timing_seconds(row) for row in campaign.rows)
        if value is not None
    ]
    return {
        "schema_version": 1,
        "measured_observations": len(values),
        "p50_seconds": _safe_percentile(values, 0.50),
        "p95_seconds": _safe_percentile(values, 0.95),
        "maximum_seconds": max(values) if values else None,
        "acceptance_max_overhead_ratio": float(campaign.cfg["acceptance_latency_ratio"]),
    }


def _final_recipe_registry(
    knockouts: dict[str, Any],
    blind: dict[str, Any],
    harm_evidence: dict[str, Any] | None = None,
    *,
    campaign: Test2Campaign | None = None,
) -> list[dict[str, Any]]:
    blind_effects = blind.get("effects") or {}
    harm_evidence = harm_evidence or {}
    result: list[dict[str, Any]] = []

    if blind.get("locked_policy_proved_exactly") is True:
        policy = copy.deepcopy(blind.get("locked_policy") or {})
        policy_lock = str(blind.get("policy_lock_sha256") or "")
        policy_key = next(iter(blind_effects), "POLICY-" + policy_lock[:16])
        blind_summary = copy.deepcopy(blind_effects.get(policy_key))

        referenced_ids: list[str] = []
        for value in (
            policy.get("intervention_id"),
            policy.get("fallback_intervention_id"),
        ):
            ident = str(value or "")
            if ident and ident not in referenced_ids:
                referenced_ids.append(ident)
        for value in (policy.get("route_map") or {}).values():
            ident = str(value or "")
            if ident and ident != "DIRECT" and ident not in referenced_ids:
                referenced_ids.append(ident)

        harm_by_intervention: dict[str, Any] = {}
        all_harm_safe = True
        for ident in referenced_ids:
            matching_recipe = next(
                (
                    recipe for recipe in getattr(campaign, "recipes", [])
                    if str(recipe.get("intervention_id") or "") == ident
                ),
                None,
            ) if campaign is not None else None
            if matching_recipe is None:
                harm_by_intervention[ident] = {
                    "verification_status":"MISSING_HARM_EVIDENCE",
                    "harm_safe":False,
                }
                all_harm_safe = False
                continue
            summary = copy.deepcopy(harm_evidence.get(_recipe_key(matching_recipe)))
            harm_by_intervention[ident] = summary
            if not (
                summary
                and summary.get("harm_evidence_sufficient") is True
                and summary.get("harm_safe") is True
            ):
                all_harm_safe = False

        # DIRECT has no applied control and therefore no control-specific harm debt.
        if not referenced_ids:
            all_harm_safe = True

        blind_ok = bool(
            blind_summary
            and blind_summary.get("classification")
            in {"STRONG", "PROMISING", "NULL"}
            and blind_summary.get("classification") != "CENSORING_DOMINATED"
        )
        return [{
            "recipe_id":"POLICY-" + (policy_lock[:12] or "UNLOCKED"),
            "source_recipe_key":"FROZEN_TEST1.2_POLICY",
            "recipe_key":policy_key,
            "recipe":{
                "exact_locked_policy":copy.deepcopy(policy),
                "policy_lock_sha256":policy_lock,
                "ingredient_ids":[f"POLICY:{policy.get('policy_id','UNKNOWN')}"],
                "representation":"exact-policy",
                "placement":"semantic",
                "dose":1.0,
            },
            "blind_summary":blind_summary,
            "harm_summary":{
                "referenced_interventions":harm_by_intervention,
                "all_referenced_controls_harm_safe":all_harm_safe,
            },
            "verified_for_shipping":bool(blind_ok and all_harm_safe),
            "required_ingredients":referenced_ids,
            "removed_ingredients":[],
            "semantic_translation_used":False,
        }]
    for source_key, row in knockouts.items():
        recipe = copy.deepcopy(row.get("minimal_recipe") or {})
        if not recipe:
            continue
        key = _recipe_key(recipe)
        blind_summary = blind_effects.get(key)
        harm_summary = copy.deepcopy(harm_evidence.get(key))
        result.append({
            "recipe_id": "REC-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12],
            "source_recipe_key": source_key,
            "recipe_key": key,
            "recipe": recipe,
            "blind_summary": copy.deepcopy(blind_summary),
            "harm_summary": harm_summary,
            "verified_for_shipping": bool(
                harm_summary
                and harm_summary.get("harm_evidence_sufficient") is True
                and harm_summary.get("harm_safe") is True
                and blind_summary
                and blind_summary.get("classification") in {"STRONG", "PROMISING", "NULL"}
            ),
            "required_ingredients": list(row.get("required_ingredients") or []),
            "removed_ingredients": list(row.get("removable_ingredients") or []),
        })
    result.sort(
        key=lambda row: float((row.get("blind_summary") or {}).get("normalized_effect", -999.0)),
        reverse=True,
    )
    return result


def _build_finalization_contract(
    campaign: Test2Campaign,
    recovery: dict[str, Any],
    negative_transfer: dict[str, Any],
    knockouts: dict[str, Any],
    blind: dict[str, Any],
    finetune: list[dict[str, Any]],
    latency: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    recipes = _final_recipe_registry(
        knockouts,
        blind,
        campaign.harm_evidence,
        campaign=campaign,
    )
    verified_recipes = [row for row in recipes if row.get("verified_for_shipping") is True]
    primary = verified_recipes[0] if verified_recipes else {
        "recipe_id": "REC-NONE",
        "recipe_key": "",
        "recipe": {
            "ingredient_ids": [],
            "dose": 1.0,
            "representation": "prose",
            "placement": "prefix",
        },
        "blind_summary": None,
    }
    general_recoveries = [
        copy.deepcopy(row)
        for row in (recovery.get("matrix") or {}).values()
        if row.get("status") == "GENERAL_RECOVERY"
    ]
    harmful = [
        {"key": key, **copy.deepcopy(row)}
        for key, row in negative_transfer.items()
        if row.get("boundary_class") == "NEGATIVE_TRANSFER"
    ]

    ingredient_index = {item["id"]: item for item in INGREDIENTS}
    primary_recipe = primary.get("recipe") or {}
    exact_policy = primary_recipe.get("exact_locked_policy")
    if exact_policy is not None:
        rendered_control_text = None
        rendering_rule = {
            "mode":"EXACT_TEST1.2_POLICY",
            "semantic_translation_allowed":False,
            "policy_lock_sha256":primary_recipe.get("policy_lock_sha256"),
        }
        ingredient_definitions: list[dict[str, Any]] = []
    else:
        rendered_parts = []
        for ingredient_id in primary_recipe.get("ingredient_ids") or []:
            definition = ingredient_index.get(str(ingredient_id))
            if definition is None:
                continue
            rendered_parts.append(str(definition.get("full") or definition.get("short") or ""))
        rendered_control_text = "\n".join(rendered_parts)
        rendering_rule = {
            "mode":"LEGACY_INGREDIENT_RECIPE",
            "ingredient_order_is_semantic": True,
            "placement": primary_recipe.get("placement"),
            "representation": primary_recipe.get("representation"),
            "dose": primary_recipe.get("dose"),
            "recurrence_count": len(primary_recipe.get("ingredient_ids") or []),
        }
        ingredient_definitions = copy.deepcopy(list(INGREDIENTS))

    exact_model = {
        "model": str(campaign.runner.model),
        "thinking_mode": bool(campaign.cfg["thinking_mode"]),
        "reasoning_effort": campaign.cfg.get("reasoning_effort"),
        "generation_budget": int(campaign.cfg["generation_budget"]),
        "generation_budget_by_family": copy.deepcopy(
            campaign.generation_budget_by_family
        ),
        "temperature": 0.0,
        "seed": 42,
        "source_test1_run": campaign.handoff.get("run_id"),
    }

    rollback = {
        "schema_version": 1,
        "trigger": "any final acceptance hard gate fails",
        "base_model": str(campaign.runner.model),
        "disable_fine_tune_adapter": True,
        "restore_recipe_id": primary["recipe_id"],
        "restore_recipe": copy.deepcopy(primary["recipe"]),
        "retain_failure_snapshots": True,
        "prohibit_history_rewrite": True,
    }

    contract = {
        "schema_version": 1,
        "status": "PROVISIONAL_PENDING_TEST3_AND_FINAL_ACCEPTANCE",
        "base_model": str(campaign.runner.model),
        "primary_recipe": copy.deepcopy(primary),
        "rendered_primary_control_text": rendered_control_text,
        "rendering_rule": rendering_rule,
        "alternate_minimal_recipes": copy.deepcopy(verified_recipes[1:]),
        "unverified_minimal_recipes": copy.deepcopy(
            [row for row in recipes if row.get("verified_for_shipping") is not True]
        ),
        "ingredient_definitions": ingredient_definitions,
        "recovery_policies": general_recoveries,
        "negative_transfer_boundaries": harmful,
        "tool_boundary": {
            "model_role": "propose tool/action intent and arguments",
            "system_role": "validate tool availability, argument schema, permissions, authorization, and execution",
            "model_output_is_authority": False,
            "permission_bypass_allowed": False,
            "failed_tool_action_policy": "snapshot exact state; one bounded recipe recovery; then escalate",
        },
        "ownership_rules": {
            "prompt_or_recipe_recoverable": "INVERTED_RECIPE",
            "runtime_or_permission_failure": "SYSTEM_OR_TOOL",
            "recurrent_independent_model_owned": "TEST3_FINE_TUNING",
            "unresolved_low_evidence": "INSUFFICIENT_EVIDENCE",
        },
        "routing_and_retry": {
            "ordinary_attempts": 1,
            "bounded_recovery_attempts": 1,
            "repeated_external_retry_loop": False,
            "on_repeat_failure": "snapshot_and_escalate",
        },
        "context_memory_policy": {
            "authoritative_current_state_wins": True,
            "obsolete_state_must_be_rejected": True,
            "external_evidence_over_model_memory": True,
            "protected_eval_material_may_enter_context": False,
            "failure_metadata_may_be_reused_only_through_versioned_evidence": True,
        },
        "model_configuration": exact_model,
        "latency_resource_envelope": copy.deepcopy(latency),
        "resource_evidence": {
            "telemetry": "telemetry.jsonl",
            "hardware": "hardware.json",
            "raw_runtime_exchanges": "raw/runtime/exchanges/",
        },
        "unresolved_model_owned_failures": copy.deepcopy(finetune),
        "rollback_configuration": copy.deepcopy(rollback),
        "acceptance_gates": {
            "protected_set_leakage": 0,
            "new_severe_regression_count": 0,
            "verified_control_requires_harm_evidence": True,
            "harm_max_break_rate": float(campaign.cfg["harm_max_break_rate"]),
            "harm_gate_uses_ci_upper_bound": True,
            "policy_cost_ratio_ceiling": float(campaign.cfg["policy_cost_ratio_ceiling"]),
            "minimum_accuracy_advantage_when_over_cost_ceiling": float(
                campaign.cfg["minimum_accuracy_advantage_when_over_cost_ceiling"]
            ),
            "tool_boundary_violation_count": 0,
            "general_recovery_min_success_rate": float(campaign.cfg["general_recovery_threshold"]),
            "tuned_inverted_protected_score_must_not_regress": True,
            "tuned_raw_must_be_compared_to_base_raw": True,
            "tuned_inverted_must_be_compared_to_base_inverted": True,
            "max_p95_latency_ratio_vs_frozen_base_inverted": float(campaign.cfg["acceptance_latency_ratio"]),
            "manifest_integrity": 1.0,
        },
    }

    acceptance = {
        "schema_version": 1,
        "purpose": "post-Test3 release qualification; not a new discovery campaign",
        "protected_partition": "TEST3_PROTECTED",
        "conditions": [
            "GPT20B_RAW",
            "GPT20B_PLUS_INVERTED",
            "TUNED_GPT20B_RAW",
            "TUNED_GPT20B_PLUS_INVERTED",
        ],
        "metrics": [
            "capability_score",
            "failure_rate",
            "failure_phenotypes",
            "recovery_success_rate",
            "negative_transfer",
            "tool_boundary_violations",
            "latency_p50",
            "latency_p95",
            "resource_cost",
            "manifest_integrity",
        ],
        "hard_gates": copy.deepcopy(contract["acceptance_gates"]),
        "recipe_freeze": copy.deepcopy(primary),
        "rollback_on_failure": copy.deepcopy(rollback),
    }
    return contract, acceptance, rollback


def write_test2_outputs(
    campaign: Test2Campaign,
    recurrence: dict[str, dict[str, Any]],
    recovery: dict[str, Any],
    negative_transfer: dict[str, Any],
    unicorns: dict[str, Any],
    knockouts: dict[str, Any],
    blind: dict[str, Any],
) -> None:
    store = campaign.runner.store
    store.write_json(
        "recurrence-map.json",
        {"schema_version": 1, "recipes": recurrence},
        producer="test2",
        stage="report",
    )
    store.write_json(
        "proof-value-scheduler.json",
        copy.deepcopy(campaign.proof_scheduler_audit) or {
            "schema_version":1,
            "mode":"LEGACY_OR_NO_EXACT_SCHEDULER",
            "candidates":{},
        },
        producer="test2",
        stage="report",
    )
    hyperedges = {
        key: value for key, value in recurrence.items()
        if len(key.split("->")) >= 3
    }
    store.write_json(
        "higher-order-hypergraph.json",
        {"schema_version": 1, "hyperedges": hyperedges},
        producer="test2",
        stage="report",
    )
    store.write_json(
        "failure-phenotype-registry.json",
        {"schema_version": 1, "phenotypes": recovery.get("phenotypes") or {}},
        producer="test2",
        stage="report",
    )
    store.write_json(
        "failure-recovery-matrix.json",
        {"schema_version": 1, "matrix": recovery.get("matrix") or {}},
        producer="test2",
        stage="report",
    )
    store.write_json(
        "negative-transfer-boundaries.json",
        {"schema_version": 1, "boundaries": negative_transfer},
        producer="test2",
        stage="report",
    )
    store.write_json(
        "harm-sentinel-evidence.json",
        {
            "schema_version":1,
            "policy":"VERIFIED_CONTROL_REQUIRES_DEDICATED_BASELINE_PASS_HARM_EVIDENCE_AND_WILSON90_UPPER_BOUND_AT_OR_BELOW_CEILING",
            "recipes":copy.deepcopy(campaign.harm_evidence),
        },
        producer="test2",
        stage="report",
    )
    store.write_json(
        "purple-unicorn-registry.json",
        {"schema_version": 1, "candidates": unicorns},
        producer="test2",
        stage="report",
    )
    store.write_json(
        "knockout-registry.json",
        {"schema_version": 1, "recipes": knockouts},
        producer="test2",
        stage="report",
    )

    blind_payload = copy.deepcopy(blind)
    store.write_json(
        "blind-confirmation-results.json",
        blind_payload,
        producer="test2",
        stage="report",
    )

    minimal = _final_recipe_registry(
        knockouts,
        blind,
        campaign.harm_evidence,
        campaign=campaign,
    )
    store.write_json(
        "minimal-recipe-registry.json",
        {"schema_version": 1, "recipes": minimal},
        producer="test2",
        stage="report",
    )

    limits, finetune, dataset = _build_model_limit_and_finetuning(
        campaign,
        recovery,
        negative_transfer,
    )
    store.write_json(
        "model-limit-registry.json",
        {"schema_version": 1, **limits},
        producer="test2",
        stage="report",
    )
    store.write_json(
        "fine-tuning-candidate-queue.json",
        {"schema_version": 1, "candidates": finetune},
        producer="test2",
        stage="report",
    )

    training_yield = {
        "schema_version":1,
        "raw_test2_observations":len(campaign.rows),
        "qualified_fine_tuning_phenotypes":len(finetune),
        "valid_train_eligible_examples":len(dataset),
        "training_pipeline_has_input":bool(dataset),
        "success_metric":"VALID_SCIENTIFICALLY_ELIGIBLE_EXAMPLES_NOT_RAW_OBSERVATION_VOLUME",
        "status":(
            "TRAINING_INPUT_AVAILABLE"
            if dataset
            else "NO_VALID_WEIGHT_TRAINING_INPUT_YET"
        ),
        "interpretation":(
            "A small number of valid examples is preferable to a large contaminated corpus."
        ),
    }
    store.write_json(
        "training-asset-yield.json",
        training_yield,
        producer="test2",
        stage="report",
    )

    stopping = _build_harness_stopping_rules(campaign, blind, limits)
    store.write_json(
        "harness-stopping-rules.json",
        stopping,
        producer="test2",
        stage="decision",
    )

    protected_ids = [
        _fixture_id(case) for case in campaign.partitions["TEST3_PROTECTED"]
    ]
    store.write_json(
        "test3-protected-manifest.json",
        {
            "schema_version": 1,
            "partition": "TEST3_PROTECTED",
            "fixture_ids": protected_ids,
            "fixture_count": len(protected_ids),
            "exposed_by_test2": False,
            "train_eligible": False,
            "partition_fingerprint":_fixture_set_fingerprint(
                campaign.partitions["TEST3_PROTECTED"]
            ),
            "acceptance_use_budget":1,
            "retire_permanently_after_first_exposure":True,
            "same_run_resume_allowed_after_exposure":True,
            "cross_run_reuse_prohibited":True,
            "must_claim_holdout_before_exposure":True,
            "next_cycle_partition_must_not_exist_during_current_cycle_discovery":True,
        },
        producer="test2",
        stage="report",
    )
    store.write_json(
        "test3-dataset-manifest.json",
        {
            "schema_version": 1,
            "train_eligible_examples": [
                row for row in dataset
                if row.get("train_eligible") is True
                and row.get("partition") in {"DISCOVERY", "VALIDATION"}
            ],
            "protected_fixture_ids": protected_ids,
            "candidate_phenotypes": [row["phenotype_id"] for row in finetune],
            "leakage_rule": "TEST3_PROTECTED fixtures and siblings derived from them are forbidden from training",
        },
        producer="test2",
        stage="report",
    )

    blind_claim = copy.deepcopy(blind.get("holdout_claim") or {})
    replenishment = {
        "schema_version":1,
        "current_cycle_run_id":getattr(store, "run_id", None),
        "consumed_partition":"TEST2_BLIND",
        "current_blind_claim":blind_claim,
        "test3_protected_partition_fingerprint":_fixture_set_fingerprint(
            campaign.partitions["TEST3_PROTECTED"]
        ),
        "next_cycle_requirements":{
            "generate_new_fixtures":True,
            "source":"NEW_FIELD_FAILURE_PHENOTYPES_NOT_ALREADY_FIXED",
            "zero_fixture_overlap_with_consumed_holdouts":True,
            "cycle_n_partition_must_not_exist_during_cycle_n_minus_1_discovery":True,
            "retire_each_partition_after_acceptance_budget_is_spent":True,
        },
    }
    store.write_json(
        "holdout-replenishment-plan.json",
        replenishment,
        producer="test2",
        stage="holdout-governance",
    )

    unresolved = []
    for row in finetune:
        unresolved.append({
            "question": f"Can Test 3 remove phenotype {row['phenotype_id']} without regression?",
            "evidence": row,
            "confidence": "QUALIFIED_FOR_TEST3",
            "value_of_resolving": "HIGH",
            "estimated_cost": "TEST3",
            "next_test": "fine-tuning causal experiment",
            "owner": "TEST3_FINE_TUNING",
        })
    for key, candidate in unicorns.items():
        unresolved.append({
            "question": f"Map local failure region for {key}",
            "evidence": candidate,
            "confidence": "BOUNDARY_CANDIDATE",
            "value_of_resolving": "MEDIUM",
            "estimated_cost": "REPLAY",
            "next_test": "protected sibling replay after Test3 if phenotype remains",
            "owner": "TEST3_OR_RELEASE_QUALIFICATION",
        })
    store.write_json(
        "remaining-unknowns.json",
        {"schema_version": 1, "unknowns": unresolved},
        producer="test2",
        stage="report",
    )

    latency = _latency_envelope(campaign)
    store.write_json(
        "latency-resource-envelope.json",
        latency,
        producer="test2",
        stage="report",
    )
    contract, acceptance, rollback = _build_finalization_contract(
        campaign,
        recovery,
        negative_transfer,
        knockouts,
        blind,
        finetune,
        latency,
    )
    contract["harness_stopping_rules"] = copy.deepcopy(stopping)
    contract["harness_ceiling_reached"] = bool(stopping.get("harness_ceiling_reached"))
    contract["next_strategy"] = stopping.get("next_strategy")
    contract["training_asset_yield"] = copy.deepcopy(training_yield)
    acceptance["holdout_governance"] = copy.deepcopy(replenishment)
    acceptance["harness_stopping_rules"] = copy.deepcopy(stopping)
    acceptance["training_asset_yield"] = copy.deepcopy(training_yield)
    store.write_json(
        "inverted-finalization-contract.json",
        contract,
        producer="test2",
        stage="report",
    )
    store.write_json(
        "final-acceptance-manifest.json",
        acceptance,
        producer="test2",
        stage="report",
    )
    store.write_json(
        "rollback-configuration.json",
        rollback,
        producer="test2",
        stage="report",
    )
    store.write_json(
        "exact-model-configuration.json",
        contract["model_configuration"],
        producer="test2",
        stage="report",
    )


def run_test2_campaign(
    runner: Any,
    cases: list[dict[str, Any]],
    *,
    test1_run: str,
    clock: Callable[[], float] = time.monotonic,
    started_monotonic: float | None = None,
) -> list[dict[str, Any]]:
    assert runner.store is not None
    plan = build_test2_plan(cases, test1_run=test1_run)
    validate_test2_plan(plan)
    handoff = load_test1_handoff(Path(runner.results_root), test1_run, cases)
    if handoff.get("handoff_mode") == "TEST12_EXACT":
        runtime_path = runner.store.run_dir / "runtime.json"
        if not runtime_path.is_file():
            raise ValueError("Test 2 requires current runtime.json before exact Test 1.2 proof")
        current_runtime = _read_json(runtime_path)
        identity = (handoff.get("runtime_profile") or {}).get("identity") or {}
        if str(identity.get("model")) != str(runner.model):
            raise ValueError("Test 2 model identity differs from Stage 0 characterization")
        if identity.get("runtime_version") != current_runtime.get("version"):
            raise ValueError(
                "Test 2 runtime version differs from Stage 0 characterization; "
                "this is a new onboarding event, not a valid proof continuation"
            )

    campaign = Test2Campaign(
        runner,
        cases,
        handoff,
        clock=clock,
        started_monotonic=started_monotonic,
    )
    if not (runner.store.run_dir / "test2-plan.json").is_file():
        runner.store.write_json("test2-plan.json", plan, producer="test2", stage="preflight")
    runner.store.write_json(
        "test1-handoff-summary.json",
        {
            "schema_version": 1,
            "source_run": test1_run,
            "source_files": list(REQUIRED_TEST1_FILES),
            "noise_sigma": campaign.noise_sigma,
            "promoted_recipe_count": len(campaign.recipes),
            "failure_count": len((handoff.get("failures") or {}).get("failures", []) or []),
            "test3_protected_count": len(campaign.partitions["TEST3_PROTECTED"]),
        },
        producer="test2",
        stage="preflight",
    )

    carry = 0.0
    recurrence: dict[str, dict[str, Any]] = {}
    recovery: dict[str, Any] = {"phenotypes": {}, "matrix": {}}
    negative_transfer: dict[str, Any] = {}
    censoring_tradeoff: dict[str, Any] = {"findings": {}}
    unicorns: dict[str, Any] = {}
    knockouts: dict[str, Any] = {}
    blind: dict[str, Any] = {"effects": {}, "observations": 0}

    for phase_name, nominal_seconds in PHASES:
        phase_start = clock()
        deadline = min(campaign.active_end, phase_start + nominal_seconds + carry)
        if phase_name == "recurrence_higher_order":
            recurrence = phase_recurrence(campaign, deadline)
        elif phase_name == "failure_recovery":
            recovery = phase_failure_recovery(campaign, deadline)
        elif phase_name == "negative_transfer":
            negative_transfer = phase_negative_transfer(campaign, deadline, recurrence)
        elif phase_name == "censoring_cost_tradeoff":
            censoring_tradeoff = phase_censoring_cost_tradeoff(campaign, deadline, recurrence)
        elif phase_name == "purple_unicorn":
            unicorns = phase_purple_unicorn(campaign, deadline, recurrence)
        elif phase_name == "knockout_distillation":
            knockouts = phase_knockout(campaign, deadline, recurrence)
        elif phase_name == "blind_confirmation":
            blind = phase_blind(campaign, deadline, knockouts, recurrence)
        phase_end = clock()
        carry = max(0.0, deadline - phase_end)
        runner.store.append_jsonl(
            "test2-phase-events.jsonl",
            {
                "phase": phase_name,
                "started_monotonic": phase_start,
                "ended_monotonic": phase_end,
                "nominal_seconds": nominal_seconds,
                "carry_forward_seconds": carry,
                "physical_observations": len(campaign.rows),
            },
        )
        if phase_end >= campaign.call_start_cutoff:
            break

    campaign.runner.store.write_json(
        "censoring-cost-tradeoff.json",
        censoring_tradeoff,
        producer="test2",
        stage="report",
    )
    write_test2_outputs(
        campaign,
        recurrence,
        recovery,
        negative_transfer,
        unicorns,
        knockouts,
        blind,
    )
    return campaign.rows
