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
    ("failure_recovery", 120 * 60),
    ("negative_transfer", 55 * 60),
    ("purple_unicorn", 60 * 60),
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
    "higher-order-hypergraph.json",
    "failure-phenotype-registry.json",
    "failure-recovery-matrix.json",
    "negative-transfer-boundaries.json",
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
)

DEFAULT_TEST2_CONFIG: dict[str, Any] = {
    "expected_calls": 4100,
    "safety_call_cap": 10000,
    "generation_budget": 256,
    "thinking_mode": False,
    "reasoning_effort": None,
    "bootstrap_samples": 500,
    "top_recipes": 12,
    "max_recovery_recipes": 8,
    "negative_transfer_recipes": 8,
    "blind_recipes": 4,
    "control_interval": 12,
    "fine_tuning_min_independent_failures": 3,
    "general_recovery_threshold": 0.80,
    "partial_recovery_threshold": 0.60,
    "acceptance_latency_ratio": 1.25,
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
        generation_budget=int(cfg["generation_budget"]),
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
        self.noise_sigma = max(
            EPSILON_NOISE,
            float((handoff.get("noise_model") or {}).get("global_noise_sigma", EPSILON_NOISE)),
        )
        self.treatment_calls = 0
        self.recipes = source_recipes(handoff, limit=int(self.cfg["top_recipes"]))

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
        if not force and fixture_id in self.current_baselines:
            return self.current_baselines[fixture_id]
        if not self.can_start(deadline):
            return None
        self.sequence += 1
        spec = _spec(
            sequence=self.sequence,
            case=case,
            label=f"{phase}-control",
            cfg=self.cfg,
            baseline=True,
        )
        label = f"test2 {phase} control {fixture_id}"
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
        self.current_baselines[fixture_id] = numeric
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
            source_key=None,
        )
        return numeric

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
    ) -> dict[str, Any] | None:
        self._assert_partition_allowed(case, blind=blind)
        if not self.can_start(deadline):
            return None

        fixture_id = _fixture_id(case)
        baseline = self.current_baselines.get(fixture_id)
        if baseline is None:
            baseline = self.control(case, deadline, phase=phase, blind=blind)
        elif self.treatment_calls and self.treatment_calls % int(self.cfg["control_interval"]) == 0:
            refreshed = self.control(case, deadline, phase=phase, blind=blind, force=True)
            if refreshed is not None:
                baseline = refreshed
        if baseline is None or not self.can_start(deadline):
            return None

        ids = [str(value) for value in recipe.get("ingredient_ids") or []]
        if not ids:
            return None
        self.sequence += 1
        spec = _spec(
            sequence=self.sequence,
            case=case,
            label=label,
            cfg=self.cfg,
            baseline=False,
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
        source_key: str | None,
    ) -> dict[str, Any]:
        score = row.get("score")
        valid = (row.get("classification") or {}).get("valid_for_capability") is True
        numeric = (
            float(score)
            if valid and isinstance(score, (int, float)) and not isinstance(score, bool)
            else 0.0
        )
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
            "score": numeric,
            "baseline_score": baseline_score,
            "delta": numeric - baseline_score,
            "recipe": copy.deepcopy(recipe),
            "source_key": source_key,
            "timing": copy.deepcopy(row.get("timing") or {}),
            "evidence_refs": copy.deepcopy(row.get("evidence_refs") or {}),
        }
        self.rows.append(record)
        self.runner.store.append_jsonl("test2-observations.jsonl", record)
        return record


def _effect_map(
    rows: Iterable[dict[str, Any]],
    *,
    noise_sigma: float,
    bootstrap_samples: int,
    key_fn: Callable[[dict[str, Any]], str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    samples: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("kind") == "control":
            continue
        key = key_fn(row)
        grouped[key].append(float(row.get("delta", 0.0)))
        samples.setdefault(key, copy.deepcopy(row.get("recipe") or {}))
    result: dict[str, dict[str, Any]] = {}
    for key, deltas in grouped.items():
        summary = effect_summary(
            deltas,
            noise_sigma,
            bootstrap_samples=bootstrap_samples,
            seed_key=f"test2:{key}",
        )
        summary["recipe"] = samples[key]
        result[key] = summary
    return result


def _recipe_key(recipe: dict[str, Any]) -> str:
    ids = [str(value) for value in recipe.get("ingredient_ids") or []]
    return "->".join(ids)


def _recurrence_variants(recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def phase_recurrence(campaign: Test2Campaign, deadline: float) -> dict[str, dict[str, Any]]:
    variants = _recurrence_variants(campaign.recipes)
    fixtures = _balanced_cases(
        campaign.partitions["VALIDATION"],
        len(campaign.partitions["VALIDATION"]),
    )
    if not fixtures:
        fixtures = _balanced_cases(campaign.partitions["DISCOVERY"], len(campaign.partitions["DISCOVERY"]))
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
    return _effect_map(
        [row for row in campaign.rows if row["phase"] == "recurrence_higher_order"],
        noise_sigma=campaign.noise_sigma,
        bootstrap_samples=int(campaign.cfg["bootstrap_samples"]),
        key_fn=lambda row: _recipe_key(row["recipe"]),
    )


def _recovery_candidates(campaign: Test2Campaign, case: dict[str, Any]) -> list[dict[str, Any]]:
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
    if not recipes:
        recipes = [copy.deepcopy(row) for row in campaign.recipes[:4]]

    fixtures = _balanced_cases(campaign.partitions["VALIDATION"], len(campaign.partitions["VALIDATION"]))
    cursor = 0
    while recipes and fixtures and campaign.can_start(deadline):
        recipe = recipes[cursor % len(recipes)]
        case = fixtures[(cursor // len(recipes)) % len(fixtures)]
        campaign.treatment(
            case,
            deadline,
            phase="negative_transfer",
            kind="negative_transfer",
            recipe=recipe,
            label="transfer-" + _recipe_key(recipe),
            source_key="recurrence-map",
        )
        cursor += 1
        if cursor >= len(recipes) * len(fixtures):
            break

    effects = _effect_map(
        [row for row in campaign.rows if row["phase"] == "negative_transfer"],
        noise_sigma=campaign.noise_sigma,
        bootstrap_samples=int(campaign.cfg["bootstrap_samples"]),
        key_fn=lambda row: f"{_recipe_key(row['recipe'])}|family={row['family_id']}",
    )
    boundaries: dict[str, Any] = {}
    for key, summary in effects.items():
        if summary["classification"] == "HARMFUL":
            label = "NEGATIVE_TRANSFER"
        elif summary["classification"] in {"STRONG", "PROMISING"}:
            label = "DOMAIN_POSITIVE"
        elif summary["classification"] == "NULL":
            label = "NEUTRAL"
        else:
            label = "CONTEXT_DEPENDENT"
        boundaries[key] = {**copy.deepcopy(summary), "boundary_class": label}
    return boundaries


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
            if result_class != "ANSWER_CORRECT" or float(observation.get("delta", 0.0)) < (-0.5 * campaign.noise_sigma):
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
            if row is not None:
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
        key_fn=lambda row: _recipe_key(row["recipe"]),
    )
    return {
        "recipes_frozen_before_phase": True,
        "fixture_count": len(fixtures),
        "effects": effects,
        "observations": len(blind_rows),
    }


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
    for row in campaign.rows:
        if row.get("partition") == "TEST3_PROTECTED":
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
        valid_rows = [
            row for row in rows
            if (row.get("classification") or {}).get("valid_for_capability") is True
        ]
        invalid_rows = [row for row in rows if row not in valid_rows]
        recoveries = recovery_by_phenotype.get(phenotype, [])
        general = [row for row in recoveries if row.get("status") == "GENERAL_RECOVERY"]
        best_general = max(
            general,
            key=lambda row: float(row.get("success_rate", 0.0)),
            default=None,
        )
        cases = [campaign.case_by_id.get(fixture_id) for fixture_id in fixture_ids]
        observable_target = all(
            case is not None and case.get("scorer") is not None and "expected" in case
            for case in cases
        )
        levels = [
            int(case.get("difficulty_level", 0))
            for case in cases
            if case is not None
        ]

        if invalid_rows and not valid_rows:
            toolish = any("tool" in family.lower() for family in family_ids)
            owner = "TOOL_SOLVABLE" if toolish else "SYSTEM_SOLVABLE"
        elif not observable_target:
            owner = "DATA_SOLVABLE"
        elif best_general is not None:
            recipe_len = len((best_general.get("recipe") or {}).get("ingredient_ids") or [])
            owner = "PROMPT_SOLVABLE" if recipe_len <= 1 else "RECIPE_SOLVABLE"
        elif len(fixture_ids) >= min_failures:
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
            "independent_fixture_count": len(fixture_ids),
            "fixture_ids": fixture_ids,
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
        }
        limits[phenotype] = entry

        if owner == "FINE_TUNING_CANDIDATE":
            package = {
                **copy.deepcopy(entry),
                "qualification": {
                    "recurrent": True,
                    "independent": len(fixture_ids) >= min_failures,
                    "model_owned": True,
                    "prior_generalization_evidence": len(fixture_ids) >= min_failures,
                    "cheaper_prompt_recipe_owner_resolved": best_general is None,
                    "tool_system_owner_resolved": not invalid_rows,
                    "observable_target": observable_target,
                    "protected_fixtures_excluded": True,
                    "negative_transfer_evidence_available": negative_transfer_available,
                    "leakage_safe_partitioning": True,
                },
                "next_action": "TEST3_FINE_TUNING_QUALIFICATION",
            }
            finetune.append(package)
            for fixture_id in fixture_ids:
                case = campaign.case_by_id.get(fixture_id)
                if case is None or campaign._partition(case) == "TEST3_PROTECTED":
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
                    "train_eligible": campaign._partition(case) in {"DISCOVERY", "VALIDATION"},
                    "source_experiment_ids": [
                        str(row.get("experiment_id"))
                        for row in rows
                        if row.get("fixture_id") == fixture_id and row.get("experiment_id")
                    ],
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
) -> list[dict[str, Any]]:
    blind_effects = blind.get("effects") or {}
    result: list[dict[str, Any]] = []
    for source_key, row in knockouts.items():
        recipe = copy.deepcopy(row.get("minimal_recipe") or {})
        if not recipe:
            continue
        key = _recipe_key(recipe)
        blind_summary = blind_effects.get(key)
        result.append({
            "recipe_id": "REC-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12],
            "source_recipe_key": source_key,
            "recipe_key": key,
            "recipe": recipe,
            "blind_summary": copy.deepcopy(blind_summary),
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
    recipes = _final_recipe_registry(knockouts, blind)
    primary = recipes[0] if recipes else {
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
    rendered_parts = []
    for ingredient_id in primary_recipe.get("ingredient_ids") or []:
        definition = ingredient_index.get(str(ingredient_id))
        if definition is None:
            continue
        rendered_parts.append(str(definition.get("full") or definition.get("short") or ""))
    rendered_control_text = "\n".join(rendered_parts)

    exact_model = {
        "model": str(campaign.runner.model),
        "thinking_mode": bool(campaign.cfg["thinking_mode"]),
        "reasoning_effort": campaign.cfg.get("reasoning_effort"),
        "generation_budget": int(campaign.cfg["generation_budget"]),
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
        "rendering_rule": {
            "ingredient_order_is_semantic": True,
            "placement": primary_recipe.get("placement"),
            "representation": primary_recipe.get("representation"),
            "dose": primary_recipe.get("dose"),
            "recurrence_count": len(primary_recipe.get("ingredient_ids") or []),
        },
        "alternate_minimal_recipes": copy.deepcopy(recipes[1:]),
        "ingredient_definitions": copy.deepcopy(list(INGREDIENTS)),
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

    minimal = _final_recipe_registry(knockouts, blind)
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
        },
        producer="test2",
        stage="report",
    )
    store.write_json(
        "test3-dataset-manifest.json",
        {
            "schema_version": 1,
            "train_eligible_examples": dataset,
            "protected_fixture_ids": protected_ids,
            "candidate_phenotypes": [row["phenotype_id"] for row in finetune],
            "leakage_rule": "TEST3_PROTECTED fixtures and siblings derived from them are forbidden from training",
        },
        producer="test2",
        stage="report",
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
