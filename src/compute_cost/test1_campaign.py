"""Seven-hour GPT-20B Test-1 discovery and interaction campaign.

This module reuses the proven compute-cost execution/evidence path while changing
only the scientific scheduler.  It never touches TEST2_BLIND or TEST3_PROTECTED
fixtures and is deliberately wall-clock governed rather than call-count governed.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable

from .characterization import execute_experiment
from .experiments import ExperimentSpec, make_experiment_id

ACTIVE_SECONDS = 6 * 3600 + 50 * 60
HARD_SECONDS = 7 * 3600
CALL_START_CUTOFF_SECONDS = 6 * 3600 + 45 * 60
EPSILON_NOISE = 0.05

PHASES = (
    ("calibration", 30 * 60),
    ("broad_discovery", 110 * 60),
    ("dose_representation_placement", 80 * 60),
    ("pair_order_interactions", 100 * 60),
    ("higher_order_scout", 50 * 60),
    ("confirmation_uncertainty", 40 * 60),
)

PARTITIONS = (
    ("DISCOVERY", 55),
    ("VALIDATION", 75),
    ("TEST2_BLIND", 90),
    ("TEST3_PROTECTED", 100),
)

DEFAULT_TEST1_CONFIG: dict[str, Any] = {
    "expected_calls": 4300,
    "safety_call_cap": 10000,
    "generation_budget": 256,
    "thinking_mode": False,
    "reasoning_effort": None,
    "calibration_anchor_count": 64,
    "calibration_repeats": 5,
    "discovery_min_observations": 16,
    "discovery_max_observations": 64,
    "characterization_top_ingredients": 6,
    "pair_top_ingredients": 8,
    "confirmation_candidates": 24,
    "bootstrap_samples": 500,
}

INGREDIENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "ING-001",
        "family": "constraint_control",
        "label": "explicit_success_criteria",
        "short": "Satisfy every explicit requirement.",
        "full": "Identify every explicit requirement in the task and ensure the final answer satisfies all of them.",
    },
    {
        "id": "ING-002",
        "family": "planning",
        "label": "minimal_decomposition",
        "short": "Break the task into the minimum necessary steps.",
        "full": "Decompose the task into the minimum necessary dependent steps before producing the final answer.",
    },
    {
        "id": "ING-003",
        "family": "verification",
        "label": "preanswer_verification",
        "short": "Verify the answer before returning it.",
        "full": "Before returning the final answer, verify the result against the task requirements and correct any detected error.",
    },
    {
        "id": "ING-004",
        "family": "evidence",
        "label": "evidence_grounding",
        "short": "Use only evidence actually provided or derived.",
        "full": "Ground the answer only in evidence provided by the task or directly derived from it; do not invent missing facts.",
    },
    {
        "id": "ING-005",
        "family": "state",
        "label": "latest_state_priority",
        "short": "Prefer the latest authoritative state over obsolete state.",
        "full": "When facts conflict over time, use the latest authoritative state and explicitly reject superseded or obsolete state.",
    },
    {
        "id": "ING-006",
        "family": "consistency",
        "label": "contradiction_check",
        "short": "Check for contradictions before answering.",
        "full": "Check the relevant facts and constraints for contradictions before committing to the final answer.",
    },
    {
        "id": "ING-007",
        "family": "format",
        "label": "output_contract",
        "short": "Obey the requested output format exactly.",
        "full": "Treat the requested output shape, keys, types, ordering, and no-extra-prose requirements as a strict contract.",
    },
    {
        "id": "ING-008",
        "family": "reasoning",
        "label": "independent_recompute",
        "short": "Recompute critical results independently.",
        "full": "Independently recompute the critical result once before finalizing instead of trusting the first intermediate result.",
    },
    {
        "id": "ING-009",
        "family": "critique",
        "label": "counterexample_check",
        "short": "Look for one counterexample to your proposed answer.",
        "full": "Before finalizing, actively look for a counterexample or condition that would make the proposed answer wrong.",
    },
    {
        "id": "ING-010",
        "family": "instruction_following",
        "label": "negative_constraint_scan",
        "short": "Check prohibitions and negative constraints.",
        "full": "Scan specifically for prohibitions, exclusions, and negative constraints and ensure none are violated.",
    },
    {
        "id": "ING-011",
        "family": "tool_use",
        "label": "tool_argument_validation",
        "short": "Validate tool choice, arguments, types, and dependency order.",
        "full": "For tool-like tasks, validate the selected tool, exact argument values and types, and dependency order before returning the call.",
    },
    {
        "id": "ING-012",
        "family": "arithmetic",
        "label": "arithmetic_sanity_check",
        "short": "Sanity-check arithmetic and units.",
        "full": "After computing a numerical result, sanity-check the arithmetic, sign, scale, and units before finalizing.",
    },
    {
        "id": "ING-013",
        "family": "logic",
        "label": "premise_conclusion_trace",
        "short": "Trace premises to the conclusion without adding assumptions.",
        "full": "Trace the required conclusion from the supplied premises step by step and reject any unstated assumption.",
    },
    {
        "id": "ING-014",
        "family": "code",
        "label": "execution_trace",
        "short": "Trace the code's actual execution before diagnosing it.",
        "full": "Trace the code's actual execution state before naming the root cause or proposing the corrected result.",
    },
    {
        "id": "ING-015",
        "family": "uncertainty",
        "label": "ambiguity_resolution",
        "short": "Resolve ambiguity from the strongest explicit evidence.",
        "full": "If the task appears ambiguous, resolve it using the strongest explicit evidence and constraints rather than guessing.",
    },
    {
        "id": "ING-016",
        "family": "finalization",
        "label": "answer_only_after_checks",
        "short": "Do not finalize until required checks pass.",
        "full": "Delay the final answer until constraint, evidence, computation, and format checks applicable to this task have passed.",
    },
)


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_TEST1_CONFIG)
    incoming = config.get("test1_campaign")
    if isinstance(incoming, dict):
        merged.update(copy.deepcopy(incoming))
    return merged


def _fixture_id(case: dict[str, Any]) -> str:
    return str(case.get("id") or "")


def _family(case: dict[str, Any]) -> str:
    return str(case.get("family_id") or case.get("category") or "unknown")


def partition_cases(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {name: [] for name, _ in PARTITIONS}
    for case in cases:
        digest = hashlib.sha256(_fixture_id(case).encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % 100
        lower = 0
        for name, upper in PARTITIONS:
            if lower <= bucket < upper:
                result[name].append(case)
                break
            lower = upper
    return result


def build_test1_plan(cases: list[dict[str, Any]]) -> dict[str, Any]:
    partitions = partition_cases(cases)
    return {
        "schema_version": 1,
        "campaign": "gpt20b-test1-discovery-interaction-atlas",
        "wall_clock_seconds": HARD_SECONDS,
        "active_model_seconds": ACTIVE_SECONDS,
        "call_start_cutoff_seconds": CALL_START_CUTOFF_SECONDS,
        "phases": [{"name": name, "seconds": seconds} for name, seconds in PHASES],
        "partition_policy": {
            "DISCOVERY": 55,
            "VALIDATION": 20,
            "TEST2_BLIND": 15,
            "TEST3_PROTECTED": 10,
        },
        "partition_counts": {name: len(rows) for name, rows in partitions.items()},
        "ingredient_count": len(INGREDIENTS),
        "ingredient_ids": [item["id"] for item in INGREDIENTS],
        "prohibited_partitions": ["TEST2_BLIND", "TEST3_PROTECTED"],
        "required_outputs": [
            "noise-model.json",
            "ingredient-registry.json",
            "dose-response-map.json",
            "representation-map.json",
            "placement-map.json",
            "pair-interaction-graph.json",
            "directional-order-graph.json",
            "higher-order-candidate-queue.json",
            "failure-registry.json",
            "negative-effect-registry.json",
            "uncertainty-ledger.json",
            "test2-priority-queue.json",
            "fixture-partitions.json",
        ],
    }


def validate_test1_plan(plan: dict[str, Any]) -> None:
    if sum(int(row["seconds"]) for row in plan["phases"]) != ACTIVE_SECONDS:
        raise ValueError("Test 1 active phase durations must total 6h50m")
    if plan["wall_clock_seconds"] != HARD_SECONDS:
        raise ValueError("Test 1 wall clock must be exactly seven hours")
    ids = list(plan["ingredient_ids"])
    if len(ids) != len(set(ids)):
        raise ValueError("ingredient IDs must be unique")
    counts = plan["partition_counts"]
    if counts.get("DISCOVERY", 0) <= 0:
        raise ValueError("DISCOVERY partition is empty")
    if counts.get("VALIDATION", 0) <= 0:
        raise ValueError("VALIDATION partition is empty")
    if counts.get("TEST2_BLIND", 0) <= 0:
        raise ValueError("TEST2_BLIND partition is empty")
    if counts.get("TEST3_PROTECTED", 0) <= 0:
        raise ValueError("TEST3_PROTECTED partition is empty")


def _render_ingredient(item: dict[str, Any], dose: float, representation: str) -> str:
    base = str(item["short"] if dose <= 0.5 else item["full"])
    if dose >= 2.0:
        base = base + " Treat this check as mandatory before the final answer."
    if representation == "bullets":
        return "- " + base
    if representation == "schema":
        return json.dumps(
            {"control": item["label"], "requirement": base, "mandatory": True},
            sort_keys=True,
            separators=(",", ":"),
        )
    return base


def build_treatment_messages(
    case: dict[str, Any],
    ingredient_ids: list[str],
    *,
    dose: float = 1.0,
    representation: str = "prose",
    placement: str = "prefix",
) -> list[dict[str, str]]:
    index = {item["id"]: item for item in INGREDIENTS}
    selected = [index[value] for value in ingredient_ids]
    rendered = [_render_ingredient(item, dose, representation) for item in selected]
    treatment = "\n".join(rendered)
    prompt = str(case["prompt"])

    if placement == "system":
        return [
            {"role": "system", "content": treatment},
            {"role": "user", "content": prompt},
        ]
    if placement == "suffix":
        user = prompt + "\n\nADDITIONAL CONTROL:\n" + treatment
    elif placement == "middle":
        words = prompt.split()
        pivot = max(1, len(words) // 2)
        user = " ".join(words[:pivot]) + "\n\nCONTROL:\n" + treatment + "\n\n" + " ".join(words[pivot:])
    else:
        user = "CONTROL:\n" + treatment + "\n\nTASK:\n" + prompt
    return [{"role": "user", "content": user}]


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    center = median(values)
    return median([abs(value - center) for value in values])


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = fraction * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(len(sorted_values) - 1, lower + 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def effect_summary(
    deltas: list[float],
    noise_sigma: float,
    *,
    bootstrap_samples: int = 500,
    seed_key: str = "",
) -> dict[str, Any]:
    sigma = max(float(noise_sigma), EPSILON_NOISE)
    if not deltas:
        return {
            "n": 0,
            "median_delta": 0.0,
            "normalized_effect": 0.0,
            "probability_positive": 0.0,
            "ci80": [0.0, 0.0],
            "win_rate": 0.0,
            "classification": "UNTESTED",
        }

    center = float(median(deltas))
    wins = sum(1 for value in deltas if value > 0)
    win_rate = wins / len(deltas)
    if len(deltas) < 2:
        boot = [center]
    else:
        digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        boot = []
        for _ in range(max(20, int(bootstrap_samples))):
            sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
            boot.append(float(median(sample)))
    boot.sort()
    probability_positive = sum(1 for value in boot if value > 0) / len(boot)
    lower = _quantile(boot, 0.10)
    upper = _quantile(boot, 0.90)
    normalized = center / sigma

    severe_negative = any(value < (-2.0 * sigma) for value in deltas)
    if upper < (-0.5 * sigma) or severe_negative:
        classification = "HARMFUL"
    elif lower >= (-0.5 * sigma) and upper <= (0.5 * sigma):
        classification = "NULL"
    elif lower > (0.5 * sigma) and win_rate >= 0.65:
        classification = "STRONG"
    elif center > (0.5 * sigma) and probability_positive >= 0.80:
        classification = "PROMISING"
    else:
        classification = "UNCERTAIN"

    return {
        "n": len(deltas),
        "median_delta": center,
        "normalized_effect": normalized,
        "probability_positive": probability_positive,
        "ci80": [lower, upper],
        "win_rate": win_rate,
        "classification": classification,
    }


def _balanced_cases(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in sorted(cases, key=lambda row: (_family(row), int(row.get("difficulty_level", 0)), _fixture_id(row))):
        by_family[_family(case)].append(case)
    result: list[dict[str, Any]] = []
    families = sorted(by_family)
    cursor = 0
    while families and len(result) < limit:
        family = families[cursor % len(families)]
        bucket = by_family[family]
        if bucket:
            result.append(bucket.pop(0))
        if not bucket:
            families.remove(family)
            cursor = 0
        else:
            cursor += 1
    return result


def _historical_baselines(results_root: Path, run_id: str | None) -> dict[str, dict[str, Any]]:
    if not run_id:
        return {}
    run_dir = results_root / run_id
    obs_path = run_dir / "capability-observations.jsonl"
    exp_path = run_dir / "experiments.jsonl"
    if not obs_path.is_file() or not exp_path.is_file():
        return {}

    fixture_by_experiment: dict[str, str] = {}
    for raw in obs_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        experiment_id = row.get("experiment_id")
        fixture_id = row.get("fixture_id")
        if isinstance(experiment_id, str) and isinstance(fixture_id, str):
            fixture_by_experiment[experiment_id] = fixture_id

    values: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for raw in exp_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        experiment = row.get("experiment") or {}
        experiment_id = experiment.get("experiment_id")
        fixture_id = fixture_by_experiment.get(str(experiment_id))
        score = row.get("score")
        valid = (row.get("classification") or {}).get("valid_for_capability") is True
        if fixture_id and valid and isinstance(score, (int, float)) and not isinstance(score, bool):
            values[fixture_id].append((float(score), str(experiment_id)))

    result: dict[str, dict[str, Any]] = {}
    for fixture_id, rows in values.items():
        result[fixture_id] = {
            "score": float(median([value for value, _ in rows])),
            "experiment_id": rows[-1][1],
            "source": "historical",
        }
    return result


def _spec(
    *,
    sequence: int,
    case: dict[str, Any],
    label: str,
    parent_experiment_id: str | None,
    baseline: bool,
    cfg: dict[str, Any],
) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=make_experiment_id(sequence, _fixture_id(case), label),
        parent_experiment_id=parent_experiment_id,
        task_id=_fixture_id(case),
        task_family=_family(case),
        difficulty_level=int(case.get("difficulty_level", 0)),
        hypothesis="Test-1 controlled treatment effect",
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


class _Campaign:
    def __init__(
        self,
        runner: Any,
        cases: list[dict[str, Any]],
        *,
        baseline_run: str | None,
        clock: Callable[[], float],
    ) -> None:
        self.runner = runner
        self.cases = cases
        self.clock = clock
        self.cfg = _cfg(runner.config)
        self.start = clock()
        self.call_start_cutoff = self.start + CALL_START_CUTOFF_SECONDS
        self.active_end = self.start + ACTIVE_SECONDS
        self.partitions = partition_cases(cases)
        self.sequence = 0
        self.rows: list[dict[str, Any]] = []
        self.baselines = _historical_baselines(Path(runner.results_root), baseline_run)
        self.live_baseline_specs: dict[str, ExperimentSpec] = {}
        self.noise_sigma = EPSILON_NOISE

    def can_start(self, deadline: float) -> bool:
        now = self.clock()
        return now < min(deadline, self.call_start_cutoff, self.active_end)

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
                    reason="wall-clock campaign capacity extended",
                )
            self.runner._progress_begin(label)
        else:
            self.runner._progress_complete(label)

    def baseline(self, case: dict[str, Any], deadline: float, *, calibration: bool = False) -> dict[str, Any] | None:
        fixture_id = _fixture_id(case)
        if not calibration and fixture_id in self.baselines:
            return self.baselines[fixture_id]
        if not self.can_start(deadline):
            return None
        self.sequence += 1
        spec = _spec(
            sequence=self.sequence,
            case=case,
            label="base",
            parent_experiment_id=None,
            baseline=True,
            cfg=self.cfg,
        )
        label = f"test1 baseline {fixture_id}"
        self._progress(label, True)
        try:
            row = execute_experiment(self.runner, case, spec, parent=None)
        finally:
            self._progress(label, False)
        score = row.get("score")
        valid = (row.get("classification") or {}).get("valid_for_capability") is True
        record = {
            "score": float(score) if valid and isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0,
            "experiment_id": spec.experiment_id,
            "source": "test1",
        }
        if not calibration:
            self.baselines[fixture_id] = record
            self.live_baseline_specs[fixture_id] = spec
        self._record(
            case,
            row,
            phase="calibration" if calibration else "baseline",
            kind="baseline",
            treatment={},
            baseline_score=record["score"],
        )
        return record

    def treatment(
        self,
        case: dict[str, Any],
        deadline: float,
        *,
        phase: str,
        kind: str,
        ingredient_ids: list[str],
        dose: float = 1.0,
        representation: str = "prose",
        placement: str = "prefix",
        treatment_label: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.can_start(deadline):
            return None
        baseline = self.baseline(case, deadline)
        if baseline is None or not self.can_start(deadline):
            return None
        fixture_id = _fixture_id(case)
        parent = self.live_baseline_specs.get(fixture_id)
        self.sequence += 1
        label_value = treatment_label or (
            kind + "-" + "-".join(ingredient_ids) + f"-d{dose:g}-{representation}-{placement}"
        )
        spec = _spec(
            sequence=self.sequence,
            case=case,
            label=label_value,
            parent_experiment_id=str(baseline.get("experiment_id") or "") or None,
            baseline=False,
            cfg=self.cfg,
        )
        messages = build_treatment_messages(
            case,
            ingredient_ids,
            dose=dose,
            representation=representation,
            placement=placement,
        )
        progress_label = f"test1 {phase} {fixture_id} {label_value}"
        self._progress(progress_label, True)
        try:
            row = execute_experiment(
                self.runner,
                case,
                spec,
                parent=parent,
                messages_override=messages,
            )
        finally:
            self._progress(progress_label, False)
        treatment = {
            "ingredient_ids": list(ingredient_ids),
            "dose": dose,
            "representation": representation,
            "placement": placement,
            "label": label_value,
        }
        return self._record(
            case,
            row,
            phase=phase,
            kind=kind,
            treatment=treatment,
            baseline_score=float(baseline["score"]),
        )

    def _record(
        self,
        case: dict[str, Any],
        row: dict[str, Any],
        *,
        phase: str,
        kind: str,
        treatment: dict[str, Any],
        baseline_score: float,
    ) -> dict[str, Any]:
        score = row.get("score")
        valid = (row.get("classification") or {}).get("valid_for_capability") is True
        numeric_score = (
            float(score)
            if valid and isinstance(score, (int, float)) and not isinstance(score, bool)
            else 0.0
        )
        delta = numeric_score - baseline_score
        observation = {
            "schema_version": 1,
            "timestamp_utc": self.runner._utc(),
            "phase": phase,
            "kind": kind,
            "fixture_id": _fixture_id(case),
            "family_id": _family(case),
            "difficulty_level": int(case.get("difficulty_level", 0)),
            "partition": self._partition_name(case),
            "experiment_id": (row.get("experiment") or {}).get("experiment_id"),
            "classification": copy.deepcopy(row.get("classification") or {}),
            "score": numeric_score,
            "baseline_score": baseline_score,
            "delta": delta,
            "treatment": copy.deepcopy(treatment),
            "evidence_refs": copy.deepcopy(row.get("evidence_refs") or {}),
        }
        self.rows.append(observation)
        self.runner.store.append_jsonl("test1-observations.jsonl", observation)
        return observation

    def _partition_name(self, case: dict[str, Any]) -> str:
        fixture_id = _fixture_id(case)
        for name, rows in self.partitions.items():
            if any(_fixture_id(row) == fixture_id for row in rows):
                return name
        return "UNKNOWN"


def _calibration(campaign: _Campaign, deadline: float) -> None:
    cfg = campaign.cfg
    anchors = _balanced_cases(
        campaign.partitions["DISCOVERY"],
        min(int(cfg["calibration_anchor_count"]), len(campaign.partitions["DISCOVERY"])),
    )
    scores_by_family: dict[str, list[float]] = defaultdict(list)
    repeats = int(cfg["calibration_repeats"])
    for repeat in range(repeats):
        for case in anchors:
            if not campaign.can_start(deadline):
                break
            record = campaign.baseline(case, deadline, calibration=True)
            if record is not None:
                scores_by_family[_family(case)].append(float(record["score"]))
        if not campaign.can_start(deadline):
            break

    sigmas: list[float] = []
    families: dict[str, Any] = {}
    for family, values in sorted(scores_by_family.items()):
        sigma = max(EPSILON_NOISE, 1.4826 * _mad(values))
        sigmas.append(sigma)
        families[family] = {
            "n": len(values),
            "median_score": float(median(values)) if values else None,
            "mad": _mad(values),
            "noise_sigma": sigma,
        }
    campaign.noise_sigma = max(EPSILON_NOISE, float(median(sigmas)) if sigmas else EPSILON_NOISE)
    campaign.runner.store.write_json(
        "noise-model.json",
        {
            "schema_version": 1,
            "global_noise_sigma": campaign.noise_sigma,
            "epsilon_floor": EPSILON_NOISE,
            "families": families,
        },
        producer="test1",
        stage="calibration",
    )


def _group_effects(
    rows: list[dict[str, Any]],
    *,
    key_fn: Callable[[dict[str, Any]], str],
    noise_sigma: float,
    bootstrap_samples: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    examples: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["kind"] == "baseline":
            continue
        key = key_fn(row)
        grouped[key].append(float(row["delta"]))
        examples.setdefault(key, copy.deepcopy(row["treatment"]))
    result: dict[str, dict[str, Any]] = {}
    for key, deltas in grouped.items():
        summary = effect_summary(
            deltas,
            noise_sigma,
            bootstrap_samples=bootstrap_samples,
            seed_key=key,
        )
        summary["treatment"] = examples[key]
        result[key] = summary
    return result


def _ingredient_key(row: dict[str, Any]) -> str:
    return "+".join(row["treatment"].get("ingredient_ids") or [])


def _broad_discovery(campaign: _Campaign, deadline: float) -> dict[str, dict[str, Any]]:
    discovery = _balanced_cases(campaign.partitions["DISCOVERY"], len(campaign.partitions["DISCOVERY"]))
    min_n = int(campaign.cfg["discovery_min_observations"])
    max_n = int(campaign.cfg["discovery_max_observations"])
    bootstrap = int(campaign.cfg["bootstrap_samples"])
    counts = {item["id"]: 0 for item in INGREDIENTS}
    active = [item["id"] for item in INGREDIENTS]
    cursor = 0

    while active and campaign.can_start(deadline):
        ingredient_id = active[cursor % len(active)]
        case = discovery[counts[ingredient_id] % len(discovery)]
        campaign.treatment(
            case,
            deadline,
            phase="broad_discovery",
            kind="single",
            ingredient_ids=[ingredient_id],
        )
        counts[ingredient_id] += 1

        if counts[ingredient_id] >= min_n:
            summaries = _group_effects(
                [row for row in campaign.rows if row["phase"] == "broad_discovery"],
                key_fn=_ingredient_key,
                noise_sigma=campaign.noise_sigma,
                bootstrap_samples=bootstrap,
            )
            state = summaries.get(ingredient_id, {}).get("classification")
            if state in {"NULL", "HARMFUL"} or counts[ingredient_id] >= max_n:
                active.remove(ingredient_id)
                cursor = 0
                continue
        cursor += 1

    return _group_effects(
        [row for row in campaign.rows if row["phase"] == "broad_discovery"],
        key_fn=_ingredient_key,
        noise_sigma=campaign.noise_sigma,
        bootstrap_samples=bootstrap,
    )


def _rank_ingredients(summaries: dict[str, dict[str, Any]]) -> list[str]:
    rank = {"STRONG": 4, "PROMISING": 3, "UNCERTAIN": 2, "NULL": 1, "HARMFUL": 0}
    return [
        key
        for key, _ in sorted(
            summaries.items(),
            key=lambda item: (
                rank.get(str(item[1].get("classification")), -1),
                float(item[1].get("normalized_effect", 0.0)),
                int(item[1].get("n", 0)),
            ),
            reverse=True,
        )
    ]


def _characterize_controls(
    campaign: _Campaign,
    deadline: float,
    discovery_summary: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    top = _rank_ingredients(discovery_summary)[: int(campaign.cfg["characterization_top_ingredients"])]
    fixtures = _balanced_cases(campaign.partitions["DISCOVERY"], min(96, len(campaign.partitions["DISCOVERY"])))
    variants = list(itertools.product((0.5, 1.0, 2.0), ("prose", "bullets", "schema"), ("prefix", "middle", "suffix", "system")))
    for ingredient_id in top:
        for dose, representation, placement in variants:
            for case in fixtures[:6]:
                if not campaign.can_start(deadline):
                    break
                campaign.treatment(
                    case,
                    deadline,
                    phase="dose_representation_placement",
                    kind="characterization",
                    ingredient_ids=[ingredient_id],
                    dose=float(dose),
                    representation=str(representation),
                    placement=str(placement),
                )
            if not campaign.can_start(deadline):
                break
        if not campaign.can_start(deadline):
            break

    def key(row: dict[str, Any]) -> str:
        t = row["treatment"]
        return f"{'+'.join(t['ingredient_ids'])}|d={t['dose']}|r={t['representation']}|p={t['placement']}"

    return _group_effects(
        [row for row in campaign.rows if row["phase"] == "dose_representation_placement"],
        key_fn=key,
        noise_sigma=campaign.noise_sigma,
        bootstrap_samples=int(campaign.cfg["bootstrap_samples"]),
    )


def _best_variant_by_ingredient(characterized: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for summary in characterized.values():
        treatment = summary.get("treatment") or {}
        ids = treatment.get("ingredient_ids") or []
        if len(ids) != 1:
            continue
        ingredient_id = ids[0]
        current = result.get(ingredient_id)
        if current is None or float(summary.get("normalized_effect", 0.0)) > float(current.get("normalized_effect", 0.0)):
            result[ingredient_id] = summary
    return result


def _pair_order(
    campaign: _Campaign,
    deadline: float,
    discovery_summary: dict[str, dict[str, Any]],
    characterized: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    ranked = _rank_ingredients(discovery_summary)[: int(campaign.cfg["pair_top_ingredients"])]
    best = _best_variant_by_ingredient(characterized)
    fixtures = _balanced_cases(campaign.partitions["DISCOVERY"], min(128, len(campaign.partitions["DISCOVERY"])))

    for a, b in itertools.combinations(ranked, 2):
        base_treatment = best.get(a, {}).get("treatment") or {"dose": 1.0, "representation": "prose", "placement": "prefix"}
        for order, label in (([a, b], "AB"), ([b, a], "BA")):
            for case in fixtures[:12]:
                if not campaign.can_start(deadline):
                    break
                campaign.treatment(
                    case,
                    deadline,
                    phase="pair_order_interactions",
                    kind="pair",
                    ingredient_ids=order,
                    dose=float(base_treatment.get("dose", 1.0)),
                    representation=str(base_treatment.get("representation", "prose")),
                    placement=str(base_treatment.get("placement", "prefix")),
                    treatment_label=f"pair-{a}-{b}-{label}",
                )
            if not campaign.can_start(deadline):
                break
        if not campaign.can_start(deadline):
            break

    def key(row: dict[str, Any]) -> str:
        return "->".join(row["treatment"].get("ingredient_ids") or [])

    return _group_effects(
        [row for row in campaign.rows if row["phase"] == "pair_order_interactions"],
        key_fn=key,
        noise_sigma=campaign.noise_sigma,
        bootstrap_samples=int(campaign.cfg["bootstrap_samples"]),
    )


def _higher_order(
    campaign: _Campaign,
    deadline: float,
    discovery_summary: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    ranked = _rank_ingredients(discovery_summary)[:6]
    fixtures = _balanced_cases(campaign.partitions["DISCOVERY"], min(128, len(campaign.partitions["DISCOVERY"])))
    for a, b, c in itertools.combinations(ranked, 3):
        variants = ([a, b, c], [a, b, a], [b, a, b])
        for ingredients in variants:
            for case in fixtures[:8]:
                if not campaign.can_start(deadline):
                    break
                campaign.treatment(
                    case,
                    deadline,
                    phase="higher_order_scout",
                    kind="higher_order",
                    ingredient_ids=list(ingredients),
                    treatment_label="higher-" + "-".join(ingredients),
                )
            if not campaign.can_start(deadline):
                break
        if not campaign.can_start(deadline):
            break

    def key(row: dict[str, Any]) -> str:
        return "->".join(row["treatment"].get("ingredient_ids") or [])

    return _group_effects(
        [row for row in campaign.rows if row["phase"] == "higher_order_scout"],
        key_fn=key,
        noise_sigma=campaign.noise_sigma,
        bootstrap_samples=int(campaign.cfg["bootstrap_samples"]),
    )


def _confirmation(
    campaign: _Campaign,
    deadline: float,
    discovery_summary: dict[str, dict[str, Any]],
    characterized: dict[str, dict[str, Any]],
    pair_summary: dict[str, dict[str, Any]],
    higher_summary: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    pools: list[dict[str, Any]] = []
    for source in (characterized, pair_summary, higher_summary):
        for summary in source.values():
            if summary.get("classification") in {"STRONG", "PROMISING", "UNCERTAIN"}:
                pools.append(summary)
    pools.sort(
        key=lambda row: (
            row.get("classification") == "STRONG",
            row.get("classification") == "PROMISING",
            abs(float(row.get("normalized_effect", 0.0))),
        ),
        reverse=True,
    )
    pools = pools[: int(campaign.cfg["confirmation_candidates"])]
    validation = _balanced_cases(campaign.partitions["VALIDATION"], len(campaign.partitions["VALIDATION"]))
    if not validation:
        return {}

    index = 0
    while pools and campaign.can_start(deadline):
        candidate = pools[index % len(pools)]
        treatment = candidate.get("treatment") or {}
        ingredients = list(treatment.get("ingredient_ids") or [])
        if not ingredients:
            index += 1
            continue
        case = validation[(index // len(pools)) % len(validation)]
        campaign.treatment(
            case,
            deadline,
            phase="confirmation_uncertainty",
            kind="confirmation",
            ingredient_ids=ingredients,
            dose=float(treatment.get("dose", 1.0)),
            representation=str(treatment.get("representation", "prose")),
            placement=str(treatment.get("placement", "prefix")),
            treatment_label="confirm-" + "-".join(ingredients),
        )
        index += 1
        if index >= len(pools) * max(1, min(20, len(validation))):
            break

    def key(row: dict[str, Any]) -> str:
        t = row["treatment"]
        return f"{'->'.join(t.get('ingredient_ids') or [])}|d={t.get('dose')}|r={t.get('representation')}|p={t.get('placement')}"

    return _group_effects(
        [row for row in campaign.rows if row["phase"] == "confirmation_uncertainty"],
        key_fn=key,
        noise_sigma=campaign.noise_sigma,
        bootstrap_samples=int(campaign.cfg["bootstrap_samples"]),
    )


def _write_outputs(
    campaign: _Campaign,
    discovery: dict[str, dict[str, Any]],
    characterized: dict[str, dict[str, Any]],
    pairs: dict[str, dict[str, Any]],
    higher: dict[str, dict[str, Any]],
    confirmation: dict[str, dict[str, Any]],
) -> None:
    store = campaign.runner.store
    registry = []
    for ingredient in INGREDIENTS:
        row = copy.deepcopy(ingredient)
        row["discovery"] = copy.deepcopy(discovery.get(ingredient["id"], {"classification": "UNTESTED"}))
        registry.append(row)
    store.write_json(
        "ingredient-registry.json",
        {"schema_version": 1, "ingredients": registry},
        producer="test1",
        stage="report",
    )

    def filtered_map(field: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, summary in characterized.items():
            treatment = summary.get("treatment") or {}
            value = str(treatment.get(field))
            result.setdefault(value, {})[key] = summary
        return result

    store.write_json("dose-response-map.json", {"schema_version": 1, "groups": filtered_map("dose")}, producer="test1", stage="report")
    store.write_json("representation-map.json", {"schema_version": 1, "groups": filtered_map("representation")}, producer="test1", stage="report")
    store.write_json("placement-map.json", {"schema_version": 1, "groups": filtered_map("placement")}, producer="test1", stage="report")

    single_effect = {
        key: float(value.get("median_delta", 0.0))
        for key, value in discovery.items()
    }
    pair_graph: dict[str, Any] = {}
    direction_graph: dict[str, Any] = {}
    for key, summary in pairs.items():
        parts = key.split("->")
        if len(parts) != 2:
            continue
        a, b = parts
        expected = single_effect.get(a, 0.0) + single_effect.get(b, 0.0)
        observed = float(summary.get("median_delta", 0.0))
        edge = copy.deepcopy(summary)
        edge["expected_additive_delta"] = expected
        edge["interaction_delta"] = observed - expected
        pair_graph[key] = edge
        reverse = pairs.get(f"{b}->{a}")
        if reverse is not None:
            direction_graph[key] = {
                "forward_delta": observed,
                "reverse_delta": float(reverse.get("median_delta", 0.0)),
                "order_delta": observed - float(reverse.get("median_delta", 0.0)),
                "forward_classification": summary.get("classification"),
                "reverse_classification": reverse.get("classification"),
            }

    store.write_json("pair-interaction-graph.json", {"schema_version": 1, "edges": pair_graph}, producer="test1", stage="report")
    store.write_json("directional-order-graph.json", {"schema_version": 1, "edges": direction_graph}, producer="test1", stage="report")

    higher_queue = [
        {"key": key, **copy.deepcopy(summary)}
        for key, summary in higher.items()
        if summary.get("classification") in {"STRONG", "PROMISING", "UNCERTAIN"}
    ]
    higher_queue.sort(key=lambda row: abs(float(row.get("normalized_effect", 0.0))), reverse=True)
    store.write_json(
        "higher-order-candidate-queue.json",
        {"schema_version": 1, "candidates": higher_queue},
        producer="test1",
        stage="report",
    )

    failures = [
        row for row in campaign.rows
        if (row.get("classification") or {}).get("result_class") != "ANSWER_CORRECT"
    ]
    store.write_json(
        "failure-registry.json",
        {"schema_version": 1, "failures": failures},
        producer="test1",
        stage="report",
    )

    all_summaries = {}
    for prefix, source in (
        ("single", discovery),
        ("characterization", characterized),
        ("pair", pairs),
        ("higher", higher),
        ("confirmation", confirmation),
    ):
        for key, summary in source.items():
            all_summaries[f"{prefix}:{key}"] = summary

    harmful = [
        {"key": key, **copy.deepcopy(summary)}
        for key, summary in all_summaries.items()
        if summary.get("classification") == "HARMFUL"
    ]
    uncertain = [
        {
            "key": key,
            **copy.deepcopy(summary),
            "next_action": "GENERALIZATION_TEST" if key.startswith("confirmation:") else "REPLICATE",
        }
        for key, summary in all_summaries.items()
        if summary.get("classification") == "UNCERTAIN"
    ]
    store.write_json("negative-effect-registry.json", {"schema_version": 1, "effects": harmful}, producer="test1", stage="report")
    store.write_json("uncertainty-ledger.json", {"schema_version": 1, "unknowns": uncertain}, producer="test1", stage="report")

    priority = [
        {
            "key": key,
            **copy.deepcopy(summary),
            "next_action": (
                "ESCALATE_LAYERING"
                if key.startswith(("pair:", "higher:"))
                else "GENERALIZATION_TEST"
            ),
        }
        for key, summary in all_summaries.items()
        if summary.get("classification") in {"STRONG", "PROMISING"}
    ]
    priority.extend(uncertain[:32])
    priority.sort(
        key=lambda row: (
            row.get("classification") == "STRONG",
            abs(float(row.get("normalized_effect", 0.0))),
            int(row.get("n", 0)),
        ),
        reverse=True,
    )
    store.write_json("test2-priority-queue.json", {"schema_version": 1, "queue": priority}, producer="test1", stage="report")


def run_test1_campaign(
    runner: Any,
    cases: list[dict[str, Any]],
    *,
    baseline_run: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> list[dict[str, Any]]:
    """Execute the frozen Test-1 campaign without touching protected partitions."""
    assert runner.store is not None
    plan = build_test1_plan(cases)
    validate_test1_plan(plan)
    campaign = _Campaign(runner, cases, baseline_run=baseline_run, clock=clock)

    runner.store.write_json("test1-plan.json", plan, producer="test1", stage="preflight")
    runner.store.write_json(
        "fixture-partitions.json",
        {
            "schema_version": 1,
            "partitions": {
                name: [_fixture_id(case) for case in rows]
                for name, rows in campaign.partitions.items()
            },
        },
        producer="test1",
        stage="preflight",
    )

    carry = 0.0
    discovery: dict[str, dict[str, Any]] = {}
    characterized: dict[str, dict[str, Any]] = {}
    pairs: dict[str, dict[str, Any]] = {}
    higher: dict[str, dict[str, Any]] = {}
    confirmation: dict[str, dict[str, Any]] = {}

    for phase_name, nominal_seconds in PHASES:
        phase_start = clock()
        deadline = min(campaign.active_end, phase_start + nominal_seconds + carry)
        if phase_name == "calibration":
            _calibration(campaign, deadline)
        elif phase_name == "broad_discovery":
            discovery = _broad_discovery(campaign, deadline)
        elif phase_name == "dose_representation_placement":
            characterized = _characterize_controls(campaign, deadline, discovery)
        elif phase_name == "pair_order_interactions":
            pairs = _pair_order(campaign, deadline, discovery, characterized)
        elif phase_name == "higher_order_scout":
            higher = _higher_order(campaign, deadline, discovery)
        elif phase_name == "confirmation_uncertainty":
            confirmation = _confirmation(
                campaign,
                deadline,
                discovery,
                characterized,
                pairs,
                higher,
            )
        phase_end = clock()
        carry = max(0.0, deadline - phase_end)
        runner.store.append_jsonl(
            "test1-phase-events.jsonl",
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

    _write_outputs(campaign, discovery, characterized, pairs, higher, confirmation)
    return campaign.rows
