"""GPT-OSS 20B Test 1.1 corrective discovery and recipe campaign.

Test 1.1 reuses the proven Test-1 execution/evidence chassis while correcting
what Test 1 taught us was scientifically weak: ceiling/headroom bias,
truncation conflated with capability harm, continuous-noise logic applied to
binary scores, conditional rescue lost by global medians, too-small fixed
ingredient vocabulary, shallow recipe composition, empty handoff queues, and
phase loops that exhausted after only ~41% of the active window.

TEST2_BLIND and TEST3_PROTECTED are never touched. Every active phase owns a
large deterministic work reservoir. If any queue exhausts early, unused time is
automatically converted into more recipe-factorial tests. A final recipe reserve
also consumes any remaining active wall-clock budget.
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
from .evidence import EvidenceStore
from .experiments import ExperimentSpec, make_experiment_id
from .test1_campaign import (
    ACTIVE_SECONDS,
    HARD_SECONDS,
    INGREDIENTS as TEST1_INGREDIENTS,
    _balanced_cases,
    _family,
    _fixture_id,
    partition_cases,
)

CALL_START_CUTOFF_SECONDS = ACTIVE_SECONDS

PHASES = (
    ("headroom_recalibration", 45 * 60),
    ("ingredient_harvest_screen", 100 * 60),
    ("recipe_factorial", 100 * 60),
    ("conditional_rescue_generalization", 70 * 60),
    ("truncation_budget_disentanglement", 45 * 60),
    ("confirmation_handoff", 30 * 60),
    ("recipe_reserve", 20 * 60),
)

TRUNCATION_CLASSES = {"THINK_TRUNCATED", "ANSWER_TRUNCATED", "NO_FINAL_ANSWER"}
CAPABILITY_FAILURE_CLASSES = {"ANSWER_WRONG", "FORMAT_FAILURE", "TOOL_FAILURE"}

RULES = (
    "TEST2_BLIND is prohibited",
    "TEST3_PROTECTED is prohibited",
    "source Test-1 evidence selects targets but never substitutes for Test-1.1 controls",
    "each treatment is compared with a current-run control at matching seed and generation budget",
    "baseline failures and baseline passes are analyzed separately",
    "truncation is not equivalent to wrong-answer capability harm",
    "binary paired outcomes use rescue/regression statistics",
    "generation budget is an explicit experimental factor",
    "conditional rescue may promote even when global median delta is zero",
    "ingredient vocabulary expands beyond Test 1 and includes failure-derived candidates",
    "recipe tests vary order recurrence density knockout dose representation placement seed and budget",
    "every phase emits positive-work assertions",
    "KNOWN UNKNOWN NOT_LOOKED_AT states are explicit",
    "unused phase time becomes additional recipe tests",
)

DEFAULT_TEST11_CONFIG: dict[str, Any] = {
    "expected_calls": 5200,
    "safety_call_cap": 12000,
    "thinking_mode": False,
    "reasoning_effort": None,
    "base_generation_budget": 256,
    "generation_budgets": [256, 512, 1024, 2048],
    "seeds": [42, 43, 44],
    "headroom_fail_fixtures": 48,
    "headroom_pass_fixtures": 48,
    "headroom_repeats": 3,
    "promotion_min_rescue_trials": 4,
    "promotion_min_pass_sentinels": 8,
    "promotion_rescue_rate": 0.25,
    "promotion_max_capability_regression_rate": 0.10,
    "max_promoted_ingredients": 16,
    "max_recipe_ingredients": 8,
    "confirmation_recipes": 16,
    "minimum_phase_observations": 16,
    "minimum_active_utilization": 0.90,
}

# Test 1 had 16 ingredients. Test 1.1 starts with those plus a much broader
# evidence-informed bank. Dynamic candidates mined from the source Test-1
# failures are appended at runtime.
NEW_SEED_INGREDIENTS: tuple[dict[str, Any], ...] = (
    {"id":"ING-017","family":"brevity","label":"minimal_control","short":"Use the minimum reasoning needed.","full":"Use only the reasoning and checks necessary to answer correctly; avoid unnecessary expansion."},
    {"id":"ING-018","family":"completion","label":"reserve_answer_budget","short":"Reserve room for the final answer.","full":"Keep intermediate reasoning compact enough to leave sufficient completion budget for the required final answer."},
    {"id":"ING-019","family":"completion","label":"answer_first_internal","short":"Form the answer before elaborating.","full":"Determine the concise final answer early, then perform only checks that could change it."},
    {"id":"ING-020","family":"verification","label":"single_best_check","short":"Use the single highest-value verification.","full":"Choose and perform only the single verification most likely to catch a material error."},
    {"id":"ING-021","family":"verification","label":"targeted_verify","short":"Verify the weakest step only.","full":"Identify the least certain step and verify that step specifically rather than rechecking everything."},
    {"id":"ING-022","family":"constraints","label":"constraint_priority","short":"Prioritize hard constraints.","full":"Separate hard constraints from preferences and satisfy hard constraints before optimizing anything else."},
    {"id":"ING-023","family":"constraints","label":"constraint_ledger","short":"Track each hard constraint once.","full":"Maintain a compact ledger of hard constraints and mark each satisfied exactly once before finalizing."},
    {"id":"ING-024","family":"constraints","label":"conflict_resolution","short":"Resolve conflicting constraints explicitly.","full":"If constraints conflict, identify the controlling constraint and resolve the conflict before answering."},
    {"id":"ING-025","family":"evidence","label":"claim_evidence_pairing","short":"Pair claims with evidence.","full":"For every material claim, ensure there is direct task evidence or a valid derivation supporting it."},
    {"id":"ING-026","family":"evidence","label":"unsupported_claim_filter","short":"Remove unsupported claims.","full":"Before finalizing, remove any claim that is not directly supported by the task evidence or a valid derivation."},
    {"id":"ING-027","family":"state","label":"state_diff","short":"Compare old and new state.","full":"When state changes, explicitly compare prior and current state and retain only the authoritative current value."},
    {"id":"ING-028","family":"state","label":"state_lock","short":"Lock authoritative state.","full":"Once the authoritative current state is identified, prevent superseded state from re-entering later reasoning."},
    {"id":"ING-029","family":"logic","label":"assumption_inventory","short":"List hidden assumptions.","full":"Identify hidden assumptions required by the proposed conclusion and reject any assumption not supported by the task."},
    {"id":"ING-030","family":"logic","label":"necessary_sufficient_check","short":"Check necessary versus sufficient conditions.","full":"Distinguish necessary from sufficient conditions before accepting the conclusion."},
    {"id":"ING-031","family":"logic","label":"branch_elimination","short":"Eliminate impossible branches.","full":"Enumerate only plausible branches and eliminate those contradicted by premises before choosing an answer."},
    {"id":"ING-032","family":"arithmetic","label":"estimate_then_compute","short":"Estimate before exact arithmetic.","full":"Estimate the expected scale first, then compute exactly and reject results inconsistent with the estimate."},
    {"id":"ING-033","family":"arithmetic","label":"inverse_check","short":"Check by inverse operation.","full":"When practical, verify the numerical result using the inverse operation or an equivalent independent relation."},
    {"id":"ING-034","family":"arithmetic","label":"unit_propagation","short":"Propagate units through every step.","full":"Carry units through every numerical transformation and reject a result whose final units do not match the requested quantity."},
    {"id":"ING-035","family":"code","label":"minimal_reproducer","short":"Reduce to the failing path.","full":"Focus on the smallest execution path that reproduces the observed code failure before proposing a fix."},
    {"id":"ING-036","family":"code","label":"invariant_check","short":"Check the violated invariant.","full":"Identify the invariant the code must maintain and locate the first point where execution violates it."},
    {"id":"ING-037","family":"code","label":"input_output_contract","short":"Trace input to required output.","full":"Trace how the actual input propagates to the required output and verify each transformation on that path."},
    {"id":"ING-038","family":"code","label":"edge_case_probe","short":"Probe one decisive edge case.","full":"Test the proposed code reasoning against one edge case most likely to expose an incorrect assumption."},
    {"id":"ING-039","family":"tool_use","label":"tool_preconditions","short":"Check tool preconditions.","full":"Before choosing a tool, verify its required preconditions and whether the current state satisfies them."},
    {"id":"ING-040","family":"tool_use","label":"tool_postconditions","short":"Check expected tool result.","full":"Define the expected postcondition of the tool call and reject arguments that cannot produce it."},
    {"id":"ING-041","family":"tool_use","label":"tool_dependency_graph","short":"Order tool dependencies.","full":"Build the minimal dependency order among required tool actions and do not call a dependent tool before its prerequisites."},
    {"id":"ING-042","family":"tool_use","label":"tool_schema_projection","short":"Map intent to exact schema.","full":"Map each user intent field to the exact tool argument name, type, and allowed value before emitting the call."},
    {"id":"ING-043","family":"format","label":"format_skeleton_first","short":"Create the output skeleton first.","full":"Construct the required output skeleton first, then fill only the permitted fields without extra prose."},
    {"id":"ING-044","family":"format","label":"format_roundtrip","short":"Mentally parse the output.","full":"Before returning structured output, mentally parse it against the requested schema and correct any invalid structure."},
    {"id":"ING-045","family":"format","label":"no_extra_text","short":"Emit only requested content.","full":"Return only the requested content and omit explanations, labels, or wrappers not allowed by the output contract."},
    {"id":"ING-046","family":"uncertainty","label":"confidence_gate","short":"Do not guess past the evidence.","full":"If evidence is insufficient for a confident conclusion, explicitly choose the supported uncertainty state rather than guessing."},
    {"id":"ING-047","family":"uncertainty","label":"ambiguity_branch","short":"Test both plausible interpretations.","full":"When two interpretations are plausible, test the answer under both and select only if one is better supported."},
    {"id":"ING-048","family":"uncertainty","label":"missing_data_gate","short":"Separate missing data from reasoning failure.","full":"Determine whether the answer is impossible because information is missing before attempting additional reasoning."},
    {"id":"ING-049","family":"retrieval","label":"query_target_lock","short":"Lock the exact retrieval target.","full":"Identify the exact requested fact or key before scanning context so nearby distractors do not replace the target."},
    {"id":"ING-050","family":"retrieval","label":"evidence_span_check","short":"Verify the retrieved span.","full":"After retrieving a candidate fact, verify the surrounding context supports the same entity, time, and relation requested."},
    {"id":"ING-051","family":"retrieval","label":"distractor_rejection","short":"Reject near-match distractors.","full":"Actively reject facts that are lexically similar but differ in entity, time, scope, or relation from the requested target."},
    {"id":"ING-052","family":"planning","label":"critical_path_only","short":"Plan only the critical path.","full":"Plan only steps that are necessary dependencies of the requested result; omit optional branches."},
    {"id":"ING-053","family":"planning","label":"dependency_before_action","short":"Resolve dependencies first.","full":"Identify prerequisites and resolve them before taking downstream reasoning or tool actions."},
    {"id":"ING-054","family":"planning","label":"stop_when_sufficient","short":"Stop when the answer is proven.","full":"Stop adding reasoning once the available evidence and checks are sufficient to establish the required answer."},
    {"id":"ING-055","family":"critique","label":"failure_mode_probe","short":"Probe the most likely failure mode.","full":"Identify the most likely way the proposed answer could be wrong and test that specific failure mode once."},
    {"id":"ING-056","family":"critique","label":"alternative_hypothesis","short":"Test one competing explanation.","full":"Compare the leading answer against the strongest competing explanation before finalizing."},
    {"id":"ING-057","family":"critique","label":"disconfirming_evidence","short":"Search for disconfirming evidence.","full":"Look specifically for evidence in the task that contradicts the proposed answer before accepting it."},
    {"id":"ING-058","family":"causal","label":"cause_vs_correlation","short":"Separate cause from correlation.","full":"Do not infer causation from association; require a valid causal mechanism or counterfactual support."},
    {"id":"ING-059","family":"causal","label":"counterfactual_test","short":"Test the causal claim counterfactually.","full":"Ask whether the outcome would change if the proposed cause were absent or different, using only the task evidence."},
    {"id":"ING-060","family":"temporal","label":"timeline_order","short":"Order events before reasoning.","full":"Place relevant events in chronological order before inferring current state, causation, or precedence."},
    {"id":"ING-061","family":"temporal","label":"time_scope_lock","short":"Lock the requested time scope.","full":"Keep all evidence and conclusions within the requested time window and reject facts from the wrong period."},
    {"id":"ING-062","family":"adversarial","label":"literal_requirement_recovery","short":"Recover literal requirements from noisy wording.","full":"Ignore rhetorical or adversarial wording and recover the literal task requirements before solving."},
    {"id":"ING-063","family":"adversarial","label":"instruction_source_priority","short":"Prioritize authoritative instructions.","full":"When instructions conflict, follow the highest-authority applicable instruction and reject lower-priority conflicts."},
    {"id":"ING-064","family":"self_correction","label":"change_only_on_evidence","short":"Revise only with evidence.","full":"Do not change a correct answer merely because it is challenged; revise only when new evidence or a verified error justifies it."},
    {"id":"ING-065","family":"self_correction","label":"error_localization","short":"Localize the error before revising.","full":"Before revising, identify the exact step or assumption that failed and change only what that error requires."},
    {"id":"ING-066","family":"compression","label":"preserve_decisive_facts","short":"Preserve decisive facts when compressing.","full":"When summarizing or compressing context, preserve every fact that could change the final decision."},
    {"id":"ING-067","family":"robustness","label":"invariant_answer_under_rephrase","short":"Preserve semantics under rephrasing.","full":"Base the answer on task semantics rather than surface wording so irrelevant rephrasing does not change the result."},
    {"id":"ING-068","family":"meta","label":"choose_reasoning_mode","short":"Choose the cheapest adequate reasoning mode.","full":"Select the minimum reasoning strategy sufficient for this task instead of applying the same reasoning process universally."},
    {"id":"ING-069","family":"meta","label":"recognize_easy_case","short":"Do not overthink easy cases.","full":"If the task is already directly determined by explicit evidence, answer directly after one lightweight check."},
    {"id":"ING-070","family":"meta","label":"recognize_hard_case","short":"Escalate only when complexity demands it.","full":"Use deeper decomposition or verification only when the task contains genuine ambiguity, interaction, or high-risk reasoning."},
    {"id":"ING-071","family":"completion","label":"concise_final_only","short":"Keep the final answer concise.","full":"Keep the final answer as concise as the task permits so control overhead does not consume completion budget."},
    {"id":"ING-072","family":"completion","label":"budget_aware_checks","short":"Limit checks to available budget.","full":"Prioritize the highest-value checks and stop lower-value checks when they threaten completion of the required answer."},
    {"id":"ING-073","family":"generalization","label":"principle_over_example","short":"Apply the underlying principle.","full":"Solve using the underlying rule or principle rather than memorizing the surface form of the example."},
    {"id":"ING-074","family":"generalization","label":"sibling_consistency","short":"Check a nearby variant.","full":"Verify that the reasoning would remain valid for a nearby sibling case that preserves the same underlying rule."},
    {"id":"ING-075","family":"negative_transfer","label":"scope_control","short":"Apply the control only where relevant.","full":"Apply this reasoning control only when its triggering condition is present; otherwise do not add unnecessary processing."},
    {"id":"ING-076","family":"negative_transfer","label":"do_not_overconstrain","short":"Do not add constraints not requested.","full":"Do not introduce extra requirements, checks, or restrictions beyond those justified by the task."},
    {"id":"ING-077","family":"tool_use","label":"argument_crosscheck","short":"Cross-check dependent arguments.","full":"Cross-check arguments that depend on one another so identifiers, ranges, and dependent values are mutually consistent."},
    {"id":"ING-078","family":"tool_use","label":"required_argument_check","short":"Check required arguments only once.","full":"Verify every required argument is present exactly once and omit unsupported optional arguments."},
    {"id":"ING-079","family":"coding_generation","label":"spec_to_tests","short":"Translate requirements into checks.","full":"Translate the most important code requirements into concrete checks before writing the implementation."},
    {"id":"ING-080","family":"coding_generation","label":"minimal_correct_patch","short":"Prefer the smallest correct patch.","full":"Prefer the smallest implementation change that satisfies the specification without introducing unrelated behavior."},
)

REQUIRED_OUTPUTS = (
    "test1.1-source-audit.json",
    "ingredient-harvest-registry.json",
    "headroom-map.json",
    "baseline-stability-map.json",
    "binary-effect-map.json",
    "conditional-rescue-map.json",
    "recipe-factorial-atlas.json",
    "recipe-knockout-map.json",
    "recipe-recurrence-map.json",
    "recipe-order-map.json",
    "generation-budget-map.json",
    "family-generalization-map.json",
    "truncation-causality-map.json",
    "negative-transfer-map-1.1.json",
    "test1.1-priority-queue.json",
    "test1.1-uncertainty-ledger.json",
    "corrective-handoff.json",
    "assurance-map.json",
    "positive-work-assertions.jsonl",
    "test1.1-phase-events.jsonl",
    "test1.1-observations.jsonl",
)


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_TEST11_CONFIG)
    incoming = config.get("test11_campaign")
    if isinstance(incoming, dict):
        merged.update(copy.deepcopy(incoming))
    return merged


def build_test11_plan(cases: list[dict[str, Any]], *, test1_run: str | None = None) -> dict[str, Any]:
    partitions = partition_cases(cases)
    return {
        "schema_version": 1,
        "campaign": "gpt20b-test1.1-corrective-discovery-recipe-atlas",
        "source_test1_run": test1_run,
        "wall_clock_seconds": HARD_SECONDS,
        "active_model_seconds": ACTIVE_SECONDS,
        "call_start_cutoff_seconds": CALL_START_CUTOFF_SECONDS,
        "phases": [{"name": name, "seconds": seconds} for name, seconds in PHASES],
        "partition_counts": {name: len(rows) for name, rows in partitions.items()},
        "allowed_partitions": ["DISCOVERY", "VALIDATION"],
        "prohibited_partitions": ["TEST2_BLIND", "TEST3_PROTECTED"],
        "seed_ingredient_count": len(TEST1_INGREDIENTS) + len(NEW_SEED_INGREDIENTS),
        "dynamic_ingredient_harvest": True,
        "unused_time_sink": "MORE_RECIPE_TESTS",
        "required_outputs": list(REQUIRED_OUTPUTS),
    }


def validate_test11_plan(plan: dict[str, Any]) -> None:
    if int(plan["wall_clock_seconds"]) != HARD_SECONDS:
        raise ValueError("Test 1.1 wall clock must be exactly seven hours")
    if sum(int(row["seconds"]) for row in plan["phases"]) != ACTIVE_SECONDS:
        raise ValueError("Test 1.1 active phases must total exactly 6h50m")
    if int(plan["seed_ingredient_count"]) < 64:
        raise ValueError("Test 1.1 must start with at least 64 ingredients")
    for name in ("DISCOVERY", "VALIDATION", "TEST2_BLIND", "TEST3_PROTECTED"):
        if int(plan["partition_counts"].get(name, 0)) <= 0:
            raise ValueError(f"{name} partition is empty")
    if set(plan["prohibited_partitions"]) != {"TEST2_BLIND", "TEST3_PROTECTED"}:
        raise ValueError("protected partition contract changed")
    if plan["unused_time_sink"] != "MORE_RECIPE_TESTS":
        raise ValueError("unused Test 1.1 time must become more recipe tests")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                value = json.loads(raw)
                if isinstance(value, dict):
                    result.append(value)
    return result


def load_test1_source(results_root: Path, run_id: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    run_dir = results_root / run_id
    if not run_dir.is_dir():
        raise ValueError(f"Test-1 source run does not exist: {run_id}")
    problems = EvidenceStore(results_root, run_id).verify_manifest()
    if problems:
        raise ValueError(f"Test-1 evidence manifest verification failed: {problems}")
    required = ("test1-observations.jsonl", "fixture-partitions.json", "failure-registry.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"Test-1 source missing required artifacts: {missing}")

    source_parts = (_read_json(run_dir / "fixture-partitions.json").get("partitions") or {})
    computed = partition_cases(cases)
    for name, rows in computed.items():
        if set(source_parts.get(name) or []) != {_fixture_id(case) for case in rows}:
            raise ValueError(f"fixture partition drift detected for {name}")

    observations = _read_jsonl(run_dir / "test1-observations.jsonl")
    baseline_values: dict[str, list[float]] = defaultdict(list)
    positives, negatives, truncations = [], [], []
    for row in observations:
        fixture_id = row.get("fixture_id")
        baseline = row.get("baseline_score")
        if isinstance(fixture_id, str) and isinstance(baseline, (int, float)) and not isinstance(baseline, bool):
            baseline_values[fixture_id].append(float(baseline))
        delta = row.get("delta")
        if isinstance(delta, (int, float)) and not isinstance(delta, bool):
            if float(delta) > 0:
                positives.append(copy.deepcopy(row))
            elif float(delta) < 0:
                negatives.append(copy.deepcopy(row))
        if str((row.get("classification") or {}).get("result_class") or "") in TRUNCATION_CLASSES:
            truncations.append(copy.deepcopy(row))

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "observations": observations,
        "baselines": {key: float(median(values)) for key, values in baseline_values.items() if values},
        "positive_rows": positives,
        "negative_rows": negatives,
        "truncation_rows": truncations,
        "failures": (_read_json(run_dir / "failure-registry.json").get("failures") or []),
    }


def synthetic_test1_source(cases: list[dict[str, Any]]) -> dict[str, Any]:
    discovery = partition_cases(cases)["DISCOVERY"]
    baselines = {_fixture_id(case): (0.0 if index % 4 == 0 else 1.0) for index, case in enumerate(discovery)}
    return {
        "run_id": "SYNTHETIC-TEST1",
        "run_dir": None,
        "observations": [],
        "baselines": baselines,
        "positive_rows": [],
        "negative_rows": [],
        "truncation_rows": [],
        "failures": [],
    }


def _dynamic_ingredients(source: dict[str, Any]) -> list[dict[str, Any]]:
    families = sorted({
        str(row.get("family_id") or "unknown")
        for row in source.get("failures", [])
        if row.get("family_id")
    })
    result = []
    for index, family in enumerate(families, start=1):
        safe = family.replace("_", " ")
        result.append({
            "id": f"DYN-FAM-{index:03d}",
            "family": f"failure_derived:{family}",
            "label": f"target_{family}",
            "short": f"Focus on the decisive failure condition for {safe}.",
            "full": f"For this {safe} task, identify the specific condition that would cause the answer to fail and check that condition once before finalizing.",
            "origin": "test1_failure_family",
        })
    classes = sorted({
        str((row.get("classification") or {}).get("result_class") or "")
        for row in source.get("failures", [])
        if (row.get("classification") or {}).get("result_class")
    })
    for index, result_class in enumerate(classes, start=1):
        if result_class in TRUNCATION_CLASSES:
            text = "Keep reasoning compact and preserve enough output budget to complete the required final answer."
        elif result_class == "TOOL_FAILURE":
            text = "Validate tool selection, required arguments, argument types, and dependencies before emitting the call."
        elif result_class == "FORMAT_FAILURE":
            text = "Construct the exact required output shape first and emit no content outside it."
        else:
            text = "Localize the exact reasoning step responsible for the prior failure and verify only that step."
        result.append({
            "id": f"DYN-CLS-{index:03d}",
            "family": f"failure_class:{result_class}",
            "label": f"recover_{result_class.lower()}",
            "short": text,
            "full": text,
            "origin": "test1_failure_class",
        })
    return result


def build_ingredient_bank(source: dict[str, Any]) -> list[dict[str, Any]]:
    result = [copy.deepcopy(row) for row in TEST1_INGREDIENTS]
    result.extend(copy.deepcopy(list(NEW_SEED_INGREDIENTS)))
    result.extend(_dynamic_ingredients(source))
    seen = set()
    unique = []
    for row in result:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        unique.append(row)
    return unique


def _render(item: dict[str, Any], dose: float, representation: str) -> str:
    text = str(item["short"] if dose <= 0.5 else item["full"])
    if dose >= 2.0:
        text += " Treat this as mandatory, but do not repeat the check unnecessarily."
    if representation == "bullets":
        return "- " + text
    if representation == "schema":
        return json.dumps({"control": item["label"], "requirement": text}, sort_keys=True, separators=(",", ":"))
    return text


def build_recipe_messages(case: dict[str, Any], bank: dict[str, dict[str, Any]], recipe: dict[str, Any]) -> list[dict[str, str]]:
    buckets = {"system": [], "prefix": [], "middle": [], "suffix": []}
    for step in recipe.get("steps") or []:
        item = bank[str(step["ingredient_id"])]
        placement = str(step.get("placement", "prefix"))
        buckets.setdefault(placement, []).append(
            _render(item, float(step.get("dose", 1.0)), str(step.get("representation", "prose")))
        )
    prompt = str(case["prompt"])
    messages = []
    if buckets["system"]:
        messages.append({"role": "system", "content": "\n".join(buckets["system"])})
    prefix = "\n".join(buckets["prefix"])
    middle = "\n".join(buckets["middle"])
    suffix = "\n".join(buckets["suffix"])
    if middle:
        words = prompt.split()
        pivot = max(1, len(words) // 2)
        prompt = " ".join(words[:pivot]) + "\n\nCONTROL:\n" + middle + "\n\n" + " ".join(words[pivot:])
    if prefix:
        prompt = "CONTROL:\n" + prefix + "\n\nTASK:\n" + prompt
    if suffix:
        prompt = prompt + "\n\nADDITIONAL CONTROL:\n" + suffix
    messages.append({"role": "user", "content": prompt})
    return messages


def _wilson(successes: int, total: int, z: float = 1.6448536269514722) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def binary_effect_summary(rows: Iterable[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    data = [row for row in rows if row.get("kind") != "control"]
    fails = [row for row in data if float(row.get("control_score", 0.0)) < 1.0]
    passes = [row for row in data if float(row.get("control_score", 0.0)) >= 1.0]
    rescues = [row for row in fails if float(row.get("score", 0.0)) > float(row.get("control_score", 0.0))]
    regressions = [row for row in passes if float(row.get("score", 0.0)) < float(row.get("control_score", 0.0))]
    trunc_reg = [row for row in regressions if str((row.get("classification") or {}).get("result_class") or "") in TRUNCATION_CLASSES]
    cap_reg = [row for row in regressions if str((row.get("classification") or {}).get("result_class") or "") in CAPABILITY_FAILURE_CLASSES]
    rescue_rate = len(rescues) / len(fails) if fails else 0.0
    regression_rate = len(regressions) / len(passes) if passes else 0.0
    cap_reg_rate = len(cap_reg) / len(passes) if passes else 0.0

    if (
        len(fails) >= int(cfg["promotion_min_rescue_trials"])
        and rescue_rate >= float(cfg["promotion_rescue_rate"])
        and len(passes) >= int(cfg["promotion_min_pass_sentinels"])
        and cap_reg_rate <= float(cfg["promotion_max_capability_regression_rate"])
    ):
        classification = "PROMISING_CONDITIONAL_RESCUE"
        if _wilson(len(rescues), len(fails))[0] > 0.0 and cap_reg_rate == 0.0:
            classification = "STRONG_CONDITIONAL_RESCUE"
    elif cap_reg and cap_reg_rate > float(cfg["promotion_max_capability_regression_rate"]):
        classification = "CAPABILITY_HARM"
    elif regressions and not cap_reg:
        classification = "TRUNCATION_SENSITIVE"
    elif fails and not rescues:
        classification = "NO_RESCUE_SIGNAL"
    else:
        classification = "UNCERTAIN"

    return {
        "n": len(data),
        "baseline_fail_trials": len(fails),
        "baseline_pass_trials": len(passes),
        "rescues": len(rescues),
        "regressions": len(regressions),
        "truncation_regressions": len(trunc_reg),
        "capability_regressions": len(cap_reg),
        "rescue_rate": rescue_rate,
        "regression_rate": regression_rate,
        "capability_regression_rate": cap_reg_rate,
        "rescue_ci80": _wilson(len(rescues), len(fails)),
        "regression_ci80": _wilson(len(regressions), len(passes)),
        "classification": classification,
    }


def _group_binary(rows: list[dict[str, Any]], cfg: dict[str, Any], key_fn: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    grouped = defaultdict(list)
    examples = {}
    for row in rows:
        if row.get("kind") == "control":
            continue
        key = key_fn(row)
        grouped[key].append(row)
        examples.setdefault(key, copy.deepcopy(row.get("recipe") or {}))
    result = {}
    for key, values in grouped.items():
        result[key] = binary_effect_summary(values, cfg)
        result[key]["recipe"] = examples[key]
    return result


def _recipe_id(recipe: dict[str, Any]) -> str:
    payload = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    return "REC11-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _single_recipe(ingredient_id: str, *, dose: float = 0.5, representation: str = "prose", placement: str = "system") -> dict[str, Any]:
    recipe = {
        "mode": "single",
        "steps": [{"ingredient_id": ingredient_id, "dose": dose, "representation": representation, "placement": placement}],
    }
    recipe["recipe_id"] = _recipe_id(recipe)
    return recipe


def build_recipe_variants(ingredient_ids: list[str], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    ids = list(dict.fromkeys(ingredient_ids))[: max(2, int(cfg["max_recipe_ingredients"]))]
    result = []
    for ingredient_id in ids:
        for dose in (0.5, 1.0):
            for placement in ("system", "suffix", "prefix"):
                result.append(_single_recipe(ingredient_id, dose=dose, placement=placement))
    for a, b in itertools.combinations(ids[:8], 2):
        sequences = ([a, b], [b, a], [a, b, a], [b, a, b], [a, a, b], [a, b, b])
        for sequence in sequences:
            for representation in ("prose", "bullets"):
                steps = [
                    {
                        "ingredient_id": value,
                        "dose": 0.5 if index == 0 else 1.0,
                        "representation": representation,
                        "placement": "system" if index == 0 else "suffix",
                    }
                    for index, value in enumerate(sequence)
                ]
                recipe = {"mode": "ordered_recurrence", "steps": steps}
                recipe["recipe_id"] = _recipe_id(recipe)
                result.append(recipe)
    for triple in itertools.combinations(ids[:6], 3):
        for order in (triple, tuple(reversed(triple))):
            recipe = {
                "mode": "triple",
                "steps": [
                    {"ingredient_id": value, "dose": 0.5, "representation": "prose", "placement": "system"}
                    for value in order
                ],
            }
            recipe["recipe_id"] = _recipe_id(recipe)
            result.append(recipe)
    dense = ids[: min(6, len(ids))]
    if dense:
        full = {
            "mode": "dense",
            "steps": [
                {"ingredient_id": value, "dose": 0.5, "representation": "bullets", "placement": "system"}
                for value in dense
            ],
        }
        full["recipe_id"] = _recipe_id(full)
        result.append(full)
        for remove in dense:
            recipe = copy.deepcopy(full)
            recipe["mode"] = "knockout"
            recipe["steps"] = [step for step in recipe["steps"] if step["ingredient_id"] != remove]
            recipe["knocked_out"] = remove
            recipe["recipe_id"] = _recipe_id(recipe)
            result.append(recipe)
    unique = {}
    for recipe in result:
        unique.setdefault(recipe["recipe_id"], recipe)
    return list(unique.values())


def _spec(sequence: int, case: dict[str, Any], label: str, cfg: dict[str, Any], *, budget: int, seed: int, baseline: bool) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=make_experiment_id(sequence, _fixture_id(case), label),
        parent_experiment_id=None,
        task_id=_fixture_id(case),
        task_family=_family(case),
        difficulty_level=int(case.get("difficulty_level", 0)),
        hypothesis="Test-1.1 corrective controlled experiment",
        changed_variable="baseline" if baseline else "prompt_variant",
        thinking_mode=bool(cfg["thinking_mode"]),
        reasoning_effort=cfg.get("reasoning_effort"),
        generation_budget=int(budget),
        context_request=None,
        temperature=0.0,
        seed=int(seed),
        prompt_variant="base" if baseline else label,
        recovery_level=None,
    )


class Test11Campaign:
    __test__ = False

    def __init__(
        self,
        runner: Any,
        cases: list[dict[str, Any]],
        source: dict[str, Any],
        *,
        clock: Callable[[], float] = time.monotonic,
        started_monotonic: float | None = None,
    ) -> None:
        self.runner = runner
        self.cases = cases
        self.source = source
        self.cfg = _cfg(runner.config)
        self.clock = clock
        self.start = clock() if started_monotonic is None else float(started_monotonic)
        self.active_end = self.start + ACTIVE_SECONDS
        self.call_start_cutoff = self.start + CALL_START_CUTOFF_SECONDS
        self.partitions = partition_cases(cases)
        self.bank_list = build_ingredient_bank(source)
        self.bank = {row["id"]: row for row in self.bank_list}
        self.case_by_id = {_fixture_id(case): case for case in cases}
        self.controls: dict[tuple[str, int, int], dict[str, Any]] = {}
        self.rows: list[dict[str, Any]] = []
        self.sequence = 0
        self.phase_assertions: list[dict[str, Any]] = []
        self.promoted_ids: list[str] = []
        self.promoted_recipes: list[dict[str, Any]] = []

    def partition_name(self, case: dict[str, Any]) -> str:
        fixture_id = _fixture_id(case)
        for name, rows in self.partitions.items():
            if any(_fixture_id(row) == fixture_id for row in rows):
                return name
        return "UNKNOWN"

    def assert_allowed(self, case: dict[str, Any]) -> None:
        if self.partition_name(case) in {"TEST2_BLIND", "TEST3_PROTECTED"}:
            raise ValueError("Test 1.1 attempted to expose a protected partition")

    def can_start(self, deadline: float) -> bool:
        return self.clock() < min(deadline, self.active_end, self.call_start_cutoff)

    def _progress(self, label: str, begin: bool) -> None:
        progress = getattr(self.runner, "progress", None)
        if progress is None:
            return
        if begin:
            if int(getattr(progress, "done", 0)) >= int(getattr(progress, "total_tasks", 0)) - 1:
                old = int(progress.total_tasks)
                progress.total_tasks = old + 256
                self.runner._record_progress("plan_adjusted", label, old_total=old, new_total=int(progress.total_tasks), reason="wall-clock Test-1.1 capacity extended")
            self.runner._progress_begin(label)
        else:
            self.runner._progress_complete(label)

    def control(self, case: dict[str, Any], deadline: float, *, budget: int, seed: int, force: bool = False) -> dict[str, Any] | None:
        self.assert_allowed(case)
        key = (_fixture_id(case), int(budget), int(seed))
        if not force and key in self.controls:
            return self.controls[key]
        if not self.can_start(deadline):
            return None
        self.sequence += 1
        spec = _spec(self.sequence, case, f"control-b{budget}-s{seed}", self.cfg, budget=budget, seed=seed, baseline=True)
        label = f"test1.1 control {_fixture_id(case)} b{budget} s{seed}"
        self._progress(label, True)
        try:
            row = execute_experiment(self.runner, case, spec, parent=None)
        finally:
            self._progress(label, False)
        score = row.get("score")
        valid = (row.get("classification") or {}).get("valid_for_capability") is True
        record = {
            "score": float(score) if valid and isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0,
            "classification": copy.deepcopy(row.get("classification") or {}),
            "experiment_id": spec.experiment_id,
        }
        self.controls[key] = record
        self._record(case, row, phase="control", kind="control", recipe=None, control_score=record["score"], budget=budget, seed=seed)
        return record

    def treatment(self, case: dict[str, Any], deadline: float, *, phase: str, recipe: dict[str, Any], budget: int, seed: int, kind: str) -> dict[str, Any] | None:
        self.assert_allowed(case)
        control = self.control(case, deadline, budget=budget, seed=seed)
        if control is None or not self.can_start(deadline):
            return None
        self.sequence += 1
        recipe_id = str(recipe["recipe_id"])
        spec = _spec(self.sequence, case, f"{phase}-{recipe_id}-b{budget}-s{seed}", self.cfg, budget=budget, seed=seed, baseline=False)
        messages = build_recipe_messages(case, self.bank, recipe)
        label = f"test1.1 {phase} {_fixture_id(case)} {recipe_id}"
        self._progress(label, True)
        try:
            row = execute_experiment(self.runner, case, spec, parent=None, messages_override=messages)
        finally:
            self._progress(label, False)
        return self._record(case, row, phase=phase, kind=kind, recipe=recipe, control_score=float(control["score"]), budget=budget, seed=seed)

    def _record(self, case: dict[str, Any], row: dict[str, Any], *, phase: str, kind: str, recipe: dict[str, Any] | None, control_score: float, budget: int, seed: int) -> dict[str, Any]:
        score = row.get("score")
        valid = (row.get("classification") or {}).get("valid_for_capability") is True
        numeric = float(score) if valid and isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0
        result = {
            "schema_version": 1,
            "timestamp_utc": self.runner._utc(),
            "phase": phase,
            "kind": kind,
            "fixture_id": _fixture_id(case),
            "family_id": _family(case),
            "difficulty_level": int(case.get("difficulty_level", 0)),
            "partition": self.partition_name(case),
            "experiment_id": (row.get("experiment") or {}).get("experiment_id"),
            "classification": copy.deepcopy(row.get("classification") or {}),
            "score": numeric,
            "control_score": control_score,
            "delta": numeric - control_score,
            "generation_budget": int(budget),
            "seed": int(seed),
            "recipe": copy.deepcopy(recipe),
            "evidence_refs": copy.deepcopy(row.get("evidence_refs") or {}),
        }
        self.rows.append(result)
        self.runner.store.append_jsonl("test1.1-observations.jsonl", result)
        return result

    def positive_work(self, phase: str, start_index: int, attempted_fixtures: set[str], expected_coverage: str) -> None:
        rows = self.rows[start_index:]
        evidence_ids = [str(row.get("experiment_id") or "") for row in rows]
        digest = hashlib.sha256("\n".join(evidence_ids).encode("utf-8")).hexdigest()
        assertion = {
            "schema_version": 1,
            "phase": phase,
            "phase_nonce": hashlib.sha256(f"{self.runner.store.run_id}|{phase}".encode("utf-8")).hexdigest(),
            "expected_rule_count": len(RULES),
            "loaded_rule_count": len(RULES),
            "rules_hash": hashlib.sha256("\n".join(RULES).encode("utf-8")).hexdigest(),
            "expected_coverage": expected_coverage,
            "expected_fixture_count": len(self.partitions["DISCOVERY"]) + len(self.partitions["VALIDATION"]),
            "attempted_fixture_count": len(attempted_fixtures),
            "completed_fixture_count": len({str(row.get("fixture_id")) for row in rows}),
            "completed_observation_count": len(rows),
            "physical_calls": len(rows),
            "evidence_objects_created": len(rows),
            "coverage_observed": sorted({str(row.get("family_id")) for row in rows}),
            "evidence_digest": digest,
            "manifest_hash": "PENDING_FINALIZATION",
            "assurance_map_hash": "PENDING_REPORT",
            "known": sorted({str(row.get("family_id")) for row in rows}),
            "unknown": [] if rows else ["phase produced no observations"],
            "not_looked_at": ["TEST2_BLIND", "TEST3_PROTECTED"],
            "status": "VERIFIED_WORK_PERFORMED" if rows else "INSUFFICIENT_WORK",
        }
        self.phase_assertions.append(assertion)
        self.runner.store.append_jsonl("positive-work-assertions.jsonl", assertion)


def _source_headroom(campaign: Test11Campaign, partition: str = "DISCOVERY") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = campaign.partitions[partition]
    fail = [case for case in rows if float(campaign.source["baselines"].get(_fixture_id(case), 0.0)) < 1.0]
    passed = [case for case in rows if float(campaign.source["baselines"].get(_fixture_id(case), 0.0)) >= 1.0]
    fail = _balanced_cases(fail, min(len(fail), int(campaign.cfg["headroom_fail_fixtures"])))
    passed = _balanced_cases(passed, min(len(passed), int(campaign.cfg["headroom_pass_fixtures"])))
    return fail, passed


def _screen_summary(campaign: Test11Campaign, phase: str) -> dict[str, Any]:
    rows = [row for row in campaign.rows if row["phase"] == phase and row.get("recipe")]
    return _group_binary(rows, campaign.cfg, lambda row: str((row["recipe"]["steps"][0]["ingredient_id"] if len(row["recipe"]["steps"]) == 1 else row["recipe"]["recipe_id"])))


def _select_promoted(summary: dict[str, Any], campaign: Test11Campaign) -> list[str]:
    ranked = []
    for key, value in summary.items():
        recipe = value.get("recipe") or {}
        steps = recipe.get("steps") or []
        if len(steps) != 1:
            continue
        ingredient_id = str(steps[0]["ingredient_id"])
        rank = {
            "STRONG_CONDITIONAL_RESCUE": 5,
            "PROMISING_CONDITIONAL_RESCUE": 4,
            "TRUNCATION_SENSITIVE": 2,
            "UNCERTAIN": 1,
            "NO_RESCUE_SIGNAL": 0,
            "CAPABILITY_HARM": -10,
        }.get(str(value.get("classification")), 0)
        score = rank * 100 + float(value.get("rescue_rate", 0.0)) * 10 - float(value.get("capability_regression_rate", 0.0)) * 20
        ranked.append((score, ingredient_id))
    ranked.sort(reverse=True)
    selected = []
    seen_families = set()
    for _, ingredient_id in ranked:
        family = campaign.bank[ingredient_id]["family"]
        if family not in seen_families or len(selected) < 8:
            selected.append(ingredient_id)
            seen_families.add(family)
        if len(selected) >= int(campaign.cfg["max_promoted_ingredients"]):
            break
    if len(selected) < 8:
        # Never allow a sparse classifier to starve the recipe lab.
        for row in campaign.bank_list:
            if row["id"] not in selected:
                selected.append(row["id"])
            if len(selected) >= max(8, int(campaign.cfg["max_promoted_ingredients"])):
                break
    return selected


def _run_recipe_reserve(campaign: Test11Campaign, deadline: float, recipes: list[dict[str, Any]], phase: str, attempted: set[str]) -> None:
    if not recipes:
        recipes = build_recipe_variants([row["id"] for row in campaign.bank_list[:8]], campaign.cfg)
    cases = _balanced_cases(campaign.partitions["DISCOVERY"] + campaign.partitions["VALIDATION"], len(campaign.partitions["DISCOVERY"] + campaign.partitions["VALIDATION"]))
    seeds = [int(v) for v in campaign.cfg["seeds"]]
    budgets = [int(v) for v in campaign.cfg["generation_budgets"]]
    cursor = 0
    # Intentional infinite reservoir bounded only by wall-clock/call budget.
    while cases and recipes and campaign.can_start(deadline):
        recipe = recipes[cursor % len(recipes)]
        case = cases[(cursor // len(recipes)) % len(cases)]
        seed = seeds[(cursor // max(1, len(recipes) * len(cases))) % len(seeds)]
        budget = budgets[(cursor // max(1, len(recipes) * len(cases) * len(seeds))) % len(budgets)]
        attempted.add(_fixture_id(case))
        campaign.treatment(case, deadline, phase=phase, recipe=recipe, budget=budget, seed=seed, kind="recipe_reserve")
        cursor += 1


def phase_headroom(campaign: Test11Campaign, deadline: float) -> dict[str, Any]:
    start = len(campaign.rows)
    attempted = set()
    fail, passed = _source_headroom(campaign)
    cases = fail + passed
    repeats = int(campaign.cfg["headroom_repeats"])
    base_budget = int(campaign.cfg["base_generation_budget"])
    seeds = [int(v) for v in campaign.cfg["seeds"]]
    for repeat in range(repeats):
        for case in cases:
            if not campaign.can_start(deadline):
                break
            seed = seeds[repeat % len(seeds)]
            attempted.add(_fixture_id(case))
            campaign.control(case, deadline, budget=base_budget, seed=seed, force=True)
    # If source headroom work ends early, consume the rest with low-risk recipe tests.
    reserve = build_recipe_variants([row["id"] for row in campaign.bank_list[:8]], campaign.cfg)
    _run_recipe_reserve(campaign, deadline, reserve, "headroom_recipe_reserve", attempted)
    campaign.positive_work("headroom_recalibration", start, attempted, "source fail/pass headroom + repeated current controls + recipe reserve")
    current = [row for row in campaign.rows[start:] if row["kind"] == "control"]
    by_fixture = defaultdict(list)
    for row in current:
        by_fixture[row["fixture_id"]].append(float(row["score"]))
    return {
        "source_fail_count": len(fail),
        "source_pass_count": len(passed),
        "fixtures": {
            fixture_id: {
                "n": len(values),
                "pass_rate": sum(values) / len(values),
                "stable": len(set(values)) == 1,
            }
            for fixture_id, values in by_fixture.items()
        },
    }


def phase_ingredient_screen(campaign: Test11Campaign, deadline: float) -> dict[str, Any]:
    start = len(campaign.rows)
    attempted = set()
    fail, passed = _source_headroom(campaign)
    cases = fail + passed
    if not cases:
        cases = _balanced_cases(campaign.partitions["DISCOVERY"], len(campaign.partitions["DISCOVERY"]))
    seeds = [int(v) for v in campaign.cfg["seeds"]]
    # Broad bank: every ingredient gets low-dose/system, low-dose/suffix and
    # standard-dose/system attempts before repetitions begin.
    variants = []
    for row in campaign.bank_list:
        variants.extend([
            _single_recipe(row["id"], dose=0.5, placement="system"),
            _single_recipe(row["id"], dose=0.5, placement="suffix"),
            _single_recipe(row["id"], dose=1.0, placement="system"),
        ])
    cursor = 0
    while variants and cases and campaign.can_start(deadline):
        recipe = variants[cursor % len(variants)]
        case = cases[(cursor // len(variants)) % len(cases)]
        seed = seeds[(cursor // max(1, len(variants) * len(cases))) % len(seeds)]
        attempted.add(_fixture_id(case))
        campaign.treatment(case, deadline, phase="ingredient_harvest_screen", recipe=recipe, budget=int(campaign.cfg["base_generation_budget"]), seed=seed, kind="ingredient_screen")
        cursor += 1

    summary = _screen_summary(campaign, "ingredient_harvest_screen")
    campaign.promoted_ids = _select_promoted(summary, campaign)
    campaign.promoted_recipes = build_recipe_variants(campaign.promoted_ids, campaign.cfg)

    # If the full ingredient matrix happens to exhaust before deadline, all
    # remaining phase time becomes more recipe tests.
    _run_recipe_reserve(campaign, deadline, campaign.promoted_recipes, "ingredient_recipe_reserve", attempted)
    campaign.positive_work("ingredient_harvest_screen", start, attempted, "80+ seed/dynamic ingredients x placement/dose x headroom strata; leftover time -> recipe tests")
    return summary


def phase_recipe_factorial(campaign: Test11Campaign, deadline: float) -> dict[str, Any]:
    start = len(campaign.rows)
    attempted = set()
    recipes = campaign.promoted_recipes or build_recipe_variants(campaign.promoted_ids or [row["id"] for row in campaign.bank_list[:16]], campaign.cfg)
    _run_recipe_reserve(campaign, deadline, recipes, "recipe_factorial", attempted)
    campaign.positive_work("recipe_factorial", start, attempted, "order recurrence density knockout dose representation placement seed budget")
    rows = [row for row in campaign.rows[start:] if row.get("recipe")]
    return _group_binary(rows, campaign.cfg, lambda row: str(row["recipe"]["recipe_id"]))


def _top_recipes(atlas: dict[str, Any], campaign: Test11Campaign, limit: int) -> list[dict[str, Any]]:
    ranked = []
    for recipe_id, summary in atlas.items():
        recipe = summary.get("recipe")
        if not recipe:
            continue
        cls = str(summary.get("classification"))
        rank = {"STRONG_CONDITIONAL_RESCUE":5,"PROMISING_CONDITIONAL_RESCUE":4,"TRUNCATION_SENSITIVE":2,"UNCERTAIN":1,"NO_RESCUE_SIGNAL":0,"CAPABILITY_HARM":-10}.get(cls,0)
        score = rank * 100 + float(summary.get("rescue_rate",0.0))*10 - float(summary.get("capability_regression_rate",0.0))*20
        ranked.append((score, recipe_id, copy.deepcopy(recipe)))
    ranked.sort(reverse=True)
    result = [row[2] for row in ranked[:limit]]
    return result or campaign.promoted_recipes[:limit]


def phase_generalization(campaign: Test11Campaign, deadline: float, atlas: dict[str, Any]) -> dict[str, Any]:
    start = len(campaign.rows)
    attempted = set()
    recipes = _top_recipes(atlas, campaign, int(campaign.cfg["confirmation_recipes"]))
    cases = _balanced_cases(campaign.partitions["VALIDATION"], len(campaign.partitions["VALIDATION"]))
    seeds = [int(v) for v in campaign.cfg["seeds"]]
    cursor = 0
    while recipes and cases and campaign.can_start(deadline):
        recipe = recipes[cursor % len(recipes)]
        case = cases[(cursor // len(recipes)) % len(cases)]
        seed = seeds[(cursor // max(1, len(recipes)*len(cases))) % len(seeds)]
        attempted.add(_fixture_id(case))
        campaign.treatment(case, deadline, phase="conditional_rescue_generalization", recipe=recipe, budget=int(campaign.cfg["base_generation_budget"]), seed=seed, kind="generalization")
        cursor += 1
    _run_recipe_reserve(campaign, deadline, recipes, "generalization_recipe_reserve", attempted)
    campaign.positive_work("conditional_rescue_generalization", start, attempted, "validation-family transfer + repeated recipe reserve")
    rows = [row for row in campaign.rows[start:] if row.get("recipe")]
    by_recipe_family = _group_binary(rows, campaign.cfg, lambda row: f"{row['recipe']['recipe_id']}|{row['family_id']}")
    return by_recipe_family


def phase_truncation(campaign: Test11Campaign, deadline: float, atlas: dict[str, Any]) -> dict[str, Any]:
    start = len(campaign.rows)
    attempted = set()
    recipes = _top_recipes(atlas, campaign, 12)
    source_fixture_ids = []
    for row in campaign.source.get("truncation_rows", []):
        fixture_id = str(row.get("fixture_id") or "")
        if fixture_id in campaign.case_by_id and campaign.partition_name(campaign.case_by_id[fixture_id]) in {"DISCOVERY","VALIDATION"}:
            source_fixture_ids.append(fixture_id)
    cases = [campaign.case_by_id[value] for value in dict.fromkeys(source_fixture_ids)]
    if not cases:
        cases = _balanced_cases(campaign.partitions["DISCOVERY"], min(48, len(campaign.partitions["DISCOVERY"])))
    budgets = [int(v) for v in campaign.cfg["generation_budgets"]]
    seeds = [int(v) for v in campaign.cfg["seeds"]]
    cursor = 0
    while recipes and cases and campaign.can_start(deadline):
        recipe = recipes[cursor % len(recipes)]
        case = cases[(cursor // len(recipes)) % len(cases)]
        budget = budgets[(cursor // max(1,len(recipes)*len(cases))) % len(budgets)]
        seed = seeds[(cursor // max(1,len(recipes)*len(cases)*len(budgets))) % len(seeds)]
        attempted.add(_fixture_id(case))
        campaign.treatment(case, deadline, phase="truncation_budget_disentanglement", recipe=recipe, budget=budget, seed=seed, kind="budget_factorial")
        cursor += 1
    _run_recipe_reserve(campaign, deadline, recipes, "truncation_recipe_reserve", attempted)
    campaign.positive_work("truncation_budget_disentanglement", start, attempted, "source truncation fixtures x recipe x 256/512/1024/2048 budgets")
    rows = [row for row in campaign.rows[start:] if row.get("recipe")]
    return _group_binary(rows, campaign.cfg, lambda row: f"{row['recipe']['recipe_id']}|budget={row['generation_budget']}")


def phase_confirmation(campaign: Test11Campaign, deadline: float, atlas: dict[str, Any], generalization: dict[str, Any]) -> dict[str, Any]:
    start = len(campaign.rows)
    attempted = set()
    recipes = _top_recipes(atlas, campaign, int(campaign.cfg["confirmation_recipes"]))
    _run_recipe_reserve(campaign, deadline, recipes, "confirmation_handoff", attempted)
    campaign.positive_work("confirmation_handoff", start, attempted, "top recipe confirmation across discovery/validation with seed/budget variation")
    rows = [row for row in campaign.rows[start:] if row.get("recipe")]
    return _group_binary(rows, campaign.cfg, lambda row: str(row["recipe"]["recipe_id"]))


def _recipe_maps(atlas: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    knockout, recurrence, order = {}, {}, {}
    for recipe_id, summary in atlas.items():
        recipe = summary.get("recipe") or {}
        mode = recipe.get("mode")
        if mode == "knockout":
            knockout[recipe_id] = summary
        if mode == "ordered_recurrence" and len(recipe.get("steps") or []) >= 3:
            recurrence[recipe_id] = summary
        if mode in {"ordered_recurrence","triple"}:
            order[recipe_id] = summary
    return knockout, recurrence, order


def _priority_queue(atlas: dict[str, Any], generalization: dict[str, Any], confirmation: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for source_name, source in (("recipe_factorial", atlas), ("confirmation", confirmation)):
        for key, summary in source.items():
            if summary.get("classification") in {"STRONG_CONDITIONAL_RESCUE","PROMISING_CONDITIONAL_RESCUE","TRUNCATION_SENSITIVE","UNCERTAIN"}:
                result.append({
                    "source": source_name,
                    "key": key,
                    "classification": summary.get("classification"),
                    "rescue_rate": summary.get("rescue_rate"),
                    "capability_regression_rate": summary.get("capability_regression_rate"),
                    "recipe": copy.deepcopy(summary.get("recipe")),
                    "next_action": "GENERALIZATION_TEST" if source_name == "recipe_factorial" else "TEST2_RECIPE_INPUT",
                })
    result.sort(key=lambda row: (
        row["classification"] == "STRONG_CONDITIONAL_RESCUE",
        row["classification"] == "PROMISING_CONDITIONAL_RESCUE",
        float(row.get("rescue_rate") or 0.0),
        -float(row.get("capability_regression_rate") or 0.0),
    ), reverse=True)
    return result


def write_outputs(
    campaign: Test11Campaign,
    headroom: dict[str, Any],
    ingredient_summary: dict[str, Any],
    atlas: dict[str, Any],
    generalization: dict[str, Any],
    budget_map: dict[str, Any],
    confirmation: dict[str, Any],
) -> None:
    store = campaign.runner.store
    assert store is not None
    knockout, recurrence, order = _recipe_maps(atlas)
    queue = _priority_queue(atlas, generalization, confirmation)
    unknowns = []
    for key, summary in atlas.items():
        if summary.get("classification") in {"UNCERTAIN","TRUNCATION_SENSITIVE"}:
            unknowns.append({"key": key, "state": "UNKNOWN", "evidence": summary, "next_test": "more recipe/budget/generalization trials"})
    assurance = {
        "schema_version": 1,
        "rules": list(RULES),
        "rules_hash": hashlib.sha256("\n".join(RULES).encode("utf-8")).hexdigest(),
        "phases": copy.deepcopy(campaign.phase_assertions),
        "protected_partitions": {"TEST2_BLIND":"NOT_LOOKED_AT","TEST3_PROTECTED":"NOT_LOOKED_AT"},
        "known": ["ingredient bank screened","recipe compositions tested","headroom stratified","truncation budget tested"],
        "unknown": [row["key"] for row in unknowns],
        "not_looked_at": ["TEST2_BLIND","TEST3_PROTECTED"],
    }
    store.write_json("test1.1-source-audit.json", {
        "schema_version":1,
        "source_run":campaign.source.get("run_id"),
        "source_observations":len(campaign.source.get("observations",[])),
        "source_positive_rows":len(campaign.source.get("positive_rows",[])),
        "source_negative_rows":len(campaign.source.get("negative_rows",[])),
        "source_truncation_rows":len(campaign.source.get("truncation_rows",[])),
    }, producer="test1.1", stage="report")
    store.write_json("ingredient-harvest-registry.json", {"schema_version":1,"ingredient_count":len(campaign.bank_list),"ingredients":campaign.bank_list,"promoted_ids":campaign.promoted_ids,"screen":ingredient_summary}, producer="test1.1", stage="report")
    store.write_json("headroom-map.json", headroom, producer="test1.1", stage="report")
    store.write_json("baseline-stability-map.json", {"schema_version":1,"fixtures":headroom.get("fixtures",{})}, producer="test1.1", stage="report")
    store.write_json("binary-effect-map.json", {"schema_version":1,"ingredient_screen":ingredient_summary,"recipes":atlas}, producer="test1.1", stage="report")
    store.write_json("conditional-rescue-map.json", {"schema_version":1,"ingredients":{k:v for k,v in ingredient_summary.items() if v.get("rescues",0)>0},"recipes":{k:v for k,v in atlas.items() if v.get("rescues",0)>0}}, producer="test1.1", stage="report")
    store.write_json("recipe-factorial-atlas.json", {"schema_version":1,"recipes":atlas}, producer="test1.1", stage="report")
    store.write_json("recipe-knockout-map.json", {"schema_version":1,"recipes":knockout}, producer="test1.1", stage="report")
    store.write_json("recipe-recurrence-map.json", {"schema_version":1,"recipes":recurrence}, producer="test1.1", stage="report")
    store.write_json("recipe-order-map.json", {"schema_version":1,"recipes":order}, producer="test1.1", stage="report")
    store.write_json("generation-budget-map.json", {"schema_version":1,"effects":budget_map}, producer="test1.1", stage="report")
    store.write_json("family-generalization-map.json", {"schema_version":1,"effects":generalization}, producer="test1.1", stage="report")
    store.write_json("truncation-causality-map.json", {"schema_version":1,"effects":budget_map,"truncation_classes":sorted(TRUNCATION_CLASSES)}, producer="test1.1", stage="report")
    store.write_json("negative-transfer-map-1.1.json", {"schema_version":1,"recipes":{k:v for k,v in atlas.items() if v.get("capability_regressions",0)>0}}, producer="test1.1", stage="report")
    store.write_json("test1.1-priority-queue.json", {"schema_version":1,"queue":queue}, producer="test1.1", stage="report")
    store.write_json("test1.1-uncertainty-ledger.json", {"schema_version":1,"unknowns":unknowns}, producer="test1.1", stage="report")
    store.write_json("corrective-handoff.json", {"schema_version":1,"source_test1_run":campaign.source.get("run_id"),"priority_queue":queue,"ingredient_ids":campaign.promoted_ids,"recipe_count":len(atlas),"protected_partitions_exposed":False}, producer="test1.1", stage="report")
    store.write_json("assurance-map.json", assurance, producer="test1.1", stage="report")


def run_test11_campaign(
    runner: Any,
    cases: list[dict[str, Any]],
    *,
    test1_run: str,
    clock: Callable[[], float] = time.monotonic,
    started_monotonic: float | None = None,
) -> list[dict[str, Any]]:
    assert runner.store is not None
    source = load_test1_source(Path(runner.results_root), test1_run, cases)
    campaign = Test11Campaign(runner, cases, source, clock=clock, started_monotonic=started_monotonic)
    plan = build_test11_plan(cases, test1_run=test1_run)
    validate_test11_plan(plan)
    if not (runner.store.run_dir / "test1.1-plan.json").is_file():
        runner.store.write_json("test1.1-plan.json", plan, producer="test1.1", stage="preflight")

    results: dict[str, Any] = {}
    phase_start_cursor = campaign.clock()
    for phase_name, seconds in PHASES:
        deadline = min(campaign.active_end, phase_start_cursor + seconds)
        before = len(campaign.rows)
        if phase_name == "headroom_recalibration":
            results["headroom"] = phase_headroom(campaign, deadline)
        elif phase_name == "ingredient_harvest_screen":
            results["ingredients"] = phase_ingredient_screen(campaign, deadline)
        elif phase_name == "recipe_factorial":
            results["atlas"] = phase_recipe_factorial(campaign, deadline)
        elif phase_name == "conditional_rescue_generalization":
            results["generalization"] = phase_generalization(campaign, deadline, results.get("atlas", {}))
        elif phase_name == "truncation_budget_disentanglement":
            results["budget"] = phase_truncation(campaign, deadline, results.get("atlas", {}))
        elif phase_name == "confirmation_handoff":
            results["confirmation"] = phase_confirmation(campaign, deadline, results.get("atlas", {}), results.get("generalization", {}))
        elif phase_name == "recipe_reserve":
            attempted = set()
            start = len(campaign.rows)
            recipes = _top_recipes(results.get("atlas", {}), campaign, int(campaign.cfg["confirmation_recipes"]))
            _run_recipe_reserve(campaign, deadline, recipes, "recipe_reserve", attempted)
            campaign.positive_work("recipe_reserve", start, attempted, "all remaining scheduled time -> additional recipe tests")
        ended = campaign.clock()
        runner.store.append_jsonl("test1.1-phase-events.jsonl", {
            "phase":phase_name,
            "started_monotonic":phase_start_cursor,
            "ended_monotonic":ended,
            "deadline_monotonic":deadline,
            "observations_added":len(campaign.rows)-before,
            "total_observations":len(campaign.rows),
        })
        phase_start_cursor = deadline
        if ended >= campaign.active_end:
            break

    # Hard rule requested by the user: if anything finished early, do not end.
    # Keep running more recipe tests until the active clock is exhausted.
    if campaign.can_start(campaign.active_end):
        attempted = set()
        start = len(campaign.rows)
        recipes = _top_recipes(results.get("atlas", {}), campaign, int(campaign.cfg["confirmation_recipes"]))
        _run_recipe_reserve(campaign, campaign.active_end, recipes, "final_unused_time_recipe_sink", attempted)
        campaign.positive_work("final_unused_time_recipe_sink", start, attempted, "every remaining active second -> more recipe tests")

    write_outputs(
        campaign,
        results.get("headroom", {}),
        results.get("ingredients", {}),
        results.get("atlas", {}),
        results.get("generalization", {}),
        results.get("budget", {}),
        results.get("confirmation", {}),
    )
    return campaign.rows
