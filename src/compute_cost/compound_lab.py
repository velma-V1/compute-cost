"""Bounded compound-capability frontier measurement and composition penalties."""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

from .adaptive import AdaptiveDifficultyController, DifficultyObservation
from .characterization import execute_experiment
from .experiments import ExperimentSpec, make_experiment_id
from .frontier import build_family_frontier

ALIGNMENT_POLICY = "COMPONENT_LEVEL_EQUALS_COMPOUND_LEVEL"

COMPOUND_RECIPES: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        (
            "extract_calculate_json",
            (
                "extraction_transformation",
                "arithmetic_numerical_reasoning",
                "strict_structured_output",
            ),
        ),
        (
            "logic_constraints_json",
            (
                "formal_logic_deduction",
                "instruction_following_constraint_stacking",
                "strict_structured_output",
            ),
        ),
        (
            "temporal_plan_json",
            (
                "temporal_reasoning",
                "planning_optimization",
                "strict_structured_output",
            ),
        ),
        (
            "tool_chain",
            (
                "tool_selection",
                "tool_argument_correctness",
                "multi_tool_sequencing",
            ),
        ),
        (
            "context_state",
            (
                "context_retrieval",
                "contradictory_information_handling",
                "updated_obsolete_state_rejection",
            ),
        ),
        (
            "code_debug_verify",
            (
                "code_comprehension",
                "debugging_root_cause_diagnosis",
                "verification_critique",
            ),
        ),
    ]
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _extract_calculate(level: int) -> tuple[str, dict[str, Any]]:
    count = 2 + level
    rows = []
    total = 0
    for index in range(count):
        price = 3 + index + level
        quantity = 1 + (index % 3)
        owner = "Mira" if index % 2 == 0 else "Jo"
        rows.append(f"{owner}: item{index} price={price} qty={quantity}.")
        if owner == "Mira":
            total += price * quantity
    expected = {"owner": "Mira", "total": total}
    prompt = (
        f"Difficulty L{level}. Facts: {' '.join(rows)} Extract only Mira's rows, compute "
        "sum(price*qty), and return only JSON with keys owner and total. "
        f"The owner value must be Mira."
    )
    return prompt, expected


def _logic_constraints(level: int) -> tuple[str, dict[str, Any]]:
    depth = 2 + level
    predicates = [f"P{i}" for i in range(depth + 1)]
    rules = [f"All {predicates[i]} objects are {predicates[i + 1]}." for i in range(depth)]
    distractors = [f"No D{i} objects are Z{i}." for i in range(level)]
    expected = {"answer": "YES", "token": f"L{level}"}
    prompt = (
        f"Difficulty L{level}. {' '.join(rules + distractors)} Object K is {predicates[0]}. "
        f"Determine whether K is {predicates[-1]}. Return only JSON equal in shape to "
        '{"answer":"YES|NO","token":"LEVEL"}; use token '
        f"L{level}' and no extra keys or prose."
    )
    return prompt, expected


def _temporal_plan(level: int) -> tuple[str, dict[str, Any]]:
    offset = 5 + (level * 4)
    start_hour = 8
    absolute = start_hour + offset
    day_names = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
    day = day_names[(absolute // 24) % 7]
    hour = absolute % 24
    extra_tasks = [f"X{i}" for i in range(level)]
    order = ["A", "C", "B", *extra_tasks]
    expected = {"event_b": f"{day} {hour:02d}:00", "order": order}
    extras = ", ".join(extra_tasks) if extra_tasks else "none"
    prompt = (
        f"Difficulty L{level}. Event A is Monday 08:00 and event B is exactly {offset} hours later. "
        "Scheduling tasks: A must finish before C; C must finish before B; all other listed tasks "
        f"({extras}) have no dependencies and must be placed after B for this test. Return only JSON "
        "with keys event_b and order, where order is the exact task array."
    )
    return prompt, expected


def _tool_chain(level: int) -> tuple[str, dict[str, Any]]:
    user_id = str(40 + level)
    expected = {
        "sequence": ["search_user", "get_user", "calculator"],
        "calls": [
            {"tool": "search_user", "arguments": {"name": "Mira"}},
            {"tool": "get_user", "arguments": {"id": user_id}},
            {"tool": "calculator", "arguments": {"expression": f"{level + 2}*3"}},
        ],
    }
    prompt = (
        f"Difficulty L{level}. Available pseudo-tools: search_user(name:string), get_user(id:string), "
        "calculator(expression:string). First find Mira, then retrieve the returned user id. For this "
        f"fixture the search result id is '{user_id}'. Finally calculate {(level + 2)}*3. Return only "
        "JSON with keys sequence and calls, preserving that dependency order and exact argument types."
    )
    return prompt, expected


def _context_state(level: int) -> tuple[str, dict[str, Any]]:
    stale = f"OLD-{level:02d}"
    current = f"NEW-{level:02d}"
    noise = " ".join(f"note{i}=irrelevant" for i in range(level + 1))
    expected = {"code": current, "source": "LATEST"}
    prompt = (
        f"Difficulty L{level}. Earlier state says access_code={stale}. {noise} A later authoritative "
        f"update explicitly replaces it: access_code={current}. A contradictory stale note then repeats "
        f"access_code={stale} but is marked OBSOLETE. Return only JSON with current code and source; "
        "source must be LATEST."
    )
    return prompt, expected


def _code_debug_verify(level: int) -> tuple[str, dict[str, Any]]:
    values = list(range(1, 4 + level))
    correct = sum(values)
    wrong = values[-1]
    expected = {"root_cause": "WRONG_ACCUMULATOR", "correct_result": correct, "wrong_result": wrong}
    prompt = (
        f"Difficulty L{level}. Python-like code: total=0; for x in {values}: total=x; return total. "
        "The intended behavior is to sum every element. Comprehend the executed result, identify the "
        "single root cause, verify the corrected result, and return only JSON with keys root_cause, "
        "correct_result, wrong_result. root_cause must use the canonical diagnostic token."
    )
    return prompt, expected


_BUILDERS = {
    "extract_calculate_json": _extract_calculate,
    "logic_constraints_json": _logic_constraints,
    "temporal_plan_json": _temporal_plan,
    "tool_chain": _tool_chain,
    "context_state": _context_state,
    "code_debug_verify": _code_debug_verify,
}


def build_compound_cases() -> list[dict[str, Any]]:
    """Build six deterministic aligned L0-L10 compound ladders."""
    cases: list[dict[str, Any]] = []
    for compound_id, capabilities in COMPOUND_RECIPES.items():
        builder = _BUILDERS[compound_id]
        for level in range(11):
            prompt, expected = builder(level)
            cases.append(
                {
                    "id": f"compound-{compound_id}-L{level}",
                    "category": "composite_agent_tasks",
                    "family_id": "composite_agent_tasks",
                    "compound_id": compound_id,
                    "difficulty_level": level,
                    "difficulty": {
                        "level": level,
                        "rubric_version": "compound-aligned-v1",
                        "dimensions": {
                            "component_count": len(capabilities),
                            "aligned_component_level": level,
                        },
                    },
                    "prompt": prompt,
                    "scorer": "json",
                    "scorer_version": "1",
                    "expected": expected,
                    "required": list(expected),
                    "timeout_s": 120,
                    "capabilities_required": list(capabilities),
                    "component_level_map": {capability: level for capability in capabilities},
                    "recovery_eligible": False,
                    "robustness_eligible": False,
                    "compound": True,
                    "tags": ["compound", compound_id, f"L{level}"],
                }
            )
    return cases


def select_compound_targets(
    frontiers: dict[str, Any],
    *,
    max_compounds: int = 6,
) -> dict[str, dict[str, Any]]:
    """Select only recipes whose aligned component floors are all proven."""
    if isinstance(max_compounds, bool) or not isinstance(max_compounds, int) or max_compounds <= 0:
        raise ValueError("max_compounds must be a positive integer")
    family_rows = frontiers.get("families") or {}
    selected: dict[str, dict[str, Any]] = {}
    for compound_id, capabilities in COMPOUND_RECIPES.items():
        component_frontiers: dict[str, int] = {}
        valid = True
        for capability in capabilities:
            family = family_rows.get(capability) or {}
            floor = family.get("reliable_floor")
            if isinstance(floor, bool) or not isinstance(floor, int) or not 0 <= floor <= 10:
                valid = False
                break
            component_frontiers[capability] = floor
        if not valid:
            continue
        selected[compound_id] = {
            "compound_id": compound_id,
            "capabilities_required": list(capabilities),
            "component_frontiers": component_frontiers,
            "expected_frontier": min(component_frontiers.values()),
            "alignment_policy": ALIGNMENT_POLICY,
        }
        if len(selected) >= max_compounds:
            break
    return selected


def composition_penalty(expected_frontier: int, observed_frontier: int | None) -> int | None:
    """Return observed minus expected aligned frontier; None means unresolved."""
    if observed_frontier is None:
        return None
    return int(observed_frontier) - int(expected_frontier)


def _spec(
    *,
    sequence: int,
    compound_id: str,
    case: dict[str, Any],
    parent: ExperimentSpec | None,
    changed_variable: str,
    hypothesis: str,
    thinking_mode: bool,
    reasoning_effort: str | None,
    generation_budget: int,
) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=make_experiment_id(sequence, compound_id, f"compound-L{case['difficulty_level']}"),
        parent_experiment_id=None if parent is None else parent.experiment_id,
        task_id=compound_id,
        task_family=compound_id,
        difficulty_level=int(case["difficulty_level"]),
        hypothesis=hypothesis,
        changed_variable=changed_variable,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
        generation_budget=generation_budget,
        context_request=None,
        temperature=0.0,
        seed=42,
        prompt_variant="compound-base",
        recovery_level=None,
    )


def _run_one_compound(
    runner: Any,
    compound_id: str,
    ladder: dict[int, dict[str, Any]],
    target: dict[str, Any],
    *,
    sequence_start: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    cfg = runner.config["compound_lab"]
    campaign_cfg = runner.config["capability_campaign"]
    expected = int(target["expected_frontier"])
    controller = AdaptiveDifficultyController(
        min_level=0,
        max_level=expected,
        anchor_level=expected,
        jump=int(cfg["jump"]),
        boundary_repeats=int(cfg["boundary_repeats"]),
    )
    max_experiments = int(cfg["max_experiments_per_compound"])
    observations: list[DifficultyObservation] = []
    observation_rows: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    latest_spec_by_level: dict[int, ExperimentSpec] = {}
    previous_spec: ExperimentSpec | None = None
    sequence = sequence_start

    while len(generated) < max_experiments:
        decision = controller.next(observations)
        if decision.action == "STOP":
            break
        if decision.level is None:
            break
        level = int(decision.level)
        case = ladder[level]
        if decision.action == "REPLICATE":
            parent = latest_spec_by_level.get(level)
            if parent is None:
                raise ValueError(f"cannot replicate compound {compound_id} L{level} without parent")
            changed_variable = "replication"
        else:
            parent = previous_spec
            changed_variable = "baseline" if parent is None else "difficulty_level"

        sequence += 1
        spec = _spec(
            sequence=sequence,
            compound_id=compound_id,
            case=case,
            parent=parent,
            changed_variable=changed_variable,
            hypothesis=decision.reason,
            thinking_mode=bool(campaign_cfg["thinking_mode"]),
            reasoning_effort=campaign_cfg.get("reasoning_effort"),
            generation_budget=int(campaign_cfg["generation_budget"]),
        )
        row = execute_experiment(runner, case, spec, parent=parent)
        generated.append(row)
        classification = row.get("classification") or {}
        valid = classification.get("valid_for_capability") is True
        result_class = str(classification.get("result_class"))
        passed: bool | None = None if not valid else result_class == "ANSWER_CORRECT"
        observation = {
            "compound_id": compound_id,
            "level": level,
            "passed": passed,
            "valid_for_capability": valid,
            "result_class": result_class,
            "experiment_id": spec.experiment_id,
            "fixture_id": case["id"],
        }
        observation_rows.append(observation)
        runner.store.append_jsonl("compound-observations.jsonl", observation)
        observations.append(DifficultyObservation(level=level, passed=passed, valid_for_capability=valid))
        latest_spec_by_level[level] = spec
        previous_spec = spec

    frontier = build_family_frontier(
        compound_id,
        observation_rows,
        reliable_threshold=float(campaign_cfg.get("reliable_threshold", 0.90)),
        unstable_threshold=float(campaign_cfg.get("unstable_threshold", 0.40)),
    )
    observed = frontier["reliable_floor"]
    penalty = composition_penalty(expected, observed)
    result = {
        "compound_id": compound_id,
        "capabilities_required": list(target["capabilities_required"]),
        "component_frontiers": dict(target["component_frontiers"]),
        "expected_component_frontier": expected,
        "observed_compound_frontier": observed,
        "first_failure_level": frontier["first_failure_level"],
        "composition_penalty": penalty,
        "interaction_delta": penalty,
        "status": "MEASURED" if observed is not None else "UNRESOLVED",
        "frontier": frontier,
    }
    return generated, result, sequence


def run_compound_lab(
    runner: Any,
    frontiers: dict[str, Any],
    *,
    sequence_start: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Run bounded compound searches only where every component frontier is proven."""
    cfg = runner.config.get("compound_lab") or {}
    if cfg.get("enabled") is not True:
        return [], {
            "schema_version": 1,
            "model": str(runner.model),
            "measurement_policy": {"component_alignment": ALIGNMENT_POLICY},
            "compounds": {},
        }, sequence_start

    cases = build_compound_cases()
    ladders: dict[str, dict[int, dict[str, Any]]] = {}
    for case in cases:
        ladders.setdefault(str(case["compound_id"]), {})[int(case["difficulty_level"])] = case

    targets = select_compound_targets(frontiers, max_compounds=int(cfg["max_compounds"]))
    all_rows: list[dict[str, Any]] = []
    compounds: dict[str, dict[str, Any]] = {}
    sequence = sequence_start
    for compound_id, target in targets.items():
        rows, result, sequence = _run_one_compound(
            runner,
            compound_id,
            ladders[compound_id],
            target,
            sequence_start=sequence,
        )
        all_rows.extend(rows)
        compounds[compound_id] = result

    compound_map = {
        "schema_version": 1,
        "model": str(runner.model),
        "measurement_policy": {
            "component_alignment": ALIGNMENT_POLICY,
            "expected_frontier": "minimum proven aligned component reliable floor",
            "penalty": "observed_compound_frontier - expected_component_frontier",
        },
        "selected_count": len(targets),
        "compounds": compounds,
    }
    return all_rows, compound_map, sequence
