"""GPT-OSS 20B Test 1.2 full-system improvement campaign.

Test 1.2 reuses the seven-hour Test 1.1 scientific/evidence chassis but widens
the optimization surface from prompt recipes to the complete locally testable
agent stack. It consumes the FULL Test 1.1 handoff, not the older Test 1
handoff. Prompt controls compete directly with reasoning modes, generation and
context budgets, planning, verification, critique, retry, state tracking,
memory/context transforms, tool-policy controls, delegation/ensemble patterns,
conditional escalation, adaptive routing, and A+B+A compositions.

The campaign is adaptive but not greedy: every mechanism family receives a
coverage floor on current baseline failures and baseline-pass sentinels before
additional calls are allocated. Promotion requires measured rescue, bounded
capability regression, and cost efficiency. TEST2_BLIND and TEST3_PROTECTED
remain untouched.
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
from statistics import mean, median
from typing import Any, Callable, Iterable

from .characterization import execute_experiment
from .evidence import EvidenceStore
from .experiments import ExperimentSpec, make_experiment_id
from .test1_campaign import _balanced_cases, _family, _fixture_id, partition_cases
from .test11_campaign import TRUNCATION_CLASSES, CAPABILITY_FAILURE_CLASSES
from .test12_toollab import (
    TOOL_HARNESS_POLICIES,
    TOOL_MICROCASES,
    execute_tool,
    parse_action,
    score_final,
    tool_system_prompt,
)

COLLECTION_HARD_SECONDS = (6 * 60 * 60) + (15 * 60)
COLLECTION_ACTIVE_SECONDS = 6 * 60 * 60
HARD_SECONDS = COLLECTION_HARD_SECONDS
ACTIVE_SECONDS = COLLECTION_ACTIVE_SECONDS
CALL_START_CUTOFF_SECONDS = COLLECTION_ACTIVE_SECONDS

# Six active hours; the extra 15 minutes is reserved for preflight/finalization.
# Adaptive early-stop may finish sooner once coverage + uncertainty criteria are met.
PHASES = (
    ("baseline_capability_map", 40 * 60),
    ("fractional_compute_surface", 30 * 60),
    ("mechanism_coverage_floor", 60 * 60),
    ("real_tool_execution", 30 * 60),
    ("failure_phenotype_replay", 45 * 60),
    ("interaction_scout", 40 * 60),
    ("dose_activation_boundaries", 40 * 60),
    ("negative_transfer_sentinels", 45 * 60),
    ("information_gain_reserve", 30 * 60),
)


IMPROVEMENT_SURFACE = (
    "PROMPT_CONTROL",
    "REASONING_MODE",
    "GENERATION_BUDGET",
    "CONTEXT_WINDOW",
    "PLANNING",
    "VERIFICATION",
    "CRITIQUE",
    "RETRY_RECOVERY",
    "STATE_TRACKING",
    "MEMORY",
    "CONTEXT_SELECTION_COMPRESSION",
    "TOOL_POLICY",
    "REAL_TOOL_EXECUTION",
    "DELEGATION",
    "ENSEMBLE_CONSENSUS",
    "ADAPTIVE_ROUTING",
    "STOP_ESCALATE_POLICY",
    "COMPOSITION_LAYERING",
    "COMPUTE_COST_ROUTING",
    "FINE_TUNING_QUALIFICATION",
)

RULES = (
    "a new model requires no prior model-specific Test 1 or Test 1.1 run; historical mechanisms are seeds, never evidence for the new model",
    "collection + tuning hard ceilings sum to 12h30m, leaving operational buffer below the 14-hour end-to-end target",
    "TEST2_BLIND is prohibited",
    "TEST3_PROTECTED is prohibited",
    "every mechanism family receives a coverage floor before adaptive pruning",
    "every promoted mechanism is tested on baseline failures and baseline-pass sentinels",
    "current-run controls are matched by seed budget thinking mode and context where applicable",
    "oracle scores may evaluate outcomes but may never choose a deployment-time routing action",
    "adaptive routing may use only model-visible task text and model-generated state",
    "multi-call controllers record every physical model call and compete on value per call token and second",
    "intermediate planner critic memory and router outputs are evidence but are not scored as final answers",
    "negative transfer blocks global promotion even when a mechanism rescues a local family",
    "thinking mode reasoning effort generation budget context window temperature and seed are explicit factors",
    "memory and context compression must preserve decisive task evidence rather than invent state",
    "tool-policy mechanisms are tested both on benchmark tool outputs and on a deterministic in-process synthetic tool harness with real execution and error feedback",
    "independent ensemble branches execute serially on one local GPU to avoid compute contention confounds",
    "A+B+A and other composition effects are measured rather than assumed additive",
    "residual failures are eligible for fine-tuning only after prompt controller compute context retry and tool-policy owners are tested",
    "unused active time is allocated to uncertainty reduction and replication, never arbitrary repeated prompting",
)

REQUIRED_TEST11_FILES = (
    "test1.1-observations.jsonl",
    "fixture-partitions.json",
    "ingredient-harvest-registry.json",
    "recipe-factorial-atlas.json",
    "system-operator-atlas.json",
    "family-generalization-map.json",
    "generation-budget-map.json",
    "residual-failure-ownership.json",
    "fine-tuning-readiness-map.json",
    "test1.1-priority-queue.json",
    "test1.1-uncertainty-ledger.json",
    "corrective-handoff.json",
    "assurance-map.json",
)

REQUIRED_OUTPUTS = (
    "test1.2-source-audit.json",
    "test1.2-plan.json",
    "mechanism-registry.json",
    "full-control-candidate-registry.json",
    "control-grammar-coverage.json",
    "mechanism-coverage-ledger.json",
    "reasoning-compute-map.json",
    "controller-mechanism-map.json",
    "context-memory-state-map.json",
    "tool-action-verification-map.json",
    "real-tool-execution-map.json",
    "retry-replay-recovery-map.json",
    "composition-interaction-atlas.json",
    "adaptive-routing-map.json",
    "family-generalization-map-1.2.json",
    "activation-boundary-map.json",
    "negative-transfer-map-1.2.json",
    "cost-value-frontier-1.2.json",
    "residual-failure-ownership-1.2.json",
    "fine-tuning-readiness-map-1.2.json",
    "test1.2-priority-queue.json",
    "test1.2-uncertainty-ledger.json",
    "test1.2-handoff.json",
    "tuning-example-corpus.jsonl",
    "harness-policy-blueprint.json",
    "scope-boundaries.json",
    "assurance-map-1.2.json",
    "positive-work-assertions-1.2.jsonl",
    "test1.2-phase-events.jsonl",
    "test1.2-observations.jsonl",
)

DEFAULT_TEST12_CONFIG: dict[str, Any] = {
    "expected_calls": 5600,
    "safety_call_cap": 14000,
    "base_generation_budget": 256,
    "generation_budgets": [256, 512, 1024, 2048],
    "context_windows": [4096, 8192, 16384, 32768],
    "seeds": [42, 43, 44],
    "coverage_floor_failures": 4,
    "coverage_floor_sentinels": 4,
    "promotion_min_rescue_trials": 4,
    "promotion_min_pass_sentinels": 8,
    "promotion_rescue_rate": 0.25,
    "promotion_max_capability_regression_rate": 0.10,
    "max_promoted_mechanisms": 16,
    "max_source_recipes": 8,
    "max_composition_arms": 24,
    "confirmation_mechanisms": 16,
    "minimum_phase_observations": 16,
    "router_confidence_threshold": 0.65,
    "call_cost_penalty": 0.06,
    "token_cost_penalty": 0.00002,
    "latency_cost_penalty": 0.002,
}

CORE_INTERVENTIONS: tuple[dict[str, Any], ...] = (
    {"id":"CTRL-DECOMPOSE","category":"PROMPT_CONTROL","mode":"single","label":"minimal_decomposition","instruction":"Decompose the task into the minimum necessary dependent steps before producing the final answer."},
    {"id":"CTRL-CONSTRAINTS","category":"PROMPT_CONTROL","mode":"single","label":"constraint_ledger","instruction":"Track every explicit hard requirement once and verify each is satisfied before finalizing."},
    {"id":"CTRL-EVIDENCE","category":"PROMPT_CONTROL","mode":"single","label":"evidence_grounding","instruction":"Use only task evidence or direct derivations. Do not invent missing facts."},
    {"id":"CTRL-STATE","category":"STATE_TRACKING","mode":"single","label":"authoritative_state","instruction":"Identify the latest authoritative state, reject superseded values, and answer from current state only."},
    {"id":"CTRL-SCHEMA","category":"PROMPT_CONTROL","mode":"single","label":"schema_first","instruction":"Construct the exact required output shape first, then fill it without extra content."},
    {"id":"REASON-LOW","category":"REASONING_MODE","mode":"single","label":"thinking_low","reasoning_effort":"low"},
    {"id":"REASON-MEDIUM","category":"REASONING_MODE","mode":"single","label":"thinking_medium","reasoning_effort":"medium"},
    {"id":"REASON-HIGH","category":"REASONING_MODE","mode":"single","label":"thinking_high","reasoning_effort":"high"},
    {"id":"BUDGET-512","category":"GENERATION_BUDGET","mode":"single","label":"budget_512","generation_budget":512},
    {"id":"BUDGET-1024","category":"GENERATION_BUDGET","mode":"single","label":"budget_1024","generation_budget":1024},
    {"id":"BUDGET-2048","category":"GENERATION_BUDGET","mode":"single","label":"budget_2048","generation_budget":2048},
    {"id":"CTX-4096","category":"CONTEXT_WINDOW","mode":"single","label":"context_4096","context_request":4096},
    {"id":"CTX-8192","category":"CONTEXT_WINDOW","mode":"single","label":"context_8192","context_request":8192},
    {"id":"CTX-16384","category":"CONTEXT_WINDOW","mode":"single","label":"context_16384","context_request":16384},
    {"id":"CTX-32768","category":"CONTEXT_WINDOW","mode":"single","label":"context_32768","context_request":32768},
    {"id":"PLAN-SOLVE","category":"PLANNING","mode":"precompute","label":"plan_then_solve","aux_instruction":"Produce the minimum dependency-ordered plan needed to solve the task. Do not answer yet.","final_instruction":"Use the plan only where it is necessary; solve the original task and return the required final answer."},
    {"id":"SOLVE-VERIFY","category":"VERIFICATION","mode":"repair","label":"solve_then_verify","instruction":"Verify the candidate against every material task requirement. Correct only detected errors and return only the final answer."},
    {"id":"CRITIQUE-REPAIR","category":"CRITIQUE","mode":"critique_repair","label":"critique_then_repair","aux_instruction":"Identify the single strongest reason the candidate could be wrong. Cite the exact task evidence. Do not rewrite yet.","final_instruction":"Use the critique only if supported. Repair the candidate and return only the final answer."},
    {"id":"PLAN-SOLVE-VERIFY","category":"PLANNING","mode":"three_stage","label":"plan_solve_verify","aux_instruction":"Produce a compact dependency-ordered plan. Do not answer.","middle_instruction":"Solve the task using the plan. Produce a candidate answer.","final_instruction":"Verify the candidate against the task, repair any material defect, and return only the final answer."},
    {"id":"INDEPENDENT-ADJUDICATE","category":"ENSEMBLE_CONSENSUS","mode":"ensemble","label":"two_independent_then_adjudicate","aux_instruction":"Solve the task independently and return a candidate answer.","final_instruction":"Compare the two independent candidates against the original task. Return only the better supported final answer, repairing it if needed."},
    {"id":"EVIDENCE-LEDGER","category":"MEMORY","mode":"precompute","label":"evidence_ledger","aux_instruction":"Extract a compact ledger of decisive facts, constraints, and uncertainties from the task. Do not solve it.","final_instruction":"Solve the original task using the ledger as memory. Ignore any ledger item not supported by the original task."},
    {"id":"STATE-MEMORY","category":"MEMORY","mode":"precompute","label":"state_memory","aux_instruction":"Extract only authoritative current state, superseded state, and state transitions relevant to the requested answer.","final_instruction":"Use the extracted state memory to solve the original task. Prefer authoritative current state."},
    {"id":"CONTEXT-SELECT","category":"CONTEXT_SELECTION_COMPRESSION","mode":"precompute","label":"select_relevant_context","aux_instruction":"Select only the evidence spans and constraints that can change the answer. Preserve exact entities, values, relations, and time scope.","final_instruction":"Solve from the selected context while checking it against the original task. Do not use omitted distractors."},
    {"id":"CONTEXT-COMPRESS","category":"CONTEXT_SELECTION_COMPRESSION","mode":"precompute","label":"compress_decisive_context","aux_instruction":"Compress the task while preserving every fact or constraint that could change the answer. Mark uncertainty explicitly.","final_instruction":"Solve the original task using the compressed context as a working memory. Correct it if it conflicts with the original."},
    {"id":"TOOL-SCHEMA","category":"TOOL_POLICY","mode":"precompute","label":"tool_schema_projection","aux_instruction":"For any tool-like output, map intent to exact tool choice, required argument names, types, dependencies, and forbidden unsupported values. Do not emit the final call yet.","final_instruction":"Produce the final answer or tool call after applying the schema map exactly."},
    {"id":"TOOL-DEPENDENCY","category":"TOOL_POLICY","mode":"precompute","label":"tool_dependency_plan","aux_instruction":"Identify tool/action prerequisites and the minimum legal dependency order. Do not execute or emit the final call yet.","final_instruction":"Return the final requested tool-like output respecting prerequisite order and argument dependencies."},
    {"id":"TOOL-POSTCHECK","category":"TOOL_POLICY","mode":"repair","label":"tool_postcondition_repair","instruction":"For tool-like output, check tool choice, required arguments, types, dependencies, and expected postcondition. Repair only errors and return the corrected output."},
    {"id":"FAILURE-DIAGNOSE-RETRY","category":"RETRY_RECOVERY","mode":"retry","label":"diagnose_then_retry","aux_instruction":"Diagnose the exact assumption, step, constraint, or output contract most likely responsible for failure in the candidate. Do not merely restate the task.","final_instruction":"Retry from the original task using a different strategy that directly addresses the diagnosed failure. Return only the final answer."},
    {"id":"STRATEGY-RESET","category":"RETRY_RECOVERY","mode":"retry","label":"strategy_reset","aux_instruction":"Assume the candidate strategy is unreliable. Identify an independent solution strategy that does not reuse its central assumption.","final_instruction":"Solve again using the independent strategy. Compare only at the end and return the better-supported final answer."},
    {"id":"MINIMAL-REPAIR","category":"RETRY_RECOVERY","mode":"repair","label":"minimal_repair","instruction":"Locate the single material defect in the candidate and change only what is required to fix it. If no defect is supported, return it unchanged."},
    {"id":"SKEPTICAL-AUDITOR","category":"DELEGATION","mode":"delegation","label":"skeptical_auditor","aux_instruction":"Act as a skeptical specialist. Audit the task and candidate for one decisive hidden error, missed constraint, stale state, or unsupported inference.","final_instruction":"As the final solver, use the specialist audit only if supported by the original task and return the corrected final answer."},
    {"id":"SPECIALIST-DELEGATE","category":"DELEGATION","mode":"delegation","label":"specialist_delegate","aux_instruction":"Act as a specialist analyst. Extract the hardest subproblem, solve only that subproblem, and state the result and evidence.","final_instruction":"Integrate the specialist result into a complete solution of the original task and return only the final answer."},
    {"id":"A-B-A","category":"COMPOSITION_LAYERING","mode":"aba","label":"plan_solve_plancheck","aux_instruction":"A: build the minimum dependency plan and identify the highest-risk step. Do not answer.","middle_instruction":"B: solve the task independently using the plan only as guidance.","final_instruction":"A again: check the candidate specifically against the original plan's highest-risk step and task constraints; repair and return only the final answer."},
    {"id":"SELF-ROUTER","category":"ADAPTIVE_ROUTING","mode":"router","label":"self_router","aux_instruction":"Classify the task using only its visible text. Return exactly one token from: TOOL, STATE, EVIDENCE, FORMAT, PLAN, VERIFY, DIRECT.","final_instruction":"Apply the selected controller minimally and solve the original task."},
    {"id":"RISK-GATED-VERIFY","category":"STOP_ESCALATE_POLICY","mode":"confidence_gate","label":"risk_gated_verify","aux_instruction":"Assess whether this task has material ambiguity, interacting constraints, tool/schema risk, stale state, or multi-step reasoning. Return exactly RISK=HIGH or RISK=LOW.","final_instruction":"Solve the original task. If risk is high, perform one targeted verification before finalizing; if low, answer directly."},
    {"id":"STOP-WHEN-SUFFICIENT","category":"STOP_ESCALATE_POLICY","mode":"repair","label":"stop_when_sufficient","instruction":"Decide whether the candidate is already fully supported. If yes return it unchanged. If not perform exactly one highest-value correction and return the result."},
    {"id":"DIVERSE-CONSENSUS","category":"ENSEMBLE_CONSENSUS","mode":"ensemble","label":"diverse_consensus","aux_instruction":"Solve independently. Prefer a different reasoning route from any obvious default approach.","final_instruction":"Adjudicate the independent candidates using original task evidence only. Return the best final answer."},
    {"id":"TEMP-0","category":"COMPUTE_COST_ROUTING","mode":"single","label":"temperature_0","temperature":0.0},
    {"id":"TEMP-02","category":"COMPUTE_COST_ROUTING","mode":"single","label":"temperature_0_2","temperature":0.2},
    {"id":"TEMP-07","category":"COMPUTE_COST_ROUTING","mode":"single","label":"temperature_0_7","temperature":0.7},
    {"id":"FAILURE-SYNTH-INJECT","category":"RETRY_RECOVERY","mode":"failure_synth","label":"failure_derived_injection","aux_instruction":"From the original task and failed candidate, write one minimal corrective instruction that targets the exact failure without adding unrelated guidance.","final_instruction":"Apply the generated corrective instruction to the original task and return only the corrected final answer."},
    {"id":"COUNTEREXAMPLE-RETRY","category":"RETRY_RECOVERY","mode":"retry","label":"counterexample_retry","aux_instruction":"Find a concrete counterexample or violated requirement that proves the candidate wrong. If none exists, say NONE.","final_instruction":"Retry only if the counterexample is supported; otherwise keep the candidate. Return only the final answer."},
    {"id":"CONSTRAINT-RETRY","category":"RETRY_RECOVERY","mode":"retry","label":"constraint_retry","aux_instruction":"Identify the first hard constraint the candidate violates, if any. Name only that constraint and the evidence.","final_instruction":"Retry from the original task while satisfying the violated constraint and preserving already-correct parts."},
    {"id":"TOOL-SCHEMA-RETRY","category":"TOOL_POLICY","mode":"retry","label":"tool_schema_retry","aux_instruction":"Identify the exact tool-selection or argument-schema defect in the candidate tool-like output.","final_instruction":"Rebuild the tool-like output from the required schema and dependencies; return only the corrected output."},
)

PROMPT_PRIMITIVES: tuple[dict[str, str], ...] = (
    {"id":"REQ","label":"requirements","instruction":"Identify every material requirement and satisfy each exactly once."},
    {"id":"DEC","label":"decomposition","instruction":"Decompose the task into the minimum dependency-ordered steps needed to solve it."},
    {"id":"EVD","label":"evidence","instruction":"Ground every material claim in task evidence or a direct derivation; do not invent missing facts."},
    {"id":"STA","label":"state","instruction":"Resolve current versus superseded state and answer only from the latest authoritative state."},
    {"id":"SCH","label":"schema","instruction":"Construct the exact required output schema before filling it; emit nothing outside the required shape."},
    {"id":"VER","label":"verification","instruction":"Before finalizing, verify the weakest material step and correct only a supported defect."},
    {"id":"ASM","label":"assumptions","instruction":"List the assumptions that could change the answer and reject any assumption not supported by the task."},
    {"id":"CTR","label":"counterexample","instruction":"Try to produce one counterexample to the candidate conclusion; if one survives, repair the answer."},
    {"id":"TMP","label":"temporal","instruction":"Normalize time/order relationships explicitly before reasoning from them."},
    {"id":"QNT","label":"quantitative","instruction":"Track units, signs, ranges, and arithmetic dependencies explicitly before finalizing."},
    {"id":"LOG","label":"logic","instruction":"Separate premises, derived implications, and conclusions; do not reverse implications."},
    {"id":"AMB","label":"ambiguity","instruction":"Identify ambiguity that can change the answer; resolve only from evidence and otherwise state the uncertainty."},
    {"id":"CON","label":"conflict","instruction":"When instructions or facts conflict, identify precedence and preserve the highest-authority constraint."},
    {"id":"MIN","label":"minimality","instruction":"Use the minimum sufficient reasoning and output; do not add unsupported or unnecessary content."},
    {"id":"ALT","label":"alternate_strategy","instruction":"Before committing, consider one materially different solution route and prefer the better-supported result."},
    {"id":"PST","label":"postcondition","instruction":"Define the observable postcondition of success and verify the answer satisfies it before stopping."},
)

# A balanced covering design. Every primitive is exposed to every placement,
# representation, dose, and recurrence level without paying for the full
# Cartesian product (5*3*3*3 = 135 variants per primitive).
CONTROL_DESIGN_ROWS: tuple[tuple[str, str, float, int], ...] = (
    ("system", "prose", 0.5, 1),
    ("prefix", "bullets", 1.0, 1),
    ("middle", "schema", 2.0, 1),
    ("suffix", "prose", 1.0, 2),
    ("system", "bullets", 2.0, 2),
    ("prefix", "schema", 0.5, 2),
    ("middle", "prose", 1.0, 3),
    ("suffix", "bullets", 0.5, 3),
    ("system", "schema", 1.0, 3),
    ("prefix", "prose", 2.0, 3),
    ("middle", "bullets", 0.5, 1),
    ("suffix", "schema", 2.0, 2),
)

CONTROL_GRAMMAR = {
    "prompt_primitives": [row["id"] for row in PROMPT_PRIMITIVES],
    "placements": ["system", "prefix", "middle", "suffix", "post_candidate"],
    "representations": ["prose", "bullets", "schema"],
    "doses": [0.5, 1.0, 2.0],
    "recurrence_counts": [1, 2, 3],
    "retry_policies": [
        "minimal_repair",
        "diagnose_retry",
        "strategy_reset",
        "counterexample_retry",
        "constraint_retry",
        "tool_schema_retry",
    ],
    "multi_call_topologies": [
        "plan_solve",
        "solve_verify",
        "critique_repair",
        "plan_solve_verify",
        "two_independent_adjudicate",
        "specialist_delegate",
        "skeptical_auditor",
        "A+B+A",
    ],
    "memory_context_controls": [
        "evidence_ledger",
        "state_memory",
        "context_select",
        "context_compress",
    ],
    "tool_controls": [
        "tool_schema_projection",
        "tool_dependency_plan",
        "tool_postcondition_repair",
    ],
    "routing_controls": [
        "self_router",
        "risk_gated_verify",
        "stop_when_sufficient",
    ],
    "compute_controls": [
        "thinking_low",
        "thinking_medium",
        "thinking_high",
        "budget_512",
        "budget_1024",
        "budget_2048",
        "context_4096",
        "context_8192",
        "context_16384",
        "context_32768",
        "temperature_0",
        "temperature_0_2",
        "temperature_0_7",
    ],
}


def _render_control_primitive(primitive: dict[str, str], *, dose: float, representation: str) -> str:
    text = str(primitive["instruction"])
    if dose <= 0.5:
        text = "Briefly: " + text
    elif dose >= 2.0:
        text = text + " Treat this as mandatory and verify compliance before stopping."
    if representation == "bullets":
        return "- " + text
    if representation == "schema":
        return json.dumps(
            {"control": primitive["label"], "instruction": text},
            sort_keys=True,
            separators=(",", ":"),
        )
    return text


def generate_prompt_control_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for primitive_index, primitive in enumerate(PROMPT_PRIMITIVES):
        for design_index, design in enumerate(CONTROL_DESIGN_ROWS):
            placement, representation, dose, recurrence = design
            # Rotate design rows per primitive so pairwise level combinations are
            # distributed rather than aligned to the same primitive/order.
            rotated = CONTROL_DESIGN_ROWS[(design_index + primitive_index) % len(CONTROL_DESIGN_ROWS)]
            placement, representation, dose, recurrence = rotated
            instruction = _render_control_primitive(
                primitive,
                dose=float(dose),
                representation=str(representation),
            )
            rows.append({
                "id": f"GRAM-{primitive['id']}-{design_index+1:02d}",
                "category": "PROMPT_CONTROL",
                "mode": "grammar_control",
                "label": f"{primitive['label']}|{placement}|{representation}|dose={dose}|r={recurrence}",
                "primitive_id": primitive["id"],
                "placement": placement,
                "representation": representation,
                "dose": float(dose),
                "recurrence": int(recurrence),
                "instruction": instruction,
            })

        # Every primitive also receives a post-candidate injection arm. This is
        # materially different from pre-answer placement and cannot be inferred
        # from system/prefix/middle/suffix trials.
        rows.append({
            "id": f"GRAM-{primitive['id']}-POST",
            "category": "PROMPT_CONTROL",
            "mode": "post_candidate_injection",
            "label": f"{primitive['label']}|post_candidate",
            "primitive_id": primitive["id"],
            "placement": "post_candidate",
            "representation": "prose",
            "dose": 1.0,
            "recurrence": 1,
            "instruction": str(primitive["instruction"]),
        })
    return rows


ROUTER_INSTRUCTIONS = {
    "TOOL": "Validate tool choice, schema, required arguments, dependencies, and postconditions before answering.",
    "STATE": "Reconcile current versus superseded state before answering.",
    "EVIDENCE": "Ground material claims in task evidence and reject unsupported facts.",
    "FORMAT": "Construct the exact required output shape first and emit nothing extra.",
    "PLAN": "Use a minimal dependency-ordered plan, then solve.",
    "VERIFY": "Solve, then perform one targeted verification of the weakest step.",
    "DIRECT": "Answer directly with the minimum processing needed.",
}


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_TEST12_CONFIG)
    incoming = config.get("test12_campaign")
    if isinstance(incoming, dict):
        merged.update(copy.deepcopy(incoming))
    return merged


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            value = json.loads(raw)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def build_test12_plan(cases: list[dict[str, Any]], *, seed_run: str | None = None) -> dict[str, Any]:
    parts = partition_cases(cases)
    return {
        "schema_version": 2,
        "campaign": "model-harness-compiler-test1.2-collection",
        "source_seed_run": seed_run,
        "stage": "COLLECTION",
        "wall_clock_seconds": COLLECTION_HARD_SECONDS,
        "active_model_seconds": COLLECTION_ACTIVE_SECONDS,
        "call_start_cutoff_seconds": CALL_START_CUTOFF_SECONDS,
        "phases": [{"name": name, "seconds": seconds} for name, seconds in PHASES],
        "partition_counts": {name: len(rows) for name, rows in parts.items()},
        "allowed_partitions": ["DISCOVERY"],
        "reserved_for_tuning": ["VALIDATION"],
        "prohibited_partitions": ["TEST2_BLIND", "TEST3_PROTECTED"],
        "improvement_surface": list(IMPROVEMENT_SURFACE),
        "core_mechanism_count": len(CORE_INTERVENTIONS),
        "generated_prompt_control_count": len(generate_prompt_control_candidates()),
        "finite_control_grammar": copy.deepcopy(CONTROL_GRAMMAR),
        "control_search_contract": "EVERY_DECLARED_CONTROL_CANDIDATE_GETS_MINIMUM_COVERAGE_BEFORE_REPLICATION_DEPTH_IS_ADAPTIVE",
        "adaptive_allocation": {
            "coverage_floor_first": True,
            "screening_design": "BALANCED_COVERING_ARRAY_PLUS_FRACTIONAL_FACTORIAL",
            "then_allocate_by": [
                "expected_information_gain",
                "rescue_probability",
                "negative_transfer_risk",
                "value_per_call",
                "family_coverage_gap",
            ],
            "successive_halving": True,
            "exact_failure_replay": True,
            "oracle_routing_prohibited": True,
            "early_stop_when": "coverage complete and no material decision boundary remains above uncertainty threshold",
        },
        "required_outputs": list(REQUIRED_OUTPUTS),
        "scope_boundaries": {
            "real_external_tool_execution": "SYNTHETIC_IN_PROCESS_TOOL_EXECUTION_ONLY",
            "cross_vendor_harness_comparison": "NOT_REQUIRED_FOR_MODEL_HARNESS_COMPILATION",
            "weight_update": "QUALIFICATION_ONLY",
            "logical_parallel_branches": "SERIALIZED_FOR_LOCAL_COMPUTE_CONTROL",
            "validation_partition": "RESERVED_FOR_TUNING_RUN",
        },
    }



def validate_test12_plan(plan: dict[str, Any]) -> None:
    if int(plan["wall_clock_seconds"]) != COLLECTION_HARD_SECONDS:
        raise ValueError("Test 1.2 collection hard ceiling must be 6h15m")
    if sum(int(row["seconds"]) for row in plan["phases"]) != COLLECTION_ACTIVE_SECONDS:
        raise ValueError("Test 1.2 collection active phases must total exactly six hours")
    if plan.get("allowed_partitions") != ["DISCOVERY"]:
        raise ValueError("Test 1.2 collection may use DISCOVERY only")
    if plan.get("reserved_for_tuning") != ["VALIDATION"]:
        raise ValueError("Test 1.2 must reserve VALIDATION for the tuning run")
    if set(plan["prohibited_partitions"]) != {"TEST2_BLIND", "TEST3_PROTECTED"}:
        raise ValueError("Test 1.2 protected partition contract changed")
    for name in ("DISCOVERY", "VALIDATION", "TEST2_BLIND", "TEST3_PROTECTED"):
        if int(plan["partition_counts"].get(name, 0)) <= 0:
            raise ValueError(f"{name} partition is empty")
    missing_surface = sorted(set(IMPROVEMENT_SURFACE) - set(plan.get("improvement_surface") or []))
    if missing_surface:
        raise ValueError(f"Test 1.2 improvement surface incomplete: {missing_surface}")
    if int(plan["core_mechanism_count"]) < 30:
        raise ValueError("Test 1.2 must expose at least 30 core improvement mechanisms")
    if int(plan.get("generated_prompt_control_count", 0)) < 200:
        raise ValueError("Test 1.2 prompt/injection grammar is too small for full control-surface collection")
    if plan.get("control_search_contract") != "EVERY_DECLARED_CONTROL_CANDIDATE_GETS_MINIMUM_COVERAGE_BEFORE_REPLICATION_DEPTH_IS_ADAPTIVE":
        raise ValueError("Test 1.2 may not prune declared controls before minimum coverage")
    missing_outputs = [name for name in REQUIRED_OUTPUTS if name not in plan["required_outputs"]]
    if missing_outputs:
        raise ValueError(f"Test 1.2 output contract incomplete: {missing_outputs}")


def load_test11_source(results_root: Path, run_id: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    run_dir = results_root / run_id
    if not run_dir.is_dir():
        raise ValueError(f"Test-1.1 source run does not exist: {run_id}")
    problems = EvidenceStore(results_root, run_id).verify_manifest_paths(REQUIRED_TEST11_FILES)
    if problems:
        raise ValueError(f"Test-1.1 consumed-artifact verification failed: {problems}")
    missing = [name for name in REQUIRED_TEST11_FILES if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"Test-1.1 source missing required artifacts: {missing}")

    source_parts = (_read_json(run_dir / "fixture-partitions.json").get("partitions") or {})
    computed = partition_cases(cases)
    for name, rows in computed.items():
        if set(source_parts.get(name) or []) != {_fixture_id(case) for case in rows}:
            raise ValueError(f"fixture partition drift detected for {name}")

    observations = _read_jsonl(run_dir / "test1.1-observations.jsonl")
    baseline_values: dict[str, list[float]] = defaultdict(list)
    for row in observations:
        fixture_id = str(row.get("fixture_id") or "")
        if not fixture_id:
            continue
        value = row.get("control_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            baseline_values[fixture_id].append(float(value))

    manifest_bytes = (run_dir / EvidenceStore.MANIFEST_NAME).read_bytes()
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "partitions": source_parts,
        "observations": observations,
        "baselines": {key: float(median(values)) for key, values in baseline_values.items() if values},
        "ingredient_registry": _read_json(run_dir / "ingredient-harvest-registry.json"),
        "recipe_atlas": _read_json(run_dir / "recipe-factorial-atlas.json"),
        "operator_atlas": _read_json(run_dir / "system-operator-atlas.json"),
        "family_generalization": _read_json(run_dir / "family-generalization-map.json"),
        "generation_budget": _read_json(run_dir / "generation-budget-map.json"),
        "residual_ownership": _read_json(run_dir / "residual-failure-ownership.json"),
        "fine_tuning_readiness": _read_json(run_dir / "fine-tuning-readiness-map.json"),
        "priority_queue": _read_json(run_dir / "test1.1-priority-queue.json"),
        "uncertainty": _read_json(run_dir / "test1.1-uncertainty-ledger.json"),
        "handoff": _read_json(run_dir / "corrective-handoff.json"),
        "assurance": _read_json(run_dir / "assurance-map.json"),
        "source_integrity": {
            "verification_mode": "FULL_TEST11_HANDOFF",
            "verified_paths": list(REQUIRED_TEST11_FILES),
            "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
    }


def fresh_model_source(cases: list[dict[str, Any]]) -> dict[str, Any]:
    parts = partition_cases(cases)
    return {
        "run_id": "FRESH-MODEL",
        "run_dir": None,
        "partitions": {name: [_fixture_id(case) for case in rows] for name, rows in parts.items()},
        "observations": [],
        "baselines": {},
        "ingredient_registry": {"schema_version": 1, "ingredients": [], "promoted_ids": [], "screen": {}},
        "recipe_atlas": {"schema_version": 1, "recipes": {}},
        "operator_atlas": {"schema_version": 1, "operators": {}},
        "family_generalization": {"schema_version": 1, "effects": {}},
        "generation_budget": {"schema_version": 1, "effects": {}},
        "residual_ownership": {"schema_version": 1, "fixtures": {}, "phenotypes": {}},
        "fine_tuning_readiness": {"schema_version": 1, "candidates": {}},
        "priority_queue": {"schema_version": 1, "queue": []},
        "uncertainty": {"schema_version": 1, "unknowns": []},
        "handoff": {"schema_version": 1, "priority_queue": [], "ingredient_ids": []},
        "assurance": {"schema_version": 1, "not_looked_at": ["VALIDATION", "TEST2_BLIND", "TEST3_PROTECTED"]},
        "source_integrity": {"verification_mode": "FRESH_MODEL_NO_PRIOR_EVIDENCE"},
    }


def synthetic_test11_source(cases: list[dict[str, Any]]) -> dict[str, Any]:
    # Backwards-compatible dry-run alias. A real new-model collection also starts
    # without model-specific prior evidence.
    return fresh_model_source(cases)



def _source_recipe_messages(case: dict[str, Any], source: dict[str, Any], recipe: dict[str, Any]) -> list[dict[str, str]]:
    bank = {str(row["id"]): row for row in (source.get("ingredient_registry") or {}).get("ingredients", []) or []}
    buckets: dict[str, list[str]] = {"system": [], "prefix": [], "middle": [], "suffix": []}
    for step in recipe.get("steps") or []:
        item = bank.get(str(step.get("ingredient_id")))
        if not item:
            continue
        dose = float(step.get("dose", 1.0))
        text = str(item.get("short") if dose <= 0.5 else item.get("full") or item.get("short") or "")
        if dose >= 2.0:
            text += " Treat this as mandatory, but do not repeat the check unnecessarily."
        representation = str(step.get("representation") or "prose")
        if representation == "bullets":
            text = "- " + text
        elif representation == "schema":
            text = json.dumps({"control": item.get("label"), "requirement": text}, sort_keys=True, separators=(",", ":"))
        buckets.setdefault(str(step.get("placement") or "prefix"), []).append(text)

    prompt = str(case["prompt"])
    messages: list[dict[str, str]] = []
    if buckets["system"]:
        messages.append({"role": "system", "content": "\n".join(buckets["system"])})
    if buckets["middle"]:
        words = prompt.split()
        pivot = max(1, len(words) // 2)
        prompt = " ".join(words[:pivot]) + "\n\nCONTROL:\n" + "\n".join(buckets["middle"]) + "\n\n" + " ".join(words[pivot:])
    if buckets["prefix"]:
        prompt = "CONTROL:\n" + "\n".join(buckets["prefix"]) + "\n\nTASK:\n" + prompt
    if buckets["suffix"]:
        prompt += "\n\nADDITIONAL CONTROL:\n" + "\n".join(buckets["suffix"])
    messages.append({"role": "user", "content": prompt})
    return messages


def _source_prompt_arms(source: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = list((source.get("priority_queue") or {}).get("queue", []) or [])
    result = []
    seen = set()
    for row in rows:
        recipe = row.get("recipe")
        if not isinstance(recipe, dict) or not recipe.get("steps"):
            continue
        recipe_id = str(recipe.get("recipe_id") or row.get("key") or "")
        if not recipe_id or recipe_id in seen:
            continue
        seen.add(recipe_id)
        result.append({
            "id": "SRC-" + recipe_id[:28],
            "category": "PROMPT_CONTROL",
            "mode": "source_recipe",
            "label": "test11_promoted_recipe",
            "source_recipe": copy.deepcopy(recipe),
            "source_classification": row.get("classification"),
        })
        if len(result) >= int(limit):
            break
    return result


def build_intervention_bank(source: dict[str, Any], *, max_source_recipes: int = 8) -> list[dict[str, Any]]:
    rows = [copy.deepcopy(row) for row in CORE_INTERVENTIONS]
    rows.extend(generate_prompt_control_candidates())
    rows.extend(_source_prompt_arms(source, max_source_recipes))
    seen = set()
    unique = []
    for row in rows:
        key = str(row["id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _spec(
    sequence: int,
    case: dict[str, Any],
    label: str,
    cfg: dict[str, Any],
    intervention: dict[str, Any] | None,
    *,
    seed: int,
    baseline: bool = False,
) -> ExperimentSpec:
    iv = intervention or {}
    effort = iv.get("reasoning_effort")
    thinking = isinstance(effort, str) and effort in {"low", "medium", "high"}
    budget = int(iv.get("generation_budget") or cfg["base_generation_budget"])
    context_request = iv.get("context_request")
    return ExperimentSpec(
        experiment_id=make_experiment_id(sequence, _fixture_id(case), label),
        parent_experiment_id=None,
        task_id=_fixture_id(case),
        task_family=_family(case),
        difficulty_level=int(case.get("difficulty_level", 0)),
        hypothesis="Test 1.2 full-system improvement intervention",
        changed_variable="baseline" if baseline else "prompt_variant",
        thinking_mode=thinking,
        reasoning_effort=str(effort) if thinking else None,
        generation_budget=budget,
        context_request=int(context_request) if isinstance(context_request, int) else None,
        temperature=float(iv.get("temperature", 0.0)),
        seed=int(seed),
        prompt_variant="base" if baseline else str(iv.get("id") or "intervention"),
        recovery_level=None,
    )


def _metric_number(row: dict[str, Any], key: str) -> float:
    value = (row.get("metrics") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _latency_seconds(row: dict[str, Any]) -> float:
    timing = row.get("timing") or {}
    value = timing.get("client_latency_ns")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) / 1_000_000_000.0
    return 0.0


class Test12Campaign:
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
        self.case_by_id = {_fixture_id(case): case for case in cases}
        self.interventions = build_intervention_bank(source, max_source_recipes=int(self.cfg["max_source_recipes"]))
        self.intervention_by_id = {str(row["id"]): row for row in self.interventions}
        self.allowed_partitions = {"DISCOVERY"}
        self.controls: dict[tuple[str, int], dict[str, Any]] = {}
        self.rows: list[dict[str, Any]] = []
        self.sequence = 0
        self.phase_assertions: list[dict[str, Any]] = []

    def partition_name(self, case: dict[str, Any]) -> str:
        fixture_id = _fixture_id(case)
        for name, rows in self.partitions.items():
            if any(_fixture_id(row) == fixture_id for row in rows):
                return name
        return "UNKNOWN"

    def assert_allowed(self, case: dict[str, Any]) -> None:
        partition = self.partition_name(case)
        if partition not in self.allowed_partitions:
            raise ValueError(
                f"Test 1.2 attempted to use partition {partition}; allowed={sorted(self.allowed_partitions)}"
            )

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
                self.runner._record_progress("plan_adjusted", label, old_total=old, new_total=int(progress.total_tasks), reason="wall-clock Test-1.2 capacity extended")
            self.runner._progress_begin(label)
        else:
            self.runner._progress_complete(label)

    def control(self, case: dict[str, Any], deadline: float, *, seed: int, force: bool = False) -> dict[str, Any] | None:
        self.assert_allowed(case)
        key = (_fixture_id(case), int(seed))
        if not force and key in self.controls:
            return self.controls[key]
        if not self.can_start(deadline):
            return None
        self.sequence += 1
        spec = _spec(self.sequence, case, f"control-s{seed}", self.cfg, None, seed=seed, baseline=True)
        label = f"test1.2 control {_fixture_id(case)} s{seed}"
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
            "response_text": str(row.get("response_text") or ""),
            "experiment_id": spec.experiment_id,
            "metrics": copy.deepcopy(row.get("metrics") or {}),
            "timing": copy.deepcopy(row.get("timing") or {}),
        }
        self.controls[key] = record
        self._record(case, row, phase="control", intervention={"id":"CONTROL","category":"CONTROL","mode":"control"}, control=record, seed=seed, aux=[])
        return record

    def _aux(
        self,
        case: dict[str, Any],
        deadline: float,
        *,
        stage: str,
        messages: list[dict[str, str]],
        intervention: dict[str, Any],
        seed: int,
        call_index: int,
    ) -> dict[str, Any] | None:
        if not self.can_start(deadline):
            return None
        self.sequence += 1
        effort = intervention.get("reasoning_effort")
        thinking = isinstance(effort, str) and effort in {"low","medium","high"}
        budget = int(intervention.get("aux_generation_budget") or min(512, int(intervention.get("generation_budget") or self.cfg["base_generation_budget"])))
        options = self.runner._generation_options(case, {
            "num_predict": budget,
            "temperature": float(intervention.get("temperature", 0.0)),
            "seed": int(seed + 1000 + call_index),
        })
        context_request = intervention.get("context_request")
        if isinstance(context_request, int):
            options["num_ctx"] = int(context_request)
        label = f"test1.2 {stage} {_fixture_id(case)} {intervention['id']} aux{call_index}"
        self._progress(label, True)
        try:
            generation, invocation, refs = self.runner._invoke_generation(
                stage="test1.2-aux",
                case_id=f"{_fixture_id(case)}-{intervention['id']}-aux{call_index}-{self.sequence}",
                messages=messages,
                options=options,
                request_fields={"think": str(effort) if thinking else False},
            )
        finally:
            self._progress(label, False)
        text = str((generation.get("normalized") or {}).get("text") or "")
        return {
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text_chars": len(text),
            "ok": bool(generation.get("ok")),
            "metrics": copy.deepcopy(generation.get("metrics") or {}),
            "timing": copy.deepcopy(generation.get("timing") or {}),
            "evidence_refs": refs,
            "invocation": {"request_fields": copy.deepcopy((invocation or {}).get("request_fields") or {})},
        }

    @staticmethod
    def _router_choice(text: str) -> str:
        upper = str(text).upper()
        for key in ("TOOL","STATE","EVIDENCE","FORMAT","PLAN","VERIFY","DIRECT"):
            if key in upper:
                return key
        return "VERIFY"

    def treatment(
        self,
        case: dict[str, Any],
        deadline: float,
        *,
        phase: str,
        intervention: dict[str, Any],
        seed: int,
    ) -> dict[str, Any] | None:
        self.assert_allowed(case)
        control = self.control(case, deadline, seed=seed)
        if control is None or not self.can_start(deadline):
            return None
        mode = str(intervention.get("mode") or "single")
        prompt = str(case["prompt"])
        candidate = str(control.get("response_text") or "")
        aux: list[dict[str, Any]] = []
        messages: list[dict[str, str]]
        final_instruction = str(intervention.get("instruction") or intervention.get("final_instruction") or "")

        def add_aux(stage: str, msgs: list[dict[str, str]]) -> str:
            row = self._aux(case, deadline, stage=stage, messages=msgs, intervention=intervention, seed=seed, call_index=len(aux)+1)
            if row is None:
                return ""
            aux.append(row)
            return str(row.get("text") or "")

        if mode == "source_recipe":
            messages = _source_recipe_messages(case, self.source, intervention["source_recipe"])
        elif mode == "single":
            messages = [{"role":"user","content":prompt}]
            if final_instruction:
                messages = [{"role":"system","content":final_instruction}, *messages]
        elif mode == "grammar_control":
            placement = str(intervention.get("placement") or "prefix")
            recurrence = max(1, int(intervention.get("recurrence") or 1))
            instruction = "\n".join([str(intervention.get("instruction") or "")] * recurrence)
            if placement == "system":
                messages = [{"role":"system","content":instruction},{"role":"user","content":prompt}]
            elif placement == "suffix":
                messages = [{"role":"user","content":prompt + "\n\nADDITIONAL CONTROL:\n" + instruction}]
            elif placement == "middle":
                words = prompt.split()
                pivot = max(1, len(words)//2)
                mixed = " ".join(words[:pivot]) + "\n\nCONTROL:\n" + instruction + "\n\n" + " ".join(words[pivot:])
                messages = [{"role":"user","content":mixed}]
            else:
                messages = [{"role":"user","content":"CONTROL:\n" + instruction + "\n\nTASK:\n" + prompt}]
        elif mode == "post_candidate_injection":
            messages = [
                {"role":"user","content":prompt},
                {"role":"assistant","content":candidate},
                {"role":"user","content":str(intervention.get("instruction") or "") + "\nReturn only the final answer."},
            ]
        elif mode == "precompute":
            artifact = add_aux("precompute", [{"role":"user","content":prompt + "\n\n" + str(intervention.get("aux_instruction") or "")}])
            messages = [
                {"role":"user","content":prompt},
                {"role":"assistant","content":"AUXILIARY WORKING STATE:\n" + artifact},
                {"role":"user","content":str(intervention.get("final_instruction") or "Solve the original task and return the final answer.")},
            ]
        elif mode == "repair":
            messages = [
                {"role":"user","content":prompt},
                {"role":"assistant","content":candidate},
                {"role":"user","content":final_instruction},
            ]
        elif mode == "critique_repair":
            critique = add_aux("critique", [
                {"role":"user","content":prompt},
                {"role":"assistant","content":candidate},
                {"role":"user","content":str(intervention.get("aux_instruction") or "")},
            ])
            messages = [
                {"role":"user","content":prompt},
                {"role":"assistant","content":candidate},
                {"role":"user","content":"CRITIQUE:\n" + critique + "\n\n" + str(intervention.get("final_instruction") or "")},
            ]
        elif mode == "failure_synth":
            corrective = add_aux("synthesize-control", [
                {"role":"user","content":prompt},
                {"role":"assistant","content":candidate},
                {"role":"user","content":str(intervention.get("aux_instruction") or "")},
            ])
            messages = [
                {"role":"system","content":corrective},
                {"role":"user","content":prompt},
                {"role":"assistant","content":candidate},
                {"role":"user","content":str(intervention.get("final_instruction") or "")},
            ]
        elif mode == "retry":
            diagnosis = add_aux("diagnose", [
                {"role":"user","content":prompt},
                {"role":"assistant","content":candidate},
                {"role":"user","content":str(intervention.get("aux_instruction") or "")},
            ])
            messages = [
                {"role":"user","content":prompt},
                {"role":"assistant","content":"PRIOR CANDIDATE:\n" + candidate},
                {"role":"user","content":"DIAGNOSIS OR ALTERNATE STRATEGY:\n" + diagnosis + "\n\n" + str(intervention.get("final_instruction") or "")},
            ]
        elif mode == "delegation":
            specialist = add_aux("specialist", [
                {"role":"user","content":prompt + "\n\nSPECIALIST ROLE:\n" + str(intervention.get("aux_instruction") or "")}
            ])
            messages = [
                {"role":"user","content":prompt},
                {"role":"assistant","content":"SPECIALIST REPORT:\n" + specialist},
                {"role":"user","content":str(intervention.get("final_instruction") or "")},
            ]
        elif mode == "ensemble":
            a = add_aux("branch-a", [{"role":"user","content":prompt + "\n\n" + str(intervention.get("aux_instruction") or "Solve independently.")}])
            b = add_aux("branch-b", [{"role":"user","content":prompt + "\n\nSolve independently using a materially different reasoning route."}])
            messages = [
                {"role":"user","content":prompt},
                {"role":"assistant","content":"CANDIDATE A:\n" + a + "\n\nCANDIDATE B:\n" + b},
                {"role":"user","content":str(intervention.get("final_instruction") or "")},
            ]
        elif mode == "three_stage":
            plan = add_aux("plan", [{"role":"user","content":prompt + "\n\n" + str(intervention.get("aux_instruction") or "")}])
            solve = add_aux("solve", [
                {"role":"user","content":prompt},
                {"role":"assistant","content":"PLAN:\n" + plan},
                {"role":"user","content":str(intervention.get("middle_instruction") or "Solve and produce a candidate.")},
            ])
            messages = [
                {"role":"user","content":prompt},
                {"role":"assistant","content":"PLAN:\n" + plan + "\n\nCANDIDATE:\n" + solve},
                {"role":"user","content":str(intervention.get("final_instruction") or "")},
            ]
        elif mode == "aba":
            a1 = add_aux("a1", [{"role":"user","content":prompt + "\n\n" + str(intervention.get("aux_instruction") or "")}])
            b = add_aux("b", [
                {"role":"user","content":prompt},
                {"role":"assistant","content":"A1:\n" + a1},
                {"role":"user","content":str(intervention.get("middle_instruction") or "")},
            ])
            messages = [
                {"role":"user","content":prompt},
                {"role":"assistant","content":"A1:\n" + a1 + "\n\nB CANDIDATE:\n" + b},
                {"role":"user","content":str(intervention.get("final_instruction") or "")},
            ]
        elif mode == "router":
            route_text = add_aux("router", [{"role":"user","content":prompt + "\n\nROUTER:\n" + str(intervention.get("aux_instruction") or "")}])
            choice = self._router_choice(route_text)
            intervention = {**copy.deepcopy(intervention), "router_choice": choice}
            messages = [
                {"role":"system","content":ROUTER_INSTRUCTIONS[choice]},
                {"role":"user","content":prompt},
            ]
        elif mode == "confidence_gate":
            risk = add_aux("risk", [{"role":"user","content":prompt + "\n\nRISK ASSESSMENT:\n" + str(intervention.get("aux_instruction") or "")}])
            high = "HIGH" in risk.upper()
            intervention = {**copy.deepcopy(intervention), "risk_gate": "HIGH" if high else "LOW"}
            gate_instruction = (
                "Perform one targeted verification of the weakest material step before finalizing."
                if high else
                "Answer directly after one lightweight consistency check."
            )
            messages = [{"role":"system","content":gate_instruction},{"role":"user","content":prompt}]
        else:
            raise ValueError(f"unknown Test 1.2 intervention mode: {mode}")

        if not self.can_start(deadline):
            return None
        self.sequence += 1
        spec = _spec(self.sequence, case, f"{phase}-{intervention['id']}-s{seed}", self.cfg, intervention, seed=seed)
        label = f"test1.2 {phase} {_fixture_id(case)} {intervention['id']}"
        self._progress(label, True)
        try:
            row = execute_experiment(self.runner, case, spec, parent=None, messages_override=messages)
        finally:
            self._progress(label, False)
        return self._record(case, row, phase=phase, intervention=intervention, control=control, seed=seed, aux=aux)

    def _record(
        self,
        case: dict[str, Any],
        row: dict[str, Any],
        *,
        phase: str,
        intervention: dict[str, Any],
        control: dict[str, Any],
        seed: int,
        aux: list[dict[str, Any]],
    ) -> dict[str, Any]:
        score = row.get("score")
        valid = (row.get("classification") or {}).get("valid_for_capability") is True
        numeric = float(score) if valid and isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0
        aux_prompt = sum(float((item.get("metrics") or {}).get("prompt_eval_count") or 0) for item in aux)
        aux_output = sum(float((item.get("metrics") or {}).get("eval_count") or 0) for item in aux)
        aux_latency = sum(
            float((item.get("timing") or {}).get("client_latency_ns") or 0) / 1_000_000_000.0
            for item in aux
        )
        prompt_tokens = aux_prompt + _metric_number(row, "prompt_eval_count")
        output_tokens = aux_output + _metric_number(row, "eval_count")
        latency_s = aux_latency + _latency_seconds(row)
        result = {
            "schema_version": 1,
            "timestamp_utc": self.runner._utc(),
            "phase": phase,
            "fixture_id": _fixture_id(case),
            "family_id": _family(case),
            "task_text": str(case.get("prompt") or ""),
            "difficulty_level": int(case.get("difficulty_level", 0)),
            "partition": self.partition_name(case),
            "experiment_id": (row.get("experiment") or {}).get("experiment_id"),
            "intervention_id": str(intervention.get("id") or "CONTROL"),
            "intervention_category": str(intervention.get("category") or "CONTROL"),
            "intervention_mode": str(intervention.get("mode") or "control"),
            "router_choice": intervention.get("router_choice"),
            "risk_gate": intervention.get("risk_gate"),
            "classification": copy.deepcopy(row.get("classification") or {}),
            "score": numeric,
            "control_score": float(control.get("score") or 0.0),
            "delta": numeric - float(control.get("score") or 0.0),
            "control_response_text": str(control.get("response_text") or ""),
            "treatment_response_text": str(row.get("response_text") or ""),
            "generation_budget": int((row.get("experiment") or {}).get("generation_budget") or self.cfg["base_generation_budget"]),
            "reasoning_effort": (row.get("experiment") or {}).get("reasoning_effort"),
            "context_request": (row.get("experiment") or {}).get("context_request"),
            "temperature": (row.get("experiment") or {}).get("temperature"),
            "seed": int(seed),
            "model_calls_per_application": 0 if str(intervention.get("id")) == "CONTROL" else len(aux) + 1,
            "auxiliary_stage_count": len(aux),
            "auxiliary_stages": [
                {
                    "text_sha256": item.get("text_sha256"),
                    "text_chars": item.get("text_chars"),
                    "ok": item.get("ok"),
                    "metrics": item.get("metrics"),
                    "timing": item.get("timing"),
                    "evidence_refs": item.get("evidence_refs"),
                }
                for item in aux
            ],
            "cost": {
                "prompt_tokens_observed": prompt_tokens,
                "output_tokens_observed": output_tokens,
                "wall_seconds": latency_s,
            },
            "evidence_refs": copy.deepcopy(row.get("evidence_refs") or {}),
        }
        self.rows.append(result)
        self.runner.store.append_jsonl("test1.2-observations.jsonl", result)
        return result

    def positive_work(self, phase: str, start_index: int, expected_coverage: str) -> None:
        rows = self.rows[start_index:]
        assertion = {
            "schema_version": 1,
            "phase": phase,
            "phase_nonce": hashlib.sha256(f"{self.runner.store.run_id}|{phase}".encode("utf-8")).hexdigest(),
            "rules_hash": hashlib.sha256("\n".join(RULES).encode("utf-8")).hexdigest(),
            "expected_coverage": expected_coverage,
            "completed_observation_count": len(rows),
            "completed_fixture_count": len({str(row.get("fixture_id")) for row in rows}),
            "mechanisms_tested": sorted({str(row.get("intervention_id")) for row in rows if row.get("intervention_id") != "CONTROL"}),
            "categories_tested": sorted({str(row.get("intervention_category")) for row in rows if row.get("intervention_category") != "CONTROL"}),
            "families_tested": sorted({str(row.get("family_id")) for row in rows}),
            "protected_partitions": ["TEST2_BLIND", "TEST3_PROTECTED"],
            "status": "VERIFIED_WORK_PERFORMED" if rows else "INSUFFICIENT_WORK",
        }
        self.phase_assertions.append(assertion)
        self.runner.store.append_jsonl("positive-work-assertions-1.2.jsonl", assertion)


def _wilson(successes: int, total: int, z: float = 1.6448536269514722) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def mechanism_summary(rows: Iterable[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    data = [row for row in rows if row.get("intervention_id") not in {None, "CONTROL"}]
    fails = [row for row in data if float(row.get("control_score", 0.0)) < 1.0]
    passes = [row for row in data if float(row.get("control_score", 0.0)) >= 1.0]
    rescues = [row for row in fails if float(row.get("score", 0.0)) > float(row.get("control_score", 0.0))]
    regressions = [row for row in passes if float(row.get("score", 0.0)) < float(row.get("control_score", 0.0))]
    cap_reg = [
        row for row in regressions
        if str((row.get("classification") or {}).get("result_class") or "") in CAPABILITY_FAILURE_CLASSES
    ]
    trunc_reg = [
        row for row in regressions
        if str((row.get("classification") or {}).get("result_class") or "") in TRUNCATION_CLASSES
    ]
    rescue_rate = len(rescues) / len(fails) if fails else 0.0
    cap_reg_rate = len(cap_reg) / len(passes) if passes else 0.0
    mean_calls = mean([float(row.get("model_calls_per_application") or 0) for row in data]) if data else 0.0
    mean_tokens = mean([
        float((row.get("cost") or {}).get("prompt_tokens_observed") or 0)
        + float((row.get("cost") or {}).get("output_tokens_observed") or 0)
        for row in data
    ]) if data else 0.0
    mean_latency = mean([float((row.get("cost") or {}).get("wall_seconds") or 0) for row in data]) if data else 0.0

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
    elif fails and not rescues:
        classification = "NO_RESCUE_SIGNAL"
    elif regressions and not cap_reg:
        classification = "TRUNCATION_SENSITIVE"
    else:
        classification = "UNCERTAIN"

    raw_value = rescue_rate - 2.0 * cap_reg_rate
    cost_penalty = (
        float(cfg["call_cost_penalty"]) * mean_calls
        + float(cfg["token_cost_penalty"]) * mean_tokens
        + float(cfg["latency_cost_penalty"]) * mean_latency
    )
    return {
        "n": len(data),
        "baseline_fail_trials": len(fails),
        "baseline_pass_trials": len(passes),
        "rescues": len(rescues),
        "regressions": len(regressions),
        "capability_regressions": len(cap_reg),
        "truncation_regressions": len(trunc_reg),
        "rescue_rate": rescue_rate,
        "capability_regression_rate": cap_reg_rate,
        "rescue_wilson90": _wilson(len(rescues), len(fails)),
        "mean_model_calls": mean_calls,
        "mean_tokens_observed": mean_tokens,
        "mean_wall_seconds": mean_latency,
        "raw_value": raw_value,
        "cost_penalty": cost_penalty,
        "net_value": raw_value - cost_penalty,
        "value_per_call": raw_value / mean_calls if mean_calls > 0 else 0.0,
        "classification": classification,
        "fixture_ids": sorted({str(row.get("fixture_id")) for row in data}),
        "family_ids": sorted({str(row.get("family_id")) for row in data}),
    }


def _group_summary(rows: list[dict[str, Any]], cfg: dict[str, Any], key_fn: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("intervention_id") in {None, "CONTROL"}:
            continue
        groups[key_fn(row)].append(row)
    return {key: mechanism_summary(values, cfg) for key, values in groups.items()}


def _source_headroom(campaign: Test12Campaign, partition: str = "DISCOVERY") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = campaign.partitions[partition]
    current: dict[str, list[float]] = defaultdict(list)
    for row in campaign.rows:
        if row.get("intervention_id") == "CONTROL" and row.get("partition") == partition:
            current[str(row["fixture_id"])].append(float(row.get("score", 0.0)))
    baseline = {
        key: float(median(values))
        for key, values in current.items()
        if values
    }
    fail = [case for case in rows if baseline.get(_fixture_id(case), 0.0) < 1.0]
    passed = [case for case in rows if baseline.get(_fixture_id(case), 0.0) >= 1.0]
    return (
        _balanced_cases(fail, min(96, len(fail))),
        _balanced_cases(passed, min(96, len(passed))),
    )



def _breadth_cover(
    campaign: Test12Campaign,
    deadline: float,
    *,
    phase: str,
    interventions: list[dict[str, Any]],
    failure_cases: list[dict[str, Any]],
    sentinel_cases: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Mandatory breadth pass with family-rotating fixture assignment.

    Every candidate receives one failure-side and one pass-sentinel trial when
    both pools exist. Candidate-to-fixture assignment rotates so the full
    catalog is not judged on the same tiny pair of fixtures.
    """
    start = len(campaign.rows)
    failures = _balanced_cases(failure_cases, len(failure_cases))
    sentinels = _balanced_cases(sentinel_cases, len(sentinel_cases))
    for index, intervention in enumerate(interventions):
        if not campaign.can_start(deadline):
            break
        if failures:
            case = failures[index % len(failures)]
            campaign.treatment(
                case,
                deadline,
                phase=phase,
                intervention=intervention,
                seed=seed,
            )
        if not campaign.can_start(deadline):
            break
        if sentinels:
            # Prime stride reduces repeated pair alignment when pool sizes share
            # small factors with the candidate catalog.
            case = sentinels[(index * 7 + 3) % len(sentinels)]
            campaign.treatment(
                case,
                deadline,
                phase=phase,
                intervention=intervention,
                seed=seed,
            )
    return campaign.rows[start:]


def _matrix(
    campaign: Test12Campaign,
    deadline: float,
    *,
    phase: str,
    interventions: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    seeds: list[int] | None = None,
    coverage_rounds: int | None = None,
) -> list[dict[str, Any]]:
    if not interventions or not cases:
        return []
    seed_values = seeds or [int(v) for v in campaign.cfg["seeds"]]
    start = len(campaign.rows)
    cursor = 0
    max_rounds = coverage_rounds
    max_trials = None if max_rounds is None else len(interventions) * len(cases) * max_rounds
    while campaign.can_start(deadline):
        intervention = interventions[cursor % len(interventions)]
        case = cases[(cursor // len(interventions)) % len(cases)]
        seed = seed_values[(cursor // max(1, len(interventions) * len(cases))) % len(seed_values)]
        campaign.treatment(case, deadline, phase=phase, intervention=intervention, seed=seed)
        cursor += 1
        if max_trials is not None and cursor >= max_trials:
            break
        if max_trials is None and cursor >= len(interventions) * len(cases) * len(seed_values):
            break
    return campaign.rows[start:]


def _coverage_cases(campaign: Test12Campaign, partition: str = "DISCOVERY") -> list[dict[str, Any]]:
    fail, passed = _source_headroom(campaign, partition)
    fail_n = int(campaign.cfg["coverage_floor_failures"])
    pass_n = int(campaign.cfg["coverage_floor_sentinels"])
    return _balanced_cases(fail, min(fail_n, len(fail))) + _balanced_cases(passed, min(pass_n, len(passed)))


def _rank_mechanisms(summary: dict[str, Any], campaign: Test12Campaign, limit: int) -> list[dict[str, Any]]:
    ranked = []
    for key, value in summary.items():
        iv = campaign.intervention_by_id.get(str(key))
        if iv is None:
            continue
        cls = str(value.get("classification"))
        rank = {"STRONG_CONDITIONAL_RESCUE":5,"PROMISING_CONDITIONAL_RESCUE":4,"UNCERTAIN":2,"TRUNCATION_SENSITIVE":1,"NO_RESCUE_SIGNAL":0,"CAPABILITY_HARM":-10}.get(cls,0)
        ranked.append((rank, float(value.get("net_value", 0.0)), float(value.get("value_per_call", 0.0)), str(key), iv))
    ranked.sort(reverse=True)
    return [copy.deepcopy(row[-1]) for row in ranked[:limit]]


def capability_frontier_cover(
    cases: list[dict[str, Any]],
    *,
    target_levels: tuple[int, ...] = (0, 2, 5, 8, 10),
) -> list[dict[str, Any]]:
    """Pick difficulty-spanning representatives for every capability family."""
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_family[_family(case)].append(case)
    selected: list[dict[str, Any]] = []
    for family in sorted(by_family):
        pool = sorted(
            by_family[family],
            key=lambda row: (int(row.get("difficulty_level", 0)), _fixture_id(row)),
        )
        used: set[str] = set()
        for target in target_levels:
            candidates = [row for row in pool if _fixture_id(row) not in used]
            if not candidates:
                break
            chosen = min(
                candidates,
                key=lambda row: (
                    abs(int(row.get("difficulty_level", 0)) - int(target)),
                    int(row.get("difficulty_level", 0)),
                    _fixture_id(row),
                ),
            )
            used.add(_fixture_id(chosen))
            selected.append(chosen)
    return selected


def phase_baseline(campaign: Test12Campaign, deadline: float) -> dict[str, Any]:
    start = len(campaign.rows)
    # Map every capability family across easy, middle, hard, and boundary
    # difficulty points rather than exhausting redundant easy fixtures.
    discovery = capability_frontier_cover(campaign.partitions["DISCOVERY"])
    primary_seed = int(campaign.cfg["seeds"][0])
    for case in discovery:
        if not campaign.can_start(deadline):
            break
        campaign.control(case, deadline, seed=primary_seed, force=True)

    # Replicate a balanced slice across families and difficulty so instability is
    # measurable without paying for full duplicate coverage.
    first_rows = [
        row for row in campaign.rows[start:]
        if row.get("intervention_id") == "CONTROL"
    ]
    failures = [
        campaign.case_by_id[str(row["fixture_id"])]
        for row in first_rows
        if float(row.get("score", 0.0)) < 1.0 and str(row["fixture_id"]) in campaign.case_by_id
    ]
    passes = [
        campaign.case_by_id[str(row["fixture_id"])]
        for row in first_rows
        if float(row.get("score", 0.0)) >= 1.0 and str(row["fixture_id"]) in campaign.case_by_id
    ]
    replicate = _balanced_cases(failures, min(40, len(failures))) + _balanced_cases(passes, min(40, len(passes)))
    for seed in [int(v) for v in campaign.cfg["seeds"][1:]]:
        for case in replicate:
            if not campaign.can_start(deadline):
                break
            campaign.control(case, deadline, seed=seed, force=True)

    campaign.positive_work(
        "baseline_capability_map",
        start,
        "balanced full-family/difficulty capability map + targeted instability replication",
    )
    rows = [
        row for row in campaign.rows[start:]
        if row.get("intervention_id") == "CONTROL"
    ]
    by_fixture: dict[str, list[float]] = defaultdict(list)
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_fixture[str(row["fixture_id"])].append(float(row["score"]))
        by_family[str(row["family_id"])].append(float(row["score"]))
    return {
        "fixtures": {
            key: {
                "n": len(values),
                "pass_rate": sum(values) / len(values),
                "stable": len(set(values)) == 1,
            }
            for key, values in by_fixture.items()
        },
        "families": {
            key: {
                "n": len(values),
                "pass_rate": sum(values) / len(values),
            }
            for key, values in by_family.items()
        },
        "capability_family_count": len(by_family),
    }



def phase_reasoning_compute(campaign: Test12Campaign, deadline: float) -> dict[str, Any]:
    start = len(campaign.rows)
    allowed = {"REASONING_MODE","GENERATION_BUDGET","CONTEXT_WINDOW","COMPUTE_COST_ROUTING"}
    interventions = [row for row in campaign.interventions if row["category"] in allowed]
    rows = _matrix(campaign, deadline, phase="reasoning_compute_surface", interventions=interventions, cases=_coverage_cases(campaign), coverage_rounds=2)
    campaign.positive_work("reasoning_compute_surface", start, "thinking effort x generation budget x context window with matched controls")
    return _group_summary(rows, campaign.cfg, lambda row: str(row["intervention_id"]))


def phase_controller_screen(campaign: Test12Campaign, deadline: float) -> dict[str, Any]:
    start = len(campaign.rows)
    excluded = {"REASONING_MODE","GENERATION_BUDGET","CONTEXT_WINDOW","COMPUTE_COST_ROUTING"}
    interventions = [row for row in campaign.interventions if row["category"] not in excluded]
    fail, passed = _source_headroom(campaign)
    failures = _balanced_cases(fail, min(64, len(fail)))
    sentinels = _balanced_cases(passed, min(64, len(passed)))

    # Non-negotiable breadth pass: every declared candidate receives coverage,
    # distributed across capability families rather than on the same fixture.
    rows = _breadth_cover(
        campaign,
        deadline,
        phase="mechanism_coverage_floor",
        interventions=interventions,
        failure_cases=failures,
        sentinel_cases=sentinels,
        seed=int(campaign.cfg["seeds"][0]),
    )

    # Replication depth is adaptive only after the complete breadth catalog has
    # had its first look.
    depth_ids = {
        str(row["id"])
        for row in CORE_INTERVENTIONS
        if row["category"] not in excluded
    }
    depth = [row for row in interventions if str(row["id"]) in depth_ids]
    if campaign.can_start(deadline):
        rows.extend(
            _matrix(
                campaign,
                deadline,
                phase="mechanism_coverage_floor",
                interventions=depth,
                cases=_coverage_cases(campaign),
                seeds=[int(v) for v in campaign.cfg["seeds"][:2]],
                coverage_rounds=1,
            )
        )
    campaign.positive_work(
        "mechanism_coverage_floor",
        start,
        "EVERY declared prompt/injection/retry/controller candidate receives rotating failure + sentinel coverage before replication is adaptive",
    )
    return _group_summary(rows, campaign.cfg, lambda row: str(row["intervention_id"]))



def phase_category_focus(campaign: Test12Campaign, deadline: float, *, phase: str, categories: set[str], target_cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    start = len(campaign.rows)
    interventions = [row for row in campaign.interventions if row["category"] in categories]
    cases = target_cases or _coverage_cases(campaign)
    rows = _matrix(campaign, deadline, phase=phase, interventions=interventions, cases=cases)
    campaign.positive_work(phase, start, "targeted category stress + non-target sentinels")
    return _group_summary(rows, campaign.cfg, lambda row: str(row["intervention_id"]))


def _retry_cases(campaign: Test12Campaign) -> list[dict[str, Any]]:
    fixtures = []
    ownership = (campaign.source.get("residual_ownership") or {}).get("fixtures", {}) or {}
    for fixture_id, record in ownership.items():
        if (
            fixture_id in campaign.case_by_id
            and campaign.partition_name(campaign.case_by_id[fixture_id]) in {"DISCOVERY","VALIDATION"}
            and not record.get("rescued_by_recipe")
            and not record.get("rescued_by_operator")
            and not record.get("rescued_by_larger_budget")
        ):
            fixtures.append(campaign.case_by_id[fixture_id])
    return _balanced_cases(fixtures, min(48, len(fixtures))) if fixtures else _coverage_cases(campaign)


def _tool_cases(campaign: Test12Campaign) -> list[dict[str, Any]]:
    tool = [case for case in campaign.partitions["DISCOVERY"] if "tool" in _family(case).lower()]
    sentinels = [case for case in campaign.partitions["DISCOVERY"] if "tool" not in _family(case).lower()]
    return _balanced_cases(tool, min(24, len(tool))) + _balanced_cases(sentinels, min(12, len(sentinels)))


def build_compositions(promoted: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    usable = [row for row in promoted if row.get("mode") != "source_recipe"]
    result = []
    for a, b in itertools.permutations(usable[:8], 2):
        if a.get("category") == b.get("category"):
            continue
        a_text = str(a.get("instruction") or a.get("final_instruction") or a.get("aux_instruction") or a.get("label"))
        b_text = str(b.get("instruction") or b.get("final_instruction") or b.get("aux_instruction") or b.get("label"))
        result.append({
            "id": f"COMP-{a['id']}-{b['id']}-A",
            "category": "COMPOSITION_LAYERING",
            "mode": "aba",
            "label": f"{a['id']}+{b['id']}+{a['id']}",
            "aux_instruction": "A: " + a_text + " Build only the working state needed for the task.",
            "middle_instruction": "B: " + b_text + " Produce a candidate answer.",
            "final_instruction": "A again: " + a_text + " Verify the candidate and return only the final answer.",
            "parents": [a["id"], b["id"], a["id"]],
        })
        if len(result) >= int(limit):
            break
    return result


def phase_real_tool_execution(campaign: Test12Campaign, deadline: float) -> dict[str, Any]:
    start_count = len(campaign.rows)
    results: list[dict[str, Any]] = []
    for policy in TOOL_HARNESS_POLICIES:
        for case in TOOL_MICROCASES:
            if not campaign.can_start(deadline):
                break
            state: dict[str, str] = {}
            transcript: list[dict[str, Any]] = []
            messages = [
                {"role":"system","content":tool_system_prompt(policy)},
                {"role":"user","content":str(case["prompt"])},
            ]
            success = False
            final_value = None
            parse_failures = 0
            tool_errors = 0
            for step in range(1, int(case["max_steps"]) + 1):
                if not campaign.can_start(deadline):
                    break
                campaign.sequence += 1
                options = campaign.runner._generation_options(
                    {"id":case["id"],"prompt":case["prompt"]},
                    {"num_predict":256,"temperature":0.0,"seed":42+step},
                )
                label=f"test1.2 real-tool {case['id']} {policy['id']} step{step}"
                campaign._progress(label, True)
                try:
                    generation, invocation, refs = campaign.runner._invoke_generation(
                        stage="test1.2-real-tool",
                        case_id=f"{case['id']}-{policy['id']}-{step}",
                        messages=messages,
                        options=options,
                        request_fields={"think":False},
                    )
                finally:
                    campaign._progress(label, False)
                text=str((generation.get("normalized") or {}).get("text") or "")
                action=parse_action(text)
                event={
                    "step":step,
                    "model_text":text,
                    "action":action,
                    "metrics":copy.deepcopy(generation.get("metrics") or {}),
                    "timing":copy.deepcopy(generation.get("timing") or {}),
                    "evidence_refs":refs,
                }
                if action is None:
                    parse_failures += 1
                    event["tool_result"]={"ok":False,"error":"UNPARSEABLE_ACTION"}
                    transcript.append(event)
                    messages.extend([
                        {"role":"assistant","content":text},
                        {"role":"user","content":'ERROR: action must be JSON tool call or {"final":"value"}. Retry minimally.'},
                    ])
                    continue
                if "final" in action:
                    final_value=action.get("final")
                    success=score_final(str(case["expected_final"]),final_value)
                    event["final_correct"]=success
                    transcript.append(event)
                    break
                tool_name=str(action.get("tool") or "")
                arguments=action.get("arguments")
                if not isinstance(arguments,dict):
                    result={"ok":False,"error":"ARGUMENTS_MUST_BE_OBJECT"}
                else:
                    result=execute_tool(tool_name,arguments,state)
                if not result.get("ok"):
                    tool_errors += 1
                event["tool_result"]=result
                transcript.append(event)
                messages.extend([
                    {"role":"assistant","content":text},
                    {"role":"user","content":"TOOL RESULT:\n"+json.dumps(result,sort_keys=True)+"\nContinue with the next tool call or final JSON object."},
                ])
            result_row={
                "schema_version":1,
                "phase":"real_tool_execution",
                "fixture_id":case["id"],
                "family_id":"real_tool_execution",
                "partition":"SYNTHETIC_TOOL_LAB",
                "intervention_id":policy["id"],
                "intervention_category":"REAL_TOOL_EXECUTION",
                "intervention_mode":"tool_loop",
                "score":1.0 if success else 0.0,
                "control_score":0.0,
                "delta":1.0 if success else 0.0,
                "model_calls_per_application":len(transcript),
                "parse_failures":parse_failures,
                "tool_errors":tool_errors,
                "final_value":final_value,
                "expected_final":case["expected_final"],
                "tool_transcript":transcript,
                "cost":{
                    "prompt_tokens_observed":sum(float((row.get("metrics") or {}).get("prompt_eval_count") or 0) for row in transcript),
                    "output_tokens_observed":sum(float((row.get("metrics") or {}).get("eval_count") or 0) for row in transcript),
                    "wall_seconds":sum(float((row.get("timing") or {}).get("client_latency_ns") or 0)/1_000_000_000.0 for row in transcript),
                },
            }
            campaign.rows.append(result_row)
            campaign.runner.store.append_jsonl("test1.2-observations.jsonl",result_row)
            results.append(result_row)
    campaign.positive_work(
        "real_tool_execution",
        start_count,
        "real deterministic in-process tool execution across selection, arguments, sequencing, state handoff, and error recovery",
    )
    grouped: dict[str, Any] = {}
    for policy in TOOL_HARNESS_POLICIES:
        rows=[row for row in results if row.get("intervention_id")==policy["id"]]
        grouped[policy["id"]]={
            "n":len(rows),
            "successes":sum(1 for row in rows if float(row.get("score",0.0))>=1.0),
            "success_rate":sum(float(row.get("score",0.0)) for row in rows)/len(rows) if rows else 0.0,
            "tool_errors":sum(int(row.get("tool_errors",0)) for row in rows),
            "parse_failures":sum(int(row.get("parse_failures",0)) for row in rows),
            "mean_model_calls":mean([float(row.get("model_calls_per_application",0)) for row in rows]) if rows else 0.0,
        }
    return grouped


def phase_composition(campaign: Test12Campaign, deadline: float, combined_summary: dict[str, Any]) -> dict[str, Any]:
    start = len(campaign.rows)
    promoted = _rank_mechanisms(combined_summary, campaign, 12)
    compositions = build_compositions(promoted, int(campaign.cfg["max_composition_arms"]))
    for intervention in compositions:
        ident = str(intervention["id"])
        if ident not in campaign.intervention_by_id:
            campaign.interventions.append(copy.deepcopy(intervention))
            campaign.intervention_by_id[ident] = copy.deepcopy(intervention)
    fail, passed = _source_headroom(campaign)
    rows = _breadth_cover(
        campaign,
        deadline,
        phase="composition_interaction",
        interventions=compositions,
        failure_cases=_balanced_cases(fail, min(16, len(fail))),
        sentinel_cases=_balanced_cases(passed, min(16, len(passed))),
        seed=int(campaign.cfg["seeds"][0]),
    )
    if campaign.can_start(deadline) and compositions:
        ranked = _group_summary(rows, campaign.cfg, lambda row: str(row["intervention_id"]))
        survivors = _rank_mechanisms(ranked, campaign, min(8, len(compositions)))
        if survivors:
            rows.extend(
                _matrix(
                    campaign,
                    deadline,
                    phase="composition_interaction",
                    interventions=survivors,
                    cases=_coverage_cases(campaign),
                    seeds=[int(campaign.cfg["seeds"][1])],
                    coverage_rounds=1,
                )
            )
    campaign.positive_work(
        "composition_interaction",
        start,
        "all generated composition candidates receive first coverage; replication depth is adaptive",
    )
    return _group_summary(rows, campaign.cfg, lambda row: str(row["intervention_id"]))



def phase_routing(campaign: Test12Campaign, deadline: float) -> dict[str, Any]:
    start = len(campaign.rows)
    ids = {"SELF-ROUTER","RISK-GATED-VERIFY","STOP-WHEN-SUFFICIENT"}
    interventions = [row for row in campaign.interventions if row["id"] in ids]
    cases = _balanced_cases(campaign.partitions["DISCOVERY"], min(64, len(campaign.partitions["DISCOVERY"])))
    rows = _matrix(campaign, deadline, phase="adaptive_routing", interventions=interventions, cases=cases)
    campaign.positive_work("adaptive_routing", start, "deployment-valid routing using only visible prompt/model-generated state; no oracle family/difficulty routing")
    return _group_summary(rows, campaign.cfg, lambda row: str(row["intervention_id"]))


def phase_generalization(campaign: Test12Campaign, deadline: float, summary: dict[str, Any]) -> dict[str, Any]:
    start = len(campaign.rows)
    interventions = _rank_mechanisms(summary, campaign, int(campaign.cfg["confirmation_mechanisms"]))
    cases = _balanced_cases(campaign.partitions["DISCOVERY"], min(96, len(campaign.partitions["DISCOVERY"])))
    rows = _matrix(campaign, deadline, phase="generalization_confirmation", interventions=interventions, cases=cases)
    campaign.positive_work("generalization_confirmation", start, "top mechanisms x validation families x seeds")
    return _group_summary(rows, campaign.cfg, lambda row: f"{row['intervention_id']}|family={row['family_id']}")


def _merge_summaries(*maps: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for mapping in maps:
        for key, value in mapping.items():
            current = result.get(key)
            if current is None or float(value.get("n", 0)) > float(current.get("n", 0)):
                result[key] = copy.deepcopy(value)
    return result


def phase_negative_transfer(
    campaign: Test12Campaign,
    deadline: float,
    summary: dict[str, Any],
) -> dict[str, Any]:
    start = len(campaign.rows)
    candidates = _rank_mechanisms(summary, campaign, 32)
    _, passed = _source_headroom(campaign)
    sentinels = _balanced_cases(passed, min(64, len(passed)))
    if not candidates or not sentinels:
        return {}
    rows = _matrix(
        campaign,
        deadline,
        phase="negative_transfer_sentinels",
        interventions=candidates,
        cases=sentinels,
        seeds=[int(campaign.cfg["seeds"][1]), int(campaign.cfg["seeds"][2])],
        coverage_rounds=1,
    )
    campaign.positive_work(
        "negative_transfer_sentinels",
        start,
        "highest-value controls challenged on balanced baseline-pass sentinels across families",
    )
    return _group_summary(rows, campaign.cfg, lambda row: str(row["intervention_id"]))


def phase_reserve(campaign: Test12Campaign, deadline: float, summary: dict[str, Any]) -> dict[str, Any]:
    start = len(campaign.rows)
    uncertain = [
        campaign.intervention_by_id[key]
        for key, value in summary.items()
        if key in campaign.intervention_by_id and value.get("classification") in {"UNCERTAIN","TRUNCATION_SENSITIVE"}
    ]
    best = _rank_mechanisms(summary, campaign, 12)
    interventions = list({row["id"]: row for row in [*uncertain, *best]}.values()) or campaign.interventions[:12]
    cases = _coverage_cases(campaign, "DISCOVERY")
    rows = _matrix(campaign, deadline, phase="uncertainty_reserve", interventions=interventions, cases=cases)
    campaign.positive_work("uncertainty_reserve", start, "uncertainty reduction and replication only")
    return _group_summary(rows, campaign.cfg, lambda row: str(row["intervention_id"]))


def _coverage_ledger(campaign: Test12Campaign) -> dict[str, Any]:
    by_mechanism: dict[str, dict[str, Any]] = {}
    for intervention in campaign.interventions:
        ident = str(intervention["id"])
        rows = [row for row in campaign.rows if row.get("intervention_id") == ident]
        by_mechanism[ident] = {
            "category": intervention["category"],
            "observations": len(rows),
            "failure_trials": sum(1 for row in rows if float(row.get("control_score",0.0)) < 1.0),
            "sentinel_trials": sum(1 for row in rows if float(row.get("control_score",0.0)) >= 1.0),
            "families": sorted({str(row.get("family_id")) for row in rows}),
            "phases": sorted({str(row.get("phase")) for row in rows}),
        }
    categories = {
        category: {
            "mechanisms": sum(1 for row in campaign.interventions if row["category"] == category),
            "observations": sum(1 for row in campaign.rows if row.get("intervention_category") == category),
        }
        for category in sorted({row["category"] for row in campaign.interventions})
    }
    return {"schema_version":1,"mechanisms":by_mechanism,"categories":categories}


def _activation_and_negative_transfer(campaign: Test12Campaign) -> tuple[dict[str, Any], dict[str, Any]]:
    effects = _group_summary(
        campaign.rows,
        campaign.cfg,
        lambda row: f"{row['intervention_id']}|family={row['family_id']}",
    )
    negative = {key:value for key,value in effects.items() if value.get("classification") == "CAPABILITY_HARM" or int(value.get("capability_regressions",0)) > 0}
    return effects, negative


def _cost_value_frontier(campaign: Test12Campaign) -> dict[str, Any]:
    summaries = _group_summary(campaign.rows, campaign.cfg, lambda row: str(row["intervention_id"]))
    rows = []
    for key, value in summaries.items():
        iv = campaign.intervention_by_id.get(key, {})
        rows.append({"intervention_id":key,"category":iv.get("category"),**copy.deepcopy(value)})
    rows.sort(key=lambda row:(float(row.get("net_value",0.0)),float(row.get("value_per_call",0.0))),reverse=True)
    return {
        "schema_version":1,
        "objective":"maximize measured rescue/generalization while minimizing capability regression, model calls, tokens, latency, and controller complexity",
        "ranked":rows,
        "pareto_candidates":[row for row in rows if float(row.get("raw_value",0.0)) > 0 and row.get("classification") != "CAPABILITY_HARM"][:25],
    }


def _residual_ownership(campaign: Test12Campaign) -> tuple[dict[str, Any], dict[str, Any]]:
    by_fixture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in campaign.rows:
        if row.get("fixture_id"):
            by_fixture[str(row["fixture_id"])].append(row)
    fixtures = {}
    phenotypes: dict[str, list[str]] = defaultdict(list)
    for fixture_id, rows in sorted(by_fixture.items()):
        controls = [row for row in rows if row.get("intervention_id") == "CONTROL"]
        if not controls or not any(float(row.get("score",0.0)) < 1.0 for row in controls):
            continue
        treatment = [row for row in rows if row.get("intervention_id") != "CONTROL"]
        rescued_categories = sorted({
            str(row.get("intervention_category"))
            for row in treatment
            if float(row.get("control_score",0.0)) < 1.0 and float(row.get("score",0.0)) >= 1.0
        })
        failed_classes = [
            str((row.get("classification") or {}).get("result_class") or "UNKNOWN")
            for row in treatment
            if float(row.get("score",0.0)) < 1.0
        ]
        dominant = max(set(failed_classes), key=failed_classes.count) if failed_classes else "UNKNOWN"
        family = str(rows[0].get("family_id") or "unknown")
        phenotype = f"{family}|{dominant}"
        phenotypes[phenotype].append(fixture_id)
        tested_categories = sorted({str(row.get("intervention_category")) for row in treatment})
        fixtures[fixture_id] = {
            "fixture_id":fixture_id,
            "family_id":family,
            "dominant_failure_class":dominant,
            "phenotype_id":phenotype,
            "tested_categories":tested_categories,
            "rescued_categories":rescued_categories,
            "unresolved_after_full_system_search":not bool(rescued_categories),
        }

    phenotype_records = {}
    fine = {}
    required_owner_categories = {"PROMPT_CONTROL","REASONING_MODE","VERIFICATION","RETRY_RECOVERY","CONTEXT_SELECTION_COMPRESSION","TOOL_POLICY"}
    for phenotype, fixture_ids in phenotypes.items():
        records = [fixtures[value] for value in fixture_ids]
        unresolved = [row for row in records if row["unresolved_after_full_system_search"]]
        tested = set().union(*(set(row["tested_categories"]) for row in records)) if records else set()
        recurrent = len(unresolved) >= 3
        owner_tested = len(required_owner_categories & tested) >= 4
        record = {
            "phenotype_id":phenotype,
            "independent_fixture_count":len(fixture_ids),
            "independent_unresolved_fixture_count":len(unresolved),
            "fixture_ids":sorted(fixture_ids),
            "unresolved_fixture_ids":sorted(row["fixture_id"] for row in unresolved),
            "tested_categories":sorted(tested),
            "owner_search_sufficient_for_tuning":owner_tested,
            "owner":"FINE_TUNING_CANDIDATE" if recurrent and owner_tested else ("RECOVERED_BY_SYSTEM" if not unresolved else "MORE_SYSTEM_SEARCH_REQUIRED"),
        }
        phenotype_records[phenotype]=record
        if record["owner"]=="FINE_TUNING_CANDIDATE":
            fine[phenotype]={
                **copy.deepcopy(record),
                "qualification":{
                    "recurrent":True,
                    "independent":True,
                    "prompt_owner_tested":"PROMPT_CONTROL" in tested,
                    "reasoning_compute_owner_tested":"REASONING_MODE" in tested,
                    "verification_retry_owner_tested":bool({"VERIFICATION","RETRY_RECOVERY"} & tested),
                    "context_memory_owner_tested":bool({"CONTEXT_SELECTION_COMPRESSION","MEMORY","STATE_TRACKING"} & tested),
                    "tool_policy_owner_tested":"TOOL_POLICY" in tested,
                    "protected_partitions_excluded":True,
                    "next_action":"TEST3_FINE_TUNING_QUALIFICATION",
                },
            }
    return (
        {"schema_version":1,"fixtures":fixtures,"phenotypes":phenotype_records},
        {"schema_version":1,"candidates":fine},
    )


def _control_grammar_coverage(campaign: Test12Campaign) -> dict[str, Any]:
    generated = generate_prompt_control_candidates()
    tested = {
        str(row.get("intervention_id"))
        for row in campaign.rows
        if row.get("intervention_id") not in {None, "CONTROL"}
    }
    campaign_declared = {str(row["id"]) for row in campaign.interventions}
    tool_declared = {str(row["id"]) for row in TOOL_HARNESS_POLICIES}
    all_declared = campaign_declared | tool_declared
    prompt_declared = {str(row["id"]) for row in generated}
    by_dimension: dict[str, set[str]] = {
        "primitive": set(),
        "placement": set(),
        "representation": set(),
        "dose": set(),
        "recurrence": set(),
    }
    for row in generated:
        if str(row["id"]) not in tested:
            continue
        by_dimension["primitive"].add(str(row.get("primitive_id")))
        by_dimension["placement"].add(str(row.get("placement")))
        by_dimension["representation"].add(str(row.get("representation")))
        by_dimension["dose"].add(str(row.get("dose")))
        by_dimension["recurrence"].add(str(row.get("recurrence")))
    return {
        "schema_version": 1,
        "declared_candidate_count": len(all_declared),
        "tested_candidate_count": len(all_declared & tested),
        "untested_candidate_ids": sorted(all_declared - tested),
        "all_declared_candidates_tested": not bool(all_declared - tested),
        "prompt_grammar_declared_count": len(prompt_declared),
        "prompt_grammar_tested_count": len(prompt_declared & tested),
        "tool_harness_policy_ids": sorted(tool_declared),
        "tool_harness_policies_tested": sorted(tool_declared & tested),
        "dimension_levels_tested": {key: sorted(value) for key, value in by_dimension.items()},
        "grammar": copy.deepcopy(CONTROL_GRAMMAR),
    }



def _tuning_corpus_rows(campaign: Test12Campaign) -> list[dict[str, Any]]:
    rows = []
    for row in campaign.rows:
        if row.get("intervention_id") in {None, "CONTROL"}:
            continue
        delta = float(row.get("delta") or 0.0)
        if delta > 0:
            label = "POSITIVE_CONTROL"
        elif delta < 0:
            label = "NEGATIVE_CONTROL"
        else:
            label = "NULL_CONTROL"
        rows.append({
            "schema_version": 1,
            "fixture_id": row.get("fixture_id"),
            "family_id": row.get("family_id"),
            "difficulty_level": row.get("difficulty_level"),
            "intervention_id": row.get("intervention_id"),
            "intervention_category": row.get("intervention_category"),
            "intervention_mode": row.get("intervention_mode"),
            "router_choice": row.get("router_choice"),
            "risk_gate": row.get("risk_gate"),
            "task_text": row.get("task_text"),
            "control_response_text": row.get("control_response_text"),
            "treatment_response_text": row.get("treatment_response_text"),
            "control_score": row.get("control_score"),
            "treatment_score": row.get("score"),
            "delta": delta,
            "label": label,
            "model_calls_per_application": row.get("model_calls_per_application"),
            "cost": copy.deepcopy(row.get("cost") or {}),
            "classification": copy.deepcopy(row.get("classification") or {}),
            "activation_context": {
                "family_id": row.get("family_id"),
                "difficulty_level": row.get("difficulty_level"),
                "partition": row.get("partition"),
            },
        })
    return rows


def _harness_policy_blueprint(campaign: Test12Campaign) -> dict[str, Any]:
    frontier = _cost_value_frontier(campaign)
    activation, negative = _activation_and_negative_transfer(campaign)
    promoted = [
        row for row in frontier.get("ranked", [])
        if row.get("classification") in {"STRONG_CONDITIONAL_RESCUE","PROMISING_CONDITIONAL_RESCUE"}
        and row.get("classification") != "CAPABILITY_HARM"
    ]
    return {
        "schema_version": 1,
        "model": getattr(campaign.runner, "model", None),
        "policy_type": "MODEL_SPECIFIC_ADAPTIVE_HARNESS_BLUEPRINT",
        "global_candidates": promoted[:16],
        "activation_boundaries": activation,
        "negative_transfer_boundaries": negative,
        "default_rule": "DIRECT_UNLESS_A_VALIDATED_CONDITIONAL_CONTROL_HAS_POSITIVE_EXPECTED_VALUE",
        "routing_inputs_allowed": [
            "visible task text",
            "model-generated risk/router state",
            "observable failure/retry state",
        ],
        "oracle_routing_prohibited": True,
        "tuning_run_objectives": [
            "learn control selection",
            "learn activation boundaries",
            "reduce unnecessary controller calls",
            "preserve baseline-pass sentinels",
            "repair recurrent model-owned failures if tuning is qualified",
        ],
    }


def write_outputs(campaign: Test12Campaign, results: dict[str, Any]) -> None:
    store = campaign.runner.store
    assert store is not None
    combined = _merge_summaries(
        results.get("reasoning",{}),
        results.get("controllers",{}),
        results.get("context",{}),
        results.get("tool",{}),
        results.get("retry",{}),
        results.get("routing",{}),
        results.get("reserve",{}),
    )
    activation, negative = _activation_and_negative_transfer(campaign)
    cost_value = _cost_value_frontier(campaign)
    residual, fine = _residual_ownership(campaign)
    queue = [
        {
            "intervention_id":row["intervention_id"],
            "category":row.get("category"),
            "classification":row.get("classification"),
            "net_value":row.get("net_value"),
            "value_per_call":row.get("value_per_call"),
            "next_action":"GENERALIZATION_OR_CAUSAL_REPLICATION",
        }
        for row in cost_value["ranked"][:32]
        if row.get("classification") in {"STRONG_CONDITIONAL_RESCUE","PROMISING_CONDITIONAL_RESCUE","UNCERTAIN"}
    ]
    unknowns = [
        {"intervention_id":key,"state":"UNKNOWN","next_test":"replicate on more failure/sentinel families","evidence":value}
        for key,value in combined.items()
        if value.get("classification") in {"UNCERTAIN","TRUNCATION_SENSITIVE"}
    ]
    scope = build_test12_plan(campaign.cases, seed_run=(campaign.source.get("run_id") if campaign.source.get("run_id") not in {None, "FRESH-MODEL"} else None))["scope_boundaries"]
    assurance = {
        "schema_version":1,
        "rules":list(RULES),
        "rules_hash":hashlib.sha256("\n".join(RULES).encode("utf-8")).hexdigest(),
        "improvement_surface":list(IMPROVEMENT_SURFACE),
        "phases":copy.deepcopy(campaign.phase_assertions),
        "protected_partitions":{"TEST2_BLIND":"NOT_LOOKED_AT","TEST3_PROTECTED":"NOT_LOOKED_AT"},
        "not_looked_at":["TEST2_BLIND","TEST3_PROTECTED","REAL_EXTERNAL_TOOL_EXECUTION","CROSS_VENDOR_HARNESS","MODEL_WEIGHT_UPDATE"],
    }
    grammar_coverage = _control_grammar_coverage(campaign)
    tuning_rows = _tuning_corpus_rows(campaign)
    harness_blueprint = _harness_policy_blueprint(campaign)
    store.write_json("full-control-candidate-registry.json", {
        "schema_version":1,
        "candidates":campaign.interventions,
        "tool_harness_policies":list(TOOL_HARNESS_POLICIES),
        "control_grammar":copy.deepcopy(CONTROL_GRAMMAR),
    }, producer="test1.2", stage="report")
    store.write_json("control-grammar-coverage.json", grammar_coverage, producer="test1.2", stage="report")
    for tuning_row in tuning_rows:
        store.append_jsonl("tuning-example-corpus.jsonl", tuning_row)
    store.write_json("harness-policy-blueprint.json", harness_blueprint, producer="test1.2", stage="report")
    store.write_json("test1.2-source-audit.json", {
        "schema_version":1,
        "source_run":campaign.source.get("run_id"),
        "source_integrity":campaign.source.get("source_integrity"),
        "source_observations":len(campaign.source.get("observations",[])),
        "source_priority_items":len((campaign.source.get("priority_queue") or {}).get("queue",[]) or []),
        "source_ingredients":len((campaign.source.get("ingredient_registry") or {}).get("ingredients",[]) or []),
    }, producer="test1.2", stage="report")
    store.write_json("mechanism-registry.json", {"schema_version":1,"mechanisms":campaign.interventions,"surface":list(IMPROVEMENT_SURFACE)}, producer="test1.2", stage="report")
    store.write_json("mechanism-coverage-ledger.json", _coverage_ledger(campaign), producer="test1.2", stage="report")
    store.write_json("reasoning-compute-map.json", {"schema_version":1,"effects":results.get("reasoning",{})}, producer="test1.2", stage="report")
    store.write_json("controller-mechanism-map.json", {"schema_version":1,"effects":results.get("controllers",{})}, producer="test1.2", stage="report")
    store.write_json("context-memory-state-map.json", {"schema_version":1,"effects":results.get("context",{})}, producer="test1.2", stage="report")
    store.write_json("tool-action-verification-map.json", {"schema_version":1,"effects":results.get("tool",{})}, producer="test1.2", stage="report")
    store.write_json("real-tool-execution-map.json", {"schema_version":1,"effects":results.get("real_tool",{})}, producer="test1.2", stage="report")
    store.write_json("retry-replay-recovery-map.json", {"schema_version":1,"effects":results.get("retry",{})}, producer="test1.2", stage="report")
    store.write_json("composition-interaction-atlas.json", {"schema_version":1,"effects":results.get("composition",{})}, producer="test1.2", stage="report")
    store.write_json("adaptive-routing-map.json", {"schema_version":1,"effects":results.get("routing",{})}, producer="test1.2", stage="report")
    store.write_json("family-generalization-map-1.2.json", {"schema_version":1,"effects":results.get("generalization",{})}, producer="test1.2", stage="report")
    store.write_json("activation-boundary-map.json", {"schema_version":1,"effects":activation}, producer="test1.2", stage="report")
    store.write_json("negative-transfer-map-1.2.json", {"schema_version":1,"effects":negative}, producer="test1.2", stage="report")
    store.write_json("cost-value-frontier-1.2.json", cost_value, producer="test1.2", stage="report")
    store.write_json("residual-failure-ownership-1.2.json", residual, producer="test1.2", stage="report")
    store.write_json("fine-tuning-readiness-map-1.2.json", fine, producer="test1.2", stage="report")
    store.write_json("test1.2-priority-queue.json", {"schema_version":1,"queue":queue}, producer="test1.2", stage="report")
    store.write_json("test1.2-uncertainty-ledger.json", {"schema_version":1,"unknowns":unknowns}, producer="test1.2", stage="report")
    store.write_json("test1.2-handoff.json", {
        "schema_version":1,
        "source_seed_run":campaign.source.get("run_id"),
        "priority_queue":queue,
        "fine_tuning_candidates":list((fine.get("candidates") or {}).keys()),
        "negative_transfer_keys":sorted(negative),
        "improvement_surface":list(IMPROVEMENT_SURFACE),
        "protected_partitions_exposed":False,
        "next_test":"TEST1.2_TUNING_RUN",
        "full_control_candidate_coverage":grammar_coverage,
        "tuning_example_count":len(tuning_rows),
        "harness_policy_blueprint":"harness-policy-blueprint.json",
    }, producer="test1.2", stage="report")
    store.write_json("scope-boundaries.json", {"schema_version":1,**scope}, producer="test1.2", stage="report")
    store.write_json("assurance-map-1.2.json", assurance, producer="test1.2", stage="report")


def run_test12_campaign(
    runner: Any,
    cases: list[dict[str, Any]],
    *,
    seed_run: str | None = None,
    clock: Callable[[], float] = time.monotonic,
    started_monotonic: float | None = None,
) -> list[dict[str, Any]]:
    assert runner.store is not None
    source = (
        load_test11_source(Path(runner.results_root), seed_run, cases)
        if seed_run
        else fresh_model_source(cases)
    )
    campaign = Test12Campaign(runner, cases, source, clock=clock, started_monotonic=started_monotonic)
    plan = build_test12_plan(cases, seed_run=seed_run)
    validate_test12_plan(plan)
    if not (runner.store.run_dir / "test1.2-plan.json").is_file():
        runner.store.write_json("test1.2-plan.json", plan, producer="test1.2", stage="preflight")

    results: dict[str, Any] = {}
    phase_start = campaign.clock()
    for phase_name, seconds in PHASES:
        deadline = min(campaign.active_end, phase_start + seconds)
        before = len(campaign.rows)
        if phase_name == "baseline_capability_map":
            results["baseline"] = phase_baseline(campaign, deadline)
        elif phase_name == "fractional_compute_surface":
            results["reasoning"] = phase_reasoning_compute(campaign, deadline)
        elif phase_name == "mechanism_coverage_floor":
            results["controllers"] = phase_controller_screen(campaign, deadline)
        elif phase_name == "real_tool_execution":
            results["real_tool"] = phase_real_tool_execution(campaign, deadline)
        elif phase_name == "failure_phenotype_replay":
            results["retry"] = phase_category_focus(
                campaign, deadline, phase=phase_name,
                categories={"RETRY_RECOVERY","CRITIQUE","DELEGATION","TOOL_POLICY","VERIFICATION"},
                target_cases=_retry_cases(campaign),
            )
        elif phase_name == "interaction_scout":
            combined = _merge_summaries(
                results.get("reasoning",{}),
                results.get("controllers",{}),
                results.get("retry",{}),
            )
            results["composition"] = phase_composition(campaign, deadline, combined)
        elif phase_name == "dose_activation_boundaries":
            results["context"] = phase_category_focus(
                campaign, deadline, phase=phase_name,
                categories={
                    "REASONING_MODE","GENERATION_BUDGET","CONTEXT_WINDOW",
                    "STATE_TRACKING","MEMORY","CONTEXT_SELECTION_COMPRESSION",
                    "STOP_ESCALATE_POLICY","ADAPTIVE_ROUTING",
                },
            )
        elif phase_name == "negative_transfer_sentinels":
            combined = _merge_summaries(
                results.get("reasoning",{}),
                results.get("controllers",{}),
                results.get("context",{}),
                results.get("retry",{}),
                results.get("composition",{}),
            )
            results["tool"] = phase_negative_transfer(campaign, deadline, combined)
            results["routing"] = phase_routing(campaign, deadline)
        elif phase_name == "information_gain_reserve":
            combined = _merge_summaries(
                results.get("reasoning",{}),
                results.get("controllers",{}),
                results.get("context",{}),
                results.get("tool",{}),
                results.get("retry",{}),
                results.get("routing",{}),
            )
            results["reserve"] = phase_reserve(campaign, deadline, combined)

        ended = campaign.clock()
        runner.store.append_jsonl("test1.2-phase-events.jsonl", {
            "phase":phase_name,
            "started_monotonic":phase_start,
            "ended_monotonic":ended,
            "deadline_monotonic":deadline,
            "observations_added":len(campaign.rows)-before,
            "total_observations":len(campaign.rows),
        })
        phase_start = deadline
        if ended >= campaign.active_end:
            break

    if campaign.can_start(campaign.active_end):
        combined = _merge_summaries(
            results.get("reasoning",{}),
            results.get("controllers",{}),
            results.get("context",{}),
            results.get("tool",{}),
            results.get("retry",{}),
            results.get("routing",{}),
            results.get("reserve",{}),
        )
        extra = phase_reserve(campaign, campaign.active_end, combined)
        results["reserve"] = _merge_summaries(results.get("reserve",{}), extra)

    write_outputs(campaign, results)
    return campaign.rows
