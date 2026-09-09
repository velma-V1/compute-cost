# GPT-OSS 20B Capability Map Design

**Status:** Approved design captured for implementation
**Version:** capability-map-v1
**Date:** 2026-09-08
**Base branch:** `feat/adaptive-qwen-characterization`

## Objective

Extend the existing adaptive characterization harness into a versioned GPT-OSS 20B capability-mapping campaign without changing or invalidating the existing `qwen-characterization-v1` fixture format or historical results.

The campaign must map what GPT-OSS 20B can do, where reliability breaks, how thinking/reasoning budget changes the boundary, what failure mechanism occurred, what minimum intervention repairs it, how robust the repaired behavior is, and what each reliable capability costs on the measured machine.

The resulting data is intended to become an empirical operating policy for Inverted, not a single benchmark score.

## Compatibility Rule

`benchmarks/qwen-characterization-v1.json` is immutable for this work. It remains a valid legacy suite and continues to execute through the existing characterization path.

The GPT-OSS campaign copies that suite's core fixture skeleton:

- stable case ID
- capability/category identifier
- difficulty
- prompt
- deterministic scorer
- expected result/evidence
- timeout

and extends it with versioned capability-map metadata. The old fields remain accepted. New fields are additive.

Historical run schemas are not rewritten. New capability-map outputs use a new schema version and coexist with existing `characterization-summary.json`/report outputs.

## Scope Boundary

"Everything" means exhaustive coverage of the declared, versioned capability taxonomy plus newly discovered failure modes and compound interactions. It does not mean every possible language-model task.

Every family and discovered mechanism must have an explicit coverage state: `PROVEN`, `PARTIAL`, `UNCERTAIN`, or `UNTESTED`.

## Capability Taxonomy v1

The first taxonomy contains these 40 capability families:

1. instruction_following_constraint_stacking
2. strict_structured_output
3. extraction_transformation
4. arithmetic_numerical_reasoning
5. algebra_quantitative_reasoning
6. formal_logic_deduction
7. causal_counterfactual_reasoning
8. temporal_reasoning
9. spatial_reasoning
10. planning_optimization
11. coding_generation
12. code_comprehension
13. debugging_root_cause_diagnosis
14. refactoring_under_constraints
15. test_generation_verification
16. tool_selection
17. tool_argument_correctness
18. multi_tool_sequencing
19. tool_error_recovery
20. ambiguity_detection
21. missing_information_handling
22. uncertainty_calibration
23. hallucination_resistance
24. context_retrieval
25. context_reasoning
26. lost_in_middle_resistance
27. distractor_noise_resistance
28. contradictory_information_handling
29. multi_turn_state_tracking
30. updated_obsolete_state_rejection
31. memory_compression_summary_fidelity
32. decomposition
33. self_correction
34. verification_critique
35. meta_reasoning
36. prompt_instruction_conflict_handling
37. format_robustness
38. adversarial_wording_robustness
39. sibling_transfer_generalization
40. composite_agent_tasks

The taxonomy is stored separately from individual fixtures so future families can be added under a new taxonomy version without changing historical family definitions.

## Difficulty Model

Each capability family has a family-local L0-L10 rubric:

- L0 sanity / trivial
- L1 basic
- L2 routine
- L3 moderate
- L4 competent
- L5 difficult
- L6 advanced
- L7 expert
- L8 adversarial
- L9 pathological / edge case
- L10 boundary-breaking

Difficulty is not globally interchangeable between families. `L7` means level 7 under that family's rubric, not equal absolute difficulty across all families.

Every fixture records:

- `difficulty.level`
- `difficulty.rubric_version`
- optional structured `difficulty.dimensions`

The dimensions expose what made the task harder, such as reasoning steps, distractor count, constraint count, context length, ambiguity, tool count, or state transitions.

## Extended Fixture Contract

A capability-map fixture extends the existing template with the following normalized shape:

```json
{
  "id": "arith-L6-001",
  "category": "arithmetic_numerical_reasoning",
  "family_id": "arithmetic_numerical_reasoning",
  "taxonomy_version": "capability-taxonomy-v1",
  "difficulty_level": 6,
  "difficulty": {
    "level": 6,
    "rubric_version": "arithmetic-v1",
    "dimensions": {
      "reasoning_steps": 7,
      "distractors": 2,
      "constraint_count": 3
    }
  },
  "prompt": "...",
  "scorer": "numeric",
  "scorer_version": "1",
  "expected": 147.25,
  "tolerance": 0,
  "timeout_s": 120,
  "capabilities_required": ["arithmetic_numerical_reasoning"],
  "recovery_eligible": true,
  "robustness_eligible": true,
  "compound": false,
  "tags": []
}
```

Compatibility rules:

1. `category` remains populated because existing runner/scoring code already consumes it.
2. `difficulty_level` remains populated because existing experiment metadata consumes it.
3. `family_id` and structured `difficulty` are the new canonical fields for capability-map analysis.
4. Existing scalar scorer names remain valid so deterministic scoring can be reused.
5. The suite loader must normalize legacy fixtures into the extended internal representation without modifying source files.

## Experiment Identity and Provenance

Every experiment must be traceable to:

- suite version
- taxonomy version
- fixture ID
- fixture/source hash
- instantiated prompt hash
- scorer/version hash or identifier
- model/runtime configuration
- machine baseline
- seed
- thinking mode
- generation budget
- context request
- prompt variant
- parent experiment
- recovery level
- replay lineage

The existing immutable `ExperimentSpec` lineage mechanism is retained and extended only where a new controlled variable is required.

## Adaptive Difficulty Search

The current generation-budget controller is retained for reasoning-budget boundaries. A separate family difficulty controller finds the capability frontier efficiently.

Per family:

```text
easy anchor PASS
  -> jump upward
PASS
  -> jump upward
FAIL
  -> bisect level bracket
  -> replicate transition
  -> classify reliable / unstable / failure region
```

The controller must not mechanically execute all 11 levels.

Transition-zone results are replicated. The raw pass rate is always stored. Default derived labels are:

- reliable: pass rate >= 0.90
- unstable: 0.40 <= pass rate < 0.90
- failure region: pass rate < 0.40

Thresholds are configuration/versioned policy, not irreversible truth.

## Reasoning Curves

Near the discovered difficulty frontier, characterize:

- THINK OFF frontier
- THINK ON frontier
- minimum reproduced useful generation/reasoning budget
- diminishing-return region
- overthinking/corruption observations when measurable

No unexposed internal reasoning state may be fabricated. Runtime-exposed thinking is behavioral evidence only. Aggregate generation counts remain aggregate unless the runtime exposes a trustworthy split.

## Failure Separation

A failed experiment is first classified by source:

- `MODEL_FAILURE`
- `EVALUATOR_FAILURE`
- `TOOL_FAILURE`
- `INFRA_FAILURE`
- `TIMEOUT`
- `INVALID_FIXTURE`

Only valid model behavior contributes to capability frontiers.

Model failures can then receive mechanism labels including:

- knowledge_gap
- logic_error
- arithmetic_error
- assumption_error
- premature_commit
- instruction_loss
- distractor_capture
- state_loss
- stale_state_use
- hallucinated_fact
- tool_selection_error
- tool_argument_error
- verification_failure
- cascading_error

Mechanism labels must retain the evidence basis and may be `UNKNOWN` when the evidence does not support a confident mechanism.

## Permanent Replay Rule

Every meaningful model failure, frontier transition, anomaly, and successful recovery becomes a replay fixture or replay snapshot. Replays preserve the exact original evidence and lineage.

Replay classes:

```text
replay/
  failures/
  boundaries/
  anomalies/
  recoveries/
```

Harness/evaluator defects remain separated from model replay evidence.

## Recovery Ladder

Recovery tests apply the weakest intervention first:

- R0 exact replay
- R1 reasoning/generation budget
- R2 thinking mode
- R3 minimal clarification
- R4 failure evidence supplied
- R5 decomposition
- R6 structured state
- R7 targeted correction
- R8 Inverted intervention

The campaign records the minimum intervention that changes the outcome and its reproduced reliability/cost. Recovery does not erase the original failure.

## Robustness

Once a frontier or recovery is established, perturb only dimensions relevant to the hypothesis:

- prompt wording
- seed
- distractors/noise
- context placement
- contradictory context
- state history
- format pressure
- sibling task transfer

Robustness experiments preserve controlled-variable lineage. They do not silently alter multiple independent variables at once unless the experiment is explicitly a compound interaction test.

## Compound Tasks and Composition Penalty

Compound fixtures declare multiple `capabilities_required` and set `compound: true`.

The campaign compares the observed compound frontier against the component capability frontiers and records a composition penalty/interference delta where meaningful.

Example:

```text
component frontier estimate: L7
compound observed frontier: L4
composition penalty: -3 levels
```

The raw component and compound observations remain authoritative; the delta is a derived metric.

## Cost Attachment

Every capability observation links to available measured evidence for:

- wall-clock duration
- load time
- prompt tokens
- generated tokens
- prompt throughput
- generation throughput
- observable thinking duration
- time to first answer/chunk
- RAM average/peak
- VRAM average/peak
- CPU utilization
- GPU utilization
- GPU power / integrated Wh
- process disk reads/writes where observable
- context requested/used
- retry/recovery cost

Derived capability economics may include:

- reliable capability / second
- reliable capability / 1K generated tokens
- reliable capability / Wh
- THINK ON vs OFF cost
- cost per frontier level gained
- recovery cost
- avoided-escalation cost when comparison evidence exists

Unavailable measurements stay unavailable rather than estimated from unrelated values.

## Campaign Phases

The execution order is:

A. machine baseline
B. breadth screen using cheap anchor probes across all 40 families
C. vertical difficulty frontier search
D. reasoning curves near the frontier
E. failure autopsy and mechanism classification
F. recovery ladder
G. robustness perturbations
H. compound capability tasks
I. value map
J. Inverted operating policy

Later phases consume earlier evidence and spend calls only when the added experiment can change the capability map, failure map, recovery policy, robustness conclusion, or routing decision.

## Required Outputs

A completed campaign run may retain existing evidence files and additionally produces:

```text
results/<run>/
  experiments.jsonl
  telemetry.jsonl
  capability-map.json
  capability-frontiers.json
  reasoning-curves.json
  weakness-map.json
  failure-atlas.json
  recovery-map.json
  robustness-map.json
  cost-map.json
  value-map.json
  inverted-operating-policy.json
  coverage-ledger.json
  characterization-report.md
  replay/
    failures/
    boundaries/
    anomalies/
    recoveries/
  manifest.json
```

Each derived output records schema version and provenance back to raw experiment/evidence IDs.

## Operating Policy Output

The final policy answers:

```text
task -> required capabilities -> estimated difficulty
     -> known GPT-OSS envelope
     -> cheapest proven reliable raw configuration
     -> minimum proven recovery if needed
     -> Inverted intervention if justified
     -> escalation when local evidence says the envelope is exceeded
```

The policy is evidence-backed. It must distinguish proven rules from partial/uncertain recommendations.

## Initial Implementation Increment

The first implementation increment requested by the user is specifically to **copy the existing characterization test template and extend it**. That increment must:

1. Leave `qwen-characterization-v1.json` unchanged.
2. Add `capability-taxonomy-v1.json` containing the 40 stable family IDs and L0-L10 labels.
3. Add `gpt-oss-20b-capability-v1.json` using the existing fixture skeleton plus the additive capability-map fields.
4. Include at least one deterministic breadth-screen anchor fixture for every one of the 40 families, using existing scorers where possible and explicit scorer requirements where new deterministic scorer support is needed.
5. Add suite normalization/validation that accepts both the legacy template and the extended template.
6. Add tests proving legacy compatibility, taxonomy completeness, difficulty consistency, stable family references, and extended-field normalization.
7. Do not start expensive model execution as part of repository unit tests.

This increment establishes the reusable template and coverage ledger foundation. Adaptive per-family frontier search and later campaign phases build on this contract rather than introducing a second fixture system.

## Acceptance Criteria for the Initial Increment

1. The original Qwen suite file is byte-for-byte unchanged on the feature branch.
2. The new taxonomy contains exactly the approved 40 family IDs and all eleven level labels L0-L10.
3. The GPT-OSS suite contains at least one breadth-screen anchor fixture referencing every taxonomy family.
4. Every new fixture has a valid stable ID, family ID, taxonomy version, difficulty, scorer declaration, timeout, and recovery/robustness eligibility flags.
5. Legacy fixtures normalize successfully without requiring new fields in their source JSON.
6. Extended fixtures normalize into one canonical internal representation.
7. Invalid family IDs, mismatched scalar/structured difficulty, unknown taxonomy versions, and malformed compound capability lists fail validation deterministically.
8. Unit tests require no Ollama server, GPU, model download, or network access.
9. No result data, secrets, machine-specific paths, or personal information are committed.
10. The full existing test suite plus the new validation tests is green before the increment is described as complete.
