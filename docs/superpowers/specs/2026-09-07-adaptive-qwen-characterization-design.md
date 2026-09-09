# Adaptive Qwen Characterization Lab — Design Specification

**Date:** 2026-09-07  
**Status:** Approved architecture; implementation not yet started  
**Repository:** `velma-V1/compute-cost`  
**Primary target:** Qwen-family local models through Ollama, beginning with `qwen3.5:27b-q4_K_M`

## 1. Purpose

`compute-cost` must evolve from a fixed cost/capability benchmark into a closed-loop model-characterization laboratory.

The objective is not merely to produce a leaderboard score. The system must learn, with replayable evidence:

- what a model can and cannot do;
- how much observable reasoning is required for different task classes;
- when more reasoning stops helping or actively hurts;
- where context, prompt structure, quantization, runtime state, and resources change behavior;
- how failures develop;
- whether failures are reproducible;
- the minimum intervention that reliably recovers each failure;
- which model/runtime/prompt/thinking configuration is the cheapest reliable operating point for a given task.

The long-term output is an evidence-backed operating manual that Inverted can use to select model, quant, thinking mode, context strategy, reasoning budget, prompt ingredients, verification policy, and recovery strategy automatically.

## 2. Governing Laws

1. **Every run must teach something new.** A test that does not reduce uncertainty, expose a boundary, reproduce an anomaly, validate a recovery, or establish a baseline is low value.
2. **Failures are research assets.** A failure is incomplete until reproducibility, cause class, recoverability, and cheapest reliable recovery have been investigated or explicitly marked unresolved.
3. **One variable changes at a time.** Multi-variable changes may be used only after the effects of constituent variables are known or when explicitly testing interaction effects.
4. **Raw evidence is authoritative.** Derived conclusions never replace exact prompts, runtime requests, streamed responses, exposed thinking, final answers, telemetry, scorer evidence, configuration, or replay snapshots.
5. **Test defects are not model defects.** Scorer defects, invalid budgets, harness bugs, capture gaps, or runtime misconfiguration must be classified separately from model capability failures.
6. **Thinking is not inherently valuable.** Only reasoning that increases reliable capability, reduces failure, or lowers total system cost is valuable.
7. **Maximum accepted context is not effective context.** Effective context is the largest context region where the required capability remains reliable and operationally worthwhile.
8. **Data collection is cheap; retesting is not.** Preserve enough evidence to support offline analysis, exact replay, and future cross-model comparison without repeating expensive model calls.
9. **A PASS is not automatically reliable.** Near boundaries, reliability must be established with controlled replication.
10. **No fabricated precision.** Measurements must be labeled `MEASURED`, `DERIVED`, `ESTIMATED`, or `UNAVAILABLE`. In particular, exact thinking-token counts must not be claimed unless the runtime or a validated tokenizer method makes them exact.
11. **Boundaries are more valuable than averages.** Compute should concentrate around transitions between reliable pass, unstable behavior, and reliable failure.
12. **Anomalies are first-class results.** Unexpected passes, unexpected failures, cross-quant reversals, overthinking failures, and unusually effective recovery interventions must trigger targeted follow-up.

## 3. Architecture

The system consists of six logical subsystems with explicit boundaries:

1. **Experiment Planner** — chooses the next experiment and the single variable to change.
2. **Execution Harness** — runs one model invocation under a fully specified configuration.
3. **Evidence Recorder** — preserves raw and normalized evidence with hashes and provenance.
4. **Classifier/Scorer** — separates answer correctness, formatting, truncation, runtime faults, harness faults, and behavioral failure signatures.
5. **Adaptive Controller** — updates uncertainty, brackets boundaries, schedules replication, and selects the next highest-value experiment.
6. **Knowledge Builder** — derives model profiles, operating rules, recovery recipes, anomalies, and replay libraries while retaining links to source evidence.

The existing runner/evidence/runtime/telemetry foundation should be extended rather than replaced where practical.

## 4. Experiment Unit

Every experiment must have a stable identity and lineage.

Required fields include:

- `experiment_id`
- `parent_experiment_id` or `null`
- `model`
- `runtime`
- `task_id`
- `task_family`
- `difficulty_level`
- `hypothesis`
- `changed_variable`
- `old_value`
- `new_value`
- `thinking_mode`
- `generation_budget`
- `context_request`
- actual runtime-reported prompt/eval counts where exposed
- `temperature`
- `seed`
- prompt/system/messages exact bytes or canonical serialized request
- hardware/runtime snapshot references
- result classification
- scorer result
- evidence references
- information-gain/result summary

A child experiment must identify exactly how it differs from its parent.

## 5. Baseline and Adaptive Search

For a new task, establish at least two operating baselines when the runtime supports them:

- thinking disabled;
- thinking enabled with enough headroom to avoid immediate truncation.

The controller then adapts according to observed failure mode.

Examples:

- easy pass → reduce budget or increase difficulty;
- reasoning truncation → increase generation budget;
- answer truncation → increase answer headroom;
- persistent semantic failure despite larger budget → stop blind scaling and test a different hypothesis;
- unstable pass/fail → replicate near the boundary;
- test/scorer defect → quarantine the case and do not charge it against model capability;
- resource limit → bracket the resource boundary;
- interesting anomaly → reproduce before generalizing.

Budget values are experimental variables, not assumptions. The fixed case-level limits in the current baseline suite must no longer be interpreted as model capability limits.

## 6. Thought-Budget Curves

For tasks where thinking can affect outcome, the system should characterize the relationship between reasoning allowance and result.

Candidate budgets may include powers or near-powers such as 32, 64, 128, 256, 512, 1024, and 2048, but the controller should not exhaustively run all values when a bracket can be found more cheaply.

Derived operating regions:

- `minimum_viable_budget`
- `transition_zone`
- `stable_budget`
- `optimal_budget`
- `waste_threshold`
- `overthinking_threshold`

Near a FAIL→PASS transition, perform local bracketing and replication rather than jumping directly to very large budgets.

## 7. Observable Thought Anatomy

Exposed reasoning text is a behavioral trace, not ground-truth access to hidden neural computation. The system may analyze it as observable evidence while avoiding claims that it fully describes internal cognition.

For each invocation, preserve and derive when available:

- time to first thinking chunk;
- time to first final-answer chunk;
- thinking duration;
- answer duration;
- total duration;
- streamed thinking chunks and timestamps;
- streamed answer chunks and timestamps;
- `done_reason`;
- runtime-reported prompt/eval counts;
- thinking bytes/chars/chunks;
- final-answer bytes/chars/chunks;
- corrections/revisions/repetition indicators;
- telemetry aligned to the thinking and answer periods.

Behavioral signatures may include:

- `DIRECT_SOLVE`
- `SEARCH_THEN_SOLVE`
- `MULTI_HYPOTHESIS`
- `SELF_CORRECT`
- `REASONING_OSCILLATION`
- `PREMATURE_COMMIT`
- `OVERTHINK_CORRUPTION`
- `FORMAT_DRIFT`
- `CORRECT_REASONING_WRONG_OUTPUT`
- `WRONG_REASONING_CORRECT_OUTPUT`
- `NO_FINAL_ANSWER`

Automatic signature classification must remain evidence-backed and inspectable.

## 8. Solution Emergence and Excess Reasoning

For objectively checkable tasks, the system may estimate the earliest observable point at which the trace contains a stable correct solution.

Derived fields may include:

- `earliest_observable_solution`
- `final_answer_commit`
- `excess_reasoning_duration`
- `excess_reasoning_events`

These are derived behavioral metrics and must be labeled accordingly.

The purpose is to discover whether the model routinely solves a task early and then spends additional compute without benefit, or whether extra reasoning corrupts an initially correct result.

## 9. Failure Taxonomy

Results must not collapse into a single PASS/FAIL bit.

Minimum classes:

- `ANSWER_CORRECT`
- `ANSWER_WRONG`
- `THINK_TRUNCATED`
- `ANSWER_TRUNCATED`
- `NO_FINAL_ANSWER`
- `FORMAT_FAILURE`
- `TOOL_FAILURE`
- `CONTEXT_FAILURE`
- `SELF_CORRECTED`
- `REASONING_LOOP`
- `OVERTHINK_CORRUPTION`
- `TIMEOUT`
- `RESOURCE_LIMIT`
- `RUNTIME_FAILURE`
- `TEST_DEFECT`
- `SCORER_DEFECT`
- `CAPTURE_GAP`

The exact Qwen 27B Q4 Test #1 event where a 16-token allowance was consumed by exposed thinking and the model emitted no final answer should be representable as a truncation/test-design failure rather than a false capability score of zero.

## 10. Failure Snapshot and Replay

Every meaningful failure must create an immutable replay fixture containing or referencing:

- exact model/tag/quant;
- exact serialized request;
- system prompt/messages;
- thinking mode and all runtime options;
- generation/context limits;
- seed/temperature;
- raw thinking stream;
- raw answer stream;
- runtime response metadata;
- timing;
- scorer evidence;
- aligned telemetry;
- hardware/runtime state;
- failure classification;
- artifact hashes.

Replay fixtures must be usable against the same model or a different model without reconstructing the original experiment from memory.

## 11. Recovery Laboratory

A retry is valid only when it tests a hypothesis.

Recovery interventions include:

- exact replay;
- more/less generation budget;
- thinking ON/OFF;
- reduced context/noise;
- minimal prompt clarification;
- stronger output constraints;
- explicit failure evidence;
- exact failed assertion or error;
- decomposition;
- structured intermediate state;
- verification pass;
- examples;
- seed change;
- controlled temperature change;
- targeted Inverted/mentor intervention.

Recommended intervention ladder:

- `R0` exact replay
- `R1` budget adjustment
- `R2` thinking-mode change
- `R3` minimal prompt clarification
- `R4` failure evidence
- `R5` decomposition
- `R6` structured intermediate state
- `R7` targeted corrective strategy
- `R8` external mentor/Inverted intervention

The system should identify the **minimum effective intervention**, not merely the strongest intervention that succeeds.

For successful recoveries, derive:

- recovery reliability;
- added wall time;
- added generated work;
- added energy/resources where measurable;
- capability gain;
- transferability across sibling tasks;
- minimum intervention needed.

## 12. Capability Families and Difficulty Frontiers

Initial capability families:

1. instruction control;
2. structured output;
3. extraction/transformation;
4. reasoning/math/logic/planning;
5. coding/debugging;
6. tool selection and arguments;
7. ambiguity/uncertainty handling;
8. context retrieval/reasoning;
9. state tracking and obsolete-information rejection;
10. adversarial/noise robustness;
11. self-correction;
12. meta-reasoning.

Difficulty levels are relative to the model, not absolute human labels. Use an adaptive ladder from sanity/basic through expert/adversarial/boundary/edge-case levels.

The goal for each family is to identify:

- known easy region;
- reliable operating region;
- unstable frontier;
- known failure region;
- important anomalies.

## 13. Controlled Task Mutations

High-value tasks should support controlled siblings where one factor changes:

- wording;
- distractor level;
- constraint count;
- misleading premise;
- missing information;
- context size;
- fact position;
- exact failure evidence;
- difficulty.

Mutation lineage must be retained so the system can distinguish underlying reasoning capability from wording, instruction, or context sensitivity.

## 14. Context Characterization

The system must distinguish:

1. **load limit** — largest context accepted without resource/runtime failure;
2. **retrieval limit** — largest context with reliable fact retrieval;
3. **reasoning limit** — largest context with reliable retrieval plus reasoning;
4. **operational limit** — largest context where capability, latency, energy, and resources remain worthwhile.

Actual runtime-reported prompt counts are authoritative. Requested `num_ctx` values must not be represented as actual prompt-token counts.

Adaptive context experiments should vary:

- total context size;
- fact position (beginning/middle/end and intermediate positions);
- number of planted facts;
- information density;
- distractor similarity;
- contradictory/obsolete facts;
- state updates across turns;
- context format;
- context compression strategy;
- thinking budget at each context region.

Important derived concepts:

- lost-in-the-middle onset;
- context-noise penalty;
- dense-context penalty;
- state-update reliability;
- obsolete-information rejection;
- effective context envelope;
- best memory representation for Qwen.

## 15. Prompt / Instruction Control Surface

Prompt engineering must become measured rather than intuitive.

Controlled ingredients include:

- role;
- objective;
- constraints;
- success criteria;
- output format;
- examples;
- decomposition;
- verification;
- failure/uncertainty rules;
- context hints;
- confidence/calibration rules;
- instruction ordering;
- positive versus negative wording;
- prompt verbosity.

Use ablation and interaction testing to identify:

- critical ingredients;
- unnecessary overhead;
- synergistic combinations;
- interfering combinations;
- minimal effective prompt;
- prompt robustness to minor wording changes;
- transferability across tasks/models/quants.

A successful long prompt should be compressed until removing another ingredient causes a measurable loss. Preserve the smallest reliable recipe.

## 16. Reliability and Variance

Near boundaries, characterize separately:

- answer stability;
- reasoning-signature stability;
- latency/performance stability;
- resource stability.

Replication count should be adaptive. Obvious interior-region cases need little repetition; unstable boundaries and anomalies warrant more.

Seed and temperature experiments should be used selectively to distinguish deterministic failure, seed sensitivity, and cases where a small amount of sampling variation improves recovery.

## 17. Cross-Quant and Cross-Model Experiments

After Qwen3.5 27B Q4 is characterized, Q8 and later models should inherit the highest-value replay cases rather than rerunning only a generic fixed suite.

Transfer sets should include:

- known Q4 pass;
- unstable Q4 frontier;
- known Q4 failure;
- Q4 recovery case;
- Q4 overthinking case;
- Q4 context boundary;
- Q4 anomaly.

Measure what a new quant/model actually buys in:

- capability;
- reasoning efficiency;
- context envelope;
- stability;
- recovery behavior;
- latency;
- VRAM/RAM;
- energy.

The planned independent model sequence remains:

1. `qwen3.5:27b-q4_K_M`
2. `qwen3.5:27b-q8_0`
3. `qwen3.5:35b-a3b-q4_K_M`
4. `qwen3.5:35b-a3b-q8_0`
5. `devstral-small-2:24b-instruct-2512-q8_0`

Each remains an independent test/run with its own evidence and run identity.

## 18. Research-Value Scheduler

Candidate next experiments should be ranked by an explicit value concept such as:

`value = boundary_uncertainty × anomaly_importance × expected_information_gain × transferability / estimated_compute_cost`

The exact numeric formula may evolve, but the scheduler must prefer:

- uncertain PASS/FAIL boundaries;
- unexplained anomalies;
- cross-model disagreements;
- high-impact recovery hypotheses;
- context failures below expectation;
- thinking behavior that contradicts the current model profile.

It should deprioritize:

- repeated trivial passes;
- budgets far above a known stable optimum;
- duplicate prompt variants without a hypothesis;
- repeated failures that no longer reduce uncertainty.

## 19. Stop Rules

An experimental branch should stop when one of the following is true:

- a stable boundary has been bracketed to the configured precision;
- repeated larger budgets produce the same semantic failure and truncation is not implicated;
- a recovery is reproduced at the minimum effective intervention;
- additional replications no longer materially change reliability confidence;
- resource/time limits are reached;
- a test/scorer defect invalidates the branch;
- expected information gain falls below the configured threshold.

Stopping must be evidence-driven and recorded.

## 20. Knowledge Base and Operating Rules

Derived knowledge should be stored in a structured, queryable form under model identity, including:

- capability frontiers;
- reasoning curves;
- context envelopes;
- prompt recipes;
- failures;
- recoveries;
- anomalies;
- reliability estimates;
- resource profiles;
- replay cases;
- derived operating rules.

Every rule must include:

- claim;
- scope;
- supporting experiment IDs;
- evidence references;
- reliability/confidence basis;
- known exceptions;
- date/model/runtime/version context.

No derived rule may exist without provenance.

## 21. Automatic Hypothesis Generation

Once enough evidence exists, the system may propose follow-up hypotheses from observed patterns.

Examples:

- middle-position retrieval fails while start/end retrieval succeeds → test positional sensitivity at additional positions;
- Q4 succeeds at a smaller reasoning budget than Q8 → test paired reasoning dynamics before assuming the quant is inferior;
- thinking hurts strict JSON but helps math → generate family-specific thinking-policy hypotheses;
- compressed context beats full context → test alternate memory encodings.

Automatically proposed experiments remain subject to the same one-variable, evidence, value, and stop rules as human-authored experiments.

## 22. Progress and Wall-Clock Visibility

All characterization runs inherit the adaptive terminal progress requirement:

- terminal-width-aware progress bar;
- percentage complete;
- completed/total/left experiments or tasks;
- current stage/task;
- elapsed wall-clock time;
- continuously recalibrated ETA;
- estimated wall-clock finish time.

Because adaptive runs can add or remove planned experiments, progress evidence must record plan adjustments rather than silently changing the denominator.

## 23. Evidence Requirements

The existing lossless-evidence contract remains authoritative and is extended for characterization.

Preserve, when observable:

- exact serialized runtime request;
- every raw runtime stream chunk;
- exposed thinking and final response separately;
- unknown runtime fields;
- runtime metrics;
- task/scorer inputs and outputs;
- exact failure snapshots;
- recovery lineage;
- telemetry samples and raw NVIDIA collector output;
- progress events;
- hardware/runtime snapshots;
- derived metric provenance;
- hashes/byte counts in the evidence manifest.

If something cannot be captured, record `UNAVAILABLE` or a capture-gap event. Do not infer missing raw data.

Periodic live progress redraw frames do not need to become raw research evidence unless explicitly enabled; task/experiment boundary progress and plan changes must be persisted.

## 24. Safety Against False Conclusions

The analysis layer must explicitly guard against:

- calling a truncated thought-only response a capability failure;
- comparing models with different effective budgets without labeling the difference;
- treating one pass as stable capability;
- treating requested context as actual prompt tokens;
- interpreting exposed reasoning text as full hidden cognition;
- attributing scorer/harness faults to the model;
- changing multiple factors and claiming a causal mechanism;
- using aggregate averages to hide important boundary/anomaly behavior.

## 25. First Implementation Target

Before model #2 is run, the first implementation slice should make Qwen3.5 27B Q4 scientifically valid to characterize.

Minimum first slice:

1. repair scorer/test defects exposed by run `20260907-215638-6ee38d88`;
2. add explicit thinking mode control where Ollama supports it;
3. separate thinking-only truncation from semantic model failure;
4. make generation budget an experimental variable rather than a fixed assumption;
5. add experiment identity/lineage and hypothesis/change metadata;
6. add adaptive budget bracketing for a small set of representative tasks;
7. create exact replay fixtures for failures;
8. retain progress and all evidence under the existing manifest contract;
9. validate the adaptive logic with deterministic fake-runtime tests before spending local model compute.

The first real adaptive characterization target is `qwen3.5:27b-q4_K_M`. Model #2 must not begin until the Q4 harness produces scientifically valid capability/thinking results.

## 26. Acceptance Criteria

The architecture is implemented successfully when the system can, for a representative task:

1. run thinking OFF and ON baselines;
2. distinguish answer correctness from truncation/harness/scorer failure;
3. adaptively bracket a reasoning-budget boundary without exhaustively testing all budgets;
4. reproduce a boundary or anomaly when uncertainty requires it;
5. create a replayable failure snapshot;
6. test at least one controlled recovery intervention;
7. identify the minimum reliable recovery among attempted interventions;
8. preserve exact evidence and provenance;
9. produce a model/task operating profile containing reliable/unstable/failure regions and resource cost;
10. complete with a verified manifest and no unclassified silent evidence loss.

## 27. End State

The final system should be able to answer, from evidence rather than intuition:

- Which Qwen should Inverted call for this task?
- Should thinking be enabled?
- How much reasoning headroom is enough?
- What context representation and size should be supplied?
- Which prompt ingredients matter?
- How reliable is this configuration?
- What failure is most likely?
- What is the cheapest proven recovery?
- What capability/resource tradeoff does Q4 versus Q8 versus another model actually provide?

The system is complete only when model behavior is transformed from a black-box score into an evidence-backed, replayable operating specification.