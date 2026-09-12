# Test 1.2 Frontier-Gap Research Audit

Date: 2026-09-12

This audit searched recent agent and inference-time research for capability
classes that were not already measured strongly enough by Test 1.2. The rule
for inclusion was additive value: a candidate had to introduce a distinct
controller decision, reliability surface, or manufacturing signal rather than
another wording of an existing prompt/retry experiment.

## Added gaps

### 1. Adaptive width-vs-depth search

Research signal:
- *Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching
  Tree Search* (arXiv:2503.04412, 2025).

Why it matters:
Repeated sampling and ordinary self-refinement test different uses of
inference-time compute. The useful controller question is when to branch wider,
when to refine deeper, and when extra search is wasted.

Test 1.2 addition:
- two independent branches;
- verifier comparison;
- conditional third refinement only when verifier returns REFINE;
- final adjudicated answer;
- rescue/regression/cost evidence.

Artifact: `adaptive-search-map.json`.

### 2. Metamorphic reliability

Research signal:
- *ReliabilityBench: Evaluating LLM Agent Reliability Under Production-Like
  Stress Conditions* (arXiv:2601.06112, 2026);
- *Metamorphic Testing of Large Language Models for Natural Language
  Processing* (arXiv:2511.02108, 2025).

Why it matters:
A model that succeeds on one exact prompt but fails under a semantics-preserving
wrapper is not reliably improved. End-state equivalence and invariance are
different from ordinary accuracy.

Test 1.2 addition:
- semantics-preserving wrappers;
- formatting perturbations;
- per-family stable/improved/regressed counts;
- regression sentinels for wrappers that damage a family.

Artifact: `metamorphic-reliability-map.json`.

### 3. Calibrated abstention

Research signal:
- *AgentAbstain: Do LLM Agents Know When Not to Act?*
  (arXiv:2607.10059, 2026).

Why it matters:
Task-solving ability and knowing when not to act are partly independent.
Autonomous systems need paired act/abstain evidence, especially before
irreversible or under-specified actions.

Test 1.2 addition:
- paired ACT/ABSTAIN synthetic cases;
- paired accuracy;
- false-act and false-abstain counts;
- compiler rules for pre-action guards and evidence-gather-before-abstain.

Artifact: `abstention-calibration-map.json`.

### 4. Active memory write/manage/read

Research signal:
- *Context as a Tool: Context Management for Long-Horizon SWE-Agents*
  (arXiv:2512.22087, 2025);
- *Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving
  Memory* (arXiv:2511.20857, 2025).

Why it matters:
Static context compression does not test whether an agent can decide what to
write, remove superseded state, and later read the right compact memory.

Test 1.2 addition:
- append-only baseline;
- explicit memory-write/manage step;
- final read/answer step;
- rescue and regression comparison;
- enable active memory only where measured value is non-negative.

Artifact: `active-memory-evolution-map.json`.

### 5. Controlled reflection transfer

Research signal:
- *BenchTrace: A Benchmark for Testing Reflection Ability and Controlled
  Evolution in LLM Agents* (arXiv:2605.29225, 2026).

Why it matters:
A reflection is not useful because it sounds plausible. It is useful only when
a lesson derived from one failure prevents the same failure class on a
different task without causing negative transfer.

Test 1.2 addition:
- failed source attempt;
- no oracle answer revealed;
- compact general lesson;
- apply lesson to a sibling fixture;
- measure sibling rescue versus negative transfer.

Artifact: `reflection-transfer-map.json`.

### 6. Tool-chaos recovery and corrupted-success detection

Research signal:
- *When Tools Fail: Benchmarking Dynamic Replanning and Anomaly Recovery in LLM
  Agents* / ToolMaze (arXiv:2606.05806, 2026);
- *ToolFailBench: Diagnosing Tool-Use Failures in LLM Agents*
  (arXiv:2607.04686, 2026).

Why it matters:
Ordinary tool-error tests mostly exercise explicit failures. Production agents
also fail when a tool returns `ok=true` with stale or semantically corrupted
data. Blindly trusting successful responses is a separate failure mode.

Test 1.2 addition:
- explicit transient timeout;
- explicit permanent service removal;
- implicit semantic corruption;
- stale successful value;
- backup/verification paths;
- blind-identical-retry detection.

Artifact: `tool-chaos-recovery-map.json`.

### 7. Dependency-aware tool scheduling

Research signal:
- *TPS-Bench: Evaluating AI Agents' Tool Planning & Scheduling Abilities in
  Compounding Tasks* (arXiv:2511.01527, 2025).

Why it matters:
Correct tool selection is not enough. A harness can be materially faster by
parallelizing independent tools while preserving dependency constraints.

Test 1.2 addition:
- deterministic dependency DAGs;
- explicit start-time schedules;
- dependency validation;
- exact critical-path optimum;
- makespan and efficiency;
- compiler enables parallel scheduling only above reliability/efficiency
  thresholds.

Artifact: `tool-scheduling-map.json`.

## Manufacturing contract

The seven additions are mandatory and additive. No existing Test 1.2 phase was
removed. The collection now has a 7h40m hard ceiling and 7h25m active ceiling.
The tuning run remains 6h15m hard / 6h active. Combined hard ceiling is 13h55m.

The tuning run refuses a collection unless all seven labs completed and each
map contains measured output. The compiled harness receives a
`frontier_gap_policy` that converts the measurements into concrete activation,
fallback, guard, verification, memory, search, reflection, and scheduling
rules.
