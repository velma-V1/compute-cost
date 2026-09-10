# 700-Call Full Comparability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved <=700-call benchmark so every executed capability task, reasoning condition, retry, autonomous decision/turn, family, scenario, and model aggregate is fully scored and directly comparable.

**Architecture:** Preserve the existing capability runner and evidence pipeline. Add a single authoritative call ledger, a fixed-core matrix scheduler for L2/L5/L8/L10 across native reasoning conditions, multidimensional attempt scoring, autonomous semantic-equivalence scoring, and zero-call synthesis/reporting. Adaptive diagnostics use only post-core reserve and never affect official matched-core scores.

**Tech Stack:** Python 3.11/3.12, pytest, Ollama runtime, existing EvidenceStore/runtime adapters.

**Spec:** `docs/superpowers/specs/2026-09-10-700-call-full-comparability-design.md`

## Global Constraints

- Absolute model-call ceiling: 700 per run, enforced before runtime inference.
- GPT-OSS fixed core: 40 families x L2/L5/L8/L10 x low/medium/high = 480 capability calls; 6 scenarios x 8 turns x low/medium/high = 144 autonomous calls; 76-call reserve.
- Qwen uses only real supported controls: think=false and think=true. Unsupported cells are explicit and excluded from denominators.
- Every executed call creates one scored attempt dossier.
- Semantic correctness, contract/format compliance, tool/procedure compliance, autonomous state/decision/recovery/verification, and cost metrics remain separate dimensions.
- Token-budget escalation occurs only after actual truncation and resets for the next independent task.
- Diagnostic/adaptive rows are excluded from official fixed-core aggregates.
- Existing raw evidence, replay, Windows 11, local Ollama, and privacy contracts remain intact.

---

### Task 1: Authoritative Call Ledger

**Files:**
- Create: `src/compute_cost/call_ledger.py`
- Create: `tests/test_call_ledger.py`
- Modify: `src/compute_cost/runner_core.py`
- Modify: `config/default.toml`

**Interfaces:**
- Produces: `CallLedger(limit: int)`, `authorize(category, family=None, scenario=None)`, `snapshot()`, and a runner-owned ledger persisted as `call-ledger.json`.

- [ ] Write tests proving calls 1..700 are authorized, call 701 is blocked before runtime invocation, categories/families/scenarios reconcile, and snapshot remaining count is exact.
- [ ] Run targeted tests and verify RED because `CallLedger` does not exist.
- [ ] Implement minimal ledger and wire authorization immediately before every model inference in `runner_core`.
- [ ] Set `max_model_calls_per_run = 700` in config/defaults.
- [ ] Run targeted tests and full suite; commit.

### Task 2: Native Reasoning Conditions and Fixed Comparable Matrix

**Files:**
- Create: `src/compute_cost/comparison_matrix.py`
- Create: `tests/test_comparison_matrix.py`
- Modify: `src/compute_cost/compact_patch.py`
- Modify: `src/compute_cost/capability_campaign.py` only where required for shared experiment construction.

**Interfaces:**
- Produces: `native_reasoning_conditions(model)`, `fixed_comparison_cells(cases, model)`, and `run_fixed_capability_matrix(...)`.

- [ ] Write failing tests for GPT-OSS low/medium/high, Qwen think=false/true, explicit unsupported MAX_NATIVE for Qwen, exact L2/L5/L8/L10 task identity, fixed-core labeling, and planned GPT capability count=480.
- [ ] Verify RED.
- [ ] Implement model-native condition resolver and deterministic fixed-cell scheduler.
- [ ] Ensure each fixed task starts at base output budget; only truncation retries change budget.
- [ ] Run targeted/full tests; commit.

### Task 3: Multidimensional Attempt Score Vector and Full Dossiers

**Files:**
- Create: `src/compute_cost/score_vector.py`
- Create: `tests/test_score_vector.py`
- Modify: `src/compute_cost/attempt_dossier.py`

**Interfaces:**
- Produces: `build_score_vector(case, scoring, classification, generation, telemetry) -> dict` and persists `score_vector` for every executed call.

- [ ] Write failing tests showing semantic correctness can be 100 while contract format is 0, non-applicable fields use `NOT_APPLICABLE`, unavailable metrics use `UNAVAILABLE`, and value/cost metrics are calculated from retained evidence.
- [ ] Add a test proving a successful non-retry call also creates a dossier; verify RED under the current failure/retry-only policy.
- [ ] Implement score-vector grouping and change dossier persistence from failure/retry-only to every executed call.
- [ ] Preserve right/wrong checks, exposed thinking, hidden reasoning=`UNOBSERVABLE`, raw refs, and lineage.
- [ ] Run targeted/full tests; commit.

### Task 4: Autonomous Matrix and Decision Scoring

**Files:**
- Modify: `src/compute_cost/autonomous_simulation.py`
- Modify: `src/compute_cost/autonomous_dossier.py`
- Create: `tests/test_autonomous_comparability.py`

**Interfaces:**
- Extends each scenario turn with standardized/native reasoning condition, full score vector, and declared semantic-equivalence action rubric.

- [ ] Write failing tests for 6x8 turns per supported reasoning condition, GPT planned autonomous core=144, Qwen planned autonomous core=96, state/checkpoint scoring, semantic-equivalent `TEST`/`RETEST` credit with exact contract distinction, and per-turn dossier creation.
- [ ] Verify RED.
- [ ] Implement reason-mode outer loop without changing transcript semantics inside each independent scenario/mode run.
- [ ] Add declared equivalence classes to fixture/rubric before scoring; never post-hoc infer equivalence.
- [ ] Run targeted/full tests; commit.

### Task 5: Retry Deltas, Family/Task/Reasoning Aggregation, and Comparison Cells

**Files:**
- Create: `src/compute_cost/comparability_report.py`
- Create: `tests/test_comparability_report.py`
- Modify: `src/compute_cost/compact_scorecard.py`
- Modify: `src/compute_cost/compact_patch.py`

**Interfaces:**
- Produces `task-scorecard.json`, `reasoning-comparison.json/.md`, `retry-deltas.json`, `comparison-cells.json`, updated `scorecard.json/.md`.

- [ ] Write failing tests for attempt->task->difficulty->reasoning->family aggregation, macro weighting, adaptive diagnostic exclusion, retry parent/child deltas, unsupported-cell exclusion, and component-vector preservation.
- [ ] Verify RED.
- [ ] Implement zero-call synthesis over retained experiment/dossier evidence.
- [ ] Rename raw adaptive ratio to `observed_adaptive_accuracy`; do not expose it as official capability score.
- [ ] Render human-readable tables for every family, fixed task, reasoning condition, autonomous turn, retry chain, and model component.
- [ ] Run targeted/full tests; commit.

### Task 6: Reporting Integrity and Known Defects

**Files:**
- Modify: `src/compute_cost/run_synthesis.py`
- Modify: `src/compute_cost/failure_atlas.py`
- Modify: `src/compute_cost/compact_patch.py`
- Create/modify tests under `tests/` for synthesis integrity.

**Interfaces:**
- Final run emits/reconciles `call-ledger.json`, all comparison reports, complete failure atlas, and correct native reasoning/autonomous status.

- [ ] Write failing regressions for autonomous evidence never being reported UNTESTED, retained native controls overriding stale `medium`, and failure atlas count reconciling with valid failure dossiers.
- [ ] Verify RED.
- [ ] Implement minimal synthesis fixes and integrity checks.
- [ ] Add final reconciliation: runtime request count == ledger used calls == scored dossier count for model calls; mismatch marks run-integrity failure.
- [ ] Run full tests on supported Python versions through CI; commit.

### Task 7: Acceptance Audit and Retest Handoff

**Files:**
- Verify all files from Tasks 1-6 plus design spec.

- [ ] Run full pytest suite in CI and require zero failures on supported Python versions.
- [ ] Verify default config says 700 and fixed-core counts are mathematically correct.
- [ ] Verify every spec acceptance criterion maps to a passing test or explicit evidence artifact.
- [ ] Verify no personal data introduced.
- [ ] Verify no model call can occur beyond ledger authorization.
- [ ] Only after fresh green verification, provide the user the exact PowerShell retest command for `gpt-oss:20b` and `qwen3.5:35b-a3b-q4_K_M`.
