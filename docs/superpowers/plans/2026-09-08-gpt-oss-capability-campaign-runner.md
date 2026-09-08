# GPT-OSS Capability Campaign Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect versioned GPT-OSS family/level fixtures to the adaptive difficulty controller so the harness executes only informative L0-L10 probes and emits evidence-backed `capability-frontiers.json` without changing the legacy Qwen `characterize` path.

**Architecture:** Keep capability difficulty search in a new `capability_campaign` module and treat the existing suite `cases` array as the fixture bank. A ladder index groups normalized fixtures by family and level; the scheduler asks `AdaptiveDifficultyController` for levels and resolves those decisions only to declared fixtures. `BenchmarkRunner.capability_characterize()` reuses existing preflight, evidence, scoring, telemetry, and manifest machinery but writes capability-specific outputs. Legacy `characterize()` remains behaviorally unchanged.

**Tech Stack:** Python 3.11+, pytest, JSON benchmark fixtures, existing `compute_cost` evidence/scoring/runtime layers, GitHub Actions Windows 3.11/3.12 matrix.

**Spec:** `docs/superpowers/specs/2026-09-08-gpt-oss-20b-capability-map-design.md`

## Global Constraints

- Preserve `benchmarks/qwen-characterization-v1.json` byte-for-byte.
- Difficulty is family-local integer `0..10`.
- Unit tests make no Ollama/model/network calls.
- Invalid harness/runtime observations never count as capability pass/fail observations.
- The controller must not mechanically execute all 11 levels.
- Every executed capability experiment retains exact existing generation/scoring evidence and stable experiment lineage.
- Missing ladder levels remain explicit coverage gaps; the runner must not invent fixtures.
- Default reliability thresholds remain `reliable >= 0.90`, `unstable >= 0.40`, `failure < 0.40`.
- Legacy `BenchmarkRunner.characterize()` and `characterize-campaign` behavior remain unchanged.

---

### Task 1: Difficulty lineage + ladder index

**Files:**
- Modify: `src/compute_cost/experiments.py`
- Create: `src/compute_cost/capability_ladders.py`
- Create: `tests/test_capability_ladders.py`
- Modify: `tests/test_experiments.py`

**Interfaces:**
- Produces: `build_ladder_index(cases: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]`.
- Produces: `resolve_requested_level(ladder: dict[int, dict[str, Any]], requested_level: int, attempted_levels: set[int]) -> int | None`.
- Extends `CONTROLLED_FIELDS` with `difficulty_level` so a parent/child experiment can explicitly declare `changed_variable="difficulty_level"`.

- [ ] **Step 1: Write RED tests** proving normalized family IDs are grouped by level, duplicate family/level fixtures fail deterministically, sparse ladders return only declared levels, and `changed_fields()` reports `difficulty_level` when only difficulty changes.
- [ ] **Step 2: Run** `python -m pytest tests/test_capability_ladders.py tests/test_experiments.py -q` and verify RED is caused by the missing ladder module / missing controlled field.
- [ ] **Step 3: Implement** the minimal index and resolver. `resolve_requested_level` returns the requested level when available and unattempted; otherwise it chooses the nearest unattempted declared level by `(abs(level-requested), level)`. It returns `None` when no declared unattempted level remains.
- [ ] **Step 4: Re-run targeted tests** and require PASS.
- [ ] **Step 5: Commit** with `feat: add capability ladder indexing`.

---

### Task 2: Adaptive family campaign executor

**Files:**
- Create: `src/compute_cost/capability_campaign.py`
- Create: `tests/test_capability_campaign.py`

**Interfaces:**
- Produces: `run_family_frontier(runner: Any, family_id: str, ladder: dict[int, dict[str, Any]], *, sequence_start: int = 0) -> tuple[list[dict[str, Any]], int]`.
- Produces: `run_capability_campaign(runner: Any, cases: list[dict[str, Any]]) -> list[dict[str, Any]]`.
- Uses `AdaptiveDifficultyController`, `ExperimentSpec`, and existing `characterization.execute_experiment` for actual evidence/scoring.

- [ ] **Step 1: Write RED tests** with a fake runner/executor surface proving the call order follows anchor -> jump -> bisect -> boundary replication rather than 0..10 enumeration; invalid observations retry without entering pass-rate denominators; sparse fixture banks stop with a recorded `MISSING_FIXTURE_COVERAGE` event rather than synthesizing a task.
- [ ] **Step 2: Verify RED** with `python -m pytest tests/test_capability_campaign.py -q`.
- [ ] **Step 3: Implement minimal execution**. Every spec uses stable family search lineage, `changed_variable="difficulty_level"` for level moves and `changed_variable="replication"` for exact repeats, fixed campaign thinking/budget config, and the declared case prompt/scorer for that level.
- [ ] **Step 4: Convert result rows into frontier observations** shaped as `{family_id, level, passed, valid_for_capability, result_class, experiment_id, fixture_id}` while retaining the full experiment rows in `experiments.jsonl`.
- [ ] **Step 5: Re-run targeted and full tests**; require PASS.
- [ ] **Step 6: Commit** with `feat: execute adaptive capability frontiers`.

---

### Task 3: Runner artifacts and coverage ledger

**Files:**
- Modify: `src/compute_cost/runner.py`
- Create: `tests/test_capability_runner.py`

**Interfaces:**
- Produces: `BenchmarkRunner.capability_characterize(model: str, *, pull: bool = False) -> Path`.
- Writes: `capability-frontiers.json`, `coverage-ledger.json`, existing `experiments.jsonl`, telemetry/evidence, and manifest.

- [ ] **Step 1: Write RED integration tests** using the existing fake-runtime pattern. Assert no model/network dependency, capability mode reuses preflight evidence, only controller-selected levels are generated, `capability-frontiers.json` is built via `build_capability_frontiers`, and every taxonomy family is represented in `coverage-ledger.json` as `PROVEN`, `PARTIAL`, `UNCERTAIN`, or `UNTESTED`.
- [ ] **Step 2: Verify RED** with `python -m pytest tests/test_capability_runner.py -q`.
- [ ] **Step 3: Implement `capability_characterize`** as a sibling of `characterize`, not a replacement. It validates/normalizes the capability suite, runs preflight, calls `run_capability_campaign`, groups observations by family, writes frontier and coverage artifacts, finalizes evidence manifest, and records `CAPABILITY_CHARACTERIZATION_COMPLETE`.
- [ ] **Step 4: Progress accounting** begins with preflight + one planned anchor per declared family + finalize and grows dynamically for adaptive probes using the existing `plan_adjusted` mechanism.
- [ ] **Step 5: Run targeted + full tests** and require PASS.
- [ ] **Step 6: Commit** with `feat: add capability characterization runner`.

---

### Task 4: CLI/config execution contract

**Files:**
- Modify: `src/compute_cost/cli.py`
- Modify: `config/default.toml`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Adds command: `compute-cost capability-characterize --model <model> --suite <path> --taxonomy <path> [--pull]`.
- Default suite: `benchmarks/gpt-oss-20b-capability-v1.json`.
- Default taxonomy: `benchmarks/capability-taxonomy-v1.json`.
- Adds `[capability_campaign]`: `anchor_level=2`, `jump=3`, `boundary_repeats=5`, `max_experiments_per_family=24`, `thinking_mode=true`, `generation_budget=256`, `reliable_threshold=0.90`, `unstable_threshold=0.40`.

- [ ] **Step 1: Write RED CLI tests** proving parser defaults, taxonomy loading/validation, and dispatch to `capability_characterize` without altering `characterize` defaults.
- [ ] **Step 2: Verify RED** with `python -m pytest tests/test_cli.py -q`.
- [ ] **Step 3: Implement minimal CLI/config wiring** and preserve all existing commands.
- [ ] **Step 4: Run full suite** on the branch.
- [ ] **Step 5: Commit** with `feat: expose GPT-OSS capability campaign CLI`.

---

### Task 5: Ladder completeness contract and fixture expansion boundary

**Files:**
- Modify: `tests/test_capability_suite.py`
- Modify: `src/compute_cost/capability_suite.py`
- Modify: `benchmarks/gpt-oss-20b-capability-v1.json` in subsequent fixture-data commits.

**Interfaces:**
- Produces: `build_ladder_coverage(suite: dict[str, Any], taxonomy: dict[str, Any]) -> dict[str, Any]` with per-family declared/testable levels and missing levels.

- [ ] **Step 1: Write RED tests** requiring deterministic ladder coverage reporting for all 40 families without requiring all 11 levels to exist yet.
- [ ] **Step 2: Implement coverage derivation**; missing levels are data gaps, not validator failures.
- [ ] **Step 3: Add fixture-data commits family-by-family until every family has L0-L10 declared fixtures.** Each fixture keeps a deterministic scorer and family-local rubric dimensions; fixture-only commits must pass the validator before the next family is added.
- [ ] **Step 4: When all 40×11 levels exist, add a final contract assertion** that `missing_levels == []` for every family and re-run the complete suite.

---

### Task 6: Verification and handoff

- [ ] Verify the full Windows Python 3.11 and 3.12 GitHub Actions matrix is green on the final head.
- [ ] Verify `benchmarks/qwen-characterization-v1.json` SHA-256 remains `813acd321edefeeb42a4d5e11c98906aea022f0973029c3553db924575189b0a`.
- [ ] Verify unit tests made zero real model calls.
- [ ] Verify the diff contains no result directories, secrets, personal data, or machine-specific paths.
- [ ] Only after these gates merge the branch into `feat/adaptive-qwen-characterization`.
