# GPT-OSS 20B Frontier Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the remaining initial-increment contract gaps, then add a pure family-local L0-L10 adaptive difficulty controller and deterministic frontier summarization without changing the legacy Qwen execution path.

**Architecture:** Normalize both legacy and extended source fixtures into one canonical internal shape before frontier analysis. Keep generation-budget search separate from difficulty search: `AdaptiveBudgetController` remains responsible for reasoning/generation headroom, while a new `AdaptiveDifficultyController` consumes level outcomes and chooses the next family-local difficulty probe. Frontier labels are derived from raw replicated pass rates using versioned thresholds.

**Tech Stack:** Python 3.11+, pytest, existing JSON benchmark fixtures and `compute_cost` package.

**Spec:** `docs/superpowers/specs/2026-09-08-gpt-oss-20b-capability-map-design.md`

## Global Constraints

- Preserve `benchmarks/qwen-characterization-v1.json` byte-for-byte.
- Canonical taxonomy IDs follow the approved design spec exactly, including `debugging_root_cause_diagnosis` and `memory_compression_summary_fidelity`.
- Source fixture formats remain additive/backward compatible; normalization may not require edits to legacy source files.
- Difficulty remains family-local and integer `0..10`.
- Default frontier labels are `reliable >= 0.90`, `unstable >= 0.40 and < 0.90`, `failure < 0.40`.
- Invalid/harness/runtime observations do not count as model capability passes or failures.
- No model calls occur in unit tests.

---

### Task 1: Repair canonical fixture normalization contract

**Files:**
- Modify: `tests/test_capability_suite.py`
- Modify: `src/compute_cost/capability_suite.py`
- Modify: `benchmarks/capability-taxonomy-v1.json`
- Modify: `benchmarks/gpt-oss-20b-capability-v1.json`

**Interfaces:**
- Produces: `normalize_capability_case(case: dict[str, Any], *, suite_version: str, taxonomy_version: str | None = None) -> dict[str, Any]`.
- Produces: `normalize_capability_suite(suite: dict[str, Any], taxonomy: dict[str, Any] | None = None) -> dict[str, Any]`.

- [ ] Write failing tests proving a legacy Qwen case normalizes without source mutation.
- [ ] Write failing tests proving an extended nested `capability_map` case normalizes to canonical top-level `family_id`, `taxonomy_version`, `difficulty`, `capabilities_required`, eligibility flags, `compound`, `tags`, and `scorer_version`.
- [ ] Write failing test locking the two spec family IDs.
- [ ] Run CI and confirm RED is caused only by missing normalization / ID drift.
- [ ] Implement minimal normalization and rename the two family IDs in taxonomy/suite source.
- [ ] Run full suite to GREEN.

Canonical normalized legacy case defaults:

```python
{
    "family_id": case["category"],
    "taxonomy_version": None,
    "difficulty": {
        "level": case.get("difficulty_level", 0),
        "rubric_version": "legacy",
        "dimensions": {},
    },
    "scorer_version": "legacy",
    "capabilities_required": [case["category"]],
    "recovery_eligible": True,
    "robustness_eligible": True,
    "compound": False,
    "tags": [],
}
```

---

### Task 2: Pure adaptive difficulty controller

**Files:**
- Modify: `tests/test_adaptive.py`
- Modify: `src/compute_cost/adaptive.py`

**Interfaces:**
- Produces: `DifficultyObservation(level: int, passed: bool | None, valid_for_capability: bool = True)`.
- Produces: `DifficultyDecision(action: str, level: int | None, reason: str)`.
- Produces: `AdaptiveDifficultyController(min_level=0, max_level=10, anchor_level=1, jump=3, boundary_repeats=5)` with `.next(observations)`.

Controller behavior:

```text
no observations -> PROBE anchor
PASS with no higher fail -> jump upward by configured jump, bounded at L10
PASS/FAIL bracket wider than one level -> bisect integer level
adjacent pass/fail bracket -> replicate both transition levels until boundary_repeats valid observations exist at each
invalid-only observations -> retry same level rather than treating them as capability failure
replicated boundary -> STOP
```

- [ ] Write RED tests for anchor, jump, bracket+bisection, adjacent replication, invalid observation handling, and L10 ceiling.
- [ ] Implement minimal pure controller.
- [ ] Run targeted and full tests GREEN.

---

### Task 3: Probabilistic frontier derivation

**Files:**
- Create: `tests/test_frontier.py`
- Create: `src/compute_cost/frontier.py`

**Interfaces:**
- Produces: `classify_pass_rate(pass_rate: float, *, reliable_threshold: float = 0.90, unstable_threshold: float = 0.40) -> str`.
- Produces: `build_family_frontier(family_id: str, observations: list[dict[str, Any]], *, reliable_threshold: float = 0.90, unstable_threshold: float = 0.40) -> dict[str, Any]`.

The frontier output must retain raw counts and rates per level and derive:

```text
reliable_floor = highest tested level labeled reliable with no lower-level contradiction
unstable_levels = tested levels labeled unstable
first_failure_level = lowest tested level labeled failure above the reliable region
transition_bracket = nearest reliable/failure or reliable/unstable boundary
coverage = tested / untested levels
```

Invalid observations are counted separately and excluded from pass-rate denominators.

- [ ] Write RED tests for threshold boundaries, invalid exclusion, raw counts, frontier selection, and sparse levels.
- [ ] Implement minimal derivation functions.
- [ ] Run targeted and full tests GREEN.

---

### Task 4: Frontier artifact contract

**Files:**
- Create: `tests/test_frontier_artifact.py`
- Modify: `src/compute_cost/frontier.py`

**Interfaces:**
- Produces: `build_capability_frontiers(taxonomy_version: str, family_observations: dict[str, list[dict[str, Any]]], *, thresholds: dict[str, float] | None = None) -> dict[str, Any]`.

Top-level output:

```json
{
  "schema_version": 1,
  "taxonomy_version": "capability-taxonomy-v1",
  "thresholds": {"reliable": 0.9, "unstable": 0.4},
  "families": {}
}
```

- [ ] Write RED artifact tests.
- [ ] Implement aggregate artifact builder with deterministic family ordering.
- [ ] Run full suite GREEN.

---

### Task 5: Verification and next boundary

- [ ] Verify Windows Python 3.11 and 3.12 CI are both green.
- [ ] Verify Qwen fixture hash remains `813acd321edefeeb42a4d5e11c98906aea022f0973029c3553db924575189b0a`.
- [ ] Verify no model execution was added to tests.
- [ ] After GREEN, next slice adds the per-family L0-L10 fixture ladders and runner orchestration that feeds actual model observations into this controller/artifact layer.
