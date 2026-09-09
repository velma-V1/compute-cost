# GPT-OSS 20B Capability Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Copy the proven Qwen characterization fixture skeleton into a new, backward-compatible GPT-OSS capability suite that declares all 40 capability families and carries versioned L0-L10 metadata without modifying historical Qwen fixtures.

**Architecture:** Keep `benchmarks/qwen-characterization-v1.json` byte-for-byte unchanged. Add a standalone taxonomy catalog plus a new GPT-OSS suite whose cases retain the legacy flat fields consumed by `run_characterization` (`id`, `category`, `difficulty_level`, `prompt`, `scorer`, expected/scorer-specific fields, `timeout_s`) and add namespaced capability-map metadata. Add a small validator module so the new contract is mechanically checkable while the existing runner remains untouched in this slice.

**Tech Stack:** Python 3.11+, JSON fixtures, pytest 8+, existing `compute_cost` package and GitHub Actions Windows test matrix.

**Spec:** `docs/superpowers/specs/2026-09-08-gpt-oss-20b-capability-map-design.md`

## Global Constraints

- `benchmarks/qwen-characterization-v1.json` must remain byte-for-byte unchanged; expected SHA-256 is `8d7cd2eadaa3c105a491f234200f57c6332d019938104619d11088330c855618`.
- New schema is additive and must preserve the flat fields already consumed by the current characterization runner/scorers.
- Taxonomy version is `capability-taxonomy-v1`; suite version is `gpt-oss-20b-capability-v1`.
- Difficulty levels are integers `0..10` and each case must name a family-local rubric version plus explicit difficulty dimensions.
- All 40 declared capability families must exist in the taxonomy and have a coverage state from `PROVEN`, `PARTIAL`, `UNCERTAIN`, `UNTESTED`.
- The first breadth template must provide at least one deterministic anchor case for every family; deeper L0-L10 ladders are a subsequent slice.
- No model execution, user benchmark evidence, secrets, or machine-specific paths are committed.

---

### Task 1: Contract Tests (RED)

**Files:**
- Create: `tests/test_capability_suite.py`
- Modify: none

**Interfaces:**
- Consumes: JSON files under `benchmarks/`; `compute_cost.capability_suite.validate_capability_suite` which does not yet exist.
- Produces: executable contract for taxonomy completeness, suite compatibility, legacy-fixture immutability, case metadata consistency, and coverage ledger completeness.

- [ ] **Step 1: Write failing tests**

Create tests that assert:

```python
EXPECTED_QWEN_SHA256 = "8d7cd2eadaa3c105a491f234200f57c6332d019938104619d11088330c855618"
EXPECTED_FAMILIES = {
    "instruction_following_constraint_stacking",
    "strict_structured_output",
    "extraction_transformation",
    "arithmetic_numerical_reasoning",
    "algebra_quantitative_reasoning",
    "formal_logic_deduction",
    "causal_counterfactual_reasoning",
    "temporal_reasoning",
    "spatial_reasoning",
    "planning_optimization",
    "coding_generation",
    "code_comprehension",
    "debugging_root_cause",
    "refactoring_under_constraints",
    "test_generation_verification",
    "tool_selection",
    "tool_argument_correctness",
    "multi_tool_sequencing",
    "tool_error_recovery",
    "ambiguity_detection",
    "missing_information_handling",
    "uncertainty_calibration",
    "hallucination_resistance",
    "context_retrieval",
    "context_reasoning",
    "lost_in_middle_resistance",
    "distractor_noise_resistance",
    "contradictory_information_handling",
    "multi_turn_state_tracking",
    "updated_obsolete_state_rejection",
    "memory_compression_summarization_fidelity",
    "decomposition",
    "self_correction",
    "verification_critique",
    "meta_reasoning",
    "prompt_instruction_conflict_handling",
    "format_robustness",
    "adversarial_wording_robustness",
    "sibling_transfer_generalization",
    "composite_agent_tasks",
}
```

For every GPT-OSS case require the existing flat fields plus:

```python
case["capability_map"] == {
    "taxonomy_version": "capability-taxonomy-v1",
    "family_id": <known family>,
    "rubric_version": <non-empty str>,
    "difficulty": {"level": case["difficulty_level"], "dimensions": <dict>},
    "capabilities_required": <non-empty list>,
    "recovery_eligible": <bool>,
    "robustness_eligible": <bool>,
}
```

Also assert one or more anchor cases per family and exact coverage-ledger keys.

- [ ] **Step 2: Commit RED tests**

```bash
git add tests/test_capability_suite.py
git commit -m "test: define GPT-OSS capability suite contract"
```

- [ ] **Step 3: Open/update PR and verify RED in GitHub Actions**

Run the full `python -m pytest` CI matrix. Expected result: failure because `benchmarks/capability-taxonomy-v1.json`, `benchmarks/gpt-oss-20b-capability-v1.json`, and `compute_cost.capability_suite` do not exist yet.

---

### Task 2: Taxonomy Catalog + Extended Template (GREEN data)

**Files:**
- Create: `benchmarks/capability-taxonomy-v1.json`
- Create: `benchmarks/gpt-oss-20b-capability-v1.json`
- Modify: none

**Interfaces:**
- Consumes: existing scorer names from `src/compute_cost/scoring.py`.
- Produces: a 40-family versioned taxonomy and a directly executable breadth suite using only currently supported deterministic scorers.

- [ ] **Step 1: Create taxonomy catalog**

Top-level contract:

```json
{
  "taxonomy_version": "capability-taxonomy-v1",
  "difficulty_scale": {
    "0": "sanity / trivial",
    "1": "basic",
    "2": "routine",
    "3": "moderate",
    "4": "competent",
    "5": "difficult",
    "6": "advanced",
    "7": "expert",
    "8": "adversarial",
    "9": "pathological / edge case",
    "10": "boundary-breaking"
  },
  "families": [
    {
      "id": "instruction_following_constraint_stacking",
      "name": "Instruction following and constraint stacking",
      "rubric_version": "instruction-following-v1",
      "difficulty_dimensions": ["constraint_count", "constraint_interaction", "negative_constraints", "ordering_pressure"]
    }
  ]
}
```

Repeat for all 40 families with family-appropriate difficulty dimensions.

- [ ] **Step 2: Create GPT-OSS breadth suite by extending the old fixture skeleton**

Top-level contract:

```json
{
  "benchmark_version": "gpt-oss-20b-capability-v1",
  "schema_version": "capability-suite-v1",
  "taxonomy_version": "capability-taxonomy-v1",
  "description": "Breadth anchors for adaptive GPT-OSS 20B capability mapping.",
  "coverage": {"<family-id>": "PARTIAL"},
  "cases": []
}
```

Every family gets at least one deterministic anchor. Use only existing scorers: `exact`, `numeric`, `json`, `extraction_set`, `tool_call`, `ambiguity`, `context_retrieval`, `python_function`. Preserve old flat fields and attach `capability_map` metadata. The initial anchors should be cheap L1-L3 probes, not the full ladder.

- [ ] **Step 3: Confirm the old Qwen file was not edited**

Compute SHA-256 of `benchmarks/qwen-characterization-v1.json`; it must equal the global-constraint digest.

---

### Task 3: Capability Suite Validator (GREEN code)

**Files:**
- Create: `src/compute_cost/capability_suite.py`
- Test: `tests/test_capability_suite.py`

**Interfaces:**
- Consumes: Python dictionaries parsed from taxonomy/suite JSON.
- Produces: `validate_capability_suite(suite: dict[str, Any], taxonomy: dict[str, Any]) -> None`; raises `ValueError` with deterministic messages for contract violations.

- [ ] **Step 1: Implement minimal validator required by tests**

Validator checks:

```python
def validate_capability_suite(suite: dict[str, Any], taxonomy: dict[str, Any]) -> None:
    # exact version match
    # taxonomy has unique 40 family ids
    # coverage ledger exactly matches taxonomy ids and uses allowed states
    # case ids unique
    # every family has >=1 case
    # legacy flat fields present and difficulty_level in 0..10
    # capability_map taxonomy/family/rubric/difficulty fields are consistent
    # capabilities_required contains only known family ids
    # recovery_eligible and robustness_eligible are bool
```

Do not add runner orchestration, frontier search, recovery policy, or report generation in this slice.

- [ ] **Step 2: Run targeted contract tests**

```bash
python -m pytest tests/test_capability_suite.py -q
```

Expected: PASS.

- [ ] **Step 3: Run full suite**

```bash
python -m pytest
```

Expected: all existing tests plus new capability-suite tests pass on Python 3.11 and 3.12.

- [ ] **Step 4: Commit implementation**

```bash
git add benchmarks/capability-taxonomy-v1.json benchmarks/gpt-oss-20b-capability-v1.json src/compute_cost/capability_suite.py tests/test_capability_suite.py
git commit -m "feat: add GPT-OSS capability suite template"
```

---

### Task 4: Verification + Handoff to Frontier Search Slice

**Files:**
- Modify: none unless verification reveals a defect.

**Interfaces:**
- Consumes: GitHub Actions status for the feature branch/PR.
- Produces: verified base contract ready for the next implementation slice: adaptive L0-L10 family-local frontier search.

- [ ] **Step 1: Verify GitHub Actions matrix**

Both Windows jobs (`3.11`, `3.12`) must pass the repository's existing `python -m pytest` workflow.

- [ ] **Step 2: Verify invariants from repository contents**

Confirm:

```text
qwen-characterization-v1.json unchanged
40 taxonomy families
40/40 breadth coverage
all cases executable by current scorer interface
no runner behavior changed
no historical schema rewritten
```

- [ ] **Step 3: Record next boundary**

The next slice begins only after this contract is green: add family-local adaptive difficulty search, probabilistic replication thresholds, frontier artifacts, and recovery/robustness scheduling on top of this suite.
