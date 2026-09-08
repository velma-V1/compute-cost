# Qwen Characterization Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `qwen3.5:27b-q4_K_M` scientifically valid to characterize by separating harness/scorer/truncation failures from capability failures, controlling thinking explicitly, adaptively bracketing reasoning budgets, preserving experiment lineage/replays, and producing a verified task operating profile before any model #2 run.

**Architecture:** Preserve the existing evidence-first runner, Ollama adapter, telemetry, manifest, and progress systems. Add a small classification layer, pure adaptive-budget controller, experiment metadata model, and characterization orchestration that reuses the existing runtime/evidence paths. `onboard` remains backward-compatible; a new `characterize` command drives the adaptive path. Characterization repeats the same task under different settings only through unique experiment identities, so raw/scoring/replay artifacts never overwrite one another.

**Tech Stack:** Python 3.11/3.12, standard library, pytest, Ollama `/api/chat`, existing `EvidenceStore`, `BenchmarkRunner`, `ProgressDisplay`, and Windows GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-07-adaptive-qwen-characterization-design.md`

## Global Constraints

- The first real characterization target is exactly `qwen3.5:27b-q4_K_M`; model #2 does not run until Phase 1 produces scientifically valid capability/thinking results.
- Raw runtime requests, every stream chunk, exposed thinking, final content, scorer evidence, telemetry, experiment lineage, replay fixtures, progress events, and derived provenance remain preserved under the existing manifest contract.
- Exposed reasoning is behavioral evidence, not ground-truth hidden cognition.
- Measurements/conclusions are labeled `MEASURED`, `DERIVED`, `ESTIMATED`, or `UNAVAILABLE`; do not fabricate exact thinking-token counts.
- One experimental variable changes at a time except for explicitly labeled interaction tests.
- Test/scorer/harness defects never count as model capability failures.
- Requested context is never reported as actual prompt tokens; runtime `prompt_eval_count` is authoritative when available.
- Adaptive search concentrates calls around PASS/FAIL/truncation boundaries and stops redundant budget scaling when failure semantics no longer implicate insufficient budget.
- Each model invocation is evidence-isolated and replayable. Characterization never batch-runs multiple model names in one invocation.
- No live Ollama model call is used to validate implementation. Deterministic fake-runtime tests must pass first on Windows Python 3.11 and 3.12.
- Existing `onboard`, `verify`, progress behavior, and prior run verification remain backward-compatible.

---

## File Structure

### Create

- `src/compute_cost/classification.py` — deterministic mapping from runtime/scorer evidence to behavioral result class.
- `src/compute_cost/experiments.py` — immutable experiment identity, lineage, request settings, and serialization helpers.
- `src/compute_cost/adaptive.py` — pure reasoning-budget boundary controller with no runtime/file-system dependency.
- `src/compute_cost/characterization.py` — Phase-1 orchestration and task-profile derivation.
- `benchmarks/qwen-characterization-v1.json` — small representative Phase-1 characterization suite.
- `tests/test_classification.py` — result-classification tests, including the exact Test #1 failure shape.
- `tests/test_experiments.py` — experiment identity/lineage tests.
- `tests/test_adaptive.py` — deterministic budget-search/stop-rule tests.
- `tests/test_characterization.py` — fake-runtime end-to-end adaptive characterization tests.

### Modify

- `src/compute_cost/schema.py` — stable result/measurement enums.
- `src/compute_cost/scoring.py` — malformed/empty structured output becomes scored model output evidence, not `SCORER_ERROR`.
- `src/compute_cost/runtimes/ollama.py` — derive observable thinking/content phase timings and chunk counts while retaining raw evidence.
- `src/compute_cost/runner_core.py` — unique evidence keys, explicit request fields, reusable run/preflight helpers, richer replay metadata.
- `src/compute_cost/runner.py` — progress-aware characterization entry point and adaptive plan adjustments.
- `src/compute_cost/config.py` and `config/default.toml` — Phase-1 characterization defaults.
- `src/compute_cost/report.py` — characterization summary/profile rendering without contaminating ordinary benchmark reporting.
- `src/compute_cost/cli.py` — `characterize` command, single-model only.
- `tests/test_schema.py`, `tests/test_scoring.py`, `tests/test_ollama_adapter.py`, `tests/test_runner.py`, `tests/test_runner_progress.py`, `tests/test_cli.py`, `tests/test_report.py` — regression and new-path tests.
- `README.md` — document characterization command and evidence interpretation.

---

### Task 1: Stable Characterization Result Schema

**Files:**
- Modify: `src/compute_cost/schema.py`
- Modify: `tests/test_schema.py`

**Interfaces:**
- Consumes: existing `StageStatus`, `FailureCode`.
- Produces: `ResultClass`, `MeasurementKind` enums used by classification, experiments, profiles, and reports.

- [ ] **Step 1: Write failing enum tests**

Add to `tests/test_schema.py`:

```python
from compute_cost.schema import MeasurementKind, ResultClass


def test_characterization_result_classes_are_stable_strings():
    assert ResultClass.ANSWER_CORRECT.value == "ANSWER_CORRECT"
    assert ResultClass.ANSWER_WRONG.value == "ANSWER_WRONG"
    assert ResultClass.THINK_TRUNCATED.value == "THINK_TRUNCATED"
    assert ResultClass.ANSWER_TRUNCATED.value == "ANSWER_TRUNCATED"
    assert ResultClass.NO_FINAL_ANSWER.value == "NO_FINAL_ANSWER"
    assert ResultClass.FORMAT_FAILURE.value == "FORMAT_FAILURE"
    assert ResultClass.TOOL_FAILURE.value == "TOOL_FAILURE"
    assert ResultClass.CONTEXT_FAILURE.value == "CONTEXT_FAILURE"
    assert ResultClass.SCORER_DEFECT.value == "SCORER_DEFECT"
    assert ResultClass.RUNTIME_FAILURE.value == "RUNTIME_FAILURE"


def test_measurement_kinds_prevent_false_precision():
    assert [kind.value for kind in MeasurementKind] == [
        "MEASURED",
        "DERIVED",
        "ESTIMATED",
        "UNAVAILABLE",
    ]
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_schema.py -v
```

Expected: import failure because `ResultClass` and `MeasurementKind` do not exist.

- [ ] **Step 3: Add the stable enums**

Add to `src/compute_cost/schema.py`:

```python
class ResultClass(str, Enum):
    ANSWER_CORRECT = "ANSWER_CORRECT"
    ANSWER_WRONG = "ANSWER_WRONG"
    THINK_TRUNCATED = "THINK_TRUNCATED"
    ANSWER_TRUNCATED = "ANSWER_TRUNCATED"
    NO_FINAL_ANSWER = "NO_FINAL_ANSWER"
    FORMAT_FAILURE = "FORMAT_FAILURE"
    TOOL_FAILURE = "TOOL_FAILURE"
    CONTEXT_FAILURE = "CONTEXT_FAILURE"
    SELF_CORRECTED = "SELF_CORRECTED"
    REASONING_LOOP = "REASONING_LOOP"
    OVERTHINK_CORRUPTION = "OVERTHINK_CORRUPTION"
    TIMEOUT = "TIMEOUT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    RUNTIME_FAILURE = "RUNTIME_FAILURE"
    TEST_DEFECT = "TEST_DEFECT"
    SCORER_DEFECT = "SCORER_DEFECT"
    CAPTURE_GAP = "CAPTURE_GAP"


class MeasurementKind(str, Enum):
    MEASURED = "MEASURED"
    DERIVED = "DERIVED"
    ESTIMATED = "ESTIMATED"
    UNAVAILABLE = "UNAVAILABLE"
```

- [ ] **Step 4: Run schema tests GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_schema.py -v
```

Expected: all schema tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/compute_cost/schema.py tests/test_schema.py
git commit -m "feat: add characterization result schema"
```

---

### Task 2: Structured-Output Scorer Hardening

**Files:**
- Modify: `src/compute_cost/scoring.py`
- Modify: `tests/test_scoring.py`

**Interfaces:**
- Consumes: existing `score_case(case, response, workspace=None)`.
- Produces: invalid JSON/extraction/tool output returns `status="SCORED"`, `score=0.0`, an explicit failed `valid_json` check, and parse-error evidence. `SCORER_ERROR` is reserved for benchmark implementation defects.

- [ ] **Step 1: Add regression tests for empty/malformed structured output**

Add to `tests/test_scoring.py`:

```python

def test_empty_json_output_is_model_format_failure_not_scorer_error():
    result = score_case(case("json", {"name": "Ada"}, required=["name"]), "")
    assert result["status"] == "SCORED"
    assert result["score"] == 0.0
    assert result["checks"][0] == {"name": "valid_json", "pass": False}
    assert result["evidence"]["parse_error"]["type"] == "JSONDecodeError"


def test_malformed_extraction_and_tool_output_are_scored_failures():
    extraction = score_case(case("extraction_set", ["red", "blue"]), "[red, blue]")
    tool = score_case(case("tool_call", {"tool": "lookup", "arguments": {"id": 7}}), "")
    assert extraction["status"] == "SCORED"
    assert extraction["score"] == 0.0
    assert tool["status"] == "SCORED"
    assert tool["score"] == 0.0
```

- [ ] **Step 2: Run focused scorer tests RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_scoring.py -v
```

Expected: the new cases currently return `SCORER_ERROR`.

- [ ] **Step 3: Add a deterministic JSON parse helper**

Add near the top of `src/compute_cost/scoring.py`:

```python
def _parse_json_response(response: str) -> tuple[Any | None, dict[str, str] | None]:
    try:
        return json.loads(response), None
    except json.JSONDecodeError as exc:
        return None, {"type": "JSONDecodeError", "message": str(exc)}


def _invalid_json_result(response: str, error: dict[str, str]) -> dict[str, Any]:
    return {
        "score": 0.0,
        "status": "SCORED",
        "checks": [{"name": "valid_json", "pass": False}],
        "evidence": {"raw_response": response, "parsed": None, "parse_error": error},
    }
```

Replace direct `json.loads(response)` calls inside the `json`, `extraction_set`, and `tool_call` branches with:

```python
parsed, parse_error = _parse_json_response(response)
if parse_error is not None:
    return _invalid_json_result(response, parse_error)
```

Keep the outer exception handler unchanged so actual scorer bugs still become `SCORER_ERROR`.

- [ ] **Step 4: Run scorer tests GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_scoring.py -v
```

Expected: all scorer tests pass, including valid structured-output cases.

- [ ] **Step 5: Commit**

```powershell
git add src/compute_cost/scoring.py tests/test_scoring.py
git commit -m "fix: distinguish malformed output from scorer defects"
```

---

### Task 3: Observable Thinking/Answer Phase Metrics in Ollama Adapter

**Files:**
- Modify: `src/compute_cost/runtimes/ollama.py`
- Modify: `tests/test_ollama_adapter.py`

**Interfaces:**
- Consumes: `OllamaAdapter.generate(..., request_fields={"think": bool})`, which already preserves the explicit `think` request field.
- Produces: `generation["phase_metrics"]` containing measured stream-event timings/counts only; no fabricated token split.

- [ ] **Step 1: Add a phase-metrics test**

Add to `tests/test_ollama_adapter.py`:

```python

def test_generate_derives_observable_thinking_and_answer_phase_metrics():
    events = [
        b'{"message":{"role":"assistant","thinking":"plan","content":""},"done":false}\n',
        b'{"message":{"role":"assistant","thinking":" more","content":""},"done":false}\n',
        b'{"message":{"role":"assistant","content":"42"},"done":false}\n',
        b'{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","eval_count":5}\n',
    ]
    transport = FakeTransport([
        HttpExchange(200, {}, [chunk(events[0], 110), chunk(events[1], 120), chunk(events[2], 140), chunk(events[3], 150)])
    ])
    adapter = OllamaAdapter("http://127.0.0.1:11434", transport=transport, monotonic_ns=lambda: 100)

    result = adapter.generate(
        "fake",
        [{"role": "user", "content": "solve"}],
        {"num_predict": 64},
        request_fields={"think": True},
    )

    phase = result["phase_metrics"]
    assert phase["thinking_chunks"] == 2
    assert phase["answer_chunks"] == 1
    assert phase["thinking_chars"] == 9
    assert phase["answer_chars"] == 2
    assert phase["time_to_first_thinking_ns"] == 10
    assert phase["time_to_first_answer_ns"] == 40
    assert phase["thinking_span_ns"] == 10
    assert phase["answer_span_ns"] == 0
    assert "thinking_token_count" not in phase
```

- [ ] **Step 2: Run the adapter test RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ollama_adapter.py -v
```

Expected: `phase_metrics` is missing.

- [ ] **Step 3: Derive metrics from retained stream events**

Add a static helper to `OllamaAdapter`:

```python
@staticmethod
def _phase_metrics(stream_events: list[dict[str, Any]], started_ns: int) -> dict[str, Any]:
    thinking_events: list[tuple[int, str]] = []
    answer_events: list[tuple[int, str]] = []
    for event in stream_events:
        received = event.get("received_monotonic_ns")
        parsed = event.get("parsed")
        if not isinstance(received, int) or not isinstance(parsed, dict):
            continue
        message = parsed.get("message")
        if not isinstance(message, dict):
            continue
        thinking = message.get("thinking")
        content = message.get("content")
        if isinstance(thinking, str) and thinking:
            thinking_events.append((received, thinking))
        if isinstance(content, str) and content:
            answer_events.append((received, content))

    def first_latency(items: list[tuple[int, str]]) -> int | None:
        return None if not items else items[0][0] - started_ns

    def span(items: list[tuple[int, str]]) -> int | None:
        return None if not items else items[-1][0] - items[0][0]

    return {
        "measurement_kind": "MEASURED",
        "thinking_chunks": len(thinking_events),
        "answer_chunks": len(answer_events),
        "thinking_chars": sum(len(text) for _, text in thinking_events),
        "answer_chars": sum(len(text) for _, text in answer_events),
        "time_to_first_thinking_ns": first_latency(thinking_events),
        "time_to_first_answer_ns": first_latency(answer_events),
        "thinking_span_ns": span(thinking_events),
        "answer_span_ns": span(answer_events),
    }
```

After normalized output and aggregate runtime metrics are created in `generate`, add:

```python
envelope["phase_metrics"] = self._phase_metrics(envelope["stream_events"], envelope["request"]["started_monotonic_ns"])
```

Do not infer per-phase token counts from `eval_count`.

- [ ] **Step 4: Run adapter tests GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ollama_adapter.py -v
```

Expected: all adapter tests pass and the existing raw/unknown-field preservation test remains green.

- [ ] **Step 5: Commit**

```powershell
git add src/compute_cost/runtimes/ollama.py tests/test_ollama_adapter.py
git commit -m "feat: measure observable thinking and answer phases"
```

---

### Task 4: Deterministic Failure/Behavior Classification

**Files:**
- Create: `src/compute_cost/classification.py`
- Create: `tests/test_classification.py`

**Interfaces:**
- Consumes: a case dict, Ollama generation envelope, and scorer result.
- Produces: `classify_result(case, generation, scoring) -> dict[str, Any]` with `result_class`, `basis`, and `measurement_kind`.

- [ ] **Step 1: Write the exact Test #1 regression and neighboring cases**

Create `tests/test_classification.py`:

```python
from compute_cost.classification import classify_result


def generation(*, text="", thinking="", done_reason="stop", ok=True):
    return {
        "ok": ok,
        "normalized": {"text": text, "thinking": thinking, "done_reason": done_reason},
        "metrics": {"eval_count": 16},
    }


def test_thinking_consumes_budget_without_final_answer_is_think_truncated():
    result = classify_result(
        {"scorer": "exact"},
        generation(thinking="Thinking Process: analyze request", done_reason="length"),
        {"status": "SCORED", "score": 0.0, "checks": []},
    )
    assert result["result_class"] == "THINK_TRUNCATED"
    assert result["basis"] == "length stop with exposed thinking and no final content"


def test_correct_answer_wins_even_if_runtime_reports_length_stop():
    result = classify_result(
        {"scorer": "exact"},
        generation(text="BLUE", thinking="plan", done_reason="length"),
        {"status": "SCORED", "score": 1.0, "checks": []},
    )
    assert result["result_class"] == "ANSWER_CORRECT"


def test_invalid_json_is_format_failure_and_scorer_error_is_scorer_defect():
    malformed = classify_result(
        {"scorer": "json"},
        generation(text="not json"),
        {"status": "SCORED", "score": 0.0, "checks": [{"name": "valid_json", "pass": False}]},
    )
    defect = classify_result(
        {"scorer": "json"},
        generation(text="{}"),
        {"status": "SCORER_ERROR", "score": None, "checks": []},
    )
    assert malformed["result_class"] == "FORMAT_FAILURE"
    assert defect["result_class"] == "SCORER_DEFECT"
```

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_classification.py -v
```

Expected: module import fails.

- [ ] **Step 3: Implement precedence rules**

Create `src/compute_cost/classification.py`:

```python
from __future__ import annotations

from typing import Any

from .schema import MeasurementKind, ResultClass


def classify_result(case: dict[str, Any], generation: dict[str, Any], scoring: dict[str, Any]) -> dict[str, Any]:
    normalized = generation.get("normalized") or {}
    text = str(normalized.get("text") or "")
    thinking = str(normalized.get("thinking") or "")
    done_reason = normalized.get("done_reason")
    scorer = str(case.get("scorer") or "")

    if not generation.get("ok", False):
        result = ResultClass.RUNTIME_FAILURE
        basis = "runtime generation failed"
    elif scoring.get("status") == "SCORER_ERROR":
        result = ResultClass.SCORER_DEFECT
        basis = "scorer reported benchmark defect"
    elif scoring.get("score") == 1.0:
        result = ResultClass.ANSWER_CORRECT
        basis = "deterministic scorer passed"
    elif done_reason == "length" and thinking and not text:
        result = ResultClass.THINK_TRUNCATED
        basis = "length stop with exposed thinking and no final content"
    elif done_reason == "length" and text:
        result = ResultClass.ANSWER_TRUNCATED
        basis = "length stop after final content began"
    elif not text:
        result = ResultClass.NO_FINAL_ANSWER
        basis = "runtime succeeded but emitted no final content"
    elif any(check.get("name") == "valid_json" and check.get("pass") is False for check in scoring.get("checks", [])):
        result = ResultClass.FORMAT_FAILURE
        basis = "structured output was not valid JSON"
    elif scorer == "tool_call":
        result = ResultClass.TOOL_FAILURE
        basis = "tool-call scorer failed"
    elif scorer == "context_retrieval":
        result = ResultClass.CONTEXT_FAILURE
        basis = "context-retrieval scorer failed"
    else:
        result = ResultClass.ANSWER_WRONG
        basis = "runtime completed and deterministic scorer failed"

    return {
        "result_class": result.value,
        "basis": basis,
        "measurement_kind": MeasurementKind.DERIVED.value,
    }
```

- [ ] **Step 4: Run GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_classification.py -v
```

Expected: all classification tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/compute_cost/classification.py tests/test_classification.py
git commit -m "feat: classify model and harness failure modes"
```

---

### Task 5: Experiment Identity, Lineage, and One-Variable Contract

**Files:**
- Create: `src/compute_cost/experiments.py`
- Create: `tests/test_experiments.py`

**Interfaces:**
- Produces: `ExperimentSpec`, `make_experiment_id`, `changed_fields(parent, child)`, and serialization used by the controller/runner.
- Contract: child experiments outside an explicit interaction test differ in exactly one controlled field among `thinking_mode`, `generation_budget`, `context_request`, `temperature`, `seed`, and `prompt_variant`.

- [ ] **Step 1: Write identity/lineage tests**

Create `tests/test_experiments.py`:

```python
from compute_cost.experiments import ExperimentSpec, changed_fields, make_experiment_id


def base_spec(**overrides):
    values = dict(
        experiment_id="exp-root",
        parent_experiment_id=None,
        task_id="math-001",
        task_family="reasoning_math",
        difficulty_level=3,
        hypothesis="thinking baseline",
        changed_variable="baseline",
        thinking_mode=True,
        generation_budget=256,
        context_request=None,
        temperature=0.0,
        seed=42,
        prompt_variant="base",
        recovery_level=None,
    )
    values.update(overrides)
    return ExperimentSpec(**values)


def test_experiment_id_is_stable_for_run_sequence_and_task():
    assert make_experiment_id(7, "math-001", "think-on-256") == "exp-000007-math-001-think-on-256"


def test_child_budget_probe_changes_exactly_one_controlled_field():
    parent = base_spec()
    child = base_spec(
        experiment_id="exp-child",
        parent_experiment_id="exp-root",
        hypothesis="find lower passing budget",
        changed_variable="generation_budget",
        generation_budget=192,
    )
    assert changed_fields(parent, child) == ["generation_budget"]
    assert child.to_dict()["parent_experiment_id"] == "exp-root"
```

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_experiments.py -v
```

Expected: module import fails.

- [ ] **Step 3: Implement immutable experiment metadata**

Create `src/compute_cost/experiments.py`:

```python
from __future__ import annotations

import re
from dataclasses import asdict, dataclass


CONTROLLED_FIELDS = (
    "thinking_mode",
    "generation_budget",
    "context_request",
    "temperature",
    "seed",
    "prompt_variant",
)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "x"


def make_experiment_id(sequence: int, task_id: str, label: str) -> str:
    return f"exp-{sequence:06d}-{_safe(task_id)}-{_safe(label)}"


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    parent_experiment_id: str | None
    task_id: str
    task_family: str
    difficulty_level: int
    hypothesis: str
    changed_variable: str
    thinking_mode: bool
    generation_budget: int
    context_request: int | None
    temperature: float
    seed: int
    prompt_variant: str
    recovery_level: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def changed_fields(parent: ExperimentSpec, child: ExperimentSpec) -> list[str]:
    return [name for name in CONTROLLED_FIELDS if getattr(parent, name) != getattr(child, name)]
```

- [ ] **Step 4: Run GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_experiments.py -v
```

Expected: all experiment tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/compute_cost/experiments.py tests/test_experiments.py
git commit -m "feat: add experiment identity and lineage"
```

---

### Task 6: Pure Adaptive Thought-Budget Controller

**Files:**
- Create: `src/compute_cost/adaptive.py`
- Create: `tests/test_adaptive.py`

**Interfaces:**
- Consumes: ordered observations `(budget, result_class)` for thinking-ON attempts.
- Produces: `BudgetDecision(action, budget, reason)` where action is `PROBE`, `REPLICATE`, or `STOP`.
- Phase-1 semantics: truncation may justify increasing budget; persistent semantic failure without a known PASS does not trigger unbounded budget escalation; once a FAIL/PASS bracket exists, bisect to configured granularity and replicate the minimum PASS boundary.

- [ ] **Step 1: Write adaptive boundary tests**

Create `tests/test_adaptive.py`:

```python
from compute_cost.adaptive import AdaptiveBudgetController, BudgetObservation


def obs(budget, result):
    return BudgetObservation(budget=budget, result_class=result)


def test_pass_256_fail_128_brackets_with_192_then_160():
    controller = AdaptiveBudgetController(initial_budget=256, min_budget=32, max_budget=2048, granularity=32, boundary_repeats=3)
    assert controller.next([obs(256, "ANSWER_CORRECT")]).budget == 128
    assert controller.next([obs(256, "ANSWER_CORRECT"), obs(128, "ANSWER_WRONG")]).budget == 192
    assert controller.next([obs(256, "ANSWER_CORRECT"), obs(128, "ANSWER_WRONG"), obs(192, "ANSWER_CORRECT")]).budget == 160


def test_think_truncation_scales_up_but_semantic_failure_does_not_blindly_scale():
    controller = AdaptiveBudgetController(initial_budget=256, min_budget=32, max_budget=2048, granularity=32, boundary_repeats=3)
    assert controller.next([obs(256, "THINK_TRUNCATED")]).budget == 512
    decision = controller.next([obs(256, "ANSWER_WRONG")])
    assert decision.action == "STOP"
    assert decision.reason == "semantic failure without evidence that more budget will help"


def test_boundary_is_replicated_before_stop():
    controller = AdaptiveBudgetController(initial_budget=256, min_budget=32, max_budget=2048, granularity=32, boundary_repeats=3)
    observations = [
        obs(128, "ANSWER_WRONG"),
        obs(160, "ANSWER_WRONG"),
        obs(192, "ANSWER_CORRECT"),
    ]
    assert controller.next(observations).action == "REPLICATE"
    observations.extend([obs(192, "ANSWER_CORRECT"), obs(192, "ANSWER_CORRECT")])
    decision = controller.next(observations)
    assert decision.action == "STOP"
    assert decision.reason == "minimum passing boundary reproduced"
```

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_adaptive.py -v
```

Expected: module import fails.

- [ ] **Step 3: Implement deterministic controller**

Create `src/compute_cost/adaptive.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


PASS = {"ANSWER_CORRECT"}
TRUNCATION = {"THINK_TRUNCATED", "ANSWER_TRUNCATED", "NO_FINAL_ANSWER"}


@dataclass(frozen=True)
class BudgetObservation:
    budget: int
    result_class: str


@dataclass(frozen=True)
class BudgetDecision:
    action: str
    budget: int | None
    reason: str


class AdaptiveBudgetController:
    def __init__(self, *, initial_budget: int, min_budget: int, max_budget: int, granularity: int, boundary_repeats: int):
        self.initial_budget = initial_budget
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.granularity = granularity
        self.boundary_repeats = boundary_repeats

    def _round(self, value: int) -> int:
        return max(self.min_budget, min(self.max_budget, (value // self.granularity) * self.granularity))

    def next(self, observations: list[BudgetObservation]) -> BudgetDecision:
        if not observations:
            return BudgetDecision("PROBE", self.initial_budget, "establish thinking-on baseline")

        passes = [item for item in observations if item.result_class in PASS]
        failures = [item for item in observations if item.result_class not in PASS]

        if not passes:
            latest = observations[-1]
            if latest.result_class in TRUNCATION and latest.budget < self.max_budget:
                return BudgetDecision("PROBE", min(self.max_budget, latest.budget * 2), "truncation justifies more generation headroom")
            return BudgetDecision("STOP", None, "semantic failure without evidence that more budget will help")

        minimum_pass = min(item.budget for item in passes)
        lower_failures = [item.budget for item in failures if item.budget < minimum_pass]
        if not lower_failures:
            candidate = self._round(minimum_pass // 2)
            if candidate >= minimum_pass or candidate < self.min_budget:
                candidate = self.min_budget
            if candidate != minimum_pass and not any(item.budget == candidate for item in observations):
                return BudgetDecision("PROBE", candidate, "search below current passing budget")

        lower_fail = max(lower_failures) if lower_failures else self.min_budget - self.granularity
        width = minimum_pass - lower_fail
        if width > self.granularity:
            candidate = self._round(lower_fail + width // 2)
            if candidate <= lower_fail:
                candidate = lower_fail + self.granularity
            if candidate >= minimum_pass:
                candidate = minimum_pass - self.granularity
            return BudgetDecision("PROBE", candidate, "bisect fail/pass budget bracket")

        pass_count = sum(1 for item in observations if item.budget == minimum_pass and item.result_class in PASS)
        if pass_count < self.boundary_repeats:
            return BudgetDecision("REPLICATE", minimum_pass, "reproduce minimum passing boundary")
        return BudgetDecision("STOP", None, "minimum passing boundary reproduced")
```

- [ ] **Step 4: Run adaptive tests GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_adaptive.py -v
```

Expected: all adaptive-controller tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/compute_cost/adaptive.py tests/test_adaptive.py
git commit -m "feat: add adaptive thought budget controller"
```

---

### Task 7: Make Core Runner Reusable and Evidence-Unique for Repeated Experiments

**Files:**
- Modify: `src/compute_cost/runner_core.py`
- Modify: `src/compute_cost/runner.py`
- Modify: `tests/test_runner.py`
- Modify: `tests/test_runner_progress.py`

**Interfaces:**
- Extend `_execute_case(..., request_fields=None, evidence_key=None)` and `run_case(..., request_fields=None, evidence_key=None, experiment=None)`.
- Add `_begin_run(model) -> EvidenceStore` and `_preflight_model(model, pull=False) -> bool` shared by `onboard` and characterization.
- Existing calls without new parameters remain byte/path compatible where practical.

- [ ] **Step 1: Write a regression test proving repeated case IDs do not overwrite evidence**

Add to `tests/test_runner.py`:

```python

def test_repeated_case_can_use_unique_evidence_keys(tmp_path: Path):
    runner = BenchmarkRunner(FakeRuntime(), config(), suite(), results_root=tmp_path, telemetry=FakeTelemetry())
    runner.model = "fake"
    runner.store = EvidenceStore(tmp_path, "repeat-evidence")
    case = suite()["cases"][0]

    runner.run_case(case, stage="characterize", evidence_key="exp-000001", request_fields={"think": False})
    runner.run_case(case, stage="characterize", evidence_key="exp-000002", request_fields={"think": True})

    scoring = sorted(path.name for path in (runner.store.run_dir / "raw/scoring").glob("*.json"))
    assert scoring == ["characterize-exp-000001.json", "characterize-exp-000002.json"]
    assert runner.runtime.calls[-2]["request_fields"] == {"think": False}
    assert runner.runtime.calls[-1]["request_fields"] == {"think": True}
```

- [ ] **Step 2: Run focused runner test RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_runner.py::test_repeated_case_can_use_unique_evidence_keys -v
```

Expected: `run_case` rejects `evidence_key`/`request_fields`.

- [ ] **Step 3: Thread explicit request fields and evidence keys through the core**

Change the internal call shape in `runner_core.py` so `_execute_case` accepts:

```python
request_fields: dict[str, Any] | None = None,
evidence_key: str | None = None,
```

Pass `request_fields` to `_invoke_generation`. Use:

```python
artifact_key = evidence_key or str(case["id"])
```

for `_next_request_id`, `_persist_scoring`, and replay naming while keeping `record["case_id"]` as the original task ID. Add to each case record:

```python
"evidence_key": artifact_key,
"classification": classify_result(case, generation, scoring),
```

Import `classify_result` from `compute_cost.classification`.

Change `_persist_scoring` to accept `evidence_key` and build the safe path from it rather than the logical task ID.

- [ ] **Step 4: Extract reusable run-start/preflight helpers without changing onboarding semantics**

Move the existing run-ID/store/config/suite/event/telemetry initialization into:

```python
def _begin_run(self, model: str) -> EvidenceStore:
    self.model = model
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    self.store = EvidenceStore(self.results_root, run_id)
    self.store.write_json("resolved-config.json", self.config, producer="runner", stage="preflight")
    self.store.write_json("benchmark-snapshot.json", self.suite, producer="runner", stage="preflight")
    self._event("RUN_START", model=model, benchmark_version=self.suite.get("benchmark_version"))
    self._start_telemetry()
    return self.store
```

Move hardware/version/tags/pull/show/runtime snapshot logic into:

```python
def _preflight_model(self, model: str, *, pull: bool = False) -> bool:
    # implementation is the existing preflight block moved without semantic changes
```

The implementation must return `False` after persisting `RUN_FAILED: MODEL_NOT_FOUND`; otherwise return `True` after `PREFLIGHT_COMPLETE`.

Change `onboard` to call `_begin_run`, `_preflight_model`, then the existing cold/warm/base/context/sustained sequence.

- [ ] **Step 5: Preserve progress-wrapper signatures**

Update `src/compute_cost/runner.py::_execute_case` with the same new optional parameters and pass them through to `super()._execute_case`. Do not alter ordinary onboarding progress accounting in this task.

- [ ] **Step 6: Run runner/progress regression suite GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_runner.py tests/test_runner_progress.py -v
```

Expected: all existing onboarding/replay/progress tests and the new evidence-isolation test pass.

- [ ] **Step 7: Commit**

```powershell
git add src/compute_cost/runner_core.py src/compute_cost/runner.py tests/test_runner.py tests/test_runner_progress.py
git commit -m "refactor: isolate repeated experiment evidence"
```

---

### Task 8: Phase-1 Characterization Suite and Configuration

**Files:**
- Create: `benchmarks/qwen-characterization-v1.json`
- Modify: `config/default.toml`
- Modify: `src/compute_cost/config.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_benchmark_suite.py`

**Interfaces:**
- Produces: three representative deterministic Phase-1 tasks and validated controller defaults.

- [ ] **Step 1: Add config/suite tests**

Add assertions that the default config exposes:

```python
characterization = config["characterization"]
assert characterization["initial_think_budget"] == 256
assert characterization["min_think_budget"] == 32
assert characterization["max_think_budget"] == 2048
assert characterization["budget_granularity"] == 32
assert characterization["boundary_repeats"] == 3
assert characterization["max_experiments_per_task"] == 12
```

Add a benchmark-suite test that loads `benchmarks/qwen-characterization-v1.json` and asserts exact task IDs:

```python
assert [case["id"] for case in suite["cases"]] == [
    "char-if-001",
    "char-json-001",
    "char-math-001",
]
```

- [ ] **Step 2: Run config/suite tests RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_benchmark_suite.py -v
```

Expected: characterization config/file is missing.

- [ ] **Step 3: Add deterministic Phase-1 suite**

Create `benchmarks/qwen-characterization-v1.json`:

```json
{
  "benchmark_version": "qwen-characterization-v1",
  "description": "Representative deterministic tasks for adaptive thinking characterization.",
  "cases": [
    {
      "id": "char-if-001",
      "category": "instruction_following",
      "difficulty_level": 1,
      "prompt": "Reply with exactly: BLUE",
      "scorer": "exact",
      "expected": "BLUE",
      "timeout_s": 120
    },
    {
      "id": "char-json-001",
      "category": "structured_output",
      "difficulty_level": 2,
      "prompt": "Return only valid JSON with name set to Ada and count set to 3. No markdown.",
      "scorer": "json",
      "expected": {"name": "Ada", "count": 3},
      "required": ["name", "count"],
      "timeout_s": 120
    },
    {
      "id": "char-math-001",
      "category": "reasoning_math",
      "difficulty_level": 3,
      "prompt": "A 240 GB dataset is reduced by 25 percent, then 20 GB is added. What is the final size in GB? Give the number.",
      "scorer": "numeric",
      "expected": 200,
      "tolerance": 0,
      "timeout_s": 120
    }
  ]
}
```

No fixed `max_output_tokens` belongs in this suite; the characterization controller owns the experimental budget.

- [ ] **Step 4: Add default config**

Append to `config/default.toml`:

```toml
[characterization]
initial_think_budget = 256
min_think_budget = 32
max_think_budget = 2048
budget_granularity = 32
boundary_repeats = 3
max_experiments_per_task = 12
think_off_budget = 256
```

Update config validation so all values are positive integers and `min_think_budget <= initial_think_budget <= max_think_budget`.

- [ ] **Step 5: Run config/suite tests GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_benchmark_suite.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add benchmarks/qwen-characterization-v1.json config/default.toml src/compute_cost/config.py tests/test_config.py tests/test_benchmark_suite.py
git commit -m "feat: add phase one characterization suite"
```

---

### Task 9: Characterization Orchestrator with OFF/ON Baselines and Adaptive Bracketing

**Files:**
- Create: `src/compute_cost/characterization.py`
- Create: `tests/test_characterization.py`
- Modify: `src/compute_cost/runner_core.py`
- Modify: `src/compute_cost/runner.py`

**Interfaces:**
- Produces: `run_characterization(runner, cases) -> list[dict]` and `BenchmarkRunner.characterize(model, pull=False) -> Path`.
- Evidence: `experiments.jsonl` contains one row per model call with immutable experiment metadata, classification, scoring summary, runtime metrics, phase metrics, and evidence key.

- [ ] **Step 1: Build a deterministic fake runtime that makes a known budget boundary**

Create `tests/test_characterization.py` with a fake runtime whose `generate` method returns:

- think OFF: wrong answer;
- think ON with `num_predict < 192`: thinking-only `done_reason="length"`;
- think ON with `num_predict >= 192`: correct final answer;
- all control/preflight methods return valid envelopes.

Use the existing `tests/test_runner.py` envelope shape, but set `normalized["thinking"]`, `normalized["done_reason"]`, and `phase_metrics` explicitly.

- [ ] **Step 2: Write end-to-end assertions before implementation**

The test must assert:

```python
run_dir = runner.characterize("fake")
rows = [json.loads(line) for line in (run_dir / "experiments.jsonl").read_text().splitlines()]

assert rows[0]["experiment"]["thinking_mode"] is False
assert rows[1]["experiment"]["thinking_mode"] is True
assert any(row["classification"]["result_class"] == "THINK_TRUNCATED" for row in rows)
assert any(row["experiment"]["generation_budget"] == 192 and row["classification"]["result_class"] == "ANSWER_CORRECT" for row in rows)
assert sum(1 for row in rows if row["experiment"]["generation_budget"] == 192 and row["classification"]["result_class"] == "ANSWER_CORRECT") >= 3
assert EvidenceStore(tmp_path, run_dir.name).verify_manifest() == []
```

Also assert every `evidence_key` is unique.

- [ ] **Step 3: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_characterization.py -v
```

Expected: `BenchmarkRunner.characterize` does not exist.

- [ ] **Step 4: Implement one experiment execution helper**

In `src/compute_cost/characterization.py`, add a helper that:

1. creates `ExperimentSpec`;
2. validates a non-baseline child changes exactly the declared variable;
3. calls `runner.run_case(..., stage="characterize", append=False, evidence_key=experiment_id, request_fields={"think": thinking_mode}, option_overrides={"num_predict": generation_budget, "temperature": temperature, "seed": seed})`;
4. obtains the stored classification from the record;
5. appends a row to `experiments.jsonl` containing:

```python
{
    "experiment": spec.to_dict(),
    "classification": record["classification"],
    "score": record["score"],
    "status": record["status"],
    "metrics": record["metrics"],
    "timing": record["timing"],
    "phase_metrics": record.get("phase_metrics", {}),
    "evidence_key": record["evidence_key"],
}
```

Ensure `_execute_case` copies `generation.get("phase_metrics")` into the record.

- [ ] **Step 5: Implement task sequence**

For each case:

1. run think-OFF baseline at `think_off_budget` with `changed_variable="baseline"`;
2. create `AdaptiveBudgetController`;
3. run think-ON baseline at the controller's first requested budget;
4. feed each observed `result_class` back to the controller;
5. execute `PROBE`/`REPLICATE` decisions until `STOP` or `max_experiments_per_task`;
6. for every non-root experiment, set `parent_experiment_id` to the immediately preceding experiment and `changed_variable="generation_budget"` unless the transition is OFF→ON baseline, which is labeled `thinking_mode`;
7. if the maximum experiment count is reached, persist `stop_reason="MAX_EXPERIMENTS_PER_TASK"` rather than silently stopping.

- [ ] **Step 6: Add a core/public runner `characterize` lifecycle**

In `runner_core.py`, add `characterize(model, pull=False)` that:

```python
self._begin_run(model)
if not self._preflight_model(model, pull=pull):
    return self._finalize_run()
run_characterization(self, self.suite.get("cases", []) or [])
self._event("CHARACTERIZATION_COMPLETE", model=model)
return self._finalize_run()
```

In `runner.py`, wrap the method with `ProgressDisplay`. The initial denominator is:

```python
1 + len(cases) * 2 + 1
```

(preflight + minimum OFF/ON per case + finalize). Whenever the adaptive controller adds another probe/replication, increment `progress.total_tasks` and persist `plan_adjusted` with reason `adaptive experiment added`. Do not claim a fixed total for an adaptive run.

- [ ] **Step 7: Run characterization and progress tests GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_characterization.py tests/test_runner_progress.py -v
```

Expected: deterministic boundary discovery, repeat confirmation, unique evidence, progress plan adjustments, and verified manifest all pass.

- [ ] **Step 8: Commit**

```powershell
git add src/compute_cost/characterization.py src/compute_cost/runner_core.py src/compute_cost/runner.py tests/test_characterization.py tests/test_runner_progress.py
git commit -m "feat: add adaptive Qwen characterization runner"
```

---

### Task 10: Replay Fixtures Carry Experiment Classification and Recovery Lineage

**Files:**
- Modify: `src/compute_cost/runner_core.py`
- Modify: `tests/test_runner.py`
- Modify: `tests/test_characterization.py`

**Interfaces:**
- Replay schema version becomes `2` for new snapshots.
- New fields: `experiment`, `classification`, `evidence_key`, and `recovery` metadata while retaining all v1 fields.

- [ ] **Step 1: Write replay-v2 assertions**

For a truncation experiment, assert the replay JSON contains:

```python
assert replay["schema_version"] == 2
assert replay["experiment"]["experiment_id"].startswith("exp-")
assert replay["classification"]["result_class"] == "THINK_TRUNCATED"
assert replay["evidence_key"] == replay["experiment"]["experiment_id"]
assert replay["generation"]["normalized"]["thinking"]
assert replay["invocation"]["request_fields"] == {"think": True}
```

For the later successful higher-budget child, assert its experiment row names the truncation experiment as `parent_experiment_id` and uses `recovery_level="R1"` when the controller raised budget due to truncation.

- [ ] **Step 2: Run replay tests RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_runner.py tests/test_characterization.py -v
```

Expected: replay metadata fields are missing.

- [ ] **Step 3: Extend `_write_replay` without dropping old evidence**

Add optional parameters:

```python
experiment: dict[str, Any] | None = None,
classification: dict[str, Any] | None = None,
evidence_key: str | None = None,
```

Write:

```python
"schema_version": 2,
"experiment": copy.deepcopy(experiment),
"classification": copy.deepcopy(classification),
"evidence_key": evidence_key,
```

alongside the existing exact case/invocation/generation/scoring/telemetry/config snapshot.

Create replay for any characterization result whose class is not `ANSWER_CORRECT`; do not create a failure replay for `SCORER_DEFECT` as though it were a model failure. Instead write it under `replay/harness/` or retain the experiment evidence and classify it as harness-invalid.

- [ ] **Step 4: Mark budget recovery lineage**

When a prior observation is `THINK_TRUNCATED`, `ANSWER_TRUNCATED`, or `NO_FINAL_ANSWER` and the next controller action increases budget, set:

```python
recovery_level="R1"
hypothesis="insufficient generation headroom caused truncation"
changed_variable="generation_budget"
```

This turns the first successful higher-budget attempt into an explicitly testable recovery rather than an anonymous retry.

- [ ] **Step 5: Run replay/characterization tests GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_runner.py tests/test_characterization.py -v
```

Expected: old replay behavior remains readable and new fixtures include classification/lineage.

- [ ] **Step 6: Commit**

```powershell
git add src/compute_cost/runner_core.py tests/test_runner.py tests/test_characterization.py
git commit -m "feat: retain characterization replay lineage"
```

---

### Task 11: Task Operating Profile and Characterization Report

**Files:**
- Modify: `src/compute_cost/characterization.py`
- Modify: `src/compute_cost/report.py`
- Modify: `tests/test_characterization.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Produces: `characterization-summary.json` and `characterization-report.md` before manifest finalization.
- Profile contains think-OFF result, observed budget curve, minimum reproduced passing budget when known, transition bracket, failure classes, and measured runtime/resource references.

- [ ] **Step 1: Write profile derivation tests**

Given observations containing `128 FAIL`, `160 FAIL`, and three `192 PASS` rows, assert:

```python
profile = build_task_profile("char-math-001", rows)
assert profile["minimum_reproduced_pass_budget"] == {"value": 192, "kind": "DERIVED"}
assert profile["transition_bracket"] == {"lower_fail": 160, "upper_pass": 192, "kind": "DERIVED"}
assert profile["think_off"]["result_class"] == "ANSWER_WRONG"
assert profile["observations"][0]["generation_budget"] is not None
```

The report test must assert the Markdown contains the labels `MEASURED`/`DERIVED` and never prints a fabricated `thinking tokens` field.

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_characterization.py tests/test_report.py -v
```

Expected: profile/report functions are missing.

- [ ] **Step 3: Implement deterministic task-profile derivation**

Add `build_task_profile(task_id, rows)` in `characterization.py`. It must:

- separate think-OFF and think-ON observations;
- sort think-ON observations by generation budget and experiment order;
- find the highest observed failing budget below the minimum budget with at least `boundary_repeats` correct observations;
- include exact `metrics`, `timing`, and `phase_metrics` as measured observations rather than collapsing them into invented precision;
- return `None` for unknown boundaries rather than guessing.

- [ ] **Step 4: Persist summary/report before manifest finalization**

After all characterization tasks complete and before `_finalize_run`, write:

```python
runner.store.write_json(
    "characterization-summary.json",
    summary,
    producer="characterization",
    stage="report",
)
runner.store.write_raw(
    "characterization-report.md",
    render_characterization_report(summary),
    producer="characterization",
    stage="report",
    media_type="text/markdown",
)
```

The report should state that exposed thinking is an observable trace and that aggregate `eval_count` is not presented as exact per-phase token counts.

- [ ] **Step 5: Run profile/report tests GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_characterization.py tests/test_report.py -v
```

Expected: profile boundaries and provenance labels are deterministic.

- [ ] **Step 6: Commit**

```powershell
git add src/compute_cost/characterization.py src/compute_cost/report.py tests/test_characterization.py tests/test_report.py
git commit -m "feat: build evidence backed Qwen operating profiles"
```

---

### Task 12: CLI `characterize` Command and Single-Model Contract

**Files:**
- Modify: `src/compute_cost/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- New command: `compute-cost characterize --model MODEL [--pull] [--suite PATH]`.
- Default suite for `characterize` is `benchmarks/qwen-characterization-v1.json`; ordinary `onboard` keeps its existing suite behavior.
- Output remains JSON with `run_id` and `run_dir`.

- [ ] **Step 1: Write CLI tests**

Add a fake runner and parser test asserting:

```python
args = parser.parse_args(["characterize", "--model", "fake"])
assert args.command == "characterize"
assert args.model == "fake"
```

Add an execution test that monkeypatches the runner and asserts exactly one model string reaches `runner.characterize("fake", pull=False)` and the printed JSON contains `run_id`/`run_dir`.

- [ ] **Step 2: Run CLI tests RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cli.py -v
```

Expected: `characterize` is not a recognized subcommand.

- [ ] **Step 3: Add the command without model batching**

Add parser arguments:

```python
characterize = subparsers.add_parser("characterize")
characterize.add_argument("--model", required=True)
characterize.add_argument("--pull", action="store_true")
characterize.add_argument("--suite", default="benchmarks/qwen-characterization-v1.json")
```

Dispatch to:

```python
run_dir = runner.characterize(args.model, pull=args.pull)
print(json.dumps({"run_id": run_dir.name, "run_dir": str(run_dir)}, indent=2))
```

Keep the parameter singular; do not introduce `--models`.

- [ ] **Step 4: Run CLI tests GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cli.py -v
```

Expected: all CLI commands remain green.

- [ ] **Step 5: Commit**

```powershell
git add src/compute_cost/cli.py tests/test_cli.py
git commit -m "feat: add single model characterize command"
```

---

### Task 13: Full Fake-Runtime Acceptance Test for the Test #1 Failure Mechanism

**Files:**
- Modify: `tests/test_characterization.py`
- Modify: `tests/test_runner_progress.py`

**Interfaces:**
- This is the no-model-call acceptance gate before Qwen is allowed to run again.

- [ ] **Step 1: Add a fake reproduction of run `20260907-215638-6ee38d88`**

The fake runtime must return for a 16-token thinking-ON probe:

```python
{
    "normalized": {
        "text": "",
        "thinking": "Thinking Process:\n\n1. Analyze the Request",
        "done": True,
        "done_reason": "length",
    },
    "metrics": {"eval_count": 16},
}
```

Assert:

```python
assert row["classification"]["result_class"] == "THINK_TRUNCATED"
assert row["score"] == 0.0
assert row["classification"]["result_class"] != "ANSWER_WRONG"
```

Then return correct output at the higher budget and assert the controller records an `R1` recovery and reproduces the minimum passing boundary.

- [ ] **Step 2: Assert scorer-defect quarantine**

Inject one fake scorer result with `status="SCORER_ERROR"`. Assert:

```python
assert row["classification"]["result_class"] == "SCORER_DEFECT"
assert row["valid_for_capability"] is False
```

and that it is excluded from pass-rate/boundary calculations.

- [ ] **Step 3: Assert adaptive progress denominator changes are retained**

Read `progress.jsonl` and assert at least one row has:

```python
assert event["event"] == "plan_adjusted"
assert event["reason"] == "adaptive experiment added"
```

and the final progress snapshot is exactly 100%.

- [ ] **Step 4: Run the entire deterministic suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: full suite passes locally without contacting Ollama.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_characterization.py tests/test_runner_progress.py
git commit -m "test: prove Qwen truncation is not capability failure"
```

---

### Task 14: README, Evidence Interpretation, and Windows CI Gate

**Files:**
- Modify: `README.md`
- Existing: `.github/workflows/test.yml`

**Interfaces:**
- Documents exactly how to run Phase 1 and how to interpret thought/capability results.

- [ ] **Step 1: Document the new command**

Add a section containing:

```powershell
.\.venv\Scripts\compute-cost.exe characterize --model "qwen3.5:27b-q4_K_M"
```

Document that:

- thinking and final content are preserved separately when Ollama exposes them;
- `eval_count` is aggregate runtime generation evidence, not claimed as an exact thinking-token count;
- `THINK_TRUNCATED` is not a semantic capability failure;
- malformed final JSON is a model-format failure while `SCORER_DEFECT` is a harness defect;
- adaptive runs can change their progress denominator as new boundary probes are scheduled;
- raw machine metadata remains local/gitignored and may contain host/runtime paths.

- [ ] **Step 2: Run the full local test suite again**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit documentation**

```powershell
git add README.md
git commit -m "docs: explain adaptive Qwen characterization"
```

- [ ] **Step 4: Push the implementation branch and inspect fresh Windows CI**

Push the implementation branch. Confirm the workflow runs against the exact final head SHA and both configured Python jobs (3.11 and 3.12) succeed.

- [ ] **Step 5: Verify no evidence/privacy artifacts entered Git**

Run repository checks for:

```text
results/
C:\Users\
raw runtime output
credentials/tokens
machine-specific benchmark artifacts
```

Expected: no local run evidence or private machine paths are tracked outside intentional synthetic test fixtures.

- [ ] **Step 6: Final implementation commit only if verification required a repository change**

If verification finds a tracked-file defect, fix that exact defect under a failing test and commit it separately. If no repository change is required, do not create an empty commit.

---

## Phase-1 Acceptance Gate Before Spending Another Qwen Call

All conditions below must be true:

1. Full deterministic pytest suite passes.
2. Fresh Windows CI passes Python 3.11 and 3.12 on the exact implementation head.
3. Fake Test #1 reproduction classifies thinking-only length exhaustion as `THINK_TRUNCATED`, not `ANSWER_WRONG` or `SCORER_DEFECT`.
4. Empty/malformed JSON, extraction, and pseudo-tool outputs do not crash their scorer paths.
5. Think OFF and think ON are sent as explicit, preserved Ollama request fields.
6. Repeated experiments for one task use unique raw/scoring/replay evidence keys.
7. Adaptive search can discover and reproduce a synthetic FAIL→PASS budget boundary.
8. A truncation recovery is labeled with parent experiment and `R1` lineage.
9. `experiments.jsonl`, `characterization-summary.json`, `characterization-report.md`, progress events, replay fixtures, and the manifest are internally consistent.
10. `verify <run-id>` succeeds on the fake-runtime characterization fixture generated by tests where verification is exercised.
11. No exact per-phase thinking-token count is claimed unless directly measured by a validated mechanism.
12. `onboard` regression tests remain green.

Only after this gate is green should the user run:

```powershell
.\.venv\Scripts\compute-cost.exe characterize --model "qwen3.5:27b-q4_K_M"
```

The resulting Q4 evidence is reviewed before any `qwen3.5:27b-q8_0` call is started.

---

## Deferred Beyond Phase 1

These approved-spec capabilities remain intentionally outside this implementation slice until the Phase-1 scientific foundation is proven:

- broad L0–L10 task-family expansion;
- context-position/density/noise matrices;
- prompt ingredient ablation and interaction search;
- seed/temperature frontier research beyond recovery-specific probes;
- automatic reasoning-signature classifiers such as oscillation/overthinking detection;
- automatic hypothesis generation;
- cross-quant transfer scheduling;
- full research-value scheduler;
- persistent global Qwen knowledge base consumed by Inverted.

They are not dropped. Phase 1 creates the trustworthy experiment/evidence primitives those later systems require.