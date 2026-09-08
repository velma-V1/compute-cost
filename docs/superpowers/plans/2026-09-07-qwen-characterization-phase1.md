# Qwen Characterization Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `qwen3.5:27b-q4_K_M` scientifically valid to characterize by separating harness/scorer/truncation failures from capability failures, controlling thinking explicitly, adaptively bracketing reasoning budgets, preserving experiment lineage/replays, and producing a verified task operating profile before any model #2 run.

**Architecture:** Keep the existing evidence-first Ollama adapter, runner, telemetry, manifest, replay, and progress machinery. Add deterministic classification, immutable experiment metadata, a pure adaptive-budget controller, and a separate `characterize` lifecycle that reuses the same low-level evidence paths. Repeated attempts for one logical task use unique experiment/evidence IDs so no raw exchange, scorer result, replay, or progress event is overwritten.

**Tech Stack:** Python 3.11/3.12, standard library, pytest, Ollama `/api/chat`, existing `EvidenceStore`, `BenchmarkRunner`, `ProgressDisplay`, Windows GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-07-adaptive-qwen-characterization-design.md`

## Global Constraints

- First real target: exactly `qwen3.5:27b-q4_K_M`; do not start model #2 until this phase is valid and reviewed.
- Raw request bytes, every runtime stream chunk, exposed thinking, final content, scorer evidence, telemetry, replay data, experiment lineage, progress events, and derived provenance remain under the manifest contract.
- Exposed reasoning is observable behavioral evidence, not hidden neural ground truth.
- Label claims `MEASURED`, `DERIVED`, `ESTIMATED`, or `UNAVAILABLE`; never invent exact thinking-token counts.
- One controlled variable changes per child experiment except explicitly labeled interaction tests.
- Harness/scorer/capture/runtime-invalid attempts do not enter semantic capability denominators.
- Truncation is a valid behavioral observation but not a semantic capability failure.
- Requested context is not actual context usage; runtime `prompt_eval_count` is authoritative when exposed.
- No live Ollama call validates code. Fake-runtime tests and fresh Windows Python 3.11/3.12 CI must pass first.
- Preserve current `onboard`, `verify`, replay, and progress behavior.
- No multi-model batch command.

---

## File Map

**Create**
- `src/compute_cost/classification.py` — result-class rules and capability-validity flag.
- `src/compute_cost/experiments.py` — experiment identity/lineage/controlled fields.
- `src/compute_cost/adaptive.py` — pure budget-boundary search.
- `src/compute_cost/characterization.py` — experiment execution, orchestration, profiles.
- `benchmarks/qwen-characterization-v1.json` — three representative deterministic tasks.
- `tests/test_classification.py`
- `tests/test_experiments.py`
- `tests/test_adaptive.py`
- `tests/test_characterization.py`

**Modify**
- `src/compute_cost/schema.py`
- `src/compute_cost/scoring.py`
- `src/compute_cost/runtimes/ollama.py`
- `src/compute_cost/runner_core.py`
- `src/compute_cost/runner.py`
- `src/compute_cost/config.py`
- `config/default.toml`
- `src/compute_cost/report.py`
- `src/compute_cost/cli.py`
- related existing tests and `README.md`

---

### Task 1: Stable Result and Measurement Schema

**Files:** `src/compute_cost/schema.py`, `tests/test_schema.py`

**Produces:** `ResultClass`, `MeasurementKind`.

- [ ] **Step 1: Write RED tests**

```python
from compute_cost.schema import MeasurementKind, ResultClass


def test_characterization_schema_is_stable():
    assert ResultClass.ANSWER_CORRECT.value == "ANSWER_CORRECT"
    assert ResultClass.THINK_TRUNCATED.value == "THINK_TRUNCATED"
    assert ResultClass.SCORER_DEFECT.value == "SCORER_DEFECT"
    assert [x.value for x in MeasurementKind] == ["MEASURED", "DERIVED", "ESTIMATED", "UNAVAILABLE"]
```

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_schema.py -v
```
Expected: import failure for the new enums.

- [ ] **Step 2: Implement enums**

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

- [ ] **Step 3: Run GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_schema.py -v
git add src/compute_cost/schema.py tests/test_schema.py
git commit -m "feat: add characterization result schema"
```

---

### Task 2: Structured-Output Scorer Hardening

**Files:** `src/compute_cost/scoring.py`, `tests/test_scoring.py`

**Produces:** malformed/empty JSON-like model output is `status="SCORED", score=0.0`, not `SCORER_ERROR`.

- [ ] **Step 1: Write RED regressions**

```python
def test_empty_json_is_scored_format_failure_not_scorer_error():
    result = score_case(case("json", {"name": "Ada"}, required=["name"]), "")
    assert result["status"] == "SCORED"
    assert result["score"] == 0.0
    assert result["checks"][0] == {"name": "valid_json", "pass": False}
    assert result["evidence"]["parse_error"]["type"] == "JSONDecodeError"


def test_malformed_extraction_and_tool_output_are_scored():
    assert score_case(case("extraction_set", ["red"]), "[red]")["status"] == "SCORED"
    assert score_case(case("tool_call", {"tool": "lookup", "arguments": {"id": 7}}), "")["status"] == "SCORED"
```

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_scoring.py -v
```
Expected: the new cases currently become `SCORER_ERROR`.

- [ ] **Step 2: Implement local parse failure handling**

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

Inside `json`, `extraction_set`, and `tool_call` branches:

```python
parsed, parse_error = _parse_json_response(response)
if parse_error is not None:
    return _invalid_json_result(response, parse_error)
```

Keep the outer exception handler so actual scorer-code defects still become `SCORER_ERROR`.

- [ ] **Step 3: Run GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_scoring.py -v
git add src/compute_cost/scoring.py tests/test_scoring.py
git commit -m "fix: distinguish malformed output from scorer defects"
```

---

### Task 3: Observable Thinking/Answer Phase Metrics

**Files:** `src/compute_cost/runtimes/ollama.py`, `tests/test_ollama_adapter.py`

**Consumes:** existing `request_fields={"think": True|False}` support.

**Produces:** measured phase chunk/character/timing fields; no synthetic token split.

- [ ] **Step 1: Write RED adapter test**

```python
def test_generate_derives_observable_thinking_and_answer_phases():
    events = [
        b'{"message":{"role":"assistant","thinking":"plan","content":""},"done":false}\n',
        b'{"message":{"role":"assistant","thinking":" more","content":""},"done":false}\n',
        b'{"message":{"role":"assistant","content":"42"},"done":false}\n',
        b'{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","eval_count":5}\n',
    ]
    transport = FakeTransport([HttpExchange(200, {}, [chunk(events[0], 110), chunk(events[1], 120), chunk(events[2], 140), chunk(events[3], 150)])])
    adapter = OllamaAdapter("http://127.0.0.1:11434", transport=transport, monotonic_ns=lambda: 100)
    result = adapter.generate("fake", [{"role": "user", "content": "solve"}], {"num_predict": 64}, request_fields={"think": True})
    phase = result["phase_metrics"]
    assert phase["thinking_chunks"] == 2
    assert phase["answer_chunks"] == 1
    assert phase["thinking_chars"] == 9
    assert phase["answer_chars"] == 2
    assert phase["time_to_first_thinking_ns"] == 10
    assert phase["time_to_first_answer_ns"] == 40
    assert "thinking_token_count" not in phase
```

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ollama_adapter.py -v
```
Expected: `phase_metrics` missing.

- [ ] **Step 2: Implement measured phase extraction**

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

    def latency(items: list[tuple[int, str]]) -> int | None:
        return None if not items else items[0][0] - started_ns

    def span(items: list[tuple[int, str]]) -> int | None:
        return None if not items else items[-1][0] - items[0][0]

    return {
        "measurement_kind": "MEASURED",
        "thinking_chunks": len(thinking_events),
        "answer_chunks": len(answer_events),
        "thinking_chars": sum(len(text) for _, text in thinking_events),
        "answer_chars": sum(len(text) for _, text in answer_events),
        "time_to_first_thinking_ns": latency(thinking_events),
        "time_to_first_answer_ns": latency(answer_events),
        "thinking_span_ns": span(thinking_events),
        "answer_span_ns": span(answer_events),
    }
```

After normalized output is built:

```python
envelope["phase_metrics"] = self._phase_metrics(
    envelope["stream_events"],
    envelope["request"]["started_monotonic_ns"],
)
```

- [ ] **Step 3: Run GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ollama_adapter.py -v
git add src/compute_cost/runtimes/ollama.py tests/test_ollama_adapter.py
git commit -m "feat: measure observable thinking and answer phases"
```

---

### Task 4: Deterministic Result Classification

**Files:** create `src/compute_cost/classification.py`, `tests/test_classification.py`

**Produces:** `classify_result(case, generation, scoring)` with `result_class`, `basis`, `measurement_kind`, and `valid_for_capability`.

- [ ] **Step 1: Write RED tests including exact Test #1 shape**

```python
from compute_cost.classification import classify_result


def generation(*, text="", thinking="", done_reason="stop", ok=True, error=None):
    return {
        "ok": ok,
        "normalized": {"text": text, "thinking": thinking, "done_reason": done_reason},
        "metrics": {"eval_count": 16},
        "error": error,
    }


def test_thinking_only_length_stop_is_not_semantic_failure():
    result = classify_result(
        {"scorer": "exact"},
        generation(thinking="Thinking Process: analyze request", done_reason="length"),
        {"status": "SCORED", "score": 0.0, "checks": []},
    )
    assert result["result_class"] == "THINK_TRUNCATED"
    assert result["valid_for_capability"] is False


def test_scorer_defect_is_quarantined():
    result = classify_result(
        {"scorer": "json"},
        generation(text="{}"),
        {"status": "SCORER_ERROR", "score": None, "checks": []},
    )
    assert result["result_class"] == "SCORER_DEFECT"
    assert result["valid_for_capability"] is False


def test_invalid_json_and_correct_answer_classify_separately():
    malformed = classify_result(
        {"scorer": "json"},
        generation(text="not-json"),
        {"status": "SCORED", "score": 0.0, "checks": [{"name": "valid_json", "pass": False}]},
    )
    correct = classify_result(
        {"scorer": "exact"},
        generation(text="BLUE", thinking="plan", done_reason="length"),
        {"status": "SCORED", "score": 1.0, "checks": []},
    )
    assert malformed["result_class"] == "FORMAT_FAILURE"
    assert malformed["valid_for_capability"] is True
    assert correct["result_class"] == "ANSWER_CORRECT"
```

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_classification.py -v
```
Expected: module missing.

- [ ] **Step 3: Implement precedence and validity**

```python
from __future__ import annotations

from typing import Any

from .schema import MeasurementKind, ResultClass


CAPABILITY_RESULTS = {
    ResultClass.ANSWER_CORRECT,
    ResultClass.ANSWER_WRONG,
    ResultClass.FORMAT_FAILURE,
    ResultClass.TOOL_FAILURE,
    ResultClass.CONTEXT_FAILURE,
}


def classify_result(case: dict[str, Any], generation: dict[str, Any], scoring: dict[str, Any]) -> dict[str, Any]:
    normalized = generation.get("normalized") or {}
    text = str(normalized.get("text") or "")
    thinking = str(normalized.get("thinking") or "")
    done_reason = normalized.get("done_reason")
    scorer = str(case.get("scorer") or "")
    error_text = json.dumps(generation.get("error") or {}).lower()

    if not generation.get("ok", False) and "timeout" in error_text:
        result, basis = ResultClass.TIMEOUT, "runtime request timed out"
    elif not generation.get("ok", False) and any(token in error_text for token in ("out of memory", "oom", "resource")):
        result, basis = ResultClass.RESOURCE_LIMIT, "runtime reported resource exhaustion"
    elif not generation.get("ok", False):
        result, basis = ResultClass.RUNTIME_FAILURE, "runtime generation failed"
    elif scoring.get("status") == "SCORER_ERROR":
        result, basis = ResultClass.SCORER_DEFECT, "scorer reported benchmark defect"
    elif scoring.get("score") == 1.0:
        result, basis = ResultClass.ANSWER_CORRECT, "deterministic scorer passed"
    elif done_reason == "length" and thinking and not text:
        result, basis = ResultClass.THINK_TRUNCATED, "length stop with exposed thinking and no final content"
    elif done_reason == "length" and text:
        result, basis = ResultClass.ANSWER_TRUNCATED, "length stop after final content began"
    elif not text:
        result, basis = ResultClass.NO_FINAL_ANSWER, "runtime succeeded but emitted no final content"
    elif any(check.get("name") == "valid_json" and check.get("pass") is False for check in scoring.get("checks", [])):
        result, basis = ResultClass.FORMAT_FAILURE, "structured output was not valid JSON"
    elif scorer == "tool_call":
        result, basis = ResultClass.TOOL_FAILURE, "tool-call scorer failed"
    elif scorer == "context_retrieval":
        result, basis = ResultClass.CONTEXT_FAILURE, "context-retrieval scorer failed"
    else:
        result, basis = ResultClass.ANSWER_WRONG, "runtime completed and deterministic scorer failed"

    return {
        "result_class": result.value,
        "basis": basis,
        "measurement_kind": MeasurementKind.DERIVED.value,
        "valid_for_capability": result in CAPABILITY_RESULTS,
    }
```

Also add `import json`.

- [ ] **Step 4: Run GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_classification.py -v
git add src/compute_cost/classification.py tests/test_classification.py
git commit -m "feat: classify model and harness failure modes"
```

---

### Task 5: Immutable Experiment Identity and One-Variable Lineage

**Files:** create `src/compute_cost/experiments.py`, `tests/test_experiments.py`

**Produces:** `ExperimentSpec`, `make_experiment_id`, `changed_fields`.

- [ ] **Step 1: Write RED tests**

```python
from compute_cost.experiments import ExperimentSpec, changed_fields, make_experiment_id


def make_spec(**overrides):
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


def test_id_and_one_variable_change():
    assert make_experiment_id(7, "math-001", "think-on-256") == "exp-000007-math-001-think-on-256"
    parent = make_spec()
    child = make_spec(
        experiment_id="exp-child",
        parent_experiment_id="exp-root",
        hypothesis="find lower passing budget",
        changed_variable="generation_budget",
        generation_budget=192,
    )
    assert changed_fields(parent, child) == ["generation_budget"]
```

- [ ] **Step 2: Implement**

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

- [ ] **Step 3: Run RED/GREEN cycle and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_experiments.py -v
git add src/compute_cost/experiments.py tests/test_experiments.py
git commit -m "feat: add experiment identity and lineage"
```

---

### Task 6: Pure Adaptive Thought-Budget Controller

**Files:** create `src/compute_cost/adaptive.py`, `tests/test_adaptive.py`

**Produces:** `BudgetObservation`, `BudgetDecision`, `AdaptiveBudgetController.next()`.

- [ ] **Step 1: Write RED boundary/stop tests**

```python
from compute_cost.adaptive import AdaptiveBudgetController, BudgetObservation


def obs(budget, result):
    return BudgetObservation(budget=budget, result_class=result)


def controller():
    return AdaptiveBudgetController(initial_budget=256, min_budget=32, max_budget=2048, granularity=32, boundary_repeats=3)


def test_brackets_pass_fail_boundary():
    c = controller()
    assert c.next([obs(256, "ANSWER_CORRECT")]).budget == 128
    assert c.next([obs(256, "ANSWER_CORRECT"), obs(128, "ANSWER_WRONG")]).budget == 192
    assert c.next([obs(256, "ANSWER_CORRECT"), obs(128, "ANSWER_WRONG"), obs(192, "ANSWER_CORRECT")]).budget == 160


def test_only_evidenced_truncation_scales_up():
    assert controller().next([obs(256, "THINK_TRUNCATED")]).budget == 512
    assert controller().next([obs(256, "ANSWER_WRONG")]).action == "STOP"
    assert controller().next([obs(256, "NO_FINAL_ANSWER")]).action == "STOP"


def test_invalid_observation_stops_branch():
    decision = controller().next([obs(256, "SCORER_DEFECT")])
    assert decision.action == "STOP"
    assert decision.reason == "invalid experiment observation"


def test_minimum_pass_is_replicated():
    c = controller()
    rows = [obs(128, "ANSWER_WRONG"), obs(160, "ANSWER_WRONG"), obs(192, "ANSWER_CORRECT")]
    assert c.next(rows).action == "REPLICATE"
    rows += [obs(192, "ANSWER_CORRECT"), obs(192, "ANSWER_CORRECT")]
    assert c.next(rows).reason == "minimum passing boundary reproduced"
```

- [ ] **Step 2: Implement controller**

```python
from __future__ import annotations

from dataclasses import dataclass

PASS = {"ANSWER_CORRECT"}
TRUNCATION = {"THINK_TRUNCATED", "ANSWER_TRUNCATED"}
INVALID = {"SCORER_DEFECT", "TEST_DEFECT", "CAPTURE_GAP", "RUNTIME_FAILURE", "TIMEOUT", "RESOURCE_LIMIT"}


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
        if observations[-1].result_class in INVALID:
            return BudgetDecision("STOP", None, "invalid experiment observation")

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
            if candidate != minimum_pass and not any(item.budget == candidate for item in observations):
                return BudgetDecision("PROBE", candidate, "search below current passing budget")

        lower_fail = max(lower_failures) if lower_failures else self.min_budget - self.granularity
        if minimum_pass - lower_fail > self.granularity:
            candidate = self._round(lower_fail + (minimum_pass - lower_fail) // 2)
            candidate = max(lower_fail + self.granularity, min(minimum_pass - self.granularity, candidate))
            return BudgetDecision("PROBE", candidate, "bisect fail/pass budget bracket")

        pass_count = sum(1 for item in observations if item.budget == minimum_pass and item.result_class in PASS)
        if pass_count < self.boundary_repeats:
            return BudgetDecision("REPLICATE", minimum_pass, "reproduce minimum passing boundary")
        return BudgetDecision("STOP", None, "minimum passing boundary reproduced")
```

- [ ] **Step 3: Run GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_adaptive.py -v
git add src/compute_cost/adaptive.py tests/test_adaptive.py
git commit -m "feat: add adaptive thought budget controller"
```

---

### Task 7: Reusable Run Preflight + Unique Evidence Keys

**Files:** `src/compute_cost/runner_core.py`, `src/compute_cost/runner.py`, `tests/test_runner.py`, `tests/test_runner_progress.py`

**Changes:** extend `_execute_case`/`run_case` with `request_fields`, `evidence_key`, `experiment`; extract `_begin_run` and `_preflight_model`; retain onboarding behavior.

- [ ] **Step 1: Write RED evidence-isolation test**

```python
def test_repeated_case_uses_unique_evidence_keys(tmp_path: Path):
    runner = BenchmarkRunner(FakeRuntime(), config(), suite(), results_root=tmp_path, telemetry=FakeTelemetry())
    runner.model = "fake"
    runner.store = EvidenceStore(tmp_path, "repeat-evidence")
    case = suite()["cases"][0]
    runner.run_case(case, stage="characterize", evidence_key="exp-000001", request_fields={"think": False})
    runner.run_case(case, stage="characterize", evidence_key="exp-000002", request_fields={"think": True})
    names = sorted(path.name for path in (runner.store.run_dir / "raw/scoring").glob("*.json"))
    assert names == ["characterize-exp-000001.json", "characterize-exp-000002.json"]
    assert runner.runtime.calls[-2]["request_fields"] == {"think": False}
    assert runner.runtime.calls[-1]["request_fields"] == {"think": True}
```

- [ ] **Step 2: Thread request fields/evidence key through existing execution**

Add optional parameters:

```python
request_fields: dict[str, Any] | None = None,
evidence_key: str | None = None,
experiment: dict[str, Any] | None = None,
```

In `_execute_case`:

```python
artifact_key = evidence_key or str(case["id"])
generation, invocation, refs = self._invoke_generation(
    stage=stage,
    case_id=artifact_key,
    messages=messages,
    options=options,
    request_fields=request_fields,
)
classification = classify_result(case, generation, scoring)
record = {
    "case_id": case["id"],
    "category": case.get("category"),
    "stage": stage,
    "status": scoring.get("status"),
    "score": scoring.get("score"),
    "timing": copy.deepcopy(generation.get("timing") or {}),
    "metrics": copy.deepcopy(generation.get("metrics") or {}),
    "phase_metrics": copy.deepcopy(generation.get("phase_metrics") or {}),
    "runtime_ok": bool(generation.get("ok", False)),
    "evidence_refs": refs,
    "evidence_key": artifact_key,
    "scorer": case.get("scorer"),
    "classification": classification,
    "experiment": copy.deepcopy(experiment),
}
```

Pass `artifact_key` to `_persist_scoring`. Preserve `case_id` as the logical task ID.

- [ ] **Step 3: Extract exact run initialization**

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

- [ ] **Step 4: Extract exact model preflight from current `onboard`**

```python
def _preflight_model(self, model: str, *, pull: bool = False) -> bool:
    assert self.store is not None
    store = self.store
    hardware = self.hardware_collector()
    store.write_json("hardware.json", hardware, producer="hardware", stage="preflight")

    version = self.runtime.version()
    version_refs = self._persist_control_exchange("runtime-version", version)
    tags = self.runtime.list_models()
    tags_refs = self._persist_control_exchange("model-list", tags)
    available = self.runtime.model_available_in(tags, model) if hasattr(self.runtime, "model_available_in") else self.runtime.is_model_available(model)

    pull_refs = None
    if not available and pull:
        pull_result = self.runtime.pull(model)
        pull_refs = self._persist_control_exchange("model-pull", pull_result)
        tags = self.runtime.list_models()
        tags_refs = self._persist_control_exchange("model-list-after-pull", tags)
        available = self.runtime.model_available_in(tags, model) if hasattr(self.runtime, "model_available_in") else self.runtime.is_model_available(model)

    if not available:
        store.write_json(
            "runtime.json",
            {
                "model": model,
                "available": False,
                "version": version.get("parsed"),
                "model_size_bytes": None,
                "evidence_refs": {"version": version_refs, "tags": tags_refs, "pull": pull_refs},
            },
            producer="runner",
            stage="preflight",
        )
        self._event("RUN_FAILED", failure="MODEL_NOT_FOUND", model=model)
        return False

    info = self.runtime.model_info(model)
    info_refs = self._persist_control_exchange("model-info", info)
    store.write_json(
        "runtime.json",
        {
            "model": model,
            "available": True,
            "version": version.get("parsed"),
            "model_size_bytes": self._find_model_size(tags, model),
            "model_info": info.get("parsed"),
            "evidence_refs": {"version": version_refs, "tags": tags_refs, "show": info_refs, "pull": pull_refs},
        },
        producer="runner",
        stage="preflight",
    )
    self._event("PREFLIGHT_COMPLETE", model=model)
    return True
```

Change `onboard` to call `_begin_run`, then `_preflight_model`; if preflight returns false, finalize; otherwise execute the existing cold/warm/base/context/sustained flow unchanged.

- [ ] **Step 5: Mirror new optional `_execute_case` arguments through public progress runner**

Use identical defaults in `src/compute_cost/runner.py` and pass them to `super()._execute_case`.

- [ ] **Step 6: Run GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_runner.py tests/test_runner_progress.py -v
git add src/compute_cost/runner_core.py src/compute_cost/runner.py tests/test_runner.py tests/test_runner_progress.py
git commit -m "refactor: isolate repeated experiment evidence"
```

---

### Task 8: Phase-1 Suite and Characterization Config

**Files:** create `benchmarks/qwen-characterization-v1.json`; modify `config/default.toml`, `src/compute_cost/config.py`, `tests/test_config.py`, `tests/test_benchmark_suite.py`.

- [ ] **Step 1: Write RED config/suite assertions**

```python
characterization = config["characterization"]
assert characterization["initial_think_budget"] == 256
assert characterization["min_think_budget"] == 32
assert characterization["max_think_budget"] == 2048
assert characterization["budget_granularity"] == 32
assert characterization["boundary_repeats"] == 3
assert characterization["max_experiments_per_task"] == 12
```

Suite IDs must be exactly:

```python
["char-if-001", "char-json-001", "char-math-001"]
```

- [ ] **Step 2: Create suite**

```json
{
  "benchmark_version": "qwen-characterization-v1",
  "description": "Representative deterministic tasks for adaptive thinking characterization.",
  "cases": [
    {"id":"char-if-001","category":"instruction_following","difficulty_level":1,"prompt":"Reply with exactly: BLUE","scorer":"exact","expected":"BLUE","timeout_s":120},
    {"id":"char-json-001","category":"structured_output","difficulty_level":2,"prompt":"Return only valid JSON with name set to Ada and count set to 3. No markdown.","scorer":"json","expected":{"name":"Ada","count":3},"required":["name","count"],"timeout_s":120},
    {"id":"char-math-001","category":"reasoning_math","difficulty_level":3,"prompt":"A 240 GB dataset is reduced by 25 percent, then 20 GB is added. What is the final size in GB? Give the number.","scorer":"numeric","expected":200,"tolerance":0,"timeout_s":120}
  ]
}
```

No fixed `max_output_tokens`; the experiment owns generation budget.

- [ ] **Step 3: Add defaults and validation**

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

Validation: each value is a positive integer and:

```python
if not (c["min_think_budget"] <= c["initial_think_budget"] <= c["max_think_budget"]):
    raise ValueError("characterization budgets must satisfy min <= initial <= max")
```

- [ ] **Step 4: Run GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_benchmark_suite.py -v
git add benchmarks/qwen-characterization-v1.json config/default.toml src/compute_cost/config.py tests/test_config.py tests/test_benchmark_suite.py
git commit -m "feat: add phase one characterization suite"
```

---

### Task 9: Adaptive Characterization Orchestrator

**Files:** create `src/compute_cost/characterization.py`, `tests/test_characterization.py`; modify runner files and progress tests.

**Produces:** `experiments.jsonl`, OFF/ON baselines, budget bracketing, boundary replication, plan-adjusted progress.

- [ ] **Step 1: Write fake-runtime RED test**

Fake behavior for one task:
- `think=False` -> deterministic wrong final answer.
- `think=True` and budget `<192` -> exposed thinking, no final content, `done_reason="length"`.
- `think=True` and budget `>=192` -> correct final answer.

Assertions:

```python
run_dir = runner.characterize("fake")
rows = [json.loads(line) for line in (run_dir / "experiments.jsonl").read_text().splitlines()]
assert rows[0]["experiment"]["thinking_mode"] is False
assert rows[1]["experiment"]["thinking_mode"] is True
assert any(row["classification"]["result_class"] == "THINK_TRUNCATED" for row in rows)
assert sum(1 for row in rows if row["experiment"]["generation_budget"] == 192 and row["classification"]["result_class"] == "ANSWER_CORRECT") >= 3
assert len({row["evidence_key"] for row in rows}) == len(rows)
assert EvidenceStore(tmp_path, run_dir.name).verify_manifest() == []
```

- [ ] **Step 2: Implement a single experiment executor**

In `characterization.py` implement `execute_experiment(runner, case, spec)`:

```python
record = runner.run_case(
    case,
    stage="characterize",
    append=False,
    evidence_key=spec.experiment_id,
    request_fields={"think": spec.thinking_mode},
    option_overrides={
        "num_predict": spec.generation_budget,
        "temperature": spec.temperature,
        "seed": spec.seed,
    },
    experiment=spec.to_dict(),
)
row = {
    "experiment": spec.to_dict(),
    "classification": record["classification"],
    "score": record["score"],
    "status": record["status"],
    "metrics": record["metrics"],
    "timing": record["timing"],
    "phase_metrics": record["phase_metrics"],
    "evidence_key": record["evidence_key"],
}
runner.store.append_jsonl("experiments.jsonl", row)
return row
```

Before a child executes:

```python
changes = changed_fields(parent_spec, child_spec)
if child_spec.changed_variable not in {"baseline", "interaction"} and changes != [child_spec.changed_variable]:
    raise ValueError(f"experiment must change exactly {child_spec.changed_variable}: {changes}")
```

- [ ] **Step 3: Implement per-task sequence**

For every case:
1. OFF baseline at `think_off_budget`.
2. ON baseline from `controller.next([])`.
3. Feed only thinking-ON behavioral observations to controller.
4. For `PROBE`/`REPLICATE`, create a child with parent equal to the preceding ON experiment.
5. OFF→ON changes only `thinking_mode` because both use budget 256 initially.
6. Budget probes change only `generation_budget`.
7. If the previous class is `THINK_TRUNCATED` or `ANSWER_TRUNCATED` and budget increases, set `recovery_level="R1"` and hypothesis `insufficient generation headroom caused truncation`.
8. Stop at `max_experiments_per_task` and persist `stop_reason="MAX_EXPERIMENTS_PER_TASK"`.
9. Do not feed `SCORER_DEFECT`, runtime-invalid, or capture-invalid observations into capability profiles; controller stops the affected branch.

- [ ] **Step 4: Add `characterize` lifecycle**

Core:

```python
def characterize(self, model: str, *, pull: bool = False) -> Path:
    self._begin_run(model)
    if not self._preflight_model(model, pull=pull):
        return self._finalize_run()
    run_characterization(self, self.suite.get("cases", []) or [])
    self._event("CHARACTERIZATION_COMPLETE", model=model)
    return self._finalize_run()
```

Public progress wrapper starts with:

```python
initial_total = 1 + (len(self.suite.get("cases", []) or []) * 2) + 1
```

Each adaptive extra call increments `progress.total_tasks` before execution and writes:

```python
self._record_progress(
    "plan_adjusted",
    task_label,
    old_total=old_total,
    new_total=self.progress.total_tasks,
    reason="adaptive experiment added",
)
```

- [ ] **Step 5: Run GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_characterization.py tests/test_runner_progress.py -v
git add src/compute_cost/characterization.py src/compute_cost/runner_core.py src/compute_cost/runner.py tests/test_characterization.py tests/test_runner_progress.py
git commit -m "feat: add adaptive Qwen characterization runner"
```

---

### Task 10: Replay v2 with Experiment/Recovery Lineage

**Files:** `src/compute_cost/runner_core.py`, `tests/test_runner.py`, `tests/test_characterization.py`

**Produces:** new replay snapshots keep all v1 fields plus `experiment`, `classification`, `evidence_key`; recovery is represented by `experiment.recovery_level`, not a separate competing field.

- [ ] **Step 1: Write RED replay assertions**

```python
assert replay["schema_version"] == 2
assert replay["experiment"]["experiment_id"].startswith("exp-")
assert replay["classification"]["result_class"] == "THINK_TRUNCATED"
assert replay["evidence_key"] == replay["experiment"]["experiment_id"]
assert replay["generation"]["normalized"]["thinking"]
assert replay["invocation"]["request_fields"] == {"think": True}
```

- [ ] **Step 2: Extend `_write_replay`**

Add:

```python
experiment: dict[str, Any] | None = None,
classification: dict[str, Any] | None = None,
evidence_key: str | None = None,
```

Snapshot fields:

```python
"schema_version": 2,
"experiment": copy.deepcopy(experiment),
"classification": copy.deepcopy(classification),
"evidence_key": evidence_key,
```

Retain existing exact `case`, `invocation`, `generation`, `scoring`, telemetry, config, and artifact evidence.

Routing rule:
- semantic/truncation characterization failures -> `replay/<experiment_id>.json`;
- `SCORER_DEFECT`, `TEST_DEFECT`, `CAPTURE_GAP` -> `replay/harness/<experiment_id>.json` and never label them model failures.

- [ ] **Step 3: Run GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_runner.py tests/test_characterization.py -v
git add src/compute_cost/runner_core.py tests/test_runner.py tests/test_characterization.py
git commit -m "feat: retain characterization replay lineage"
```

---

### Task 11: Task Operating Profile + Characterization Report

**Files:** `src/compute_cost/characterization.py`, `src/compute_cost/report.py`, `tests/test_characterization.py`, `tests/test_report.py`

**Produces:** `characterization-summary.json`, `characterization-report.md`.

- [ ] **Step 1: Write RED profile test**

```python
profile = build_task_profile("char-math-001", rows, boundary_repeats=3)
assert profile["minimum_reproduced_pass_budget"] == {"value": 192, "kind": "DERIVED"}
assert profile["transition_bracket"] == {"lower_fail": 160, "upper_pass": 192, "kind": "DERIVED"}
assert profile["think_off"]["result_class"] == "ANSWER_WRONG"
```

Assert rows with `valid_for_capability=False` are excluded from semantic pass-rate calculations but retained under `behavioral_observations`.

- [ ] **Step 2: Implement `build_task_profile(task_id, rows, boundary_repeats)`**

Rules:
- identify OFF baseline separately;
- keep all observations in evidence order;
- capability subset requires `classification.valid_for_capability=True`;
- budget boundary uses ON rows and requires `boundary_repeats` correct observations at the candidate minimum pass;
- highest lower failure becomes `transition_bracket.lower_fail`;
- unknown values are `None`, never guessed;
- exact `metrics`, `timing`, `phase_metrics` remain measured observation payloads.

Return at minimum:

```python
{
    "task_id": task_id,
    "think_off": think_off_summary,
    "minimum_reproduced_pass_budget": minimum_pass,
    "transition_bracket": bracket,
    "capability_observations": capability_rows,
    "behavioral_observations": rows,
}
```

- [ ] **Step 3: Persist summary/report before final manifest**

```python
runner.store.write_json("characterization-summary.json", summary, producer="characterization", stage="report")
runner.store.write_raw(
    "characterization-report.md",
    render_characterization_report(summary),
    producer="characterization",
    stage="report",
    media_type="text/markdown",
)
```

Report text must state exposed thinking is an observable trace and aggregate `eval_count` is not an exact per-phase token count.

- [ ] **Step 4: Run GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_characterization.py tests/test_report.py -v
git add src/compute_cost/characterization.py src/compute_cost/report.py tests/test_characterization.py tests/test_report.py
git commit -m "feat: build evidence backed Qwen operating profiles"
```

---

### Task 12: Single-Model `characterize` CLI

**Files:** `src/compute_cost/cli.py`, `tests/test_cli.py`

- [ ] **Step 1: Write RED parser/dispatch tests**

```python
args = build_parser().parse_args(["characterize", "--model", "fake"])
assert args.command == "characterize"
assert args.model == "fake"
```

Execution test must prove one string reaches `runner.characterize("fake", pull=False)` and output contains `run_id`/`run_dir`.

- [ ] **Step 2: Add constant/parser**

```python
DEFAULT_CHARACTERIZATION_SUITE_PATH = PROJECT_ROOT / "benchmarks" / "qwen-characterization-v1.json"
```

Inside `build_parser`:

```python
characterize = sub.add_parser("characterize", help="Adaptively characterize one local model.")
characterize.add_argument("--model", required=True)
characterize.add_argument("--suite", default=str(DEFAULT_CHARACTERIZATION_SUITE_PATH))
characterize.add_argument("--pull", action="store_true", help="Pull the model if it is not already local.")
```

Dispatch after runner construction:

```python
if args.command == "characterize":
    run_dir = runner.characterize(args.model, pull=bool(args.pull))
    print(json.dumps({"run_id": run_dir.name, "run_dir": str(run_dir)}, indent=2))
    return 0
```

Do not add `--models`.

- [ ] **Step 3: Run GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cli.py -v
git add src/compute_cost/cli.py tests/test_cli.py
git commit -m "feat: add single model characterize command"
```

---

### Task 13: Full Fake Test #1 Acceptance Gate

**Files:** `tests/test_characterization.py`, `tests/test_runner_progress.py`

- [ ] **Step 1: Reproduce exact failure mechanism**

Fake generation at budget 16:

```python
{
    "ok": True,
    "normalized": {
        "text": "",
        "thinking": "Thinking Process:\n\n1. Analyze the Request",
        "done": True,
        "done_reason": "length",
    },
    "metrics": {"eval_count": 16},
}
```

Assertions:

```python
assert row["classification"]["result_class"] == "THINK_TRUNCATED"
assert row["classification"]["valid_for_capability"] is False
assert row["classification"]["result_class"] != "ANSWER_WRONG"
```

Then make the higher-budget child correct and assert `experiment.recovery_level == "R1"` plus three reproduced PASS observations at the discovered boundary.

- [ ] **Step 2: Prove scorer-defect quarantine and adaptive progress**

```python
assert scorer_defect_row["classification"]["result_class"] == "SCORER_DEFECT"
assert scorer_defect_row["classification"]["valid_for_capability"] is False
```

Read `progress.jsonl` and assert one event has:

```python
assert event["event"] == "plan_adjusted"
assert event["reason"] == "adaptive experiment added"
```

Final progress snapshot must equal 100%.

- [ ] **Step 3: Run complete deterministic suite and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git add tests/test_characterization.py tests/test_runner_progress.py
git commit -m "test: prove Qwen truncation is not capability failure"
```

---

### Task 14: Documentation, Privacy Scan, and Windows CI

**Files:** `README.md`, existing `.github/workflows/test.yml`

- [ ] **Step 1: Document Phase-1 command and interpretation**

```powershell
.\.venv\Scripts\compute-cost.exe characterize --model "qwen3.5:27b-q4_K_M"
```

Document:
- separate thinking/final streams when Ollama exposes them;
- `eval_count` is aggregate generation evidence;
- `THINK_TRUNCATED` is behavioral/truncation evidence, not semantic failure;
- malformed final JSON is a model-format failure; `SCORER_DEFECT` is harness-invalid;
- adaptive denominator changes are recorded;
- machine metadata/raw results remain local/gitignored and can contain host/runtime paths.

- [ ] **Step 2: Full local deterministic verification**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: all tests pass without an Ollama model call.

- [ ] **Step 3: Commit README**

```powershell
git add README.md
git commit -m "docs: explain adaptive Qwen characterization"
```

- [ ] **Step 4: Push implementation branch and verify fresh CI**

Required evidence:
- workflow run head SHA equals implementation head;
- Windows Python 3.11 job succeeds;
- Windows Python 3.12 job succeeds.

- [ ] **Step 5: Repository privacy/evidence scan**

Check tracked content for:

```text
results/
C:\Users\
raw runtime model output
credentials or tokens
machine-specific benchmark artifacts
```

Expected: no real local run evidence or private machine identifiers are tracked outside intentional synthetic fixtures.

If the scan finds a defect, add a failing regression test, fix that exact defect, rerun full tests/CI, and commit the fix separately. If the scan is clean, create no empty commit.

---

## Phase-1 Acceptance Gate Before Another Qwen Call

All must be true:

1. Full deterministic pytest suite passes.
2. Fresh Windows CI is green on Python 3.11 and 3.12 at exact head.
3. Fake Test #1 shape becomes `THINK_TRUNCATED`, `valid_for_capability=false`.
4. Empty/malformed JSON/extraction/tool outputs never masquerade as scorer defects.
5. Think OFF/ON are explicit and preserved in exact Ollama request evidence.
6. Repeated task experiments have unique raw/scoring/replay keys.
7. Adaptive search brackets and reproduces a synthetic FAIL→PASS boundary.
8. Budget recovery carries parent experiment + `R1` lineage.
9. Harness-invalid observations are retained but excluded from semantic capability profiles.
10. `experiments.jsonl`, characterization summary/report, progress, replay fixtures, and manifest are consistent.
11. No fabricated exact thought-token count is emitted.
12. Existing `onboard` and `verify` regression tests remain green.

Only then run:

```powershell
.\.venv\Scripts\compute-cost.exe characterize --model "qwen3.5:27b-q4_K_M"
```

Review that Q4 run before any `qwen3.5:27b-q8_0` invocation.

---

## Deferred Beyond Phase 1

The approved architecture remains larger than this first implementation slice. Defer until Phase 1 is scientifically proven:

- broad L0–L10 capability ladders;
- context position/density/noise matrices;
- prompt ingredient ablation/interactions;
- seed/temperature frontier research outside recovery probes;
- automatic reasoning-signature classifiers;
- automatic hypothesis generation;
- cross-quant transfer scheduling;
- full information-gain research scheduler;
- persistent global Qwen knowledge base consumed by Inverted.

Phase 1 builds the trustworthy experiment/evidence substrate required by those later layers.