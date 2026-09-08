# Model Onboarding + Compute Cost Benchmark Design

**Status:** Approved architecture baseline
**Version:** base-v1
**Date:** 2026-09-07

## Objective

Build a local-first benchmark harness for evaluating new model tryouts on the user's PC. The harness must answer two decisions with evidence:

1. Is this model worth keeping on this machine?
2. If kept, what does it cost—in time, memory, energy, storage, stability, and capability—to obtain useful work from it?

The harness is not only a speed test and not only an academic quality benchmark. It combines onboarding cost, machine-fit telemetry, deterministic capability checks, context scaling, sustained-load behavior, and replayable evidence.

## Design Principles

- **Purpose-built core.** Native Python CLI with runtime adapters; Ollama is the first adapter.
- **Local-first.** No paid API dependency is required for base-v1.
- **Evidence before score.** Preserve raw inputs, raw outputs, telemetry samples, timings, runtime metadata, and scoring evidence.
- **Measured vs derived vs estimated.** Reports must identify which values came directly from observation, which were computed from measured values, and which depend on configurable assumptions.
- **Comparable runs.** Every model receives the same versioned benchmark contract unless an explicit compatibility exception is recorded.
- **Safe boundary discovery.** Context and sustained-load tests stop at configured memory, latency, timeout, or runtime-failure boundaries rather than forcing unsafe exhaustion.
- **Replayable failures.** Failed benchmark cases retain enough evidence to reproduce the prompt/configuration and rescore or replay later without rerunning unrelated cases.
- **YAGNI.** No dashboard, database server, container stack, or distributed scheduler in base-v1.

## Supported Environment

Base-v1 targets:

- Windows 11 host
- NVIDIA GPU telemetry when NVIDIA tooling is available
- CPU/RAM telemetry on the host
- Ollama as the first model runtime
- Python 3.11+

The architecture must permit later adapters for llama.cpp, Transformers/vLLM, LM Studio-compatible APIs, or other runtimes without changing the run schema.

## Benchmark Run Lifecycle

A complete onboarding run is:

`preflight -> cold start -> warmup -> base capability suite -> context sweep -> sustained-load test -> report`

Each stage emits structured evidence and may end in `PASS`, `PARTIAL`, `FAILED`, or `SKIPPED` with a machine-readable reason.

### 1. Preflight

Capture:

- benchmark version and git commit when available
- UTC/local timestamps
- OS and Python version
- CPU identity and logical-core count
- total/available RAM
- GPU identity, driver, total/free VRAM when observable
- runtime identity/version
- model identifier/tag
- model metadata available from the runtime
- configurable electricity price
- benchmark settings and stop thresholds

A preflight failure that prevents inference stops the run while preserving the preflight artifact.

### 2. Cold Start / Onboarding Cost

Measure or record when observable:

- model storage footprint
- model availability before run
- pull/download duration if the harness performs the pull
- runtime setup errors
- cold model-load duration
- cold first-token latency / time-to-first-token
- cold end-to-end latency
- peak RAM/VRAM during cold load and first response

Download/setup cost is only labeled **measured** when the harness actually observes it. Existing local model files are not assigned invented download cost.

### 3. Warmup

Run a small fixed request to stabilize model/runtime state. Record warm first-token and completion latency but exclude warmup quality from the capability score.

### 4. Base Capability Suite

Base-v1 uses deterministic or mechanically scoreable tasks wherever practical. Categories:

1. **Instruction following** — exact constraints and requested transformations.
2. **Structured output** — valid JSON and schema-conforming values.
3. **Extraction** — recover specified facts from supplied text.
4. **Reasoning/math** — answer problems with deterministic expected answers.
5. **Coding** — generate a small function judged by executable unit tests in an isolated temporary directory; no network access is required.
6. **Tool-call shaped output** — select a named pseudo-tool and emit schema-valid arguments without executing an external tool.
7. **Ambiguity handling** — distinguish answerable instructions from cases that require a bounded clarification or explicit assumption.
8. **Long-context retrieval** — recover planted facts from synthetic context to measure retrieval degradation independent of world knowledge.

Every case declares:

- unique case ID
- benchmark version
- category
- prompt template/input
- generation parameters
- scorer type
- expected evidence or executable tests
- timeout
- maximum output budget
- compatibility requirements, if any

No model-graded judge is required for the base-v1 score. Subjective/open-ended tests may be added later as separately labeled research suites.

### 5. Context Sweep

Run synthetic long-context retrieval at increasing context sizes. The configured schedule is recorded in the run artifact.

For each level capture:

- requested context size
- observed prompt token count if runtime reports it
- prompt processing rate if observable
- generation rate
- TTFT
- total latency
- answer correctness
- peak RAM/VRAM
- sampled GPU utilization/power
- failure reason

Stop the sweep when any configured hard boundary is reached:

- runtime/model error
- timeout
- configurable host RAM ceiling
- configurable VRAM ceiling when observable
- consecutive correctness failures threshold

The last successful level and first failed/stopped level form the model's observed context boundary on this machine for this runtime/configuration.

### 6. Sustained-Load Test

Run a fixed sequence of representative prompts for a bounded duration/count to expose:

- throughput decay
- memory growth/leak symptoms
- thermal or power throttling signals when observable
- runtime crashes/timeouts
- output instability under repeated load

This is a bounded stability test, not an unrestricted stress test.

### 7. Report

Generate both machine-readable JSON and human-readable Markdown.

The report contains:

- model/runtime/config identity
- hardware snapshot
- stage status
- onboarding metrics
- capability category scores and aggregate score
- context boundary
- sustained-load results
- peak resource use
- throughput/latency metrics
- energy and electricity-cost metrics when power data is available
- failure/replay references
- explicit `measured`, `derived`, and `estimated` labels

## Metrics

### Directly measured when available

- wall-clock duration
- TTFT
- total request latency
- prompt/completion token counts reported by runtime
- prompt/completion throughput
- process/host RAM
- VRAM usage
- GPU utilization
- GPU power draw
- model disk footprint
- failures/timeouts

### Derived

Examples:

- successful cases per minute
- generated tokens per joule when power integration is available
- joules per successful benchmark case
- watt-hours per run
- electricity cost = measured kWh × configured electricity price
- capability-per-second
- capability-per-GB-VRAM
- capability-per-watt-hour

Derived metrics must preserve enough source measurements to recompute them.

### Estimated / Optional

Hardware amortization is optional and disabled by default. If enabled, the user supplies hardware cost and amortization assumptions. The report must label resulting per-hour/per-run figures as estimates.

## Capability Scoring

Each case is scored in `[0, 1]`. Base-v1 category scores are arithmetic means of compatible cases. The aggregate baseline capability score is the arithmetic mean of category scores that have at least one compatible executed case.

The report must show category scores alongside the aggregate; the aggregate cannot hide a zero or missing category.

A separate **usable-efficiency** section combines capability with compute metrics, but base-v1 does not collapse all tradeoffs into one authoritative scalar. It reports at minimum:

- capability / second
- capability / peak VRAM GiB
- capability / Wh when energy is measured
- successful benchmark cases / minute

This avoids pretending that one weighting of speed, memory, energy, and quality is universally correct.

## Evidence Layout

Each run receives a unique directory under `results/<run-id>/` containing:

- `manifest.json` — schema version, benchmark version, model/runtime/config, stage status, artifact hashes
- `hardware.json` — hardware/runtime preflight snapshot
- `events.jsonl` — ordered lifecycle/events
- `telemetry.jsonl` — timestamped resource samples
- `cases.jsonl` — one record per benchmark case with prompt, raw response, timing, token data, scorer evidence, score, and status
- `summary.json` — normalized machine-readable metrics
- `report.md` — human-readable report
- `replay/` — compact failure records sufficient to rerun individual failed cases

`results/` is gitignored except for schemas/examples intentionally committed to the repository.

Artifacts include SHA-256 hashes in the manifest so evidence corruption or accidental edits are detectable.

## CLI Contract

Primary commands:

```text
compute-cost preflight
compute-cost onboard --model <runtime-model-id>
compute-cost benchmark --model <runtime-model-id> [--suite base-v1]
compute-cost compare <run-a> <run-b> [<run-c> ...]
compute-cost replay <run-id> <case-id>
```

Common configuration is loaded from `config/default.toml` with CLI overrides for values such as runtime endpoint, timeouts, context schedule, telemetry interval, electricity price, and safety ceilings.

## Runtime Adapter Boundary

The core runner depends on a runtime protocol rather than Ollama-specific calls. The adapter interface must provide enough information for:

- runtime health/version
- model metadata
- ensuring a model is locally available when explicitly requested
- generation with a fixed request config
- token/timing fields available from the runtime
- unload/reload when supported so cold-start measurements are meaningful

Unsupported runtime capabilities are recorded as unavailable rather than fabricated.

## Telemetry Boundary

Telemetry collection is best-effort and must never invalidate otherwise valid capability evidence solely because one sensor is unavailable.

Base-v1 collectors:

- host CPU/RAM via Python system APIs
- NVIDIA GPU/VRAM/utilization/power through `nvidia-smi` when present

Power integration uses timestamped samples and trapezoidal numerical integration over the observed interval. If power cannot be measured, energy/electricity metrics are `unavailable`, not estimated from TDP.

## Failure Model

Failures are first-class evidence. Every stage/case records a typed failure such as:

- `RUNTIME_UNAVAILABLE`
- `MODEL_NOT_FOUND`
- `MODEL_PULL_FAILED`
- `MODEL_LOAD_FAILED`
- `TIMEOUT`
- `OOM_OR_RESOURCE_LIMIT`
- `INVALID_STRUCTURED_OUTPUT`
- `SCORER_ERROR`
- `RUNTIME_ERROR`
- `USER_LIMIT_STOP`

A benchmark implementation bug is distinct from a model failure and must not reduce the model's capability score.

## Safety / Isolation

- No destructive host actions.
- Coding tests execute only generated code needed for the test, in an isolated temporary working directory with bounded timeout and no benchmark-provided network dependency.
- Context/stress tests use configurable resource ceilings and bounded durations.
- Secrets/environment values are not dumped wholesale into evidence.
- Raw evidence is local by default; the repository stores code, schemas, and synthetic examples, not the user's run data.

## Planned Repository Structure

```text
compute-cost/
  README.md
  pyproject.toml
  .gitignore
  config/default.toml
  benchmarks/base-v1.json
  docs/superpowers/specs/2026-09-07-model-onboarding-compute-cost-design.md
  docs/superpowers/plans/2026-09-07-model-onboarding-compute-cost.md
  src/compute_cost/
    __init__.py
    cli.py
    config.py
    schema.py
    hardware.py
    telemetry.py
    runner.py
    scoring.py
    report.py
    evidence.py
    runtimes/
      __init__.py
      base.py
      ollama.py
  tests/
    test_config.py
    test_schema.py
    test_scoring.py
    test_evidence.py
    test_telemetry.py
    test_ollama_adapter.py
    test_runner.py
    test_report.py
    test_cli.py
```

## Acceptance Criteria

Base-v1 is complete when:

1. A new Ollama model can be run through one CLI onboarding command.
2. The run produces reproducible structured evidence and a Markdown summary.
3. Cold/warm latency, throughput, RAM, and NVIDIA VRAM/utilization/power are captured when observable.
4. Missing telemetry fields are explicitly unavailable, not guessed.
5. The deterministic capability suite scores all eight base categories that are compatible with the runtime.
6. Context sweep records its last success and first stop/failure boundary.
7. Sustained-load testing is bounded and reports throughput/resource drift and failures.
8. Failed cases generate replay artifacts and can be rerun individually.
9. Run artifacts are hashed and integrity-verifiable.
10. `compare` can compare at least two completed runs without rerunning models.
11. Unit/integration tests use mocked runtime/telemetry and pass without requiring Ollama or an NVIDIA GPU.
12. The repository contains no user benchmark results, secrets, or machine-specific paths.
13. The complete test suite is green on a clean Python 3.11+ environment.
