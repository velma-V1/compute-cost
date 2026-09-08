# compute-cost

Lossless local-model onboarding and compute-cost benchmark for deciding whether a new model is worth keeping on a specific PC.

`compute-cost` measures **machine fit + inference cost + deterministic baseline capability** in one repeatable run. Raw evidence is authoritative; summaries are derived views.

## What one onboarding run measures

A full run executes:

```text
preflight -> cold start -> warmup -> base-v1 capability -> context sweep -> sustained load -> report
```

It records, when observable:

- model identity, runtime version, model metadata, model storage size
- whether the model was already local and all observed pull/download progress when a pull is requested
- cold load duration, first observable streamed event, and end-to-end latency
- warm load/latency
- prompt token count/rate and generated token count/rate
- host RAM usage
- identified Ollama process RAM, CPU, CPU time, I/O counters, threads, and Windows handle count
- NVIDIA GPU identity, driver, VRAM, utilization, temperature, power, clocks, p-state, and fan speed
- every telemetry sample plus exact `nvidia-smi` stdout/stderr/exit state
- bounded context scaling and the observed prompt token count returned by the runtime
- bounded sustained-load throughput/resource drift
- deterministic capability scores
- every raw model request, streamed response event, final response, exposed thinking/reasoning fields, tool-call fields, runtime timings, warnings/errors, and unknown future response fields
- scorer inputs, checks, outputs, exceptions, and coding-test subprocess evidence
- exact failure-point replay snapshots
- SHA-256 integrity information for every retained run artifact

### Observable-data boundary

"Lossless" means **everything the benchmark can observe through the interfaces it uses is retained before normalization**. It does not claim access to model internals, hidden runtime state, CPU package power, or sensors that the operating system/runtime does not expose.

Missing channels are represented as unavailable or `CAPTURE_GAP`; they are not silently omitted or guessed.

Secrets are intentionally excluded. The benchmark does not dump arbitrary environment variables, auth headers, cookies, credentials, or unrelated files.

## Install

Requirements:

- Windows 11 is the primary target
- Python 3.11+
- Ollama running locally
- NVIDIA GPU recommended for GPU telemetry; the benchmark still runs if `nvidia-smi` is unavailable

From the repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest
```

## Run a new-model onboarding test

For a model that is already installed in Ollama:

```powershell
compute-cost onboard --model <ollama-model-tag>
```

If the model is not local and you explicitly want the benchmark to pull it so download/onboarding evidence is captured:

```powershell
compute-cost onboard --model <ollama-model-tag> --pull
```

To include electricity-price accounting for measured GPU energy:

```powershell
compute-cost --electricity-per-kwh 0.15 onboard --model <ollama-model-tag>
```

The electricity figure is based on **sampled NVIDIA GPU power**, not whole-wall system power. CPU/system energy is not invented when no supported sensor exposes it. For whole-machine electricity cost, use a wall-power collector in a future adapter or compare the retained benchmark timeline with an external meter.

## Other commands

```powershell
# Hardware/runtime visibility only
compute-cost preflight

# Benchmark an already-local model
compute-cost benchmark --model <ollama-model-tag>

# Compare completed runs without running a model again
compute-cost compare <run-a> <run-b> [<run-c> ...]

# Replay one retained failed case as a separate evidence run
compute-cost replay <run-id> <case-id>

# Verify all hashes in a completed run
compute-cost verify <run-id>
```

## Base-v1 capability benchmark

`benchmarks/base-v1.json` uses mechanically scoreable tasks so new models can be compared without a model-as-judge dependency.

The eight baseline categories are:

1. instruction following
2. structured JSON output
3. extraction
4. reasoning/math
5. executable Python coding
6. tool-call-shaped output
7. ambiguity handling
8. long-context retrieval

The aggregate score is the arithmetic mean of category scores that actually executed. Category scores are always reported separately so an aggregate cannot hide a weak or missing category.

Coding cases run a deliberately restricted Python subset in an isolated temporary directory with a timeout. Imports and high-risk built-ins are rejected by the base-v1 scorer.

## Context sweep

The default runtime context targets are configured in `config/default.toml`.

For every level the run retains:

- requested runtime context window (`num_ctx`)
- synthetic prompt and planted key
- runtime-reported prompt token count/duration
- generation count/duration
- raw request/stream/response
- correctness
- telemetry around the request
- stop-boundary reason

The **runtime-reported prompt token count is the observed value**. The target window is not mislabeled as an exact token count for generated text because tokenization is model-dependent.

The sweep stops at configured resource/runtime/correctness limits rather than deliberately exhausting the machine.

## Evidence layout

Each run is local under `results/<run-id>/` and is gitignored.

```text
results/<run-id>/
  manifest.json
  resolved-config.json
  benchmark-snapshot.json
  hardware.json
  runtime.json
  events.jsonl
  telemetry.jsonl
  cases.jsonl
  summary.json
  report.md
  raw/
    runtime/
      requests/
      responses/
      streams/
      exchanges/
      errors/
    telemetry/
      nvidia/
    scoring/
  replay/
```

Raw request/response and collector bytes are retained independently of parsed JSON. Unknown fields are kept. Normalization never overwrites the original observation.

`manifest.json` inventories retained artifacts with SHA-256 and byte size. Run:

```powershell
compute-cost verify <run-id>
```

to detect missing or modified evidence.

## Cost interpretation

The report separates three evidence classes:

- **measured** — directly observed values such as client latency, runtime-reported load duration, sampled RAM/VRAM, GPU power, process I/O, and model size
- **derived** — calculations over measured evidence such as tokens/s, Wh, electricity cost, I/O deltas, throughput drift, and capability-efficiency ratios
- **estimated** — user-supplied assumptions, such as an electricity price or future hardware-amortization inputs

Do not interpret host RAM or total sampled GPU power as perfectly isolated model-only consumption if unrelated workloads are active. For clean comparisons, close unrelated GPU/CPU-heavy workloads and use the same benchmark version/configuration.

## Why raw evidence is kept

A benchmark result is not just a final score. Every failure can become a smaller research experiment later.

Failed cases write `replay/<case-id>.json` containing the exact case definition, invocation, raw generation envelope, scorer evidence, recent telemetry, and resolved configuration. That allows a failed model, another model, or a changed prompting/system strategy to retest the same failure without paying to rerun unrelated benchmark cases.

## Development gate

The repository CI runs the full test suite on Windows with Python 3.11 and 3.12. Tests use fake runtimes/telemetry and require neither Ollama nor an NVIDIA GPU.

Design contracts:

- `docs/superpowers/specs/2026-09-07-model-onboarding-compute-cost-design.md`
- `docs/superpowers/specs/2026-09-07-lossless-evidence-contract.md`
