# compute-cost

Lossless local-model onboarding, compute-cost measurement, and adaptive behavioral characterization for deciding how a local model should actually be operated on a specific PC.

`compute-cost` treats raw evidence as authoritative. Summaries, capability frontiers, and operating recommendations are derived views that must remain traceable to exact runtime evidence.

## Two run modes

### Onboarding benchmark

A full onboarding run executes:

```text
preflight -> cold start -> warmup -> base-v1 capability -> context sweep -> sustained load -> report
```

Use it to measure machine fit, baseline capability, load behavior, throughput, context scaling, resource use, and cost.

### Adaptive characterization

A characterization run executes one model independently and adaptively spends calls near informative behavioral boundaries:

```text
preflight
  -> thinking OFF baseline
  -> thinking ON baseline
  -> adaptive budget probes
  -> boundary replication
  -> replay/recovery lineage
  -> task operating profile
  -> report
```

For the first Qwen characterization target:

```powershell
.\.venv\Scripts\compute-cost.exe characterize --model "qwen3.5:27b-q4_K_M"
```

The default characterization suite is `benchmarks/qwen-characterization-v1.json`. It deliberately does **not** hard-code output budgets. Generation budget is experiment metadata controlled by the adaptive search.

Characterization preserves explicit `think: false` and `think: true` requests when Ollama supports them. Exposed thinking and final content are retained separately.

Important interpretation rules:

- exposed thinking is an **observable behavioral trace**, not proof of complete hidden neural cognition;
- Ollama `eval_count` is retained as aggregate generated-token evidence and is **not** presented as an exact thinking-token count;
- `THINK_TRUNCATED` means the generation allowance ended while exposed reasoning was still being emitted and no usable final answer had appeared; it is not counted as a semantic capability failure;
- malformed JSON/tool/extraction output is a model-format failure when the scorer itself worked;
- `SCORER_DEFECT`, runtime faults, capture gaps, and test defects are retained but excluded from semantic capability frontiers;
- adaptive runs can add probes after they begin, so progress-denominator changes are persisted in `progress.jsonl` instead of being hidden.

## What one onboarding run measures

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

Raw run evidence may contain machine-specific metadata such as host/runtime paths and hardware identifiers. `results/` is therefore local and gitignored; inspect evidence before deliberately sharing it.

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

# Adaptively characterize exactly one model
compute-cost characterize --model <ollama-model-tag>

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

## Adaptive Qwen characterization

Phase 1 begins with three deterministic task families—exact instruction control, strict JSON, and reasoning/math—to validate the experimental machinery before expanding the capability map.

For each task the controller establishes thinking-OFF and thinking-ON baselines, then changes only one controlled variable at a time. Generation-budget probes bracket the transition between lower-budget failure/truncation and reproduced success. Near a discovered boundary, the same configuration is repeated to distinguish a single pass from a reliable operating point.

Every model call receives a unique experiment ID and evidence key. `experiments.jsonl` records:

- experiment identity and parent lineage
- task family and difficulty
- hypothesis and changed variable
- thinking mode
- generation budget
- seed/temperature
- result classification and capability-validity flag
- scorer result
- runtime metrics and observable phase metrics
- raw-evidence references

A budget increase following `THINK_TRUNCATED` or `ANSWER_TRUNCATED` is retained as an `R1` recovery hypothesis rather than an anonymous retry.

The run derives `characterization-summary.json` and `characterization-report.md` only after preserving the individual experiments. A derived minimum passing budget requires repeated success at that budget; unknown boundaries remain unknown rather than being guessed.

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
  progress.jsonl
  cases.jsonl                      # onboarding runs
  experiments.jsonl                # characterization runs
  characterization-events.jsonl    # characterization task-stop evidence
  summary.json
  report.md
  characterization-summary.json    # characterization runs
  characterization-report.md       # characterization runs
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
    harness/
```

Raw request/response and collector bytes are retained independently of parsed JSON. Unknown fields are kept. Normalization never overwrites the original observation.

`manifest.json` inventories retained artifacts with SHA-256 and byte size. Run:

```powershell
compute-cost verify <run-id>
```

to detect missing, modified, or unexpected post-finalization evidence.

## Cost interpretation

The report separates four evidence classes:

- **MEASURED** — directly observed values such as client latency, runtime-reported counts/durations, sampled RAM/VRAM, GPU power, process I/O, and observable thinking/answer chunks
- **DERIVED** — calculations over measured evidence such as tokens/s, Wh, transition brackets, minimum reproduced passing budgets, throughput drift, and efficiency ratios
- **ESTIMATED** — explicit assumptions or approximations
- **UNAVAILABLE** — data the active interfaces did not expose reliably

Do not interpret host RAM or total sampled GPU power as perfectly isolated model-only consumption if unrelated workloads are active. For clean comparisons, close unrelated GPU/CPU-heavy workloads and use the same benchmark version/configuration.

## Why raw evidence is kept

A benchmark result is not just a final score. Every failure can become a smaller research experiment later.

Onboarding failures retain replay snapshots containing the exact case definition, invocation, generation envelope, scorer evidence, recent telemetry, and resolved configuration. Characterization uses replay schema v2 to add experiment identity, parent/recovery lineage, classification, and unique evidence key.

This allows the same model, another quant, another architecture, or a changed prompting/system strategy to retest the exact failure without paying to reconstruct or rerun unrelated work.

## Development gate

Repository CI runs the full test suite on Windows with Python 3.11 and 3.12. Tests use fake runtimes/telemetry and require neither Ollama nor an NVIDIA GPU.

Before another real Qwen call, Phase 1 requires:

- deterministic fake-runtime tests green;
- fresh Windows Python 3.11 and 3.12 CI green on the exact implementation head;
- the old Test #1 thinking-only length exhaustion reproduced as `THINK_TRUNCATED`, not semantic failure;
- unique evidence for repeated experiments;
- a synthetic FAIL/truncation-to-PASS boundary adaptively bracketed and reproduced;
- characterization manifest verification green.

Design contracts:

- `docs/superpowers/specs/2026-09-07-model-onboarding-compute-cost-design.md`
- `docs/superpowers/specs/2026-09-07-lossless-evidence-contract.md`
- `docs/superpowers/specs/2026-09-07-adaptive-qwen-characterization-design.md`
- `docs/superpowers/plans/2026-09-07-qwen-characterization-phase1.md`
