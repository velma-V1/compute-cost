# Four-Model Capability Showdown Design

Date: 2026-09-09
Status: Approved design, implementation pending

## Purpose

Add a fair, reproducible four-model showdown to `compute-cost` using the existing `benchmarks/gpt-oss-20b-capability-v1.json` cases unchanged.

The showdown answers two separate questions:

1. Which model solves more of the exact same capability tasks?
2. Which model provides the most practical value on this specific PC once latency and resource cost are included?

The comparison must never convert a slow answer into a semantic capability failure.

## Model roster

The initial fixed roster is:

1. `gpt-oss:20b` — Ollama
2. `qwen3.5:35b-a3b-q4_K_M` — Ollama
3. `devstral-small-2:24b-instruct-2512-q8_0` — Ollama
   - observed local Ollama model ID: `691a8e03a6dd`
   - observed local size: approximately 25 GB
4. Qwen3-Next-80B-A3B-Instruct Q4_K_M — Oversized MoE runtime through its persistent OpenAI-compatible server

The 80B model path and endpoint are configuration, not benchmark content. The benchmark must not depend on a machine-specific absolute path committed to the repository.

## Benchmark invariant

The authoritative showdown fixture remains:

`benchmarks/gpt-oss-20b-capability-v1.json`

The showdown must not rewrite, simplify, expand, reorder, or otherwise specialize prompts for any model.

For a given showdown campaign all four models receive:

- identical case definitions
- identical prompt text
- identical expected outputs
- identical scorers
- identical temperature and deterministic settings where the runtime exposes equivalent controls
- identical generation envelopes unless a runtime cannot represent an option, in which case the mismatch is recorded explicitly
- identical case order

The existing file name is historical provenance only. The cases are treated as model-agnostic showdown fixtures.

Each run stores the exact benchmark snapshot and its digest so later comparisons can prove that all four models saw the same fixture version.

## Runtime architecture

Introduce a runtime-neutral showdown path instead of forcing every model through Ollama.

```text
                        capability-showdown
                               |
                    fixed benchmark snapshot
                               |
              +----------------+----------------+
              |                |                |
          OllamaAdapter    OllamaAdapter    OversizedMoEAdapter
              |                |                |
        GPT-OSS 20B      Qwen 35B /       Qwen3-Next 80B
                         Devstral 24B
```

### Ollama models

The three Ollama models use the existing `OllamaAdapter` and existing evidence-capture conventions.

### 80B model

Add an `OversizedMoEAdapter` implementing the runtime interface needed by the showdown runner against the persistent `oversized-moe serve` OpenAI-compatible endpoint.

The adapter must not own model loading policy. `oversized-moe` remains responsible for mmap, expert residency, fit policy, and CUDA/CPU placement.

The adapter records server/runtime identity and request/response evidence, but does not pretend Ollama-only fields exist. Unsupported telemetry fields are represented as unavailable rather than synthesized.

The showdown assumes the 80B server is already started for the first implementation. Automatic server lifecycle management is out of scope unless later evidence proves it is necessary.

## Execution model

Models run sequentially. Only one contender should be active as the tested model at a time.

For each model:

```text
preflight
  -> verify model/runtime availability
  -> warmup
  -> execute every fixed capability case once
  -> score each case
  -> preserve raw evidence
  -> derive model capability summary
  -> verify run evidence
```

After all four model runs complete:

```text
four verified model runs
  -> showdown comparison
  -> capability leaderboard
  -> machine-utility leaderboard
  -> per-family separation map
  -> escalation recommendations
```

The showdown is a breadth comparison, not the adaptive L0-L10 frontier campaign. Adaptive frontier tests are a later follow-up only for families where the fixed showdown reveals useful separation.

## Capability score versus machine utility

The output must keep semantic capability and machine practicality separate.

### Capability result

A case is evaluated only by its existing scorer and expected result.

Examples:

- `PASS`
- `FAIL`
- `INVALID_TEST`
- `RUNTIME_ERROR`
- `CAPTURE_GAP`

A response that is correct after 150 seconds is still a capability pass if execution was allowed to complete.

### Utility result

Record the cost of obtaining that capability separately, including when observable:

- cold/load time
- warm/request latency
- prompt processing rate
- decode/generation rate
- total case wall time
- RAM
- VRAM
- GPU utilization
- GPU power samples
- process CPU and I/O where supported
- timeout-budget overruns
- runtime/backend type

The existing 120-second fixture timeout is treated as a utility threshold for cross-model comparison, not automatically as semantic incapability for a model known to run more slowly. The showdown runner therefore needs a larger hard safety ceiling configurable independently from the benchmark's utility timeout. If a model exceeds the fixture timeout but finishes before the hard ceiling, score the answer semantically and mark `UTILITY_TIMEOUT_EXCEEDED=true`.

If the hard safety ceiling is reached, classify the model call as a runtime/time-budget failure; do not infer that the model lacked the underlying capability.

## Comparison outputs

Create a campaign-level showdown artifact with at least:

### Overall capability leaderboard

- passed cases / valid cases
- percentage
- mechanically scoreable aggregate
- per-family scores
- invalid/test-fault counts kept separate

### Machine-utility leaderboard

Do not collapse this into a single unexplained magic score. Report the component measurements directly and, if a derived value score is produced, preserve the formula and source fields.

At minimum report:

- median successful-case wall time
- total showdown wall time per model
- median generation rate where comparable
- load/warmup cost
- peak observed RAM/VRAM where available
- number of utility-timeout overruns

### Separation map

For every capability family, identify patterns such as:

- all models pass
- only 20B fails
- 20B and Devstral fail while 35B/80B pass
- only 80B passes
- 35B beats 80B
- Devstral wins a coding/debugging family
- all models fail

This map determines where expensive adaptive frontier testing is worth doing next.

## Fairness rules

1. No model-specific prompt repairs during the showdown.
2. No retry after a semantic fail in the fixed breadth score.
3. Runtime faults may be rerun only as explicit recovery/replay evidence and must not silently overwrite the first attempt.
4. One model cannot receive a larger reasoning prompt or hidden helper instruction than another.
5. Backend-specific chat templates are permitted only as the normal model/runtime interface; the user prompt content remains identical.
6. Deterministic settings must be requested where supported; unsupported settings are recorded.
7. Exact model identity, quantization/tag, runtime identity, and benchmark digest are retained with every run.
8. The campaign must fail visibly if a requested contender is missing; it must not silently substitute another quant or tag.

## CLI

Add a dedicated command rather than overloading adaptive characterization:

```text
compute-cost capability-showdown
```

Initial defaults use the four-model roster above and the existing capability suite.

Useful overrides should be narrow and explicit, for example:

```text
--suite PATH
--ollama-endpoint URL
--oversized-endpoint URL
--hard-timeout-s N
```

A future `--models` override can be added if there is a demonstrated need, but the first version should optimize for one reproducible named campaign rather than become a generic orchestration framework.

## Evidence layout

Each contender keeps an independent evidence run using existing lossless evidence rules.

The showdown campaign adds a small campaign directory containing references to the four completed runs rather than duplicating their raw evidence.

Suggested layout:

```text
results/showdowns/<showdown-id>/
  showdown.json
  benchmark-digest.json
  models.json
  comparison.json
  separation-map.json
  report.md
```

`showdown.json` records the four run IDs and their backend types. A model run is eligible for final comparison only if its evidence manifest verifies.

## Failure behavior

- Missing Ollama model: stop before model execution and report exact missing tag.
- Unreachable 80B endpoint: stop that contender with runtime-unavailable evidence; do not substitute Ollama.
- Scorer defect: exclude from capability denominator and preserve evidence.
- Runtime error: preserve first attempt; optionally create a replay/recovery run, never replace history.
- Hard timeout: record runtime/time-budget failure distinct from semantic fail.
- Manifest failure: campaign comparison must mark the contender invalid until evidence integrity is restored.

## Testing strategy

Implementation follows TDD.

Required deterministic tests include:

1. exact four-model default roster, including `devstral-small-2:24b-instruct-2512-q8_0`
2. runtime selection routes three models to Ollama and the 80B contender to `OversizedMoEAdapter`
3. showdown uses the existing capability suite without mutating case content
4. all contenders receive identical case order and prompt content
5. a correct response exceeding 120 seconds remains a semantic PASS while recording utility timeout overrun
6. hard safety timeout does not become semantic FAIL
7. missing contender fails visibly with exact requested model identity
8. one contender's runtime failure cannot overwrite another contender's run
9. only manifest-verified runs enter final comparison
10. separation-map classifications are deterministic
11. comparison keeps invalid/scorer/runtime results out of semantic capability denominators
12. fake adapters allow the full four-model orchestration to run in CI without Ollama, CUDA, or the 80B model

## Non-goals

This change does not:

- modify `gpt-oss-20b-capability-v1.json`
- tune any model during the showdown
- run the entire L0-L10 frontier ladder for all four models
- add Inverted prompting or Brain interventions
- automatically optimize the 80B runtime
- automatically start/stop the 80B server in v1
- declare the largest model the winner by parameter count
- invent telemetry unavailable from a backend

## Success criteria

The feature is complete when one command can execute or deterministically simulate the four-model campaign, preserve independent evidence for each contender, and produce a comparison proving:

- what each model solved
- where the models differ
- how much each answer cost on this machine
- whether the 80B's capability gains, if any, justify its latency/resource penalty
- whether Devstral Small 2 24B Q8 earns a specialist or general role relative to GPT-OSS 20B and Qwen 35B

The benchmark questions themselves must remain unchanged throughout the comparison.
