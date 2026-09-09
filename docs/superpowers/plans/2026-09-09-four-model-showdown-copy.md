# Four-Model Showdown Copy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the existing GPT-OSS capability campaign and existing sequential campaign pattern to run the same capability suite against GPT-OSS 20B, Qwen3.5 35B, Devstral Small 2 24B Q8, and Qwen3-Next 80B.

**Architecture:** Do not build a new benchmark engine. Keep `benchmarks/gpt-oss-20b-capability-v1.json` and its scorers unchanged. Copy the existing `characterize-campaign` orchestration pattern into a new `capability-showdown` command that calls the already-implemented `BenchmarkRunner.capability_characterize()`. The first three models use `OllamaAdapter`; the 80B uses one small OpenAI-compatible adapter pointed at the already-running `oversized-moe serve` endpoint.

**Tech Stack:** Python 3.11+, pytest, urllib, existing `compute_cost` runner/scorers/evidence, Ollama, oversized-moe/llama.cpp OpenAI-compatible HTTP server.

**Spec:** `docs/superpowers/specs/2026-09-09-four-model-capability-showdown-design.md`

## Global Constraints

- Do not modify capability prompts, expected answers, scorers, or case order.
- Fixed roster: `gpt-oss:20b`, `qwen3.5:35b-a3b-q4_K_M`, `devstral-small-2:24b-instruct-2512-q8_0`, `qwen3-next-80b-a3b-instruct-q4_k_m`.
- Devstral local identity supplied by user: `691a8e03a6dd`, approximately 25 GB; do not substitute another tag.
- 80B is served externally; the adapter does not start, stop, or reconfigure oversized-moe.
- No new benchmark framework and no prompt duplication.
- Existing commands remain behaviorally unchanged.

---

### Task 1: 80B runtime adapter

**Files:**
- Create: `src/compute_cost/runtimes/oversized_moe.py`
- Modify: `src/compute_cost/runtimes/__init__.py`
- Create: `tests/test_oversized_moe_runtime.py`

**Interfaces:**
- Produces `OversizedMoEAdapter(endpoint: str = "http://127.0.0.1:8080", timeout_s: float = 600.0, served_model: str = "qwen3-next-80b-a3b-instruct-q4_k_m", transport=None)`.
- Implements the existing runtime protocol methods used by `BenchmarkRunner.capability_characterize`: `version`, `list_models`, `model_info`, `model_available_in`, `is_model_available`, `generate`, `pull`, `unload`.
- `generate()` POSTs to `/v1/chat/completions`, maps `num_predict` to `max_tokens`, passes `temperature` and `seed` when present, and normalizes `choices[0].message.content` into `normalized.text`.

- [ ] **Step 1: Write RED tests** proving health/model availability and OpenAI response normalization without network access.
- [ ] **Step 2: Run the runtime test and require failure because the adapter does not exist.**
- [ ] **Step 3: Copy the existing lossless urllib envelope pattern from `OllamaAdapter` and implement only the protocol surface above.**
- [ ] **Step 4: Re-run the runtime test and require PASS.**

---

### Task 2: Copy the sequential campaign path

**Files:**
- Modify: `src/compute_cost/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Adds `SHOWDOWN_MODELS` with exact `(model, backend)` entries.
- Adds command: `compute-cost capability-showdown`.
- Defaults to `benchmarks/gpt-oss-20b-capability-v1.json` and `benchmarks/capability-taxonomy-v1.json`.
- Adds `--oversized-endpoint`, default `http://127.0.0.1:8080`, and `--hard-timeout-s`, default `600`.
- Reuses suite validation/materialization/normalization from `capability-characterize` once, then creates a fresh runner per contender.
- Calls `runner.capability_characterize(model, pull=False)` sequentially and verifies each manifest before starting the next model.

- [ ] **Step 1: Copy the existing `characterize-campaign` tests into showdown-focused RED tests asserting exact four-model order and backend routing.**
- [ ] **Step 2: Run CLI tests and require RED because `capability-showdown` is missing.**
- [ ] **Step 3: Copy the existing campaign loop, change only roster, completion event, runtime selection, and capability method.**
- [ ] **Step 4: Re-run CLI tests and require PASS.**

---

### Task 3: Verification

**Files:**
- No production scope expansion unless tests reveal a defect.

- [ ] **Step 1: Run targeted runtime + CLI tests.**
- [ ] **Step 2: Run the full pytest suite.**
- [ ] **Step 3: Open/update a PR so Windows Python 3.11 and 3.12 CI runs.**
- [ ] **Step 4: Require both CI jobs green before calling the test ready.**
