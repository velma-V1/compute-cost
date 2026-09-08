# Model Onboarding + Compute Cost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows/NVIDIA/Ollama-first local model onboarding benchmark that preserves lossless evidence and produces complete compute-cost and deterministic baseline-capability reports.

**Architecture:** A Python 3.11+ CLI drives a runtime-agnostic runner through an Ollama adapter, continuously samples host/NVIDIA telemetry, writes append-only/raw evidence before normalization, scores deterministic benchmark cases, performs bounded context and sustained-load probes, then generates summary/compare/replay views from retained evidence. Raw artifacts and unknown fields are content-hashed and never replaced by derived summaries.

**Tech Stack:** Python 3.11+, stdlib (`argparse`, `urllib`, `subprocess`, `hashlib`, `json`, `tomllib`, `tempfile`, `platform`, `time`), `psutil`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-07-model-onboarding-compute-cost-design.md` plus required extension `docs/superpowers/specs/2026-09-07-lossless-evidence-contract.md`.

## Global Constraints

- Windows 11 is a first-class target.
- Base-v1 requires no paid API or network service other than the user's local Ollama endpoint and optional Ollama model pull.
- Every observable benchmark datum is retained in original form before normalization.
- Unknown runtime fields are retained.
- Missing sensors/capabilities produce explicit `unavailable`/`CAPTURE_GAP` evidence rather than fabricated values.
- Raw result data is gitignored.
- Secrets, authorization headers, arbitrary environment variables, and unrelated host files are never captured.
- Context/stress execution is bounded by configured time/resource limits.
- Unit/integration tests must not require Ollama or an NVIDIA GPU.

---

### Task 1: Project skeleton, schemas, and evidence store

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/compute_cost/__init__.py`
- Create: `src/compute_cost/schema.py`
- Create: `src/compute_cost/evidence.py`
- Test: `tests/test_evidence.py`
- Create: `.github/workflows/test.yml`

**Interfaces:**
- Produces: `EvidenceStore(root: Path, run_id: str)`, `write_json()`, `append_jsonl()`, `write_raw()`, `record_capture_gap()`, `finalize_manifest()`, `verify_manifest()`.
- Produces typed status/failure constants and JSON-safe record helpers in `schema.py`.

- [ ] Write tests proving raw bytes/text are retained exactly, JSONL is append-only, SHA-256/byte size are inventoried, unknown dictionary fields survive capture, and manifest verification detects mutation/missing files.
- [ ] Run CI/test and verify RED because `compute_cost.evidence` does not exist.
- [ ] Implement only the evidence/schema functionality required by those tests.
- [ ] Run tests and verify GREEN.
- [ ] Commit task.

### Task 2: Configuration and hardware/runtime preflight

**Files:**
- Create: `config/default.toml`
- Create: `src/compute_cost/config.py`
- Create: `src/compute_cost/hardware.py`
- Test: `tests/test_config.py`
- Test: `tests/test_hardware.py`

**Interfaces:**
- Produces: `load_config(path=None, overrides=None) -> dict`.
- Produces: `collect_hardware_snapshot() -> dict` and `run_command_capture(argv, timeout) -> dict` preserving stdout/stderr/return code/timing.

- [ ] Write tests for defaults/overrides, safe serialization, unavailable fields, and captured subprocess evidence.
- [ ] Verify RED.
- [ ] Implement minimal config/hardware functions using stdlib + psutil and best-effort Windows commands.
- [ ] Verify GREEN and commit.

### Task 3: Lossless telemetry sampler

**Files:**
- Create: `src/compute_cost/telemetry.py`
- Test: `tests/test_telemetry.py`

**Interfaces:**
- Produces: `TelemetrySampler.sample() -> dict`, `parse_nvidia_csv(text) -> list[dict]`, `integrate_power_wh(samples) -> float | None`.
- Every sample includes normalized host/GPU fields plus raw collector output/error and timestamps.

- [ ] Write tests for NVIDIA CSV parsing, preservation of unknown/raw fields, unavailable GPU handling, and trapezoidal Wh integration.
- [ ] Verify RED.
- [ ] Implement sampler with psutil and `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`.
- [ ] Verify GREEN and commit.

### Task 4: Runtime protocol and Ollama adapter

**Files:**
- Create: `src/compute_cost/runtimes/__init__.py`
- Create: `src/compute_cost/runtimes/base.py`
- Create: `src/compute_cost/runtimes/ollama.py`
- Test: `tests/test_ollama_adapter.py`

**Interfaces:**
- Produces: runtime protocol methods `health()`, `version()`, `model_info(model)`, `is_model_available(model)`, `generate(model, messages, options, stream=True)`, `unload(model)`, `pull(model)`.
- `generate()` returns an evidence envelope containing adapter request, raw request body, every raw streaming event/chunk, normalized response, unknown response fields, timings, and errors.

- [ ] Write mocked HTTP tests proving exact request payload retention, chunk sequence/timestamps, timing fields, unknown response-field retention, and raw error-body capture.
- [ ] Verify RED.
- [ ] Implement adapter using stdlib HTTP/JSON; no external Ollama Python SDK dependency.
- [ ] Verify GREEN and commit.

### Task 5: Deterministic base-v1 benchmark and scorers

**Files:**
- Create: `benchmarks/base-v1.json`
- Create: `src/compute_cost/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Produces: `score_case(case, response, workspace=None) -> ScoreResult` with complete check/subscore evidence.
- Scorers: exact/normalized text, JSON schema-lite validation, extraction set, numeric answer, tool-call shape, ambiguity decision, long-context planted-fact retrieval, bounded Python coding tests.

- [ ] Write tests for all eight categories and scorer-error separation from model failure.
- [ ] Verify RED.
- [ ] Implement deterministic scorers and a compact benchmark containing at least two cases per non-context category plus generated context cases.
- [ ] Coding scorer records source, executed source, test source, command, stdout, stderr, exit code, duration, timeout, and hashes.
- [ ] Verify GREEN and commit.

### Task 6: Runner lifecycle, context sweep, sustained load, and failure snapshots

**Files:**
- Create: `src/compute_cost/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Produces: `BenchmarkRunner.onboard(model) -> Path`, `run_case(case)`, `run_context_sweep()`, `run_sustained_load()`, `replay_case(run_id, case_id)`.

- [ ] Write fake-runtime/fake-telemetry tests for lifecycle ordering, cold/warm separation, every-case evidence, context last-success/first-stop boundaries, bounded sustained load, capture gaps, and replay artifacts containing exact failure-point inputs/raw output/telemetry/config.
- [ ] Verify RED.
- [ ] Implement lifecycle using the evidence store as the first write destination before scoring/summary transformations.
- [ ] Verify GREEN and commit.

### Task 7: Reporting, cost accounting, and comparison

**Files:**
- Create: `src/compute_cost/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Produces: `build_summary(run_dir) -> dict`, `render_report(summary) -> str`, `compare_runs(run_dirs) -> dict`.

- [ ] Write tests for measured/derived/estimated classification, energy/electricity math, capability/category scores, capability-per-second/VRAM/Wh, evidence-byte count, context boundary, sustained-load drift, and two-run comparison.
- [ ] Verify RED.
- [ ] Implement report strictly from retained evidence; reports never mutate/delete source evidence.
- [ ] Verify GREEN and commit.

### Task 8: CLI and operator UX

**Files:**
- Create: `src/compute_cost/cli.py`
- Create: `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Commands: `preflight`, `onboard`, `benchmark`, `compare`, `replay`, `verify`.

- [ ] Write CLI tests for parsing, safe defaults, exit status, run-directory output, compare/replay/verify dispatch, and explicit `--pull` behavior.
- [ ] Verify RED.
- [ ] Implement CLI and README with Windows/Ollama setup, exact commands, evidence layout, safety boundaries, and metric definitions.
- [ ] Verify GREEN and commit.

### Task 9: Full verification and release gate

**Files:**
- Modify only files required by discovered defects.

- [ ] Run complete `pytest` on Python 3.11+.
- [ ] Run compile/import checks.
- [ ] Validate `benchmarks/base-v1.json` loads and every case has unique ID/category/scorer/time/output limits.
- [ ] Run a fake-runtime end-to-end onboarding test and verify manifest integrity.
- [ ] Scan repository for accidentally committed `results/`, machine-specific paths, credential-looking strings, and placeholder markers.
- [ ] Confirm every acceptance criterion in both specs has a corresponding passing test or explicit integration-only verification command.
- [ ] Commit any verification fixes and leave branch ready for merge/review.
