# 700-Call Full Comparability Benchmark Design

## Purpose

Expand the existing compact capability campaign into a high-information, fully scored model characterization run with a hard ceiling of 700 model calls per model. The benchmark must maximize cross-model comparability while preserving the existing evidence-first architecture, model-specific reasoning controls, exact retry lineage, and long autonomous simulation.

The benchmark is not a leaderboard. Its purpose is to produce an operating manual for Inverted: what a model can do, where it fails, how reasoning mode changes performance, how much compute each success costs, how it behaves across long autonomous work, and which model should own which task class.

## Hard constraints

1. `max_model_calls_per_run = 700` is an absolute pre-call hard stop. No path may exceed it, including truncation retries, reasoning-mode probes, autonomous simulation, recovery diagnostics, or adaptive boundary work.
2. Every executed model call must be scored and retained. No executed attempt may exist only as an unscored log row.
3. Every family, task, difficulty, reasoning condition, retry, autonomous decision, autonomous turn, scenario, and model aggregate must have explicit comparable score output.
4. Every failure and retry must preserve the full forensic attempt dossier: exact model-visible messages, request fields/options, raw request/response refs, raw stream refs, exposed thinking when available, final answer, scorer checks, right/wrong checks, runtime status, timing, throughput, token counts, telemetry, parent/retry lineage, changed variable, and reason for retry.
5. Hidden internal reasoning that the runtime does not expose must remain explicitly `UNOBSERVABLE`; it must never be fabricated or inferred as captured chain-of-thought.
6. Unsupported reasoning controls must be reported as `UNSUPPORTED`, never simulated and never counted as model failure.
7. Token escalation follows the existing binding law: increase output budget only when the token ceiling was actually exhausted; retry the exact same task one ladder step at a time; reset to base for the next independent task/family.
8. Raw semantic capability, contract/format compliance, and tool/procedure compliance must be separate score dimensions. A formatting failure must not be reported as zero semantic capability unless the semantic answer itself is also wrong.
9. Adaptive diagnostic calls may not alter the official fixed-core comparison score. They are reported separately as diagnostic/frontier evidence.
10. Windows 11 Home, local Ollama runtime, and current evidence/replay layout remain supported. No paid API or activation dependency may be introduced.

## Call-budget architecture

### GPT-OSS fixed comparable core

The fixed core uses 40 capability families, four matched difficulty cells per family, and three native GPT-OSS reasoning efforts.

- Capability matrix: `40 families × 4 fixed tasks × 3 efforts = 480 calls`
- Autonomous matrix: `6 scenarios × 8 turns × 3 efforts = 144 calls`
- Fixed core total: `624 calls`
- Adaptive/retry reserve: `76 calls`
- Absolute total: `700 calls`

The four fixed difficulty cells are L2, L5, L8, and L10 where fixtures exist. If a required fixed cell is missing, the benchmark records `MISSING_FIXTURE_COVERAGE` for that cell and does not silently substitute a different level in the official fixed comparison matrix. Adaptive reserve may separately investigate nearby available levels.

### Qwen and other models with fewer native reasoning controls

Qwen uses its actual native controls only:

- `think=false`
- `think=true`

A model with one native control state gets one supported condition; a model with two gets two; GPT-OSS gets low/medium/high. Missing conditions are `UNSUPPORTED` rather than duplicated.

The fixed task identities remain identical across models. Unused call budget from unsupported reasoning conditions becomes diagnostic reserve, but those extra calls are not allowed to contaminate the matched fixed-core comparison score.

### Standardized reasoning-condition labels

Native controls map to standardized comparison roles without pretending that different APIs are identical:

- `BASELINE_MINIMAL`: cheapest supported/native low-reasoning condition (`gpt-oss=low`, `qwen=think:false`, model default where no lower control exists)
- `ENHANCED`: next genuinely distinct supported reasoning condition (`gpt-oss=medium`, `qwen=think:true` where supported)
- `MAX_NATIVE`: highest distinct supported reasoning condition (`gpt-oss=high`; Qwen is `UNSUPPORTED` if `think:true` is already its highest and only enabled state)

Reports always include both standardized role and exact native request fields.

## Fixed-core comparability

The benchmark must compare identical task IDs, identical prompts, identical or explicitly documented generation constraints, identical seeds where supported, and identical scoring rubrics across models. Model names must not change task semantics.

The official comparison views are:

1. `baseline_matched`: all models on the identical fixed task matrix using `BASELINE_MINIMAL`.
2. `enhanced_matched`: only models that support a genuinely distinct `ENHANCED` condition; unsupported cells remain explicit.
3. `max_native`: each model at its highest supported native reasoning condition; this is a best-available comparison and must be labeled as such rather than strictly apples-to-apples.
4. `within_model_reasoning_curve`: per-task and per-family deltas between a model's supported reasoning conditions.

No overall model ranking may average unsupported cells as zero.

## Scoring model

### Attempt score vector

Every executed call receives a structured score vector. Components are 0-100 where applicable, plus explicit status fields.

Required attempt dimensions:

- `semantic_correctness`
- `contract_format_compliance`
- `tool_procedure_compliance`
- `constraint_compliance`
- `state_checkpoint_accuracy`
- `decision_quality`
- `recovery_quality`
- `verification_quality`
- `evidence_use_quality`
- `goal_preservation`
- `reasoning_condition_status`
- `runtime_validity`
- `truncation_status`
- `latency_seconds`
- `prompt_tokens_per_second`
- `generation_tokens_per_second`
- `prompt_tokens`
- `generated_tokens`
- `total_tokens`
- `ram_peak_bytes` when measured
- `vram_peak_mib` when measured
- `gpu_energy_wh` when measured
- `wall_clock_seconds`
- `value_per_second`
- `value_per_generated_token`
- `value_per_wh` when energy is measured

A component that is not applicable must be `NOT_APPLICABLE`, not zero. A component that could not be observed must be `UNOBSERVABLE` or `UNAVAILABLE` as appropriate.

### Partial credit

Deterministic scorers must retain all individual checks. Component scores are the percentage of applicable checks passed within that component. A task may therefore be semantically correct while failing contract compliance, or may preserve state while choosing the wrong action.

The benchmark must never collapse all such cases into one binary `ANSWER_WRONG` score for higher-level reporting.

### Semantic equivalence for autonomous actions

Autonomous action scoring must support declared semantic-equivalence classes and state-aware equivalents. Examples such as `TEST` versus `RETEST` may receive equivalent decision credit when the intended operation, state transition, and rationale satisfy the same rubric. Exact-token matching remains available as a separate contract-compliance signal.

Equivalence rules must be declared in the scenario fixture/rubric, not improvised after seeing a model answer.

## Aggregation hierarchy

Every score is traceable upward through this hierarchy:

`attempt -> task -> difficulty -> reasoning condition -> family -> autonomous turn -> scenario -> model -> cross-model comparison`

### Task score

A task score includes all component scores for every attempt and every supported reasoning condition. Truncation retries remain separate attempts and also produce a resolved task result after the retry chain terminates.

### Difficulty score

Difficulty scores aggregate fixed tasks at that difficulty and supported reasoning condition. Adaptive-only diagnostic tasks are shown separately.

### Family score

Each family report must contain:

- fixed-core semantic score
- fixed-core contract score
- fixed-core procedure/tool score
- fixed-core composite score
- score by difficulty
- score by reasoning condition
- reasoning benefit/cost deltas
- reliable lower bound and observed failure upper bound where supported by evidence
- invalid/truncation counts
- latency/throughput/token/VRAM/RAM/energy summaries
- diagnostic reserve findings
- source experiment IDs
- confidence/evidence count

The current `ANSWER_CORRECT / valid observations` metric may remain as a raw observed statistic but must be labeled `observed_adaptive_accuracy` and may not serve as the official family capability score.

### Model score

Model-level comparison must use macro-aggregation over the fixed comparable cells so families with more adaptive probes do not receive extra weight. Required top-level views:

- baseline semantic capability
- baseline contract compliance
- baseline tool/procedure compliance
- baseline autonomous score
- enhanced semantic/autonomous scores where supported
- max-native semantic/autonomous scores
- reasoning gain per added second/token/Wh
- reliability/invalid rate
- throughput and latency
- state retention
- recovery
- verification
- overall value-per-compute

The report must expose the component vector next to any composite. No single composite may hide why one model won.

## Reasoning-mode characterization

For every fixed capability task, run all distinct native reasoning conditions supported by that model, subject to the 700-call ceiling.

For every reasoning-condition cell record:

- exact native control sent to runtime
- standardized condition role
- semantic/contract/procedure score
- exposed thinking availability and size/timing if returned
- final answer
- token count and truncation
- latency and throughput
- compute/resource metrics
- delta from baseline
- whether extra reasoning changed correctness
- whether extra reasoning merely increased cost
- whether extra reasoning caused regression/overthinking
- minimum condition that achieved the best observed task score

Reasoning-condition comparisons use the exact same task fixture. They may not compare two different prompts and call the difference a reasoning effect.

## Retry scoring

Every retry is an independently scored attempt and also has a delta record against its parent.

Required retry delta fields:

- parent experiment ID
- child experiment ID
- changed variable
- retry reason
- semantic score delta
- contract score delta
- tool/procedure score delta
- decision/state/recovery/verification deltas where applicable
- token delta
- latency delta
- throughput delta
- RAM/VRAM/energy delta where available
- result-class transition
- whether the retry fixed the original failure
- whether it introduced a new failure
- whether the retry was cost-effective

Exact failed attempts must never be overwritten by successful retries.

## Autonomous simulation scoring

The existing six scenarios and eight-turn growing transcripts remain the fixed autonomous core unless a fixture is explicitly versioned.

Every turn must be scored independently on all applicable dimensions:

- action/decision quality
- state/checkpoint accuracy
- goal preservation
- constraint compliance
- evidence use
- tool/procedure choice
- change/failure recovery
- verification behavior
- completion progress
- semantic correctness
- contract compliance
- latency, tokens, throughput, and resources

Every scenario must report:

- all eight turn score vectors
- cumulative state integrity
- first drift turn if any
- recovery after injected change/failure
- completion status
- completion quality
- total and median cost/latency
- reasoning-condition comparison
- failure/retry lineage
- final scenario composite plus component vector

A scenario is not marked `UNTESTED` when autonomous evidence exists.

## Diagnostic/adaptive reserve

The 76-call GPT reserve, or larger unused reserve for models with fewer native reasoning modes, is scheduled only after fixed-core coverage is protected.

Priority order:

1. token-exhaustion retries required by the binding retry law
2. missing/uncertain frontier boundary confirmation
3. suspicious L0/L2 failure diagnostic sibling fixture
4. semantic-vs-format scorer discrimination
5. reasoning-mode disagreement confirmation
6. autonomous decision ambiguity confirmation
7. highest-value cross-model disagreement probes

Reserve exhaustion stops optional diagnostics before fixed-core calls are sacrificed.

Diagnostic results are scored identically but labeled `diagnostic` and excluded from official matched-core aggregates.

## Call-ledger enforcement

Introduce a single authoritative call ledger shared by every model-calling phase. Before every runtime inference, the runner asks the ledger for authorization. When `used_calls >= 700`, the call is rejected before the runtime request is sent and a `MODEL_CALL_BUDGET_EXHAUSTED` event is persisted.

Required ledger fields:

- configured hard ceiling
- calls used
- fixed-core calls used
- reasoning-mode calls used
- autonomous calls used
- truncation-retry calls used
- diagnostic calls used
- remaining calls
- per-family/per-scenario counts
- rejected-call attempts

The final report must reconcile ledger count to retained runtime request count. A mismatch is a run-integrity failure.

## Evidence and forensic completeness

Every model call must have an attempt dossier, including successful fixed-core calls, failed calls, retries, autonomous turns, and reasoning-mode probes.

Required dossier sections:

- question/task identity and taxonomy metadata
- exact fixture snapshot and oracle/rubric
- exact model-visible messages/transcript
- exact runtime request fields and generation options
- native reasoning control and standardized condition
- raw serialized request reference
- raw response reference
- raw stream references
- exposed thinking text/reference and observability status
- final answer/tool calls
- all scorer checks, grouped by score dimension
- right/wrong/partial-credit findings
- score vector
- runtime metrics
- throughput/timing/token metrics
- telemetry/resource metrics
- lineage and retry delta
- parent/child links
- failure signature and origin classification
- replay eligibility/index references

The dossier index must allow reconstruction of an entire task's reasoning-mode matrix and retry tree without rerunning the model.

## Report outputs

The run must produce machine-readable JSON and human-readable Markdown for at least:

- `scorecard.json` / `scorecard.md`
- `task-scorecard.json`
- `reasoning-comparison.json` / `.md`
- `autonomous-simulation.json` / `.md`
- `retry-deltas.json`
- `comparison-cells.json`
- `call-ledger.json`
- `failure-atlas.json`
- `attempt-dossiers/index.jsonl`
- existing capability/frontier/cost/value evidence

The human report must include explicit tables for:

- every family
- every executed fixed task
- every supported reasoning condition
- every autonomous turn
- every retry chain
- every model-level comparison component

Large detailed tables may live in separate Markdown files linked from the top-level scorecard, but no required score may exist only in an opaque raw JSON file.

## Cross-model comparison contract

A fixed comparison cell is identified by:

`benchmark_version + family_id + task_id + difficulty_level + standardized_reasoning_role + scoring_rubric_version`

Two model results may be directly compared only when the comparison cell identity matches. Best-native comparisons must be separately labeled when native reasoning controls differ.

Each comparison cell reports:

- model score vectors side by side
- winner/tie by component
- absolute and relative latency difference
- throughput difference
- token-cost difference
- reasoning-cost difference
- reliability/invalid difference
- source experiment IDs

No score from an adaptive-only task may be silently compared against a different task from another model.

## Known defects to repair as part of this design

1. Autonomous results currently exist but synthesized characterization reports may label `autonomous_simulation` as `UNTESTED`. This must be corrected.
2. Synthesized reasoning effort may report stale/default `medium` rather than the actual model-native control. Reports must derive this from retained experiment evidence.
3. Semantic capability and strict format failure are currently conflated in some family scores. They must be separated.
4. Autonomous exact-action matching is too literal for semantically equivalent decisions. Declared equivalence classes must support partial/full semantic credit while preserving exact contract scoring.
5. Failure-atlas generation must include every valid model failure present in attempt dossiers; an empty atlas with non-empty failure dossiers is a run-integrity defect.
6. Raw adaptive accuracy must not be presented as a calibrated global capability score.

## Acceptance criteria

The expansion is accepted only when all of the following are true:

- hard ceiling is 700 and a unit/integration test proves the 701st attempted model call is blocked before runtime invocation
- every executed model call creates exactly one scored attempt dossier
- runtime request count equals call-ledger used count
- fixed comparison cells use identical task fixtures across models
- every fixed task has per-component scores
- every supported reasoning condition has per-task/per-family scores and cost deltas
- unsupported reasoning conditions are explicit and excluded from denominators
- every autonomous turn has a score vector
- every autonomous scenario has a reasoning-condition comparison
- every retry has a parent-linked delta record
- format/contract failures are separable from semantic correctness
- semantic-equivalent autonomous actions can receive semantic credit while exact-token compliance remains separately measurable
- family aggregates are macro-comparable and not biased by adaptive probe count
- adaptive/diagnostic reserve results cannot alter fixed-core official scores
- autonomous evidence appears in the main report when present
- actual native reasoning controls appear correctly in reports
- failure atlas reconciles with failure dossiers
- all required JSON/Markdown reports are emitted
- no personal information is introduced into the repository
- existing lossless-evidence, replay, and token-escalation contracts remain intact
- full tests pass on supported Python versions

## Completion rule

A model run is complete when the fixed supported comparison matrix and autonomous matrix have either been executed or explicitly marked unsupported/missing, all executed attempts are scored and forensically indexed, the call ledger reconciles with retained requests, all report surfaces are generated, and the runner has not exceeded 700 calls.

`RUN_COMPLETE` does not imply every possible L0-L10 fixture or every hypothetical reasoning condition was executed. Unexecuted cells remain explicit and are never imputed. The benchmark's guarantee is that every executed item is completely scored and comparable, while the fixed core provides identical model-to-model comparison coverage within the finite 700-call budget.
