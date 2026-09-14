# Test 1.2 Zero-Clock Model-Building Refinery

Date: 2026-09-12

This pass looks specifically for ways to improve the **model itself** using
evidence Test 1.2 already collected. No additional model inference, runtime
call, campaign phase, or active-test second is permitted.

The refinery runs as deterministic finalization over existing DISCOVERY
observations.

## Research basis

### Failure data should not be discarded

- *AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling*
  (arXiv:2603.21357, 2026) shows that failed agent trajectories contain useful
  training signal and can be repackaged into SFT and preference data rather
  than discarded.
- *Learning from Failure: Inference-Time Self-Improvement for Computer-Use
  Agents* (arXiv:2606.31270, 2026) demonstrates that failure evidence can
  materially improve agent performance instead of serving only as a score.

### Preference quality matters more than raw pair count

- *Hard Negative Sample-Augmented DPO Post-Training for Small Language Models*
  (arXiv:2512.19728, 2025) reports that structured, verifier-informed hard
  negatives and sample weighting are more targeted than unweighted preference
  tuning.
- *Contrastive Instruction Tuning* (arXiv:2402.11138, 2024) motivates
  same-task/near-task hard negatives rather than trivial far-OOD negatives.

### Distillation supervision should be weighted by reliability

- *When Are Teacher Tokens Reliable? Position-Weighted On-Policy
  Self-Distillation for Reasoning* (arXiv:2605.21606, 2026) finds that
  distillation supervision is not uniformly valuable and that weighting
  existing supervision can improve reasoning without extra teacher compute.

### Fine-tuning must preserve already-correct behavior

- *Preventing Catastrophic Forgetting: Behavior-Aware Sampling for Safer
  Language Model Fine-Tuning* (arXiv:2510.21885, 2025) shows that a small,
  targeted rehearsal mixture can preserve important prior behaviors more
  efficiently than random replay.

### Routing itself is trainable behavior

- *When Are Experts Misrouted? Counterfactual Routing Analysis in
  Mixture-of-Experts Language Models* (arXiv:2605.07260, 2026) finds that
  GPT-OSS-20B and other MoE models contain equal-compute alternative routes
  whose utility differs strongly on fragile reasoning tokens, and that
  router-only updates can improve performance.
- Test 1.2 applies the analogous principle at the **harness/control routing**
  level: train from measured action utility rather than hard-code every control.

## Twelve zero-clock products

### 1. Harness-to-weight distillation

When the raw model fails a task and a measured harness intervention rescues it,
Test 1.2 already owns both responses.

The refinery creates:

`original raw task -> successful treatment response`

The intervention prompt itself is not included in the training input. This
creates a direct path for later SFT to internalize behavior that previously
required an external control.

Artifact:

`harness-to-weight-distillation-corpus.jsonl`

Use:
- SFT;
- LoRA/QLoRA;
- capability-specific adapter training;
- later self-distillation.

### 2. Weighted hard-negative preferences

Every treatment comparison already supplies same-task alternatives.

If treatment improves quality:
- chosen = treatment;
- rejected = raw response.

If treatment regresses:
- chosen = raw response;
- rejected = harmful treatment.

If quality ties:
- chosen = lower-call/token/latency response.

Weights increase for measured score gaps and harder fixtures.

Artifact:

`weighted-preference-corpus.jsonl`

Use:
- DPO;
- ORPO;
- KTO-style conversion;
- reward-model/preference-head training.

This converts negative effects directly into model-improvement data.

### 3. Capability curriculum

The refinery computes family-level priority from:
- baseline weakness;
- hard-case failure rate;
- measured rescueability;
- instability;
- negative-transfer risk.

It also labels measured difficulty levels as anchor/frontier/weakness regions.

Artifact:

`capability-curriculum.json`

Use:
- training-mixture weights;
- easy-to-frontier progression;
- more gradient budget for rescueable weaknesses;
- less oversampling of already-saturated capabilities.

### 4. Router/activation supervision

For each measured fixture, the refinery compares:
- DIRECT;
- every observed control.

The label is the highest-quality action, with calls/tokens/latency breaking
quality ties.

Artifact:

`router-supervision-corpus.jsonl`

Use:
- small router model;
- classifier head;
- policy distillation;
- future internal routing supervision.

This is how Test 1.2 can teach **when not to use a harness block**, not just how
to use one.

### 5. Stability-anchor rehearsal corpus

Every raw-model pass is a potential asset that later fine-tuning can erase.
Passes exposed to observed regressing controls receive higher rehearsal weight.

Artifact:

`stability-anchor-corpus.jsonl`

Use:
- SFT replay;
- mixed fine-tuning anchors;
- catastrophic-forgetting protection;
- regression-sensitive curriculum.

### 6. Cross-family transfer graph

For every intervention, the refinery records its effect across capability
families and classifies it as:
- GENERALIZER;
- CONDITIONAL_SPECIALIST;
- GLOBAL_VETO_CANDIDATE;
- SPARSE_POSITIVE;
- NULL_OR_UNRESOLVED.

Artifact:

`cross-family-transfer-graph.json`

Use:
- decide what behavior is worth baking into shared weights;
- isolate family-specific behavior behind routing/adapters;
- avoid globally training a control with known negative transfer;
- design multi-task mixture composition.

### 7. Quality/compute Pareto targets

For each fixture, Test 1.2 may have several correct outputs with different
compute cost.

The refinery forms the nondominated quality/calls/tokens/latency frontier and
chooses the cheapest observed response among the maximum-quality candidates.

Artifact:

`pareto-training-targets.jsonl`

Use:
- SFT for concise efficient behavior;
- preference tuning for equal-quality lower-compute output;
- distillation of test-time compute back into cheaper single-pass behavior.

## Second zero-clock quality pass

A deeper data-quality audit added five more deterministic products. These do
not create any new model/runtime calls and do not add a campaign phase.

### 8. Reliability-weighted distillation

A one-off harness rescue is not treated as an equally reliable teacher target
to a rescue reproduced across seeds or distinct interventions.

The refinery scores each distillation target using:
- independent support count;
- seed support;
- intervention/category support;
- observed score margin;
- contradictory positive-target count.

Artifact:

`reliability-weighted-distillation-corpus.jsonl`

Use:
- SFT/LoRA target weighting;
- filter provisional one-off targets;
- prioritize targets supported by multiple measured mechanisms.

Research basis:
- *When Are Teacher Tokens Reliable? Position-Weighted On-Policy
  Self-Distillation for Reasoning* (arXiv:2605.21606, 2026);
- recent preference/data-selection work showing that training examples have
  strongly model-dependent value.

### 9. Long-horizon balanced training mix

A curriculum focused only on today's weakest capability can over-concentrate
gradients and cause forgetting or reduce future adaptability.

The refinery blends:
- current weakness;
- uniform capability coverage;
- stability-anchor retention need;
- cross-family transfer evidence;
- an anti-concentration cap.

Artifact:

`long-horizon-training-mix.json`

Use:
- multi-capability fine-tuning mixture weights;
- anti-forgetting rehearsal allocation;
- prevent one weak family from monopolizing the update budget.

Research basis:
- *The Long-Term Effects of Data Selection in LLM Fine-Tuning*
  (arXiv:2605.30537, 2026);
- *MSSR: Memory-Aware Adaptive Replay for Continual LLM Fine-Tuning*
  (arXiv:2603.09892, 2026).

### 10. Preference-quality filtering

Not every DPO/ORPO pair is equally useful. The refinery ranks each existing
same-task preference pair by:
- outcome-sign consistency;
- replication count;
- seed count;
- quality margin;
- hard-case value.

Artifact:

`preference-quality-index.jsonl`

Use:
- select/weight DPO, ORPO, KTO, or reward-model pairs;
- retain harmful-control negatives only when their measured sign is stable;
- avoid spending training compute on noisy or contradictory pairs.

Research basis:
- *Towards Understanding Valuable Preference Data for Large Language Model
  Alignment* (arXiv:2510.13212, 2025), which finds preference-pair value is
  model dependent and that better selection can outperform larger raw sets.

### 11. Failure credit assignment

Failed trajectories are converted into supervised ownership/repair records
rather than being treated as undifferentiated negatives.

The deterministic labels distinguish:
- base-model behavior rescued by a measured control;
- control-induced negative transfer;
- unresolved residual model/control failure.

Artifact:

`failure-credit-assignment-corpus.jsonl`

Use:
- train failure detectors;
- train repair/router heads;
- separate "improve model weights" from "do not activate this controller";
- salvage useful evidence from failures.

Research basis:
- *Exploring Expert Failures Improves LLM Agent Tuning*
  (arXiv:2504.13145, 2025);
- *Where LLM Agents Fail and How They can Learn From Failures*
  (arXiv:2509.25370, 2025).

### 12. Calibration / verify supervision

Test 1.2 already measures raw correctness, repeats, instability, and whether
verification/retry controllers rescue a fixture. The refinery converts that
evidence into supervision labels:

- `TRUST_DIRECT`
- `VERIFY`
- `ESCALATE_TO_VALIDATED_CONTROL`
- `ESCALATE_OR_ABSTAIN`

Artifact:

`calibration-verify-supervision-corpus.jsonl`

Use:
- train a confidence/verification head;
- teach the model when its raw answer is reliable;
- reduce unnecessary verification on stable strengths;
- escalate unreliable or repeatedly failed regions.

Research basis:
- *Beyond Accuracy: The Role of Calibration in Self-Improving Large Language
  Models* (arXiv:2504.02902, 2025), which reports that self-improvement can
  increase overconfidence unless calibration is handled explicitly.

These five additions change **data quality and training policy**, not the test
clock. The hard ceiling remains 13h59m.

## Hard zero-clock contract

The summary artifact:

`zero-clock-model-manufacturing-map.json`

must contain:

- `zero_model_calls_added = true`
- `zero_active_test_seconds_added = true`

The tuning run refuses the collection if either value is false or if any of the
twelve product classes disappears.

These outputs therefore increase model-training value without changing the
Test 1.2 phase schedule or its 13h59m combined hard ceiling.
