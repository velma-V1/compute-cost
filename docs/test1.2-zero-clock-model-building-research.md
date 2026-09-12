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

## Seven zero-clock products

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

## Hard zero-clock contract

The summary artifact:

`zero-clock-model-manufacturing-map.json`

must contain:

- `zero_model_calls_added = true`
- `zero_active_test_seconds_added = true`

The tuning run refuses the collection if either value is false or if any of the
seven product classes disappears.

These outputs therefore increase model-training value without changing the
Test 1.2 phase schedule or its 13h59m combined hard ceiling.
