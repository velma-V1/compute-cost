# REVIEW ROUND 1 — IMPLEMENTATION RESOLUTION

**Round-1 external review basis:** commit `81ecec37d56cfed7c6617c28596d6a3b7b31ac90`  
**Current corrected branch:** `build/gpt20b-test1.2-full-improvement`  
**Latest validated scientific-fix head:** `a8440bda4ceeeb2dfe9a72ee90dcae65b6e02351`  
**CI:** Python 3.11 PASS + Python 3.12 PASS

The first external reviewer found five material issues. They have been implemented as follows.

## A1 — Undefined deltas persisted as numeric zero — FIXED

Scientific contract is now:

```
delta_valid == false  =>  delta == null
```

This applies to:

- Test 1.2 Collection;
- Test 1.2 recovery/sanitization;
- Test 1.2 tuning policy rows;
- Test 2 observations.

Consumers have been hardened to refuse undefined deltas instead of coercing them back to zero.

Regression tests explicitly assert that invalid observations serialize `delta: null`.

## A2 — One-observation mutable family budget ratchet — FIXED

Budget calibration is now owned by **Stage 0 Runtime Characterization**, not Test 2.

Stage 0:

- uses 3 independent replicates by default;
- does not accept the first lucky completion;
- finds the first 3/3 capability-valid generation boundary;
- applies a safety factor (default 1.5);
- rounds upward to the tested budget ladder;
- writes the resolved family budgets into a hashed runtime-characterization profile;
- does not mutate campaign config while measuring the boundary.

Test 2:

- consumes resolved family budgets as immutable proof inputs;
- no longer learns or ratchets family budgets while proof is running;
- records the budget used for every baseline and treatment.

## A3 — Informative censoring — FIXED AS A FIRST-CLASS OUTCOME

Matched-budget truncation is no longer treated as a null result.

Rows may now carry:

```
censored_for_capability = true
censoring_class = CONTROL_EXCEEDS_BASELINE_BUDGET
```

Per-control summaries include:

- raw observations;
- valid comparable observations;
- censored observations;
- censoring rate.

If censoring exceeds the configured threshold, classification becomes:

```
CENSORING_DOMINATED
```

and ordinary null/pruning claims are prohibited.

Test 2 also contains a separate **own-budget capability+cost probe**. A censored control can be retried with bounded extra compute, but that result is explicitly classified as a capability-plus-cost finding and may never be counted as a matched-budget rescue.

## A4 — Harm invariant unfalsifiable — FIXED WITH DEDICATED HARM SEEKING

Test 2 now builds a baseline-pass sentinel pool and deliberately attacks it with candidate controls.

Default verification requirement:

- at least 16 valid baseline-pass sentinel observations per control;
- at least 4 distinct capability families;
- explicit break count and break rate;
- explicit censoring rate;
- maximum accepted break rate = 5%.

A recipe cannot receive `verified_for_shipping = true` without sufficient dedicated harm evidence.

The artifact:

```
harm-sentinel-evidence.json
```

contains the proof.

## A5 — Runtime-semantics conformance invariant missing — FIXED

Stage 0 is now a hard prerequisite before capability claims.

It produces:

```
runtime-characterization-profile.json
```

bound to:

- exact model;
- exact runtime version;
- model information / quant context exposed by runtime;
- runtime semantics;
- replicated family generation budgets;
- role-specialization evidence.

Capability observations are blocked unless the Stage-0 profile exists and passed.

Every Test 1.2 capability row carries:

```
runtime_characterization_profile_sha256
scoring_source_channel = content
```

The scoring path explicitly declares that the thinking channel is excluded.

## Stage ownership after Round 1

```
STAGE 0
runtime semantics
replicated safe operating budgets
role economics
        ↓
TEST 1.2
opportunity discovery
failure phenotypes
harder frontiers
unique valid rescues
        ↓
TEST 2
recurrence
generalization
harm seeking
censoring/cost tradeoffs
robustness
blind proof
        ↓
COMPILER / ACCEPTANCE
verified controls only
routing / vetoes / limits
release / constrain / reject
```

## STILL STOP-CLOSED — DO NOT BYPASS

### Exact Test 1.2 → Test 2 control execution

Test 2's legacy recipe language is not semantically equivalent to the full Test 1.2 intervention language.

Until a shared exact-control executor is implemented, Test 2 intentionally rejects a Test 1.2 handoff rather than translating it approximately.

The required architecture is:

```
single canonical intervention definition
           ↓
single shared renderer/executor
       ↙           ↘
Test 1.2          Test 2
discovery         proof
```

Required invariant:

```
proof_semantic_hash == discovery_semantic_hash
```

Do not remove the fail-closed guard merely to make a campaign run.

## ROUND-2 REVIEW PRIORITIES

The next reviewer must now focus on items not fully reviewed in Round 1:

1. shared exact-control executor and semantic hashing;
2. cluster-aware inference and multiple-comparison control;
3. training-asset firewall through `test12_model_manufacturing.py`;
4. tuning/recovery correctness through `test12_tuning.py`;
5. raw Ollama channel semantics through `runtimes/ollama.py`;
6. classification validity rules;
7. EvidenceStore / manifest / protected-partition isolation;
8. zero-call reanalysis products:
   - per-control censoring audit,
   - rescue-signature redundancy clustering,
   - unresolved capability-floor registry;
9. whether Stage-0 budget calibration should use smarter bracket/bisect search while preserving replicated validity;
10. whether auditor-versus-executor evidence is deep enough to make an architectural decision.

**Do not recommend another long run until these are reviewed.**

---

# TEST 1.2 + TEST 2 EXTERNAL SCIENTIFIC REVIEW HANDOFF

**Repository:** `velma-V1/compute-cost`  
**Branch:** `build/gpt20b-test1.2-full-improvement`  
**Purpose:** Independent adversarial review before another long model campaign  
**Status:** Review required before the next full Test 1.2 → Test 2 production run

---

# 1. REVIEWER MISSION

Do not treat this as a normal code review.

Treat the repository as a **scientific measurement instrument** that has already demonstrated that it can produce plausible-looking but false conclusions when runtime failures are allowed to enter capability statistics.

Your job is to determine:

1. whether Test 1.2 is actually measuring the model rather than Ollama/runtime/harness behavior;
2. whether Test 1.2 is spending its fixed clock on **new opportunities** rather than proving the same finding repeatedly;
3. whether Test 2 is a valid recurrence/robustness/proof stage;
4. whether the Test 1.2 → Test 2 handoff preserves the **exact semantics** of discovered controls;
5. whether any invalid observation can still leak into:
   - capability scores,
   - rescue counts,
   - regressions,
   - control ranking,
   - routing,
   - training assets,
   - negative assets,
   - fine-tuning candidates,
   - final acceptance;
6. whether the complete system is extracting the maximum useful model improvement from the available wall-clock budget;
7. what is still missing for the strongest possible INVERTED model harness.

Do not assume green CI means the experiment is scientifically correct.

Trace:

```
raw model generation
  ↓
runtime capture
  ↓
classification
  ↓
capability validity
  ↓
baseline
  ↓
treatment
  ↓
delta
  ↓
candidate promotion
  ↓
proof / recurrence
  ↓
negative-transfer analysis
  ↓
routing / harness compile
  ↓
training assets
  ↓
acceptance decision
```

Try to break every arrow.

---

# 2. PROJECT GOAL

The project is not trying to produce a benchmark score.

The goal is to build the highest-value external model system possible around a fixed local model.

The harness should make the model:

- more capable;
- more reliable;
- more efficient;
- more difficult to derail;
- better at hard cases;
- better with tools;
- better at state, context, and memory;
- better at recovery;
- better at verification;
- better at knowing when to stop or escalate;
- safer against negative transfer;
- more useful as both executor and auditor;
- capable of producing high-quality later training assets.

The governing objective is:

> **Maximize new decision-changing information and usable capability gain per physical model call and per wall-clock second.**

Project laws:

- Suggestions are a floor, not a ceiling.
- Data collection is cheap; repeating a long campaign is not.
- Never remove a valuable test simply to simplify the harness.
- A known opportunity does not deserve endless proof during discovery.
- Negative effects are useful evidence and must become routing/veto/boundary knowledge.
- Invalid runtime behavior is evidence about the runtime, not automatically evidence about model capability.
- A model improvement is only real if the comparison is scientifically valid.
- Test 1.2 should search broadly.
- Test 2 should prove deeply.
- Final acceptance should be harder than either discovery or proof.

---

# 3. AUTHORITATIVE STAGE OWNERSHIP

This is the architecture that should be reviewed and improved.

## Test 1.2 Collection

**Role: opportunity discovery.**

It should search for:

- new capability failures;
- new failure phenotypes;
- new hard boundaries;
- new valid rescues;
- new control categories;
- new negative boundaries;
- new runtime operating conditions;
- new tool behavior;
- new context/state/memory limits;
- new role-specialization effects;
- new combinations worth handing forward.

It should not spend most of the clock proving recurrence.

### Test 1.2 discovery law

Once a fixture receives one **valid full rescue**, that fixture/control path should lose discovery priority.

Clock should move toward:

1. unresolved failures;
2. unseen failure phenotypes;
3. underexplored families;
4. harder levels in strong families;
5. unused control categories;
6. unexplored interactions;
7. negative-transfer boundaries;
8. model/runtime frontier questions.

Proof debt follows the candidate into Test 2.

---

## Test 2

**Role: dedicated proof / recurrence / robustness / distillation stage.**

Test 2 owns:

- recurrence;
- repeated success;
- seed stability;
- cross-fixture generalization;
- cross-family transfer;
- negative transfer;
- failure recovery;
- robustness;
- interaction recurrence;
- minimal recipe / knockout analysis;
- blind confirmation;
- cost/latency envelope;
- proof that the discovered effect is not noise.

Test 2 must **not rediscover the world from scratch**.

It should consume Test 1.2 opportunities and spend the majority of its clock proving or rejecting them.

---

## Final compiler / acceptance

After proof, the system should compile:

- exact operating configuration;
- routing policy;
- capability allowlist;
- negative boundaries;
- stop/escalate rules;
- tool policies;
- context/state/memory rules;
- verified controls;
- do-not-use controls;
- latency/cost envelope;
- rollback state.

The final acceptance stage should use untouched evidence and must never tune on its own holdout.

---

# 4. MAJOR FAILURE DISCOVERED IN THE FIRST TEST 1.2 RUN

Completed Collection run:

`test1.2-20260913-092435-e9cfd1c8`

Original aggregate results appeared to show:

- 3,698 treatment observations;
- 537 positive observations;
- 537 improved failed cases;
- 537 full rescues;
- 34 unique rescued fixtures;
- 305 regressions;
- 53 strong controls;
- 2 promising controls.

That interpretation was wrong.

## Critical clue

These three values were identical:

```
positive observations = 537
improved failed cases = 537
full failure rescues = 537
```

The scorer was effectively binary:

```
0 → 0
0 → 1
1 → 0
1 → 1
```

Observed transitions:

```
1.0 -> 1.0    1713
0.0 -> 0.0    1143
0.0 -> 1.0     537
1.0 -> 0.0     305
```

Binary scoring itself is not inherently wrong.

The fatal problem was **invalid observations being collapsed to numeric zero**.

---

# 5. FOUNDING MEASUREMENT BUG

The previous Test 1.2 path effectively did:

```python
valid = classification.valid_for_capability is True

numeric = score if valid else 0.0

delta = numeric - control_score
```

This meant:

- THINK_TRUNCATED could become score 0;
- ANSWER_TRUNCATED could become score 0;
- NO_FINAL_ANSWER could become score 0;
- runtime failures could become score 0.

An invalid observation therefore became a fake capability failure.

## Consequences actually observed

Reported regressions:

```
305
```

Real valid regressions:

```
5
```

Invalid regressions:

```
300
```

Breakdown:

```
THINK_TRUNCATED     293
ANSWER_TRUNCATED      6
ANSWER_WRONG          4
RUNTIME_FAILURE       1
FORMAT_FAILURE        1
```

Therefore:

> **98.4% of the original reported regressions were not capability regressions.**

---

# 6. THE 537 RESCUES WERE ALSO CONTAMINATED

A baseline-validity audit showed that the apparent rescues overwhelmingly started from invalid baselines.

Invalid baseline classes behind reported rescues:

```
THINK_TRUNCATED     508
ANSWER_TRUNCATED     28
```

That is 536 of the 537 originally reported rescues.

Therefore the original headline:

```
537 rescues
53 strong controls
34 rescued fixtures
```

must **not** be used as scientific truth.

The correct interpretation is:

> Test 1.2 discovered a massive **generation-budget/runtime boundary**, not 536 independent capability rescues.

This evidence is still valuable, but it belongs in the runtime/budget channel.

---

# 7. FAMILY CONTAMINATION

The 17 value-incomplete capability families contained heavy invalid-output contamination.

Observed result classes across those families:

```
THINK_TRUNCATED     333
ANSWER_CORRECT      252
TOOL_FAILURE         52
ANSWER_TRUNCATED     26
ANSWER_WRONG         13
NO_FINAL_ANSWER       3
RUNTIME_FAILURE       2
```

Examples:

```
instruction_following_constraint_stacking  44 / 57 invalid
planning_optimization                      37 / 46 invalid
test_generation_verification               36 / 36 invalid
refactoring_under_constraints              32 / 46 invalid
sibling_transfer_generalization            32 / 37 invalid
self_correction                            30 / 43 invalid
multi_turn_state_tracking                  30 / 52 invalid
tool_error_recovery                        24 / 35 invalid
verification_critique                      23 / 38 invalid
```

This is especially important because:

- verification,
- critique,
- self-correction,
- uncertainty,
- tools,

are central to the INVERTED architecture.

---

# 8. REPLICATION PROBLEM

The original Collection produced:

```
2554 intervention × fixture cells
```

Replication distribution:

```
1 observation: 1932 cells
2 observations:  100 cells
3 observations:  522 cells
```

Therefore:

```
75.6% of cells were singletons
```

The original “strong” classifications were largely pooled intervention statistics, not repeated proof on the same intervention × failure relationship.

This is now considered correct **for discovery only**.

It is not sufficient for proof.

Test 2 must explicitly measure:

- recurrence;
- cross-seed stability;
- cross-fixture generalization;
- family transfer;
- harm rate;
- confidence intervals.

---

# 9. TEST 1.2 DISCOVERY REDESIGN ALREADY IMPLEMENTED

The Collection scheduler has been changed so that:

- one valid full rescue creates an opportunity candidate;
- a rescued fixture leaves the priority rescue queue;
- rare/new failure phenotypes outrank repeated copies of known failures;
- strong families are pushed toward harder unseen cases;
- different control categories are directed toward underexplored phenotypes;
- same fixture × same control seed replication is no longer a Collection priority;
- negative-transfer discovery samples novel sentinels instead of repeatedly proving one boundary;
- all 40 capability families remain part of the mandatory coverage floor;
- new clock is directed toward discovery rather than proof.

A new opportunity map tracks:

- unique failing fixtures;
- unique rescued fixtures;
- distinct failure phenotypes;
- unresolved failures;
- frontier state;
- repeated-pair waste.

Review this logic in:

`src/compute_cost/test12_campaign.py`

---

# 10. CAPABILITY-VALIDITY FIX ALREADY IMPLEMENTED

The corrected rule is:

> **A capability delta exists only if the baseline and treatment are both capability-valid.**

Invalid rows remain evidence.

They may be used for:

- runtime analysis;
- truncation analysis;
- budget calibration;
- resource behavior;
- capture diagnostics.

They may **not** become:

- capability failures;
- capability rescues;
- regressions;
- null controls;
- pruning candidates;
- training positives;
- training negatives;
- fine-tuning candidates;
- routing labels.

Review these files carefully:

- `src/compute_cost/test12_campaign.py`
- `src/compute_cost/test12_value.py`
- `src/compute_cost/test12_model_manufacturing.py`
- `src/compute_cost/test12_tuning.py`

---

# 11. GENERATION-BUDGET FIX ALREADY IMPLEMENTED

The 536 fake rescues are now treated as useful evidence about the model's required operating budget.

Current rules:

1. determine a capability-valid baseline budget per family;
2. invalid baseline output is not a capability failure;
3. non-budget controls must be compared at the same generation budget as baseline;
4. explicit GENERATION_BUDGET experiments are allowed to change budget;
5. old evidence measured at obsolete budgets must not suppress new trials;
6. recovered invalid/obsolete-budget evidence remains on disk but is quarantined from capability scoring.

Reviewer should try to break this invariant:

> A prompt/control/routing/tool intervention must never receive credit simply because it was allowed more generation tokens than its baseline.

---

# 12. TEST 2 STOP-SHIP BUGS THAT HAVE ALREADY BEEN FIXED

The old `test2_campaign.py` had the same invalid→0 failure.

That has been changed.

## Test 2 now:

- records capability validity;
- refuses to calculate a capability delta from an invalid baseline or invalid treatment;
- filters invalid rows out of effect maps;
- tracks baseline and treatment generation budgets;
- requires a matched budget for capability comparison;
- escalates baseline budget through a configurable ladder when the model truncates;
- stores truncation attempts as runtime evidence;
- uses the first valid operating point instead of calling truncation a failure.

The default ladder is currently:

```
256
512
1024
2048
```

Reviewer must determine whether this is the optimal search method or whether a faster bracket/bisect strategy is better.

---

# 13. TEST 1.2 → TEST 2 SEMANTIC COMPATIBILITY

This is now **fail-closed**.

Old Test 2 knows the old `INGREDIENTS` prompt-recipe system.

Test 1.2 can discover controls involving:

- reasoning effort;
- generation budget;
- context;
- prompt structure;
- verification;
- retry;
- routing;
- tools;
- state;
- memory;
- controller chains;
- interaction/composition;
- recovery.

It is scientifically invalid to translate those controls into a generic old prompt recipe.

Therefore current Test 2 intentionally refuses a Test 1.2 handoff until it has an **exact-control execution adapter**.

Current fail-closed rule:

> **No Test 1.2 control may enter Test 2 unless Test 2 can execute the exact intervention definition with the same semantics.**

This is a remaining implementation requirement, but the unsafe behavior is blocked.

## Highest-priority remaining implementation task

Build a shared exact intervention executor so both stages call the same implementation:

```
Test 1.2 discovery
      │
      ▼
shared intervention definition + executor
      │
      ├── Test 1.2 discovery observation
      │
      └── Test 2 proof observation
```

Do not duplicate intervention rendering logic.

Do not convert an intervention to another control language.

Identity should be content-addressed, for example:

```
intervention_semantic_sha256
```

Test 2 should reject any candidate if its proof-time semantic hash differs from discovery-time semantic hash.

---

# 14. STAGE-CONTRACT CONFLICT THAT HAS BEEN FIXED

The previous Test 1.2 terminal output claimed:

```
no_test2_followup_required = true
```

That contradicted the project law that proof belongs to Test 2.

The contract now states that Test 2 is the authoritative:

```
RECURRENCE
ROBUSTNESS
NEGATIVE_TRANSFER
DISTILLATION
PROOF
```

stage.

Reviewer should verify there is no remaining artifact that tells another model or operator to skip Test 2.

---

# 15. FOUNDATIONAL GPT-OSS / OLLAMA QUESTIONS

These questions must be treated as upstream model/runtime characterization, not ordinary capability tests.

## Runtime semantics

1. Does Ollama accept `think: false` for gpt-oss?
2. If not, does it error or silently substitute another mode?
3. Does `format_json: true` or native JSON format produce empty content with populated thinking?
4. At what rate does this happen per task family?
5. Is reasoning always separated into `message.thinking`?
6. Does reasoning ever leak into `content`?
7. Does `done_reason` distinguish thinking exhaustion from answer exhaustion?
8. Does `num_predict` cap thinking + answer together or answer only?
9. What exactly does `eval_count` contain?
10. Can thinking tokens and answer tokens be separated through this interface?
11. Does the harness preserve/drop prior analysis correctly through tool loops?
12. Does a tool call terminate generation cleanly?

## Sampling

13. Is `top_p` falling to Ollama default behavior when omitted?
14. What changes when `top_p=1.0`?
15. Temperature 0.0 versus 1.0 by capability family.
16. Which setting lies on the accuracy/cost Pareto frontier?

## Reasoning budget

17. Minimum reproduced passing generation budget per family.
18. Minimum passing budget per reasoning effort.
19. Thinking/answer proportion if observable.
20. Sharpness of THINK_TRUNCATED → PASS boundary.
21. Boundary versus difficulty.
22. 2× / 4× / 10× minimum-budget behavior.
23. Overthinking/accuracy decline.
24. Low effort versus high effort.
25. Accuracy/token Pareto front.
26. Analysis-loop rate.
27. Early doomed-generation prediction.

## Output contract

28. First-attempt parse rate.
29. Developer JSON instructions versus runtime-enforced JSON.
30. Schema/grammar effect.
31. Cheapest reliable output contract.
32. Retry-on-parse-failure convergence.
33. Wrong answer versus unparseable answer separation.

## Context

34. Real usable context.
35. Context limit versus reasoning effort.
36. Observable truncation behavior.
37. planted-key position sensitivity.
38. VRAM/latency knee.

## Role specialization

39. Auditor accuracy versus executor accuracy.
40. Auditor versus executor required budget.
41. False-accept versus false-reject asymmetry.
42. Candidate-quality sweep.
43. Candidate alone versus candidate + reasoning.
44. second-audit stability.
45. low-effort versus high-effort audit economics.

## Reliability / trust

Also measure:

- seed spread;
- warm versus cold;
- sustained-load degradation;
- wall-clock and energy cost;
- prompt injection into auditor candidates;
- rationale/verdict divergence;
- reasoning-channel leakage into scored output.

The most important ordering is:

```
runtime semantics
    ↓
minimum valid budget
    ↓
role specialization
    ↓
capability opportunity search
```

Do not invert that order.

---

# 16. WHAT TEST 1.2 SHOULD OPTIMIZE FOR

A future Collection should report at minimum:

```
unique valid baselines
unique valid failures
unique failure phenotypes
unique valid rescues
unique rescued phenotypes
unique control categories producing rescue
hardest valid pass by family
easiest valid fail by family
new negative boundaries
runtime-invalid observations
invalid rate by family
calls per new discovery
seconds per new discovery
repeated-cell waste
unresolved capability floor
```

The scorecard should punish repeated proof during discovery.

Suggested discovery utility:

```
utility =
    new_failure_phenotype_value
  + new_unique_rescue_value
  + harder_frontier_value
  + new_control_category_value
  + new_negative_boundary_value
  + information_gain
  - repetition_cost
  - invalid_call_cost
  - latency_cost
```

Do not let the raw number of observations become the success metric.

---

# 17. WHAT TEST 2 SHOULD OPTIMIZE FOR

Test 2 should receive an opportunity registry like:

```
candidate_id
semantic_hash
discovery_family
discovery_fixture
failure_phenotype
control_category
discovery_budget
baseline_budget
discovery_seed
valid_rescue_count
verification_debt
known_negative_boundaries
estimated_call_cost
```

Then prove:

### Recurrence

- same candidate;
- independent seeds;
- same failure phenotype;
- no semantic drift.

### Generalization

- independent fixtures;
- sibling difficulty;
- nearby difficulty;
- other families where applicable.

### Harm

For every verified control:

```
rescue_rate
break_rate
rescue_to_break_ratio
family-specific harm
global harm
severity
```

A control with:

```
3 rescues : 1 break
```

is not equivalent to:

```
30 rescues : 1 break
```

### Confidence

Use clustered inference where repeated observations from one fixture are not treated as independent trials.

The unit of independence must be explicit.

Reviewer should decide whether the correct cluster is:

- fixture;
- fixture × seed;
- failure phenotype;
- family;
- campaign block.

---

# 18. BINARY SCORER LIMITATION

Current core capability scoring is often binary.

This is acceptable for hard acceptance:

```
PASS / FAIL
```

but insufficient for every analytical task.

Reviewer should determine where richer scoring is needed.

Potential secondary signals:

- exact correctness;
- partial constraint satisfaction;
- tool-call correctness;
- schema validity;
- argument correctness;
- factual support;
- reasoning completeness where observable;
- number of violated constraints;
- edit distance from valid structure;
- plan quality;
- recovery quality;
- calibration error.

Do not replace a reliable binary acceptance gate with a vague model judge.

Instead use:

```
hard pass/fail
+
diagnostic subscore vector
```

when deterministic scoring is possible.

---

# 19. FAILURE PHENOTYPE DESIGN

A fixture ID is not a failure type.

Test 1.2 should cluster failures by mechanism.

Candidate phenotype dimensions:

```
capability family
result class
scorer type
failure subtype
difficulty band
runtime validity
tool phase
context position
state age
constraint count
dependency depth
answer-contract type
```

Reviewer should search for cases where many “different” failures are actually the same phenotype.

Likewise, cluster controls by their rescued-fixture signature.

If 20 controls rescue the same 8 fixtures, they may represent one underlying mechanism with 20 names.

Find the real mechanism count.

---

# 20. UNRESOLVED FAILURE FLOOR

The most valuable output may not be the controls that work.

It may be the valid failures that survive every tested control.

For each unresolved failure, determine:

- valid baseline?
- enough generation budget?
- repeated?
- same phenotype elsewhere?
- tool/environment limitation?
- context limitation?
- genuine model capability limit?
- recoverable by another model?
- fine-tuning candidate?
- should route to stronger model?
- should abstain?

The unresolved set should become a **model-limit registry**, not a pile of zeros.

---

# 21. AUDITOR / EXECUTOR THESIS

INVERTED depends heavily on role specialization.

The reviewer must determine whether there is a real economic advantage to using the model as an auditor.

Matched design:

```
same task
same candidate
same operating budget accounting
executor condition
auditor condition
```

Measure:

- executor accuracy;
- auditor accuracy;
- false accept;
- false reject;
- required budget;
- latency;
- seed stability;
- difficulty sensitivity.

If low-effort audit performs like high-effort execution, that materially changes the architecture.

If audit is unreliable or easily steered by candidate text, that also changes the architecture.

---

# 22. TOOL TESTING REQUIREMENTS

Do not confuse:

```
model knows which tool should be used
```

with:

```
model successfully completed a real tool loop
```

Test separately:

1. tool selection;
2. argument schema;
3. argument values;
4. tool invocation;
5. stop behavior;
6. tool result ingestion;
7. state update;
8. recovery from tool error;
9. multi-tool dependency order;
10. final answer after tool result;
11. malicious/untrusted tool output;
12. stale tool state.

The raw Ollama tool-call interface must be tested independently of synthetic tool simulations.

---

# 23. CONTEXT / MEMORY / STATE REQUIREMENTS

Measure independently:

- retrieval;
- reasoning over retrieved material;
- lost-in-middle;
- contradiction;
- stale state;
- superseded state;
- context compression;
- memory fidelity;
- active memory evolution;
- tool-state interaction;
- long-horizon state transitions.

A large context window is not evidence that the model can use that context.

---

# 24. NEGATIVE EFFECTS MUST BECOME PRODUCT VALUE

Negative observations are not wasted tests.

A valid negative effect should produce one or more of:

- route veto;
- activation boundary;
- control exclusion;
- family-specific exception;
- safer replacement;
- retry prohibition;
- cost avoidance rule;
- training negative;
- regression sentinel;
- rollback trigger.

However:

> Invalid runtime observations must never become negative capability assets.

---

# 25. TRAINING-ASSET FIREWALL

Review every path into:

- distillation;
- preference data;
- fine-tuning examples;
- router supervision;
- calibration data;
- failure-credit data;
- stability anchors.

Required invariant:

```
training asset
    ⇒ source observation scientifically valid
    ⇒ baseline scientifically valid where comparative
    ⇒ treatment scientifically valid
    ⇒ partition eligible
    ⇒ no protected leakage
```

Any training corpus that violates this must fail closed.

---

# 26. PROTECTED-DATA CONTRACT

The reviewer must verify:

- DISCOVERY may influence candidate generation.
- VALIDATION may influence selection only where explicitly allowed.
- TEST2_BLIND may not tune Test 2 candidates after blind opening.
- TEST3_PROTECTED must not be exposed before its owner stage.
- protected siblings must not leak into training.
- report generation must not accidentally reload protected content into optimization.

Partition names alone are not proof of isolation.

Trace file access.

---

# 27. RECOVERY CONTRACT

Long tests must never require a full rerun after a process interruption.

Required behavior:

- same run ID;
- valid atomic rows survive;
- elapsed active time survives;
- physical model-call count survives;
- completed valid trials survive;
- malformed atomic record quarantined individually;
- invalid scientific evidence preserved as diagnostic evidence;
- only missing/invalid scientific slice is eligible for replay;
- winner lock cannot reopen after holdout acceptance begins.

A model/runtime/benchmark-contract change creates a **new onboarding event**.

---

# 28. CURRENT TEST 2 SAFETY STATUS

The exact-control handoff is now implemented.

For a Test 1.2 source, Test 2:

1. loads the frozen provisional Test 1.2 policy;
2. imports the exact Test 1.2 intervention definitions;
3. recomputes and verifies semantic hashes;
4. executes those controls through the same `Test12Campaign.treatment()` path;
5. preserves the Stage 0 family budget map as immutable proof input;
6. blind-tests the frozen policy through the same policy executor used during compilation;
7. refuses semantic translation into the legacy ingredient recipe language.

Required invariant:

```
discovery_semantic_hash == proof_semantic_hash
```

The current fail-closed behavior should remain: any hash mismatch, runtime-profile incompatibility, consumed holdout, or missing exact source evidence must block proof rather than approximate it.

---

# 29. HIGHEST-VALUE ITEMS STILL MISSING

The following items remain genuine improvement opportunities after Round 2. Items already implemented are deliberately omitted.

## 1. Budget search efficiency

Stage 0 now requires 3/3 reproducible validity plus safety headroom, but the ladder is still discrete.

Improve it with bracket/bisect search while preserving:
- independent replicates;
- family-level safety factor;
- fail-closed behavior when the tested maximum has no headroom.

## 2. Richer role-specialization sweep

Stage 0 now measures executor versus low/high-effort auditor at the calibrated family budget and excludes invalid role observations.

Still add:
- controlled candidate-quality sweep;
- candidate-alone versus candidate-plus-reasoning;
- second-audit stability;
- adversarial candidate-text steering;
- false-accept/false-reject curves by candidate quality.

## 3. Control redundancy clustering

Cluster discovered controls by rescued-fixture / failure-phenotype signature so multiple labels for one underlying mechanism do not consume Test 2 proof budget.

## 4. Proof-value scheduler

Allocate Test 2 recurrence calls by expected decision value:
- near promotion/rejection boundary;
- high-value but uncertain controls;
- under-sampled families;
- high censoring;
- unresolved harm confidence;
- unresolved generalization.

Do not spend equal proof clock on already-settled candidates.

## 5. Runtime-invalidity predictive model

Track truncation/runtime failure by:
- family;
- difficulty;
- reasoning effort;
- generation budget;
- output contract;
- context;
- tool phase.

Use it to predict doomed calls early and reallocate clock.

## 6. Output-contract optimizer

Determine the cheapest reliable contract per family:
- free text + extraction;
- explicit JSON instruction;
- native JSON;
- schema/grammar.

## 7. Context knee and context/reasoning competition

Measure real usable context, lost-in-middle, position sensitivity, and latency/VRAM knee under each reasoning effort.

## 8. Sustained-load capability validity

Separate warm-load latency degradation from actual capability degradation over a multi-hour run.

## 9. Energy / hardware economics

Add Wh and thermal state to quality-per-cost decisions for local deployment.

## 10. Manifest finalization efficiency

Current manifest finalization recursively hashes retained files and can create a long invisible post-processing tail. Replace full-file re-reading with streaming/incremental content hashes without weakening integrity.

### Implemented since the first review

The following are **no longer missing**:

- shared exact Test 1.2 control execution inside Test 2;
- semantic intervention hashing;
- fixture-clustered Test 2 effect statistics;
- Benjamini-Hochberg FDR on positive Test 2 effects;
- dedicated baseline-pass harm seeking with confidence-bound gating;
- censoring as a first-class outcome;
- capability-plus-cost probing for censored controls;
- immutable Stage 0 family budgets during Test 2;
- replicated Stage 0 budget calibration with safety headroom;
- validation-only Test 1.2 compiler;
- consumable holdout lifecycle and retirement;
- blind/protected holdout recovery quarantine;
- invalid-delta JSON null contract;
- content-only scoring contract;
- training-asset validity filtering;
- TEST2_BLIND exclusion from fine-tuning qualification/data;
- valid unresolved-failure provenance across Test 1.2 → Test 2;
- exact frozen-policy blind proof;
- executable harness-to-weights stopping rules.

---

# 30. WHAT THE REVIEWER MUST DELIVER

Do not return a generic prose review.

Deliver:

## A. Stop-ship findings

Every issue capable of invalidating a long run.

For each:

```
severity
file
function
exact failure mode
scientific consequence
minimal reproduction
correct fix
regression test
```

## B. Architecture decision

State the final role of:

- Test 1.2 Collection;
- Test 2;
- final compiler;
- Test 3 / protected acceptance if retained.

No overlapping ownership.

## C. Missing high-value experiments

Rank by:

```
expected information value
expected capability value
model-call cost
wall-clock cost
implementation complexity
risk of confounding
```

## D. Exact-control handoff design

Provide the schema and implementation plan.

## E. Statistical validity plan

Define:

- unit of independence;
- recurrence threshold;
- confidence method;
- clustering;
- multiple-comparison handling;
- effect representation for binary outcomes;
- harm threshold.

## F. Scheduler design

Provide discovery scheduler and proof scheduler separately.

## G. Training firewall audit

Prove invalid/protected observations cannot enter training assets.

## H. Test suite

Add regression tests for every discovered stop-ship failure.

## I. Final answer

Answer:

> **What exact Test 1.2 + Test 2 architecture will produce the most trustworthy and highest-value model improvement under the existing fixed wall-clock constraints?**

Do not optimize for minimal code.

Optimize for highest scientific value and shipping quality.

---

# 31. FILES TO INSPECT FIRST

Start here:

```
src/compute_cost/test12_campaign.py
src/compute_cost/test12_foundation_labs.py
src/compute_cost/test12_value.py
src/compute_cost/test12_model_manufacturing.py
src/compute_cost/test12_tuning.py
src/compute_cost/test2_campaign.py
tests/test_test12_campaign.py
tests/test_test12_model_manufacturing.py
tests/test_test2_campaign.py
docs/test1.2-model-harness-compiler.md
```

Then trace:

```
characterization.py
runtimes/ollama.py
runner_core.py
scoring / classification
EvidenceStore / manifest
CLI recovery paths
```

Do not begin with reports.

Begin with raw-generation capture and work upward.

---

# 32. CURRENT EMPIRICAL FACTS THAT MUST NOT BE LOST

These facts came from the first real Test 1.2 Collection and should remain available as forensic evidence:

```
Treatment observations: 3698
Reported positive observations: 537
Reported full rescues: 537
Original unique rescued fixtures: 34
Reported regressions: 305

Valid regressions after audit: 5
Invalid regressions: 300

Invalid-baseline reported rescues:
  THINK_TRUNCATED: 508
  ANSWER_TRUNCATED: 28

Intervention × fixture cells:
  1 sample: 1932
  2 samples: 100
  3 samples: 522
```

Do not delete these results.

They are evidence of both:

- the model/runtime budget boundary;
- the failure mode of the old measurement system.

---

# 33. CURRENT RUN IDS

Collection:

```
test1.2-20260913-092435-e9cfd1c8
```

Test 1.2 tuning/compile run:

```
test1.2-tune-20260913-182713-e0af9b02
```

The tuning run was already partially executed when the validity issue was discovered.

Its evidence must be treated according to the recovery/quarantine rules, not discarded or blindly trusted.

---

# 34. NON-NEGOTIABLE SCIENTIFIC INVARIANTS

The reviewer should turn these into executable assertions wherever possible.

### Validity

```
capability_delta_defined
    ⇒ baseline_valid
    AND treatment_valid
```

### Budget

```
non_budget_control_delta_defined
    ⇒ baseline_budget == treatment_budget
```

### Semantic identity

```
test2_proof_candidate
    ⇒ proof_semantic_hash == discovery_semantic_hash
```

### Discovery

```
valid_full_rescue_found
    ⇒ fixture/control proof priority drops in Test 1.2
```

### Proof

```
promotion_to_verified
    ⇒ independent recurrence evidence exists
```

### Harm

```
verified_control
    ⇒ measured negative-transfer boundary exists
```

### Training

```
training_example
    ⇒ scientifically valid source evidence
```

### Protection

```
protected_fixture
    ⇒ never influences candidate generation or tuning
```

### Recovery

```
process interruption
    ⇒ valid completed evidence survives
```

---

# 35. FINAL REVIEW STANDARD

Do not approve the system because it “looks much better.”

Approve it only when you can answer yes to all of these:

- Can a truncation still become a capability failure?
- Can an invalid baseline create a rescue?
- Can a budget increase masquerade as a prompt improvement?
- Can Test 2 execute a different control than Test 1.2 discovered?
- Can repeated observations on one fixture inflate confidence?
- Can one control hide a large break rate behind a rescue rate?
- Can protected evidence influence tuning?
- Can invalid evidence enter training?
- Can a strong family consume clock on easy repeated cases?
- Can a weak family consume clock repeating one failure phenotype?
- Can Test 1.2 spend proof calls that belong in Test 2?
- Can Test 2 waste proof calls on already-settled questions?
- Can the harness claim full integration without unresolved families being explicit?
- Can an interruption force a full rerun?
- Can final acceptance reopen tuning?

If any answer is **yes**, identify and patch it before approving another long campaign.

---

# 36. REVIEWER DIRECTIVE

You are not being asked to preserve the current architecture.

You are being asked to preserve the **goal**.

If the best design requires:

- moving phases;
- replacing a scheduler;
- changing statistics;
- creating a shared executor;
- changing handoff schemas;
- adding instrumentation;
- removing redundant proof from discovery;
- increasing difficulty;
- redistributing the same fixed clock;

do it.

Do not add wall-clock time unless there is no way to recover equivalent or greater value by removing redundancy.

The final system should feel sophisticated internally but simple operationally:

```
run discovery
→ run proof
→ compile exact model policy
→ accept / constrain / reject
```

Everything else exists to make those four steps scientifically trustworthy and maximally valuable.


---

# 23. FINAL CONTINUOUS-IMPROVEMENT GOVERNANCE

This section supersedes any earlier wording that implies Test 1.2 may consume
blind/protected holdouts or directly authorize deployment.

## 23.1 Holdouts are consumable resources

A holdout partition is not permanently "clean" merely because it is excluded
from candidate generation. Every acceptance result leaks information back into
the development process.

Therefore:

1. each acceptance partition has a fixed cross-run use budget;
2. the default use budget is **one acceptance cycle**;
3. exposure consumes the partition even if the run later crashes;
4. same-run atomic resume may continue the already-consumed partition;
5. a different run/cycle may not reuse that partition;
6. fixture IDs from retired acceptance partitions may not reappear in later
   acceptance partitions;
7. the next cycle must generate new fixtures from **new field failure
   phenotypes**, not merely variants of already-fixed failures;
8. cycle N acceptance fixtures must not have existed during cycle N-1
   discovery/tuning.

Required machine-readable evidence:

- `holdout-consumption-ledger.json`
- `holdout-replenishment-plan.json`
- partition fingerprint;
- exact fixture IDs;
- cycle/run ID;
- use count;
- retirement state;
- next-cycle zero-overlap requirement.

## 23.2 Partition ownership

The pipeline is:

```
Stage 0 runtime characterization
    ↓
Test 1.2 Collection / discovery
    ↓
Test 1.2 validation-only compiler
    ↓
Test 2 exact-control / exact-policy proof on TEST2_BLIND
    ↓
fresh protected final acceptance on TEST3_PROTECTED
    ↓
release
```

Test 1.2 tuning may expose **VALIDATION only**.

It must not expose:

- `TEST2_BLIND`;
- `TEST3_PROTECTED`.

Its output is a **frozen provisional policy**, not a shipping decision.

Required status:

```
PROVISIONAL_READY_FOR_TEST2
PROVISIONAL_CONSTRAINED_FOR_TEST2
REJECT_BEFORE_TEST2
```

It must explicitly report:

```
release_authorized = false
onboarding_complete = false
test2_blind_exposed = false
test3_protected_exposed = false
test2_proof_required = true
```

## 23.3 Exact-control semantic identity

Test 2 may not translate Test 1.2 controls into legacy prompt ingredients.

For every Test 1.2 control promoted to Test 2:

```
discovery_semantic_hash == proof_semantic_hash
```

must hold.

Test 2 must execute the original intervention through the same
`Test12Campaign.treatment()` implementation used by Test 1.2.

Proof may vary:

- independent seed;
- independent fixture;
- family;
- difficulty;
- proof phase.

Proof may not mutate the discovered control.

The frozen compiled policy must also be blind-tested through the same
`TuningRun.run_policy()` implementation used during compilation. Router and
risk-gate semantics are therefore proof objects, not reconstructed prose.

## 23.4 Runtime-conformance invariant

Stage 0 is a hard prerequisite.

Capability evidence is valid only under the characterized runtime identity and
resolved family budgets.

At minimum:

```
model identity
runtime version
runtime semantics profile hash
resolved generation budget by family
```

must remain compatible through Test 2.

A material runtime change is a **new onboarding event**, not a continuation.

## 23.5 Censoring is first-class

Token-budget censoring must never be counted as model incapability.

When the same case/control flips from invalid/censored to valid solely because
generation budget increased, classify the primary cause as:

```
INSUFFICIENT_TOKEN_BUDGET
```

The higher-budget result is capability-plus-cost evidence, not a matched-budget
rescue.

## 23.6 Harm uses confidence bounds

Shipping safety is not judged by point break rate alone.

For every reachable control in the frozen policy:

```
harm CI upper bound <= preregistered break-rate ceiling
```

is required.

The current implementation uses a Wilson 90% interval.

Controls reachable from the frozen router/risk gate create mandatory harm
evidence debt. They are tested before exploratory controls.

## 23.7 Concrete stopping rules

The harness-improvement era stops or blocks shipping when any applicable rule
fires:

1. **Marginal net value <= 0** on a fresh holdout partition.
2. **Policy cost > k × direct execution** while accuracy advantage is below the
   preregistered exchange rate.
3. **Harm CI upper bound > break-rate ceiling**, regardless of rescue rate.
4. **Capability floor unchanged across two cycles.**

Rule 4 is the strategy-transition rule:

```
HARNESS_CEILING_REACHED
    ↓
MOVE_TO_WEIGHTS
```

Required machine-readable artifact:

- `harness-stopping-rules.json`

## 23.8 Training yield is measured after scientific qualification

Raw rescue count is not a training metric.

The primary training-yield metrics are:

```
qualified_fine_tuning_phenotypes
valid_train_eligible_examples
training_pipeline_has_input
```

A cycle producing a small number of uncontaminated training examples is better
than a large contaminated corpus.

Required artifact:

- `training-asset-yield.json`

If the valid training set is empty, the correct status is:

```
NO_VALID_WEIGHT_TRAINING_INPUT_YET
```

not a fabricated fine-tuning recommendation.

## 23.9 Operational law

Do not rerun valid expensive evidence merely because architecture code changes.

Use:

```
NO_FULL_RERUN_ATOMIC_RESUME
```

Preserve:

- valid atomic observations;
- completed phases;
- elapsed active-time budget;
- physical call budget;
- frozen winner hash.

Only replay the missing or damaged atomic slice.



---

# 37. ROUND 2 IMPLEMENTATION AUDIT

This section records the second adversarial pass after the first external review.

## 37.1 Fixed stop-ship findings

### Undefined delta sentinel

Old unsafe form:

```python
delta = 0.0  # when comparison was undefined
```

Current contract:

```
delta_valid = false
delta = null
```

Applied to:
- Test 1.2 Collection;
- Test 1.2 compiler/tuning recovery;
- Test 2.

Consumers must explicitly require `delta_valid == true`.

### Stage 0 is now an executable prerequisite

Capability testing is blocked unless the exact model/runtime profile passes:

- runtime identity captured;
- critical runtime-semantics probes actually observed;
- no thinking-channel markup leakage into scored content;
- every capability family has a reproducibly valid budget;
- safe budget headroom exists above the measured boundary;
- valid matched executor/auditor evidence exists.

Every Test 1.2 capability observation carries the runtime-profile SHA256.

### Role-economics confound removed

Executor and auditor are compared using the Stage 0 calibrated family budget.

Invalid/truncated executor or auditor generations are excluded from role accuracy rather than scored as wrong.

### Test 2 budget ratchet removed

Test 2 no longer mutates a family budget based on the first fixture.

Family budgets are immutable proof inputs from Stage 0.

### Censoring is explicit

A control that cannot finish under the matched baseline budget becomes:

```
CONTROL_EXCEEDS_BASELINE_BUDGET
```

and contributes to per-control censoring rate.

High-censoring controls cannot be classified as ordinary null/no-rescue controls.

Separate capability-plus-cost probes may test them at their own budget.

### Harm evidence is mandatory

A reachable control cannot be shipping-verified without dedicated baseline-pass sentinel evidence.

Harm uses a confidence-bound gate rather than point break rate alone.

### Fixture is the unit of independence

Test 2 effect statistics first collapse repeated observations to fixture-level effects.

Reported fields include:

```
raw_valid_n
independent_fixture_n
unit_of_independence = fixture
```

Repeated seeds on one fixture no longer inflate independent N.

### Multiplicity control

Positive Test 2 effects carry exact sign-test p-values and Benjamini-Hochberg FDR q-values.

A nominal STRONG/PROMISING positive that fails the configured FDR level is downgraded to:

```
UNCERTAIN_MULTIPLICITY
```

Safety/harm evidence remains conservative and is not rescued by multiplicity.

### Training firewall hardened

Fine-tuning qualification now fails closed when runtime-invalid and capability-valid evidence are mixed under one residual owner.

Only capability-valid independent failures count toward the fine-tuning threshold.

`TEST2_BLIND` and `TEST3_PROTECTED` cannot influence fine-tuning qualification or enter the training dataset.

### Unresolved failure provenance preserved

Collection unresolved failures are derived only from capability-valid baseline evidence.

The Test 1.2 → Test 2 handoff now preserves:

- fixture ID;
- family ID;
- difficulty;
- original measured result class;
- capability-validity;
- source experiment ID;
- source Collection run;
- original partition.

Synthetic `unknown|UNRESOLVED` phenotype collapse is prohibited.

## 37.2 Recovery / integrity audit

Verified behavior:

- malformed atomic JSONL row is quarantined individually;
- record hash mismatch is quarantined;
- elapsed active time survives recovery;
- physical call count survives recovery;
- holdout exposure survives recovery;
- legacy blind/protected rows are quarantined from Test 1.2 optimization;
- consumed holdout cannot become clean again after a crash;
- Collection/Tuning handoffs verify source manifests;
- exact Test 2 handoff verifies full source manifests and semantic identity.

## 37.3 Training-manufacturing audit

`test12_model_manufacturing.py` applies capability-validity filtering before building zero-clock training products.

Comparative training rows require valid comparison evidence.

Discovery is the eligible source partition for Collection-produced training assets.

Test 2 fine-tuning data is restricted to DISCOVERY/VALIDATION and explicit capability-valid residual failures.

Blind/protected evidence is acceptance evidence, never training input.

## 37.4 Current statistical contract

For Test 2:

```
unit of independence = fixture
within-fixture repeated seeds = repeated measurement
positive promotion = effect evidence + FDR
harm = dedicated sentinel evidence + confidence upper bound
censoring = separate outcome
undefined comparison = null delta
```

Do not revert to pooled raw observation counts.

## 37.5 Remaining reviewer work

The next reviewer should focus on **new unknowns**, not re-prove the fixed bugs.

Highest-value targets:

1. budget bracket/bisection efficiency;
2. candidate-quality auditor sweep;
3. control-mechanism redundancy clustering;
4. proof-value scheduling;
5. early truncation prediction;
6. output-contract optimization;
7. context knee;
8. sustained-load capability drift;
9. energy economics;
10. manifest-finalization performance.

Any new stop-ship finding should still be patched and covered by a regression test before another long campaign.
