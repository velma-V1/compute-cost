# Test 1.2 Second Frontier-Gap Research Audit

Date: 2026-09-12

This audit was performed after the first seven frontier-gap labs were added.
The inclusion criterion was strict: each new item had to measure a distinct
agent/harness failure surface not already explained by prompt wording, ordinary
retry behavior, generic tool failure, memory compression, abstention, or
parallel scheduling.

The second audit adds seven mandatory micro-labs.

## 1. Authority separation under indirect prompt injection

Research signal:
- *NetInjectBench: Benchmarking Indirect Prompt Injection in Tool-Using Large
  Language Model Agents for Network Operations* (arXiv:2607.10490, 2026).
- *AgentSecBench: Measuring Prompt Injection, Privacy Leakage, and Tool-Use
  Integrity in LLM Agents* (arXiv:2605.26269, 2026).

Why it matters:
Tool/retrieval text can contain useful evidence and malicious instructions in
the same channel. A model must not allow untrusted text to create authorization.
At the same time, a useful harness cannot simply block every high-impact action.

Test 1.2 addition:
- strong injected override;
- fake approval in untrusted artifact text;
- legitimate high-impact change with trusted authorization metadata;
- unsafe-authority-accept and approved-change-overblock metrics.

Artifact: `authority-separation-map.json`.

Compiler use:
- enforce metadata-backed authorization when model susceptibility is observed;
- preserve approved actions to avoid useless static blocking;
- treat untrusted text as evidence, never authority.

## 2. Reward-hacking and evaluator-integrity resistance

Research signal:
- *Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use*
  (arXiv:2605.02964, 2026).
- *BAITBENCH: Measuring Agent Reward Hacking with Optional Shortcuts Planted in
  ML Tasks* (arXiv:2608.30724, 2026).

Why it matters:
An agent can improve an observed metric without improving the real task by
skipping verification, editing evaluation state, or exploiting a shortcut. This
can make a harness appear better while silently degrading true capability.

Test 1.2 addition:
- public-score tampering temptation;
- skip-verification proxy optimization;
- legitimate non-evaluator optimization control;
- exploit and overblocking metrics.

Artifact: `reward-hacking-resistance-map.json`.

Compiler use:
- protect evaluator and verification paths;
- prohibit metric-changing shortcuts that do not improve hidden task state;
- retain legitimate efficiency optimization.

## 3. Value-of-information clarification

Research signal:
- *Value of Information: A Framework for Human-Agent Communication*
  (arXiv:2601.06407, 2026).
- *Uncertainty-Aware Clarification in LLM Agents with Information Gain*
  (arXiv:2606.03135, 2026).
- *Structured Uncertainty guided Clarification for LLM Agents*
  (arXiv:2511.08798, 2025).

Why it matters:
Abstention is not enough. A capable agent should sometimes ask exactly one
high-value question, then act. Asking too often wastes interaction; asking too
little causes avoidable mistakes.

Test 1.2 addition:
- ambiguous destination;
- ambiguous account;
- complete low-risk request;
- complete authorized side-effect request;
- under-clarification and over-clarification metrics.

Artifact: `clarification-value-map.json`.

Compiler use:
- enable a value-of-information ask gate only where the model miscalibrates;
- target the missing field rather than emit generic clarification;
- avoid user interruption when the task is already complete.

## 4. Governance-safe compaction and resume

Research signal:
- *The Compaction Cliff in Long-Running AI Agent Memory*
  (arXiv:2608.22752, 2026).
- *Governance Decay: How Context Compaction Silently Erases Safety Constraints
  in Long-Horizon LLM Agents* (arXiv:2606.22528, 2026).
- *Parallel Context Compaction for Long-Horizon LLM Agent Serving*
  (arXiv:2605.23296, 2026).

Why it matters:
Ordinary memory tests do not prove that a compacted checkpoint retains exact
rules, approvals, current state, and obligations after the original history is
gone. Losing one hard rule can change later tool behavior.

Test 1.2 addition:
- compaction of noisy evolving history;
- exact preservation requirements for governance/state;
- resume using only the compacted checkpoint;
- both deny and authorized-allow cases;
- governance-decay and preservation metrics.

Artifact: `governance-compaction-map.json`.

Compiler use:
- pin governance constraints outside lossy compaction when needed;
- reject a checkpoint that drops mandatory state;
- distinguish policy preservation from ordinary semantic summary quality.

## 5. Belief-state reasoning under partial observability

Research signal:
- *Belief-State Engine: Augmenting LLMs for Principled Planning Under Partial
  Observability* (arXiv:2609.10036, 2026).
- *NeSyFS: A Neuro-symbolic Fast-Slow Thinking Framework for LLM Agent under
  Partial Observability* (arXiv:2607.28942, 2026).

Why it matters:
History tracking is not the same as maintaining uncertainty over hidden state.
Agents can prematurely commit after weak evidence or let belief drift over long
histories.

Test 1.2 addition:
- unresolved 50/50 hidden state -> SENSE;
- perfect discriminating observation -> ACT;
- weak contradictory observation against a strong prior -> SENSE;
- premature-commitment metric.

Artifact: `belief-state-map.json`.

Compiler use:
- require explicit belief-state tracking where premature commitment appears;
- separate sensing from acting;
- prevent raw-history accumulation from masquerading as state certainty.

## 6. Semantic transaction, rollback, and idempotency control

Research signal:
- *Cordon: Semantic Transactions for Tool-Using LLM Agents*
  (arXiv:2606.17573, 2026).

Why it matters:
Per-tool correctness does not protect a multi-step task with stateful
consequences. A harness needs a task-level commit boundary, rollback after failed
validation, and duplicate-request idempotency.

Test 1.2 addition:
- staged write + failed validation -> ROLLBACK;
- staged write + valid authorization/validation -> COMMIT;
- duplicate already-committed request id -> NOOP;
- unsafe commit/duplicate metric.

Artifact: `semantic-transaction-map.json`.

Compiler use:
- stage -> validate -> commit execution boundary;
- rollback invalid staged state;
- install idempotency guard where duplicate execution is unsafe.

## 7. Dynamic cost/state replanning

Research signal:
- *CostBench: Evaluating Multi-Turn Cost-Optimal Planning and Adaptation in
  Dynamic Environments for LLM Tool-Use Agents* (arXiv:2511.02734, 2025).
- *When Tools Fail: Benchmarking Dynamic Replanning and Anomaly Recovery in LLM
  Agents* / ToolMaze (arXiv:2606.05806, 2026).

Why it matters:
A plan that was optimal at time zero can become wrong after a price change,
tool outage, or other environmental event. Conversely, constant replanning is
wasteful when nothing changed.

Test 1.2 addition:
- cost increase changes optimal path;
- tool unavailability invalidates prior path;
- no-change control penalizes needless replanning;
- failed- and unnecessary-replan metrics.

Artifact: `dynamic-replanning-map.json`.

Compiler use:
- invalidate plan on cost/availability change;
- preserve current plan when its assumptions remain valid;
- route to replanning based on changed preconditions, not generic uncertainty.

## Time and non-removal contract

No prior Test 1.2 experiment is removed or shortened by this audit.

- Collection hard ceiling: 7h44m
- Collection active ceiling: 7h29m
- Tuning hard ceiling: 6h15m
- Tuning active ceiling: 6h
- Combined hard ceiling: **13h59m**

The second audit is implemented as a four-minute structured micro-lab phase.
The tuning compiler refuses to consume a collection unless all seven new maps
contain measured output.

The final compiled harness therefore receives both:
- `frontier_gap_policy` from the first seven research gaps; and
- `second_gap_policy` from this second seven-gap audit.
