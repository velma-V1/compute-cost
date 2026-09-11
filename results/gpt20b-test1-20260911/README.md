# GPT-OSS 20B Test 1 — Sanitized Results and Pattern Audit

Source run: `test1-20260911-112554-e0d1049f`

## Privacy status

The uploaded Test-1 pattern pack was scanned across 221,253,187 bytes. No direct personal data, machine-user paths, emails, device identifiers, external IP addresses, API keys, auth tokens, or sensitive JSON identity fields were found. No content redactions were required. See `privacy-scan.json` for the scan contract and file hashes.

## Executive result

Test 1 produced **3,039 observations**. 2,978 (97.99%) were unchanged, 12 (0.39%) improved, and 49 (1.61%) regressed.

The main result is **not that prompt controls damage reasoning**. Of the 49 regressions, **48 are `THINK_TRUNCATED` and 1 is `NO_FINAL_ANSWER`**. The treatment is primarily interacting with completion/generation budget.

## Strong patterns

1. **Ceiling/headroom confound.** 2,580/3,039 observations (84.9%) had baseline score 1.0, so most trials could not improve numerically but could regress.
2. **Low dose is materially safer.** In the controlled characterization grid, dose 0.5 had 0 regressions / 432 trials; dose 1.0 had 9; dose 2.0 had 12.
3. **Placement matters.** System placement: 0 regressions / 324; suffix: 3; prefix: 6; middle: 12.
4. **Schema is the riskiest representation.** Schema produced 13 regressions / 432 versus 4 each for prose and bullets.
5. **`ING-011 tool_argument_validation` contains a missed rescue signal.** It improved 2 of the 4 baseline-failing discovery fixtures while causing no regression on 12 baseline-passing fixtures. The global median classifier still labeled it `NULL`.
6. **Localized improvements exist.** All 12 positive deltas occur in `coding_generation` (10) and `context_retrieval` (2). These are useful signals but are not yet independent/generalized evidence.
7. **The handoff lost information.** `test2-priority-queue.json`, `higher-order-candidate-queue.json`, and `uncertainty-ledger.json` are empty despite the nonzero outcomes above.
8. **The frozen 7-hour campaign did not consume its planned active window.** Recorded experimental activity lasted about 2.80 hours (41.0% of the planned 6h50m active window).
9. **Binary scoring and the noise model are mismatched.** Calibration MAD is zero in every family, so the 0.05 floor makes a single 0→1 or 1→0 event a 20-sigma event. Binary paired outcomes need rate/confidence treatment in addition to the current median/MAD logic.

## Consequence for Test 2 / Test 3

The empty Test-2 queue must **not** be interpreted as “Test 1 found nothing useful.” It means the global aggregation rule discarded conditional signals. Test 2 fallback recipes should be interpreted as fallback controls, not Test-1-discovered optima. Test 3 qualification should separate model-owned correctness failures from generation-budget/truncation failures.

## Recommended corrections

- Score **rescue rate** on baseline failures separately from **regression rate** on baseline passes.
- Separate `THINK_TRUNCATED` / `ANSWER_TRUNCATED` from wrong-answer capability failures.
- Make generation budget an experimental factor before assigning prompt harm.
- Use paired binary confidence/statistics for 0/1 outcomes.
- Promote conditional rescue signals such as ING-011 even when global median delta is zero.
- Spend unused phase time on replication/generalization instead of terminating when the candidate queue is empty.

## Repository contents

The compact repo export contains `pattern-analysis.json`, `privacy-scan.json`, `failure-summary.json`, and `observation-summary.json`. The 212 MB raw-pattern evidence and large cross-product maps remain in the cleaned ZIP; their source file hashes are retained in `privacy-scan.json`.
