# Adaptive Characterization Report

Model: `qwen3.5:27b-q8_0`

Exposed thinking is an observable behavioral trace, not hidden cognition.
Aggregate `eval_count` is MEASURED runtime evidence and is not split into fabricated per-phase token counts.

## Task profiles

### char-if-001
- Think OFF: ANSWER_CORRECT (MEASURED)
- Minimum reproduced passing generation budget: 128 (DERIVED)
- Transition bracket: 96 fail/truncate -> 128 reproduced pass (DERIVED)

### char-json-001
- Think OFF: ANSWER_CORRECT (MEASURED)
- Minimum reproduced passing generation budget: 352 (DERIVED)
- Transition bracket: 320 fail/truncate -> 352 reproduced pass (DERIVED)

### char-math-001
- Think OFF: ANSWER_WRONG (MEASURED)
- Minimum reproduced passing generation budget: 576 (DERIVED)
- Transition bracket: 544 fail/truncate -> 576 reproduced pass (DERIVED)
