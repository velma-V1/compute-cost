# Adaptive Characterization Report

Model: `qwen3.5:35b-a3b-q8_0`

Exposed thinking is an observable behavioral trace, not hidden cognition.
Aggregate `eval_count` is MEASURED runtime evidence and is not split into fabricated per-phase token counts.

## Task profiles

### char-if-001
- Think OFF: ANSWER_CORRECT (MEASURED)
- Minimum reproduced passing generation budget: 160 (DERIVED)
- Transition bracket: 128 fail/truncate -> 160 reproduced pass (DERIVED)

### char-json-001
- Think OFF: ANSWER_CORRECT (MEASURED)
- Minimum reproduced passing generation budget: 256 (DERIVED)
- Transition bracket: 224 fail/truncate -> 256 reproduced pass (DERIVED)

### char-math-001
- Think OFF: ANSWER_WRONG (MEASURED)
- Minimum reproduced passing generation budget: 544 (DERIVED)
- Transition bracket: 512 fail/truncate -> 544 reproduced pass (DERIVED)
