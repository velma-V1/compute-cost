# Adaptive Characterization Report

Model: `qwen3.5:27b-q4_K_M`

Exposed thinking is an observable behavioral trace, not hidden cognition.
Aggregate `eval_count` is MEASURED runtime evidence and is not split into fabricated per-phase token counts.

## Task profiles

### char-if-001
- Think OFF: ANSWER_CORRECT (MEASURED)
- Minimum reproduced passing generation budget: 128 (DERIVED)
- Transition bracket: 96 fail/truncate -> 128 reproduced pass (DERIVED)

### char-json-001
- Think OFF: ANSWER_CORRECT (MEASURED)
- Minimum reproduced passing generation budget: 256 (DERIVED)
- Transition bracket: 224 fail/truncate -> 256 reproduced pass (DERIVED)

### char-math-001
- Think OFF: ANSWER_WRONG (MEASURED)
- Minimum reproduced passing generation budget: 704 (DERIVED)
- Transition bracket: 672 fail/truncate -> 704 reproduced pass (DERIVED)
