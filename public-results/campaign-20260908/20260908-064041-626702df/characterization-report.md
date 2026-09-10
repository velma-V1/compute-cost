# Adaptive Characterization Report

Model: `qwen3.5:35b-a3b-q4_K_M`

Exposed thinking is an observable behavioral trace, not hidden cognition.
Aggregate `eval_count` is MEASURED runtime evidence and is not split into fabricated per-phase token counts.

## Task profiles

### char-if-001
- Think OFF: ANSWER_CORRECT (MEASURED)
- Minimum reproduced passing generation budget: 192 (DERIVED)
- Transition bracket: 160 fail/truncate -> 192 reproduced pass (DERIVED)

### char-json-001
- Think OFF: ANSWER_CORRECT (MEASURED)
- Minimum reproduced passing generation budget: 288 (DERIVED)
- Transition bracket: 256 fail/truncate -> 288 reproduced pass (DERIVED)

### char-math-001
- Think OFF: ANSWER_WRONG (MEASURED)
- Minimum reproduced passing generation budget: 640 (DERIVED)
- Transition bracket: 608 fail/truncate -> 640 reproduced pass (DERIVED)
