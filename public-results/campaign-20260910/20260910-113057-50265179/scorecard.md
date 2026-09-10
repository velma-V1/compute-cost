# Capability Scorecard: gpt-oss:20b

Overall valid-observation score: 82.065% 
(151/184 correct; 16 invalid reported separately)

| Family | Score | Correct/Valid | Invalid | Tested levels | Reliable floor | First failure | Min correct budget | Median s |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| adversarial_wording_robustness | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.4816244 |
| algebra_quantitative_reasoning | 100.0% | 5/5 | 3 | L2,L5,L8,L10 | 10 | None | 256 | 4.49226065 |
| ambiguity_detection | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.5379186 |
| arithmetic_numerical_reasoning | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 1.2450141 |
| causal_counterfactual_reasoning | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 1.5563377 |
| code_comprehension | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 2.7453293 |
| coding_generation | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 2.0142895 |
| composite_agent_tasks | 0.0% | 0/3 | 0 | L0,L2 | None | 0 | None | 1.0640705 |
| context_reasoning | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 1.3271271 |
| context_retrieval | 0.0% | 0/3 | 1 | L0,L2 | None | 0 | None | 4.62031805 |
| contradictory_information_handling | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.5865666 |
| debugging_root_cause_diagnosis | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.553574 |
| decomposition | 0.0% | 0/3 | 0 | L0,L2 | None | 0 | None | 2.5410143 |
| distractor_noise_resistance | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.58869 |
| extraction_transformation | 0.0% | 0/3 | 0 | L0,L2 | None | 0 | None | 0.9092427 |
| formal_logic_deduction | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.9398896 |
| format_robustness | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 1.1680806 |
| hallucination_resistance | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.4485397 |
| instruction_following_constraint_stacking | 100.0% | 5/5 | 1 | L2,L5,L8,L10 | 10 | None | 256 | 3.9698462 |
| lost_in_middle_resistance | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 1.1815307 |
| memory_compression_summary_fidelity | 40.0% | 2/5 | 0 | L2,L3,L5 | 2 | 3 | 256 | 1.0899571 |
| meta_reasoning | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 1.376982 |
| missing_information_handling | 0.0% | 0/3 | 0 | L0,L2 | None | 0 | None | 0.7529643 |
| multi_tool_sequencing | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 2.5393215 |
| multi_turn_state_tracking | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 2.1500836 |
| planning_optimization | 83.333% | 5/6 | 6 | L2,L5,L6,L7,L8 | 7 | 8 | 512 | 5.42336985 |
| prompt_instruction_conflict_handling | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.4385136 |
| refactoring_under_constraints | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 2.0158318 |
| self_correction | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 1.9819805 |
| sibling_transfer_generalization | 66.667% | 4/6 | 2 | L2,L3,L4,L5 | 4 | 5 | 256 | 3.6782366499999997 |
| spatial_reasoning | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 1.9888058 |
| strict_structured_output | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 1.3274635 |
| temporal_reasoning | 40.0% | 2/5 | 0 | L2,L3,L5 | 2 | 3 | 256 | 1.2466738 |
| test_generation_verification | 0.0% | 0/3 | 1 | L0,L2 | None | 0 | None | 2.84245005 |
| tool_argument_correctness | 40.0% | 2/5 | 0 | L2,L3,L5 | 2 | 3 | 256 | 0.9106236 |
| tool_error_recovery | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 1.7598436 |
| tool_selection | 100.0% | 1/1 | 2 | L2,L5 | 2 | None | 256 | 0.662318 |
| uncertainty_calibration | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 1.0677233 |
| updated_obsolete_state_rejection | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.613911 |
| verification_critique | 0.0% | 0/3 | 0 | L0,L2 | None | 0 | None | 1.1948406 |

## Detailed family evidence

Each family entry in `scorecard.json` includes per-tested-level scores, result-class counts, token-budget curve, reasoning-mode breakdown, frontier data, runtime cost, and source experiment IDs. Untested levels are explicit; no score is fabricated for them.

# Long Autonomous Simulation: gpt-oss:20b

Scenarios: 6 | semantic turns: 48 | actual attempts: 48
Overall fully-correct step score: 81.2% | completion rate: 100.0%

| Scenario | Step score | Action | State | Change recovery | Complete | Attempts | Trunc retries | Median s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive_project_plan | 75.0% | 75.0% | 100.0% | 100.0% | YES | 8 | 0 | 0.98 |
| diagnostic_root_cause | 50.0% | 50.0% | 100.0% | 0.0% | YES | 8 | 0 | 1.32 |
| evidence_research | 100.0% | 100.0% | 100.0% | 100.0% | YES | 8 | 0 | 0.98 |
| repo_repair | 100.0% | 100.0% | 100.0% | 100.0% | YES | 8 | 0 | 0.89 |
| state_memory_continuation | 62.5% | 62.5% | 100.0% | 100.0% | YES | 8 | 0 | 1.65 |
| tool_chain_recovery | 100.0% | 100.0% | 100.0% | 100.0% | YES | 8 | 0 | 0.94 |

Each scenario preserves its full growing transcript plus raw request/response, scorer, telemetry, failure replay, token-budget, and per-turn evidence.
