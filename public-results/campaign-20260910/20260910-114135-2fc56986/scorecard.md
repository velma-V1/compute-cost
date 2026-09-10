# Capability Scorecard: qwen3.5:35b-a3b-q4_K_M

Overall valid-observation score: 69.231% 
(135/195 correct; 0 invalid reported separately)

| Family | Score | Correct/Valid | Invalid | Tested levels | Reliable floor | First failure | Min correct budget | Median s |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| adversarial_wording_robustness | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.6117114 |
| algebra_quantitative_reasoning | 50.0% | 3/6 | 0 | L2,L3,L4,L5 | 3 | 4 | 256 | 0.36086005 |
| ambiguity_detection | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.5320547 |
| arithmetic_numerical_reasoning | 40.0% | 2/5 | 0 | L2,L3,L5 | 2 | 3 | 256 | 0.3839188 |
| causal_counterfactual_reasoning | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.4847858 |
| code_comprehension | 0.0% | 0/3 | 0 | L0,L2 | None | 0 | None | 0.3734971 |
| coding_generation | 50.0% | 3/6 | 0 | L2,L3,L4,L5 | 3 | 4 | 256 | 1.19449805 |
| composite_agent_tasks | 0.0% | 0/3 | 0 | L0,L2 | None | 0 | None | 1.8901064 |
| context_reasoning | 60.0% | 3/5 | 0 | L0,L1,L2 | 1 | 2 | 256 | 0.4218737 |
| context_retrieval | 83.333% | 5/6 | 0 | L2,L5,L6,L7,L8 | 7 | 8 | 256 | 1.87625065 |
| contradictory_information_handling | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.5741366 |
| debugging_root_cause_diagnosis | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.638121 |
| decomposition | 0.0% | 0/3 | 0 | L0,L2 | None | 0 | None | 2.4698297 |
| distractor_noise_resistance | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.6676372 |
| extraction_transformation | 40.0% | 2/5 | 0 | L2,L3,L5 | 2 | 3 | 256 | 0.5544027 |
| formal_logic_deduction | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.4496795 |
| format_robustness | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 1.4013423 |
| hallucination_resistance | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.5559926 |
| instruction_following_constraint_stacking | 50.0% | 3/6 | 0 | L2,L3,L4,L5 | 3 | 4 | 256 | 0.70950565 |
| lost_in_middle_resistance | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 1.7226759 |
| memory_compression_summary_fidelity | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.9616488 |
| meta_reasoning | 40.0% | 2/5 | 0 | L0,L1,L2 | 0 | 1 | 256 | 0.441918 |
| missing_information_handling | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.7284973 |
| multi_tool_sequencing | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 2.0005222 |
| multi_turn_state_tracking | 66.667% | 4/6 | 0 | L2,L5,L6,L7,L8 | 6 | 7 | 256 | 0.6286441 |
| planning_optimization | 60.0% | 3/5 | 0 | L0,L1,L2 | 1 | 2 | 256 | 0.4747001 |
| prompt_instruction_conflict_handling | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.4835118 |
| refactoring_under_constraints | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.4979878 |
| self_correction | 40.0% | 2/5 | 0 | L2,L3,L5 | 2 | 3 | 256 | 0.5350484 |
| sibling_transfer_generalization | 0.0% | 0/3 | 0 | L0,L2 | None | 0 | None | 0.7280222 |
| spatial_reasoning | 50.0% | 3/6 | 0 | L2,L3,L4,L5 | 3 | 4 | 256 | 0.4732743 |
| strict_structured_output | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 1.7321824 |
| temporal_reasoning | 40.0% | 2/5 | 0 | L0,L1,L2 | 0 | 1 | 256 | 0.4988451 |
| test_generation_verification | 0.0% | 0/3 | 0 | L0,L2 | None | 0 | None | 0.7561129 |
| tool_argument_correctness | 40.0% | 2/5 | 0 | L2,L3,L5 | 2 | 3 | 256 | 0.7412565 |
| tool_error_recovery | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 1.3417081 |
| tool_selection | 40.0% | 2/5 | 0 | L2,L3,L5 | 2 | 3 | 256 | 0.6910124 |
| uncertainty_calibration | 66.667% | 4/6 | 0 | L2,L3,L4,L5 | 4 | 5 | 256 | 0.42822360000000004 |
| updated_obsolete_state_rejection | 100.0% | 5/5 | 0 | L2,L5,L8,L10 | 10 | None | 256 | 0.5682291 |
| verification_critique | 0.0% | 0/3 | 0 | L0,L2 | None | 0 | None | 0.5967565 |

## Detailed family evidence

Each family entry in `scorecard.json` includes per-tested-level scores, result-class counts, token-budget curve, reasoning-mode breakdown, frontier data, runtime cost, and source experiment IDs. Untested levels are explicit; no score is fabricated for them.

# Long Autonomous Simulation: qwen3.5:35b-a3b-q4_K_M

Scenarios: 6 | semantic turns: 48 | actual attempts: 48
Overall fully-correct step score: 93.8% | completion rate: 100.0%

| Scenario | Step score | Action | State | Change recovery | Complete | Attempts | Trunc retries | Median s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive_project_plan | 100.0% | 100.0% | 100.0% | 100.0% | YES | 8 | 0 | 2.65 |
| diagnostic_root_cause | 87.5% | 87.5% | 100.0% | 100.0% | YES | 8 | 0 | 3.24 |
| evidence_research | 100.0% | 100.0% | 100.0% | 100.0% | YES | 8 | 0 | 2.97 |
| repo_repair | 75.0% | 75.0% | 100.0% | 100.0% | YES | 8 | 0 | 2.27 |
| state_memory_continuation | 100.0% | 100.0% | 100.0% | 100.0% | YES | 8 | 0 | 2.09 |
| tool_chain_recovery | 100.0% | 100.0% | 100.0% | 100.0% | YES | 8 | 0 | 2.25 |

Each scenario preserves its full growing transcript plus raw request/response, scorer, telemetry, failure replay, token-budget, and per-turn evidence.
