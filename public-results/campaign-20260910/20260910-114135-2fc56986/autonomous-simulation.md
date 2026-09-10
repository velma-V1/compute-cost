# Long Autonomous Simulation: qwen3.5:35b-a3b-q4_K_M

Scenarios: 6 | semantic turns: 48 | actual attempts: 48
Overall fully-correct step score: 93.8% | completion rate: 100.0%

| Scenario | Step score | Action | State | Change recovery | Complete | Attempts | Trunc retries | Median s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| repo_repair | 75.0% | 75.0% | 100.0% | 100.0% | YES | 8 | 0 | 2.27 |
| tool_chain_recovery | 100.0% | 100.0% | 100.0% | 100.0% | YES | 8 | 0 | 2.25 |
| diagnostic_root_cause | 87.5% | 87.5% | 100.0% | 100.0% | YES | 8 | 0 | 3.24 |
| adaptive_project_plan | 100.0% | 100.0% | 100.0% | 100.0% | YES | 8 | 0 | 2.65 |
| evidence_research | 100.0% | 100.0% | 100.0% | 100.0% | YES | 8 | 0 | 2.97 |
| state_memory_continuation | 100.0% | 100.0% | 100.0% | 100.0% | YES | 8 | 0 | 2.09 |

Each scenario preserves its full growing transcript plus raw request/response, scorer, telemetry, failure replay, token-budget, and per-turn evidence.
