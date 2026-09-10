# Long Autonomous Simulation: gpt-oss:20b

Scenarios: 6 | semantic turns: 48 | actual attempts: 48
Overall fully-correct step score: 81.2% | completion rate: 100.0%

| Scenario | Step score | Action | State | Change recovery | Complete | Attempts | Trunc retries | Median s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| repo_repair | 100.0% | 100.0% | 100.0% | 100.0% | YES | 8 | 0 | 0.89 |
| tool_chain_recovery | 100.0% | 100.0% | 100.0% | 100.0% | YES | 8 | 0 | 0.94 |
| diagnostic_root_cause | 50.0% | 50.0% | 100.0% | 0.0% | YES | 8 | 0 | 1.32 |
| adaptive_project_plan | 75.0% | 75.0% | 100.0% | 100.0% | YES | 8 | 0 | 0.98 |
| evidence_research | 100.0% | 100.0% | 100.0% | 100.0% | YES | 8 | 0 | 0.98 |
| state_memory_continuation | 62.5% | 62.5% | 100.0% | 100.0% | YES | 8 | 0 | 1.65 |

Each scenario preserves its full growing transcript plus raw request/response, scorer, telemetry, failure replay, token-budget, and per-turn evidence.
