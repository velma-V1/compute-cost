def _case(level: int) -> dict:
    return {
        "id": f"math-L{level}",
        "family_id": "math",
        "category": "math",
        "difficulty_level": level,
        "difficulty": {
            "level": level,
            "rubric_version": "math-v1",
            "dimensions": {"steps": level},
        },
        "prompt": f"math L{level}",
        "scorer": "exact",
        "expected": "OK",
        "timeout_s": 120,
        "capabilities_required": ["math"],
        "recovery_eligible": True,
        "robustness_eligible": True,
        "compound": False,
        "tags": [],
    }


class Store:
    def __init__(self):
        self.rows = {}
        self.jsons = {}

    def append_jsonl(self, path, row):
        self.rows.setdefault(path, []).append(row)
        return {"path": path, "sha256": f"sha:{path}", "bytes": 1}

    def write_json(self, path, value, **kwargs):
        self.jsons[path] = value
        return {"path": path, "sha256": f"sha:{path}", "bytes": 1}


class Runner:
    def __init__(self):
        self.store = Store()
        self.model = "gpt-oss:20b"
        self.suite = {
            "benchmark_version": "cap-v1",
            "taxonomy_version": "capability-taxonomy-v1",
        }
        self.config = {
            "capability_campaign": {
                "anchor_level": 1,
                "jump": 3,
                "boundary_repeats": 2,
                "max_experiments_per_family": 12,
                "thinking_mode": True,
                "reasoning_effort": "medium",
                "generation_budget": 256,
                "reliable_threshold": 0.90,
                "unstable_threshold": 0.40,
            }
        }
        self.progress = None

    def _utc(self):
        return "2026-09-09T00:00:00Z"


def test_family_frontier_persists_exact_lower_and_upper_boundary_snapshots(monkeypatch):
    import compute_cost.capability_campaign as campaign

    outcomes = {1: "ANSWER_CORRECT", 4: "ANSWER_CORRECT", 5: "ANSWER_WRONG", 7: "ANSWER_WRONG"}

    def fake_execute(runner, fixture, spec, *, parent=None, snapshot_sink=None):
        level = int(fixture["difficulty_level"])
        result_class = outcomes[level]
        classification = {
            "result_class": result_class,
            "valid_for_capability": True,
        }
        if snapshot_sink is not None:
            snapshot_sink(
                {
                    "schema_version": 2,
                    "created_at_utc": runner._utc(),
                    "benchmark_version": runner.suite["benchmark_version"],
                    "stage": "characterize",
                    "model": runner.model,
                    "case": dict(fixture),
                    "experiment": spec.to_dict(),
                    "classification": dict(classification),
                    "evidence_key": spec.experiment_id,
                    "invocation": {"messages": [{"role": "user", "content": fixture["prompt"]}]},
                    "generation": {"normalized": {"text": "OK" if result_class == "ANSWER_CORRECT" else "WRONG"}},
                    "scoring": {"score": 1.0 if result_class == "ANSWER_CORRECT" else 0.0},
                    "telemetry_before_failure": [],
                    "resolved_config": runner.config,
                }
            )
        return {
            "experiment": spec.to_dict(),
            "classification": classification,
            "score": 1.0 if result_class == "ANSWER_CORRECT" else 0.0,
            "status": "SCORED",
        }

    monkeypatch.setattr(campaign, "execute_experiment", fake_execute)
    runner = Runner()
    ladder = {level: _case(level) for level in (1, 4, 5, 7)}

    rows, sequence = campaign.run_family_frontier(runner, "math", ladder)

    assert len(rows) == 6
    assert sequence == 6

    boundary_paths = sorted(path for path in runner.store.jsons if path.startswith("replay/boundaries/"))
    # Both replicated observations at the lower reliable and upper failure levels
    # are preserved, not just one representative snapshot.
    assert len(boundary_paths) == 4

    snapshots = [runner.store.jsons[path] for path in boundary_paths]
    roles = [snapshot["boundary"]["role"] for snapshot in snapshots]
    assert roles.count("lower_reliable") == 2
    assert roles.count("upper_transition") == 2
    assert {snapshot["boundary"]["level"] for snapshot in snapshots} == {4, 5}
    assert all(snapshot["source_experiment_id"] for snapshot in snapshots)
    assert all(snapshot["invocation"] for snapshot in snapshots)
    assert all(snapshot["generation"] for snapshot in snapshots)
    assert all(snapshot["scoring"] for snapshot in snapshots)

    boundary_index = [
        row for row in runner.store.rows["replay/index.jsonl"]
        if row["category"] == "boundaries"
    ]
    assert len(boundary_index) == 4
    assert len({row["replay_id"] for row in boundary_index}) == 4
    assert {row["boundary_role"] for row in boundary_index} == {"lower_reliable", "upper_transition"}
    assert all(row["replay_id"].endswith("--boundary") for row in boundary_index)
    assert all(row["source_experiment_id"] for row in boundary_index)
