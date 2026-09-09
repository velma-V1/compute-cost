import json
from pathlib import Path


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


def _row(experiment_id: str, level: int, result_class: str, request_id: str) -> dict:
    return {
        "experiment": {
            "experiment_id": experiment_id,
            "parent_experiment_id": None,
            "task_id": "math",
            "task_family": "math",
            "difficulty_level": level,
            "hypothesis": "boundary replication",
            "changed_variable": "replication",
            "thinking_mode": True,
            "reasoning_effort": "medium",
            "generation_budget": 256,
            "context_request": None,
            "temperature": 0.0,
            "seed": 42,
            "prompt_variant": "base",
            "recovery_level": None,
        },
        "classification": {
            "result_class": result_class,
            "valid_for_capability": True,
        },
        "score": 1.0 if result_class == "ANSWER_CORRECT" else 0.0,
        "status": "SCORED",
        "evidence_key": experiment_id,
        "evidence_refs": {"request_id": request_id},
    }


def _write_retained_evidence(root: Path, row: dict, case: dict) -> None:
    experiment = row["experiment"]
    experiment_id = experiment["experiment_id"]
    request_id = row["evidence_refs"]["request_id"]
    result_class = row["classification"]["result_class"]
    response = "OK" if result_class == "ANSWER_CORRECT" else "WRONG"

    exchange_dir = root / "raw" / "runtime" / "exchanges"
    scoring_dir = root / "raw" / "scoring"
    exchange_dir.mkdir(parents=True, exist_ok=True)
    scoring_dir.mkdir(parents=True, exist_ok=True)

    request_body = {
        "model": "gpt-oss:20b",
        "messages": [{"role": "user", "content": case["prompt"]}],
        "stream": True,
        "options": {
            "num_predict": 256,
            "temperature": 0.0,
            "seed": 42,
        },
        "think": "medium",
    }
    exchange = {
        "ok": True,
        "http_status": 200,
        "request": {
            "body_text": json.dumps(request_body, separators=(",", ":")),
            "started_monotonic_ns": 100,
        },
        "stream_events": [
            {
                "sequence": 0,
                "received_monotonic_ns": 200,
                "parsed": {"message": {"content": response}},
            }
        ],
        "normalized": {"text": response, "thinking": "reasoned", "tool_calls": []},
        "metrics": {"eval_count": 8},
        "timing": {
            "client_started_monotonic_ns": 100,
            "client_ended_monotonic_ns": 300,
            "client_latency_ns": 200,
        },
    }
    (exchange_dir / f"{request_id}.json").write_text(json.dumps(exchange), encoding="utf-8")
    scoring = {
        "score": row["score"],
        "status": "SCORED",
        "checks": [{"name": "exact_match", "pass": result_class == "ANSWER_CORRECT"}],
        "evidence": {"raw_response": response},
    }
    safe = experiment_id.replace("/", "-").replace("\\", "-")
    (scoring_dir / f"characterize-{safe}.json").write_text(json.dumps(scoring), encoding="utf-8")


def test_materializer_persists_exact_replicated_lower_and_upper_boundary_snapshots(tmp_path: Path):
    from compute_cost.boundary_replay import materialize_boundary_replays

    cases = [_case(4), _case(5)]
    (tmp_path / "benchmark-snapshot.json").write_text(
        json.dumps({"benchmark_version": "cap-v1", "cases": cases}),
        encoding="utf-8",
    )
    (tmp_path / "resolved-config.json").write_text(
        json.dumps({"capability_campaign": {"boundary_repeats": 2}}),
        encoding="utf-8",
    )

    rows = [
        _row("math-L4-a", 4, "ANSWER_CORRECT", "req-l4-a"),
        _row("math-L4-b", 4, "ANSWER_CORRECT", "req-l4-b"),
        _row("math-L5-a", 5, "ANSWER_WRONG", "req-l5-a"),
        _row("math-L5-b", 5, "ANSWER_WRONG", "req-l5-b"),
    ]
    for row in rows:
        level = row["experiment"]["difficulty_level"]
        _write_retained_evidence(tmp_path, row, next(case for case in cases if case["difficulty_level"] == level))

    telemetry = []
    for index, row in enumerate(rows):
        telemetry.append(
            {
                "timestamp_utc": "x",
                "monotonic_ns": 1000 + index,
                "stage": "characterize",
                "case_id": row["experiment"]["experiment_id"],
                "host": {"memory_percent": 1.0},
            }
        )
    (tmp_path / "telemetry.jsonl").write_text(
        "\n".join(json.dumps(item) for item in telemetry) + "\n",
        encoding="utf-8",
    )

    frontiers = {
        "taxonomy_version": "capability-taxonomy-v1",
        "families": {
            "math": {
                "family_id": "math",
                "reliable_floor": 4,
                "first_failure_level": 5,
                "transition_bracket": {
                    "lower_level": 4,
                    "lower_label": "reliable",
                    "upper_level": 5,
                    "upper_label": "failure",
                },
            }
        },
    }

    result = materialize_boundary_replays("gpt-oss:20b", frontiers, rows, tmp_path)

    assert result["summary"]["boundary_snapshots"] == 4
    assert result["summary"]["boundary_families"] == 1
    boundary_dir = tmp_path / "replay" / "boundaries"
    paths = sorted(boundary_dir.glob("*.json"))
    assert len(paths) == 4

    snapshots = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    roles = [snapshot["boundary"]["role"] for snapshot in snapshots]
    assert roles.count("lower_reliable") == 2
    assert roles.count("upper_transition") == 2
    assert {snapshot["boundary"]["level"] for snapshot in snapshots} == {4, 5}
    assert all(snapshot["source_experiment_id"] for snapshot in snapshots)
    assert all(snapshot["invocation"]["request_fields"] == {"think": "medium"} for snapshot in snapshots)
    assert all(snapshot["generation"]["request"]["body_text"] for snapshot in snapshots)
    assert all(snapshot["scoring"]["status"] == "SCORED" for snapshot in snapshots)
    assert all(len(snapshot["telemetry_for_experiment"]) == 1 for snapshot in snapshots)

    index = [json.loads(line) for line in (tmp_path / "replay" / "index.jsonl").read_text().splitlines()]
    assert len(index) == 4
    assert len({row["replay_id"] for row in index}) == 4
    assert {row["boundary_role"] for row in index} == {"lower_reliable", "upper_transition"}
    assert all(row["category"] == "boundaries" for row in index)
    assert all(row["replay_id"].endswith("--boundary") for row in index)
    assert all(row["source_experiment_id"] for row in index)

    # Synthesis must be idempotent: rerunning it cannot duplicate boundary files or index entries.
    second = materialize_boundary_replays("gpt-oss:20b", frontiers, rows, tmp_path)
    assert second["summary"]["boundary_snapshots"] == 4
    index_again = [json.loads(line) for line in (tmp_path / "replay" / "index.jsonl").read_text().splitlines()]
    assert len(index_again) == 4
