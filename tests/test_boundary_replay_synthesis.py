import json
from pathlib import Path


def test_post_run_synthesis_promotes_all_reproduced_upper_boundary_failures(tmp_path: Path):
    from compute_cost.run_synthesis import build_cost_value_outputs

    frontiers = {
        "taxonomy_version": "capability-taxonomy-v1",
        "families": {
            "math": {
                "family_id": "math",
                "levels": [
                    {
                        "level": 4,
                        "observation_count": 3,
                        "valid_count": 3,
                        "invalid_count": 0,
                        "pass_count": 3,
                        "fail_count": 0,
                        "pass_rate": 1.0,
                        "label": "reliable",
                    },
                    {
                        "level": 5,
                        "observation_count": 3,
                        "valid_count": 3,
                        "invalid_count": 0,
                        "pass_count": 0,
                        "fail_count": 3,
                        "pass_rate": 0.0,
                        "label": "failure",
                    },
                ],
                "reliable_floor": 4,
                "first_failure_level": 5,
                "transition_bracket": {
                    "lower_level": 4,
                    "lower_label": "reliable",
                    "upper_level": 5,
                    "upper_label": "failure",
                },
                "coverage": {"tested_levels": [4, 5], "untested_levels": [], "tested_count": 2},
            }
        },
    }
    failures = []
    replay_rows = []
    replay_failure_dir = tmp_path / "replay" / "failures"
    replay_failure_dir.mkdir(parents=True)
    for replay_id in ("math-L5-a", "math-L5-b", "math-L5-c"):
        snapshot = {
            "schema_version": 2,
            "experiment": {"experiment_id": replay_id, "difficulty_level": 5},
            "classification": {"result_class": "ANSWER_WRONG", "valid_for_capability": True},
        }
        source_path = replay_failure_dir / f"{replay_id}.json"
        source_path.write_text(json.dumps(snapshot, sort_keys=True) + "\n", encoding="utf-8")
        failures.append(
            {
                "experiment_id": replay_id,
                "family_id": "math",
                "difficulty_level": 5,
                "result_class": "ANSWER_WRONG",
                "failure_origin": "MODEL_FAILURE",
                "valid_for_capability": True,
                "recovery_level": None,
                "failure_signature": {"subtype": "arithmetic_error", "causal_claim": False},
            }
        )
        replay_rows.append(
            {
                "replay_id": replay_id,
                "category": "failures",
                "path": f"replay/failures/{replay_id}.json",
                "family_id": "math",
                "difficulty_level": 5,
                "result_class": "ANSWER_WRONG",
                "valid_for_capability": True,
                "recovery_level": None,
            }
        )

    # A harder failure remains a normal failure replay; it is not the reproduced
    # upper edge of the frozen raw frontier.
    extra_id = "math-L6-other"
    (replay_failure_dir / f"{extra_id}.json").write_text(
        json.dumps({"schema_version": 2, "experiment": {"experiment_id": extra_id}}) + "\n",
        encoding="utf-8",
    )
    failures.append(
        {
            "experiment_id": extra_id,
            "family_id": "math",
            "difficulty_level": 6,
            "result_class": "ANSWER_WRONG",
            "failure_origin": "MODEL_FAILURE",
            "valid_for_capability": True,
            "recovery_level": None,
            "failure_signature": {"subtype": "arithmetic_error", "causal_claim": False},
        }
    )
    replay_rows.append(
        {
            "replay_id": extra_id,
            "category": "failures",
            "path": f"replay/failures/{extra_id}.json",
            "family_id": "math",
            "difficulty_level": 6,
            "result_class": "ANSWER_WRONG",
            "valid_for_capability": True,
            "recovery_level": None,
        }
    )

    (tmp_path / "failure-atlas.json").write_text(
        json.dumps({"schema_version": 1, "failures": failures}), encoding="utf-8"
    )
    (tmp_path / "replay" / "index.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in replay_rows),
        encoding="utf-8",
    )
    (tmp_path / "resolved-config.json").write_text(
        json.dumps({"capability_campaign": {"boundary_repeats": 3}}), encoding="utf-8"
    )

    build_cost_value_outputs("gpt-oss:20b", [], frontiers, tmp_path)

    boundary_dir = tmp_path / "replay" / "boundaries"
    assert boundary_dir.is_dir()
    for replay_id in ("math-L5-a", "math-L5-b", "math-L5-c"):
        promoted = boundary_dir / f"{replay_id}.json"
        source = replay_failure_dir / f"{replay_id}.json"
        assert promoted.is_file()
        assert promoted.read_bytes() == source.read_bytes()
    assert not (boundary_dir / f"{extra_id}.json").exists()

    index_rows = [
        json.loads(line)
        for line in (boundary_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["replay_id"] for row in index_rows] == ["math-L5-a", "math-L5-b", "math-L5-c"]
    assert all(row["source_category"] == "failures" for row in index_rows)
    assert all(row["reason"] == "RAW_BOUNDARY_RETEST" for row in index_rows)
    assert all(row["difficulty_level"] == 5 for row in index_rows)
