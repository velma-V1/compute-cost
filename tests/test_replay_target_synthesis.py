import json
from pathlib import Path


def _level(level: int, label: str, *, passed: bool) -> dict:
    return {
        "level": level,
        "observation_count": 3,
        "valid_count": 3,
        "invalid_count": 0,
        "pass_count": 3 if passed else 0,
        "fail_count": 0 if passed else 3,
        "pass_rate": 1.0 if passed else 0.0,
        "label": label,
    }


def test_post_run_synthesis_emits_ranked_replay_targets_from_exact_registry(tmp_path: Path):
    from compute_cost.run_synthesis import build_cost_value_outputs

    frontiers = {
        "taxonomy_version": "capability-taxonomy-v1",
        "families": {
            "math": {
                "family_id": "math",
                "levels": [
                    _level(4, "reliable", passed=True),
                    _level(5, "failure", passed=False),
                ],
                "reliable_floor": 4,
                "first_failure_level": 5,
                "transition_bracket": {
                    "lower_level": 4,
                    "lower_label": "reliable",
                    "upper_level": 5,
                    "upper_label": "failure",
                },
                "coverage": {
                    "tested_levels": [4, 5],
                    "untested_levels": [0, 1, 2, 3, 6, 7, 8, 9, 10],
                    "tested_count": 2,
                },
            }
        },
    }
    (tmp_path / "coverage-ledger.json").write_text(
        json.dumps(
            {
                "families": {
                    "math": {
                        "state": "PROVEN",
                        "reliable_floor": 4,
                        "first_failure_level": 5,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "failure-atlas.json").write_text(
        json.dumps(
            {
                "failures": [
                    {
                        "experiment_id": "math-L5-fail",
                        "family_id": "math",
                        "difficulty_level": 5,
                        "result_class": "ANSWER_WRONG",
                        "failure_origin": "MODEL_FAILURE",
                        "valid_for_capability": True,
                        "recovery_level": None,
                        "failure_signature": {
                            "subtype": "arithmetic_error",
                            "causal_claim": False,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    (replay_dir / "index.jsonl").write_text(
        json.dumps(
            {
                "replay_id": "math-L5-fail",
                "category": "failures",
                "path": "replay/failures/math-L5-fail.json",
                "family_id": "math",
                "difficulty_level": 5,
                "result_class": "ANSWER_WRONG",
                "valid_for_capability": True,
                "recovery_level": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "resolved-config.json").write_text(
        json.dumps({"capability_campaign": {"boundary_repeats": 3}}),
        encoding="utf-8",
    )

    build_cost_value_outputs("gpt-oss:20b", [], frontiers, tmp_path)

    target_path = tmp_path / "replay-targets.json"
    assert target_path.is_file()
    targets = json.loads(target_path.read_text(encoding="utf-8"))
    assert targets["model"] == "gpt-oss:20b"
    assert targets["selection_policy"]["full_registry_is_source_of_truth"] is True
    assert targets["selection_policy"]["executes_model_calls"] is False
    assert targets["summary"]["selected_groups"] == 1
    assert targets["summary"]["raw_boundary_groups"] == 1
    assert targets["targets"][0]["reason"] == "RAW_BOUNDARY_RETEST"
    assert targets["targets"][0]["primary_replay_id"] == "math-L5-fail"
    assert targets["targets"][0]["primary_path"] == "replay/failures/math-L5-fail.json"
