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


def test_post_run_synthesis_persists_capability_profile_from_finalized_evidence(tmp_path: Path):
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
                "schema_version": 1,
                "taxonomy_version": "capability-taxonomy-v1",
                "families": {
                    "math": {
                        "state": "PROVEN",
                        "tested_levels": [4, 5],
                        "untested_levels": [0, 1, 2, 3, 6, 7, 8, 9, 10],
                        "valid_observations": 6,
                        "invalid_observations": 0,
                        "reliable_floor": 4,
                        "first_failure_level": 5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "failure-atlas.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "failures": [
                    {
                        "experiment_id": "math-L5-fail",
                        "family_id": "math",
                        "difficulty_level": 5,
                        "result_class": "ANSWER_WRONG",
                        "failure_origin": "MODEL_FAILURE",
                        "valid_for_capability": True,
                        "failure_signature": {
                            "subtype": "arithmetic_error",
                            "inference_kind": "FAMILY_LOCAL_SIGNATURE",
                            "causal_claim": False,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "resolved-config.json").write_text(
        json.dumps({"capability_campaign": {"boundary_repeats": 3}}),
        encoding="utf-8",
    )

    build_cost_value_outputs("gpt-oss:20b", [], frontiers, tmp_path)

    profile_path = tmp_path / "capability-profile.json"
    report_path = tmp_path / "capability-profile.md"
    assert profile_path.is_file()
    assert report_path.is_file()

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["model"] == "gpt-oss:20b"
    assert profile["summary"] == {
        "families_total": 1,
        "proven": 1,
        "partial": 0,
        "uncertain": 0,
        "untested": 0,
    }
    family = profile["families"]["math"]
    assert family["raw_reliable_through"] == 4
    assert family["first_raw_failure"] == 5
    assert family["observed_failure_signatures"] == ["arithmetic_error"]
    assert "RAW_FRONTIER_BOUNDARY" in family["signals"]

    report = report_path.read_text(encoding="utf-8")
    assert "# Capability Profile: gpt-oss:20b" in report
    assert "## math" in report
    assert "Evidence state: PROVEN" in report
    assert "Raw reliable through: L4" in report
    assert "First raw failure: L5" in report
    assert "arithmetic_error" in report
    assert "overall score" not in report.lower()
