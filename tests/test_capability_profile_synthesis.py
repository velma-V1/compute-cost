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


def test_post_run_synthesis_persists_capability_profile_and_required_outputs(tmp_path: Path):
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
    profile_report_path = tmp_path / "capability-profile.md"
    capability_map_path = tmp_path / "capability-map.json"
    weakness_map_path = tmp_path / "weakness-map.json"
    characterization_report_path = tmp_path / "characterization-report.md"
    for path in (
        profile_path,
        profile_report_path,
        capability_map_path,
        weakness_map_path,
        characterization_report_path,
    ):
        assert path.is_file(), path.name

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

    capability_map = json.loads(capability_map_path.read_text(encoding="utf-8"))
    assert capability_map == profile

    weakness_map = json.loads(weakness_map_path.read_text(encoding="utf-8"))
    assert weakness_map["model"] == "gpt-oss:20b"
    assert weakness_map["measurement_policy"]["evidence_gap_is_model_weakness"] is False
    weakness = weakness_map["families"]["math"]
    assert weakness["evidence_state"] == "PROVEN"
    assert weakness["first_raw_failure"] == 5
    assert "RAW_FRONTIER_BOUNDARY" in weakness["signals"]
    assert weakness["observed_failure_signatures"] == ["arithmetic_error"]

    profile_report = profile_report_path.read_text(encoding="utf-8")
    assert "# Capability Profile: gpt-oss:20b" in profile_report
    assert "## math" in profile_report
    assert "Evidence state: PROVEN" in profile_report
    assert "Raw reliable through: L4" in profile_report
    assert "First raw failure: L5" in profile_report
    assert "arithmetic_error" in profile_report
    assert "overall score" not in profile_report.lower()

    characterization_report = characterization_report_path.read_text(encoding="utf-8")
    assert "# Capability Characterization: gpt-oss:20b" in characterization_report
    assert "## Capability Map" in characterization_report
    assert "## Weakness Map" in characterization_report
    assert "## Operating Policy" in characterization_report
    assert "RAW_FRONTIER_BOUNDARY" in characterization_report
    assert "arithmetic_error" in characterization_report
    assert "ESCALATE" in characterization_report
    assert "overall score" not in characterization_report.lower()
