from compute_cost.frontier import build_capability_frontiers


def obs(level, passed, valid=True):
    return {
        "level": level,
        "passed": passed,
        "valid_for_capability": valid,
    }


def test_capability_frontiers_artifact_has_versioned_deterministic_shape():
    artifact = build_capability_frontiers(
        "capability-taxonomy-v1",
        {
            "z_family": [obs(2, True) for _ in range(5)],
            "a_family": [obs(1, False) for _ in range(5)],
        },
    )

    assert artifact["schema_version"] == 1
    assert artifact["taxonomy_version"] == "capability-taxonomy-v1"
    assert artifact["thresholds"] == {"reliable": 0.9, "unstable": 0.4}
    assert list(artifact["families"]) == ["a_family", "z_family"]
    assert artifact["families"]["a_family"]["first_failure_level"] == 1
    assert artifact["families"]["z_family"]["reliable_floor"] == 2


def test_capability_frontiers_artifact_propagates_custom_thresholds():
    artifact = build_capability_frontiers(
        "capability-taxonomy-v1",
        {
            "family": [
                obs(3, True),
                obs(3, True),
                obs(3, True),
                obs(3, True),
                obs(3, False),
            ]
        },
        thresholds={"reliable": 0.8, "unstable": 0.2},
    )

    assert artifact["thresholds"] == {"reliable": 0.8, "unstable": 0.2}
    assert artifact["families"]["family"]["levels"][0]["pass_rate"] == 0.8
    assert artifact["families"]["family"]["levels"][0]["label"] == "reliable"


def test_capability_frontiers_artifact_retains_unresolved_family_coverage():
    artifact = build_capability_frontiers(
        "capability-taxonomy-v1",
        {"untested_family": []},
    )
    family = artifact["families"]["untested_family"]
    assert family["levels"] == []
    assert family["reliable_floor"] is None
    assert family["coverage"]["tested_count"] == 0
    assert family["coverage"]["untested_levels"] == list(range(11))
