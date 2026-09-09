import pytest

from compute_cost.frontier import build_family_frontier, classify_pass_rate


def fobs(level, passed, valid=True):
    return {
        "level": level,
        "passed": passed,
        "valid_for_capability": valid,
    }


def test_pass_rate_threshold_boundaries_are_explicit():
    assert classify_pass_rate(1.0) == "reliable"
    assert classify_pass_rate(0.90) == "reliable"
    assert classify_pass_rate(0.899999) == "unstable"
    assert classify_pass_rate(0.40) == "unstable"
    assert classify_pass_rate(0.399999) == "failure"
    assert classify_pass_rate(0.0) == "failure"


def test_pass_rate_rejects_invalid_threshold_configuration():
    with pytest.raises(ValueError, match="threshold"):
        classify_pass_rate(0.8, reliable_threshold=0.3, unstable_threshold=0.4)
    with pytest.raises(ValueError, match="pass_rate"):
        classify_pass_rate(1.1)


def test_invalid_observations_are_retained_but_excluded_from_pass_rate():
    rows = [fobs(5, True) for _ in range(9)]
    rows += [fobs(5, False), fobs(5, None, False), fobs(5, False, False)]
    frontier = build_family_frontier("reasoning", rows)
    level = frontier["levels"][0]

    assert level["level"] == 5
    assert level["observation_count"] == 12
    assert level["valid_count"] == 10
    assert level["invalid_count"] == 2
    assert level["pass_count"] == 9
    assert level["fail_count"] == 1
    assert level["pass_rate"] == 0.9
    assert level["label"] == "reliable"


def test_frontier_derives_reliable_floor_unstable_region_and_first_failure():
    rows = []
    rows += [fobs(2, True) for _ in range(5)]
    rows += [fobs(3, True) for _ in range(5)]
    rows += [fobs(4, True), fobs(4, True), fobs(4, False), fobs(4, False), fobs(4, False)]
    rows += [fobs(5, False) for _ in range(5)]

    frontier = build_family_frontier("reasoning", rows)
    assert frontier["reliable_floor"] == 3
    assert frontier["unstable_levels"] == [4]
    assert frontier["failure_levels"] == [5]
    assert frontier["first_failure_level"] == 5
    assert frontier["transition_bracket"] == {
        "lower_level": 3,
        "lower_label": "reliable",
        "upper_level": 4,
        "upper_label": "unstable",
    }


def test_sparse_frontier_records_untested_levels_and_nearest_transition():
    rows = [fobs(2, True) for _ in range(5)] + [fobs(6, False) for _ in range(5)]
    frontier = build_family_frontier("sparse", rows)

    assert frontier["reliable_floor"] == 2
    assert frontier["first_failure_level"] == 6
    assert frontier["transition_bracket"] == {
        "lower_level": 2,
        "lower_label": "reliable",
        "upper_level": 6,
        "upper_label": "failure",
    }
    assert frontier["coverage"]["tested_levels"] == [2, 6]
    assert frontier["coverage"]["tested_count"] == 2
    assert frontier["coverage"]["untested_levels"] == [0, 1, 3, 4, 5, 7, 8, 9, 10]


def test_invalid_only_level_is_unresolved_and_does_not_break_reliable_floor():
    rows = [fobs(2, True) for _ in range(5)]
    rows += [fobs(3, None, False), fobs(3, False, False)]
    rows += [fobs(4, True) for _ in range(5)]
    frontier = build_family_frontier("invalid-gap", rows)

    levels = {item["level"]: item for item in frontier["levels"]}
    assert levels[3]["label"] == "unresolved"
    assert levels[3]["pass_rate"] is None
    assert frontier["reliable_floor"] == 4
    assert frontier["unstable_levels"] == []
    assert frontier["failure_levels"] == []
    assert frontier["first_failure_level"] is None
    assert frontier["transition_bracket"] is None
