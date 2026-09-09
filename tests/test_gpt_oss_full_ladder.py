import json
from pathlib import Path

from compute_cost.capability_suite import build_ladder_coverage, validate_capability_suite


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_gpt_oss_campaign_declares_every_family_at_every_difficulty_level():
    taxonomy = _load("benchmarks/capability-taxonomy-v1.json")
    suite = _load("benchmarks/gpt-oss-20b-capability-v1.json")

    validate_capability_suite(suite, taxonomy)
    coverage = build_ladder_coverage(suite, taxonomy)

    assert coverage["family_count"] == 40
    assert coverage["complete_family_count"] == 40
    assert coverage["complete"] is True
    assert len(suite["cases"]) == 40 * 11

    expected_levels = list(range(11))
    for family_id, family in coverage["families"].items():
        assert family["declared_levels"] == expected_levels, family_id
        assert family["missing_levels"] == [], family_id
        assert family["declared_count"] == 11, family_id
        assert family["complete"] is True, family_id

    ids = [case["id"] for case in suite["cases"]]
    assert len(ids) == len(set(ids))
