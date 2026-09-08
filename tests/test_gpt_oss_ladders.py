import copy
import importlib
import importlib.util
import json
from pathlib import Path

from compute_cost.capability_suite import build_ladder_coverage, validate_capability_suite
from compute_cost.scoring import score_case


GENERATOR_VERSION = "gpt-oss-ladders-v1"


def load_json(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def module():
    spec = importlib.util.find_spec("compute_cost.gpt_oss_ladders")
    assert spec is not None, "compute_cost.gpt_oss_ladders is not implemented"
    return importlib.import_module("compute_cost.gpt_oss_ladders")


def test_expansion_is_deterministic_complete_and_does_not_mutate_sources():
    ladders = module()
    taxonomy = load_json("benchmarks/capability-taxonomy-v1.json")
    suite = load_json("benchmarks/gpt-oss-20b-capability-v1.json")
    taxonomy_source = copy.deepcopy(taxonomy)
    suite_source = copy.deepcopy(suite)

    first = ladders.expand_gpt_oss_ladders(suite, taxonomy)
    second = ladders.expand_gpt_oss_ladders(suite, taxonomy)

    assert ladders.GENERATOR_VERSION == GENERATOR_VERSION
    assert first == second
    assert suite == suite_source
    assert taxonomy == taxonomy_source
    assert len(first["cases"]) == 40 * 11
    assert first["ladder_generator_version"] == GENERATOR_VERSION
    validate_capability_suite(first, taxonomy)

    coverage = build_ladder_coverage(first, taxonomy)
    assert coverage["complete"] is True
    assert coverage["complete_family_count"] == 40
    assert all(row["declared_levels"] == list(range(11)) for row in coverage["families"].values())


def test_existing_breadth_anchors_are_preserved_exactly_inside_expanded_suite():
    ladders = module()
    taxonomy = load_json("benchmarks/capability-taxonomy-v1.json")
    suite = load_json("benchmarks/gpt-oss-20b-capability-v1.json")
    expanded = ladders.expand_gpt_oss_ladders(suite, taxonomy)

    by_id = {case["id"]: case for case in expanded["cases"]}
    for anchor in suite["cases"]:
        assert by_id[anchor["id"]] == anchor


def test_generated_fixtures_have_stable_provenance_and_family_local_dimensions():
    ladders = module()
    taxonomy = load_json("benchmarks/capability-taxonomy-v1.json")
    suite = load_json("benchmarks/gpt-oss-20b-capability-v1.json")
    expanded = ladders.expand_gpt_oss_ladders(suite, taxonomy)
    original_ids = {case["id"] for case in suite["cases"]}
    taxonomy_by_id = {family["id"]: family for family in taxonomy["families"]}

    generated = [case for case in expanded["cases"] if case["id"] not in original_ids]
    assert len(generated) == 400
    assert len({case["id"] for case in expanded["cases"]}) == 440

    for case in generated:
        meta = case["capability_map"]
        family = taxonomy_by_id[meta["family_id"]]
        assert case["fixture_generation"] == {
            "generator_version": GENERATOR_VERSION,
            "family_id": meta["family_id"],
            "level": case["difficulty_level"],
        }
        assert set(meta["difficulty"]["dimensions"]) == set(family["difficulty_dimensions"])
        assert meta["rubric_version"] == family["rubric_version"]
        assert case["oracle_response"]


def test_every_generated_oracle_is_accepted_by_its_declared_scorer():
    ladders = module()
    taxonomy = load_json("benchmarks/capability-taxonomy-v1.json")
    suite = load_json("benchmarks/gpt-oss-20b-capability-v1.json")
    expanded = ladders.expand_gpt_oss_ladders(suite, taxonomy)
    original_ids = {case["id"] for case in suite["cases"]}

    for case in expanded["cases"]:
        if case["id"] in original_ids:
            continue
        scored = score_case(case, str(case["oracle_response"]))
        assert scored["status"] == "SCORED", case["id"]
        assert scored["score"] == 1.0, (case["id"], scored)


def test_family_prompts_are_not_reused_across_difficulty_levels():
    ladders = module()
    taxonomy = load_json("benchmarks/capability-taxonomy-v1.json")
    suite = load_json("benchmarks/gpt-oss-20b-capability-v1.json")
    expanded = ladders.expand_gpt_oss_ladders(suite, taxonomy)

    prompts = {}
    for case in expanded["cases"]:
        family = case["capability_map"]["family_id"]
        prompts.setdefault(family, []).append(case["prompt"])
    assert all(len(values) == 11 and len(set(values)) == 11 for values in prompts.values())
