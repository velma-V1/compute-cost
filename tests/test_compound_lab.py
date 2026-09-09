from compute_cost.experiments import ExperimentSpec


def component_frontiers(level=7):
    families = {}
    components = {
        "extraction_transformation",
        "arithmetic_numerical_reasoning",
        "strict_structured_output",
        "formal_logic_deduction",
        "instruction_following_constraint_stacking",
        "temporal_reasoning",
        "planning_optimization",
        "tool_selection",
        "tool_argument_correctness",
        "multi_tool_sequencing",
        "context_retrieval",
        "contradictory_information_handling",
        "updated_obsolete_state_rejection",
        "code_comprehension",
        "debugging_root_cause_diagnosis",
        "verification_critique",
    }
    for family in components:
        families[family] = {"reliable_floor": level, "first_failure_level": level + 1}
    return {"families": families}


def test_compound_generator_builds_six_aligned_l0_l10_ladders():
    from compute_cost.compound_lab import COMPOUND_RECIPES, build_compound_cases

    cases = build_compound_cases()
    assert len(COMPOUND_RECIPES) == 6
    assert len(cases) == 66
    assert len({case["id"] for case in cases}) == 66

    by_recipe = {}
    for case in cases:
        by_recipe.setdefault(case["compound_id"], []).append(case)
        assert case["compound"] is True
        assert case["recovery_eligible"] is False
        assert case["robustness_eligible"] is False
        assert len(case["capabilities_required"]) >= 3
        level = case["difficulty_level"]
        assert 0 <= level <= 10
        assert case["component_level_map"] == {
            component: level for component in case["capabilities_required"]
        }
        assert case["scorer"] == "json"
        assert isinstance(case["expected"], dict)

    assert set(by_recipe) == set(COMPOUND_RECIPES)
    assert all(sorted(case["difficulty_level"] for case in rows) == list(range(11)) for rows in by_recipe.values())


def test_compound_target_expected_frontier_is_minimum_aligned_component_floor():
    from compute_cost.compound_lab import select_compound_targets

    frontiers = component_frontiers(7)
    frontiers["families"]["strict_structured_output"]["reliable_floor"] = 6
    selected = select_compound_targets(frontiers, max_compounds=6)

    extraction = selected["extract_calculate_json"]
    assert extraction["expected_frontier"] == 6
    assert extraction["component_frontiers"]["strict_structured_output"] == 6
    assert extraction["alignment_policy"] == "COMPONENT_LEVEL_EQUALS_COMPOUND_LEVEL"


def test_compound_target_skips_recipe_when_component_frontier_is_not_proven():
    from compute_cost.compound_lab import select_compound_targets

    frontiers = component_frontiers(7)
    frontiers["families"]["tool_argument_correctness"]["reliable_floor"] = None
    selected = select_compound_targets(frontiers, max_compounds=6)

    assert "tool_chain" not in selected
    assert "extract_calculate_json" in selected


class Store:
    def __init__(self):
        self.rows = {}
        self.jsons = {}

    def append_jsonl(self, path, value):
        self.rows.setdefault(path, []).append(value)

    def write_json(self, path, value, **kwargs):
        self.jsons[path] = value


class Runner:
    def __init__(self):
        self.store = Store()
        self.model = "gpt-oss:20b"
        self.progress = None
        self.config = {
            "capability_campaign": {
                "thinking_mode": True,
                "reasoning_effort": "medium",
                "generation_budget": 256,
                "reliable_threshold": 0.90,
                "unstable_threshold": 0.40,
            },
            "compound_lab": {
                "enabled": True,
                "jump": 2,
                "boundary_repeats": 2,
                "max_experiments_per_compound": 12,
                "max_compounds": 1,
            },
        }

    def _utc(self):
        return "x"


def test_compound_frontier_anchors_at_component_expectation_searches_down_and_measures_penalty(monkeypatch):
    import compute_cost.compound_lab as compound

    calls = []

    def execute(runner, case, experiment, *, parent=None):
        level = experiment.difficulty_level
        passed = level <= 4
        calls.append({
            "level": level,
            "changed_variable": experiment.changed_variable,
            "parent": None if parent is None else parent.experiment_id,
            "experiment_id": experiment.experiment_id,
        })
        return {
            "experiment": experiment.to_dict(),
            "classification": {
                "result_class": "ANSWER_CORRECT" if passed else "ANSWER_WRONG",
                "valid_for_capability": True,
            },
            "score": 1.0 if passed else 0.0,
            "status": "PASS" if passed else "FAIL",
        }

    monkeypatch.setattr(compound, "execute_experiment", execute)
    frontiers = component_frontiers(7)

    generated, compound_map, sequence = compound.run_compound_lab(
        Runner(),
        frontiers,
        sequence_start=20,
    )

    assert [item["level"] for item in calls] == [7, 5, 3, 4, 4, 5]
    assert [item["changed_variable"] for item in calls] == [
        "baseline",
        "difficulty_level",
        "difficulty_level",
        "difficulty_level",
        "replication",
        "replication",
    ]
    assert len(generated) == 6
    assert sequence == 26

    result = compound_map["compounds"]["extract_calculate_json"]
    assert result["expected_component_frontier"] == 7
    assert result["observed_compound_frontier"] == 4
    assert result["first_failure_level"] == 5
    assert result["composition_penalty"] == -3
    assert result["interaction_delta"] == -3
    assert result["status"] == "MEASURED"
    assert compound_map["measurement_policy"]["component_alignment"] == "COMPONENT_LEVEL_EQUALS_COMPOUND_LEVEL"


def test_composition_penalty_is_none_when_no_reliable_compound_floor_is_observed():
    from compute_cost.compound_lab import composition_penalty

    assert composition_penalty(7, 4) == -3
    assert composition_penalty(7, 7) == 0
    assert composition_penalty(7, 8) == 1
    assert composition_penalty(7, None) is None
