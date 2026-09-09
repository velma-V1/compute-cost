def experiment_row(
    experiment_id,
    *,
    family="formal_logic_deduction",
    level=4,
    result_class="ANSWER_CORRECT",
    eval_count=100,
    prompt_eval_count=20,
    eval_duration_ns=2_000_000_000,
    total_duration_ns=3_000_000_000,
    client_latency_ns=3_500_000_000,
):
    return {
        "experiment": {
            "experiment_id": experiment_id,
            "task_id": family,
            "task_family": family,
            "difficulty_level": level,
        },
        "classification": {
            "result_class": result_class,
            "valid_for_capability": True,
        },
        "metrics": {
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "eval_duration_ns": eval_duration_ns,
            "total_duration_ns": total_duration_ns,
        },
        "timing": {"client_latency_ns": client_latency_ns},
    }


def test_value_map_uses_measured_compute_and_demonstrated_capability_effects_only():
    from compute_cost.capability_value import build_capability_value_map

    base_rows = [
        experiment_row("base-floor", level=4, eval_count=100),
        experiment_row("base-fail", level=5, result_class="ANSWER_WRONG", eval_count=110),
    ]
    reasoning_rows = [
        experiment_row("low-floor-1", level=4, eval_count=55, eval_duration_ns=1_100_000_000),
        experiment_row("low-floor-2", level=4, eval_count=45, eval_duration_ns=900_000_000),
        experiment_row("high-fail-1", level=5, eval_count=160, eval_duration_ns=3_200_000_000),
        experiment_row("high-fail-2", level=5, eval_count=150, eval_duration_ns=3_000_000_000),
    ]
    recovery_rows = [
        experiment_row("r3-1", level=5, eval_count=125),
        experiment_row("r3-2", level=5, eval_count=120),
    ]
    robustness_rows = [experiment_row("robust-1", level=5, eval_count=118)]
    compound_rows = [
        experiment_row(
            "compound-1",
            family="extract_calculate_json",
            level=3,
            eval_count=140,
        )
    ]

    frontiers = {
        "families": {
            "formal_logic_deduction": {
                "reliable_floor": 4,
                "first_failure_level": 5,
            }
        }
    }
    reasoning_curves = {
        "families": {
            "formal_logic_deduction": {
                "minimum_reliable_effort_at_baseline_floor": "low",
                "demonstrated_high_effort_extension_to": 5,
            }
        }
    }
    recovery_map = {
        "families": {
            "formal_logic_deduction": {
                "difficulty_level": 5,
                "minimum_successful_recovery": "R3",
                "recovered": True,
            }
        }
    }
    robustness_map = {
        "families": {
            "formal_logic_deduction": {
                "level": 5,
                "target_source": "RECOVERY_R3",
                "status": "ROBUST",
            }
        }
    }
    compound_map = {
        "compounds": {
            "extract_calculate_json": {
                "capabilities_required": [
                    "formal_logic_deduction",
                    "arithmetic_numerical_reasoning",
                    "strict_structured_output",
                ],
                "expected_component_frontier": 4,
                "observed_compound_frontier": 3,
                "composition_penalty": -1,
            }
        }
    }

    value_map = build_capability_value_map(
        "gpt-oss:20b",
        base_rows=base_rows,
        reasoning_rows=reasoning_rows,
        recovery_rows=recovery_rows,
        robustness_rows=robustness_rows,
        compound_rows=compound_rows,
        frontiers=frontiers,
        reasoning_curves=reasoning_curves,
        recovery_map=recovery_map,
        robustness_map=robustness_map,
        compound_map=compound_map,
    )

    assert value_map["schema_version"] == 1
    assert value_map["model"] == "gpt-oss:20b"
    assert value_map["measurement_policy"] == {
        "cost_basis": "MEASURED_EXPERIMENT_RUNTIME_AND_TOKEN_COUNTERS",
        "value_basis": "DEMONSTRATED_CAPABILITY_EFFECTS_ONLY",
        "scalar_value_score": False,
    }

    assert value_map["phases"]["baseline"]["model_calls"] == 2
    assert value_map["phases"]["baseline"]["generated_tokens"] == 210
    assert value_map["phases"]["reasoning"]["model_calls"] == 4
    assert value_map["phases"]["recovery"]["generated_tokens"] == 245
    assert value_map["phases"]["robustness"]["model_calls"] == 1
    assert value_map["phases"]["compound"]["model_calls"] == 1
    assert value_map["total_cost"]["model_calls"] == 10
    assert value_map["total_cost"]["generated_tokens"] == 1123

    family = value_map["families"]["formal_logic_deduction"]
    assert family["baseline_frontier"] == {
        "reliable_floor": 4,
        "first_failure_level": 5,
    }
    assert family["reasoning"]["minimum_reliable_effort_at_baseline_floor"] == "low"
    assert family["reasoning"]["demonstrated_high_effort_extension_to"] == 5
    assert family["reasoning"]["frontier_extension_levels"] == 1
    assert family["recovery"]["minimum_successful_recovery"] == "R3"
    assert family["recovery"]["recovered"] is True
    assert family["robustness"]["status"] == "ROBUST"
    assert family["compound_exposure"] == [
        {
            "compound_id": "extract_calculate_json",
            "expected_component_frontier": 4,
            "observed_compound_frontier": 3,
            "composition_penalty": -1,
        }
    ]
    assert "value_score" not in family


def test_cost_rollup_preserves_unknown_counters_instead_of_fabricating_zero():
    from compute_cost.capability_value import rollup_experiment_cost

    row = experiment_row("partial")
    row["metrics"].pop("eval_count")
    row["timing"].pop("client_latency_ns")

    cost = rollup_experiment_cost([row])

    assert cost["model_calls"] == 1
    assert cost["generated_tokens"] is None
    assert cost["prompt_tokens"] == 20
    assert cost["generation_seconds"] == 2.0
    assert cost["total_runtime_seconds"] == 3.0
    assert cost["client_latency_seconds"] is None
    assert cost["coverage"]["generated_tokens"] == {"observed": 0, "expected": 1}
