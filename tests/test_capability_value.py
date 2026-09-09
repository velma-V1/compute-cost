def experiment_row(
    experiment_id,
    *,
    eval_count=100,
    prompt_eval_count=20,
    eval_duration_ns=2_000_000_000,
    total_duration_ns=3_000_000_000,
    client_latency_ns=3_500_000_000,
):
    return {
        "experiment": {"experiment_id": experiment_id},
        "metrics": {
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "eval_duration_ns": eval_duration_ns,
            "total_duration_ns": total_duration_ns,
        },
        "timing": {"client_latency_ns": client_latency_ns},
    }


def test_cost_rollup_preserves_unknown_counters_instead_of_fabricating_zero():
    from compute_cost.cost_value import rollup_experiment_cost

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


def test_cost_rollup_sums_only_when_every_counter_is_observed():
    from compute_cost.cost_value import rollup_experiment_cost

    cost = rollup_experiment_cost(
        [
            experiment_row("a", eval_count=40, prompt_eval_count=10),
            experiment_row("b", eval_count=60, prompt_eval_count=15),
        ]
    )

    assert cost["model_calls"] == 2
    assert cost["generated_tokens"] == 100
    assert cost["prompt_tokens"] == 25
    assert cost["generation_seconds"] == 4.0
    assert cost["total_runtime_seconds"] == 6.0
    assert cost["client_latency_seconds"] == 7.0
    assert cost["coverage"]["generated_tokens"] == {"observed": 2, "expected": 2}
