from compute_cost.comparability_report import build_comparability_report


def _row(*, exp, family, level, role, semantic, fmt=100.0, scope="fixed_core", parent=None, budget=256):
    return {
        "experiment": {
            "experiment_id": exp,
            "parent_experiment_id": parent,
            "task_id": f"{family}-L{level}",
            "task_family": family,
            "difficulty_level": level,
            "generation_budget": budget,
            "reasoning_effort": role.lower() if role.startswith("GPT_") else None,
            "thinking_mode": role == "ENHANCED",
        },
        "comparison": {
            "scope": scope,
            "reasoning_role": role,
            "family_id": family,
            "difficulty_level": level,
            "task_id": f"{family}-L{level}",
            "is_retry": parent is not None,
        },
        "classification": {"result_class": "ANSWER_CORRECT" if semantic == 100.0 else "ANSWER_WRONG", "valid_for_capability": True},
        "score_vector": {
            "semantic_correctness": semantic,
            "contract_format_compliance": fmt,
            "decision_quality": "NOT_APPLICABLE",
            "runtime_validity": 100.0,
            "latency_seconds": 1.0,
            "generated_tokens": 10,
        },
    }


def test_official_aggregates_use_fixed_core_only_and_macro_weight_families():
    rows = [
        _row(exp="a", family="f1", level=2, role="BASELINE_MINIMAL", semantic=100.0),
        _row(exp="b", family="f1", level=5, role="BASELINE_MINIMAL", semantic=0.0),
        _row(exp="c", family="f2", level=2, role="BASELINE_MINIMAL", semantic=100.0),
        _row(exp="diag", family="f2", level=8, role="BASELINE_MINIMAL", semantic=0.0, scope="diagnostic"),
    ]
    report = build_comparability_report("model", rows)
    baseline = report["reasoning_comparison"]["conditions"]["BASELINE_MINIMAL"]
    # f1 mean=50, f2 mean=100; official macro score must be equal-family 75.
    assert baseline["macro_semantic_score"] == 75.0
    assert baseline["official_attempts"] == 3
    assert report["model_summary"]["official_scope"] == "fixed_core_only"
    assert report["model_summary"]["diagnostic_rows_excluded"] == 1


def test_task_scorecard_preserves_component_vector_not_single_pass_fail():
    rows = [_row(exp="a", family="f", level=2, role="BASELINE_MINIMAL", semantic=100.0, fmt=0.0)]
    report = build_comparability_report("model", rows)
    task = report["task_scorecard"][0]
    assert task["score_vector"]["semantic_correctness"] == 100.0
    assert task["score_vector"]["contract_format_compliance"] == 0.0


def test_retry_delta_compares_parent_child_without_reclassifying_official_cell():
    rows = [
        _row(exp="parent", family="f", level=8, role="BASELINE_MINIMAL", semantic=0.0, budget=256),
        _row(exp="child", family="f", level=8, role="BASELINE_MINIMAL", semantic=100.0, parent="parent", budget=512),
    ]
    report = build_comparability_report("model", rows)
    delta = report["retry_deltas"][0]
    assert delta["parent_experiment_id"] == "parent"
    assert delta["child_experiment_id"] == "child"
    assert delta["generation_budget_delta"] == 256
    assert delta["score_delta"]["semantic_correctness"] == 100.0
    assert report["model_summary"]["official_attempts"] == 1


def test_missing_or_unsupported_comparison_cells_are_explicit_and_excluded():
    report = build_comparability_report(
        "model",
        [],
        comparison_cells=[
            {"family_id": "f", "difficulty_level": 10, "reasoning_role": "MAX_NATIVE", "status": "UNSUPPORTED_REASONING_CONDITION"},
            {"family_id": "g", "difficulty_level": 8, "reasoning_role": "BASELINE_MINIMAL", "status": "MISSING_FIXTURE_COVERAGE"},
        ],
    )
    assert report["comparison_cells"]["unsupported"] == 1
    assert report["comparison_cells"]["missing_fixture"] == 1
    assert report["model_summary"]["official_attempts"] == 0
