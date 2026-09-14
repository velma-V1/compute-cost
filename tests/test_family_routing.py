from compute_cost.family_routing import (
    build_family_classifier,
    classify_family,
    evaluate_family_classifier,
)


def _case(case_id, family, level, prompt):
    return {
        "id": case_id,
        "category": family,
        "family_id": family,
        "difficulty_level": level,
        "prompt": prompt,
        "capability_map": {"family_id": family},
    }


def test_family_classifier_is_zero_call_and_holds_out_prototypes():
    cases = [
        _case("a0", "temporal_reasoning", 0, "Monday then Tuesday time sequence"),
        _case("a1", "temporal_reasoning", 1, "What weekday follows Monday after 30 hours?"),
        _case("b0", "spatial_reasoning", 0, "Move east then north on a coordinate grid"),
        _case("b1", "spatial_reasoning", 1, "Move west on the grid and return coordinates"),
        _case("c0", "tool_selection", 0, "Choose calculator tool from available tools"),
        _case("c1", "tool_selection", 1, "Select the correct tool for multiplication"),
    ]
    classifier = build_family_classifier(cases)
    report = evaluate_family_classifier(cases, classifier)

    assert classifier["model_calls_required"] == 0
    assert report["model_calls_added"] == 0
    assert report["prototype_fixture_count"] == 3
    assert report["evaluated_fixture_count"] == 3
    assert report["generalization_claim"] is False
    assert set(report["families"]) == {
        "temporal_reasoning",
        "spatial_reasoning",
        "tool_selection",
    }


def test_runtime_classification_returns_margin_not_fake_probability():
    cases = [
        _case("time0", "temporal_reasoning", 0, "weekday time hours after Monday"),
        _case("space0", "spatial_reasoning", 0, "coordinate grid move east north"),
    ]
    classifier = build_family_classifier(cases)
    result = classify_family("move east on a coordinate grid", classifier)

    assert result["family_id"] == "spatial_reasoning"
    assert result["classification_margin"] >= 0.0
    assert result["confidence_semantics"] == "COSINE_MARGIN_NOT_CALIBRATED_PROBABILITY"
