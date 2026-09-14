"""Print the zero-model-call Test 1.2 preflight audit for CI and operators."""

from __future__ import annotations

import json
from pathlib import Path

from .capability_suite import normalize_capability_suite, validate_capability_suite
from .family_routing import build_family_classifier, evaluate_family_classifier
from .gpt_oss_calibration import materialize_gpt_oss_suite
from .test12_campaign import fresh_model_source
from .test12_preflight import build_test12_cell_budget_plan


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    suite_path = root / "benchmarks" / "gpt-oss-20b-capability-v1.json"
    taxonomy_path = root / "benchmarks" / "capability-taxonomy-v1.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    validate_capability_suite(suite, taxonomy)
    suite = materialize_gpt_oss_suite(suite, taxonomy)
    validate_capability_suite(suite, taxonomy)
    suite = normalize_capability_suite(suite)
    cases = list(suite.get("cases") or [])

    classifier = build_family_classifier(cases)
    confusion = evaluate_family_classifier(cases, classifier)
    plan = build_test12_cell_budget_plan(
        fresh_model_source(cases),
        confusion,
    )
    summary = {
        "family_classifier_top1_accuracy": confusion.get("top1_accuracy"),
        "classifier_evaluated_fixtures": confusion.get("evaluated_fixture_count"),
        "top_confusable_pairs": (confusion.get("confusable_family_pairs") or [])[:10],
        "semantic_mechanism_count": plan.get("semantic_mechanism_count"),
        "built_semantic_mechanism_count": plan.get("built_semantic_mechanism_count"),
        "unbuilt_semantic_mechanism_count": plan.get("unbuilt_semantic_mechanism_count"),
        "untested_applicability_cell_count": plan.get("untested_applicability_cell_count"),
        "potential_cell_count": plan.get("potential_cell_count"),
        "applicability_counts": plan.get("applicability_counts"),
        "eligible_cell_count": plan.get("eligible_cell_count"),
        "estimated_matrix_physical_call_capacity": plan.get(
            "estimated_matrix_physical_call_capacity"
        ),
        "capacity_scenarios": {
            key: {
                "selected_cell_count": value.get("selected_cell_count"),
                "selected_cell_fraction": value.get("selected_cell_fraction"),
                "estimated_physical_calls_used": value.get(
                    "estimated_physical_calls_used"
                ),
            }
            for key, value in (plan.get("capacity_scenarios") or {}).items()
        },
        "gating_decision": plan.get("gating_decision"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
