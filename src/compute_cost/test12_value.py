"""High-value analytics for Test 1.2 model-to-harness manufacturing.

These functions are intentionally post-hoc over raw Test 1.2 observations:
one model call may contribute evidence to many value channels. Negative and
null controls are retained as first-class manufacturing evidence.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from statistics import mean, median, pstdev
from typing import Any, Iterable


FAMILY_VALUE_DIMENSIONS: tuple[str, ...] = (
    "baseline_frontier",
    "difficulty_spread",
    "seed_stability",
    "failure_rescue",
    "pass_preservation",
    "hard_case_lift",
    "prompt_control",
    "reasoning_mode",
    "generation_budget",
    "context_window",
    "compute_cost_routing",
    "planning",
    "verification",
    "retry_recovery",
    "state_tracking",
    "memory",
    "context_selection_compression",
    "tool_policy",
    "stop_escalate_policy",
    "cost_efficiency",
    "negative_effect_search",
    "activation_boundary",
    "interaction_evidence",
    "contrastive_tuning",
)

# Missing any of these means the family has not produced enough information to
# manufacture a model-specific harness block. Interaction evidence is valuable
# but is not critical for every family because the global interaction scout may
# rationally allocate its finite budget elsewhere.
CRITICAL_FAMILY_VALUE_DIMENSIONS: tuple[str, ...] = tuple(
    value
    for value in FAMILY_VALUE_DIMENSIONS
    if value != "interaction_evidence"
)


def _number(value: Any) -> float:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else 0.0
    )


def _total_tokens(cost: dict[str, Any] | None) -> float:
    cost = cost or {}
    return _number(cost.get("prompt_tokens_observed")) + _number(
        cost.get("output_tokens_observed")
    )


def _wall(cost: dict[str, Any] | None) -> float:
    return _number((cost or {}).get("wall_seconds"))


def _safe_mean(values: Iterable[float]) -> float:
    data = list(values)
    return mean(data) if data else 0.0


def _safe_median(values: Iterable[float]) -> float:
    data = list(values)
    return median(data) if data else 0.0


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _response_entry(
    family: str,
    intervention_id: str,
    rows: list[dict[str, Any]],
    intervention: dict[str, Any] | None,
) -> dict[str, Any]:
    wins = [row for row in rows if _number(row.get("delta")) > 0]
    losses = [row for row in rows if _number(row.get("delta")) < 0]
    nulls = [row for row in rows if _number(row.get("delta")) == 0]
    failure_trials = [row for row in rows if _number(row.get("control_score")) < 1.0]
    pass_trials = [row for row in rows if _number(row.get("control_score")) >= 1.0]
    rescues = [
        row
        for row in failure_trials
        if _number(row.get("score")) > _number(row.get("control_score"))
    ]
    pass_losses = [
        row
        for row in pass_trials
        if _number(row.get("score")) < _number(row.get("control_score"))
    ]
    hard_rows = [
        row for row in rows if int(row.get("difficulty_level") or 0) >= 5
    ]
    hard_wins = [row for row in hard_rows if _number(row.get("delta")) > 0]

    treatment_tokens = [_total_tokens(row.get("cost")) for row in rows]
    control_tokens = [_total_tokens(row.get("control_cost")) for row in rows]
    treatment_latency = [_wall(row.get("cost")) for row in rows]
    control_latency = [_wall(row.get("control_cost")) for row in rows]

    token_ratios = [
        ratio
        for row in rows
        if (
            ratio := _ratio(
                _total_tokens(row.get("cost")),
                _total_tokens(row.get("control_cost")),
            )
        )
        is not None
    ]
    latency_ratios = [
        ratio
        for row in rows
        if (
            ratio := _ratio(
                _wall(row.get("cost")),
                _wall(row.get("control_cost")),
            )
        )
        is not None
    ]
    equivalent_or_better_faster = [
        row
        for row in rows
        if _number(row.get("score")) >= _number(row.get("control_score"))
        and _wall(row.get("control_cost")) > 0
        and _wall(row.get("cost")) < _wall(row.get("control_cost"))
    ]
    equivalent_or_better_cheaper = [
        row
        for row in rows
        if _number(row.get("score")) >= _number(row.get("control_score"))
        and _total_tokens(row.get("control_cost")) > 0
        and _total_tokens(row.get("cost")) < _total_tokens(row.get("control_cost"))
    ]

    fixture_scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        fixture_scores[str(row.get("fixture_id"))].append(_number(row.get("score")))
    repeated_stdev = [
        pstdev(values) for values in fixture_scores.values() if len(values) > 1
    ]

    negative_deltas = [abs(_number(row.get("delta"))) for row in losses]
    intervention = intervention or {}
    first = rows[0] if rows else {}
    return {
        "family_id": family,
        "intervention_id": intervention_id,
        "category": first.get("intervention_category")
        or intervention.get("category"),
        "mode": first.get("intervention_mode") or intervention.get("mode"),
        "label": first.get("intervention_label") or intervention.get("label"),
        "n": len(rows),
        "wins": len(wins),
        "losses": len(losses),
        "nulls": len(nulls),
        "win_rate": len(wins) / len(rows) if rows else 0.0,
        "loss_rate": len(losses) / len(rows) if rows else 0.0,
        "mean_delta": _safe_mean(_number(row.get("delta")) for row in rows),
        "median_delta": _safe_median(_number(row.get("delta")) for row in rows),
        "max_delta": max((_number(row.get("delta")) for row in rows), default=0.0),
        "min_delta": min((_number(row.get("delta")) for row in rows), default=0.0),
        "failure_trials": len(failure_trials),
        "rescues": len(rescues),
        "rescue_rate": len(rescues) / len(failure_trials) if failure_trials else 0.0,
        "pass_trials": len(pass_trials),
        "pass_regressions": len(pass_losses),
        "pass_regression_rate": (
            len(pass_losses) / len(pass_trials) if pass_trials else 0.0
        ),
        "hard_case_trials": len(hard_rows),
        "hard_case_wins": len(hard_wins),
        "difficulty_levels": sorted(
            {int(row.get("difficulty_level") or 0) for row in rows}
        ),
        "seeds": sorted({int(row.get("seed") or 0) for row in rows}),
        "phases": sorted({str(row.get("phase")) for row in rows}),
        "mean_model_calls": _safe_mean(
            _number(row.get("model_calls_per_application")) for row in rows
        ),
        "mean_treatment_tokens": _safe_mean(treatment_tokens),
        "mean_control_tokens": _safe_mean(control_tokens),
        "mean_token_ratio": _safe_mean(token_ratios),
        "mean_treatment_wall_seconds": _safe_mean(treatment_latency),
        "mean_control_wall_seconds": _safe_mean(control_latency),
        "mean_latency_ratio": _safe_mean(latency_ratios),
        "equivalent_or_better_faster": len(equivalent_or_better_faster),
        "equivalent_or_better_cheaper": len(equivalent_or_better_cheaper),
        "repeat_score_stdev": _safe_mean(repeated_stdev),
        "base_score_preservation_value": _safe_mean(negative_deltas),
        "primitive_id": first.get("primitive_id") or intervention.get("primitive_id"),
        "placement": first.get("placement") or intervention.get("placement"),
        "representation": first.get("representation")
        or intervention.get("representation"),
        "dose": first.get("dose")
        if first.get("dose") is not None
        else intervention.get("dose"),
        "recurrence": first.get("recurrence")
        if first.get("recurrence") is not None
        else intervention.get("recurrence"),
        "reasoning_effort": first.get("reasoning_effort"),
        "generation_budget": first.get("generation_budget"),
        "context_request": first.get("context_request"),
        "temperature": first.get("temperature"),
        "parents": copy.deepcopy(first.get("parents") or intervention.get("parents")),
    }


def build_control_response_tensor(
    rows: list[dict[str, Any]],
    interventions: list[dict[str, Any]],
    families: Iterable[str],
) -> dict[str, Any]:
    registry = {
        str(row.get("id")): row for row in interventions if row.get("id")
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    family_set = set(families)
    for row in rows:
        family = str(row.get("family_id") or "")
        intervention_id = str(row.get("intervention_id") or "")
        if family not in family_set or intervention_id in {"", "CONTROL"}:
            continue
        grouped[(family, intervention_id)].append(row)

    entries = [
        _response_entry(
            family,
            intervention_id,
            values,
            registry.get(intervention_id),
        )
        for (family, intervention_id), values in sorted(grouped.items())
    ]
    return {
        "schema_version": 1,
        "family_count": len(family_set),
        "entry_count": len(entries),
        "entries": entries,
    }


def _baseline_frontier(
    rows: list[dict[str, Any]],
    family: str,
) -> dict[str, Any]:
    baseline = [
        row
        for row in rows
        if row.get("family_id") == family
        and row.get("intervention_id") == "CONTROL"
    ]
    by_level: dict[int, list[float]] = defaultdict(list)
    for row in baseline:
        by_level[int(row.get("difficulty_level") or 0)].append(
            _number(row.get("score"))
        )
    levels = {
        level: {
            "n": len(values),
            "pass_rate": _safe_mean(values),
            "score_stdev": pstdev(values) if len(values) > 1 else 0.0,
        }
        for level, values in sorted(by_level.items())
    }
    passing = [
        level for level, item in levels.items() if item["pass_rate"] >= 0.67
    ]
    all_rates = [item["pass_rate"] for item in levels.values()]
    if levels and all(value >= 1.0 for value in all_rates):
        state = "ALL_PASS_IN_MEASURED_RANGE"
    elif levels and all(value <= 0.0 for value in all_rates):
        state = "ALL_FAIL_IN_MEASURED_RANGE"
    elif levels:
        state = "MIXED_FRONTIER"
    else:
        state = "UNMEASURED"
    return {
        "family_id": family,
        "baseline_observations": len(baseline),
        "levels": levels,
        "measured_levels": sorted(levels),
        "reliable_pass_frontier": max(passing) if passing else None,
        "frontier_state": state,
        "unstable_levels": [
            level for level, item in levels.items() if item["score_stdev"] > 0
        ],
        "replicated_levels": [
            level for level, item in levels.items() if int(item["n"]) > 1
        ],
    }


def build_frontier_shift_map(
    rows: list[dict[str, Any]],
    response_tensor: dict[str, Any],
    families: Iterable[str],
) -> dict[str, Any]:
    entries_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in response_tensor.get("entries") or []:
        entries_by_family[str(entry.get("family_id"))].append(entry)

    result: dict[str, Any] = {}
    for family in families:
        baseline = _baseline_frontier(rows, family)
        frontier = baseline.get("reliable_pass_frontier")
        responses = entries_by_family.get(family, [])
        observed_above = []
        for entry in responses:
            if not entry.get("hard_case_wins"):
                continue
            max_level = max(entry.get("difficulty_levels") or [0])
            if frontier is None or max_level > frontier:
                observed_above.append(
                    {
                        "intervention_id": entry["intervention_id"],
                        "category": entry.get("category"),
                        "observed_level": max_level,
                        "frontier_shift_lower_bound": (
                            max_level - frontier if frontier is not None else None
                        ),
                        "hard_case_wins": entry.get("hard_case_wins"),
                    }
                )
        observed_above.sort(
            key=lambda row: (
                row.get("observed_level") or -1,
                row.get("hard_case_wins") or 0,
            ),
            reverse=True,
        )
        result[family] = {
            "baseline": baseline,
            "observed_frontier_extenders": observed_above,
            "best_observed_extender": observed_above[0] if observed_above else None,
        }
    return {
        "schema_version": 1,
        "families": result,
    }


def build_compute_quality_elasticity(
    response_tensor: dict[str, Any],
    families: Iterable[str],
) -> dict[str, Any]:
    compute_categories = {
        "REASONING_MODE",
        "GENERATION_BUDGET",
        "CONTEXT_WINDOW",
        "COMPUTE_COST_ROUTING",
    }
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in response_tensor.get("entries") or []:
        if entry.get("category") in compute_categories:
            by_family[str(entry.get("family_id"))].append(entry)

    output = {}
    for family in families:
        values = by_family.get(family, [])
        quality = sorted(
            values,
            key=lambda row: (
                _number(row.get("mean_delta")),
                -_number(row.get("mean_latency_ratio")),
                -_number(row.get("mean_token_ratio")),
            ),
            reverse=True,
        )
        efficiency = sorted(
            [
                row
                for row in values
                if _number(row.get("mean_delta")) >= 0
                and (
                    _number(row.get("equivalent_or_better_faster")) > 0
                    or _number(row.get("equivalent_or_better_cheaper")) > 0
                )
            ],
            key=lambda row: (
                -_number(row.get("mean_latency_ratio")),
                -_number(row.get("mean_token_ratio")),
                _number(row.get("mean_delta")),
            ),
            reverse=True,
        )
        output[family] = {
            "tested_controls": values,
            "best_quality_control": quality[0] if quality else None,
            "best_efficiency_control": efficiency[0] if efficiency else None,
        }
    return {
        "schema_version": 1,
        "families": output,
    }


def _negative_uses(entry: dict[str, Any]) -> list[str]:
    uses: list[str] = []
    losses = int(entry.get("losses") or 0)
    wins = int(entry.get("wins") or 0)
    nulls = int(entry.get("nulls") or 0)
    if losses:
        uses.extend(
            [
                "REGRESSION_SENTINEL",
                "CONTRASTIVE_TUNING_NEGATIVE",
                "COMPENSATION_TARGET",
            ]
        )
        if wins:
            uses.extend(["CONDITIONAL_GATE", "ACTIVATION_BOUNDARY_SENSOR"])
        else:
            uses.append("ROUTE_VETO")
    if nulls and not wins and not losses:
        uses.extend(["PRUNE_UNNECESSARY_CONTROL", "COST_AVOIDANCE_RULE"])
    if _number(entry.get("mean_latency_ratio")) > 1.0 and _number(
        entry.get("mean_delta")
    ) <= 0:
        uses.append("LATENCY_VETO")
    if _number(entry.get("mean_token_ratio")) > 1.0 and _number(
        entry.get("mean_delta")
    ) <= 0:
        uses.append("TOKEN_OVERHEAD_VETO")
    return sorted(set(uses))


def build_negative_effect_exploitation(
    response_tensor: dict[str, Any],
    families: Iterable[str],
) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    responses_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in response_tensor.get("entries") or []:
        family = str(entry.get("family_id"))
        responses_by_family[family].append(entry)
        uses = _negative_uses(entry)
        if not uses:
            continue
        enriched = {
            "intervention_id": entry.get("intervention_id"),
            "category": entry.get("category"),
            "uses": uses,
            "wins": entry.get("wins"),
            "losses": entry.get("losses"),
            "nulls": entry.get("nulls"),
            "mean_delta": entry.get("mean_delta"),
            "base_score_preservation_value": entry.get(
                "base_score_preservation_value"
            ),
            "mean_latency_ratio": entry.get("mean_latency_ratio"),
            "mean_token_ratio": entry.get("mean_token_ratio"),
            "difficulty_levels": entry.get("difficulty_levels"),
            "activation_rule": (
                "APPLY_ONLY_WHERE_POSITIVE_BOUNDARY_IS_VALIDATED"
                if "CONDITIONAL_GATE" in uses
                else "DO_NOT_APPLY_TO_THIS_FAMILY_STATE"
                if "ROUTE_VETO" in uses
                else "PRUNE_BY_DEFAULT"
            ),
        }
        by_family[family].append(enriched)

    # Turn each negative into an executable replacement choice. Prefer another
    # control from the same surface with lower observed loss and higher value;
    # otherwise preserve the raw model path.
    for family, assets in by_family.items():
        candidates = responses_by_family.get(family, [])
        for asset in assets:
            same_surface = [
                row
                for row in candidates
                if row.get("category") == asset.get("category")
                and row.get("intervention_id") != asset.get("intervention_id")
                and int(row.get("losses") or 0) == 0
            ]
            same_surface.sort(
                key=lambda row: (
                    _number(row.get("mean_delta")),
                    _number(row.get("rescue_rate")),
                    -_number(row.get("mean_latency_ratio")),
                ),
                reverse=True,
            )
            replacement = same_surface[0] if same_surface else None
            asset["safe_replacement"] = (
                {
                    "intervention_id": replacement.get("intervention_id"),
                    "mean_delta": replacement.get("mean_delta"),
                    "rescue_rate": replacement.get("rescue_rate"),
                    "loss_rate": replacement.get("loss_rate"),
                }
                if replacement
                else {
                    "intervention_id": "DIRECT",
                    "mean_delta": 0.0,
                    "loss_rate": 0.0,
                }
            )

    return {
        "schema_version": 1,
        "principle": (
            "negative evidence improves the compiled harness by preventing "
            "known regressions, defining activation boundaries, supplying "
            "contrastive tuning examples, pruning wasted compute, and selecting "
            "a safer replacement control or DIRECT fallback"
        ),
        "families": {
            family: {
                "negative_assets": by_family.get(family, []),
                "estimated_preservation_value": _safe_mean(
                    _number(row.get("base_score_preservation_value"))
                    for row in by_family.get(family, [])
                    if "ROUTE_VETO" in (row.get("uses") or [])
                    or "CONDITIONAL_GATE" in (row.get("uses") or [])
                ),
            }
            for family in families
        },
    }



def _factor_response(
    family_rows: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in family_rows:
        value = row.get(key)
        if value is None:
            continue
        grouped[str(value)].append(_number(row.get("delta")))
    return {
        value: {
            "n": len(deltas),
            "mean_delta": _safe_mean(deltas),
            "median_delta": _safe_median(deltas),
            "wins": sum(1 for delta in deltas if delta > 0),
            "losses": sum(1 for delta in deltas if delta < 0),
        }
        for value, deltas in sorted(grouped.items())
    }


def _dimension_state(
    condition: bool,
    *,
    not_applicable: bool = False,
    evidence_count: int = 0,
) -> dict[str, Any]:
    return {
        "status": "NOT_APPLICABLE"
        if not_applicable
        else "MEASURED"
        if condition
        else "MISSING",
        "evidence_count": evidence_count,
    }


def build_family_value_dossiers(
    rows: list[dict[str, Any]],
    interventions: list[dict[str, Any]],
    families: Iterable[str],
    required_surfaces: Iterable[str],
) -> dict[str, Any]:
    families = list(families)
    required_surfaces = list(required_surfaces)
    response_tensor = build_control_response_tensor(rows, interventions, families)
    frontier_map = build_frontier_shift_map(rows, response_tensor, families)
    negative_map = build_negative_effect_exploitation(response_tensor, families)

    entries_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in response_tensor.get("entries") or []:
        entries_by_family[str(entry.get("family_id"))].append(entry)

    dossiers: dict[str, Any] = {}
    for family in families:
        family_rows = [row for row in rows if row.get("family_id") == family]
        treatment_rows = [
            row for row in family_rows if row.get("intervention_id") != "CONTROL"
        ]
        responses = entries_by_family.get(family, [])
        baseline = (frontier_map.get("families") or {}).get(family, {}).get(
            "baseline", {}
        )
        state = baseline.get("frontier_state")
        categories = {
            str(row.get("intervention_category"))
            for row in treatment_rows
            if row.get("intervention_category")
        }

        positives = sorted(
            [row for row in responses if int(row.get("wins") or 0) > 0],
            key=lambda row: (
                _number(row.get("mean_delta")),
                _number(row.get("hard_case_wins")),
                -_number(row.get("pass_regression_rate")),
            ),
            reverse=True,
        )
        hard = sorted(
            [row for row in responses if int(row.get("hard_case_wins") or 0) > 0],
            key=lambda row: (
                int(row.get("hard_case_wins") or 0),
                _number(row.get("mean_delta")),
            ),
            reverse=True,
        )
        reliable = sorted(
            [
                row
                for row in responses
                if int(row.get("rescues") or 0) > 0
                and _number(row.get("pass_regression_rate")) == 0.0
            ],
            key=lambda row: (
                _number(row.get("rescue_rate")),
                -_number(row.get("repeat_score_stdev")),
            ),
            reverse=True,
        )
        speed = sorted(
            [
                row
                for row in responses
                if int(row.get("equivalent_or_better_faster") or 0) > 0
            ],
            key=lambda row: (
                -_number(row.get("mean_latency_ratio")),
                _number(row.get("mean_delta")),
            ),
            reverse=True,
        )
        tokens = sorted(
            [
                row
                for row in responses
                if int(row.get("equivalent_or_better_cheaper") or 0) > 0
            ],
            key=lambda row: (
                -_number(row.get("mean_token_ratio")),
                _number(row.get("mean_delta")),
            ),
            reverse=True,
        )

        failure_probe_count = sum(
            1 for row in treatment_rows if _number(row.get("control_score")) < 1.0
        )
        pass_probe_count = sum(
            1 for row in treatment_rows if _number(row.get("control_score")) >= 1.0
        )
        hard_probe_count = sum(
            1 for row in treatment_rows if int(row.get("difficulty_level") or 0) >= 5
        )
        has_control_cost = sum(
            1
            for row in treatment_rows
            if _wall(row.get("control_cost")) > 0
            or _total_tokens(row.get("control_cost")) > 0
        )

        dimensions: dict[str, Any] = {
            "baseline_frontier": _dimension_state(
                len(baseline.get("measured_levels") or []) >= 3,
                evidence_count=int(baseline.get("baseline_observations") or 0),
            ),
            "difficulty_spread": _dimension_state(
                len(baseline.get("measured_levels") or []) >= 3,
                evidence_count=len(baseline.get("measured_levels") or []),
            ),
            "seed_stability": _dimension_state(
                any(
                    len(entry.get("seeds") or []) > 1
                    for entry in responses
                )
                or bool(baseline.get("replicated_levels")),
                evidence_count=(
                    sum(len(entry.get("seeds") or []) for entry in responses)
                    + len(baseline.get("replicated_levels") or [])
                ),
            ),
            "failure_rescue": _dimension_state(
                failure_probe_count > 0,
                not_applicable=state == "ALL_PASS_IN_MEASURED_RANGE",
                evidence_count=failure_probe_count,
            ),
            "pass_preservation": _dimension_state(
                pass_probe_count > 0,
                not_applicable=state == "ALL_FAIL_IN_MEASURED_RANGE",
                evidence_count=pass_probe_count,
            ),
            "hard_case_lift": _dimension_state(
                hard_probe_count > 0,
                evidence_count=hard_probe_count,
            ),
            "prompt_control": _dimension_state(
                "PROMPT_CONTROL" in categories,
                evidence_count=sum(
                    1
                    for row in treatment_rows
                    if row.get("intervention_category") == "PROMPT_CONTROL"
                ),
            ),
            "reasoning_mode": _dimension_state(
                "REASONING_MODE" in categories,
                evidence_count=sum(
                    1
                    for row in treatment_rows
                    if row.get("intervention_category") == "REASONING_MODE"
                ),
            ),
            "generation_budget": _dimension_state(
                "GENERATION_BUDGET" in categories,
                evidence_count=sum(
                    1
                    for row in treatment_rows
                    if row.get("intervention_category") == "GENERATION_BUDGET"
                ),
            ),
            "context_window": _dimension_state(
                "CONTEXT_WINDOW" in categories,
                evidence_count=sum(
                    1
                    for row in treatment_rows
                    if row.get("intervention_category") == "CONTEXT_WINDOW"
                ),
            ),
            "compute_cost_routing": _dimension_state(
                "COMPUTE_COST_ROUTING" in categories,
                evidence_count=sum(
                    1
                    for row in treatment_rows
                    if row.get("intervention_category") == "COMPUTE_COST_ROUTING"
                ),
            ),
            "planning": _dimension_state(
                "PLANNING" in categories,
                evidence_count=sum(
                    1
                    for row in treatment_rows
                    if row.get("intervention_category") == "PLANNING"
                ),
            ),
            "verification": _dimension_state(
                "VERIFICATION" in categories,
                evidence_count=sum(
                    1
                    for row in treatment_rows
                    if row.get("intervention_category") == "VERIFICATION"
                ),
            ),
            "retry_recovery": _dimension_state(
                "RETRY_RECOVERY" in categories,
                evidence_count=sum(
                    1
                    for row in treatment_rows
                    if row.get("intervention_category") == "RETRY_RECOVERY"
                ),
            ),
            "state_tracking": _dimension_state(
                "STATE_TRACKING" in categories,
                evidence_count=sum(
                    1
                    for row in treatment_rows
                    if row.get("intervention_category") == "STATE_TRACKING"
                ),
            ),
            "memory": _dimension_state(
                "MEMORY" in categories,
                evidence_count=sum(
                    1
                    for row in treatment_rows
                    if row.get("intervention_category") == "MEMORY"
                ),
            ),
            "context_selection_compression": _dimension_state(
                "CONTEXT_SELECTION_COMPRESSION" in categories,
                evidence_count=sum(
                    1
                    for row in treatment_rows
                    if row.get("intervention_category")
                    == "CONTEXT_SELECTION_COMPRESSION"
                ),
            ),
            "tool_policy": _dimension_state(
                "TOOL_POLICY" in categories,
                evidence_count=sum(
                    1
                    for row in treatment_rows
                    if row.get("intervention_category") == "TOOL_POLICY"
                ),
            ),
            "stop_escalate_policy": _dimension_state(
                "STOP_ESCALATE_POLICY" in categories,
                evidence_count=sum(
                    1
                    for row in treatment_rows
                    if row.get("intervention_category") == "STOP_ESCALATE_POLICY"
                ),
            ),
            "cost_efficiency": _dimension_state(
                has_control_cost > 0,
                evidence_count=has_control_cost,
            ),
            "negative_effect_search": _dimension_state(
                pass_probe_count > 0,
                not_applicable=state == "ALL_FAIL_IN_MEASURED_RANGE",
                evidence_count=pass_probe_count,
            ),
            "activation_boundary": _dimension_state(
                len(
                    {
                        int(row.get("difficulty_level") or 0)
                        for row in treatment_rows
                    }
                )
                >= 2,
                evidence_count=len(treatment_rows),
            ),
            "interaction_evidence": _dimension_state(
                "COMPOSITION_LAYERING" in categories,
                evidence_count=sum(
                    1
                    for row in treatment_rows
                    if row.get("intervention_category") == "COMPOSITION_LAYERING"
                ),
            ),
            "contrastive_tuning": _dimension_state(
                bool(treatment_rows),
                evidence_count=len(treatment_rows),
            ),
        }

        missing_critical = [
            name
            for name in CRITICAL_FAMILY_VALUE_DIMENSIONS
            if (dimensions.get(name) or {}).get("status") == "MISSING"
        ]

        advancement_paths = []
        if positives:
            advancement_paths.append("HARNESS_CAPABILITY_LIFT")
        if hard:
            advancement_paths.append("HARD_CASE_FRONTIER_LIFT")
        if reliable:
            advancement_paths.append("RELIABILITY_LIFT")
        if speed:
            advancement_paths.append("LATENCY_SPEEDUP")
        if tokens:
            advancement_paths.append("TOKEN_COST_REDUCTION")
        negative_assets = (
            (negative_map.get("families") or {})
            .get(family, {})
            .get("negative_assets", [])
        )
        if negative_assets:
            advancement_paths.append("NEGATIVE_GATING_SCORE_PROTECTION")
        if not positives and state != "ALL_PASS_IN_MEASURED_RANGE":
            advancement_paths.append("RESIDUAL_FINE_TUNING_TARGET")
        if state == "ALL_PASS_IN_MEASURED_RANGE":
            advancement_paths.append("FRONTIER_EXTENSION_OR_EFFICIENCY_TARGET")

        dossiers[family] = {
            "family_id": family,
            "baseline": baseline,
            "critical_value_ready": not missing_critical,
            "missing_critical_value_dimensions": missing_critical,
            "value_dimensions": dimensions,
            "advancement_paths": sorted(set(advancement_paths)),
            "advancement_ready": bool(advancement_paths),
            "best_capability_amplifiers": positives[:12],
            "best_hard_case_amplifiers": hard[:12],
            "best_reliability_controls": reliable[:12],
            "best_latency_paths": speed[:12],
            "best_token_efficiency_paths": tokens[:12],
            "negative_effect_assets": negative_assets,
            "factor_response": {
                "placement": _factor_response(treatment_rows, "placement"),
                "representation": _factor_response(
                    treatment_rows, "representation"
                ),
                "dose": _factor_response(treatment_rows, "dose"),
                "recurrence": _factor_response(treatment_rows, "recurrence"),
                "reasoning_effort": _factor_response(
                    treatment_rows, "reasoning_effort"
                ),
                "generation_budget": _factor_response(
                    treatment_rows, "generation_budget"
                ),
                "context_request": _factor_response(
                    treatment_rows, "context_request"
                ),
                "temperature": _factor_response(
                    treatment_rows, "temperature"
                ),
            },
        }

    return {
        "schema_version": 1,
        "value_dimensions": list(FAMILY_VALUE_DIMENSIONS),
        "critical_value_dimensions": list(CRITICAL_FAMILY_VALUE_DIMENSIONS),
        "families": dossiers,
    }


def build_value_completeness(dossiers: dict[str, Any]) -> dict[str, Any]:
    families = dossiers.get("families") or {}
    incomplete = [
        family
        for family, row in families.items()
        if not row.get("critical_value_ready")
        or not row.get("advancement_ready")
    ]
    return {
        "schema_version": 1,
        "family_count": len(families),
        "critical_value_dimension_count": len(CRITICAL_FAMILY_VALUE_DIMENSIONS),
        "all_families_critical_value_ready": not incomplete,
        "incomplete_families": incomplete,
        "families": {
            family: {
                "critical_value_ready": row.get("critical_value_ready"),
                "advancement_ready": row.get("advancement_ready"),
                "missing_critical_value_dimensions": row.get(
                    "missing_critical_value_dimensions"
                ),
                "advancement_paths": row.get("advancement_paths"),
            }
            for family, row in families.items()
        },
    }


def contrastive_negative_corpus(
    rows: list[dict[str, Any]],
    negative_map: dict[str, Any],
) -> list[dict[str, Any]]:
    uses_by_key: dict[tuple[str, str], list[str]] = {}
    for family, payload in (negative_map.get("families") or {}).items():
        for asset in payload.get("negative_assets") or []:
            uses_by_key[(family, str(asset.get("intervention_id")))] = list(
                asset.get("uses") or []
            )

    result = []
    for row in rows:
        if row.get("intervention_id") in {None, "CONTROL"}:
            continue
        delta = _number(row.get("delta"))
        if delta >= 0:
            continue
        family = str(row.get("family_id"))
        intervention_id = str(row.get("intervention_id"))
        result.append(
            {
                "schema_version": 1,
                "family_id": family,
                "fixture_id": row.get("fixture_id"),
                "difficulty_level": row.get("difficulty_level"),
                "intervention_id": intervention_id,
                "intervention_category": row.get("intervention_category"),
                "task_text": row.get("task_text"),
                "preferred_response_text": row.get("control_response_text"),
                "rejected_response_text": row.get("treatment_response_text"),
                "score_delta": delta,
                "negative_uses": uses_by_key.get(
                    (family, intervention_id),
                    ["CONTRASTIVE_TUNING_NEGATIVE", "REGRESSION_SENTINEL"],
                ),
                "cost": copy.deepcopy(row.get("cost") or {}),
                "control_cost": copy.deepcopy(row.get("control_cost") or {}),
            }
        )
    return result


def observation_value_index(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    index = []
    for row in rows:
        channels = {"RAW_EVIDENCE", "COST_ACCOUNTING"}
        intervention_id = str(row.get("intervention_id") or "")
        if intervention_id == "CONTROL":
            channels.update(
                {
                    "BASELINE_CAPABILITY",
                    "FRONTIER_MAPPING",
                    "STABILITY_EVIDENCE",
                }
            )
        else:
            channels.update(
                {
                    "CONTROL_RESPONSE",
                    "FAMILY_MANUFACTURING",
                    "ROUTING_EVIDENCE",
                    "TUNING_EXAMPLE",
                }
            )
            delta = _number(row.get("delta"))
            if _number(row.get("control_score")) < 1.0:
                channels.add("FAILURE_RESCUE_SEARCH")
            else:
                channels.add("PASS_PRESERVATION")
            if delta > 0:
                channels.update(
                    {
                        "POSITIVE_CONTROL",
                        "CAPABILITY_LIFT",
                    }
                )
            elif delta < 0:
                channels.update(
                    {
                        "NEGATIVE_TRANSFER",
                        "ROUTE_VETO_CANDIDATE",
                        "REGRESSION_SENTINEL",
                        "CONTRASTIVE_TUNING_NEGATIVE",
                    }
                )
            else:
                channels.update(
                    {
                        "NULL_CONTROL",
                        "PRUNING_CANDIDATE",
                    }
                )
            if int(row.get("difficulty_level") or 0) >= 5:
                channels.add("HARD_CASE_EVIDENCE")
            if row.get("primitive_id") is not None:
                channels.add("PROMPT_FACTOR_RESPONSE")
            if row.get("parents"):
                channels.add("INTERACTION_EVIDENCE")
            if (
                _wall(row.get("control_cost")) > 0
                or _total_tokens(row.get("control_cost")) > 0
            ):
                channels.add("EFFICIENCY_COMPARISON")
        index.append(
            {
                "schema_version": 1,
                "experiment_id": row.get("experiment_id"),
                "fixture_id": row.get("fixture_id"),
                "family_id": row.get("family_id"),
                "intervention_id": row.get("intervention_id"),
                "value_channels": sorted(channels),
            }
        )
    return index
