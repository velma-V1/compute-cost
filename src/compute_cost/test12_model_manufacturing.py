"""Zero-clock model-building refinery for Test 1.2.

Every artifact in this module is derived deterministically from model calls the
campaign already paid for. No model/provider/runtime call is permitted here.

The goal is to convert harness experiments into training assets that can improve
the model weights themselves without extending the Test 1.2 inference clock.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Iterable


ZERO_CLOCK_MODEL_BUILDING_PRODUCTS: tuple[str, ...] = (
    "HARNESS_TO_WEIGHT_DISTILLATION",
    "WEIGHTED_HARD_NEGATIVE_PREFERENCES",
    "CAPABILITY_CURRICULUM",
    "ROUTER_ACTIVATION_SUPERVISION",
    "STABILITY_ANCHORS",
    "CROSS_FAMILY_TRANSFER_GRAPH",
    "PARETO_EFFICIENCY_TARGETS",
)


def _num(value: Any) -> float:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else 0.0
    )


def _tokens(cost: dict[str, Any] | None) -> float:
    cost = cost or {}
    return _num(cost.get("prompt_tokens_observed")) + _num(
        cost.get("output_tokens_observed")
    )


def _latency(cost: dict[str, Any] | None) -> float:
    return _num((cost or {}).get("wall_seconds"))


def _calls(row: dict[str, Any]) -> float:
    return _num(row.get("model_calls_per_application"))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _stable_id(*parts: Any) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _eligible_training_row(row: dict[str, Any]) -> bool:
    return (
        row.get("partition") == "DISCOVERY"
        and row.get("intervention_id") not in {None, "CONTROL"}
        and bool(_text(row.get("task_text")))
        and bool(_text(row.get("control_response_text")))
        and bool(_text(row.get("treatment_response_text")))
    )


def build_harness_to_weight_distillation(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Distill successful harness rescues back into the raw task distribution.

    The training input is the original task, *not* the intervention prompt.
    This intentionally turns external controller value into an internal weight
    target while preserving provenance so later training can filter by control.
    """
    result = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not _eligible_training_row(row):
            continue
        control_score = _num(row.get("control_score"))
        treatment_score = _num(row.get("score"))
        if treatment_score <= control_score:
            continue
        task = _text(row.get("task_text"))
        target = _text(row.get("treatment_response_text"))
        key = (task, target)
        if key in seen:
            continue
        seen.add(key)
        difficulty = int(row.get("difficulty_level") or 0)
        delta = treatment_score - control_score
        result.append(
            {
                "schema_version": 1,
                "record_id": _stable_id("distill", row.get("fixture_id"), row.get("intervention_id")),
                "training_type": "SFT_HARNESS_TO_WEIGHT_DISTILLATION",
                "family_id": row.get("family_id"),
                "fixture_id": row.get("fixture_id"),
                "difficulty_level": difficulty,
                "input": task,
                "target": target,
                "source_intervention_id": row.get("intervention_id"),
                "source_intervention_category": row.get("intervention_category"),
                "source_control_score": control_score,
                "source_treatment_score": treatment_score,
                "source_delta": delta,
                "sample_weight": 1.0 + delta + (0.25 if difficulty >= 5 else 0.0),
                "principle": "teach the raw model the successful answer without requiring the external harness control",
            }
        )
    return result


def build_weighted_preference_pairs(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create DPO/ORPO/KTO-ready same-task preference data.

    Score decides first. Equal-quality outputs use observable compute cost as a
    secondary preference so the model can learn shorter/faster behavior without
    sacrificing correctness.
    """
    result = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if not _eligible_training_row(row):
            continue
        control = _text(row.get("control_response_text"))
        treatment = _text(row.get("treatment_response_text"))
        if control == treatment:
            continue

        c_score = _num(row.get("control_score"))
        t_score = _num(row.get("score"))
        c_cost = row.get("control_cost") or {}
        t_cost = row.get("cost") or {}

        reason = None
        chosen = None
        rejected = None
        quality_gap = abs(t_score - c_score)
        cost_gap = 0.0

        if t_score > c_score:
            chosen, rejected = treatment, control
            reason = "QUALITY_WIN"
        elif c_score > t_score:
            chosen, rejected = control, treatment
            reason = "QUALITY_REGRESSION_NEGATIVE"
        else:
            c_scalar = _calls({"model_calls_per_application": 1}) + 0.001 * _tokens(c_cost) + 0.01 * _latency(c_cost)
            t_scalar = _calls(row) + 0.001 * _tokens(t_cost) + 0.01 * _latency(t_cost)
            if abs(c_scalar - t_scalar) < 1e-9:
                continue
            if t_scalar < c_scalar:
                chosen, rejected = treatment, control
                cost_gap = c_scalar - t_scalar
            else:
                chosen, rejected = control, treatment
                cost_gap = t_scalar - c_scalar
            reason = "EQUAL_QUALITY_LOWER_COMPUTE"

        task = _text(row.get("task_text"))
        key = (task, chosen, rejected)
        if key in seen:
            continue
        seen.add(key)
        difficulty = int(row.get("difficulty_level") or 0)
        weight = (
            1.0
            + 2.0 * quality_gap
            + min(1.0, cost_gap)
            + (0.5 if difficulty >= 5 else 0.0)
        )
        result.append(
            {
                "schema_version": 1,
                "record_id": _stable_id("pref", row.get("fixture_id"), row.get("intervention_id"), reason),
                "training_type": "WEIGHTED_SAME_TASK_PREFERENCE",
                "family_id": row.get("family_id"),
                "fixture_id": row.get("fixture_id"),
                "difficulty_level": difficulty,
                "prompt": task,
                "chosen": chosen,
                "rejected": rejected,
                "preference_reason": reason,
                "sample_weight": weight,
                "quality_gap": quality_gap,
                "cost_gap": cost_gap,
                "same_task_hard_negative": True,
                "source_intervention_id": row.get("intervention_id"),
                "source_intervention_category": row.get("intervention_category"),
            }
        )
    return result


def build_capability_curriculum(
    rows: list[dict[str, Any]],
    families: Iterable[str],
) -> dict[str, Any]:
    """Build a family/difficulty curriculum from measured weakness and headroom."""
    families = list(families)
    payload: dict[str, Any] = {}
    raw_priorities: dict[str, float] = {}

    for family in families:
        baseline = [
            row
            for row in rows
            if row.get("partition") == "DISCOVERY"
            and row.get("family_id") == family
            and row.get("intervention_id") == "CONTROL"
        ]
        treatments = [
            row
            for row in rows
            if row.get("partition") == "DISCOVERY"
            and row.get("family_id") == family
            and row.get("intervention_id") not in {None, "CONTROL"}
        ]
        pass_rate = mean([_num(row.get("score")) for row in baseline]) if baseline else 0.0
        rescue_rate = (
            sum(
                1
                for row in treatments
                if _num(row.get("control_score")) < 1.0
                and _num(row.get("score")) > _num(row.get("control_score"))
            )
            / max(
                1,
                sum(1 for row in treatments if _num(row.get("control_score")) < 1.0),
            )
        )
        regression_rate = (
            sum(
                1
                for row in treatments
                if _num(row.get("control_score")) >= 1.0
                and _num(row.get("score")) < _num(row.get("control_score"))
            )
            / max(
                1,
                sum(1 for row in treatments if _num(row.get("control_score")) >= 1.0),
            )
        )

        by_fixture: dict[str, list[float]] = defaultdict(list)
        for row in baseline:
            by_fixture[str(row.get("fixture_id"))].append(_num(row.get("score")))
        instability = mean(
            [pstdev(values) for values in by_fixture.values() if len(values) > 1]
        ) if any(len(values) > 1 for values in by_fixture.values()) else 0.0

        hard_fail_rate = (
            sum(
                1
                for row in baseline
                if int(row.get("difficulty_level") or 0) >= 5
                and _num(row.get("score")) < 1.0
            )
            / max(
                1,
                sum(1 for row in baseline if int(row.get("difficulty_level") or 0) >= 5),
            )
        )
        priority = (
            1.50 * (1.0 - pass_rate)
            + 1.25 * hard_fail_rate
            + 0.75 * rescue_rate
            + 0.50 * instability
            + 0.25 * regression_rate
        )
        raw_priorities[family] = priority

        level_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in baseline:
            level_rows[int(row.get("difficulty_level") or 0)].append(row)
        levels = {
            str(level): {
                "baseline_pass_rate": mean([_num(row.get("score")) for row in values]),
                "n": len(values),
                "curriculum_role": (
                    "ANCHOR"
                    if all(_num(row.get("score")) >= 1.0 for row in values)
                    else "FRONTIER"
                    if any(_num(row.get("score")) >= 1.0 for row in values)
                    else "WEAKNESS"
                ),
            }
            for level, values in sorted(level_rows.items())
        }

        payload[family] = {
            "baseline_pass_rate": pass_rate,
            "hard_fail_rate": hard_fail_rate,
            "rescue_rate": rescue_rate,
            "negative_transfer_rate": regression_rate,
            "instability": instability,
            "raw_priority": priority,
            "difficulty_curriculum": levels,
        }

    total = sum(raw_priorities.values()) or 1.0
    for family, item in payload.items():
        item["recommended_training_mix_weight"] = raw_priorities[family] / total

    ranked = sorted(
        payload,
        key=lambda family: payload[family]["raw_priority"],
        reverse=True,
    )
    return {
        "schema_version": 1,
        "training_type": "MEASURED_CAPABILITY_CURRICULUM",
        "families": payload,
        "priority_order": ranked,
        "principle": "allocate more gradient budget to measured weakness/frontier while retaining stable anchors",
    }


def _candidate_utility(
    *,
    score: float,
    calls: float,
    tokens: float,
    latency: float,
) -> tuple[float, float, float, float]:
    return (score, -calls, -tokens, -latency)


def build_router_supervision(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create task -> DIRECT/control labels from observed best action."""
    by_fixture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("partition") != "DISCOVERY":
            continue
        fixture_id = str(row.get("fixture_id") or "")
        if fixture_id:
            by_fixture[fixture_id].append(row)

    result = []
    for fixture_id, values in sorted(by_fixture.items()):
        baseline = next(
            (row for row in values if row.get("intervention_id") == "CONTROL"),
            None,
        )
        treatments = [
            row
            for row in values
            if row.get("intervention_id") not in {None, "CONTROL"}
            and _text(row.get("task_text"))
        ]
        if baseline is None or not treatments:
            continue

        direct_score = _num(baseline.get("score"))
        direct_tokens = _tokens(baseline.get("cost"))
        direct_latency = _latency(baseline.get("cost"))
        best_label = "DIRECT"
        best_category = "DIRECT"
        best_utility = _candidate_utility(
            score=direct_score,
            calls=1.0,
            tokens=direct_tokens,
            latency=direct_latency,
        )
        best_row = baseline

        for row in treatments:
            utility = _candidate_utility(
                score=_num(row.get("score")),
                calls=_calls(row),
                tokens=_tokens(row.get("cost")),
                latency=_latency(row.get("cost")),
            )
            if utility > best_utility:
                best_utility = utility
                best_label = str(row.get("intervention_id"))
                best_category = str(row.get("intervention_category") or "")
                best_row = row

        task = _text(
            best_row.get("task_text")
            or next((row.get("task_text") for row in treatments if row.get("task_text")), "")
        )
        if not task:
            continue
        result.append(
            {
                "schema_version": 1,
                "record_id": _stable_id("router", fixture_id),
                "training_type": "ROUTER_ACTIVATION_SUPERVISION",
                "fixture_id": fixture_id,
                "family_id": best_row.get("family_id"),
                "difficulty_level": best_row.get("difficulty_level"),
                "input": task,
                "target_action": best_label,
                "target_category": best_category,
                "direct_score": direct_score,
                "selected_score": _num(best_row.get("score")),
                "selected_calls": (
                    1.0 if best_label == "DIRECT" else _calls(best_row)
                ),
                "activation_rule": (
                    "USE_DIRECT"
                    if best_label == "DIRECT"
                    else "ACTIVATE_MEASURED_CONTROL"
                ),
            }
        )
    return result


def build_stability_anchors(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve model strengths during later fine-tuning."""
    by_fixture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("partition") == "DISCOVERY" and row.get("fixture_id"):
            by_fixture[str(row.get("fixture_id"))].append(row)

    result = []
    for fixture_id, values in sorted(by_fixture.items()):
        baseline = next(
            (
                row
                for row in values
                if row.get("intervention_id") == "CONTROL"
                and _num(row.get("score")) >= 1.0
            ),
            None,
        )
        if baseline is None:
            continue
        regressions = [
            row
            for row in values
            if row.get("intervention_id") not in {None, "CONTROL"}
            and _num(row.get("score")) < _num(row.get("control_score"))
        ]
        treatment_source = next(
            (row for row in values if _text(row.get("task_text"))),
            None,
        )
        task = _text((treatment_source or {}).get("task_text"))
        target = _text(baseline.get("response_text"))
        if not target:
            target = _text((treatment_source or {}).get("control_response_text"))
        if not task or not target:
            continue
        difficulty = int(baseline.get("difficulty_level") or 0)
        result.append(
            {
                "schema_version": 1,
                "record_id": _stable_id("anchor", fixture_id),
                "training_type": "STABILITY_REHEARSAL_ANCHOR",
                "fixture_id": fixture_id,
                "family_id": baseline.get("family_id"),
                "difficulty_level": difficulty,
                "input": task,
                "target": target,
                "observed_regressing_controls": [
                    str(row.get("intervention_id")) for row in regressions
                ],
                "sample_weight": 1.0 + 0.5 * len(regressions) + (0.25 if difficulty >= 5 else 0.0),
                "purpose": "prevent fine-tuning from erasing already-correct base behavior",
            }
        )
    return result


def build_cross_family_transfer_graph(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure which controls generalize, specialize, or negatively transfer."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if (
            row.get("partition") == "DISCOVERY"
            and row.get("intervention_id") not in {None, "CONTROL"}
            and row.get("family_id")
        ):
            grouped[
                (str(row.get("intervention_id")), str(row.get("family_id")))
            ].append(_num(row.get("delta")))

    edges = []
    by_intervention: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (intervention_id, family_id), deltas in sorted(grouped.items()):
        item = {
            "intervention_id": intervention_id,
            "family_id": family_id,
            "n": len(deltas),
            "mean_delta": mean(deltas),
            "wins": sum(1 for delta in deltas if delta > 0),
            "losses": sum(1 for delta in deltas if delta < 0),
            "nulls": sum(1 for delta in deltas if delta == 0),
        }
        edges.append(item)
        by_intervention[intervention_id].append(item)

    controls = {}
    for intervention_id, values in sorted(by_intervention.items()):
        positive = [row for row in values if row["mean_delta"] > 0]
        negative = [row for row in values if row["mean_delta"] < 0]
        if len(positive) >= 3 and not negative:
            transfer_class = "GENERALIZER"
        elif positive and negative:
            transfer_class = "CONDITIONAL_SPECIALIST"
        elif negative and not positive:
            transfer_class = "GLOBAL_VETO_CANDIDATE"
        elif positive:
            transfer_class = "SPARSE_POSITIVE"
        else:
            transfer_class = "NULL_OR_UNRESOLVED"
        controls[intervention_id] = {
            "transfer_class": transfer_class,
            "positive_families": [row["family_id"] for row in positive],
            "negative_families": [row["family_id"] for row in negative],
            "tested_families": len(values),
            "mean_cross_family_delta": mean([row["mean_delta"] for row in values]),
        }

    return {
        "schema_version": 1,
        "training_type": "CROSS_FAMILY_TRANSFER_GRAPH",
        "edges": edges,
        "controls": controls,
        "principle": "generalize only controls with cross-family evidence; isolate mixed controls behind conditional routing",
    }


def _candidate_record(
    row: dict[str, Any],
    *,
    source: str,
    response: str,
    score: float,
    calls: float,
    cost: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "source": source,
        "response": response,
        "score": score,
        "calls": calls,
        "tokens": _tokens(cost),
        "latency": _latency(cost),
        "intervention_id": row.get("intervention_id"),
        "intervention_category": row.get("intervention_category"),
    }


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    no_worse = (
        a["score"] >= b["score"]
        and a["calls"] <= b["calls"]
        and a["tokens"] <= b["tokens"]
        and a["latency"] <= b["latency"]
    )
    strictly = (
        a["score"] > b["score"]
        or a["calls"] < b["calls"]
        or a["tokens"] < b["tokens"]
        or a["latency"] < b["latency"]
    )
    return no_worse and strictly


def build_pareto_targets(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Choose highest-quality, lowest-compute training targets per fixture."""
    by_fixture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("partition") == "DISCOVERY" and row.get("fixture_id"):
            by_fixture[str(row.get("fixture_id"))].append(row)

    result = []
    for fixture_id, values in sorted(by_fixture.items()):
        baseline = next(
            (row for row in values if row.get("intervention_id") == "CONTROL"),
            None,
        )
        treatment_rows = [
            row
            for row in values
            if row.get("intervention_id") not in {None, "CONTROL"}
            and _text(row.get("task_text"))
        ]
        if baseline is None or not treatment_rows:
            continue

        task = _text(treatment_rows[0].get("task_text"))
        candidates = []
        baseline_response = _text(
            treatment_rows[0].get("control_response_text")
            or baseline.get("response_text")
        )
        if baseline_response:
            candidates.append(
                _candidate_record(
                    baseline,
                    source="DIRECT",
                    response=baseline_response,
                    score=_num(baseline.get("score")),
                    calls=1.0,
                    cost=baseline.get("cost"),
                )
            )
        for row in treatment_rows:
            response = _text(row.get("treatment_response_text"))
            if not response:
                continue
            candidates.append(
                _candidate_record(
                    row,
                    source="TREATMENT",
                    response=response,
                    score=_num(row.get("score")),
                    calls=_calls(row),
                    cost=row.get("cost"),
                )
            )
        if not candidates or not task:
            continue

        frontier = [
            candidate
            for candidate in candidates
            if not any(
                _dominates(other, candidate)
                for other in candidates
                if other is not candidate
            )
        ]
        best_score = max(candidate["score"] for candidate in frontier)
        quality_frontier = [
            candidate for candidate in frontier if candidate["score"] == best_score
        ]
        selected = min(
            quality_frontier,
            key=lambda candidate: (
                candidate["calls"],
                candidate["tokens"],
                candidate["latency"],
            ),
        )
        result.append(
            {
                "schema_version": 1,
                "record_id": _stable_id("pareto", fixture_id),
                "training_type": "PARETO_QUALITY_EFFICIENCY_TARGET",
                "fixture_id": fixture_id,
                "family_id": treatment_rows[0].get("family_id"),
                "difficulty_level": treatment_rows[0].get("difficulty_level"),
                "input": task,
                "target": selected["response"],
                "selected_source": selected["source"],
                "selected_intervention_id": selected.get("intervention_id"),
                "quality_score": selected["score"],
                "calls": selected["calls"],
                "tokens": selected["tokens"],
                "latency": selected["latency"],
                "pareto_frontier_size": len(frontier),
                "purpose": "teach the model the cheapest observed response that preserves maximum measured quality",
            }
        )
    return result


def build_zero_clock_model_manufacturing(
    rows: list[dict[str, Any]],
    families: Iterable[str],
) -> dict[str, Any]:
    distillation = build_harness_to_weight_distillation(rows)
    preferences = build_weighted_preference_pairs(rows)
    curriculum = build_capability_curriculum(rows, families)
    router = build_router_supervision(rows)
    anchors = build_stability_anchors(rows)
    transfer = build_cross_family_transfer_graph(rows)
    pareto = build_pareto_targets(rows)

    return {
        "schema_version": 1,
        "zero_model_calls_added": True,
        "zero_active_test_seconds_added": True,
        "products": list(ZERO_CLOCK_MODEL_BUILDING_PRODUCTS),
        "counts": {
            "harness_to_weight_distillation": len(distillation),
            "weighted_preference_pairs": len(preferences),
            "router_supervision": len(router),
            "stability_anchors": len(anchors),
            "pareto_targets": len(pareto),
            "curriculum_families": len(curriculum.get("families") or {}),
            "transfer_edges": len(transfer.get("edges") or []),
        },
        "distillation": distillation,
        "preferences": preferences,
        "curriculum": curriculum,
        "router": router,
        "anchors": anchors,
        "transfer": transfer,
        "pareto": pareto,
    }
