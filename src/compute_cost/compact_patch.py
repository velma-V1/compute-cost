"""Minimal compatibility repair for the existing capability campaign."""

from __future__ import annotations

import copy
from typing import Any

TRUNCATION = {"THINK_TRUNCATED", "ANSWER_TRUNCATED"}


def resolve_model_reasoning_control(
    model: str, campaign_config: dict[str, Any]
) -> tuple[bool, str | None]:
    """Return the lowest valid baseline control for known local model families."""
    name = model.lower()
    if name.startswith("gpt-oss"):
        return True, "low"
    if name.startswith("qwen3.5") or name.startswith("devstral"):
        return False, None
    effort = campaign_config.get("reasoning_effort")
    return bool(campaign_config.get("thinking_mode", False)), (
        str(effort) if effort is not None else None
    )


def _next_budget(current: int, ceiling: int) -> int | None:
    return None if current >= ceiling else min(ceiling, current * 2)


def run_family_frontier_compact(
    runner: Any,
    family_id: str,
    ladder: dict[int, dict[str, Any]],
    *,
    sequence_start: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Run adaptive probes; escalate tokens only after proven truncation.

    ``max_experiments_per_family`` limits semantic/difficulty probes. Truncation
    retries are separately bounded by the generation ceiling and global call cap.
    Every new probe starts at the base token budget; exact replications retain the
    configuration they are replicating.
    """
    import compute_cost.capability_campaign as campaign

    assert runner.store is not None
    cfg = runner.config["capability_campaign"]
    repeats = int(cfg["boundary_repeats"])
    reliable = float(cfg.get("reliable_threshold", 0.90))
    unstable = float(cfg.get("unstable_threshold", 0.40))
    controller = campaign.AdaptiveDifficultyController(
        anchor_level=int(cfg["anchor_level"]),
        jump=int(cfg["jump"]),
        boundary_repeats=repeats,
    )
    thinking, effort = resolve_model_reasoning_control(str(runner.model), cfg)
    base_budget = int(cfg["generation_budget"])
    ceiling = int(
        cfg.get(
            "max_generation_budget",
            (runner.config.get("characterization") or {}).get(
                "max_generation_budget", base_budget * 8
            ),
        )
    )
    ceiling = max(base_budget, ceiling)
    max_probes = int(cfg["max_experiments_per_family"])

    rows: list[dict[str, Any]] = []
    controller_obs: list[Any] = []
    retained_obs: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, Any]] = {}
    attempted_levels: set[int] = set()
    latest_spec_by_level: dict[int, Any] = {}
    previous_spec: Any | None = None
    invalid_retries: dict[int, int] = {}
    sequence = sequence_start
    logical_probes = 0
    forced_stop = False

    while logical_probes < max_probes and not forced_stop:
        decision = controller.next(controller_obs)
        if decision.action == "STOP":
            campaign._record_family_stop(
                runner, family_id, reason=decision.reason, experiments=len(rows)
            )
            break
        assert decision.level is not None
        requested = int(decision.level)
        level = (
            requested
            if decision.action == "REPLICATE" and requested in ladder
            else campaign.resolve_requested_level(ladder, requested, attempted_levels)
        )
        if level is None:
            campaign._record_family_stop(
                runner,
                family_id,
                reason="MISSING_FIXTURE_COVERAGE",
                experiments=len(rows),
                requested_level=requested,
            )
            break

        fixture = ladder[level]
        if decision.action == "REPLICATE":
            parent = latest_spec_by_level.get(level)
            if parent is None:
                raise ValueError(
                    f"cannot replicate family {family_id} level {level} without prior experiment"
                )
            changed_variable = "replication"
            current_budget = int(parent.generation_budget)
        else:
            # Reset to base for a new task/difficulty probe. If resetting would
            # change both difficulty and budget relative to the prior attempt,
            # begin a new baseline lineage instead of falsifying one-variable lineage.
            if previous_spec is not None and int(previous_spec.generation_budget) == base_budget:
                parent = previous_spec
                changed_variable = "difficulty_level"
            else:
                parent = None
                changed_variable = "baseline"
            current_budget = base_budget

        first_attempt = True
        final_spec = None
        final_class = None
        final_valid = False
        final_passed: bool | None = None

        while True:
            sequence += 1
            spec = campaign._spec(
                sequence=sequence,
                family_id=family_id,
                fixture=fixture,
                parent=parent,
                changed_variable=changed_variable if first_attempt else "generation_budget",
                hypothesis=(
                    decision.reason
                    if first_attempt
                    else "token ceiling exhausted; retry identical fixture at next budget"
                ),
                thinking_mode=thinking,
                reasoning_effort=effort,
                generation_budget=current_budget,
            )
            if rows:
                campaign._add_adaptive_progress_task(
                    runner,
                    f"{family_id} {decision.action.lower()} L{level} budget {current_budget}",
                )

            def capture(snapshot: dict[str, Any], *, eid: str = spec.experiment_id) -> None:
                snapshots[eid] = copy.deepcopy(snapshot)

            row = campaign._run_with_progress(
                runner,
                family_id,
                fixture,
                spec,
                parent,
                snapshot_sink=capture,
            )
            rows.append(row)
            classification = row.get("classification") or {}
            result_class = str(classification.get("result_class"))
            valid = classification.get("valid_for_capability") is True
            passed = None if not valid else result_class == "ANSWER_CORRECT"
            observation = {
                "family_id": family_id,
                "level": level,
                "passed": passed,
                "valid_for_capability": valid,
                "result_class": result_class,
                "experiment_id": spec.experiment_id,
                "fixture_id": str(fixture["id"]),
                "generation_budget": current_budget,
            }
            runner.store.append_jsonl("capability-observations.jsonl", observation)
            retained_obs.append(observation)
            final_spec, final_class, final_valid, final_passed = (
                spec,
                result_class,
                valid,
                passed,
            )

            if result_class not in TRUNCATION:
                break
            next_budget = _next_budget(current_budget, ceiling)
            if next_budget is None:
                campaign._record_family_stop(
                    runner,
                    family_id,
                    reason="TOKEN_SAFETY_CEILING",
                    experiments=len(rows),
                    requested_level=level,
                )
                forced_stop = True
                break

            # Only truncation reaches this branch. The exact fixture and all other
            # controls stay fixed; only generation_budget changes.
            parent = spec
            current_budget = next_budget
            first_attempt = False

        if final_spec is None or forced_stop:
            break

        logical_probes += 1
        controller_obs.append(
            campaign.DifficultyObservation(
                level=level,
                passed=final_passed,
                valid_for_capability=final_valid,
            )
        )

        # Preserve the most recent non-truncated attempt even when invalid. The
        # controller may request an exact retry of a runtime-invalid observation.
        latest_spec_by_level[level] = final_spec
        previous_spec = final_spec
        if final_valid:
            attempted_levels.add(level)
            invalid_retries.pop(level, None)
        else:
            invalid_retries[level] = invalid_retries.get(level, 0) + 1
            if invalid_retries[level] >= 2:
                campaign._record_family_stop(
                    runner,
                    family_id,
                    reason="REPEATED_INVALID_OBSERVATION",
                    experiments=len(rows),
                    requested_level=level,
                )
                break
    else:
        if not forced_stop:
            campaign._record_family_stop(
                runner,
                family_id,
                reason="MAX_LOGICAL_PROBES_PER_FAMILY",
                experiments=len(rows),
            )

    campaign._persist_boundary_replays(
        runner,
        family_id,
        retained_obs,
        snapshots,
        boundary_repeats=repeats,
        reliable_threshold=reliable,
        unstable_threshold=unstable,
    )
    return rows, sequence


def run_capability_campaign_compact(
    runner: Any, cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run the existing phases but honor every secondary lab's enabled flag."""
    import compute_cost.capability_campaign as campaign

    ladders = campaign.build_ladder_index(cases)
    all_rows: list[dict[str, Any]] = []
    sequence = 0
    for family_id, ladder in ladders.items():
        family_rows, sequence = campaign.run_family_frontier(
            runner, family_id, ladder, sequence_start=sequence
        )
        all_rows.extend(family_rows)

    base_rows = list(all_rows)
    frontiers = campaign._baseline_frontiers(runner, base_rows)
    effort_rows: list[dict[str, Any]] = []
    curve_cfg = runner.config.get("reasoning_curves")
    if isinstance(curve_cfg, dict) and curve_cfg.get("enabled") is True:
        effort_rows = campaign.run_reasoning_curves(
            runner, cases, base_rows, frontiers, sequence_start=sequence
        )
        sequence += len(effort_rows)
        runner.store.write_json(
            "reasoning-curves.json",
            campaign.build_reasoning_curves(
                str(runner.model),
                frontiers,
                base_rows,
                effort_rows,
                repeats=int(curve_cfg["repeats"]),
                reliable_threshold=float(
                    runner.config["capability_campaign"].get("reliable_threshold", 0.90)
                ),
            ),
            producer="reasoning-curves",
            stage="report",
        )
        all_rows.extend(effort_rows)

    pre_recovery_atlas = campaign.build_failure_atlas(str(runner.model), all_rows)
    recovery_rows: list[dict[str, Any]] = []
    recovery_map: dict[str, Any] = {"schema_version": 1, "families": {}}
    recovery_cfg = runner.config.get("recovery_lab")
    if isinstance(recovery_cfg, dict) and recovery_cfg.get("enabled") is True:
        recovery_rows, recovery_map, sequence = campaign.run_recovery_lab(
            runner,
            cases,
            base_rows,
            effort_rows,
            frontiers,
            pre_recovery_atlas,
            sequence_start=sequence,
        )
        runner.store.write_json(
            "recovery-map.json", recovery_map, producer="recovery-lab", stage="report"
        )
        all_rows.extend(recovery_rows)

    robustness_cfg = runner.config.get("robustness_lab")
    if isinstance(robustness_cfg, dict) and robustness_cfg.get("enabled") is True:
        robustness_rows, robustness_map, sequence = campaign.run_robustness_lab(
            runner,
            cases,
            base_rows,
            effort_rows,
            recovery_rows,
            frontiers,
            recovery_map,
            sequence_start=sequence,
        )
        runner.store.write_json(
            "robustness-map.json", robustness_map, producer="robustness-lab", stage="report"
        )
        all_rows.extend(robustness_rows)

    compound_cfg = runner.config.get("compound_lab")
    if isinstance(compound_cfg, dict) and compound_cfg.get("enabled") is True:
        compound_rows, compound_map, sequence = campaign.run_compound_lab(
            runner, frontiers, sequence_start=sequence
        )
        runner.store.write_json(
            "compound-map.json", compound_map, producer="compound-lab", stage="report"
        )
        all_rows.extend(compound_rows)

    runner.store.write_json(
        "failure-atlas.json",
        campaign.build_failure_atlas(str(runner.model), all_rows),
        producer="failure-atlas",
        stage="report",
    )
    return all_rows


def install() -> None:
    """Install compact behavior without replacing the benchmark architecture."""
    import compute_cost.capability_campaign as campaign
    import compute_cost.config as config
    import compute_cost.run_synthesis as synthesis

    config.DEFAULT_CONFIG["limits"]["max_model_calls_per_run"] = 120
    config.DEFAULT_CONFIG["capability_campaign"].update(
        {
            "boundary_repeats": 2,
            "max_experiments_per_family": 2,
            "thinking_mode": False,
            "reasoning_effort": None,
            "generation_budget": 256,
        }
    )
    config.DEFAULT_CONFIG["reasoning_curves"]["enabled"] = False
    config.DEFAULT_CONFIG["recovery_lab"]["enabled"] = False
    config.DEFAULT_CONFIG["robustness_lab"]["enabled"] = False
    config.DEFAULT_CONFIG["compound_lab"]["enabled"] = False

    campaign.resolve_model_reasoning_control = resolve_model_reasoning_control
    campaign.run_family_frontier = run_family_frontier_compact
    campaign.run_capability_campaign = run_capability_campaign_compact

    if getattr(synthesis.build_cost_value_outputs, "_compact_scorecard_patch", False):
        return
    original = synthesis.build_cost_value_outputs

    def with_scorecard(model, rows, frontiers, run_dir):
        import json
        from pathlib import Path

        from .compact_scorecard import build_compact_scorecard, render_compact_scorecard

        materialized = list(rows)
        cost_map, value_map = original(model, materialized, frontiers, run_dir)
        root = Path(run_dir)
        scorecard = build_compact_scorecard(model, materialized, frontiers)
        (root / "scorecard.json").write_text(
            json.dumps(scorecard, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (root / "scorecard.md").write_text(
            render_compact_scorecard(scorecard), encoding="utf-8"
        )
        return cost_map, value_map

    with_scorecard._compact_scorecard_patch = True
    synthesis.build_cost_value_outputs = with_scorecard
