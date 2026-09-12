"""Progress-aware public runner layered over the stable benchmark core."""

from __future__ import annotations

import copy
import json
import traceback
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .capability_campaign import run_capability_campaign
from .characterization import build_characterization_summary, render_characterization_report, run_characterization
from .evidence import EvidenceStore
from .frontier import build_capability_frontiers
from .hardware import collect_hardware_snapshot
from .progress import ProgressDisplay
from .report import build_summary, render_report
from .run_synthesis import build_cost_value_outputs as _build_cost_value_outputs
from .test1_campaign import build_test1_plan, partition_cases, run_test1_campaign, validate_test1_plan
from .test11_campaign import (
    build_ingredient_bank,
    build_test11_plan,
    load_test1_source,
    run_test11_campaign,
    synthetic_test1_source,
    validate_test11_plan,
)
from .test12_campaign import (
    build_intervention_bank,
    build_test12_plan,
    fresh_model_source,
    partition_test12_cases,
    load_test11_source,
    load_test12_recovery,
    run_test12_campaign,
    validate_test12_plan,
)
from .test12_tuning import (
    build_tuning_plan,
    load_collection,
    load_tuning_recovery,
    run_test12_tuning,
    validate_tuning_plan,
)
from .test2_campaign import (
    build_test2_plan,
    load_test1_handoff,
    run_test2_campaign,
    synthetic_test1_handoff,
    validate_test2_plan,
)
from .runner_core import BenchmarkRunner as _CoreBenchmarkRunner
from .runner_core import build_context_case

__all__ = ["BenchmarkRunner", "build_context_case"]


class BenchmarkRunner(_CoreBenchmarkRunner):
    """Benchmark runner with adaptive live progress and retained progress evidence."""

    def __init__(
        self,
        runtime: Any,
        config: dict[str, Any],
        suite: dict[str, Any],
        *,
        results_root: str | Path = "results",
        telemetry: Any | None = None,
        hardware_collector: Callable[[], dict[str, Any]] = collect_hardware_snapshot,
        progress_factory: Callable[[int], Any] | None = None,
    ) -> None:
        super().__init__(
            runtime,
            config,
            suite,
            results_root=results_root,
            telemetry=telemetry,
            hardware_collector=hardware_collector,
        )
        self._progress_factory = progress_factory or (lambda total: ProgressDisplay(total))
        self.progress: Any | None = None
        self._progress_preflight_complete = False
        self._progress_stopped = False

    def _planned_progress_tasks(self) -> int:
        limits = self.config.get("limits", {})
        return (
            1
            + 1
            + 1
            + len(self.suite.get("cases", []) or [])
            + len(limits.get("context_schedule", []) or [])
            + max(0, int(limits.get("sustained_iterations", 0)))
            + 1
        )

    def _planned_characterization_tasks(self) -> int:
        return 1 + (2 * len(self.suite.get("cases", []) or [])) + 1

    def _planned_capability_tasks(self) -> int:
        families = {
            str(case.get("family_id") or case.get("category"))
            for case in (self.suite.get("cases", []) or [])
            if case.get("family_id") or case.get("category")
        }
        return 1 + len(families) + 1

    def _progress_snapshot(self) -> dict[str, Any]:
        if self.progress is None:
            return {}
        snap = copy.deepcopy(self.progress.snapshot())
        finish = snap.get("finish")
        if isinstance(finish, datetime):
            snap["finish"] = finish.isoformat()
        if hasattr(self.progress, "render"):
            try:
                snap["rendered_line"] = self.progress.render()
            except Exception:
                pass
        if hasattr(self.progress, "width_getter"):
            try:
                snap["terminal_width"] = int(self.progress.width_getter())
            except Exception:
                pass
        return snap

    def _record_progress(self, event: str, task: str | None = None, **extra: Any) -> None:
        if self.store is None or self.progress is None:
            return
        row = {
            "event": event,
            "timestamp_utc": self._utc(),
            "task": task,
            "progress": self._progress_snapshot(),
            **extra,
        }
        with self._write_lock:
            self.store.append_jsonl("progress.jsonl", row)

    def _progress_begin(self, task: str) -> None:
        if self.progress is None:
            return
        self.progress.begin_task(task)
        self._record_progress("task_start", task)

    def _progress_complete(self, task: str) -> None:
        if self.progress is None:
            return
        self.progress.complete_task(task)
        self._record_progress("task_complete", task)

    def _start_telemetry(self) -> None:
        if self.progress is not None:
            self._record_progress("start", "preflight")
        super()._start_telemetry()

    @staticmethod
    def _case_progress_label(stage: str, case_id: str, target_context: int | None, iteration: int | None) -> str:
        if stage == "cold":
            return "cold start"
        if stage == "warmup":
            return "warmup"
        if stage == "context":
            return f"context {target_context or case_id}"
        if stage == "sustained":
            return f"sustained {(iteration or 0) + 1}"
        return f"{stage} {case_id}"

    def _execute_case(
        self,
        case: dict[str, Any],
        *,
        stage: str,
        target_context: int | None = None,
        iteration: int | None = None,
        option_overrides: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        if self.progress is not None and not self._progress_preflight_complete:
            self._progress_complete("preflight")
            self._progress_preflight_complete = True
        label = self._case_progress_label(stage, str(case.get("id", "case")), target_context, iteration)
        self._progress_begin(label)
        try:
            return super()._execute_case(
                case,
                stage=stage,
                target_context=target_context,
                iteration=iteration,
                option_overrides=option_overrides,
            )
        finally:
            self._progress_complete(label)

    def onboard(self, model: str, *, pull: bool = False) -> Path:
        self.progress = self._progress_factory(self._planned_progress_tasks())
        self._progress_preflight_complete = False
        self._progress_stopped = False
        self.progress.start("preflight")
        self.progress.start_live()
        try:
            return super().onboard(model, pull=pull)
        finally:
            if self.progress is not None and not self._progress_stopped:
                self.progress.stop_live(newline=True)
                self._progress_stopped = True

    def characterize(self, model: str, *, pull: bool = False) -> Path:
        """Run one independent adaptive characterization with no changes to onboarding core."""
        self.progress = self._progress_factory(self._planned_characterization_tasks())
        self._progress_preflight_complete = False
        self._progress_stopped = False
        self.progress.start("preflight")
        self.progress.start_live()

        self.model = model
        run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.store = EvidenceStore(self.results_root, run_id)
        store = self.store
        store.write_json("resolved-config.json", self.config, producer="runner", stage="preflight")
        store.write_json("benchmark-snapshot.json", self.suite, producer="runner", stage="preflight")
        self._event("RUN_START", model=model, benchmark_version=self.suite.get("benchmark_version"), mode="characterize")
        self._start_telemetry()

        try:
            hardware = self.hardware_collector()
            store.write_json("hardware.json", hardware, producer="hardware", stage="preflight")

            version = self.runtime.version()
            version_refs = self._persist_control_exchange("runtime-version", version)
            tags = self.runtime.list_models()
            tags_refs = self._persist_control_exchange("model-list", tags)
            available = (
                self.runtime.model_available_in(tags, model)
                if hasattr(self.runtime, "model_available_in")
                else self.runtime.is_model_available(model)
            )

            pull_refs = None
            if not available and pull:
                pull_result = self.runtime.pull(model)
                pull_refs = self._persist_control_exchange("model-pull", pull_result)
                tags = self.runtime.list_models()
                tags_refs = self._persist_control_exchange("model-list-after-pull", tags)
                available = (
                    self.runtime.model_available_in(tags, model)
                    if hasattr(self.runtime, "model_available_in")
                    else self.runtime.is_model_available(model)
                )

            if not available:
                store.write_json(
                    "runtime.json",
                    {
                        "model": model,
                        "available": False,
                        "version": version.get("parsed"),
                        "model_size_bytes": None,
                        "evidence_refs": {"version": version_refs, "tags": tags_refs, "pull": pull_refs},
                    },
                    producer="runner",
                    stage="preflight",
                )
                self._event("RUN_FAILED", failure="MODEL_NOT_FOUND", model=model)
                return self._finalize_run()

            info = self.runtime.model_info(model)
            info_refs = self._persist_control_exchange("model-info", info)
            store.write_json(
                "runtime.json",
                {
                    "model": model,
                    "available": True,
                    "version": version.get("parsed"),
                    "model_size_bytes": self._find_model_size(tags, model),
                    "model_info": info.get("parsed"),
                    "evidence_refs": {"version": version_refs, "tags": tags_refs, "show": info_refs, "pull": pull_refs},
                },
                producer="runner",
                stage="preflight",
            )
            self._event("PREFLIGHT_COMPLETE", model=model, mode="characterize")
            self._progress_complete("preflight")
            self._progress_preflight_complete = True

            cases = self.suite.get("cases", []) or []
            rows = run_characterization(self, cases)
            characterization_summary = build_characterization_summary(
                model,
                cases,
                rows,
                boundary_repeats=int(self.config["characterization"]["boundary_repeats"]),
            )
            store.write_json(
                "characterization-summary.json",
                characterization_summary,
                producer="characterization",
                stage="report",
            )
            store.write_raw(
                "characterization-report.md",
                render_characterization_report(characterization_summary),
                producer="characterization",
                stage="report",
                media_type="text/markdown",
            )
            self._event("CHARACTERIZATION_COMPLETE", model=model, experiments=len(rows))
            return self._finalize_run()
        except Exception as exc:
            store.write_json(
                "raw/runner-failure.json",
                {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                producer="runner",
                stage="fatal",
            )
            self._event("RUN_FAILED", failure="CHARACTERIZATION_ERROR", error_type=type(exc).__name__, error=str(exc))
            return self._finalize_run()
        finally:
            if self.progress is not None and not self._progress_stopped:
                self.progress.stop_live(newline=True)
                self._progress_stopped = True

    @staticmethod
    def _coverage_state(frontier: dict[str, Any], boundary_repeats: int) -> str:
        levels = frontier.get("levels", []) or []
        valid_total = sum(int(row.get("valid_count", 0)) for row in levels)
        invalid_total = sum(int(row.get("invalid_count", 0)) for row in levels)
        if valid_total == 0:
            return "UNCERTAIN" if invalid_total else "UNTESTED"

        by_level = {int(row["level"]): row for row in levels}
        bracket = frontier.get("transition_bracket")
        if isinstance(bracket, dict):
            lower = by_level.get(int(bracket["lower_level"]))
            upper = by_level.get(int(bracket["upper_level"]))
            if (
                lower is not None
                and upper is not None
                and int(lower.get("valid_count", 0)) >= boundary_repeats
                and int(upper.get("valid_count", 0)) >= boundary_repeats
            ):
                return "PROVEN"

        max_row = by_level.get(10)
        if (
            max_row is not None
            and max_row.get("label") == "reliable"
            and int(max_row.get("valid_count", 0)) >= boundary_repeats
        ):
            return "PROVEN"

        min_row = by_level.get(0)
        if (
            min_row is not None
            and min_row.get("label") == "failure"
            and int(min_row.get("valid_count", 0)) >= boundary_repeats
        ):
            return "PROVEN"
        return "PARTIAL"

    def _build_coverage_ledger(self, frontiers: dict[str, Any]) -> dict[str, Any]:
        boundary_repeats = int(self.config["capability_campaign"]["boundary_repeats"])
        families: dict[str, Any] = {}
        for family_id, frontier in sorted((frontiers.get("families") or {}).items()):
            levels = frontier.get("levels", []) or []
            families[family_id] = {
                "state": self._coverage_state(frontier, boundary_repeats),
                "tested_levels": list(frontier.get("coverage", {}).get("tested_levels", [])),
                "untested_levels": list(frontier.get("coverage", {}).get("untested_levels", [])),
                "valid_observations": sum(int(row.get("valid_count", 0)) for row in levels),
                "invalid_observations": sum(int(row.get("invalid_count", 0)) for row in levels),
                "reliable_floor": frontier.get("reliable_floor"),
                "first_failure_level": frontier.get("first_failure_level"),
            }
        return {
            "schema_version": 1,
            "taxonomy_version": frontiers.get("taxonomy_version"),
            "families": families,
        }

    def capability_characterize(self, model: str, *, pull: bool = False) -> Path:
        """Run adaptive family-local capability frontier search as a sibling mode."""
        self.progress = self._progress_factory(self._planned_capability_tasks())
        self._progress_preflight_complete = False
        self._progress_stopped = False
        self.progress.start("preflight")
        self.progress.start_live()

        self.model = model
        run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.store = EvidenceStore(self.results_root, run_id)
        store = self.store
        store.write_json("resolved-config.json", self.config, producer="runner", stage="preflight")
        store.write_json("benchmark-snapshot.json", self.suite, producer="runner", stage="preflight")
        self._event(
            "RUN_START",
            model=model,
            benchmark_version=self.suite.get("benchmark_version"),
            mode="capability-characterize",
        )
        self._start_telemetry()

        try:
            hardware = self.hardware_collector()
            store.write_json("hardware.json", hardware, producer="hardware", stage="preflight")

            version = self.runtime.version()
            version_refs = self._persist_control_exchange("runtime-version", version)
            tags = self.runtime.list_models()
            tags_refs = self._persist_control_exchange("model-list", tags)
            available = (
                self.runtime.model_available_in(tags, model)
                if hasattr(self.runtime, "model_available_in")
                else self.runtime.is_model_available(model)
            )

            pull_refs = None
            if not available and pull:
                pull_result = self.runtime.pull(model)
                pull_refs = self._persist_control_exchange("model-pull", pull_result)
                tags = self.runtime.list_models()
                tags_refs = self._persist_control_exchange("model-list-after-pull", tags)
                available = (
                    self.runtime.model_available_in(tags, model)
                    if hasattr(self.runtime, "model_available_in")
                    else self.runtime.is_model_available(model)
                )

            if not available:
                store.write_json(
                    "runtime.json",
                    {
                        "model": model,
                        "available": False,
                        "version": version.get("parsed"),
                        "model_size_bytes": None,
                        "evidence_refs": {"version": version_refs, "tags": tags_refs, "pull": pull_refs},
                    },
                    producer="runner",
                    stage="preflight",
                )
                self._event("RUN_FAILED", failure="MODEL_NOT_FOUND", model=model)
                return self._finalize_run()

            info = self.runtime.model_info(model)
            info_refs = self._persist_control_exchange("model-info", info)
            store.write_json(
                "runtime.json",
                {
                    "model": model,
                    "available": True,
                    "version": version.get("parsed"),
                    "model_size_bytes": self._find_model_size(tags, model),
                    "model_info": info.get("parsed"),
                    "evidence_refs": {"version": version_refs, "tags": tags_refs, "show": info_refs, "pull": pull_refs},
                },
                producer="runner",
                stage="preflight",
            )
            self._event("PREFLIGHT_COMPLETE", model=model, mode="capability-characterize")
            self._progress_complete("preflight")
            self._progress_preflight_complete = True

            cases = self.suite.get("cases", []) or []
            rows = run_capability_campaign(self, cases)

            declared_families = set((self.suite.get("coverage") or {}).keys())
            declared_families.update(
                str(case.get("family_id") or case.get("category"))
                for case in cases
                if case.get("family_id") or case.get("category")
            )
            family_observations: dict[str, list[dict[str, Any]]] = {
                family_id: [] for family_id in sorted(declared_families)
            }
            observations_path = store.run_dir / "capability-observations.jsonl"
            if observations_path.is_file():
                for raw_line in observations_path.read_text(encoding="utf-8").splitlines():
                    if not raw_line.strip():
                        continue
                    observation = json.loads(raw_line)
                    family_id = str(observation["family_id"])
                    family_observations.setdefault(family_id, []).append(observation)

            cfg = self.config["capability_campaign"]
            frontiers = build_capability_frontiers(
                str(self.suite.get("taxonomy_version") or "unknown"),
                family_observations,
                thresholds={
                    "reliable": float(cfg.get("reliable_threshold", 0.90)),
                    "unstable": float(cfg.get("unstable_threshold", 0.40)),
                },
            )
            store.write_json(
                "capability-frontiers.json",
                frontiers,
                producer="capability-characterization",
                stage="report",
            )
            ledger = self._build_coverage_ledger(frontiers)
            store.write_json(
                "coverage-ledger.json",
                ledger,
                producer="capability-characterization",
                stage="report",
            )
            self._event(
                "CAPABILITY_CHARACTERIZATION_COMPLETE",
                model=model,
                experiments=len(rows),
                families=len(frontiers["families"]),
            )
            return self._finalize_run()
        except Exception as exc:
            store.write_json(
                "raw/runner-failure.json",
                {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                producer="runner",
                stage="fatal",
            )
            self._event(
                "RUN_FAILED",
                failure="CAPABILITY_CHARACTERIZATION_ERROR",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return self._finalize_run()
        finally:
            if self.progress is not None and not self._progress_stopped:
                self.progress.stop_live(newline=True)
                self._progress_stopped = True

    def gpt20b_test1(
        self,
        model: str,
        *,
        pull: bool = False,
        baseline_run: str | None = None,
        dry_run: bool = False,
    ) -> Path:
        """Run the frozen seven-hour GPT-20B Test-1 campaign."""
        campaign_started_monotonic = time.monotonic()
        test_cfg = self.config.get("test1_campaign") or {}
        expected_calls = int(test_cfg.get("expected_calls", 4300))
        safety_cap = int(test_cfg.get("safety_call_cap", 10000))
        limits = self.config.setdefault("limits", {})
        limits["max_model_calls_per_run"] = max(
            int(limits.get("max_model_calls_per_run", 3000)),
            safety_cap,
        )

        self.progress = self._progress_factory(expected_calls + 2)
        self._progress_preflight_complete = False
        self._progress_stopped = False
        self.progress.start("preflight")
        self.progress.start_live()

        self.model = model
        run_id = f"test1-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.store = EvidenceStore(self.results_root, run_id)
        store = self.store
        store.write_json("resolved-config.json", self.config, producer="runner", stage="preflight")
        store.write_json("benchmark-snapshot.json", self.suite, producer="runner", stage="preflight")
        plan = build_test1_plan(self.suite.get("cases", []) or [])
        validate_test1_plan(plan)
        store.write_json("test1-plan.json", plan, producer="test1", stage="preflight")
        store.write_json(
            "fixture-partitions.json",
            {
                "schema_version": 1,
                "partitions": {
                    name: [str(case.get("id")) for case in rows]
                    for name, rows in partition_cases(self.suite.get("cases", []) or []).items()
                },
            },
            producer="test1",
            stage="preflight",
        )
        self._event(
            "RUN_START",
            model=model,
            benchmark_version=self.suite.get("benchmark_version"),
            mode="gpt20b-test1-dry-run" if dry_run else "gpt20b-test1",
            baseline_run=baseline_run,
        )

        if dry_run:
            self._progress_complete("preflight")
            self._progress_preflight_complete = True
            self._event(
                "TEST1_DRY_RUN_COMPLETE",
                model=model,
                planned_wall_seconds=plan["wall_clock_seconds"],
                planned_active_seconds=plan["active_model_seconds"],
                ingredient_count=plan["ingredient_count"],
            )
            return self._finalize_run()

        self._start_telemetry()
        try:
            hardware = self.hardware_collector()
            store.write_json("hardware.json", hardware, producer="hardware", stage="preflight")

            version = self.runtime.version()
            version_refs = self._persist_control_exchange("runtime-version", version)
            tags = self.runtime.list_models()
            tags_refs = self._persist_control_exchange("model-list", tags)
            available = (
                self.runtime.model_available_in(tags, model)
                if hasattr(self.runtime, "model_available_in")
                else self.runtime.is_model_available(model)
            )

            pull_refs = None
            if not available and pull:
                pull_result = self.runtime.pull(model)
                pull_refs = self._persist_control_exchange("model-pull", pull_result)
                tags = self.runtime.list_models()
                tags_refs = self._persist_control_exchange("model-list-after-pull", tags)
                available = (
                    self.runtime.model_available_in(tags, model)
                    if hasattr(self.runtime, "model_available_in")
                    else self.runtime.is_model_available(model)
                )

            if not available:
                store.write_json(
                    "runtime.json",
                    {
                        "model": model,
                        "available": False,
                        "version": version.get("parsed"),
                        "model_size_bytes": None,
                        "evidence_refs": {
                            "version": version_refs,
                            "tags": tags_refs,
                            "pull": pull_refs,
                        },
                    },
                    producer="runner",
                    stage="preflight",
                )
                self._event("RUN_FAILED", failure="MODEL_NOT_FOUND", model=model)
                return self._finalize_run()

            info = self.runtime.model_info(model)
            info_refs = self._persist_control_exchange("model-info", info)
            store.write_json(
                "runtime.json",
                {
                    "model": model,
                    "available": True,
                    "version": version.get("parsed"),
                    "model_size_bytes": self._find_model_size(tags, model),
                    "model_info": info.get("parsed"),
                    "evidence_refs": {
                        "version": version_refs,
                        "tags": tags_refs,
                        "show": info_refs,
                        "pull": pull_refs,
                    },
                },
                producer="runner",
                stage="preflight",
            )
            self._event("PREFLIGHT_COMPLETE", model=model, mode="gpt20b-test1")
            self._progress_complete("preflight")
            self._progress_preflight_complete = True

            rows = run_test1_campaign(
                self,
                self.suite.get("cases", []) or [],
                baseline_run=baseline_run,
                started_monotonic=campaign_started_monotonic,
            )
            physical_calls = self._model_call_counts.get(store.run_id, 0)
            self._event(
                "TEST1_COMPLETE",
                model=model,
                observations=len(rows),
                physical_model_calls=physical_calls,
                baseline_run=baseline_run,
            )
            return self._finalize_run()
        except Exception as exc:
            store.write_json(
                "raw/runner-failure.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                producer="runner",
                stage="fatal",
            )
            self._event(
                "RUN_FAILED",
                failure="TEST1_ERROR",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return self._finalize_run()
        finally:
            if self.progress is not None and not self._progress_stopped:
                self.progress.stop_live(newline=True)
                self._progress_stopped = True

    def gpt20b_test11(
        self,
        model: str,
        *,
        pull: bool = False,
        test1_run: str | None = None,
        dry_run: bool = False,
    ) -> Path:
        """Run the seven-hour corrective GPT-20B Test-1.1 campaign."""
        test_cfg = self.config.get("test11_campaign") or {}
        expected_calls = int(test_cfg.get("expected_calls", 5200))
        safety_cap = int(test_cfg.get("safety_call_cap", 12000))
        limits = self.config.setdefault("limits", {})
        limits["max_model_calls_per_run"] = max(
            int(limits.get("max_model_calls_per_run", 3000)),
            safety_cap,
        )

        self.progress = self._progress_factory(expected_calls + 2)
        self._progress_preflight_complete = False
        self._progress_stopped = False
        self.progress.start("preflight")
        self.progress.start_live()

        self.model = model
        run_id = f"test1.1-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.store = EvidenceStore(self.results_root, run_id)
        store = self.store
        store.write_json("resolved-config.json", self.config, producer="runner", stage="preflight")
        store.write_json("benchmark-snapshot.json", self.suite, producer="runner", stage="preflight")
        plan = build_test11_plan(self.suite.get("cases", []) or [], test1_run=test1_run)
        validate_test11_plan(plan)
        store.write_json("test1.1-plan.json", plan, producer="test1.1", stage="preflight")
        store.write_json(
            "fixture-partitions.json",
            {
                "schema_version": 1,
                "partitions": {
                    name: [str(case.get("id")) for case in rows]
                    for name, rows in partition_cases(self.suite.get("cases", []) or []).items()
                },
            },
            producer="test1.1",
            stage="preflight",
        )
        self._event(
            "RUN_START",
            model=model,
            benchmark_version=self.suite.get("benchmark_version"),
            mode="gpt20b-test1.1-dry-run" if dry_run else "gpt20b-test1.1",
            test1_run=test1_run,
        )

        if dry_run:
            if test1_run:
                source = load_test1_source(
                    self.results_root,
                    test1_run,
                    self.suite.get("cases", []) or [],
                )
            else:
                source = synthetic_test1_source(self.suite.get("cases", []) or [])
            bank = build_ingredient_bank(source)
            store.write_json(
                "test1.1-dry-run-validation.json",
                {
                    "schema_version": 1,
                    "source": source.get("run_id"),
                    "synthetic_source": source.get("run_id") == "SYNTHETIC-TEST1",
                    "planned_wall_seconds": plan["wall_clock_seconds"],
                    "planned_active_seconds": plan["active_model_seconds"],
                    "seed_ingredient_count": plan["seed_ingredient_count"],
                    "expanded_ingredient_count": len(bank),
                    "dynamic_ingredient_count": len([row for row in bank if str(row["id"]).startswith("DYN-")]),
                    "unused_time_sink": plan["unused_time_sink"],
                    "protected_partitions": plan["prohibited_partitions"],
                    "protected_partitions_exposed": False,
                    "required_outputs": plan["required_outputs"],
                },
                producer="test1.1",
                stage="preflight",
            )
            self._progress_complete("preflight")
            self._progress_preflight_complete = True
            self._event(
                "TEST11_DRY_RUN_COMPLETE",
                model=model,
                test1_run=test1_run,
                expanded_ingredient_count=len(bank),
                planned_wall_seconds=plan["wall_clock_seconds"],
            )
            return self._finalize_run()

        if not test1_run:
            self._event("RUN_FAILED", failure="TEST1_RUN_REQUIRED", model=model)
            return self._finalize_run()

        self._start_telemetry()
        try:
            # Fail before the first model call if source evidence is corrupt,
            # partition-drifted, or incomplete.
            source = load_test1_source(
                self.results_root,
                test1_run,
                self.suite.get("cases", []) or [],
            )
            store.write_json(
                "test1.1-source-preflight.json",
                {
                    "schema_version": 1,
                    "source_run": test1_run,
                    "source_observations": len(source.get("observations", [])),
                    "source_positive_rows": len(source.get("positive_rows", [])),
                    "source_negative_rows": len(source.get("negative_rows", [])),
                    "source_truncation_rows": len(source.get("truncation_rows", [])),
                    "source_integrity": copy.deepcopy(source.get("source_integrity") or {}),
                    "expanded_ingredient_count": len(build_ingredient_bank(source)),
                },
                producer="test1.1",
                stage="preflight",
            )

            hardware = self.hardware_collector()
            store.write_json("hardware.json", hardware, producer="hardware", stage="preflight")

            version = self.runtime.version()
            version_refs = self._persist_control_exchange("runtime-version", version)
            tags = self.runtime.list_models()
            tags_refs = self._persist_control_exchange("model-list", tags)
            available = (
                self.runtime.model_available_in(tags, model)
                if hasattr(self.runtime, "model_available_in")
                else self.runtime.is_model_available(model)
            )

            pull_refs = None
            if not available and pull:
                pull_result = self.runtime.pull(model)
                pull_refs = self._persist_control_exchange("model-pull", pull_result)
                tags = self.runtime.list_models()
                tags_refs = self._persist_control_exchange("model-list-after-pull", tags)
                available = (
                    self.runtime.model_available_in(tags, model)
                    if hasattr(self.runtime, "model_available_in")
                    else self.runtime.is_model_available(model)
                )

            if not available:
                store.write_json(
                    "runtime.json",
                    {
                        "model": model,
                        "available": False,
                        "version": version.get("parsed"),
                        "model_size_bytes": None,
                        "evidence_refs": {"version": version_refs, "tags": tags_refs, "pull": pull_refs},
                    },
                    producer="runner",
                    stage="preflight",
                )
                self._event("RUN_FAILED", failure="MODEL_NOT_FOUND", model=model)
                return self._finalize_run()

            info = self.runtime.model_info(model)
            info_refs = self._persist_control_exchange("model-info", info)
            store.write_json(
                "runtime.json",
                {
                    "model": model,
                    "available": True,
                    "version": version.get("parsed"),
                    "model_size_bytes": self._find_model_size(tags, model),
                    "model_info": info.get("parsed"),
                    "evidence_refs": {"version": version_refs, "tags": tags_refs, "show": info_refs, "pull": pull_refs},
                },
                producer="runner",
                stage="preflight",
            )
            self._event("PREFLIGHT_COMPLETE", model=model, mode="gpt20b-test1.1")
            self._progress_complete("preflight")
            self._progress_preflight_complete = True

            # The seven-hour experimental clock starts only after all source-integrity
            # verification and runtime preflight work has completed. Large retained
            # evidence sets can take minutes to re-hash and must not consume active
            # experiment time.
            campaign_started_monotonic = time.monotonic()
            self._event(
                "TEST11_ACTIVE_WINDOW_START",
                model=model,
                test1_run=test1_run,
            )

            rows = run_test11_campaign(
                self,
                self.suite.get("cases", []) or [],
                test1_run=test1_run,
                started_monotonic=campaign_started_monotonic,
            )
            physical_calls = self._model_call_counts.get(store.run_id, 0)
            self._event(
                "TEST11_COMPLETE",
                model=model,
                observations=len(rows),
                physical_model_calls=physical_calls,
                test1_run=test1_run,
            )
            return self._finalize_run()
        except Exception as exc:
            store.write_json(
                "raw/runner-failure.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                producer="runner",
                stage="fatal",
            )
            self._event(
                "RUN_FAILED",
                failure="TEST11_ERROR",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return self._finalize_run()
        finally:
            if self.progress is not None and not self._progress_stopped:
                self.progress.stop_live(newline=True)
                self._progress_stopped = True

    def gpt20b_test12(
        self,
        model: str,
        *,
        pull: bool = False,
        seed_run: str | None = None,
        dry_run: bool = False,
        resume_run: str | None = None,
    ) -> Path:
        """Run the <=7h44 Test-1.2 collection stage of the model-to-harness compiler."""
        test_cfg = self.config.get("test12_campaign") or {}
        expected_calls = int(test_cfg.get("expected_calls", 5600))
        safety_cap = int(test_cfg.get("safety_call_cap", 14000))
        limits = self.config.setdefault("limits", {})
        limits["max_model_calls_per_run"] = max(
            int(limits.get("max_model_calls_per_run", 3000)),
            safety_cap,
        )

        self.progress = self._progress_factory(expected_calls + 2)
        self._progress_preflight_complete = False
        self._progress_stopped = False
        self.progress.start("preflight")
        self.progress.start_live()

        self.model = model
        if dry_run and resume_run:
            raise ValueError("dry-run cannot resume an existing Test 1.2 run")

        resume_state: dict[str, Any] | None = None
        if resume_run:
            run_id = str(resume_run)
            run_dir = self.results_root / run_id
            if not run_dir.is_dir():
                raise ValueError(f"Test 1.2 resume run does not exist: {run_id}")
            resume_state = load_test12_recovery(run_dir)
            checkpoint = resume_state.get("checkpoint") or {}
            checkpoint_model = checkpoint.get("model")
            if checkpoint_model and checkpoint_model != model:
                raise ValueError(
                    f"resume model mismatch: checkpoint={checkpoint_model} requested={model}"
                )
            snapshot_path = run_dir / "benchmark-snapshot.json"
            if snapshot_path.is_file():
                prior_suite = json.loads(snapshot_path.read_text(encoding="utf-8"))
                if prior_suite != self.suite:
                    raise ValueError(
                        "resume benchmark snapshot changed; this is a new onboarding event, not a recovery"
                    )
            run_id = str(resume_run)
        else:
            run_id = f"test1.2-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        self.store = EvidenceStore(self.results_root, run_id)
        store = self.store
        if resume_state:
            checkpoint = resume_state.get("checkpoint") or {}
            self._model_call_counts[run_id] = int(
                checkpoint.get("physical_model_calls_used") or 0
            )
            self._request_seq = 1_000_000 + self._model_call_counts[run_id]
            self._telemetry_seq = 1_000_000 + len(
                (store.run_dir / "telemetry.jsonl").read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
            ) if (store.run_dir / "telemetry.jsonl").is_file() else 1_000_000
            store.append_jsonl("test1.2-recovery-ledger.jsonl", {
                "schema_version": 1,
                "event": "RESUME_REQUESTED",
                "timestamp_utc": self._utc(),
                "same_run_id": run_id,
                "valid_atomic_observations_restored": len(resume_state.get("rows") or []),
                "damaged_atomic_records_quarantined": len(resume_state.get("issues") or []),
                "issues": copy.deepcopy(resume_state.get("issues") or []),
                "active_seconds_used_before_resume": float(
                    checkpoint.get("active_seconds_used") or 0.0
                ),
                "physical_model_calls_used_before_resume": int(
                    checkpoint.get("physical_model_calls_used") or 0
                ),
                "full_rerun": False,
            })
        else:
            store.write_json("resolved-config.json", self.config, producer="runner", stage="preflight")
            store.write_json("benchmark-snapshot.json", self.suite, producer="runner", stage="preflight")

        plan = build_test12_plan(self.suite.get("cases", []) or [], seed_run=seed_run)
        validate_test12_plan(plan)
        store.write_json("test1.2-plan.json", plan, producer="test1.2", stage="preflight")
        store.write_json(
            "fixture-partitions.json",
            {
                "schema_version": 1,
                "partitions": {
                    name: [str(case.get("id")) for case in rows]
                    for name, rows in partition_test12_cases(self.suite.get("cases", []) or []).items()
                },
            },
            producer="test1.2",
            stage="preflight",
        )
        self._event(
            "RUN_RESUME" if resume_state else "RUN_START",
            model=model,
            benchmark_version=self.suite.get("benchmark_version"),
            mode=(
                "gpt20b-test1.2-resume"
                if resume_state
                else "gpt20b-test1.2-dry-run"
                if dry_run
                else "gpt20b-test1.2"
            ),
            seed_run=seed_run,
            resume_run=run_id if resume_state else None,
        )

        if dry_run:
            source = (
                load_test11_source(
                    self.results_root,
                    seed_run,
                    self.suite.get("cases", []) or [],
                )
                if seed_run
                else fresh_model_source(self.suite.get("cases", []) or [])
            )
            mechanisms = build_intervention_bank(
                source,
                max_source_recipes=int(test_cfg.get("max_source_recipes", 8)),
            )
            store.write_json(
                "test1.2-dry-run-validation.json",
                {
                    "schema_version": 1,
                    "source": source.get("run_id"),
                    "fresh_model_source": source.get("run_id") == "FRESH-MODEL",
                    "planned_wall_seconds": plan["wall_clock_seconds"],
                    "planned_active_seconds": plan["active_model_seconds"],
                    "core_mechanism_count": plan["core_mechanism_count"],
                    "expanded_mechanism_count": len(mechanisms),
                    "mechanism_categories": sorted({str(row["category"]) for row in mechanisms}),
                    "improvement_surface": plan["improvement_surface"],
                    "coverage_floor_first": plan["adaptive_allocation"]["coverage_floor_first"],
                    "oracle_routing_prohibited": plan["adaptive_allocation"]["oracle_routing_prohibited"],
                    "protected_partitions": plan["prohibited_partitions"],
                    "protected_partitions_exposed": False,
                    "required_outputs": plan["required_outputs"],
                    "scope_boundaries": plan["scope_boundaries"],
                },
                producer="test1.2",
                stage="preflight",
            )
            self._progress_complete("preflight")
            self._progress_preflight_complete = True
            self._event(
                "TEST12_DRY_RUN_COMPLETE",
                model=model,
                seed_run=seed_run,
                expanded_mechanism_count=len(mechanisms),
                planned_wall_seconds=plan["wall_clock_seconds"],
            )
            return self._finalize_run()

        self._start_telemetry()
        try:
            source = (
                load_test11_source(
                    self.results_root,
                    seed_run,
                    self.suite.get("cases", []) or [],
                )
                if seed_run
                else fresh_model_source(self.suite.get("cases", []) or [])
            )
            store.write_json(
                "test1.2-source-preflight.json",
                {
                    "schema_version": 1,
                    "source_run": seed_run,
                    "source_observations": len(source.get("observations", [])),
                    "source_priority_items": len((source.get("priority_queue") or {}).get("queue", []) or []),
                    "source_ingredients": len((source.get("ingredient_registry") or {}).get("ingredients", []) or []),
                    "source_integrity": copy.deepcopy(source.get("source_integrity") or {}),
                    "expanded_mechanism_count": len(
                        build_intervention_bank(
                            source,
                            max_source_recipes=int(test_cfg.get("max_source_recipes", 8)),
                        )
                    ),
                },
                producer="test1.2",
                stage="preflight",
            )

            hardware = self.hardware_collector()
            store.write_json("hardware.json", hardware, producer="hardware", stage="preflight")

            version = self.runtime.version()
            version_refs = self._persist_control_exchange("runtime-version", version)
            tags = self.runtime.list_models()
            tags_refs = self._persist_control_exchange("model-list", tags)
            available = (
                self.runtime.model_available_in(tags, model)
                if hasattr(self.runtime, "model_available_in")
                else self.runtime.is_model_available(model)
            )

            pull_refs = None
            if not available and pull:
                pull_result = self.runtime.pull(model)
                pull_refs = self._persist_control_exchange("model-pull", pull_result)
                tags = self.runtime.list_models()
                tags_refs = self._persist_control_exchange("model-list-after-pull", tags)
                available = (
                    self.runtime.model_available_in(tags, model)
                    if hasattr(self.runtime, "model_available_in")
                    else self.runtime.is_model_available(model)
                )

            if not available:
                store.write_json(
                    "runtime.json",
                    {
                        "model": model,
                        "available": False,
                        "version": version.get("parsed"),
                        "model_size_bytes": None,
                        "evidence_refs": {"version": version_refs, "tags": tags_refs, "pull": pull_refs},
                    },
                    producer="runner",
                    stage="preflight",
                )
                self._event("RUN_FAILED", failure="MODEL_NOT_FOUND", model=model)
                return self._finalize_run()

            info = self.runtime.model_info(model)
            info_refs = self._persist_control_exchange("model-info", info)
            store.write_json(
                "runtime.json",
                {
                    "model": model,
                    "available": True,
                    "version": version.get("parsed"),
                    "model_size_bytes": self._find_model_size(tags, model),
                    "model_info": info.get("parsed"),
                    "evidence_refs": {"version": version_refs, "tags": tags_refs, "show": info_refs, "pull": pull_refs},
                },
                producer="runner",
                stage="preflight",
            )
            self._event("PREFLIGHT_COMPLETE", model=model, mode="gpt20b-test1.2")
            self._progress_complete("preflight")
            self._progress_preflight_complete = True

            campaign_started_monotonic = time.monotonic()
            self._event(
                "TEST12_ACTIVE_WINDOW_RESUME" if resume_state else "TEST12_ACTIVE_WINDOW_START",
                model=model,
                seed_run=seed_run,
                resume_run=run_id if resume_state else None,
            )
            rows = run_test12_campaign(
                self,
                self.suite.get("cases", []) or [],
                seed_run=seed_run,
                started_monotonic=campaign_started_monotonic,
                resume_state=resume_state,
            )
            physical_calls = self._model_call_counts.get(store.run_id, 0)
            self._event(
                "TEST12_COMPLETE",
                model=model,
                observations=len(rows),
                physical_model_calls=physical_calls,
                seed_run=seed_run,
            )
            return self._finalize_run()
        except Exception as exc:
            store.write_json(
                "raw/runner-failure.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                producer="runner",
                stage="fatal",
            )
            checkpoint_path = store.run_dir / "test1.2-recovery-checkpoint.json"
            checkpoint = {}
            if checkpoint_path.is_file():
                try:
                    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                except Exception:
                    checkpoint = {}
            checkpoint.update({
                "schema_version": 1,
                "recovery_policy": "NO_FULL_RERUN_ATOMIC_RESUME",
                "state": "PAUSED_RECOVERABLE",
                "run_id": store.run_id,
                "model": model,
                "physical_model_calls_used": int(
                    self._model_call_counts.get(store.run_id, 0)
                ),
                "full_rerun_allowed": False,
            })
            store.write_json_atomic(
                "test1.2-recovery-checkpoint.json",
                checkpoint,
                producer="runner",
                stage="recovery-checkpoint",
            )
            self._event(
                "TEST12_PAUSED_RECOVERABLE",
                error_type=type(exc).__name__,
                error=str(exc),
                resume_run=store.run_id,
                full_rerun_required=False,
            )
            return self._finalize_run()
        finally:
            if self.progress is not None and not self._progress_stopped:
                self.progress.stop_live(newline=True)
                self._progress_stopped = True

    def gpt20b_test12_tune(
        self,
        model: str,
        *,
        collection_run: str,
        pull: bool = False,
        dry_run: bool = False,
        resume_run: str | None = None,
    ) -> Path:
        """Tune/compile a model-specific harness from a completed Test-1.2 collection run."""
        tune_cfg = self.config.get("test12_tuning") or {}
        expected_calls = int(tune_cfg.get("expected_calls", 4200))
        safety_cap = int(tune_cfg.get("safety_call_cap", 10000))
        limits = self.config.setdefault("limits", {})
        limits["max_model_calls_per_run"] = max(
            int(limits.get("max_model_calls_per_run", 3000)),
            safety_cap,
        )

        self.progress = self._progress_factory(expected_calls + 2)
        self._progress_preflight_complete = False
        self._progress_stopped = False
        self.progress.start("preflight")
        self.progress.start_live()

        self.model = model
        if dry_run and resume_run:
            raise ValueError("dry-run cannot resume an existing Test 1.2 tuning run")

        resume_state: dict[str, Any] | None = None
        if resume_run:
            run_id = str(resume_run)
            run_dir = self.results_root / run_id
            if not run_dir.is_dir():
                raise ValueError(f"Test 1.2 tuning resume run does not exist: {run_id}")
            resume_state = load_tuning_recovery(run_dir)
            checkpoint = resume_state.get("checkpoint") or {}
            checkpoint_model = checkpoint.get("model")
            if checkpoint_model and checkpoint_model != model:
                raise ValueError(
                    f"resume model mismatch: checkpoint={checkpoint_model} requested={model}"
                )
            checkpoint_collection = checkpoint.get("collection_run")
            if checkpoint_collection and checkpoint_collection != collection_run:
                raise ValueError(
                    "resume collection source changed; this is a new onboarding event, not recovery"
                )
            snapshot_path = run_dir / "benchmark-snapshot.json"
            if snapshot_path.is_file():
                prior_suite = json.loads(snapshot_path.read_text(encoding="utf-8"))
                if prior_suite != self.suite:
                    raise ValueError(
                        "resume benchmark snapshot changed; this is a new onboarding event, not recovery"
                    )
        else:
            run_id = f"test1.2-tune-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        self.store = EvidenceStore(self.results_root, run_id)
        store = self.store
        if resume_state:
            checkpoint = resume_state.get("checkpoint") or {}
            self._model_call_counts[run_id] = int(
                checkpoint.get("physical_model_calls_used") or 0
            )
            self._request_seq = 2_000_000 + self._model_call_counts[run_id]
            self._telemetry_seq = 2_000_000 + len(
                (store.run_dir / "telemetry.jsonl").read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
            ) if (store.run_dir / "telemetry.jsonl").is_file() else 2_000_000
            store.append_jsonl("test1.2-tuning-recovery-ledger.jsonl", {
                "schema_version": 1,
                "event": "RESUME_REQUESTED",
                "timestamp_utc": self._utc(),
                "same_run_id": run_id,
                "valid_policy_observations_restored": len(resume_state.get("rows") or []),
                "damaged_policy_records_quarantined": len(resume_state.get("issues") or []),
                "issues": copy.deepcopy(resume_state.get("issues") or []),
                "active_seconds_used_before_resume": float(
                    checkpoint.get("active_seconds_used") or 0.0
                ),
                "physical_model_calls_used_before_resume": int(
                    checkpoint.get("physical_model_calls_used") or 0
                ),
                "winner_lock_sha256": checkpoint.get("winner_lock_sha256"),
                "full_rerun": False,
            })
        else:
            store.write_json("resolved-config.json", self.config, producer="runner", stage="preflight")
            store.write_json("benchmark-snapshot.json", self.suite, producer="runner", stage="preflight")

        plan = build_tuning_plan(self.suite.get("cases", []) or [], collection_run=collection_run)
        validate_tuning_plan(plan)
        store.write_json("test1.2-tuning-plan.json", plan, producer="test1.2-tuning", stage="preflight")
        store.write_json(
            "fixture-partitions.json",
            {
                "schema_version": 1,
                "partitions": {
                    name: [str(case.get("id")) for case in rows]
                    for name, rows in partition_test12_cases(self.suite.get("cases", []) or []).items()
                },
            },
            producer="test1.2-tuning",
            stage="preflight",
        )
        self._event(
            "RUN_RESUME" if resume_state else "RUN_START",
            model=model,
            benchmark_version=self.suite.get("benchmark_version"),
            mode=(
                "gpt20b-test1.2-tune-resume"
                if resume_state
                else "gpt20b-test1.2-tune-dry-run"
                if dry_run
                else "gpt20b-test1.2-tune"
            ),
            collection_run=collection_run,
            resume_run=run_id if resume_state else None,
        )

        if dry_run:
            collection = load_collection(self.results_root, collection_run)
            store.write_json(
                "test1.2-tuning-dry-run-validation.json",
                {
                    "schema_version": 1,
                    "collection_run": collection_run,
                    "collection_candidate_count": len((collection.get("registry") or {}).get("candidates", [])),
                    "all_declared_collection_controls_tested": bool((collection.get("coverage") or {}).get("all_declared_candidates_tested")),
                    "planned_wall_seconds": plan["wall_clock_seconds"],
                    "planned_active_seconds": plan["active_model_seconds"],
                    "allowed_partitions": plan["allowed_partitions"],
                    "prohibited_partitions": plan["prohibited_partitions"],
                    "total_two_run_hard_ceiling_seconds": plan["total_two_run_hard_ceiling_seconds"],
                    "required_outputs": plan["required_outputs"],
                },
                producer="test1.2-tuning",
                stage="preflight",
            )
            self._progress_complete("preflight")
            self._progress_preflight_complete = True
            self._event("TEST12_TUNING_DRY_RUN_COMPLETE", model=model, collection_run=collection_run)
            return self._finalize_run()

        self._start_telemetry()
        try:
            collection = load_collection(self.results_root, collection_run)
            store.write_json(
                "test1.2-tuning-source-preflight.json",
                {
                    "schema_version": 1,
                    "collection_run": collection_run,
                    "collection_candidate_count": len((collection.get("registry") or {}).get("candidates", [])),
                    "all_declared_collection_controls_tested": bool((collection.get("coverage") or {}).get("all_declared_candidates_tested")),
                    "collection_tuning_examples": len(collection.get("corpus") or []),
                },
                producer="test1.2-tuning",
                stage="preflight",
            )

            hardware = self.hardware_collector()
            store.write_json("hardware.json", hardware, producer="hardware", stage="preflight")

            version = self.runtime.version()
            version_refs = self._persist_control_exchange("runtime-version", version)
            tags = self.runtime.list_models()
            tags_refs = self._persist_control_exchange("model-list", tags)
            available = (
                self.runtime.model_available_in(tags, model)
                if hasattr(self.runtime, "model_available_in")
                else self.runtime.is_model_available(model)
            )
            pull_refs = None
            if not available and pull:
                pull_result = self.runtime.pull(model)
                pull_refs = self._persist_control_exchange("model-pull", pull_result)
                tags = self.runtime.list_models()
                tags_refs = self._persist_control_exchange("model-list-after-pull", tags)
                available = (
                    self.runtime.model_available_in(tags, model)
                    if hasattr(self.runtime, "model_available_in")
                    else self.runtime.is_model_available(model)
                )
            if not available:
                store.write_json(
                    "runtime.json",
                    {
                        "model": model,
                        "available": False,
                        "version": version.get("parsed"),
                        "model_size_bytes": None,
                        "evidence_refs": {"version": version_refs, "tags": tags_refs, "pull": pull_refs},
                    },
                    producer="runner",
                    stage="preflight",
                )
                self._event("RUN_FAILED", failure="MODEL_NOT_FOUND", model=model)
                return self._finalize_run()

            info = self.runtime.model_info(model)
            info_refs = self._persist_control_exchange("model-info", info)
            store.write_json(
                "runtime.json",
                {
                    "model": model,
                    "available": True,
                    "version": version.get("parsed"),
                    "model_size_bytes": self._find_model_size(tags, model),
                    "model_info": info.get("parsed"),
                    "evidence_refs": {"version": version_refs, "tags": tags_refs, "show": info_refs, "pull": pull_refs},
                },
                producer="runner",
                stage="preflight",
            )
            self._event("PREFLIGHT_COMPLETE", model=model, mode="gpt20b-test1.2-tune")
            self._progress_complete("preflight")
            self._progress_preflight_complete = True

            started = time.monotonic()
            self._event("TEST12_TUNING_ACTIVE_WINDOW_START", model=model, collection_run=collection_run)
            rows = run_test12_tuning(
                self,
                self.suite.get("cases", []) or [],
                collection_run=collection_run,
                started_monotonic=started,
                resume_state=resume_state,
            )
            physical_calls = self._model_call_counts.get(store.run_id, 0)
            self._event(
                "TEST12_TUNING_COMPLETE",
                model=model,
                observations=len(rows),
                physical_model_calls=physical_calls,
                collection_run=collection_run,
            )
            return self._finalize_run()
        except Exception as exc:
            store.write_json(
                "raw/runner-failure.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                producer="runner",
                stage="fatal",
            )
            checkpoint_path = store.run_dir / "test1.2-tuning-recovery-checkpoint.json"
            checkpoint = {}
            if checkpoint_path.is_file():
                try:
                    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                except Exception:
                    checkpoint = {}
            checkpoint.update({
                "schema_version": 1,
                "recovery_policy": "NO_FULL_RERUN_ATOMIC_RESUME",
                "state": "PAUSED_RECOVERABLE",
                "run_id": store.run_id,
                "model": model,
                "collection_run": collection_run,
                "physical_model_calls_used": int(
                    self._model_call_counts.get(store.run_id, 0)
                ),
                "full_rerun_allowed": False,
            })
            store.write_json_atomic(
                "test1.2-tuning-recovery-checkpoint.json",
                checkpoint,
                producer="runner",
                stage="recovery-checkpoint",
            )
            self._event(
                "TEST12_TUNING_PAUSED_RECOVERABLE",
                error_type=type(exc).__name__,
                error=str(exc),
                resume_run=store.run_id,
                winner_lock_sha256=checkpoint.get("winner_lock_sha256"),
                full_rerun_required=False,
            )
            return self._finalize_run()
        finally:
            if self.progress is not None and not self._progress_stopped:
                self.progress.stop_live(newline=True)
                self._progress_stopped = True

    def gpt20b_test2(
        self,
        model: str,
        *,
        pull: bool = False,
        test1_run: str | None = None,
        dry_run: bool = False,
    ) -> Path:
        """Run the frozen seven-hour GPT-20B Test-2 finalization campaign."""
        campaign_started_monotonic = time.monotonic()
        test_cfg = self.config.get("test2_campaign") or {}
        expected_calls = int(test_cfg.get("expected_calls", 4100))
        safety_cap = int(test_cfg.get("safety_call_cap", 10000))
        limits = self.config.setdefault("limits", {})
        limits["max_model_calls_per_run"] = max(
            int(limits.get("max_model_calls_per_run", 3000)),
            safety_cap,
        )

        self.progress = self._progress_factory(expected_calls + 2)
        self._progress_preflight_complete = False
        self._progress_stopped = False
        self.progress.start("preflight")
        self.progress.start_live()

        self.model = model
        run_id = f"test2-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.store = EvidenceStore(self.results_root, run_id)
        store = self.store
        store.write_json("resolved-config.json", self.config, producer="runner", stage="preflight")
        store.write_json("benchmark-snapshot.json", self.suite, producer="runner", stage="preflight")
        plan = build_test2_plan(self.suite.get("cases", []) or [], test1_run=test1_run)
        validate_test2_plan(plan)
        store.write_json("test2-plan.json", plan, producer="test2", stage="preflight")
        self._event(
            "RUN_START",
            model=model,
            benchmark_version=self.suite.get("benchmark_version"),
            mode="gpt20b-test2-dry-run" if dry_run else "gpt20b-test2",
            test1_run=test1_run,
        )

        if dry_run:
            if test1_run:
                handoff = load_test1_handoff(
                    self.results_root,
                    test1_run,
                    self.suite.get("cases", []) or [],
                )
            else:
                handoff = synthetic_test1_handoff(self.suite.get("cases", []) or [])
                store.write_json(
                    "synthetic-test1-handoff-template.json",
                    handoff,
                    producer="test2",
                    stage="preflight",
                )
            store.write_json(
                "test2-dry-run-validation.json",
                {
                    "schema_version": 1,
                    "source": handoff.get("run_id"),
                    "synthetic_source": bool(handoff.get("synthetic")),
                    "planned_wall_seconds": plan["wall_clock_seconds"],
                    "planned_active_seconds": plan["active_model_seconds"],
                    "required_outputs": plan["required_outputs"],
                    "test3_protected_count": plan["partition_counts"]["TEST3_PROTECTED"],
                    "test3_protected_exposed": False,
                    "finalization_contract_complete": all(plan["finalization_contract"].values()),
                },
                producer="test2",
                stage="preflight",
            )
            self._progress_complete("preflight")
            self._progress_preflight_complete = True
            self._event(
                "TEST2_DRY_RUN_COMPLETE",
                model=model,
                test1_run=test1_run,
                synthetic_source=bool(handoff.get("synthetic")),
                planned_wall_seconds=plan["wall_clock_seconds"],
            )
            return self._finalize_run()

        if not test1_run:
            self._event("RUN_FAILED", failure="TEST1_RUN_REQUIRED", model=model)
            return self._finalize_run()

        self._start_telemetry()
        try:
            # Fail before touching the model if the Test-1 handoff is incomplete,
            # partition-drifted, or would expose protected fixtures.
            load_test1_handoff(
                self.results_root,
                test1_run,
                self.suite.get("cases", []) or [],
            )

            hardware = self.hardware_collector()
            store.write_json("hardware.json", hardware, producer="hardware", stage="preflight")

            version = self.runtime.version()
            version_refs = self._persist_control_exchange("runtime-version", version)
            tags = self.runtime.list_models()
            tags_refs = self._persist_control_exchange("model-list", tags)
            available = (
                self.runtime.model_available_in(tags, model)
                if hasattr(self.runtime, "model_available_in")
                else self.runtime.is_model_available(model)
            )

            pull_refs = None
            if not available and pull:
                pull_result = self.runtime.pull(model)
                pull_refs = self._persist_control_exchange("model-pull", pull_result)
                tags = self.runtime.list_models()
                tags_refs = self._persist_control_exchange("model-list-after-pull", tags)
                available = (
                    self.runtime.model_available_in(tags, model)
                    if hasattr(self.runtime, "model_available_in")
                    else self.runtime.is_model_available(model)
                )

            if not available:
                store.write_json(
                    "runtime.json",
                    {
                        "model": model,
                        "available": False,
                        "version": version.get("parsed"),
                        "model_size_bytes": None,
                        "evidence_refs": {
                            "version": version_refs,
                            "tags": tags_refs,
                            "pull": pull_refs,
                        },
                    },
                    producer="runner",
                    stage="preflight",
                )
                self._event("RUN_FAILED", failure="MODEL_NOT_FOUND", model=model)
                return self._finalize_run()

            info = self.runtime.model_info(model)
            info_refs = self._persist_control_exchange("model-info", info)
            store.write_json(
                "runtime.json",
                {
                    "model": model,
                    "available": True,
                    "version": version.get("parsed"),
                    "model_size_bytes": self._find_model_size(tags, model),
                    "model_info": info.get("parsed"),
                    "evidence_refs": {
                        "version": version_refs,
                        "tags": tags_refs,
                        "show": info_refs,
                        "pull": pull_refs,
                    },
                },
                producer="runner",
                stage="preflight",
            )
            self._event("PREFLIGHT_COMPLETE", model=model, mode="gpt20b-test2")
            self._progress_complete("preflight")
            self._progress_preflight_complete = True

            rows = run_test2_campaign(
                self,
                self.suite.get("cases", []) or [],
                test1_run=test1_run,
                started_monotonic=campaign_started_monotonic,
            )
            physical_calls = self._model_call_counts.get(store.run_id, 0)
            self._event(
                "TEST2_COMPLETE",
                model=model,
                observations=len(rows),
                physical_model_calls=physical_calls,
                test1_run=test1_run,
            )
            return self._finalize_run()
        except Exception as exc:
            store.write_json(
                "raw/runner-failure.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                producer="runner",
                stage="fatal",
            )
            self._event(
                "RUN_FAILED",
                failure="TEST2_ERROR",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return self._finalize_run()
        finally:
            if self.progress is not None and not self._progress_stopped:
                self.progress.stop_live(newline=True)
                self._progress_stopped = True

    def _finalize_run(self) -> Path:
        assert self.store is not None
        if self.progress is not None and not self._progress_preflight_complete:
            self._progress_complete("preflight")
            self._progress_preflight_complete = True

        if self.progress is not None:
            expected_before_finalize = max(0, int(self.progress.total_tasks) - 1)
            if int(getattr(self.progress, "done", 0)) < expected_before_finalize:
                old_total = int(self.progress.total_tasks)
                self.progress.total_tasks = int(getattr(self.progress, "done", 0)) + 1
                self._record_progress(
                    "plan_adjusted",
                    "finalize",
                    old_total=old_total,
                    new_total=int(self.progress.total_tasks),
                    reason="bounded stage stopped before all planned tasks executed",
                )
            self._progress_begin("finalize")

        self._stop_telemetry()

        frontiers_path = self.store.run_dir / "capability-frontiers.json"
        experiments_path = self.store.run_dir / "experiments.jsonl"
        if frontiers_path.is_file() and experiments_path.is_file():
            frontiers = json.loads(frontiers_path.read_text(encoding="utf-8"))
            rows = [
                json.loads(raw_line)
                for raw_line in experiments_path.read_text(encoding="utf-8").splitlines()
                if raw_line.strip()
            ]
            cost_map, value_map = _build_cost_value_outputs(
                str(self.model),
                rows,
                frontiers,
                self.store.run_dir,
            )
            self.store.write_json(
                "cost-map.json",
                cost_map,
                producer="cost-value",
                stage="report",
            )
            self.store.write_json(
                "value-map.json",
                value_map,
                producer="cost-value",
                stage="report",
            )

        self._event("RUN_END", model=self.model)
        summary = build_summary(self.store.run_dir)
        self.store.write_json("summary.json", summary, producer="report", stage="report")
        self.store.write_raw(
            "report.md",
            render_report(summary),
            producer="report",
            stage="report",
            media_type="text/markdown",
        )

        if self.progress is not None:
            self.progress.complete_task("complete")
            self._record_progress("complete", "complete")
            self.progress.stop_live(newline=True)
            self._progress_stopped = True

        self.store.finalize_manifest(
            metadata={
                "model": self.model,
                "benchmark_version": self.suite.get("benchmark_version"),
            }
        )
        return self.store.run_dir
