"""Progress-aware public runner layered over the stable benchmark core."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .hardware import collect_hardware_snapshot
from .progress import ProgressDisplay
from .report import build_summary, render_report
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
            1  # preflight
            + 1  # cold start
            + 1  # warmup
            + len(self.suite.get("cases", []) or [])
            + len(limits.get("context_schedule", []) or [])
            + max(0, int(limits.get("sustained_iterations", 0)))
            + 1  # finalize
        )

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
