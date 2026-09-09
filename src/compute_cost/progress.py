from __future__ import annotations

import shutil
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Callable, TextIO


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressDisplay:
    def __init__(
        self,
        total_tasks: int,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = datetime.now,
        width_getter: Callable[[], int] | None = None,
        stream: TextIO = sys.stdout,
        refresh_s: float = 0.5,
    ) -> None:
        self.total_tasks = max(1, int(total_tasks))
        self.monotonic = monotonic
        self.wall_clock = wall_clock
        self.width_getter = width_getter or (lambda: shutil.get_terminal_size((120, 20)).columns)
        self.stream = stream
        self.refresh_s = max(0.1, float(refresh_s))
        self.done = 0
        self.current_task = "starting"
        self.started_at: float | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, task: str = "starting") -> None:
        with self._lock:
            if self.started_at is None:
                self.started_at = self.monotonic()
            self.current_task = task

    def begin_task(self, task: str) -> None:
        with self._lock:
            if self.started_at is None:
                self.started_at = self.monotonic()
            self.current_task = task

    def complete_task(self, next_task: str | None = None) -> None:
        with self._lock:
            if self.started_at is None:
                self.started_at = self.monotonic()
            self.done = min(self.total_tasks, self.done + 1)
            if next_task is not None:
                self.current_task = next_task
            elif self.done >= self.total_tasks:
                self.current_task = "complete"

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            now = self.monotonic()
            elapsed = max(0.0, now - self.started_at) if self.started_at is not None else 0.0
            left = max(0, self.total_tasks - self.done)
            percent = (self.done / self.total_tasks) * 100.0
            eta = 0.0 if left == 0 else (elapsed / self.done * left if self.done > 0 else None)
            finish = self.wall_clock() + timedelta(seconds=eta or 0.0) if eta is not None else None
            return {
                "done": self.done,
                "total": self.total_tasks,
                "left": left,
                "percent": percent,
                "current_task": self.current_task,
                "elapsed_s": elapsed,
                "eta_s": eta,
                "finish": finish,
            }

    def render(self) -> str:
        snap = self.snapshot()
        width = max(24, int(self.width_getter()))
        pct = int(round(float(snap["percent"])))
        eta = "--:--:--" if snap["eta_s"] is None else format_duration(float(snap["eta_s"]))
        finish = "--:--:--" if snap["finish"] is None else snap["finish"].strftime("%H:%M:%S")
        task = str(snap["current_task"])
        core = (
            f"{pct}% | {snap['done']}/{snap['total']} done | {snap['left']} left | {task} | "
            f"elapsed {format_duration(float(snap['elapsed_s']))} | ETA {eta} | finish {finish}"
        )
        if len(core) >= width:
            compact = (
                f"{pct}% | {snap['done']}/{snap['total']} done | {snap['left']} left | "
                f"elapsed {format_duration(float(snap['elapsed_s']))} | ETA {eta}"
            )
            if len(compact) >= width:
                return compact[:width]
            available = width - len(compact) - 3
            if available > 4:
                compact += " | " + task[:available]
            return compact[:width]

        bar_room = width - len(core) - 3
        if bar_room < 8:
            return core[:width]
        filled = int(round(bar_room * float(snap["percent"]) / 100.0))
        bar = "█" * filled + "░" * max(0, bar_room - filled)
        return f"[{bar}] {core}"[:width]

    def write(self) -> None:
        line = self.render()
        self.stream.write("\r" + line)
        self.stream.flush()

    def start_live(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def worker() -> None:
            while not self._stop.wait(self.refresh_s):
                self.write()

        self.write()
        self._thread = threading.Thread(target=worker, name="compute-cost-progress", daemon=True)
        self._thread.start()

    def stop_live(self, *, newline: bool = True) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.refresh_s * 2 + 0.2)
        self.write()
        if newline:
            self.stream.write("\n")
            self.stream.flush()
