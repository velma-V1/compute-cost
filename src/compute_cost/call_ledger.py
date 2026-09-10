"""Authoritative per-run model-call accounting.

Authorization happens before runtime inference. A rejected call never increments
``calls_used`` and therefore can be proven not to have reached the model.
"""

from __future__ import annotations

from collections import Counter
from threading import RLock
from typing import Any


class CallBudgetExceeded(RuntimeError):
    """Raised when a model call is requested after the hard ceiling is full."""


class CallLedger:
    def __init__(self, limit: int = 700) -> None:
        if int(limit) <= 0:
            raise ValueError("call limit must be positive")
        self.limit = int(limit)
        self._used = 0
        self._rejected = 0
        self._categories: Counter[str] = Counter()
        self._families: Counter[str] = Counter()
        self._scenarios: Counter[str] = Counter()
        self._lock = RLock()

    def authorize(
        self,
        category: str,
        *,
        family: str | None = None,
        scenario: str | None = None,
    ) -> int:
        with self._lock:
            if self._used >= self.limit:
                self._rejected += 1
                raise CallBudgetExceeded(
                    f"model call budget exhausted: {self._used}/{self.limit}"
                )
            self._used += 1
            self._categories[str(category)] += 1
            if family:
                self._families[str(family)] += 1
            if scenario:
                self._scenarios[str(scenario)] += 1
            return self._used

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "hard_ceiling": self.limit,
                "calls_used": self._used,
                "remaining_calls": self.limit - self._used,
                "rejected_call_attempts": self._rejected,
                "categories": dict(sorted(self._categories.items())),
                "per_family": dict(sorted(self._families.items())),
                "per_scenario": dict(sorted(self._scenarios.items())),
            }
