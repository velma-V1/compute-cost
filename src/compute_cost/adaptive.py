"""Pure adaptive reasoning-budget boundary search."""

from __future__ import annotations

from dataclasses import dataclass

PASS = {"ANSWER_CORRECT"}
TRUNCATION = {"THINK_TRUNCATED", "ANSWER_TRUNCATED"}
SEMANTIC_FAILURE = {"ANSWER_WRONG", "FORMAT_FAILURE", "TOOL_FAILURE", "CONTEXT_FAILURE"}
INVALID = {"SCORER_DEFECT", "TEST_DEFECT", "CAPTURE_GAP", "RUNTIME_FAILURE", "TIMEOUT", "RESOURCE_LIMIT"}


@dataclass(frozen=True)
class BudgetObservation:
    budget: int
    result_class: str


@dataclass(frozen=True)
class BudgetDecision:
    action: str
    budget: int | None
    reason: str


class AdaptiveBudgetController:
    def __init__(
        self,
        *,
        initial_budget: int,
        min_budget: int,
        max_budget: int,
        granularity: int,
        boundary_repeats: int,
    ) -> None:
        self.initial_budget = initial_budget
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.granularity = granularity
        self.boundary_repeats = boundary_repeats

    def _round(self, value: int) -> int:
        rounded = (value // self.granularity) * self.granularity
        return max(self.min_budget, min(self.max_budget, rounded))

    def next(self, observations: list[BudgetObservation]) -> BudgetDecision:
        if not observations:
            return BudgetDecision("PROBE", self.initial_budget, "establish thinking-on baseline")
        if observations[-1].result_class in INVALID:
            return BudgetDecision("STOP", None, "invalid experiment observation")

        passes = [item for item in observations if item.result_class in PASS]
        failures = [item for item in observations if item.result_class not in PASS]
        if not passes:
            latest = observations[-1]
            if latest.result_class in TRUNCATION and latest.budget < self.max_budget:
                return BudgetDecision(
                    "PROBE",
                    min(self.max_budget, latest.budget * 2),
                    "truncation justifies more generation headroom",
                )

            semantic = [item for item in observations if item.result_class in SEMANTIC_FAILURE]
            if len(semantic) == 1 and latest.result_class in SEMANTIC_FAILURE and latest.budget < self.max_budget:
                return BudgetDecision(
                    "PROBE",
                    min(self.max_budget, latest.budget * 2),
                    "single diagnostic budget escalation for semantic failure",
                )
            if len(semantic) >= 2:
                return BudgetDecision(
                    "STOP",
                    None,
                    "semantic failure persisted after diagnostic budget escalation",
                )
            return BudgetDecision("STOP", None, "failure does not implicate generation budget")

        minimum_pass = min(item.budget for item in passes)
        lower_failures = [item.budget for item in failures if item.budget < minimum_pass]
        if not lower_failures:
            candidate = self._round(minimum_pass // 2)
            if candidate != minimum_pass and not any(item.budget == candidate for item in observations):
                return BudgetDecision("PROBE", candidate, "search below current passing budget")

        lower_fail = max(lower_failures) if lower_failures else self.min_budget - self.granularity
        if minimum_pass - lower_fail > self.granularity:
            candidate = self._round(lower_fail + (minimum_pass - lower_fail) // 2)
            candidate = max(lower_fail + self.granularity, min(minimum_pass - self.granularity, candidate))
            return BudgetDecision("PROBE", candidate, "bisect fail/pass budget bracket")

        pass_count = sum(
            1
            for item in observations
            if item.budget == minimum_pass and item.result_class in PASS
        )
        if pass_count < self.boundary_repeats:
            return BudgetDecision("REPLICATE", minimum_pass, "reproduce minimum passing boundary")
        return BudgetDecision("STOP", None, "minimum passing boundary reproduced")
