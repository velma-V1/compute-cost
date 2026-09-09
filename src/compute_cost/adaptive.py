"""Pure adaptive boundary-search controllers."""

from __future__ import annotations

from collections import Counter
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


@dataclass(frozen=True)
class DifficultyObservation:
    level: int
    passed: bool | None
    valid_for_capability: bool = True


@dataclass(frozen=True)
class DifficultyDecision:
    action: str
    level: int | None
    reason: str


class AdaptiveDifficultyController:
    """Search a family-local L0-L10 pass/fail boundary with bounded replication."""

    def __init__(
        self,
        *,
        min_level: int = 0,
        max_level: int = 10,
        anchor_level: int = 1,
        jump: int = 3,
        boundary_repeats: int = 5,
    ) -> None:
        if not 0 <= min_level <= anchor_level <= max_level <= 10:
            raise ValueError("difficulty bounds must satisfy 0 <= min <= anchor <= max <= 10")
        if jump < 1:
            raise ValueError("jump must be positive")
        if boundary_repeats < 1:
            raise ValueError("boundary_repeats must be positive")
        self.min_level = min_level
        self.max_level = max_level
        self.anchor_level = anchor_level
        self.jump = jump
        self.boundary_repeats = boundary_repeats

    def next(self, observations: list[DifficultyObservation]) -> DifficultyDecision:
        if not observations:
            return DifficultyDecision(
                "PROBE",
                self.anchor_level,
                "establish family difficulty anchor",
            )

        latest = observations[-1]
        if not latest.valid_for_capability or not isinstance(latest.passed, bool):
            return DifficultyDecision(
                "REPLICATE",
                latest.level,
                "retry invalid difficulty observation",
            )

        valid = [
            item
            for item in observations
            if item.valid_for_capability and isinstance(item.passed, bool)
        ]
        counts = Counter(item.level for item in valid)
        pass_counts = Counter(item.level for item in valid if item.passed is True)
        fail_counts = Counter(item.level for item in valid if item.passed is False)
        pass_levels = sorted(pass_counts)
        fail_levels = sorted(fail_counts)

        mixed_levels = sorted(set(pass_levels) & set(fail_levels))
        if mixed_levels:
            level = mixed_levels[0]
            if counts[level] < self.boundary_repeats:
                return DifficultyDecision(
                    "REPLICATE",
                    level,
                    "reproduce mixed difficulty behavior",
                )
            return DifficultyDecision(
                "STOP",
                None,
                "mixed difficulty behavior reproduced",
            )

        if pass_levels and not fail_levels:
            highest_pass = pass_levels[-1]
            if highest_pass < self.max_level:
                return DifficultyDecision(
                    "PROBE",
                    min(self.max_level, highest_pass + self.jump),
                    "jump upward after passing difficulty",
                )
            if pass_counts[highest_pass] < self.boundary_repeats:
                return DifficultyDecision(
                    "REPLICATE",
                    highest_pass,
                    "reproduce maximum difficulty pass",
                )
            return DifficultyDecision(
                "STOP",
                None,
                "maximum difficulty pass reproduced",
            )

        if fail_levels and not pass_levels:
            lowest_fail = fail_levels[0]
            if lowest_fail > self.min_level:
                return DifficultyDecision(
                    "PROBE",
                    max(self.min_level, lowest_fail - self.jump),
                    "search downward after failing difficulty",
                )
            if fail_counts[lowest_fail] < self.boundary_repeats:
                return DifficultyDecision(
                    "REPLICATE",
                    lowest_fail,
                    "reproduce minimum difficulty failure",
                )
            return DifficultyDecision(
                "STOP",
                None,
                "minimum difficulty failure reproduced",
            )

        monotonic_pairs = [
            (pass_level, fail_level)
            for pass_level in pass_levels
            for fail_level in fail_levels
            if pass_level < fail_level
        ]
        if not monotonic_pairs:
            level = latest.level
            if counts[level] < self.boundary_repeats:
                return DifficultyDecision(
                    "REPLICATE",
                    level,
                    "reproduce non-monotonic difficulty observation",
                )
            return DifficultyDecision(
                "STOP",
                None,
                "non-monotonic difficulty observations reproduced",
            )

        lower_pass, upper_fail = min(
            monotonic_pairs,
            key=lambda pair: (pair[1] - pair[0], -pair[0]),
        )
        gap = upper_fail - lower_pass
        if gap > 1:
            candidate = lower_pass + gap // 2
            if candidate <= lower_pass:
                candidate = lower_pass + 1
            return DifficultyDecision(
                "PROBE",
                candidate,
                "bisect difficulty pass/fail bracket",
            )

        if counts[lower_pass] < self.boundary_repeats:
            return DifficultyDecision(
                "REPLICATE",
                lower_pass,
                "reproduce passing side of difficulty boundary",
            )
        if counts[upper_fail] < self.boundary_repeats:
            return DifficultyDecision(
                "REPLICATE",
                upper_fail,
                "reproduce failing side of difficulty boundary",
            )
        return DifficultyDecision(
            "STOP",
            None,
            "adjacent difficulty boundary reproduced",
        )
