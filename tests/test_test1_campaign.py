from __future__ import annotations

import json
from pathlib import Path

from compute_cost.config import load_config
from compute_cost.runner import BenchmarkRunner
from compute_cost.test1_campaign import (
    ACTIVE_SECONDS,
    HARD_SECONDS,
    build_test1_plan,
    build_treatment_messages,
    effect_summary,
    partition_cases,
    validate_test1_plan,
)


def _cases(count: int = 400) -> list[dict]:
    return [
        {
            "id": f"fixture-{index:04d}",
            "family_id": f"family-{index % 8}",
            "category": f"family-{index % 8}",
            "difficulty_level": index % 11,
            "prompt": f"Return {index}",
            "scorer": "exact",
            "expected": str(index),
            "timeout_s": 120,
        }
        for index in range(count)
    ]


def test_plan_is_exactly_seven_hours_and_protects_future_partitions():
    cases = _cases()
    plan = build_test1_plan(cases)
    validate_test1_plan(plan)

    assert plan["wall_clock_seconds"] == HARD_SECONDS == 7 * 3600
    assert plan["active_model_seconds"] == ACTIVE_SECONDS == (6 * 3600 + 50 * 60)
    assert sum(row["seconds"] for row in plan["phases"]) == ACTIVE_SECONDS
    assert plan["prohibited_partitions"] == ["TEST2_BLIND", "TEST3_PROTECTED"]


def test_partitioning_is_deterministic_disjoint_and_complete():
    cases = _cases()
    first = partition_cases(cases)
    second = partition_cases(list(reversed(cases)))

    first_ids = {name: {row["id"] for row in rows} for name, rows in first.items()}
    second_ids = {name: {row["id"] for row in rows} for name, rows in second.items()}
    assert first_ids == second_ids

    union = set().union(*first_ids.values())
    assert union == {row["id"] for row in cases}
    names = list(first_ids)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert first_ids[left].isdisjoint(first_ids[right])


def test_treatment_can_change_true_message_placement_without_changing_fixture():
    case = _cases(1)[0]
    original = dict(case)

    system_messages = build_treatment_messages(
        case,
        ["ING-003"],
        placement="system",
    )
    suffix_messages = build_treatment_messages(
        case,
        ["ING-003"],
        placement="suffix",
    )

    assert system_messages[0]["role"] == "system"
    assert system_messages[1] == {"role": "user", "content": case["prompt"]}
    assert suffix_messages[0]["role"] == "user"
    assert suffix_messages[0]["content"].startswith(case["prompt"])
    assert case == original


def test_effect_classifier_uses_noise_normalized_evidence():
    strong = effect_summary([1.0] * 12, 0.1, seed_key="strong")
    harmful = effect_summary([-1.0] * 12, 0.1, seed_key="harmful")
    null = effect_summary([0.0] * 12, 0.1, seed_key="null")

    assert strong["classification"] == "STRONG"
    assert strong["win_rate"] == 1.0
    assert harmful["classification"] == "HARMFUL"
    assert null["classification"] == "NULL"


class _Runtime:
    def __init__(self):
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("dry run must not invoke the model")


class _Progress:
    def __init__(self, total: int):
        self.total_tasks = total
        self.done = 0

    def start(self, task: str):
        return None

    def start_live(self):
        return None

    def stop_live(self, newline: bool = True):
        return None

    def begin_task(self, task: str):
        return None

    def complete_task(self, task: str):
        self.done += 1

    def snapshot(self):
        return {"done": self.done, "total": self.total_tasks}

    def render(self):
        return ""


def test_runner_dry_run_materializes_plan_with_zero_model_calls(tmp_path: Path):
    runtime = _Runtime()
    config = load_config()
    config["telemetry"]["background"] = False
    suite = {
        "benchmark_version": "test1-dry-run",
        "cases": _cases(),
    }
    runner = BenchmarkRunner(
        runtime,
        config,
        suite,
        results_root=tmp_path,
        hardware_collector=lambda: {},
        progress_factory=lambda total: _Progress(total),
    )

    run_dir = runner.gpt20b_test1("gpt-oss:20b", dry_run=True)

    assert runtime.calls == 0
    assert (run_dir / "test1-plan.json").is_file()
    assert (run_dir / "fixture-partitions.json").is_file()
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row["type"] == "TEST1_DRY_RUN_COMPLETE" for row in events)
    assert not any(row["type"] == "RUN_FAILED" for row in events)
