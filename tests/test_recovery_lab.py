from compute_cost.experiments import ExperimentSpec


def spec(experiment_id, *, level=5, effort="medium", prompt_variant="base", recovery_level=None):
    return ExperimentSpec(
        experiment_id=experiment_id,
        parent_experiment_id=None,
        task_id="formal_logic_deduction",
        task_family="formal_logic_deduction",
        difficulty_level=level,
        hypothesis="baseline",
        changed_variable="baseline",
        thinking_mode=True,
        reasoning_effort=effort,
        generation_budget=256,
        context_request=None,
        temperature=0.0,
        seed=42,
        prompt_variant=prompt_variant,
        recovery_level=recovery_level,
    )


def row(experiment_id, result_class, *, effort="medium", level=5, prompt_variant="base", recovery_level=None):
    return {
        "experiment": spec(
            experiment_id,
            level=level,
            effort=effort,
            prompt_variant=prompt_variant,
            recovery_level=recovery_level,
        ).to_dict(),
        "classification": {
            "result_class": result_class,
            "valid_for_capability": True,
        },
        "score": 1.0 if result_class == "ANSWER_CORRECT" else 0.0,
        "status": "PASS" if result_class == "ANSWER_CORRECT" else "FAIL",
        "metrics": {"total_duration_ns": 100, "eval_count": 10},
        "phase_metrics": {"thinking_span_ns": 40, "thinking_chars": 20},
    }


def frontier(valid_count=3):
    return {
        "schema_version": 1,
        "families": {
            "formal_logic_deduction": {
                "reliable_floor": 4,
                "first_failure_level": 5,
                "levels": [
                    {"level": 4, "label": "reliable", "valid_count": 3, "pass_count": 3, "fail_count": 0},
                    {"level": 5, "label": "failure", "valid_count": valid_count, "pass_count": 0, "fail_count": valid_count},
                ],
            }
        },
    }


def fixture():
    return {
        "id": "logic-L5",
        "family_id": "formal_logic_deduction",
        "category": "formal_logic_deduction",
        "difficulty_level": 5,
        "prompt": "If A implies B and A is true, return exactly B.",
        "scorer": "exact",
        "expected": "B",
    }


def test_recovery_target_requires_reproduced_failure_boundary():
    from compute_cost.recovery_lab import select_recovery_targets

    selected = select_recovery_targets(frontier(valid_count=3), boundary_repeats=3)
    assert selected == {"formal_logic_deduction": 5}
    assert select_recovery_targets(frontier(valid_count=2), boundary_repeats=3) == {}


def test_recovery_interventions_are_isolated_and_do_not_embed_oracle():
    from compute_cost.recovery_lab import recovery_intervention

    signature = {"subtype": "logic_error", "causal_claim": False}
    r3 = recovery_intervention("R3", "formal_logic_deduction", signature)
    r4 = recovery_intervention("R4", "formal_logic_deduction", signature)
    r5 = recovery_intervention("R5", "formal_logic_deduction", signature)
    r6 = recovery_intervention("R6", "formal_logic_deduction", signature)
    r7 = recovery_intervention("R7", "formal_logic_deduction", signature)
    r8 = recovery_intervention("R8", "formal_logic_deduction", signature)

    assert r3["kind"] == "prompt_variant"
    assert r4["kind"] == "prompt_variant" and "logic_error" in r4["suffix"]
    assert r5["kind"] == "prompt_variant" and "Decompose" in r5["suffix"]
    assert r6["kind"] == "prompt_variant" and "state" in r6["suffix"].lower()
    assert r7["kind"] == "prompt_variant" and "logical" in r7["suffix"].lower()
    assert r8 == {"kind": "unavailable", "reason": "INVERTED_INTERVENTION_NOT_CONFIGURED"}
    assert "If A implies B" not in " ".join(item.get("suffix", "") for item in (r3, r4, r5, r6, r7))


class Store:
    def __init__(self):
        self.rows = {}

    def append_jsonl(self, path, value):
        self.rows.setdefault(path, []).append(value)


class Runner:
    def __init__(self):
        self.store = Store()
        self.model = "gpt-oss:20b"
        self.progress = None
        self.config = {
            "capability_campaign": {
                "generation_budget": 256,
                "reliable_threshold": 0.90,
            },
            "recovery_lab": {
                "enabled": True,
                "repeats": 3,
                "max_level": "R7",
            },
        }

    def _utc(self):
        return "x"


def test_recovery_reuses_r0_and_existing_high_effort_r2_without_model_calls(monkeypatch):
    import compute_cost.recovery_lab as recovery

    calls = []
    monkeypatch.setattr(recovery, "execute_experiment", lambda *args, **kwargs: calls.append(1))
    base = [row(f"base-{i}", "ANSWER_WRONG") for i in range(3)]
    high = [row(f"high-{i}", "ANSWER_CORRECT", effort="high") for i in range(3)]

    rows, recovery_map, sequence = recovery.run_recovery_lab(
        Runner(),
        [fixture()],
        base,
        high,
        frontier(),
        failure_atlas={
            "failures": [{
                "family_id": "formal_logic_deduction",
                "difficulty_level": 5,
                "failure_signature": {"subtype": "logic_error", "causal_claim": False},
            }]
        },
        sequence_start=20,
    )

    assert rows == []
    assert calls == []
    assert sequence == 20
    family = recovery_map["families"]["formal_logic_deduction"]
    assert family["steps"][0]["level"] == "R0"
    assert family["steps"][0]["evidence_source"] == "BASELINE_REPLICATIONS"
    assert family["steps"][1]["level"] == "R1"
    assert family["steps"][1]["status"] == "NOT_APPLICABLE"
    assert family["steps"][2]["level"] == "R2"
    assert family["steps"][2]["evidence_source"] == "REASONING_CURVE_REUSE"
    assert family["minimum_successful_recovery"] == "R2"


def test_recovery_runs_prompt_variants_from_same_baseline_parent_and_stops_at_first_reliable_success(monkeypatch):
    import compute_cost.recovery_lab as recovery

    executed = []
    outcomes = {
        "R2": ["ANSWER_WRONG"],
        "R3": ["ANSWER_WRONG"],
        "R4": ["ANSWER_CORRECT", "ANSWER_CORRECT", "ANSWER_CORRECT"],
    }

    def execute(runner, case, experiment, *, parent=None):
        level = experiment.recovery_level
        result_class = outcomes[level].pop(0)
        executed.append({
            "level": level,
            "parent": parent.experiment_id,
            "changed_variable": experiment.changed_variable,
            "prompt_variant": experiment.prompt_variant,
            "effort": experiment.reasoning_effort,
            "prompt": case["prompt"],
        })
        return row(
            experiment.experiment_id,
            result_class,
            effort=experiment.reasoning_effort or "medium",
            prompt_variant=experiment.prompt_variant,
            recovery_level=level,
        )

    monkeypatch.setattr(recovery, "execute_experiment", execute)
    base = [row(f"base-{i}", "ANSWER_WRONG") for i in range(3)]
    high_fail = [row(f"high-{i}", "ANSWER_WRONG", effort="high") for i in range(2)]

    recovery_rows, recovery_map, _ = recovery.run_recovery_lab(
        Runner(),
        [fixture()],
        base,
        high_fail,
        frontier(),
        failure_atlas={
            "failures": [{
                "family_id": "formal_logic_deduction",
                "difficulty_level": 5,
                "failure_signature": {"subtype": "logic_error", "causal_claim": False},
            }]
        },
        sequence_start=30,
    )

    assert [item["level"] for item in executed] == ["R2", "R3", "R4", "R4", "R4"]
    baseline_parent = "base-2"
    assert executed[0]["parent"] == baseline_parent
    assert executed[0]["changed_variable"] == "reasoning_effort"
    assert executed[1]["parent"] == baseline_parent
    assert executed[1]["changed_variable"] == "prompt_variant"
    assert executed[2]["parent"] == baseline_parent
    assert executed[2]["changed_variable"] == "prompt_variant"
    assert executed[3]["changed_variable"] == "replication"
    assert executed[4]["changed_variable"] == "replication"
    assert "logic_error" in executed[2]["prompt"]
    assert len(recovery_rows) == 5
    family = recovery_map["families"]["formal_logic_deduction"]
    assert family["minimum_successful_recovery"] == "R4"
    assert [step["level"] for step in family["steps"]] == ["R0", "R1", "R2", "R3", "R4"]
