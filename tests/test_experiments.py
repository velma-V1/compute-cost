from compute_cost.experiments import ExperimentSpec, changed_fields, make_experiment_id


def make_spec(**overrides):
    values = dict(
        experiment_id="exp-root",
        parent_experiment_id=None,
        task_id="math-001",
        task_family="reasoning_math",
        difficulty_level=3,
        hypothesis="thinking baseline",
        changed_variable="baseline",
        thinking_mode=True,
        generation_budget=256,
        context_request=None,
        temperature=0.0,
        seed=42,
        prompt_variant="base",
        recovery_level=None,
    )
    values.update(overrides)
    return ExperimentSpec(**values)


def test_id_is_stable():
    assert make_experiment_id(7, "math-001", "think-on-256") == "exp-000007-math-001-think-on-256"


def test_child_budget_probe_changes_exactly_one_controlled_field():
    parent = make_spec()
    child = make_spec(
        experiment_id="exp-child",
        parent_experiment_id="exp-root",
        hypothesis="find lower passing budget",
        changed_variable="generation_budget",
        generation_budget=192,
    )
    assert changed_fields(parent, child) == ["generation_budget"]
    assert child.to_dict()["parent_experiment_id"] == "exp-root"
