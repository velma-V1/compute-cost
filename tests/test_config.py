from pathlib import Path

import pytest

from compute_cost.config import load_config


def test_default_config_contains_safe_bounded_run_settings():
    config = load_config()
    assert config["runtime"]["endpoint"].startswith("http://127.0.0.1")
    assert config["limits"]["request_timeout_s"] > 0
    assert config["limits"]["context_schedule"]
    assert config["limits"]["sustained_iterations"] > 0
    assert config["cost"]["electricity_configured"] is False

    characterization = config["characterization"]
    assert characterization["initial_generation_budget"] == 256
    assert characterization["min_generation_budget"] == 32
    assert characterization["max_generation_budget"] == 2048
    assert characterization["generation_budget_granularity"] == 32
    assert characterization["boundary_repeats"] == 3
    assert characterization["max_experiments_per_task"] == 12
    assert characterization["think_off_generation_budget"] == 256

    recovery = config["recovery_lab"]
    assert recovery["enabled"] is True
    assert recovery["repeats"] == 3
    assert recovery["max_level"] == "R7"

    robustness = config["robustness_lab"]
    assert robustness["enabled"] is True
    assert robustness["repeats"] == 2
    assert robustness["max_perturbations_per_family"] == 2
    assert robustness["max_attempts_per_perturbation"] == 4

    compound = config["compound_lab"]
    assert compound["enabled"] is True
    assert compound["jump"] == 2
    assert compound["boundary_repeats"] == 3
    assert compound["max_experiments_per_compound"] == 16
    assert compound["max_compounds"] == 6


def test_user_toml_and_dotted_overrides_merge_without_erasing_other_defaults(tmp_path: Path):
    custom = tmp_path / "custom.toml"
    custom.write_text('[runtime]\nendpoint = "http://localhost:9999"\n[limits]\nsustained_iterations = 3\n', encoding="utf-8")

    config = load_config(custom, {"limits.request_timeout_s": 45, "cost.electricity_per_kwh": 0.12})

    assert config["runtime"]["endpoint"] == "http://localhost:9999"
    assert config["limits"]["sustained_iterations"] == 3
    assert config["limits"]["request_timeout_s"] == 45
    assert config["limits"]["context_schedule"]
    assert config["cost"]["electricity_per_kwh"] == 0.12
    assert config["cost"]["electricity_configured"] is True
    assert config["characterization"]["boundary_repeats"] == 3


def test_characterization_budget_order_is_validated(tmp_path: Path):
    custom = tmp_path / "bad.toml"
    custom.write_text(
        '[characterization]\nmin_generation_budget = 512\ninitial_generation_budget = 256\nmax_generation_budget = 2048\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="min <= initial <= max"):
        load_config(custom)


def test_characterization_integer_controls_must_be_positive():
    with pytest.raises(ValueError, match="positive integer"):
        load_config(overrides={"characterization.boundary_repeats": 0})


def test_recovery_lab_repeats_must_be_positive():
    with pytest.raises(ValueError, match="recovery_lab.repeats must be a positive integer"):
        load_config(overrides={"recovery_lab.repeats": 0})


def test_recovery_lab_max_level_is_validated():
    with pytest.raises(ValueError, match="recovery_lab.max_level"):
        load_config(overrides={"recovery_lab.max_level": "R9"})


def test_robustness_lab_integer_controls_are_validated():
    with pytest.raises(ValueError, match="robustness_lab.repeats must be a positive integer"):
        load_config(overrides={"robustness_lab.repeats": 0})
    with pytest.raises(ValueError, match="robustness_lab.max_perturbations_per_family must be a positive integer"):
        load_config(overrides={"robustness_lab.max_perturbations_per_family": 0})
    with pytest.raises(ValueError, match="robustness_lab.max_attempts_per_perturbation must be"):
        load_config(overrides={"robustness_lab.max_attempts_per_perturbation": 1, "robustness_lab.repeats": 2})


def test_compound_lab_integer_controls_are_validated():
    for field in ("jump", "boundary_repeats", "max_experiments_per_compound", "max_compounds"):
        with pytest.raises(ValueError, match=fr"compound_lab\.{field} must be a positive integer"):
            load_config(overrides={f"compound_lab.{field}": 0})
