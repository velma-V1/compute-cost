from pathlib import Path

from compute_cost.config import load_config


def test_default_config_contains_safe_bounded_run_settings():
    config = load_config()
    assert config["runtime"]["endpoint"].startswith("http://127.0.0.1")
    assert config["limits"]["request_timeout_s"] > 0
    assert config["limits"]["context_schedule"]
    assert config["limits"]["sustained_iterations"] > 0
    assert config["cost"]["electricity_configured"] is False


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
