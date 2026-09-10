from compute_cost.capability_campaign import resolve_model_reasoning_control
from compute_cost.config import load_config


def test_default_campaign_is_compact_and_model_specific():
    cfg = load_config()
    campaign = cfg["capability_campaign"]
    assert campaign["thinking_mode"] is False
    assert campaign["reasoning_effort"] is None
    assert campaign["boundary_repeats"] == 2
    assert campaign["max_experiments_per_family"] == 2
    assert cfg["limits"]["max_model_calls_per_run"] == 120
    assert resolve_model_reasoning_control("gpt-oss:20b", campaign) == (True, "low")
    assert resolve_model_reasoning_control("qwen3.5:35b-a3b-q4_K_M", campaign) == (False, None)
    assert resolve_model_reasoning_control("devstral-small-2:24b-instruct-2512-q8_0", campaign) == (False, None)


def test_default_reasoning_curves_are_bounded_and_opt_in():
    curves = load_config()["reasoning_curves"]
    assert curves == {"enabled": False, "repeats": 3}


def test_gpt_oss_reasoning_effort_config_rejects_unsupported_value(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[capability_campaign]\nthinking_mode = true\nreasoning_effort = "max"\n', encoding="utf-8")
    try:
        load_config(path)
    except ValueError as exc:
        assert "reasoning_effort" in str(exc)
    else:
        raise AssertionError("unsupported GPT-OSS reasoning effort must be rejected")


def test_reasoning_curve_config_rejects_nonpositive_repeats(tmp_path):
    path = tmp_path / "bad-curves.toml"
    path.write_text('[reasoning_curves]\nrepeats = 0\n', encoding="utf-8")
    try:
        load_config(path)
    except ValueError as exc:
        assert "reasoning_curves.repeats" in str(exc)
    else:
        raise AssertionError("nonpositive reasoning curve repeats must be rejected")
