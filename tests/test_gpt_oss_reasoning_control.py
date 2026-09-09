from compute_cost.config import load_config


def test_default_gpt_oss_campaign_declares_medium_reasoning_effort():
    campaign = load_config()["capability_campaign"]
    assert campaign["thinking_mode"] is True
    assert campaign["reasoning_effort"] == "medium"


def test_default_reasoning_curves_are_bounded_and_enabled():
    curves = load_config()["reasoning_curves"]
    assert curves == {"enabled": True, "repeats": 3}


def test_gpt_oss_reasoning_effort_config_rejects_unsupported_value(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[capability_campaign]\nreasoning_effort = "max"\n', encoding="utf-8")
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
