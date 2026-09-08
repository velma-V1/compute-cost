import json
from pathlib import Path

import compute_cost.cli as cli
from compute_cost.cli import main


def test_capability_cli_validates_expands_revalidates_normalizes_then_dispatches(
    tmp_path: Path, capsys, monkeypatch
):
    events = []
    calls = []

    class FakeRuntime:
        def __init__(self, endpoint, timeout_s):
            self.endpoint = endpoint
            self.timeout_s = timeout_s

    class FakeRunner:
        def __init__(self, runtime, config, suite, *, results_root):
            assert suite["normalized"] is True
            assert suite["expanded"] is True
            assert len(suite["cases"]) == 440

        def capability_characterize(self, model, *, pull=False):
            calls.append((model, pull))
            run = tmp_path / "cap-run"
            run.mkdir(exist_ok=True)
            return run

    raw_suite = {"benchmark_version": "x", "cases": [{"id": "anchor"}]}
    taxonomy = {"taxonomy_version": "test-taxonomy", "families": []}
    expanded = {
        "benchmark_version": "x",
        "cases": [{"id": f"case-{i}"} for i in range(440)],
        "expanded": True,
    }
    suite_path = tmp_path / "suite.json"
    taxonomy_path = tmp_path / "taxonomy.json"
    suite_path.write_text(json.dumps(raw_suite), encoding="utf-8")
    taxonomy_path.write_text(json.dumps(taxonomy), encoding="utf-8")

    def validate(suite, taxonomy_arg):
        assert taxonomy_arg == taxonomy
        events.append(("validate", suite))

    def expand(suite, taxonomy_arg):
        assert suite == raw_suite
        assert taxonomy_arg == taxonomy
        events.append(("expand", suite))
        return expanded

    def normalize(suite):
        assert suite is expanded
        events.append(("normalize", suite))
        return {**suite, "normalized": True}

    monkeypatch.setattr(cli, "OllamaAdapter", FakeRuntime)
    monkeypatch.setattr(cli, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(cli, "validate_capability_suite", validate)
    monkeypatch.setattr(cli, "expand_gpt_oss_ladders", expand)
    monkeypatch.setattr(cli, "normalize_capability_suite", normalize)

    code = main([
        "--results-root", str(tmp_path),
        "capability-characterize",
        "--model", "fake",
        "--suite", str(suite_path),
        "--taxonomy", str(taxonomy_path),
    ])
    out = json.loads(capsys.readouterr().out)

    assert code == 0
    assert [name for name, _ in events] == ["validate", "expand", "validate", "normalize"]
    assert events[0][1] == raw_suite
    assert events[2][1] is expanded
    assert calls == [("fake", False)]
    assert out["run_id"] == "cap-run"
