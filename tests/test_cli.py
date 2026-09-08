import json
from pathlib import Path

import compute_cost.cli as cli
from compute_cost.cli import build_parser, main


def test_parser_requires_model_for_onboarding():
    parser = build_parser()
    args = parser.parse_args(["onboard", "--model", "qwen:test", "--pull"])
    assert args.command == "onboard"
    assert args.model == "qwen:test"
    assert args.pull is True


def test_parser_supports_single_model_characterization():
    parser = build_parser()
    args = parser.parse_args(["characterize", "--model", "fake"])
    assert args.command == "characterize"
    assert args.model == "fake"
    assert args.pull is False
    assert args.suite.endswith("benchmarks\\qwen-characterization-v1.json") or args.suite.endswith("benchmarks/qwen-characterization-v1.json")


def test_parser_supports_compare_replay_and_verify():
    parser = build_parser()
    assert parser.parse_args(["compare", "run-a", "run-b"]).runs == ["run-a", "run-b"]
    replay = parser.parse_args(["replay", "run-a", "case-1"])
    assert replay.run_id == "run-a"
    assert replay.case_id == "case-1"
    assert parser.parse_args(["verify", "run-a"]).run_id == "run-a"


def test_characterize_dispatches_exactly_one_model(tmp_path: Path, capsys, monkeypatch):
    calls = []

    class FakeRuntime:
        def __init__(self, endpoint, timeout_s):
            self.endpoint = endpoint
            self.timeout_s = timeout_s

    class FakeRunner:
        def __init__(self, runtime, config, suite, *, results_root):
            assert len(suite["cases"]) == 1

        def characterize(self, model, *, pull=False):
            calls.append((model, pull))
            run = tmp_path / "char-run"
            run.mkdir(exist_ok=True)
            return run

    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"benchmark_version": "x", "cases": [{"id": "x"}]}), encoding="utf-8")
    monkeypatch.setattr(cli, "OllamaAdapter", FakeRuntime)
    monkeypatch.setattr(cli, "BenchmarkRunner", FakeRunner)

    code = main([
        "--results-root", str(tmp_path),
        "characterize", "--model", "fake", "--suite", str(suite),
    ])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert calls == [("fake", False)]
    assert out["run_id"] == "char-run"


def test_compare_command_reads_existing_runs_only(tmp_path: Path, capsys):
    for name, score in [("a", 1.0), ("b", 0.0)]:
        run = tmp_path / name
        run.mkdir()
        (run / "resolved-config.json").write_text("{}", encoding="utf-8")
        (run / "runtime.json").write_text(json.dumps({"model": name}), encoding="utf-8")
        (run / "hardware.json").write_text("{}", encoding="utf-8")
        (run / "events.jsonl").write_text('', encoding="utf-8")
        (run / "telemetry.jsonl").write_text('', encoding="utf-8")
        (run / "cases.jsonl").write_text(json.dumps({"case_id": "x", "category": "instruction_following", "stage": "base", "score": score, "status": "SCORED", "timing": {"client_latency_ns": 1}}) + "\n", encoding="utf-8")

    code = main(["--results-root", str(tmp_path), "compare", "a", "b"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert [row["run_id"] for row in out["runs"]] == ["a", "b"]


def test_verify_returns_nonzero_when_manifest_is_missing(tmp_path: Path, capsys):
    (tmp_path / "run-x").mkdir()
    code = main(["--results-root", str(tmp_path), "verify", "run-x"])
    out = json.loads(capsys.readouterr().out)
    assert code == 2
    assert out["ok"] is False
    assert out["problems"][0]["problem"] == "missing"
