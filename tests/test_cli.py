import json
from pathlib import Path

import compute_cost.cli as cli
from compute_cost.cli import build_parser, main
from compute_cost.config import load_config


PLANNED_MODELS = [
    "qwen3.5:27b-q4_K_M",
    "qwen3.5:27b-q8_0",
    "qwen3.5:35b-a3b-q4_K_M",
    "qwen3.5:35b-a3b-q8_0",
    "devstral-small-2:24b-instruct-2512-q8_0",
]


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


def test_parser_supports_capability_characterization_defaults():
    parser = build_parser()
    args = parser.parse_args(["capability-characterize", "--model", "fake"])
    assert args.command == "capability-characterize"
    assert args.model == "fake"
    assert args.pull is False
    assert args.suite.endswith("benchmarks\\gpt-oss-20b-capability-v1.json") or args.suite.endswith("benchmarks/gpt-oss-20b-capability-v1.json")
    assert args.taxonomy.endswith("benchmarks\\capability-taxonomy-v1.json") or args.taxonomy.endswith("benchmarks/capability-taxonomy-v1.json")


def test_default_config_declares_bounded_capability_campaign_controls():
    cfg = load_config()
    campaign = cfg["capability_campaign"]
    assert campaign == {
        "anchor_level": 2,
        "jump": 3,
        "boundary_repeats": 5,
        "max_experiments_per_family": 24,
        "thinking_mode": True,
        "reasoning_effort": "medium",
        "generation_budget": 256,
        "reliable_threshold": 0.90,
        "unstable_threshold": 0.40,
    }


def test_parser_supports_sequential_characterization_campaign():
    parser = build_parser()
    args = parser.parse_args(["characterize-campaign"])
    assert args.command == "characterize-campaign"
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


def test_capability_characterize_validates_materializes_normalizes_and_dispatches(tmp_path: Path, capsys, monkeypatch):
    calls = []
    validations = []

    class FakeRuntime:
        def __init__(self, endpoint, timeout_s):
            self.endpoint = endpoint
            self.timeout_s = timeout_s

    class FakeRunner:
        def __init__(self, runtime, config, suite, *, results_root):
            assert suite["normalized"] is True
            assert suite["materialized"] is True
            assert "capability_campaign" in config

        def capability_characterize(self, model, *, pull=False):
            calls.append((model, pull))
            run = tmp_path / "cap-run"
            run.mkdir(exist_ok=True)
            return run

    raw_suite = {"benchmark_version": "x", "cases": [{"id": "x"}]}
    materialized_suite = {**raw_suite, "materialized": True}
    taxonomy_data = {"taxonomy_version": "test-taxonomy", "families": []}
    suite_path = tmp_path / "suite.json"
    taxonomy_path = tmp_path / "taxonomy.json"
    suite_path.write_text(json.dumps(raw_suite), encoding="utf-8")
    taxonomy_path.write_text(json.dumps(taxonomy_data), encoding="utf-8")

    def validate(suite, taxonomy):
        validations.append((suite, taxonomy))

    def materialize(suite, taxonomy):
        assert suite == raw_suite
        assert taxonomy == taxonomy_data
        return materialized_suite

    def normalize(suite):
        assert suite is materialized_suite
        return {**suite, "normalized": True}

    monkeypatch.setattr(cli, "OllamaAdapter", FakeRuntime)
    monkeypatch.setattr(cli, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(cli, "validate_capability_suite", validate)
    monkeypatch.setattr(cli, "materialize_gpt_oss_suite", materialize)
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
    assert validations == [
        (raw_suite, taxonomy_data),
        (materialized_suite, taxonomy_data),
    ]
    assert calls == [("fake", False)]
    assert out["run_id"] == "cap-run"


def test_characterize_campaign_runs_five_models_in_order_with_fresh_runners(tmp_path: Path, capsys, monkeypatch):
    calls = []
    runner_instances = []

    class FakeRuntime:
        def __init__(self, endpoint, timeout_s):
            self.endpoint = endpoint
            self.timeout_s = timeout_s

    class FakeRunner:
        def __init__(self, runtime, config, suite, *, results_root):
            runner_instances.append(self)

        def characterize(self, model, *, pull=False):
            calls.append((model, pull))
            run = tmp_path / f"run-{len(calls)}"
            run.mkdir()
            (run / "events.jsonl").write_text(
                json.dumps({"type": "CHARACTERIZATION_COMPLETE", "model": model}) + "\n",
                encoding="utf-8",
            )
            return run

    class FakeStore:
        def __init__(self, results_root, run_id):
            self.run_id = run_id

        def verify_manifest(self):
            return []

    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"benchmark_version": "x", "cases": [{"id": "x"}]}), encoding="utf-8")
    monkeypatch.setattr(cli, "OllamaAdapter", FakeRuntime)
    monkeypatch.setattr(cli, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(cli, "EvidenceStore", FakeStore)

    code = main([
        "--results-root", str(tmp_path),
        "characterize-campaign", "--suite", str(suite),
    ])
    out = json.loads(capsys.readouterr().out)

    assert code == 0
    assert calls == [(model, False) for model in PLANNED_MODELS]
    assert len(runner_instances) == 5
    assert out["ok"] is True
    assert [row["model"] for row in out["runs"]] == PLANNED_MODELS
    assert all(row["manifest_ok"] is True for row in out["runs"])


def test_characterize_campaign_stops_before_next_model_when_manifest_fails(tmp_path: Path, capsys, monkeypatch):
    calls = []

    class FakeRuntime:
        def __init__(self, endpoint, timeout_s):
            pass

    class FakeRunner:
        def __init__(self, runtime, config, suite, *, results_root):
            pass

        def characterize(self, model, *, pull=False):
            calls.append(model)
            run = tmp_path / f"run-{len(calls)}"
            run.mkdir()
            (run / "events.jsonl").write_text(
                json.dumps({"type": "CHARACTERIZATION_COMPLETE", "model": model}) + "\n",
                encoding="utf-8",
            )
            return run

    class FakeStore:
        def __init__(self, results_root, run_id):
            self.run_id = run_id

        def verify_manifest(self):
            return [{"problem": "modified", "path": "raw/x"}] if self.run_id == "run-2" else []

    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"benchmark_version": "x", "cases": [{"id": "x"}]}), encoding="utf-8")
    monkeypatch.setattr(cli, "OllamaAdapter", FakeRuntime)
    monkeypatch.setattr(cli, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(cli, "EvidenceStore", FakeStore)

    code = main([
        "--results-root", str(tmp_path),
        "characterize-campaign", "--suite", str(suite),
    ])
    out = json.loads(capsys.readouterr().out)

    assert code == 2
    assert calls == PLANNED_MODELS[:2]
    assert out["ok"] is False
    assert out["failed_model"] == PLANNED_MODELS[1]
    assert out["failure"] == "MANIFEST_VERIFICATION_FAILED"


def test_characterize_campaign_stops_before_next_model_when_run_failed(tmp_path: Path, capsys, monkeypatch):
    calls = []

    class FakeRuntime:
        def __init__(self, endpoint, timeout_s):
            pass

    class FakeRunner:
        def __init__(self, runtime, config, suite, *, results_root):
            pass

        def characterize(self, model, *, pull=False):
            calls.append(model)
            run = tmp_path / f"run-{len(calls)}"
            run.mkdir()
            event = "RUN_FAILED" if len(calls) == 2 else "CHARACTERIZATION_COMPLETE"
            (run / "events.jsonl").write_text(
                json.dumps({"type": event, "model": model}) + "\n",
                encoding="utf-8",
            )
            return run

    class FakeStore:
        def __init__(self, results_root, run_id):
            pass

        def verify_manifest(self):
            return []

    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"benchmark_version": "x", "cases": [{"id": "x"}]}), encoding="utf-8")
    monkeypatch.setattr(cli, "OllamaAdapter", FakeRuntime)
    monkeypatch.setattr(cli, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(cli, "EvidenceStore", FakeStore)

    code = main([
        "--results-root", str(tmp_path),
        "characterize-campaign", "--suite", str(suite),
    ])
    out = json.loads(capsys.readouterr().out)

    assert code == 2
    assert calls == PLANNED_MODELS[:2]
    assert out["ok"] is False
    assert out["failed_model"] == PLANNED_MODELS[1]
    assert out["failure"] == "RUN_FAILED"


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
