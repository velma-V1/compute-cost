import json
from pathlib import Path

import compute_cost.cli as cli
from compute_cost.cli import build_parser, main


SHOWDOWN_MODELS = [
    ("gpt-oss:20b", "ollama"),
    ("qwen3.5:35b-a3b-q4_K_M", "ollama"),
    ("devstral-small-2:24b-instruct-2512-q8_0", "ollama"),
    ("qwen3-next-80b-a3b-instruct-q4_k_m", "oversized"),
]


def test_parser_supports_capability_showdown_defaults():
    parser = build_parser()
    args = parser.parse_args(["capability-showdown"])

    assert args.command == "capability-showdown"
    assert args.suite.endswith("benchmarks\\gpt-oss-20b-capability-v1.json") or args.suite.endswith("benchmarks/gpt-oss-20b-capability-v1.json")
    assert args.taxonomy.endswith("benchmarks\\capability-taxonomy-v1.json") or args.taxonomy.endswith("benchmarks/capability-taxonomy-v1.json")
    assert args.oversized_endpoint == "http://127.0.0.1:8080"
    assert args.hard_timeout_s == 600.0


def test_showdown_roster_is_exact_and_includes_devstral_q8():
    assert list(cli.SHOWDOWN_MODELS) == SHOWDOWN_MODELS


def test_capability_showdown_reuses_capability_runner_and_routes_backends(tmp_path: Path, capsys, monkeypatch):
    calls = []
    validations = []

    class FakeOllama:
        kind = "ollama"

        def __init__(self, endpoint, timeout_s):
            self.endpoint = endpoint
            self.timeout_s = timeout_s

    class FakeOversized:
        kind = "oversized"

        def __init__(self, endpoint, *, timeout_s, served_model):
            self.endpoint = endpoint
            self.timeout_s = timeout_s
            self.served_model = served_model

    class FakeRunner:
        def __init__(self, runtime, config, suite, *, results_root):
            self.runtime = runtime
            assert suite["normalized"] is True
            assert suite["materialized"] is True

        def capability_characterize(self, model, *, pull=False):
            calls.append((self.runtime.kind, model, pull))
            run = tmp_path / f"run-{len(calls)}"
            run.mkdir()
            (run / "events.jsonl").write_text(
                json.dumps({"type": "CAPABILITY_CHARACTERIZATION_COMPLETE", "model": model}) + "\n",
                encoding="utf-8",
            )
            return run

    class FakeStore:
        def __init__(self, results_root, run_id):
            self.run_id = run_id

        def verify_manifest(self):
            return []

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

    monkeypatch.setattr(cli, "OllamaAdapter", FakeOllama)
    monkeypatch.setattr(cli, "OversizedMoEAdapter", FakeOversized, raising=False)
    monkeypatch.setattr(cli, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(cli, "EvidenceStore", FakeStore)
    monkeypatch.setattr(cli, "validate_capability_suite", validate)
    monkeypatch.setattr(cli, "materialize_gpt_oss_suite", materialize)
    monkeypatch.setattr(cli, "normalize_capability_suite", normalize)

    code = main([
        "--results-root", str(tmp_path),
        "capability-showdown",
        "--suite", str(suite_path),
        "--taxonomy", str(taxonomy_path),
        "--oversized-endpoint", "http://127.0.0.1:9999",
        "--hard-timeout-s", "600",
    ])
    out = json.loads(capsys.readouterr().out)

    assert code == 0
    assert validations == [
        (raw_suite, taxonomy_data),
        (materialized_suite, taxonomy_data),
    ]
    assert calls == [
        ("ollama", "gpt-oss:20b", False),
        ("ollama", "qwen3.5:35b-a3b-q4_K_M", False),
        ("ollama", "devstral-small-2:24b-instruct-2512-q8_0", False),
        ("oversized", "qwen3-next-80b-a3b-instruct-q4_k_m", False),
    ]
    assert out["ok"] is True
    assert [row["model"] for row in out["runs"]] == [model for model, _ in SHOWDOWN_MODELS]
    assert [row["backend"] for row in out["runs"]] == [backend for _, backend in SHOWDOWN_MODELS]
    assert all(row["manifest_ok"] is True for row in out["runs"])
