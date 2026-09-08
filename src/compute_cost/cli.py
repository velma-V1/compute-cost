"""Command-line interface for local model onboarding and compute-cost analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .config import load_config
from .evidence import EvidenceStore
from .hardware import collect_hardware_snapshot
from .report import compare_runs
from .runner import BenchmarkRunner
from .runtimes.ollama import OllamaAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.toml"
DEFAULT_SUITE_PATH = PROJECT_ROOT / "benchmarks" / "base-v1.json"
DEFAULT_CHARACTERIZATION_SUITE_PATH = PROJECT_ROOT / "benchmarks" / "qwen-characterization-v1.json"
PLANNED_CHARACTERIZATION_MODELS = (
    "qwen3.5:27b-q4_K_M",
    "qwen3.5:27b-q8_0",
    "qwen3.5:35b-a3b-q4_K_M",
    "qwen3.5:35b-a3b-q8_0",
    "devstral-small-2:24b-instruct-2512-q8_0",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compute-cost",
        description="Lossless local-model onboarding, capability, and compute-cost benchmark.",
    )
    parser.add_argument("--results-root", default="results", help="Directory containing benchmark runs.")
    parser.add_argument("--config", default=None, help="Optional TOML configuration overlay.")
    parser.add_argument("--endpoint", default=None, help="Ollama endpoint override.")
    parser.add_argument("--electricity-per-kwh", type=float, default=None, help="Local electricity price per kWh.")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight", help="Print host and Ollama preflight information.")

    onboard = sub.add_parser("onboard", help="Run the complete onboarding benchmark.")
    onboard.add_argument("--model", required=True)
    onboard.add_argument("--suite", default=str(DEFAULT_SUITE_PATH))
    onboard.add_argument("--pull", action="store_true", help="Pull the model if it is not already local.")

    benchmark = sub.add_parser("benchmark", help="Benchmark an already-local model.")
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--suite", default=str(DEFAULT_SUITE_PATH))

    characterize = sub.add_parser("characterize", help="Adaptively characterize one local model.")
    characterize.add_argument("--model", required=True)
    characterize.add_argument("--suite", default=str(DEFAULT_CHARACTERIZATION_SUITE_PATH))
    characterize.add_argument("--pull", action="store_true", help="Pull the model if it is not already local.")

    campaign = sub.add_parser(
        "characterize-campaign",
        help="Characterize the planned five-model set sequentially with fail-fast verification.",
    )
    campaign.add_argument("--suite", default=str(DEFAULT_CHARACTERIZATION_SUITE_PATH))
    campaign.add_argument("--pull", action="store_true", help="Pull a planned model if it is not already local.")

    compare = sub.add_parser("compare", help="Compare completed runs without model execution.")
    compare.add_argument("runs", nargs="+", help="Two or more run IDs or run directories.")

    replay = sub.add_parser("replay", help="Replay one retained failure snapshot in a new run.")
    replay.add_argument("run_id")
    replay.add_argument("case_id")
    replay.add_argument("--suite", default=str(DEFAULT_SUITE_PATH))

    verify = sub.add_parser("verify", help="Verify SHA-256 integrity of a completed run.")
    verify.add_argument("run_id")

    return parser


def _config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.endpoint is not None:
        overrides["runtime.endpoint"] = args.endpoint
    if args.electricity_per_kwh is not None:
        overrides["cost.electricity_per_kwh"] = args.electricity_per_kwh
    config_path = Path(args.config) if args.config else None
    return load_config(config_path, overrides)


def _load_suite(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ValueError(f"invalid benchmark suite: {path}")
    return value


def _resolve_run(results_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_dir():
        return candidate
    return results_root / value


def _characterization_run_failure(run_dir: Path) -> str | None:
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        return "RUN_INCOMPLETE"
    complete = False
    for raw_line in events_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            return "RUN_INCOMPLETE"
        if event.get("event") == "RUN_FAILED":
            return "RUN_FAILED"
        if event.get("event") == "CHARACTERIZATION_COMPLETE":
            complete = True
    return None if complete else "RUN_INCOMPLETE"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    results_root = Path(args.results_root)

    if args.command == "compare":
        if len(args.runs) < 2:
            parser.error("compare requires at least two runs")
        result = compare_runs([_resolve_run(results_root, value) for value in args.runs])
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0

    if args.command == "verify":
        store = EvidenceStore(results_root, args.run_id)
        problems = store.verify_manifest()
        result = {"ok": not problems, "run_id": args.run_id, "problems": problems}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not problems else 2

    config = _config_from_args(args)
    endpoint = config.get("runtime", {}).get("endpoint", "http://127.0.0.1:11434")
    timeout = float(config.get("limits", {}).get("request_timeout_s", 120))
    runtime = OllamaAdapter(endpoint, timeout_s=timeout)

    if args.command == "preflight":
        result = {
            "hardware": collect_hardware_snapshot(),
            "runtime": runtime.version(),
            "models": runtime.list_models(),
        }
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result["runtime"].get("ok", False) else 2

    suite = _load_suite(args.suite)

    if args.command == "characterize-campaign":
        runs: list[dict[str, Any]] = []
        for model in PLANNED_CHARACTERIZATION_MODELS:
            runner = BenchmarkRunner(runtime, config, suite, results_root=results_root)
            run_dir = runner.characterize(model, pull=bool(args.pull))
            run_failure = _characterization_run_failure(run_dir)
            problems = EvidenceStore(results_root, run_dir.name).verify_manifest()
            row = {
                "model": model,
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "manifest_ok": not problems,
                "manifest_problems": problems,
            }
            runs.append(row)

            if run_failure is not None:
                print(json.dumps({
                    "ok": False,
                    "failed_model": model,
                    "failure": run_failure,
                    "runs": runs,
                }, indent=2, sort_keys=True, default=str))
                return 2
            if problems:
                print(json.dumps({
                    "ok": False,
                    "failed_model": model,
                    "failure": "MANIFEST_VERIFICATION_FAILED",
                    "runs": runs,
                }, indent=2, sort_keys=True, default=str))
                return 2

        print(json.dumps({"ok": True, "runs": runs}, indent=2, sort_keys=True, default=str))
        return 0

    runner = BenchmarkRunner(runtime, config, suite, results_root=results_root)

    if args.command in {"onboard", "benchmark"}:
        run_dir = runner.onboard(args.model, pull=bool(getattr(args, "pull", False)))
        print(json.dumps({"run_id": run_dir.name, "run_dir": str(run_dir)}, indent=2))
        return 0

    if args.command == "characterize":
        run_dir = runner.characterize(args.model, pull=bool(args.pull))
        print(json.dumps({"run_id": run_dir.name, "run_dir": str(run_dir)}, indent=2))
        return 0

    if args.command == "replay":
        replay_dir = runner.replay_case(args.run_id, args.case_id)
        print(json.dumps({"run_id": replay_dir.name, "run_dir": str(replay_dir)}, indent=2))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
