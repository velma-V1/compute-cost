import json
from pathlib import Path


def test_replay_category_routes_failures_anomalies_and_recoveries():
    from compute_cost.replay_registry import replay_category

    assert replay_category("ANSWER_WRONG", recovery_level=None) == "failures"
    assert replay_category("FORMAT_FAILURE", recovery_level=None) == "failures"
    assert replay_category("THINK_TRUNCATED", recovery_level=None) == "anomalies"
    assert replay_category("RUNTIME_FAILURE", recovery_level=None) == "anomalies"
    assert replay_category("SCORER_DEFECT", recovery_level=None) == "anomalies"
    assert replay_category("ANSWER_WRONG", recovery_level="R4") == "recoveries"


def test_replay_resolver_supports_historical_root_and_new_index(tmp_path: Path):
    from compute_cost.replay_registry import resolve_replay_snapshot

    legacy_dir = tmp_path / "legacy"
    (legacy_dir / "replay").mkdir(parents=True)
    (legacy_dir / "replay" / "old-case.json").write_text(
        json.dumps({"schema_version": 1, "id": "old"}), encoding="utf-8"
    )
    assert resolve_replay_snapshot(legacy_dir, "old-case") == legacy_dir / "replay" / "old-case.json"

    modern_dir = tmp_path / "modern"
    target = modern_dir / "replay" / "failures" / "exp-1.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"schema_version": 2, "id": "new"}), encoding="utf-8")
    (modern_dir / "replay" / "index.jsonl").write_text(
        json.dumps({
            "replay_id": "exp-1",
            "category": "failures",
            "path": "replay/failures/exp-1.json",
        }) + "\n",
        encoding="utf-8",
    )
    assert resolve_replay_snapshot(modern_dir, "exp-1") == target


def test_replay_resolver_rejects_index_path_escape(tmp_path: Path):
    from compute_cost.replay_registry import resolve_replay_snapshot

    run_dir = tmp_path / "run"
    (run_dir / "replay").mkdir(parents=True)
    (run_dir / "replay" / "index.jsonl").write_text(
        json.dumps({"replay_id": "evil", "path": "../outside.json"}) + "\n",
        encoding="utf-8",
    )
    try:
        resolve_replay_snapshot(run_dir, "evil")
    except ValueError as exc:
        assert "inside run directory" in str(exc)
    else:
        raise AssertionError("replay index path escape must be rejected")
