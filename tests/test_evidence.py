import json
from pathlib import Path

import pytest

from compute_cost.evidence import EvidenceStore


def test_raw_bytes_are_retained_exactly_and_hashed(tmp_path: Path):
    store = EvidenceStore(tmp_path, "run-1")
    payload = b"\x00raw\nbytes\xff"

    record = store.write_raw("raw/runtime/responses/r1.bin", payload, producer="ollama", stage="case")

    path = store.run_dir / "raw/runtime/responses/r1.bin"
    assert path.read_bytes() == payload
    assert record["bytes"] == len(payload)
    assert len(record["sha256"]) == 64


def test_jsonl_is_append_only_and_preserves_unknown_fields(tmp_path: Path):
    store = EvidenceStore(tmp_path, "run-1")
    first = {"known": 1, "future_runtime_field": {"x": [1, 2, 3]}}
    second = {"known": 2, "another_unknown": True}

    store.append_jsonl("events.jsonl", first)
    store.append_jsonl("events.jsonl", second)

    lines = (store.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [first, second]


def test_capture_gap_is_explicit_evidence(tmp_path: Path):
    store = EvidenceStore(tmp_path, "run-1")

    store.record_capture_gap(
        channel="nvidia",
        collector="nvidia-smi",
        error="not found",
        affected="preflight",
        continued=True,
    )

    event = json.loads((store.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert event["type"] == "CAPTURE_GAP"
    assert event["channel"] == "nvidia"
    assert event["continued"] is True


def test_manifest_inventories_artifacts_and_verifies_integrity(tmp_path: Path):
    store = EvidenceStore(tmp_path, "run-1")
    store.write_json("hardware.json", {"cpu": "test"})
    store.write_raw("raw/runtime/requests/r1.json", b'{"prompt":"hello"}')

    manifest = store.finalize_manifest(metadata={"model": "fake"})

    assert manifest["metadata"]["model"] == "fake"
    paths = {item["path"] for item in manifest["artifacts"]}
    assert "hardware.json" in paths
    assert "raw/runtime/requests/r1.json" in paths
    assert store.verify_manifest() == []


def test_manifest_verification_detects_modified_and_missing_files(tmp_path: Path):
    store = EvidenceStore(tmp_path, "run-1")
    store.write_raw("raw/a.bin", b"original")
    store.write_raw("raw/b.bin", b"keep")
    store.finalize_manifest()

    (store.run_dir / "raw/a.bin").write_bytes(b"changed")
    (store.run_dir / "raw/b.bin").unlink()

    problems = store.verify_manifest()
    assert any(p["path"] == "raw/a.bin" and p["problem"] == "sha256_mismatch" for p in problems)
    assert any(p["path"] == "raw/b.bin" and p["problem"] == "missing" for p in problems)


def test_paths_cannot_escape_run_directory(tmp_path: Path):
    store = EvidenceStore(tmp_path, "run-1")

    with pytest.raises(ValueError):
        store.write_raw("../escape.txt", b"no")
