"""Lossless, integrity-verifiable evidence storage for benchmark runs."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EvidenceStore:
    """Write original benchmark evidence before any normalization or reporting."""

    MANIFEST_NAME = "manifest.json"

    def __init__(self, root: Path | str, run_id: str):
        self.root = Path(root)
        self.run_id = run_id
        self.run_dir = self.root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, relative_path: str | Path) -> Path:
        rel = Path(relative_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"evidence path must stay inside run directory: {relative_path}")
        target = self.run_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _artifact_record(
        self,
        path: Path,
        *,
        producer: str | None = None,
        stage: str | None = None,
        case_id: str | None = None,
        media_type: str | None = None,
    ) -> dict[str, Any]:
        data = path.read_bytes()
        record: dict[str, Any] = {
            "path": path.relative_to(self.run_dir).as_posix(),
            "bytes": len(data),
            "sha256": self._sha256(data),
        }
        if producer is not None:
            record["producer"] = producer
        if stage is not None:
            record["stage"] = stage
        if case_id is not None:
            record["case_id"] = case_id
        if media_type is not None:
            record["media_type"] = media_type
        return record

    def write_raw(
        self,
        relative_path: str | Path,
        data: bytes | str,
        *,
        producer: str | None = None,
        stage: str | None = None,
        case_id: str | None = None,
        media_type: str | None = None,
    ) -> dict[str, Any]:
        path = self._path(relative_path)
        raw = data.encode("utf-8") if isinstance(data, str) else data
        path.write_bytes(raw)
        return self._artifact_record(
            path,
            producer=producer,
            stage=stage,
            case_id=case_id,
            media_type=media_type,
        )

    def write_json(
        self,
        relative_path: str | Path,
        value: Any,
        *,
        producer: str | None = None,
        stage: str | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        encoded = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n"
        ).encode("utf-8")
        return self.write_raw(
            relative_path,
            encoded,
            producer=producer,
            stage=stage,
            case_id=case_id,
            media_type="application/json",
        )

    def append_jsonl(self, relative_path: str | Path, value: Any) -> dict[str, Any]:
        path = self._path(relative_path)
        encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n").encode(
            "utf-8"
        )
        with path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
        return self._artifact_record(path, media_type="application/x-ndjson")

    def record_capture_gap(
        self,
        *,
        channel: str,
        collector: str,
        error: str,
        affected: str,
        continued: bool,
    ) -> dict[str, Any]:
        event = {
            "type": "CAPTURE_GAP",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            "channel": channel,
            "collector": collector,
            "error": error,
            "affected": affected,
            "continued": continued,
        }
        self.append_jsonl("events.jsonl", event)
        return event

    def finalize_manifest(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        artifacts: list[dict[str, Any]] = []
        for path in sorted(p for p in self.run_dir.rglob("*") if p.is_file()):
            if path.name == self.MANIFEST_NAME and path.parent == self.run_dir:
                continue
            artifacts.append(self._artifact_record(path))

        manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "artifacts": artifacts,
            "evidence_bytes": sum(item["bytes"] for item in artifacts),
        }
        self.write_json(self.MANIFEST_NAME, manifest, producer="evidence-store", stage="finalize")
        return manifest

    def verify_manifest(self) -> list[dict[str, str]]:
        manifest_path = self.run_dir / self.MANIFEST_NAME
        if not manifest_path.exists():
            return [{"path": self.MANIFEST_NAME, "problem": "missing"}]

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        problems: list[dict[str, str]] = []
        for expected in manifest.get("artifacts", []):
            path = self.run_dir / expected["path"]
            if not path.exists():
                problems.append({"path": expected["path"], "problem": "missing"})
                continue
            actual = self._artifact_record(path)
            if actual["sha256"] != expected["sha256"]:
                problems.append({"path": expected["path"], "problem": "sha256_mismatch"})
            elif actual["bytes"] != expected["bytes"]:
                problems.append({"path": expected["path"], "problem": "size_mismatch"})
        return problems
