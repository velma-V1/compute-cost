from pathlib import Path


def test_post_run_synthesis_materializes_boundary_replays(monkeypatch, tmp_path: Path):
    import compute_cost.run_synthesis as synthesis

    calls = []

    def fake_materialize(model, frontiers, rows, run_dir):
        calls.append(
            {
                "model": model,
                "frontiers": frontiers,
                "rows": list(rows),
                "run_dir": Path(run_dir),
            }
        )
        return {
            "schema_version": 1,
            "model": model,
            "summary": {"boundary_families": 0, "boundary_snapshots": 0},
        }

    monkeypatch.setattr(
        synthesis,
        "materialize_boundary_replays",
        fake_materialize,
        raising=False,
    )

    frontiers = {
        "schema_version": 1,
        "taxonomy_version": "capability-taxonomy-v1",
        "families": {},
    }
    synthesis.build_cost_value_outputs("gpt-oss:20b", [], frontiers, tmp_path)

    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-oss:20b"
    assert calls[0]["frontiers"] is frontiers
    assert calls[0]["rows"] == []
    assert calls[0]["run_dir"] == tmp_path
