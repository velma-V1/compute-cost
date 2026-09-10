"""Install the approved 700-call comparability contract additively.

The stable runner/evidence pipeline remains the execution path. This patch swaps
only campaign orchestration, call accounting, attempt scoring, and zero-call
report synthesis before runner.py captures those callables.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .call_ledger import CallBudgetExceeded, CallLedger
from .score_vector import build_score_vector


def _install_attempt_scoring() -> None:
    import compute_cost.attempt_dossier as dossier

    def persist_every_attempt(spec: Any, classification: dict[str, Any]) -> bool:
        del spec, classification
        return True

    dossier.should_persist_attempt_dossier = persist_every_attempt

    if getattr(dossier.build_attempt_dossier, "_full_score_vector", False):
        return
    original_build = dossier.build_attempt_dossier

    def build_with_vector(**kwargs: Any) -> dict[str, Any]:
        result = original_build(**kwargs)
        telemetry = list(copy.deepcopy(kwargs.get("telemetry_after") or []))
        result["score_vector"] = build_score_vector(
            kwargs["case"],
            kwargs["scoring"],
            kwargs["classification"],
            kwargs["generation"],
            telemetry=telemetry,
        )
        return result

    build_with_vector._full_score_vector = True
    dossier.build_attempt_dossier = build_with_vector


def _call_context(runner: Any, stage: str, case_id: str) -> tuple[str, str | None, str | None]:
    category = str(getattr(runner, "_call_category_context", "") or stage)
    family = getattr(runner, "_call_family_context", None)
    scenario = getattr(runner, "_call_scenario_context", None)
    if family is None and "auto-" not in case_id:
        family = getattr(runner, "_active_family", None)
    if scenario is None and case_id.startswith("auto-"):
        parts = case_id.split("-s", 1)
        if parts:
            scenario = parts[0].removeprefix("auto-")
    return category, None if family is None else str(family), None if scenario is None else str(scenario)


def _install_call_ledger() -> None:
    import compute_cost.runner_core as core

    current = core.BenchmarkRunner._invoke_generation
    if getattr(current, "_full_call_ledger", False):
        return
    original = current

    def invoke_with_ledger(
        self: Any,
        *,
        stage: str,
        case_id: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        request_fields: dict[str, Any] | None = None,
    ):
        limit = int(self.config.get("limits", {}).get("max_model_calls_per_run", 700))
        run_id = self.store.run_id
        ledger = getattr(self, "_call_ledger", None)
        ledger_run_id = getattr(self, "_call_ledger_run_id", None)
        if (
            not isinstance(ledger, CallLedger)
            or ledger.limit != limit
            or ledger_run_id != run_id
        ):
            ledger = CallLedger(limit=limit)
            self._call_ledger = ledger
            self._call_ledger_run_id = run_id

        category, family, scenario = _call_context(self, stage, case_id)
        try:
            ordinal = ledger.authorize(category, family=family, scenario=scenario)
        except CallBudgetExceeded:
            self._model_call_counts[run_id] = max(
                int(self._model_call_counts.get(run_id, 0)), limit
            )
            result = original(
                self,
                stage=stage,
                case_id=case_id,
                messages=messages,
                options=options,
                request_fields=request_fields,
            )
            self.store.write_json(
                "call-ledger.json",
                ledger.snapshot(),
                producer="call-ledger",
                stage="accounting",
            )
            return result

        result = original(
            self,
            stage=stage,
            case_id=case_id,
            messages=messages,
            options=options,
            request_fields=request_fields,
        )
        generation, invocation, refs = result
        self.store.append_jsonl(
            "model-call-request-index.jsonl",
            {
                "call_ordinal": ordinal,
                "category": category,
                "family": family,
                "scenario": scenario,
                "stage": stage,
                "case_id": case_id,
                "request_id": refs.get("request_id"),
                "runtime_ok": bool(generation.get("ok", False)),
                "request_ref": copy.deepcopy(refs.get("request")),
                "response_ref": copy.deepcopy(refs.get("response")),
                "native_request_fields": copy.deepcopy(invocation.get("request_fields") or {}),
            },
        )
        self.store.write_json(
            "call-ledger.json",
            ledger.snapshot(),
            producer="call-ledger",
            stage="accounting",
        )
        return result

    invoke_with_ledger._full_call_ledger = True
    core.BenchmarkRunner._invoke_generation = invoke_with_ledger


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _join_dossier_vectors(root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for source in rows:
        row = copy.deepcopy(source)
        if isinstance(row.get("score_vector"), dict):
            joined.append(row)
            continue
        exp = row.get("experiment") if isinstance(row.get("experiment"), dict) else {}
        exp_id = exp.get("experiment_id")
        if exp_id is not None:
            safe = str(exp_id).replace("/", "-").replace("\\", "-")
            dossier = _read_json(root / "attempt-dossiers" / f"{safe}.json")
            vector = dossier.get("score_vector")
            if isinstance(vector, dict):
                row["score_vector"] = copy.deepcopy(vector)
        joined.append(row)
    return joined


def _render_task_table(scorecard: list[dict[str, Any]]) -> str:
    lines = [
        "# Fixed Task Scorecard",
        "",
        "| Family | Task | Level | Reasoning | Semantic | Format | Procedure | Runtime | Result |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in scorecard:
        vector = row.get("score_vector") or {}
        lines.append(
            f"| {row.get('family_id')} | {row.get('task_id')} | L{row.get('difficulty_level')} | "
            f"{row.get('reasoning_role')} | {vector.get('semantic_correctness')} | "
            f"{vector.get('contract_format_compliance')} | {vector.get('tool_procedure_compliance')} | "
            f"{vector.get('runtime_validity')} | {row.get('result_class')} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_retry_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Retry Deltas",
        "",
        "| Parent | Child | Family | Task | Level | Reasoning | Budget delta | Result transition | Semantic delta |",
        "|---|---|---|---|---:|---|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('parent_experiment_id')} | {row.get('child_experiment_id')} | {row.get('family_id')} | "
            f"{row.get('task_id')} | L{row.get('difficulty_level')} | {row.get('reasoning_role')} | "
            f"{row.get('generation_budget_delta')} | {row.get('result_class_before')} -> {row.get('result_class_after')} | "
            f"{(row.get('score_delta') or {}).get('semantic_correctness')} |"
        )
    if not rows:
        lines.append("| none | none | none | none | - | - | - | - | - |")
    return "\n".join(lines).rstrip() + "\n"


def _install_full_campaign_and_reports() -> None:
    import compute_cost.capability_campaign as campaign
    import compute_cost.run_synthesis as synthesis

    from .comparability_report import build_comparability_report, render_reasoning_comparison
    from .full_run import run_full_comparability_campaign
    from .run_integrity import reconcile_run_integrity

    legacy_campaign = campaign.run_capability_campaign

    def dispatch_campaign(runner, cases):
        full_cfg = runner.config.get("full_comparability") or {}
        autonomy = runner.config.get("autonomous_simulation") or {}
        secondary = (
            runner.config.get("reasoning_curves") or {},
            runner.config.get("recovery_lab") or {},
            runner.config.get("robustness_lab") or {},
            runner.config.get("compound_lab") or {},
        )
        full_mode = (
            full_cfg.get("enabled") is True
            and autonomy.get("enabled") is True
            and not any(block.get("enabled") is True for block in secondary)
        )
        if full_mode:
            return run_full_comparability_campaign(runner, cases)
        return legacy_campaign(runner, cases)

    dispatch_campaign._full_comparability_dispatch = True
    campaign.run_capability_campaign = dispatch_campaign

    current = synthesis.build_cost_value_outputs
    if getattr(current, "_full_comparability_reports", False):
        return
    original = current

    def with_full_reports(model, rows, frontiers, run_dir):
        cost_map, value_map = original(model, rows, frontiers, run_dir)
        root = Path(run_dir)
        fixed_path = root / "fixed-capability-observations.jsonl"
        if not fixed_path.is_file():
            return cost_map, value_map

        fixed_rows = _join_dossier_vectors(root, _read_jsonl(fixed_path))
        cell_payload = _read_json(root / "comparison-cells.json")
        cells = cell_payload.get("cells") if isinstance(cell_payload.get("cells"), list) else []
        report = build_comparability_report(model, fixed_rows, comparison_cells=cells)
        autonomous = _read_json(root / "autonomous-simulation.json")
        report["autonomous"] = autonomous

        (root / "task-scorecard.json").write_text(
            json.dumps(report["task_scorecard"], ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (root / "task-scorecard.md").write_text(
            _render_task_table(report["task_scorecard"]), encoding="utf-8"
        )
        (root / "reasoning-comparison.json").write_text(
            json.dumps(report["reasoning_comparison"], ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (root / "reasoning-comparison.md").write_text(
            render_reasoning_comparison(report), encoding="utf-8"
        )
        (root / "retry-deltas.json").write_text(
            json.dumps(report["retry_deltas"], ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (root / "retry-deltas.md").write_text(
            _render_retry_table(report["retry_deltas"]), encoding="utf-8"
        )

        integrity = reconcile_run_integrity(root)
        report["run_integrity"] = integrity
        report["model_summary"]["run_integrity_ok"] = bool(integrity.get("ok"))
        report["model_summary"]["autonomous_tested"] = bool(
            autonomous.get("semantic_turns", 0)
        )
        (root / "run-integrity.json").write_text(
            json.dumps(integrity, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (root / "scorecard.json").write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        summary_md = render_reasoning_comparison(report).rstrip()
        summary_md += "\n\n## Run integrity\n\n"
        summary_md += f"Status: {'PASS' if integrity.get('ok') else 'FAIL'}\n\n"
        summary_md += "Detailed fixed tasks: `task-scorecard.md`; retries: `retry-deltas.md`; autonomy: `autonomous-simulation.md`.\n"
        (root / "scorecard.md").write_text(summary_md, encoding="utf-8")
        return cost_map, value_map

    with_full_reports._full_comparability_reports = True
    synthesis.build_cost_value_outputs = with_full_reports


def install() -> None:
    import compute_cost.config as config

    config.DEFAULT_CONFIG["limits"]["max_model_calls_per_run"] = 700
    config.DEFAULT_CONFIG.setdefault("full_comparability", {}).update(
        {
            "enabled": True,
            "fixed_levels": [2, 5, 8, 10],
            "hard_call_limit": 700,
            "protect_fixed_core": True,
        }
    )
    _install_attempt_scoring()
    _install_call_ledger()
    _install_full_campaign_and_reports()
