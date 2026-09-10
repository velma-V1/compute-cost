"""Install the approved 700-call comparability contract additively.

The project already uses import-time compatibility patches. This module follows
that pattern so the stable runner/evidence pipeline remains the single execution
path while new accounting and scoring are layered on top.
"""

from __future__ import annotations

import copy
from typing import Any

from .call_ledger import CallBudgetExceeded, CallLedger
from .score_vector import build_score_vector


def _install_attempt_scoring() -> None:
    import compute_cost.attempt_dossier as dossier

    # The full-comparability contract requires exactly one dossier for every
    # executed model call, including clean first-attempt successes.
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
            ledger.authorize(category, family=family, scenario=scenario)
        except CallBudgetExceeded:
            # Keep the legacy runner's local counter aligned so its existing
            # pre-runtime guard creates retained budget-exhausted evidence without
            # invoking the runtime.
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
        self.store.write_json(
            "call-ledger.json",
            ledger.snapshot(),
            producer="call-ledger",
            stage="accounting",
        )
        return result

    invoke_with_ledger._full_call_ledger = True
    core.BenchmarkRunner._invoke_generation = invoke_with_ledger


def install() -> None:
    import compute_cost.config as config

    # compact_patch is installed first and intentionally used 360. The approved
    # full benchmark supersedes that run ceiling while leaving its retry law intact.
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
