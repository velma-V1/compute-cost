"""Test 1.2 tuning/compile run.

Consumes a completed Test 1.2 collection run and compiles a model-specific
adaptive harness in <= 6h15m. VALIDATION is used for tuning/confirmation;
TEST2_BLIND and TEST3_PROTECTED remain untouched.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

from .evidence import EvidenceStore
from .test1_campaign import _balanced_cases, _family, _fixture_id, partition_cases
from .test12_campaign import (
    Test12Campaign,
    _cost_value_frontier,
    _group_summary,
    _rank_mechanisms,
    build_intervention_bank,
    fresh_model_source,
)

TUNING_HARD_SECONDS = (6 * 60 * 60) + (15 * 60)
TUNING_ACTIVE_SECONDS = 6 * 60 * 60

TUNING_PHASES = (
    ("validation_baseline", 30 * 60),
    ("candidate_harness_screen", 75 * 60),
    ("successive_halving", 90 * 60),
    ("routing_and_boundary_tuning", 75 * 60),
    ("residual_failure_replay", 60 * 60),
    ("final_harness_confirmation", 30 * 60),
)

REQUIRED_COLLECTION_FILES = (
    "test1.2-observations.jsonl",
    "full-control-candidate-registry.json",
    "control-grammar-coverage.json",
    "mechanism-coverage-ledger.json",
    "cost-value-frontier-1.2.json",
    "activation-boundary-map.json",
    "negative-transfer-map-1.2.json",
    "fine-tuning-readiness-map-1.2.json",
    "real-tool-execution-map.json",
    "tuning-example-corpus.jsonl",
    "harness-policy-blueprint.json",
    "test1.2-handoff.json",
)

REQUIRED_TUNING_OUTPUTS = (
    "test1.2-tuning-plan.json",
    "candidate-harness-registry.json",
    "test1.2-tuning-observations.jsonl",
    "successive-halving-ledger.json",
    "compiled-harness-policy.json",
    "compiled-harness-validation.json",
    "do-not-use-registry.json",
    "fine-tuning-training-corpus.jsonl",
    "fine-tuning-qualification.json",
    "model-harness-card.json",
)

DEFAULT_TUNING_CONFIG = {
    "expected_calls": 4200,
    "safety_call_cap": 10000,
    "screen_candidates": 24,
    "screen_cases": 16,
    "halving_cases": [24, 48, 96],
    "final_candidates": 4,
    "final_repeats": 2,
    "minimum_validation_families": 30,
    "max_capability_regression_rate": 0.05,
    "minimum_positive_value": 0.0,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_collection(results_root: Path, run_id: str) -> dict[str, Any]:
    run_dir = results_root / run_id
    if not run_dir.is_dir():
        raise ValueError(f"collection run does not exist: {run_id}")
    problems = EvidenceStore(results_root, run_id).verify_manifest_paths(REQUIRED_COLLECTION_FILES)
    if problems:
        raise ValueError(f"collection consumed-artifact verification failed: {problems}")
    coverage = _read_json(run_dir / "control-grammar-coverage.json")
    if not coverage.get("all_declared_candidates_tested"):
        raise ValueError("collection did not exercise every declared control candidate")
    registry = _read_json(run_dir / "full-control-candidate-registry.json")
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "registry": registry,
        "coverage": coverage,
        "frontier": _read_json(run_dir / "cost-value-frontier-1.2.json"),
        "activation": _read_json(run_dir / "activation-boundary-map.json"),
        "negative": _read_json(run_dir / "negative-transfer-map-1.2.json"),
        "fine_tuning": _read_json(run_dir / "fine-tuning-readiness-map-1.2.json"),
        "real_tool": _read_json(run_dir / "real-tool-execution-map.json"),
        "corpus": _read_jsonl(run_dir / "tuning-example-corpus.jsonl"),
        "blueprint": _read_json(run_dir / "harness-policy-blueprint.json"),
        "handoff": _read_json(run_dir / "test1.2-handoff.json"),
    }


def build_tuning_plan(cases: list[dict[str, Any]], *, collection_run: str) -> dict[str, Any]:
    parts = partition_cases(cases)
    return {
        "schema_version": 1,
        "campaign": "model-harness-compiler-test1.2-tuning",
        "collection_run": collection_run,
        "wall_clock_seconds": TUNING_HARD_SECONDS,
        "active_model_seconds": TUNING_ACTIVE_SECONDS,
        "phases": [{"name": n, "seconds": s} for n, s in TUNING_PHASES],
        "allowed_partitions": ["VALIDATION"],
        "prohibited_partitions": ["DISCOVERY", "TEST2_BLIND", "TEST3_PROTECTED"],
        "partition_counts": {name: len(rows) for name, rows in parts.items()},
        "objective": "compile the smallest adaptive harness that maximizes validated capability gain and minimizes regressions, model calls, tokens, and latency",
        "required_outputs": list(REQUIRED_TUNING_OUTPUTS),
        "total_two_run_hard_ceiling_seconds": TUNING_HARD_SECONDS + ((6 * 60 * 60) + (15 * 60)),
    }


def validate_tuning_plan(plan: dict[str, Any]) -> None:
    if int(plan["wall_clock_seconds"]) != TUNING_HARD_SECONDS:
        raise ValueError("tuning hard ceiling must be 6h15m")
    if sum(int(row["seconds"]) for row in plan["phases"]) != TUNING_ACTIVE_SECONDS:
        raise ValueError("tuning active phases must total six hours")
    if plan["allowed_partitions"] != ["VALIDATION"]:
        raise ValueError("tuning may use VALIDATION only")
    if set(plan["prohibited_partitions"]) != {"DISCOVERY","TEST2_BLIND","TEST3_PROTECTED"}:
        raise ValueError("tuning partition contract changed")
    if int(plan["total_two_run_hard_ceiling_seconds"]) >= 14 * 60 * 60:
        raise ValueError("two-run model-to-harness compiler exceeds 14-hour target")


def _candidate_registry(collection: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    by_id = {
        str(row["id"]): copy.deepcopy(row)
        for row in (collection.get("registry") or {}).get("candidates", [])
        if isinstance(row, dict) and row.get("id")
    }
    ranked = []
    for row in (collection.get("frontier") or {}).get("ranked", []):
        ident = str(row.get("intervention_id") or "")
        if ident not in by_id:
            continue
        if row.get("classification") == "CAPABILITY_HARM":
            continue
        ranked.append((float(row.get("net_value", 0.0)), float(row.get("value_per_call",0.0)), ident))
    ranked.sort(reverse=True)
    result = []
    for _, __, ident in ranked[:limit]:
        item = by_id[ident]
        item["collection_rank_source"] = next(
            (row for row in (collection.get("frontier") or {}).get("ranked", []) if row.get("intervention_id") == ident),
            {},
        )
        result.append(item)
    return result


def _route_map(candidates: list[dict[str, Any]]) -> dict[str, str]:
    preferences = {
        "TOOL": {"TOOL_POLICY","VERIFICATION"},
        "STATE": {"STATE_TRACKING","MEMORY"},
        "EVIDENCE": {"PROMPT_CONTROL","CONTEXT_SELECTION_COMPRESSION"},
        "FORMAT": {"PROMPT_CONTROL","VERIFICATION"},
        "PLAN": {"PLANNING","DELEGATION"},
        "VERIFY": {"VERIFICATION","CRITIQUE","RETRY_RECOVERY"},
    }
    result = {"DIRECT": "DIRECT"}
    for route, cats in preferences.items():
        found = next((row for row in candidates if row.get("category") in cats), None)
        result[route] = str(found["id"]) if found else "DIRECT"
    return result


def _policy_candidates(collection: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    base = _candidate_registry(collection, limit)
    policies = [{"policy_id":"DIRECT","mode":"direct","intervention_id":None}]
    for row in base:
        policies.append({
            "policy_id":"STATIC-"+str(row["id"]),
            "mode":"static",
            "intervention_id":str(row["id"]),
            "intervention":copy.deepcopy(row),
        })
    route_map = _route_map(base)
    policies.append({
        "policy_id":"ROUTER-COMPILED",
        "mode":"router",
        "route_map":route_map,
        "tool_execution_policy":tool_execution_policy,
    })
    policies.append({
        "policy_id":"RISK-GATED-COMPILED",
        "mode":"risk_gate",
        "fallback_intervention_id": next(
            (str(row["id"]) for row in base if row.get("category") in {"VERIFICATION","CRITIQUE","RETRY_RECOVERY"}),
            None,
        ),
    })
    return policies


def _score_policy_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n":0,"mean_delta":0.0,"regression_rate":1.0,"mean_calls":0.0,"net_value":-999.0}
    deltas=[float(row.get("delta") or 0.0) for row in rows]
    regressions=sum(1 for value in deltas if value < 0)
    calls=[float(row.get("model_calls") or 0.0) for row in rows]
    mean_delta=mean(deltas)
    regression_rate=regressions/len(rows)
    mean_calls=mean(calls)
    return {
        "n":len(rows),
        "mean_delta":mean_delta,
        "median_delta":median(deltas),
        "wins":sum(1 for value in deltas if value>0),
        "losses":regressions,
        "regression_rate":regression_rate,
        "mean_calls":mean_calls,
        "net_value":mean_delta - 0.08*mean_calls - 2.0*regression_rate,
        "families":sorted({str(row.get("family_id")) for row in rows}),
    }


class TuningRun:
    def __init__(self, runner: Any, cases: list[dict[str, Any]], collection: dict[str, Any], *, clock: Callable[[],float]=time.monotonic, started: float|None=None):
        self.runner=runner
        self.cases=cases
        self.collection=collection
        self.clock=clock
        self.start=clock() if started is None else float(started)
        self.active_end=self.start+TUNING_ACTIVE_SECONDS
        self.parts=partition_cases(cases)
        self.validation=self.parts["VALIDATION"]
        self.cfg={**DEFAULT_TUNING_CONFIG, **copy.deepcopy((runner.config.get("test12_tuning") or {}))}
        source=fresh_model_source(cases)
        source["baselines"]={}
        self.campaign=Test12Campaign(runner,cases,source,clock=clock,started_monotonic=self.start)
        self.campaign.allowed_partitions={"VALIDATION"}
        # Replace campaign bank with exact collection candidates so no mechanism
        # definition drifts between collection and tuning.
        collected=[
            copy.deepcopy(row)
            for row in (collection.get("registry") or {}).get("candidates",[])
            if isinstance(row,dict) and row.get("id")
        ]
        self.campaign.interventions=collected
        self.campaign.intervention_by_id={str(row["id"]):row for row in collected}
        self.rows=[]

    def can_start(self, deadline: float) -> bool:
        return self.clock() < min(deadline,self.active_end)

    def baseline(self, case: dict[str,Any], deadline: float, seed: int=42) -> dict[str,Any]|None:
        return self.campaign.control(case,deadline,seed=seed,force=False)

    def _router_choice(self, case: dict[str,Any], deadline: float, seed: int) -> tuple[str,dict[str,Any]|None]:
        aux=self.campaign._aux(
            case,deadline,stage="compiled-router",
            messages=[{"role":"user","content":str(case["prompt"])+"\n\nClassify with exactly one token: TOOL, STATE, EVIDENCE, FORMAT, PLAN, VERIFY, DIRECT."}],
            intervention={"id":"COMPILED-ROUTER","reasoning_effort":None},
            seed=seed,call_index=1,
        )
        if aux is None:
            return "DIRECT",None
        return self.campaign._router_choice(str(aux.get("text") or "")),aux

    def run_policy(self, policy: dict[str,Any], case: dict[str,Any], deadline: float, *, seed: int) -> dict[str,Any]|None:
        control=self.baseline(case,deadline,seed)
        if control is None:
            return None
        control_score=float(control.get("score") or 0.0)
        policy_id=str(policy["policy_id"])
        if policy["mode"]=="direct":
            row={
                "schema_version":1,"policy_id":policy_id,"fixture_id":_fixture_id(case),
                "family_id":_family(case),"seed":seed,"control_score":control_score,
                "score":control_score,"delta":0.0,"model_calls":0,"route":"DIRECT",
                "selected_intervention_id":None,
            }
        else:
            router_aux=None
            selected=None
            route=None
            if policy["mode"]=="static":
                selected=policy["intervention"]
            elif policy["mode"]=="router":
                route,router_aux=self._router_choice(case,deadline,seed)
                ident=(policy.get("route_map") or {}).get(route,"DIRECT")
                selected=self.campaign.intervention_by_id.get(str(ident)) if ident!="DIRECT" else None
            elif policy["mode"]=="risk_gate":
                route,router_aux=self._router_choice(case,deadline,seed)
                ident=policy.get("fallback_intervention_id") if route!="DIRECT" else None
                selected=self.campaign.intervention_by_id.get(str(ident)) if ident else None

            if selected is None:
                score=control_score
                calls=1 if router_aux else 0
                row={
                    "schema_version":1,"policy_id":policy_id,"fixture_id":_fixture_id(case),
                    "family_id":_family(case),"seed":seed,"control_score":control_score,
                    "score":score,"delta":0.0,"model_calls":calls,"route":route or "DIRECT",
                    "selected_intervention_id":None,
                }
            else:
                trial=self.campaign.treatment(case,deadline,phase="test1.2_tuning",intervention=selected,seed=seed)
                if trial is None:
                    return None
                calls=int(trial.get("model_calls_per_application") or 0)+(1 if router_aux else 0)
                row={
                    "schema_version":1,"policy_id":policy_id,"fixture_id":_fixture_id(case),
                    "family_id":_family(case),"seed":seed,"control_score":control_score,
                    "score":float(trial.get("score") or 0.0),
                    "delta":float(trial.get("score") or 0.0)-control_score,
                    "model_calls":calls,"route":route,
                    "selected_intervention_id":selected.get("id"),
                    "trial":trial,
                }
        self.rows.append(row)
        self.runner.store.append_jsonl("test1.2-tuning-observations.jsonl",row)
        return row


def _balanced_validation(run: TuningRun, n: int) -> list[dict[str,Any]]:
    return _balanced_cases(run.validation,min(n,len(run.validation)))


def _evaluate(run: TuningRun, policies: list[dict[str,Any]], cases: list[dict[str,Any]], deadline: float, *, seeds: list[int]) -> dict[str,Any]:
    start=len(run.rows)
    for seed in seeds:
        for policy in policies:
            for case in cases:
                if not run.can_start(deadline):
                    break
                run.run_policy(policy,case,deadline,seed=seed)
    rows=run.rows[start:]
    grouped=defaultdict(list)
    for row in rows:
        grouped[str(row["policy_id"])].append(row)
    return {key:_score_policy_rows(values) for key,values in grouped.items()}


def _top_policies(registry: list[dict[str,Any]], scores: dict[str,Any], keep: int) -> list[dict[str,Any]]:
    ranked=sorted(
        registry,
        key=lambda p: float((scores.get(str(p["policy_id"])) or {}).get("net_value",-999)),
        reverse=True,
    )
    return ranked[:max(1,keep)]


def _fine_tuning_records(run: TuningRun, winner: dict[str,Any]) -> list[dict[str,Any]]:
    winner_id=str(winner["policy_id"])
    rows=[row for row in run.rows if row.get("policy_id")==winner_id]
    result=[]
    for row in rows:
        if float(row.get("control_score",0.0))>=1.0 or float(row.get("score",0.0))>=1.0:
            continue
        trial=row.get("trial") or {}
        result.append({
            "schema_version":1,
            "fixture_id":row.get("fixture_id"),
            "family_id":row.get("family_id"),
            "policy_id":winner_id,
            "selected_intervention_id":row.get("selected_intervention_id"),
            "failure_class":((trial.get("classification") or {}).get("result_class") if isinstance(trial,dict) else None),
            "task_text":trial.get("task_text") if isinstance(trial,dict) else None,
            "failed_response_text":trial.get("treatment_response_text") if isinstance(trial,dict) else None,
            "qualification_state":"RESIDUAL_AFTER_COMPILED_HARNESS",
        })
    return result


def run_test12_tuning(runner: Any, cases: list[dict[str,Any]], *, collection_run: str, clock: Callable[[],float]=time.monotonic, started_monotonic: float|None=None) -> list[dict[str,Any]]:
    assert runner.store is not None
    collection=load_collection(Path(runner.results_root),collection_run)
    plan=build_tuning_plan(cases,collection_run=collection_run)
    validate_tuning_plan(plan)
    runner.store.write_json("test1.2-tuning-plan.json",plan,producer="test1.2-tuning",stage="preflight")

    run=TuningRun(runner,cases,collection,clock=clock,started=started_monotonic)
    policies=_policy_candidates(collection,int(run.cfg["screen_candidates"]))
    runner.store.write_json("candidate-harness-registry.json",{"schema_version":1,"policies":policies},producer="test1.2-tuning",stage="preflight")

    phase_start=run.start
    ledger=[]
    current=policies
    aggregate_scores={}
    for phase_name,seconds in TUNING_PHASES:
        deadline=min(run.active_end,phase_start+seconds)
        if phase_name=="validation_baseline":
            cases0=_balanced_validation(run,min(len(run.validation),128))
            for case in cases0:
                if not run.can_start(deadline): break
                run.baseline(case,deadline,seed=42)
            scores={}
        elif phase_name=="candidate_harness_screen":
            scores=_evaluate(run,current,_balanced_validation(run,int(run.cfg["screen_cases"])),deadline,seeds=[42])
            current=_top_policies(current,scores,max(8,len(current)//2))
        elif phase_name=="successive_halving":
            scores={}
            for n in run.cfg["halving_cases"]:
                if not run.can_start(deadline): break
                stage=_evaluate(run,current,_balanced_validation(run,int(n)),deadline,seeds=[42])
                scores.update(stage)
                current=_top_policies(current,stage,max(int(run.cfg["final_candidates"]),len(current)//2))
        elif phase_name=="routing_and_boundary_tuning":
            scores=_evaluate(run,current,_balanced_validation(run,min(96,len(run.validation))),deadline,seeds=[42,43])
            current=_top_policies(current,scores,int(run.cfg["final_candidates"]))
        elif phase_name=="residual_failure_replay":
            scores=_evaluate(run,current,_balanced_validation(run,len(run.validation)),deadline,seeds=[43])
        else:
            scores=_evaluate(run,current,_balanced_validation(run,len(run.validation)),deadline,seeds=[42,44][:int(run.cfg["final_repeats"])])
            current=_top_policies(current,scores,1)
        aggregate_scores.update(scores)
        ledger.append({"phase":phase_name,"remaining_policy_ids":[p["policy_id"] for p in current],"scores":scores})
        phase_start=deadline
        if not run.can_start(run.active_end): break

    winner=current[0] if current else {"policy_id":"DIRECT","mode":"direct"}
    winner_rows=[row for row in run.rows if row.get("policy_id")==winner["policy_id"]]
    winner_summary=_score_policy_rows(winner_rows)
    route_map=winner.get("route_map") or {}
    do_not_use=[
        key for key,value in ((collection.get("negative") or {}).get("effects") or {}).items()
        if int(value.get("capability_regressions",0))>0 or value.get("classification")=="CAPABILITY_HARM"
    ]
    tool_effects=(collection.get("real_tool") or {}).get("effects") or {}
    tool_execution_policy=None
    if tool_effects:
        tool_execution_policy=max(
            tool_effects,
            key=lambda key: (
                float((tool_effects.get(key) or {}).get("success_rate",0.0)),
                -float((tool_effects.get(key) or {}).get("mean_model_calls",999.0)),
            ),
        )
    compiled={
        "schema_version":1,
        "model":getattr(runner,"model",None),
        "collection_run":collection_run,
        "winner_policy":winner,
        "winner_validation":winner_summary,
        "route_map":route_map,
        "do_not_use":sorted(do_not_use),
        "direct_default_when_unmatched":True,
        "oracle_routing_prohibited":True,
        "hard_ceiling_total_seconds":TUNING_HARD_SECONDS+((6*60*60)+(15*60)),
    }
    runner.store.write_json("successive-halving-ledger.json",{"schema_version":1,"stages":ledger},producer="test1.2-tuning",stage="report")
    runner.store.write_json("compiled-harness-policy.json",compiled,producer="test1.2-tuning",stage="report")
    runner.store.write_json("compiled-harness-validation.json",{"schema_version":1,"winner":winner_summary,"all_policy_scores":aggregate_scores},producer="test1.2-tuning",stage="report")
    runner.store.write_json("do-not-use-registry.json",{"schema_version":1,"keys":sorted(do_not_use)},producer="test1.2-tuning",stage="report")

    ft=_fine_tuning_records(run,winner)
    for row in ft:
        runner.store.append_jsonl("fine-tuning-training-corpus.jsonl",row)
    by_family=defaultdict(int)
    for row in ft: by_family[str(row.get("family_id"))]+=1
    qualification={
        "schema_version":1,
        "residual_examples":len(ft),
        "recurrent_residual_families":{k:v for k,v in by_family.items() if v>=3},
        "weight_tuning_recommended":bool(any(v>=3 for v in by_family.values())),
        "rule":"weight tuning is recommended only for recurrent residual failures after the compiled harness is applied",
    }
    runner.store.write_json("fine-tuning-qualification.json",qualification,producer="test1.2-tuning",stage="report")
    runner.store.write_json("model-harness-card.json",{
        "schema_version":1,
        "model":getattr(runner,"model",None),
        "collection_run":collection_run,
        "compiled_policy":"compiled-harness-policy.json",
        "validated_policy_summary":winner_summary,
        "fine_tuning_qualification":qualification,
        "blind_partitions_touched":False,
        "total_two_run_hard_ceiling_hours":(TUNING_HARD_SECONDS+((6*60*60)+(15*60)))/3600.0,
    },producer="test1.2-tuning",stage="report")
    return run.rows
