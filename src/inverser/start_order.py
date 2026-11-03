import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field
from scipy.stats import binomtest

from inverser.azure_study import MODELS
from inverser.baselines import SearchConfig, geometry_actions, sampled_actions, search_goal
from inverser.diagnostic_panel import FAMILIES, load_synthetic_panel_call
from inverser.diagnostics import PairedMeasurement, summarize_paired
from inverser.evaluation import BudgetLimits, TrialLedger
from inverser.physics import SnookerSimulator
from inverser.planning_pilot import (
    PlanningCase,
    PlanningPilotConfig,
    _judge_checkpoint,
    select_planning_cases,
)
from inverser.runtime import fingerprint, write_json
from inverser.schemas import CueAction, Record
from inverser.study import GoalInput
from inverser.study_calls import load_job, lock_run, producer_hashes, save_job
from inverser.study_runner import JobEnvelope

LOGGER = logging.getLogger(__name__)
# Pre-registered from the v1 trace audit: with the v1 warm-start set and the model start moved
# last, every model is expected to match geometry at 8/32 queries and stay at 12 of 32 at 128.
PREDICTED_LAST = {8: 10, 32: 12, 128: 12}


class StartOrderConfig(Record):
    schema_version: Literal["start-order-ablation-v1"] = "start-order-ablation-v1"
    layouts: int = Field(default=32, ge=4, le=32)
    seed: int = Field(default=20261017, ge=0)
    checkpoints: tuple[int, ...] = (8, 32, 128)
    workers: int = Field(default=4, ge=1, le=4)


def _arms(case: PlanningCase, goal: GoalInput, config: StartOrderConfig) -> dict[str, list[Any]]:
    """Each arm: (explicit start list, max_starts); None marks an invalid model proposal."""
    horizon = config.checkpoints[-1]
    seed = int(fingerprint((config.seed, case.source_group))[:8], 16)
    base = SearchConfig(method="geometry_powell", seed=seed, max_objective_calls=horizon)
    geometry = geometry_actions(goal, base)
    sobol = sampled_actions(base, base.max_starts)
    arms: dict[str, list[Any]] = {
        "geometry_powell": [(), 4],
        "geometry_g1_only": [(geometry[0],) if geometry else None, 1],
    }
    for index, start in enumerate(sobol, start=1):
        arms[f"sobol_s{index}_only"] = [(start,), 1]
    for seed_record in case.native:
        model, action = seed_record.model, seed_record.action
        if action is None:
            for name in ("first", "last", "only"):
                arms[f"{model}_{name}"] = [None, 4 if name != "only" else 1]
            continue
        warm = ((action,) + geometry + sobol)[:4]
        arms[f"{model}_first"] = [(action,), 4]
        arms[f"{model}_last"] = [warm[1:] + (action,), 4]
        arms[f"{model}_only"] = [(action,), 1]
    return arms


def _run_case(
    panel: Path,
    output: Path,
    reference: Path,
    case: PlanningCase,
    config: StartOrderConfig,
    run_hash: str,
) -> list[dict[str, Any]]:
    goal = load_synthetic_panel_call(panel, case.request).input
    if not isinstance(goal, GoalInput):
        raise ValueError("Start-order ablation needs public goals")
    horizon = config.checkpoints[-1]
    folder = output / "cases" / case.request.case_id
    arms = _arms(case, goal, config)
    ledger = TrialLedger(
        folder / "ledger.sqlite",
        BudgetLimits(
            max_model_requests=0,
            max_physics_evaluations=len(arms) * (horizon + len(config.checkpoints)) + 4,
        ),
    )
    simulator: SnookerSimulator | None = None
    jobs = []
    try:
        for arm, (starts, max_starts) in arms.items():
            path = folder / f"{arm}.json"
            cached = load_job(path, run_hash)
            if cached is not None:
                jobs.append(cached)
                continue
            if simulator is None:
                simulator = SnookerSimulator(case.physics)
            seed = int(fingerprint((config.seed, case.source_group))[:8], 16)
            points: list[dict[str, object]] = []
            attempts: list[CueAction] = []
            if starts is None:
                points = [
                    {
                        "query_budget": budget,
                        "success": False,
                        "error_code": "invalid_or_missing_start",
                    }
                    for budget in config.checkpoints
                ]
            else:
                search = search_goal(
                    goal,
                    SearchConfig(
                        method="geometry_powell",
                        seed=seed,
                        max_objective_calls=horizon,
                        max_starts=max_starts,
                    ),
                    simulator,
                    ledger,
                    namespace=f"{run_hash}/{case.request.case_id}/{arm}",
                    initial_actions=starts,
                )
                attempts = [attempt.action for attempt in search.attempts]
                for budget in config.checkpoints:
                    prefix = [a for a in search.attempts[:budget] if a.loss_radii is not None]
                    best = min(prefix, key=lambda a: a.loss_radii or 0.0) if prefix else None
                    point = _judge_checkpoint(
                        goal,
                        best.action if best else None,
                        simulator,
                        ledger,
                        f"{run_hash}/{case.request.case_id}/hidden-judge",
                        budget,
                        len(search.attempts[:budget]),
                    )
                    points.append(
                        {
                            "query_budget": budget,
                            "success": bool(point.score and point.score.success),
                            "loss_radii": point.loss_radii,
                            "objective_calls": point.objective_calls,
                        }
                    )
            reproduces = None
            v1_method = {"geometry_powell": "geometry_powell"}.get(arm) or (
                f"{arm.removesuffix('_first')}_powell" if arm.endswith("_first") else None
            )
            v1_path = reference / "cases" / case.request.case_id / f"{v1_method}.json"
            if starts is not None and v1_method is not None and v1_path.exists():
                v1: Any = JobEnvelope.model_validate_json(v1_path.read_text()).payload["search"]
                reproduces = (
                    v1 is not None
                    and [CueAction.model_validate(item["action"]) for item in v1["attempts"]]
                    == attempts
                )
            payload = {
                "case_id": case.request.case_id,
                "source_group": case.source_group,
                "family": case.family,
                "arm": arm,
                "starts": max_starts,
                "objective_calls": len(attempts),
                "reproduces_v1_trace": reproduces,
                "checkpoints": points,
            }
            save_job(path, run_hash, payload)
            jobs.append(payload)
        return jobs
    finally:
        ledger.close()


def run_start_order_ablation(
    panel: Path, native_run: Path, reference: Path, output: Path, config: StartOrderConfig
) -> dict[str, object]:
    cases = select_planning_cases(
        panel,
        native_run,
        PlanningPilotConfig(
            layouts=config.layouts, seed=config.seed, checkpoints=config.checkpoints
        ),
    )
    protocol: dict[str, object] = {
        "config": config.model_dump(mode="json"),
        "reference_manifest": json.loads((reference / "manifest.json").read_text())["run_hash"],
        "case_ids": [case.request.case_id for case in cases],
        "predicted_model_last": {str(k): v for k, v in PREDICTED_LAST.items()},
        "producer": producer_hashes("start_order.py", "baselines.py", "planning_pilot.py"),
        "new_model_requests": 0,
    }
    run_hash = lock_run(output, protocol, reserve_bytes=2_000_000_000)
    if config.workers == 1:
        results = [_run_case(panel, output, reference, case, config, run_hash) for case in cases]
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [
                executor.submit(_run_case, panel, output, reference, case, config, run_hash)
                for case in cases
            ]
            results = [future.result() for future in futures]
    jobs = [job for case_jobs in results for job in case_jobs]
    table = []
    for arm in sorted({job["arm"] for job in jobs}):
        arm_jobs = [job for job in jobs if job["arm"] == arm]
        for budget in config.checkpoints:
            for family in (None, *FAMILIES):
                points = [
                    point
                    for job in arm_jobs
                    if family is None or job["family"] == family
                    for point in job["checkpoints"]
                    if point["query_budget"] == budget
                ]
                successes = sum(bool(point["success"]) for point in points)
                interval = binomtest(successes, len(points)).proportion_ci()
                table.append(
                    {
                        "arm": arm,
                        "query_budget": budget,
                        "family": family,
                        "layouts": len(points),
                        "successes": successes,
                        "success_rate_ci95": [interval.low, interval.high],
                    }
                )

    def success(arm: str, budget: int) -> dict[str, tuple[str, float]]:
        return {
            job["case_id"]: (
                job["source_group"],
                float(
                    next(p["success"] for p in job["checkpoints"] if p["query_budget"] == budget)
                ),
            )
            for job in jobs
            if job["arm"] == arm
        }

    comparisons = []
    for model in MODELS:
        for left, right in (
            (f"{model}_last", f"{model}_first"),
            (f"{model}_last", "geometry_powell"),
            (f"{model}_only", "geometry_g1_only"),
        ):
            for budget in config.checkpoints:
                a, b = success(left, budget), success(right, budget)
                keys = sorted(a.keys() & b.keys())
                comparisons.append(
                    {
                        "arm_a": left,
                        "arm_b": right,
                        "query_budget": budget,
                        "difference": summarize_paired(
                            tuple(
                                PairedMeasurement(pair_id=k, group_id=a[k][0], a=a[k][1], b=b[k][1])
                                for k in keys
                            ),
                            seed=config.seed,
                        ).model_dump(mode="json"),
                    }
                )
    random_starts = {
        str(budget): float(
            np.mean(
                [
                    np.mean([success(f"sobol_s{i}_only", budget)[case][1] for i in range(1, 5)])
                    for case in success("sobol_s1_only", budget)
                ]
            )
        )
        for budget in config.checkpoints
    }
    reproduction = [
        job["reproduces_v1_trace"] for job in jobs if job["reproduces_v1_trace"] is not None
    ]
    observed_last = {
        model: {
            str(budget): sum(value for _, value in success(f"{model}_last", budget).values())
            for budget in config.checkpoints
        }
        for model in MODELS
    }
    report: dict[str, object] = {
        "status": "completed" if len(results) == len(cases) else "partial",
        "run_hash": run_hash,
        "layouts": len(cases),
        "completed_jobs": len(jobs),
        "v1_reference_traces_checked": len(reproduction),
        "v1_reference_traces_reproduced": sum(bool(value) for value in reproduction),
        "predicted_model_last": {str(k): v for k, v in PREDICTED_LAST.items()},
        "observed_model_last": observed_last,
        "mean_random_single_start_success_by_budget": random_starts,
        "by_arm_budget_family": table,
        "paired_comparisons": comparisons,
        "new_model_requests": 0,
    }
    write_json(output / "summary.json", report)
    return report
