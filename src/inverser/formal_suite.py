import json
import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import partial
from math import hypot
from pathlib import Path
from time import monotonic
from typing import Any, Literal, Self

import httpx
import numpy as np
from pydantic import Field, model_validator
from scipy.stats import binomtest

from inverser.bank_study import free_roll_resting_state
from inverser.baselines import BaselineMethod, SearchConfig, geometry_actions, search_goal
from inverser.diagnostic_panel import FAMILIES
from inverser.diagnostics import (
    ForwardScore,
    PairedMeasurement,
    score_forward,
    score_intervention_pair,
    summarize_paired,
)
from inverser.evaluation import BudgetLimits, TrialLedger, recorded_replay, request_model
from inverser.feedback_study import FeedbackCase
from inverser.formal_panel import (
    FormalRequest,
    formal_requests,
    formal_selection,
    load_formal_call,
    validate_formal_test,
)
from inverser.metrics import EndpointScore, endpoint_loss, score_endpoint
from inverser.model_adapters import ModelCall, ModelReply
from inverser.physics import SimulationFailure, SnookerSimulator
from inverser.planning_pilot import PlanningCheckpoint, _judge_checkpoint
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.schemas import CueAction, Layout, Record, ShotRecord
from inverser.study import ForwardSubmission, GoalInput, GoalSubmission
from inverser.study_calls import (
    TOLERANCE_RADII,
    BankRankingSubmission,
    ForwardPrediction,
    ForwardQuery,
    ReasoningEffort,
    StudyAdapter,
    StudyCall,
    azure_model_backends,
    credential_ready,
    endpoint_tolerance_profile,
    forward_tolerance_profile,
    guarded_request,
    load_job,
    lock_run,
    producer_hashes,
    request_study_model,
    require_data_path,
    save_job,
    usage_summary,
)

LOGGER = logging.getLogger(__name__)
Suite = Literal["forward", "goal", "bank"]
SUITE_KINDS = {
    "forward": ("forward",),
    "goal": ("goal",),
    "bank": ("bank_forecast", "bank_ranking"),
}
CLASSICAL_METHODS = ("geometry_direct", "random", "sobol", "geometry_powell")


class FormalSuiteConfig(Record):
    schema_version: Literal["formal-suite-v1"] = "formal-suite-v1"
    models: tuple[str, ...] = Field(min_length=1, max_length=8)
    reasoning_effort: ReasoningEffort = "low"
    suites: tuple[Suite, ...] = ("forward", "goal", "bank")
    include_bev: bool = True
    max_model_requests: int = Field(ge=1, le=60000)
    max_total_tokens: int = Field(ge=1, le=400_000_000)
    max_output_tokens: int = Field(default=32768, ge=256, le=32768)
    request_token_reservation: int = Field(default=40960, ge=8192)
    timeout_s: float = Field(default=900, gt=0, le=900)

    @model_validator(mode="after")
    def validate_design(self) -> Self:
        if len(set(self.models)) != len(self.models) or len(set(self.suites)) != len(self.suites):
            raise ValueError("Formal suite models and suites must be distinct")
        if not self.suites:
            raise ValueError("Choose at least one formal suite")
        if self.request_token_reservation < self.max_output_tokens:
            raise ValueError("Each token reservation must cover its completion cap")
        return self


class FormalRow(Record):
    model: str
    request_id: str
    case_id: str
    kind: str
    view: Literal["structured", "bev"]
    condition: str | None = None
    information: str | None = None
    candidate_id: str | None = None
    valid_output: bool
    elapsed_s: float = Field(ge=0)
    reply: ModelReply | None = None
    forward_score: ForwardScore | None = None
    forward_tolerances: dict[str, dict[str, bool]] | None = None
    goal_score: EndpointScore | None = None
    goal_tolerances: dict[str, bool] | None = None
    error_code: str | None = None


def suite_requests(panel: Path, config: FormalSuiteConfig) -> tuple[FormalRequest, ...]:
    kinds = {kind for suite in config.suites for kind in SUITE_KINDS[suite]}
    return tuple(
        request
        for request in formal_requests(panel)
        if request.kind in kinds and (config.include_bev or request.view == "structured")
    )


def _score_row(
    request: FormalRequest,
    call: ModelCall | StudyCall,
    reply: ModelReply,
    labels: dict[str, Any],
    ledger: TrialLedger,
    simulators: dict[str, SnookerSimulator],
    run_hash: str,
) -> dict[str, object]:
    if request.kind in ("forward", "bank_forecast"):
        if not isinstance(call, StudyCall) or not isinstance(call.input, ForwardQuery):
            raise ValueError("Forward scoring needs a forward study call")
        forward = call.input.forward
        if request.kind == "forward":
            item = labels["forward_labels"][request.condition]
        else:
            item = labels["bank"]["forecast_truth"][request.candidate_id]
            if item is None:
                return {}
        prediction = ForwardPrediction.model_validate_json(reply.content).submission()
        truth = ForwardSubmission.model_validate(item["truth"])
        radius = forward.physics.table.ball_radius_m
        score = score_forward(
            prediction,
            truth,
            initial=forward.initial,
            moving_ball_ids=tuple(item["moving_ball_ids"]),
            position_tolerance_m=radius,
            event_time_tolerance_s=0.02,
        )
        return {
            "forward_score": score,
            "forward_tolerances": forward_tolerance_profile(prediction, truth, score, radius),
        }
    if request.kind == "goal":
        goal = call.input
        if not isinstance(goal, GoalInput):
            raise ValueError("Goal scoring needs a public goal contract")
        record = ShotRecord.model_validate(labels["record"])
        key = fingerprint(record.physics.model_dump(mode="json"))
        if key not in simulators:
            simulators[key] = SnookerSimulator(record.physics)
        replay, _ = recorded_replay(
            goal.initial,
            GoalSubmission.model_validate_json(reply.content).action,
            simulators[key],
            ledger,
            namespace=f"{run_hash}/{request.case_id}/goal-judge",
            role="hidden_judge",
        )
        radius = goal.physics.table.ball_radius_m
        endpoint = score_endpoint(
            replay.goal, goal.goal, goal.tolerance_radii * radius, no_scratch=goal.no_scratch
        )
        return {
            "goal_score": endpoint,
            "goal_tolerances": endpoint_tolerance_profile(endpoint, radius),
        }
    return {}


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _resting(row: FormalRow, radius: float) -> float | None:
    if row.forward_score is None or row.forward_score.resting_max_position_error_m is None:
        return None
    return row.forward_score.resting_max_position_error_m / radius


def _paired_forward(
    left: dict[tuple[str, str], FormalRow],
    right: dict[tuple[str, str], FormalRow],
    groups: dict[str, str],
    radius: float,
) -> dict[str, object] | None:
    keys = sorted(left.keys() & right.keys())
    if not keys:
        return None
    errors = [
        (key, a, b)
        for key in keys
        if (a := _resting(left[key], radius)) is not None
        and (b := _resting(right[key], radius)) is not None
    ]

    def success(row: FormalRow) -> float:
        return float(bool(row.forward_tolerances and row.forward_tolerances["resting"]["5.0"]))

    return {
        "planned_pairs": len(keys),
        "error_pairs": len(errors),
        "median_resting_error_difference_radii": _median([a - b for _, a, b in errors]),
        "improved_pairs": sum(a < b for _, a, b in errors),
        "resting_error_difference": summarize_paired(
            tuple(
                PairedMeasurement(pair_id="/".join(key), group_id=groups[key[0]], a=a, b=b)
                for key, a, b in errors
            ),
            seed=20261021,
        ).model_dump(mode="json")
        if errors
        else None,
        "resting_success_5r_difference": summarize_paired(
            tuple(
                PairedMeasurement(
                    pair_id="/".join(key),
                    group_id=groups[key[0]],
                    a=success(left[key]),
                    b=success(right[key]),
                )
                for key in keys
            ),
            seed=20261021,
        ).model_dump(mode="json"),
    }


def _witness_rank(order: list[str], witness: str) -> int:
    return order.index(witness) + 1


def summarize_formal(panel: Path, rows: list[FormalRow]) -> dict[str, object]:
    selection = {str(item["case_id"]): item for item in formal_selection(panel)}
    groups = {case_id: str(item["source_group"]) for case_id, item in selection.items()}
    labels: dict[str, dict[str, Any]] = {
        case_id: json.loads((panel / "private" / f"{case_id}.json").read_text())
        for case_id in sorted({row.case_id for row in rows})
    }
    index = {request.request_id: request for request in formal_requests(panel)}
    radius = json.loads((panel / "manifest.json").read_text())["resolved_physics"][0]["config"][
        "ball_radius_m"
    ]
    forward_rows = [row for row in rows if row.kind == "forward"]
    arms: dict[tuple[str, str, str], dict[tuple[str, str], FormalRow]] = defaultdict(dict)
    for row in forward_rows:
        arms[row.model, str(row.information), row.view][row.case_id, str(row.condition)] = row
    forward = []
    for (model, information, view), arm in sorted(arms.items()):
        group = list(arm.values())
        scores = [row.forward_score for row in group if row.forward_score is not None]
        errors = [error for row in group if (error := _resting(row, radius)) is not None]

        def beats(name: str, arm_rows: list[FormalRow] = group) -> int:
            return sum(
                (error := _resting(row, radius)) is not None
                and (
                    reference := labels[row.case_id]["forward_labels"][row.condition]["baselines"][
                        name
                    ]
                )
                is not None
                and error < reference
                for row in arm_rows
            )

        forward.append(
            {
                "model": model,
                "information": information,
                "view": view,
                "requests": len(group),
                "valid_outputs": sum(row.valid_output for row in group),
                "median_resting_error_radii_valid_only": _median(errors),
                "beats_static_baseline": beats("static"),
                "beats_free_roll_baseline": beats("free_roll"),
                "frames_success_by_tolerance_radii": {
                    str(k): sum(
                        bool(row.forward_tolerances and row.forward_tolerances["frames"][str(k)])
                        for row in group
                    )
                    for k in TOLERANCE_RADII
                },
                "resting_success_by_tolerance_radii": {
                    str(k): sum(
                        bool(row.forward_tolerances and row.forward_tolerances["resting"][str(k)])
                        for row in group
                    )
                    for k in TOLERANCE_RADII
                },
                "contact_f1_valid_only": float(np.mean([s.contacts.f1 for s in scores]))
                if scores
                else None,
                "usage": usage_summary([row.reply for row in group if row.reply is not None]),
            }
        )
    contrasts = []
    for model in sorted({row.model for row in forward_rows}):
        native = arms.get((model, "native", "structured"), {})
        for name, other in (
            ("oracle_kinematics", arms.get((model, "oracle", "structured"), {})),
            ("bev_view", arms.get((model, "native", "bev"), {})),
        ):
            comparison = _paired_forward(other, native, groups, radius)
            if comparison is not None:
                contrasts.append({"contrast": name, "model": model, **comparison})
    interventions = []
    for (model, information, view), arm in sorted(arms.items()):
        responses, unchanged, joint_events = [], [], 0
        for case_id in sorted({case for case, _ in arm}):
            base, altered = (
                arm.get((case_id, "base_action")),
                arm.get((case_id, "intervened_action")),
            )
            if not all(
                row is not None and row.forward_score is not None and row.reply is not None
                for row in (base, altered)
            ):
                continue
            assert base is not None and altered is not None
            assert base.reply is not None and altered.reply is not None
            item = labels[case_id]["forward_labels"]
            truths = [
                ForwardSubmission.model_validate(item[name]["truth"])
                for name in ("base_action", "intervened_action")
            ]
            call = load_formal_call(panel, index[base.request_id])
            if not isinstance(call, StudyCall) or not isinstance(call.input, ForwardQuery):
                raise ValueError("Intervention scoring needs forward study calls")
            initial = call.input.forward.initial
            moving = tuple(
                sorted(
                    set(item["base_action"]["moving_ball_ids"])
                    | set(item["intervened_action"]["moving_ball_ids"])
                )
            )
            predicted = score_intervention_pair(
                ForwardPrediction.model_validate_json(base.reply.content).submission(),
                ForwardPrediction.model_validate_json(altered.reply.content).submission(),
                *truths,
                initial=initial,
                moving_ball_ids=moving,
                position_tolerance_m=radius,
                event_time_tolerance_s=0.02,
            )
            # Predicting no change: the response baseline an action-sensitive model must beat.
            baseline = score_intervention_pair(
                truths[0],
                truths[0],
                *truths,
                initial=initial,
                moving_ball_ids=moving,
                position_tolerance_m=radius,
                event_time_tolerance_s=0.02,
            )
            if predicted.moving_response_mean_error_m is not None:
                responses.append(predicted.moving_response_mean_error_m / radius)
            if baseline.moving_response_mean_error_m is not None:
                unchanged.append(baseline.moving_response_mean_error_m / radius)
            joint_events += predicted.joint_events_success
        interventions.append(
            {
                "model": model,
                "information": information,
                "view": view,
                "complete_pairs": len(responses),
                "median_response_error_radii": _median(responses),
                "no_change_baseline_median_response_error_radii": _median(unchanged),
                "joint_events_successes": joint_events,
            }
        )
    goals = []
    for model in sorted({row.model for row in rows if row.kind == "goal"}):
        group = [row for row in rows if row.kind == "goal" and row.model == model]
        successes = sum(bool(row.goal_score and row.goal_score.success) for row in group)
        interval = binomtest(successes, len(group)).proportion_ci()
        goals.append(
            {
                "model": model,
                "goals": len(group),
                "valid_outputs": sum(row.valid_output for row in group),
                "successes": successes,
                "success_rate_ci95": [interval.low, interval.high],
                "success_by_tolerance_radii": {
                    str(k): sum(
                        bool(row.goal_tolerances and row.goal_tolerances[str(k)]) for row in group
                    )
                    for k in TOLERANCE_RADII
                },
                "successes_by_family": {
                    family: sum(
                        bool(row.goal_score and row.goal_score.success)
                        for row in group
                        if selection[row.case_id]["family"] == family
                    )
                    for family in FAMILIES
                },
                "usage": usage_summary([row.reply for row in group if row.reply is not None]),
            }
        )
    return {
        "forward": forward,
        "paired_forward_contrasts": contrasts,
        "interventions": interventions,
        "goal_direct": goals,
        "bank": summarize_bank_ranking(panel, rows, labels, groups),
    }


def summarize_bank_ranking(
    panel: Path,
    rows: list[FormalRow],
    labels: dict[str, dict[str, Any]],
    groups: dict[str, str],
) -> dict[str, object]:
    """Witness rank in three rankings; invalid model outputs rank the witness last."""
    forecasts = {
        (row.model, row.view, row.case_id, row.candidate_id): row
        for row in rows
        if row.kind == "bank_forecast"
    }
    rankings = {
        (row.model, row.view, row.case_id): row for row in rows if row.kind == "bank_ranking"
    }
    results: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    size = 0
    for (model, view, case_id), ranking_row in sorted(rankings.items()):
        bank = labels[case_id]["bank"]
        order = [item["candidate_id"] for item in bank["bank"]["candidates"]]
        size = len(order)
        witness = str(bank["witness_candidate_id"])
        goal = GoalInput.model_validate(bank["goal"])
        diagonal = hypot(goal.physics.table.width_m, goal.physics.table.length_m)

        def loss(layout: Layout) -> float:
            return endpoint_loss(
                layout,
                goal.goal,
                ball_radius_m=goal.physics.table.ball_radius_m,
                table_diagonal_m=diagonal,
                no_scratch=goal.no_scratch,
            )

        predicted: dict[str, float | None] = {}
        for cid in order:
            row = forecasts.get((model, view, case_id, cid))
            predicted[cid] = (
                loss(
                    ForwardPrediction.model_validate_json(row.reply.content)
                    .submission()
                    .resting_state
                )
                if row is not None and row.valid_output and row.reply is not None
                else None
            )
        by_forecast = sorted(
            order,
            key=lambda cid: (predicted[cid] is None, predicted[cid] or 0.0, order.index(cid)),
        )
        results[model, view, "forecast_ranking"][case_id] = _witness_rank(by_forecast, witness)
        direct = (
            list(BankRankingSubmission.model_validate_json(ranking_row.reply.content).ranking)
            if ranking_row.valid_output and ranking_row.reply is not None
            else None
        )
        results[model, view, "direct_ranking"][case_id] = (
            _witness_rank(direct, witness) if direct is not None else len(order)
        )
        if "structured" == view and case_id not in results["reference", "structured", "free_roll"]:
            actions = {
                item["candidate_id"]: CueAction.model_validate(item["action"])
                for item in bank["bank"]["candidates"]
            }
            free = sorted(
                order,
                key=lambda cid: (
                    loss(free_roll_resting_state(goal.initial, actions[cid], goal.physics)),
                    order.index(cid),
                ),
            )
            results["reference", "structured", "free_roll"][case_id] = _witness_rank(free, witness)
    chance = {
        "top1": 1 / size if size else None,
        "top3": min(3, size) / size if size else None,
        "mean_reciprocal_rank": float(np.mean([1 / k for k in range(1, size + 1)]))
        if size
        else None,
    }
    methods = []
    for (model, method_view, method), ranks in sorted(results.items()):
        top1 = sum(rank == 1 for rank in ranks.values())
        interval = binomtest(top1, len(ranks)).proportion_ci()
        methods.append(
            {
                "model": model,
                "view": method_view,
                "method": method,
                "layouts": len(ranks),
                "top1": top1,
                "top1_ci95": [interval.low, interval.high],
                "top3": sum(rank <= 3 for rank in ranks.values()),
                "mean_reciprocal_rank": float(np.mean([1 / rank for rank in ranks.values()])),
            }
        )
    comparisons = []
    reference = results.get(("reference", "structured", "free_roll"), {})
    arms = sorted({(arm_model, arm_view) for arm_model, arm_view, _ in results})
    for model, arm_view in arms:
        if model == "reference":
            continue
        forecast_ranks = results.get((model, arm_view, "forecast_ranking"), {})
        direct_ranks = results.get((model, arm_view, "direct_ranking"), {})
        contrasts: tuple[tuple[str, dict[str, int], dict[str, int] | None], ...] = (
            ("forecast_minus_chance", forecast_ranks, None),
            ("direct_minus_chance", direct_ranks, None),
            ("forecast_minus_direct", forecast_ranks, direct_ranks),
            ("forecast_minus_free_roll", forecast_ranks, reference),
        )
        for name, left, right in contrasts:
            keys = sorted(left.keys() & (right.keys() if right is not None else left.keys()))
            chance_mrr = chance["mean_reciprocal_rank"]
            if not keys or chance_mrr is None:
                continue
            comparisons.append(
                {
                    "model": model,
                    "view": arm_view,
                    "comparison": name,
                    "reciprocal_rank_difference": summarize_paired(
                        tuple(
                            PairedMeasurement(
                                pair_id=case_id,
                                group_id=groups[case_id],
                                a=1 / left[case_id],
                                b=1 / right[case_id] if right is not None else chance_mrr,
                            )
                            for case_id in keys
                        ),
                        seed=20261021,
                    ).model_dump(mode="json"),
                }
            )
    return {"chance": chance, "methods": methods, "paired_comparisons": comparisons}


def run_formal_suite(
    panel: Path,
    source: Path,
    output: Path,
    config: FormalSuiteConfig,
    *,
    endpoint_index: int = 4,
    execute: bool = False,
    token_provider: Callable[[], str] | None = None,
    transport: httpx.BaseTransport | None = None,
    limit_jobs: int | None = None,
) -> dict[str, object]:
    validation = validate_formal_test(panel)
    requests = suite_requests(panel, config)
    planned = len(requests) * len(config.models)
    if config.max_model_requests < planned:
        raise ValueError("Request allocation must cover every formal request for every model")
    backends = azure_model_backends(
        source,
        endpoint_index,
        config.models,
        reasoning_effort=config.reasoning_effort,
        max_output_tokens=config.max_output_tokens,
        request_token_reservation=config.request_token_reservation,
        timeout_s=config.timeout_s,
    )
    goals = sum(request.kind == "goal" for request in requests)
    limits = BudgetLimits(
        max_model_requests=config.max_model_requests,
        max_total_tokens=config.max_total_tokens,
        max_physics_evaluations=goals * len(config.models),
    )
    protocol: dict[str, object] = {
        "formal_test_manifest_sha256": validation["manifest_sha256"],
        "azure_inventory_sha256": file_sha256(source),
        "endpoint_hash": fingerprint(backends[0].base_url),
        "config": config.model_dump(mode="json"),
        "backends": [backend.model_dump(mode="json") for backend in backends],
        "request_ids": [request.request_id for request in requests],
        "limits": limits.model_dump(mode="json"),
        "producer": producer_hashes(
            "formal_suite.py",
            "formal_panel.py",
            "study_calls.py",
            "model_adapters.py",
            "evaluation.py",
            "diagnostics.py",
            "metrics.py",
        ),
        "real_data_sent": False,
        "formal_evaluation": True,
    }
    run_hash = lock_run(output, protocol, reserve_bytes=500_000_000)
    if not execute:
        return {"status": "planned_no_calls", "run_hash": run_hash, "planned_requests": planned}
    if token_provider is None:
        raise ValueError("Azure execution requires a runtime bearer-token provider")
    token_provider()
    rows: list[FormalRow] = []
    labels: dict[str, dict[str, Any]] = {}
    simulators: dict[str, SnookerSimulator] = {}
    ledger = TrialLedger(output / "ledger.sqlite", limits)
    before = ledger.counts()["model_requests"]
    stop_error = None
    completed = 0
    try:
        with ExitStack() as stack:
            adapters = {
                backend.alias: stack.enter_context(
                    StudyAdapter(
                        backend.model_copy(update={"execution_authorized": True}),
                        token_provider=token_provider,
                        transport=transport,
                    )
                )
                for backend in backends
            }
            for request in requests:
                for model in config.models:
                    path = output / "jobs" / model / f"{request.request_id}.json"
                    cached = load_job(path, run_hash)
                    if cached is not None:
                        row = FormalRow.model_validate(cached)
                        if (row.model, row.request_id) != (model, request.request_id):
                            raise ValueError("Cached formal job identity changed")
                        rows.append(row)
                        continue
                    if stop_error or (limit_jobs is not None and completed >= limit_jobs):
                        continue
                    if not credential_ready(token_provider):
                        stop_error = "credential_unavailable"
                        continue
                    started = monotonic()
                    call = load_formal_call(panel, request)
                    outcome = guarded_request(
                        partial(request_model, adapters[model], call, ledger)
                        if isinstance(call, ModelCall)
                        else partial(request_study_model, adapters[model], call, ledger)
                    )
                    stop_error = outcome.stop_error
                    error_code = outcome.error_code
                    scored: dict[str, object] = {}
                    if outcome.reply is not None and error_code is None:
                        if request.case_id not in labels:
                            labels[request.case_id] = json.loads(
                                (panel / "private" / f"{request.case_id}.json").read_text()
                            )
                        try:
                            scored = _score_row(
                                request,
                                call,
                                outcome.reply,
                                labels[request.case_id],
                                ledger,
                                simulators,
                                run_hash,
                            )
                        except SimulationFailure:
                            error_code = "hidden_judge_simulation_failure"
                    row = FormalRow.model_validate(
                        {
                            "model": model,
                            "request_id": request.request_id,
                            "case_id": request.case_id,
                            "kind": request.kind,
                            "view": request.view,
                            "condition": request.condition,
                            "information": request.information,
                            "candidate_id": request.candidate_id,
                            "valid_output": outcome.error_code is None,
                            "elapsed_s": monotonic() - started,
                            "reply": outcome.reply,
                            "error_code": error_code,
                            **scored,
                        }
                    )
                    save_job(path, run_hash, row.model_dump(mode="json"))
                    rows.append(row)
                    completed += 1
                    LOGGER.info("Formal job %d/%d: %s/%s", len(rows), planned, model, request.kind)
        report: dict[str, object] = {
            "status": "stopped_service_error"
            if stop_error
            else ("completed" if len(rows) == planned else "partial"),
            "run_hash": run_hash,
            "stop_error": stop_error,
            "planned_requests": planned,
            "completed_jobs": len(rows),
            "new_model_requests": ledger.counts()["model_requests"] - before,
            "resource_counts": ledger.counts(),
            "physics_usage_by_role": ledger.physics_usage_by_role(),
            "usage": ledger.usage_totals(),
            **summarize_formal(panel, rows),
            "real_data_sent": False,
            "formal_evaluation": True,
        }
        write_json(output / "summary.json", report)
        return report
    finally:
        ledger.close()


def rescore_formal_runs(panel: Path, runs: tuple[Path, ...], output: Path) -> dict[str, object]:
    """Re-score finished runs from their saved jobs with the current scoring code, so every
    model in a comparison is summarized by the same code whichever version produced it."""
    require_data_path(output)
    validation = validate_formal_test(panel)
    scored: dict[str, object] = {}
    for run in runs:
        manifest = json.loads((run / "manifest.json").read_text())
        run_hash = manifest["run_hash"]
        rows = [
            FormalRow.model_validate(load_job(path, run_hash))
            for path in (run / "jobs").rglob("*.json")
        ]
        # The run's own request order keeps floating-point sums identical to its summary.
        order = {request_id: index for index, request_id in enumerate(manifest["request_ids"])}
        models = manifest["config"]["models"]
        rows.sort(key=lambda row: (order[row.request_id], models.index(row.model)))
        scored[run.name] = {
            "run_hash": run_hash,
            "completed_jobs": len(rows),
            **summarize_formal(panel, rows),
        }
    report: dict[str, object] = {
        "formal_test_manifest_sha256": validation["manifest_sha256"],
        "scoring_code": producer_hashes("formal_suite.py", "diagnostics.py", "metrics.py"),
        "runs": scored,
    }
    write_json(output, report)
    return report


def formal_feedback_cases(panel: Path) -> tuple[tuple[FeedbackCase, ...], dict[str, object]]:
    validation = validate_formal_test(panel)
    cases = []
    for item in formal_selection(panel):
        if not item["feedback"]:
            continue
        case_id = str(item["case_id"])
        labels = json.loads((panel / "private" / f"{case_id}.json").read_text())
        cases.append(
            FeedbackCase(
                case_id=case_id,
                source_group=str(item["source_group"]),
                family=str(item["family"]),
                goal=GoalInput.model_validate_json(
                    (panel / "public" / case_id / "goal.json").read_text()
                ),
                physics=ShotRecord.model_validate(labels["record"]).physics,
            )
        )
    return tuple(cases), {"formal_test_manifest_sha256": validation["manifest_sha256"]}


class FormalClassicalConfig(Record):
    schema_version: Literal["formal-classical-v1"] = "formal-classical-v1"
    seed: int = Field(default=20261021, ge=0)
    checkpoints: tuple[int, ...] = (8, 32, 128)
    workers: int = Field(default=4, ge=1, le=4)

    @model_validator(mode="after")
    def validate_checkpoints(self) -> Self:
        if not self.checkpoints or tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("Checkpoints must be distinct and increasing")
        if any(not 1 <= item <= 128 for item in self.checkpoints):
            raise ValueError("Checkpoints must be within 1-128 queries")
        return self


def _classical_case(
    panel: Path, output: Path, case_id: str, config: FormalClassicalConfig, run_hash: str
) -> list[dict[str, object]]:
    labels = json.loads((panel / "private" / f"{case_id}.json").read_text())
    record = ShotRecord.model_validate(labels["record"])
    goal = GoalInput.model_validate_json((panel / "public" / case_id / "goal.json").read_text())
    horizon = config.checkpoints[-1]
    ledger = TrialLedger(
        output / "cases" / case_id / "ledger.sqlite",
        BudgetLimits(
            max_model_requests=0,
            max_physics_evaluations=3 * horizon + 4 * len(config.checkpoints) + 4,
        ),
    )
    simulator: SnookerSimulator | None = None
    jobs = []
    try:
        for method in CLASSICAL_METHODS:
            path = output / "cases" / case_id / f"{method}.json"
            cached = load_job(path, run_hash)
            if cached is not None:
                jobs.append(cached)
                continue
            if simulator is None:
                simulator = SnookerSimulator(record.physics)
            seed = int(fingerprint((config.seed, record.group_id))[:8], 16)
            baseline: BaselineMethod = (
                "random"
                if method == "random"
                else "sobol"
                if method == "sobol"
                else "geometry_powell"
            )
            search_config = SearchConfig(method=baseline, seed=seed, max_objective_calls=horizon)
            namespace = f"{run_hash}/{case_id}"
            points: list[PlanningCheckpoint] = []
            if method == "geometry_direct":
                actions = geometry_actions(goal, search_config)
                points.append(
                    _judge_checkpoint(
                        goal,
                        actions[0] if actions else None,
                        simulator,
                        ledger,
                        namespace + "/hidden-judge",
                        0,
                        0,
                    )
                )
            else:
                search = search_goal(
                    goal, search_config, simulator, ledger, namespace=f"{namespace}/{method}"
                )
                for budget in config.checkpoints:
                    prefix = [a for a in search.attempts[:budget] if a.loss_radii is not None]
                    best = min(prefix, key=lambda a: a.loss_radii or 0.0) if prefix else None
                    points.append(
                        _judge_checkpoint(
                            goal,
                            best.action if best else None,
                            simulator,
                            ledger,
                            namespace + "/hidden-judge",
                            budget,
                            len(search.attempts[:budget]),
                        )
                    )
            payload = {
                "case_id": case_id,
                "family": labels["family"],
                "source_group": labels["source_group"],
                "method": method,
                "checkpoints": [point.model_dump(mode="json") for point in points],
            }
            save_job(path, run_hash, payload)
            jobs.append(payload)
        return jobs
    finally:
        ledger.close()


def run_formal_classical(
    panel: Path, output: Path, config: FormalClassicalConfig
) -> dict[str, object]:
    validation = validate_formal_test(panel)
    cases = [str(item["case_id"]) for item in formal_selection(panel)]
    protocol: dict[str, object] = {
        "formal_test_manifest_sha256": validation["manifest_sha256"],
        "config": config.model_dump(mode="json"),
        "methods": list(CLASSICAL_METHODS),
        "producer": producer_hashes(
            "formal_suite.py", "formal_panel.py", "baselines.py", "planning_pilot.py", "physics.py"
        ),
        "new_model_requests": 0,
        "formal_evaluation": True,
    }
    run_hash = lock_run(output, protocol, reserve_bytes=2_000_000_000)
    if config.workers == 1:
        results = [_classical_case(panel, output, case_id, config, run_hash) for case_id in cases]
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [
                executor.submit(_classical_case, panel, output, case_id, config, run_hash)
                for case_id in cases
            ]
            results = [future.result() for future in futures]
    jobs: list[dict[str, Any]] = [job for case_jobs in results for job in case_jobs]
    table = []
    for method in CLASSICAL_METHODS:
        method_jobs = [job for job in jobs if job["method"] == method]
        budgets = sorted(
            {point["query_budget"] for job in method_jobs for point in job["checkpoints"]}
        )
        for budget in budgets:
            for family in (None, *FAMILIES):
                points = [
                    point
                    for job in method_jobs
                    if family is None or job["family"] == family
                    for point in job["checkpoints"]
                    if point["query_budget"] == budget
                ]
                successes = sum(bool(p["score"] and p["score"]["success"]) for p in points)
                interval = binomtest(successes, len(points)).proportion_ci() if points else None
                table.append(
                    {
                        "method": method,
                        "query_budget": budget,
                        "family": family,
                        "layouts": len(points),
                        "successes": successes,
                        "success_rate_ci95": [interval.low, interval.high] if interval else None,
                        "median_loss_radii": _median(
                            [p["loss_radii"] for p in points if p["loss_radii"] is not None]
                        ),
                    }
                )
    report: dict[str, object] = {
        "status": "completed" if len(results) == len(cases) else "partial",
        "run_hash": run_hash,
        "layouts": len(cases),
        "completed_jobs": len(jobs),
        "new_model_requests": 0,
        "by_method_budget_family": table,
        "outcomes": dict(
            Counter(
                (
                    p["error_code"]
                    or (p["score"]["failure"] or "success" if p["score"] else "unscored")
                )
                for job in jobs
                for p in job["checkpoints"]
            )
        ),
        "formal_evaluation": True,
    }
    write_json(output / "summary.json", report)
    return report
