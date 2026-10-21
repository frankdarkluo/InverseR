import json
import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
from math import dist
from pathlib import Path
from time import monotonic
from typing import Literal, Self

import httpx
import numpy as np
from pydantic import Field, model_validator

from inverser.azure_study import MODELS, _score_row, select_pilot_requests
from inverser.bank_study import free_roll_resting_state
from inverser.diagnostic_panel import (
    PanelRequest,
    load_synthetic_panel_call,
    validate_diagnostic_panel,
)
from inverser.diagnostics import (
    ForwardScore,
    InterventionScore,
    PairedMeasurement,
    score_forward,
    score_intervention_pair,
    summarize_paired,
)
from inverser.evaluation import BudgetLimits, TrialLedger, recorded_replay, request_model
from inverser.metrics import EndpointScore
from inverser.model_adapters import ModelReply
from inverser.physics import SimulationFailure, SnookerSimulator
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.schemas import CueAction, Record, ShotRecord
from inverser.study import ForwardInput, ForwardSubmission, PredictedFrame
from inverser.study_calls import (
    TOLERANCE_RADII,
    OracleForwardInput,
    PostImpactKinematics,
    ReasoningEffort,
    StudyAdapter,
    StudyCall,
    credential_ready,
    endpoint_tolerance_profile,
    forward_tolerance_profile,
    guarded_request,
    load_job,
    lock_run,
    producer_hashes,
    request_study_model,
    save_job,
    study_azure_backends,
    usage_summary,
)

LOGGER = logging.getLogger(__name__)
Information = Literal["native", "oracle_post_impact_kinematics"]


class NativeProbeConfig(Record):
    schema_version: Literal["native-probe-v1"] = "native-probe-v1"
    design: Literal["effort_probe", "forward_intervention"] = "effort_probe"
    layouts: int = Field(default=8, ge=4, le=64)
    reasoning_efforts: tuple[ReasoningEffort, ...] = ("minimal", "low", "medium", "high")
    max_model_requests: int = Field(default=192, ge=1, le=4000)
    max_total_tokens: int = Field(default=2_000_000, ge=1, le=40_000_000)
    max_output_tokens: int = Field(default=32768, ge=256, le=32768)
    request_token_reservation: int = Field(default=40960, ge=8192)
    timeout_s: float = Field(default=900, gt=0, le=900)

    @model_validator(mode="after")
    def validate_design(self) -> Self:
        if self.layouts % 4:
            raise ValueError("Probe layouts must be balanced across four source families")
        efforts = self.reasoning_efforts
        if not efforts or len(set(efforts)) != len(efforts):
            raise ValueError("Reasoning efforts must be distinct and nonempty")
        if self.design == "effort_probe" and self.layouts > 8:
            raise ValueError("The effort probe is limited to the eight-layout API common subset")
        if self.request_token_reservation < self.max_output_tokens:
            raise ValueError("Each token reservation must cover its completion cap")
        if self.max_model_requests < self.planned_requests():
            raise ValueError("Request allocation must cover every planned arm, model and layout")
        return self

    def planned_requests(self) -> int:
        # B0: two forwards and one goal, structured. B1: two forwards x (structured, BEV, oracle).
        per_layout = 3 if self.design == "effort_probe" else 6
        return self.layouts * per_layout * len(self.reasoning_efforts) * len(MODELS)


class ProbeItem(Record):
    reasoning_effort: ReasoningEffort
    information: Information
    request: PanelRequest


class ProbeRow(Record):
    model: str
    reasoning_effort: ReasoningEffort
    information: Information
    request_id: str
    case_id: str
    task: Literal["synthetic_forward", "synthetic_goal"]
    condition: Literal["base_action", "intervened_action", "goal"]
    view: Literal["structured", "bev"]
    valid_output: bool
    elapsed_s: float = Field(ge=0)
    reply: ModelReply | None = None
    forward_score: ForwardScore | None = None
    forward_tolerances: dict[str, dict[str, bool]] | None = None
    goal_score: EndpointScore | None = None
    goal_tolerances: dict[str, bool] | None = None
    error_code: str | None = None


def probe_plan(panel: Path, config: NativeProbeConfig) -> tuple[ProbeItem, ...]:
    if config.design == "effort_probe":
        requests = tuple(
            request
            for request in select_pilot_requests(panel, config.layouts, design="paired")
            if request.view == "structured"
        )
    else:
        cases = [
            request.case_id
            for request in select_pilot_requests(panel, config.layouts, design="goal_structured")
        ]
        index = [
            PanelRequest.model_validate_json(line)
            for line in (panel / "request_index.jsonl").open()
        ]
        requests = tuple(
            request
            for case_id in cases
            for request in index
            if request.case_id == case_id and request.task == "synthetic_forward"
        )
    items = []
    for effort in config.reasoning_efforts:
        for request in requests:
            items.append(ProbeItem(reasoning_effort=effort, information="native", request=request))
            if config.design == "forward_intervention" and request.view == "structured":
                items.append(
                    ProbeItem(
                        reasoning_effort=effort,
                        information="oracle_post_impact_kinematics",
                        request=request,
                    )
                )
    if len(items) * len(MODELS) != config.planned_requests():
        raise ValueError("Panel does not provide the declared probe request shape")
    return tuple(items)


def oracle_call(
    panel: Path,
    request: PanelRequest,
    ledger: TrialLedger,
    simulators: dict[str, SnookerSimulator],
    run_hash: str,
) -> StudyCall:
    call = load_synthetic_panel_call(panel, request)
    if not isinstance(call.input, ForwardInput) or request.view != "structured":
        raise ValueError("Oracle kinematics extend only a structured blind-forward request")
    labels = json.loads((panel / "private" / f"{request.case_id}.json").read_text())
    record = ShotRecord.model_validate(labels["record"])
    action = CueAction.model_validate(labels["forward_labels"][request.condition]["action"])
    if action != call.input.action:
        raise ValueError("Oracle kinematics must belong to the requested action")
    key = fingerprint(record.physics.model_dump(mode="json"))
    if key not in simulators:
        simulators[key] = SnookerSimulator(record.physics)
    replay, _ = recorded_replay(
        record.initial,
        action,
        simulators[key],
        ledger,
        namespace=f"{run_hash}/{request.case_id}/oracle-kinematics/{request.condition}",
        role="offline_reference",
    )
    measured = replay.cue_velocity, replay.cue_omega
    recorded = record.cue_initial_velocity_mps, record.cue_initial_angular_velocity_radps
    if request.condition == "base_action" and measured != recorded:
        raise ValueError("Witness replay no longer reproduces its recorded post-impact state")
    return StudyCall(
        view="structured",
        input=OracleForwardInput(
            forward=call.input,
            cue_ball_post_impact=PostImpactKinematics(
                velocity_mps=replay.cue_velocity, angular_velocity_radps=replay.cue_omega
            ),
        ),
    )


def _score_probe_row(
    panel: Path,
    request: PanelRequest,
    reply: ModelReply,
    ledger: TrialLedger,
    simulators: dict[str, SnookerSimulator],
    run_hash: str,
) -> dict[str, object]:
    forward_score, goal_score = _score_row(panel, request, reply, ledger, simulators, run_hash)
    radius = load_synthetic_panel_call(panel, request).input.physics.table.ball_radius_m
    scored: dict[str, object] = {"forward_score": forward_score, "goal_score": goal_score}
    if forward_score is not None:
        labels = json.loads((panel / "private" / f"{request.case_id}.json").read_text())
        scored["forward_tolerances"] = forward_tolerance_profile(
            ForwardSubmission.model_validate_json(reply.content),
            ForwardSubmission.model_validate(labels["forward_labels"][request.condition]["truth"]),
            forward_score,
            radius,
        )
    if goal_score is not None:
        scored["goal_tolerances"] = endpoint_tolerance_profile(goal_score, radius)
    return scored


def _counts(profiles: list[dict[str, bool]]) -> dict[str, int]:
    return {str(k): sum(profile[str(k)] for profile in profiles) for k in TOLERANCE_RADII}


def forward_baselines(panel: Path, request: PanelRequest) -> dict[str, float | None]:
    """Resting max error (radii) of two trivial forecasters: nothing moves, straight free roll."""
    forward = load_synthetic_panel_call(panel, request).input
    if not isinstance(forward, ForwardInput):
        raise ValueError("Forward baselines need a blind-forward request")
    labels = json.loads((panel / "private" / f"{request.case_id}.json").read_text())
    item = labels["forward_labels"][request.condition]
    truth = ForwardSubmission.model_validate(item["truth"])
    radius = forward.physics.table.ball_radius_m
    static = score_forward(
        ForwardSubmission(
            frames=tuple(
                PredictedFrame(time_s=time, state=forward.initial)
                for time in forward.sample_times_s
            ),
            resting_state=forward.initial,
            events=(),
        ),
        truth,
        initial=forward.initial,
        moving_ball_ids=tuple(item["moving_ball_ids"]),
        position_tolerance_m=radius,
        event_time_tolerance_s=0.02,
    ).resting_max_position_error_m
    rest = {ball.ball_id: ball for ball in truth.resting_state.balls}
    free = [
        dist(ball.xy_m, reference)
        for ball in free_roll_resting_state(forward.initial, forward.action, forward.physics).balls
        if ball.xy_m is not None and (reference := rest[ball.ball_id].xy_m) is not None
    ]
    return {
        "static": static / radius if static is not None else None,
        "free_roll": max(free) / radius if free else None,
    }


def _resting_error(row: ProbeRow, radius: float) -> float | None:
    if row.forward_score is None or row.forward_score.resting_max_position_error_m is None:
        return None
    return row.forward_score.resting_max_position_error_m / radius


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def compare_arms(panel: Path, rows: list[ProbeRow]) -> list[dict[str, object]]:
    """Pre-registered paired B1 contrasts on the same case and action condition."""
    radius = json.loads((panel / "manifest.json").read_text())["resolved_physics"][0]["config"][
        "ball_radius_m"
    ]
    groups = {
        item["case_id"]: item["source_group"]
        for item in json.loads((panel / "private/selection.json").read_text())["synthetic"]
    }
    arms: dict[tuple[str, str, str, str], dict[tuple[str, str], ProbeRow]] = defaultdict(dict)
    for row in rows:
        if row.task == "synthetic_forward":
            arms[row.reasoning_effort, row.information, row.view, row.model][
                row.case_id, row.condition
            ] = row
    efforts = sorted({row.reasoning_effort for row in rows})
    contrasts: list[tuple[str, str, str, str, str, str]] = []
    for model in MODELS:
        for effort in efforts:
            if effort != "minimal" and "minimal" in efforts:
                contrasts.append(("reasoning_effort", model, effort, "minimal", "native", "native"))
            contrasts.append(
                (
                    "oracle_kinematics",
                    model,
                    effort,
                    effort,
                    "oracle_post_impact_kinematics",
                    "native",
                )
            )
            contrasts.append(("bev_view", model, effort, effort, "native", "native"))
    result: list[dict[str, object]] = []
    for name, model, effort_a, effort_b, info_a, info_b in contrasts:
        view_a = "bev" if name == "bev_view" else "structured"
        left = arms.get((effort_a, info_a, view_a, model), {})
        right = arms.get((effort_b, info_b, "structured", model), {})
        keys = sorted(left.keys() & right.keys())
        if not keys:
            continue
        errors = [
            (key, a, b)
            for key in keys
            if (a := _resting_error(left[key], radius)) is not None
            and (b := _resting_error(right[key], radius)) is not None
        ]

        def success(row: ProbeRow) -> float:
            return float(bool(row.forward_tolerances and row.forward_tolerances["resting"]["5.0"]))

        f1 = [
            score_a.contacts.f1 - score_b.contacts.f1
            for key in keys
            if (score_a := left[key].forward_score) is not None
            and (score_b := right[key].forward_score) is not None
        ]
        result.append(
            {
                "contrast": name,
                "model": model,
                "arm_a": [effort_a, info_a, view_a],
                "arm_b": [effort_b, info_b, "structured"],
                "planned_pairs": len(keys),
                "error_pairs": len(errors),
                "median_resting_error_difference_radii": _median([a - b for _, a, b in errors]),
                "improved_pairs": sum(a < b for _, a, b in errors),
                "resting_error_difference": summarize_paired(
                    tuple(
                        PairedMeasurement(pair_id="/".join(key), group_id=groups[key[0]], a=a, b=b)
                        for key, a, b in errors
                    ),
                    seed=20261020,
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
                    seed=20261020,
                ).model_dump(mode="json"),
                "mean_contact_f1_difference": float(np.mean(f1)) if f1 else None,
            }
        )
    return result


def summarize_probe(panel: Path, rows: list[ProbeRow]) -> dict[str, object]:
    radius = json.loads((panel / "manifest.json").read_text())["resolved_physics"][0]["config"][
        "ball_radius_m"
    ]
    index = {
        request.request_id: request
        for request in (
            PanelRequest.model_validate_json(line)
            for line in (panel / "request_index.jsonl").open()
        )
    }
    baselines = {
        request_id: forward_baselines(panel, index[request_id])
        for request_id in sorted({r.request_id for r in rows if r.task == "synthetic_forward"})
    }
    groups: dict[tuple[str, str, str, str], list[ProbeRow]] = defaultdict(list)
    for row in rows:
        groups[row.reasoning_effort, row.information, row.view, row.model].append(row)
    arms = []
    for (effort, information, view, model), group in sorted(groups.items()):
        forwards = [row for row in group if row.task == "synthetic_forward"]
        goals = [row for row in group if row.task == "synthetic_goal"]
        scores = [row.forward_score for row in forwards if row.forward_score is not None]
        errors = [
            s.moving_mean_error_m / radius for s in scores if s.moving_mean_error_m is not None
        ]
        arms.append(
            {
                "reasoning_effort": effort,
                "information": information,
                "view": view,
                "model": model,
                "attempted_requests": len(group),
                "valid_outputs": sum(row.valid_output for row in group),
                "error_codes": dict(Counter(row.error_code for row in group if row.error_code)),
                "forward_requests": len(forwards),
                "valid_forward_outputs": len(scores),
                "moving_error_radii_valid_only": {
                    "mean": float(np.mean(errors)),
                    "median": float(np.median(errors)),
                }
                if errors
                else None,
                "contact_f1_valid_only": float(np.mean([s.contacts.f1 for s in scores]))
                if scores
                else None,
                "frames_success_by_tolerance_radii": _counts(
                    [r.forward_tolerances["frames"] for r in forwards if r.forward_tolerances]
                ),
                "resting_success_by_tolerance_radii": _counts(
                    [r.forward_tolerances["resting"] for r in forwards if r.forward_tolerances]
                ),
                "baseline_median_resting_error_radii": {
                    name: _median(
                        [
                            value
                            for row in forwards
                            if (value := baselines[row.request_id][name]) is not None
                        ]
                    )
                    for name in ("static", "free_roll")
                },
                **{
                    f"beats_{name}_baseline": sum(
                        (error := _resting_error(row, radius)) is not None
                        and (reference := baselines[row.request_id][name]) is not None
                        and error < reference
                        for row in forwards
                    )
                    for name in ("static", "free_roll")
                },
                "goal_requests": len(goals),
                "goal_success_by_tolerance_radii": _counts(
                    [row.goal_tolerances for row in goals if row.goal_tolerances]
                ),
                "usage": usage_summary([row.reply for row in group if row.reply is not None]),
            }
        )
    pairs: dict[tuple[str, str, str, str, str], dict[str, ProbeRow]] = defaultdict(dict)
    for row in rows:
        if row.forward_score is not None and row.reply is not None:
            key = row.reasoning_effort, row.information, row.view, row.model, row.case_id
            pairs[key][row.condition] = row
    interventions: dict[tuple[str, str, str, str], list[tuple[InterventionScore, float | None]]]
    interventions = defaultdict(list)
    for (effort, information, view, model, case_id), pair in sorted(pairs.items()):
        if set(pair) != {"base_action", "intervened_action"}:
            continue
        labels = json.loads((panel / "private" / f"{case_id}.json").read_text())["forward_labels"]
        base, altered = (pair[name].reply for name in ("base_action", "intervened_action"))
        if base is None or altered is None:
            raise ValueError("A scored intervention requires both billed replies")
        request = next(
            request
            for request in (
                PanelRequest.model_validate_json(line)
                for line in (panel / "request_index.jsonl").open()
            )
            if request.request_id == pair["base_action"].request_id
        )
        truths = [
            ForwardSubmission.model_validate(labels[name]["truth"])
            for name in ("base_action", "intervened_action")
        ]
        options = {
            "initial": load_synthetic_panel_call(panel, request).input.initial,
            "moving_ball_ids": tuple(
                sorted(
                    set(labels["base_action"]["moving_ball_ids"])
                    | set(labels["intervened_action"]["moving_ball_ids"])
                )
            ),
            "position_tolerance_m": radius,
            "event_time_tolerance_s": 0.02,
        }
        # Predicting no change at all: the response baseline any action-sensitive model must beat.
        unchanged = score_intervention_pair(truths[0], truths[0], *truths, **options)
        interventions[effort, information, view, model].append(
            (
                score_intervention_pair(
                    ForwardSubmission.model_validate_json(base.content),
                    ForwardSubmission.model_validate_json(altered.content),
                    *truths,
                    **options,
                ),
                unchanged.moving_response_mean_error_m / radius
                if unchanged.moving_response_mean_error_m is not None
                else None,
            )
        )
    summaries = []
    for (effort, information, view, model), scored_pairs in sorted(interventions.items()):
        pair_scores = [score for score, _ in scored_pairs]
        responses = [
            score.moving_response_mean_error_m / radius
            for score in pair_scores
            if score.moving_response_mean_error_m is not None
        ]
        unchanged_values = [value for _, value in scored_pairs if value is not None]
        summaries.append(
            {
                "reasoning_effort": effort,
                "information": information,
                "view": view,
                "model": model,
                "complete_pairs": len(pair_scores),
                "joint_positions_successes": sum(s.joint_positions_success for s in pair_scores),
                "joint_events_successes": sum(s.joint_events_success for s in pair_scores),
                "status_patterns_correct": sum(s.status_patterns_correct for s in pair_scores),
                "median_response_error_radii": _median(responses),
                "no_change_baseline_median_response_error_radii": _median(unchanged_values),
            }
        )
    return {
        "by_arm": arms,
        "interventions": summaries,
        "paired_contrasts": compare_arms(panel, rows),
    }


def run_native_probe(
    panel: Path,
    source: Path,
    output: Path,
    config: NativeProbeConfig,
    *,
    endpoint_index: int = 4,
    execute: bool = False,
    token_provider: Callable[[], str] | None = None,
    transport: httpx.BaseTransport | None = None,
    limit_jobs: int | None = None,
) -> dict[str, object]:
    validation = validate_diagnostic_panel(panel)
    plan = probe_plan(panel, config)
    backends = {
        effort: study_azure_backends(
            source,
            endpoint_index,
            reasoning_effort=effort,
            max_output_tokens=config.max_output_tokens,
            request_token_reservation=config.request_token_reservation,
            timeout_s=config.timeout_s,
        )
        for effort in config.reasoning_efforts
    }
    limits = BudgetLimits(
        max_model_requests=config.max_model_requests,
        max_total_tokens=config.max_total_tokens,
        max_physics_evaluations=config.layouts * (2 + len(config.reasoning_efforts) * len(MODELS)),
    )
    protocol: dict[str, object] = {
        "panel_manifest_sha256": validation["manifest_sha256"],
        "azure_inventory_sha256": file_sha256(source),
        "endpoint_hash": fingerprint(backends[config.reasoning_efforts[0]][0].base_url),
        "config": config.model_dump(mode="json"),
        "backends": {
            effort: [backend.model_dump(mode="json") for backend in items]
            for effort, items in backends.items()
        },
        "jobs": [
            [item.reasoning_effort, item.information, item.request.request_id] for item in plan
        ],
        "limits": limits.model_dump(mode="json"),
        "producer": producer_hashes(
            "native_probe.py",
            "study_calls.py",
            "azure_study.py",
            "model_adapters.py",
            "evaluation.py",
            "diagnostics.py",
        ),
        "real_data_sent": False,
        "agent_tool_evaluations": 0,
        "formal_evaluation": False,
    }
    run_hash = lock_run(output, protocol, reserve_bytes=200_000_000)
    planned = len(plan) * len(MODELS)
    if not execute:
        return {"status": "planned_no_calls", "run_hash": run_hash, "planned_requests": planned}
    if token_provider is None:
        raise ValueError("Azure execution requires a runtime bearer-token provider")
    token_provider()
    rows: list[ProbeRow] = []
    simulators: dict[str, SnookerSimulator] = {}
    ledger = TrialLedger(output / "ledger.sqlite", limits)
    before = ledger.counts()["model_requests"]
    stop_error = None
    completed = 0
    try:
        with ExitStack() as stack:
            adapters = {
                (effort, backend.alias): stack.enter_context(
                    StudyAdapter(
                        backend.model_copy(update={"execution_authorized": True}),
                        token_provider=token_provider,
                        transport=transport,
                    )
                )
                for effort, items in backends.items()
                for backend in items
            }
            for item in plan:
                for model in MODELS:
                    request = item.request
                    path = (
                        output
                        / "jobs"
                        / item.reasoning_effort
                        / item.information
                        / model
                        / f"{request.request_id}.json"
                    )
                    cached = load_job(path, run_hash)
                    if cached is not None:
                        row = ProbeRow.model_validate(cached)
                        identity = row.model, row.reasoning_effort, row.information, row.request_id
                        if identity != (
                            model,
                            item.reasoning_effort,
                            item.information,
                            request.request_id,
                        ):
                            raise ValueError("Cached probe job identity changed")
                        rows.append(row)
                        continue
                    if stop_error or (limit_jobs is not None and completed >= limit_jobs):
                        continue
                    if not credential_ready(token_provider):
                        stop_error = "credential_unavailable"
                        continue
                    started = monotonic()
                    adapter = adapters[item.reasoning_effort, model]
                    if item.information == "native":
                        call = load_synthetic_panel_call(panel, request)
                        outcome = guarded_request(partial(request_model, adapter, call, ledger))
                        view = call.view
                    else:
                        study_call = oracle_call(panel, request, ledger, simulators, run_hash)
                        outcome = guarded_request(
                            partial(request_study_model, adapter, study_call, ledger)
                        )
                        view = study_call.view
                    stop_error = outcome.stop_error
                    error_code = outcome.error_code
                    scored: dict[str, object] = {}
                    if outcome.reply is not None and error_code is None:
                        try:
                            scored = _score_probe_row(
                                panel, request, outcome.reply, ledger, simulators, run_hash
                            )
                        except SimulationFailure:
                            error_code = "hidden_judge_simulation_failure"
                            LOGGER.error("Hidden judge failed: %s/%s", model, request.request_id)
                    row = ProbeRow.model_validate(
                        {
                            "model": model,
                            "reasoning_effort": item.reasoning_effort,
                            "information": item.information,
                            "request_id": request.request_id,
                            "case_id": request.case_id,
                            "task": request.task,
                            "condition": request.condition,
                            "view": view,
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
                    LOGGER.info(
                        "Probe job %d/%d: %s/%s/%s",
                        len(rows),
                        planned,
                        item.reasoning_effort,
                        model,
                        request.task,
                    )
        report: dict[str, object] = {
            "status": "stopped_service_error"
            if stop_error
            else ("completed" if len(rows) == planned else "partial"),
            "run_hash": run_hash,
            "design": config.design,
            "stop_error": stop_error,
            "independent_layouts": config.layouts,
            "planned_requests": planned,
            "completed_jobs": len(rows),
            "new_model_requests": ledger.counts()["model_requests"] - before,
            "resource_counts": ledger.counts(),
            "physics_usage_by_role": ledger.physics_usage_by_role(),
            "usage": ledger.usage_totals(),
            **summarize_probe(panel, rows),
            "real_data_sent": False,
            "formal_evaluation": False,
        }
        write_json(output / "summary.json", report)
        return report
    finally:
        ledger.close()
