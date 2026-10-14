import json
import logging
from collections import defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from time import monotonic
from typing import Literal, Self

import httpx
import numpy as np
from pydantic import Field, model_validator

from inverser.diagnostic_panel import (
    PanelRequest,
    load_synthetic_panel_call,
    validate_diagnostic_panel,
)
from inverser.diagnostics import ForwardScore, score_forward, score_intervention_pair
from inverser.evaluation import BudgetLimits, TrialLedger, recorded_replay, request_model
from inverser.metrics import EndpointScore, score_endpoint
from inverser.model_adapters import BackendConfig, InvalidModelReply, ModelAdapter, ModelReply
from inverser.physics import SimulationFailure, SnookerSimulator
from inverser.runtime import PROJECT_ROOT, check_storage, file_sha256, fingerprint, write_json
from inverser.schemas import Record, ShotRecord
from inverser.study import ForwardSubmission, GoalInput, GoalSubmission
from inverser.study_runner import JobEnvelope

LOGGER = logging.getLogger(__name__)
MODELS = ("gpt-5-mini", "gpt-5")


class AzurePilotConfig(Record):
    schema_version: Literal["azure-native-pilot-v1"] = "azure-native-pilot-v1"
    layouts: int = Field(default=2, ge=1, le=8)
    max_model_requests: int = Field(default=24, ge=1, le=100)
    max_total_tokens: int = Field(default=300000, ge=1, le=1000000)
    max_output_tokens: int = Field(default=4096, ge=256, le=8192)
    request_token_reservation: int = Field(default=16384, ge=8192)
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "minimal"
    timeout_s: float = Field(default=180, gt=0, le=300)

    @model_validator(mode="after")
    def validate_allocation(self) -> Self:
        if self.max_model_requests < self.layouts * 12:
            raise ValueError("Request allocation must cover both models and all six paired tasks")
        if self.request_token_reservation < self.max_output_tokens:
            raise ValueError("Each token reservation must cover its completion cap")
        return self


class PilotRow(Record):
    model: str
    request_id: str
    case_id: str
    task: str
    condition: str
    view: Literal["structured", "bev"]
    valid_output: bool
    elapsed_s: float = Field(ge=0)
    reply: ModelReply | None = None
    forward_score: ForwardScore | None = None
    goal_score: EndpointScore | None = None
    error_code: str | None = None


def load_azure_backends(
    source: Path, endpoint_index: int, config: AzurePilotConfig
) -> tuple[BackendConfig, ...]:
    data = json.loads(source.read_text())
    endpoints = data["azure_openai_endpoints"]
    if not 0 <= endpoint_index < len(endpoints):
        raise ValueError("Azure endpoint index is outside the supplied inventory")
    endpoint = endpoints[endpoint_index]
    backends = []
    for model in MODELS:
        candidates = [
            item
            for item in endpoint["deployments"]
            if item["model"] == model
            and item["deployment_name"] == model
            and item["sku"] == "GlobalStandard"
        ]
        if len(candidates) != 1:
            raise ValueError("Selected endpoint must declare one exact GlobalStandard deployment")
        deployment = candidates[0]
        backends.append(
            BackendConfig(
                kind="api",
                alias=model,
                model_id=f"{model}-{deployment['version']}",
                model_revision=deployment["version"],
                base_url=endpoint["endpoint"],
                provider="azure_openai",
                deployment_name=deployment["deployment_name"],
                api_version=endpoint["api_version"],
                reasoning_effort=config.reasoning_effort,
                max_output_tokens=config.max_output_tokens,
                request_token_reservation=config.request_token_reservation,
                timeout_s=config.timeout_s,
            )
        )
    return tuple(backends)


def select_pilot_requests(panel: Path, layouts: int) -> tuple[PanelRequest, ...]:
    requests = [
        PanelRequest.model_validate_json(line) for line in (panel / "request_index.jsonl").open()
    ]
    selection = json.loads((panel / "private/selection.json").read_text())["synthetic"]
    eligible = {r.case_id for r in requests if r.api_common_subset and not r.local_only}
    by_family: dict[str, list[str]] = defaultdict(list)
    for item in selection:
        if item["case_id"] in eligible:
            by_family[item["family"]].append(item["case_id"])
    case_ids = [
        cases[index]
        for index in range(2)
        for family in ("free_roll", "center", "vertical", "side")
        for cases in (by_family[family],)
        if len(cases) > index
    ][:layouts]
    if len(case_ids) != layouts:
        raise ValueError("Too few API-common layouts for the requested pilot")
    selected = tuple(
        request for case_id in case_ids for request in requests if request.case_id == case_id
    )
    if len(selected) != layouts * 6 or any(
        r.local_only or not r.model_adapter_ready or not r.api_common_subset for r in selected
    ):
        raise ValueError("Pilot must contain complete synthetic-only paired request blocks")
    return selected


def _score_row(
    panel: Path,
    request: PanelRequest,
    reply: ModelReply,
    ledger: TrialLedger,
    simulators: dict[str, SnookerSimulator],
    run_hash: str,
) -> tuple[ForwardScore | None, EndpointScore | None]:
    call = load_synthetic_panel_call(panel, request)
    labels = json.loads((panel / "private" / f"{request.case_id}.json").read_text())
    if request.task == "synthetic_forward":
        truth = labels["forward_labels"][request.condition]
        return (
            score_forward(
                ForwardSubmission.model_validate_json(reply.content),
                ForwardSubmission.model_validate(truth["truth"]),
                initial=call.input.initial,
                moving_ball_ids=tuple(truth["moving_ball_ids"]),
                position_tolerance_m=call.input.physics.table.ball_radius_m,
                event_time_tolerance_s=0.02,
            ),
            None,
        )
    if not isinstance(call.input, GoalInput):
        raise ValueError("Goal request must reconstruct a public goal contract")
    record = ShotRecord.model_validate(labels["record"])
    key = fingerprint(record.physics.model_dump(mode="json"))
    if key not in simulators:
        simulators[key] = SnookerSimulator(record.physics)
    submission = GoalSubmission.model_validate_json(reply.content)
    replay, _ = recorded_replay(
        call.input.initial,
        submission.action,
        simulators[key],
        ledger,
        namespace=f"{run_hash}/{request.case_id}/native-hidden-judge",
        role="hidden_judge",
    )
    return (
        None,
        score_endpoint(
            replay.goal,
            call.input.goal,
            call.input.tolerance_radii * call.input.physics.table.ball_radius_m,
            no_scratch=call.input.no_scratch,
        ),
    )


def summarize_rows(panel: Path, rows: list[PilotRow], layouts: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    radius = json.loads((panel / "manifest.json").read_text())["resolved_physics"][0]["config"][
        "ball_radius_m"
    ]
    for model in MODELS:
        for view in ("structured", "bev"):
            group = [row for row in rows if row.model == model and row.view == view]
            forwards = [row.forward_score for row in group if row.forward_score is not None]
            goals = [row.goal_score for row in group if row.goal_score is not None]
            errors = [
                score.moving_mean_error_m / radius
                for score in forwards
                if score.moving_mean_error_m is not None
            ]
            result.append(
                {
                    "model": model,
                    "view": view,
                    "expected_requests": layouts * 3,
                    "attempted_requests": len(group),
                    "valid_outputs": sum(row.valid_output for row in group),
                    "valid_forward_outputs": len(forwards),
                    "moving_error_radii_valid_only": float(np.mean(errors)) if errors else None,
                    "forward_position_successes": sum(
                        score.positions_success for score in forwards
                    ),
                    "expected_forward_requests": layouts * 2,
                    "contact_f1_valid_only": float(np.mean([s.contacts.f1 for s in forwards]))
                    if forwards
                    else None,
                    "goal_successes": sum(score.success for score in goals),
                    "expected_goal_requests": layouts,
                    "failed_outputs": sum(row.error_code is not None for row in group),
                }
            )
    return result


def summarize_interventions(panel: Path, rows: list[PilotRow]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    groups: dict[tuple[str, str, str], dict[str, PilotRow]] = defaultdict(dict)
    for row in rows:
        if row.forward_score is not None:
            groups[row.model, row.view, row.case_id][row.condition] = row
    for (model, view, case_id), group in groups.items():
        if set(group) != {"base_action", "intervened_action"}:
            continue
        base, altered = group["base_action"], group["intervened_action"]
        if base.reply is None or altered.reply is None:
            raise ValueError("A scored intervention requires both billed replies")
        labels = json.loads((panel / "private" / f"{case_id}.json").read_text())["forward_labels"]
        request = next(
            PanelRequest.model_validate_json(line)
            for line in (panel / "request_index.jsonl").open()
            if json.loads(line)["request_id"] == base.request_id
        )
        call = load_synthetic_panel_call(panel, request)
        moving = tuple(
            sorted(
                set(labels["base_action"]["moving_ball_ids"])
                | set(labels["intervened_action"]["moving_ball_ids"])
            )
        )
        score = score_intervention_pair(
            ForwardSubmission.model_validate_json(base.reply.content),
            ForwardSubmission.model_validate_json(altered.reply.content),
            ForwardSubmission.model_validate(labels["base_action"]["truth"]),
            ForwardSubmission.model_validate(labels["intervened_action"]["truth"]),
            initial=call.input.initial,
            moving_ball_ids=moving,
            position_tolerance_m=call.input.physics.table.ball_radius_m,
            event_time_tolerance_s=0.02,
        )
        result.append(
            {
                "model": model,
                "view": view,
                "case_id": case_id,
                "response_error_radii": score.moving_response_mean_error_m
                / call.input.physics.table.ball_radius_m
                if score.moving_response_mean_error_m is not None
                else None,
                "joint_positions_success": score.joint_positions_success,
                "joint_events_success": score.joint_events_success,
            }
        )
    return result


def run_azure_panel(
    panel: Path,
    source: Path,
    output: Path,
    config: AzurePilotConfig,
    *,
    endpoint_index: int = 4,
    execute: bool = False,
    token_provider: Callable[[], str] | None = None,
    transport: httpx.BaseTransport | None = None,
    limit_jobs: int | None = None,
) -> dict[str, object]:
    validation = validate_diagnostic_panel(panel)
    backends = load_azure_backends(source, endpoint_index, config)
    requests = select_pilot_requests(panel, config.layouts)
    limits = BudgetLimits(
        max_model_requests=config.max_model_requests,
        max_total_tokens=config.max_total_tokens,
        max_physics_evaluations=config.layouts * 4,
    )
    protocol = {
        "panel_manifest_sha256": validation["manifest_sha256"],
        "azure_inventory_sha256": file_sha256(source),
        "endpoint_hash": fingerprint(backends[0].base_url),
        "config": config.model_dump(mode="json"),
        "backends": [backend.model_dump(mode="json") for backend in backends],
        "request_ids": [request.request_id for request in requests],
        "limits": limits.model_dump(mode="json"),
        "producer": {
            name: file_sha256(Path(__file__).parent / name)
            for name in ("azure_study.py", "model_adapters.py", "evaluation.py", "diagnostics.py")
        },
        "real_data_sent": False,
        "agent_tool_evaluations": 0,
        "formal_evaluation": False,
    }
    run_hash = fingerprint(protocol)
    if not output.resolve().is_relative_to((PROJECT_ROOT / "data").resolve()):
        raise ValueError("API run artifacts must remain inside the project data directory")
    manifest = {**protocol, "run_hash": run_hash}
    if output.exists():
        if json.loads((output / "manifest.json").read_text()) != manifest:
            raise ValueError("Changed API protocol requires a new output directory")
    else:
        check_storage(50_000_000)
        output.mkdir(parents=True)
        write_json(output / "manifest.json", manifest)
    if not execute:
        return {
            "status": "planned_no_calls",
            "run_hash": run_hash,
            "planned_requests": len(requests) * 2,
        }
    if token_provider is None:
        raise ValueError("Azure execution requires a runtime bearer-token provider")
    token_provider()
    rows = []
    simulators: dict[str, SnookerSimulator] = {}
    ledger = TrialLedger(output / "ledger.sqlite", limits)
    before = ledger.counts()["model_requests"]
    stop_error = None
    completed = 0
    try:
        with ExitStack() as stack:
            adapters = {
                backend.alias: stack.enter_context(
                    ModelAdapter(
                        backend.model_copy(update={"execution_authorized": True}),
                        token_provider=token_provider,
                        transport=transport,
                    )
                )
                for backend in backends
            }
            for request in requests:
                for model in MODELS:
                    path = output / "jobs" / model / f"{request.request_id}.json"
                    if path.exists():
                        envelope = JobEnvelope.model_validate_json(path.read_text())
                        row = PilotRow.model_validate(envelope.payload)
                        if envelope.run_hash != run_hash or (row.model, row.request_id) != (
                            model,
                            request.request_id,
                        ):
                            raise ValueError("Cached API job identity changed")
                        rows.append(row)
                        continue
                    if stop_error or (limit_jobs is not None and completed >= limit_jobs):
                        continue
                    started = monotonic()
                    reply = None
                    forward_score, goal_score = None, None
                    error_code = None
                    valid_output = False
                    call = load_synthetic_panel_call(panel, request)
                    try:
                        reply = request_model(adapters[model], call, ledger)
                        valid_output = True
                        forward_score, goal_score = _score_row(
                            panel, request, reply, ledger, simulators, run_hash
                        )
                    except InvalidModelReply as error:
                        reply = error.reply
                        error_code = "invalid_submission"
                        LOGGER.warning("Invalid native output: %s/%s", model, request.request_id)
                    except SimulationFailure:
                        error_code = "hidden_judge_simulation_failure"
                        LOGGER.error("Hidden judge failed: %s/%s", model, request.request_id)
                    except httpx.HTTPStatusError as error:
                        error_code = f"http_{error.response.status_code}"
                        stop_error = error_code
                        LOGGER.error("API service error %s; stopping without retries", error_code)
                    except httpx.RequestError:
                        error_code = "transport_response_unknown"
                        stop_error = error_code
                        LOGGER.error("API transport failed; stopping without rebilling")
                    row = PilotRow(
                        model=model,
                        request_id=request.request_id,
                        case_id=request.case_id,
                        task=request.task,
                        condition=request.condition,
                        view=call.view,
                        valid_output=valid_output,
                        elapsed_s=monotonic() - started,
                        reply=reply,
                        forward_score=forward_score,
                        goal_score=goal_score,
                        error_code=error_code,
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = row.model_dump(mode="json")
                    write_json(
                        path,
                        JobEnvelope(
                            run_hash=run_hash,
                            payload=payload,
                            payload_sha256=fingerprint(payload),
                        ).model_dump(mode="json"),
                    )
                    rows.append(row)
                    completed += 1
                    LOGGER.info(
                        "API job %d/%d: %s/%s", len(rows), len(requests) * 2, model, request.task
                    )
        report: dict[str, object] = {
            "status": "stopped_service_error"
            if stop_error
            else ("completed" if len(rows) == len(requests) * 2 else "partial"),
            "run_hash": run_hash,
            "stop_error": stop_error,
            "independent_layouts": config.layouts,
            "planned_requests": len(requests) * 2,
            "completed_jobs": len(rows),
            "new_model_requests": ledger.counts()["model_requests"] - before,
            "resource_counts": ledger.counts(),
            "usage": ledger.usage_totals(),
            "by_model_view": summarize_rows(panel, rows, config.layouts),
            "interventions": summarize_interventions(panel, rows),
            "real_data_sent": False,
            "formal_evaluation": False,
        }
        write_json(output / "summary.json", report)
        return report
    finally:
        ledger.close()
