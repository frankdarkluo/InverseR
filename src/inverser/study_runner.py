import fcntl
import json
from importlib.metadata import version
from pathlib import Path
from typing import Literal, Self

import httpx
from pydantic import Field, model_validator

from inverser.baselines import SearchConfig, SearchResult, make_action_bank, search_goal
from inverser.diagnostics import (
    BankSelectionScore,
    CandidateOutcome,
    ForwardScore,
    PairedMeasurement,
    PairedSummary,
    score_bank_selection,
    score_forward,
    summarize_paired,
)
from inverser.evaluation import (
    BudgetLimits,
    TrialLedger,
    recorded_replay,
    request_model,
)
from inverser.metrics import EndpointScore, endpoint_loss, score_endpoint
from inverser.model_adapters import BackendConfig, ModelAdapter, ModelCall, rendered_asset
from inverser.physics import SimulationFailure, SnookerSimulator
from inverser.pilot import source_fingerprint
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.schemas import Layout, Record, ShotRecord
from inverser.study import (
    ForwardSubmission,
    GoalInput,
    make_study_case,
    replay_forward_truth,
    translating_ball_ids,
)


class EngineeringRunConfig(Record):
    schema_version: Literal["baseline-engineering-v1"] = "baseline-engineering-v1"
    seed: int = Field(default=20261013, ge=0)
    objective_budget: int = Field(default=8, ge=1, le=16)
    candidate_count: int = Field(default=4, ge=2, le=8)
    sample_times_s: tuple[float, float] = (0.25, 0.75)

    @model_validator(mode="after")
    def validate_times(self) -> Self:
        if not 0 <= self.sample_times_s[0] < self.sample_times_s[1]:
            raise ValueError("Engineering sampling times must be nonnegative and increasing")
        return self


class JobEnvelope(Record):
    run_hash: str
    payload: dict[str, object]
    payload_sha256: str

    @model_validator(mode="after")
    def validate_checksum(self) -> Self:
        if fingerprint(self.payload) != self.payload_sha256:
            raise ValueError("Completed job payload checksum mismatch")
        return self


class SearchJob(Record):
    run_hash: str
    search: SearchResult
    hidden_judge: EndpointScore | None
    judge_failure: str | None


class ForecastRow(Record):
    candidate_id: str
    view: Literal["structured", "bev"]
    score: ForwardScore
    predicted_loss_radii: float


class BankJob(Record):
    run_hash: str
    fixture: Literal["oracle_forecast_for_engineering_only"] = (
        "oracle_forecast_for_engineering_only"
    )
    references: tuple[CandidateOutcome, ...]
    forecasts: tuple[ForecastRow, ...]
    selection_scores: dict[str, BankSelectionScore]
    paired_moving_error: PairedSummary | None


def _write_job(path: Path, job: SearchJob | BankJob) -> None:
    if path.exists():
        raise FileExistsError("Completed job artifacts are immutable")
    payload = job.model_dump(mode="json")
    envelope = JobEnvelope(
        run_hash=job.run_hash, payload=payload, payload_sha256=fingerprint(payload)
    )
    write_json(path, envelope.model_dump(mode="json"))


def _loss(layout: Layout, goal: GoalInput) -> float:
    return endpoint_loss(
        layout,
        goal.goal,
        ball_radius_m=goal.physics.table.ball_radius_m,
        table_diagonal_m=(goal.physics.table.width_m**2 + goal.physics.table.length_m**2) ** 0.5,
        no_scratch=goal.no_scratch,
    )


def _bank_job(
    record: ShotRecord,
    goal: GoalInput,
    simulator: SnookerSimulator,
    ledger: TrialLedger,
    output: Path,
    config: EngineeringRunConfig,
    run_hash: str,
) -> BankJob:
    bank = make_action_bank(goal.case_id, count=config.candidate_count, seed=config.seed)
    bank_path = output / "public_candidate_bank.json"
    if bank_path.exists():
        if json.loads(bank_path.read_text()) != bank.model_dump(mode="json"):
            raise ValueError("Frozen public candidate bank has changed")
    else:
        write_json(bank_path, bank.model_dump(mode="json"))
    references = []
    truths: dict[str, ForwardSubmission] = {}
    translating: dict[str, tuple[str, ...]] = {}
    response_by_action: dict[str, str] = {}
    for candidate in bank.candidates:
        try:
            replay, _ = recorded_replay(
                goal.initial,
                candidate.action,
                simulator,
                ledger,
                namespace=run_hash + "/offline-bank-reference",
                role="offline_reference",
            )
        except SimulationFailure:
            references.append(
                CandidateOutcome(
                    candidate_id=candidate.candidate_id,
                    goal_success=False,
                    failure="reference_simulation_failure",
                )
            )
            continue
        truth = replay_forward_truth(goal.initial, replay, record.physics, config.sample_times_s)
        truths[candidate.candidate_id] = truth
        translating[candidate.candidate_id] = translating_ball_ids(goal.initial, replay.motion)
        response_by_action[fingerprint(candidate.action.model_dump(mode="json"))] = (
            truth.model_dump_json()
        )
        references.append(
            CandidateOutcome(
                candidate_id=candidate.candidate_id,
                loss_radii=_loss(replay.goal, goal),
                goal_success=score_endpoint(
                    replay.goal,
                    goal.goal,
                    goal.tolerance_radii * goal.physics.table.ball_radius_m,
                    no_scratch=goal.no_scratch,
                ).success,
            )
        )

    def fixture(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        task = json.loads(body["messages"][1]["content"][0]["text"])["task"]
        return httpx.Response(
            200,
            json={
                "model": "engineering-forecast-fixture",
                "choices": [
                    {"message": {"content": response_by_action[fingerprint(task["action"])]}}
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    forecasts = []
    losses: dict[str, dict[str, float | None]] = {
        view: {candidate.candidate_id: None for candidate in bank.candidates}
        for view in ("structured", "bev")
    }
    case = make_study_case(record, simulator)
    with ModelAdapter(
        BackendConfig(
            kind="mock",
            alias="engineering-oracle-fixture",
            model_id="engineering-forecast-fixture",
            model_revision="fixture-v1",
            base_url="https://fixture.invalid/v1",
        ),
        transport=httpx.MockTransport(fixture),
    ) as adapter:
        for candidate in bank.candidates:
            if candidate.candidate_id not in truths:
                continue
            for view in ("structured", "bev"):
                forward = case.forward_input(candidate.action, config.sample_times_s)
                images = (
                    (rendered_asset(forward, "initial", forward.physics.table),)
                    if view == "bev"
                    else ()
                )
                call = ModelCall(view=view, input=forward, images=images, seed=config.seed)
                reply = request_model(adapter, call, ledger)
                predicted = ForwardSubmission.model_validate_json(reply.content)
                score = score_forward(
                    predicted,
                    truths[candidate.candidate_id],
                    initial=forward.initial,
                    moving_ball_ids=translating[candidate.candidate_id],
                    position_tolerance_m=goal.physics.table.ball_radius_m,
                    event_time_tolerance_s=0.02,
                )
                loss = _loss(predicted.resting_state, goal)
                losses[view][candidate.candidate_id] = loss
                forecasts.append(
                    ForecastRow(
                        candidate_id=candidate.candidate_id,
                        view=view,
                        score=score,
                        predicted_loss_radii=loss,
                    )
                )
                write_json(
                    output / f"{candidate.candidate_id}_{view}_public_input.json",
                    json.loads(call.prompt()),
                )
    selections = {}
    for view, predicted_losses in losses.items():
        valid = [(key, loss) for key, loss in predicted_losses.items() if loss is not None]
        selected = min(valid, key=lambda item: item[1])[0] if valid else None
        selections[view] = score_bank_selection(
            tuple(references), predicted_losses, selected_candidate_id=selected
        )
    indexed = {(row.candidate_id, row.view): row for row in forecasts}
    pairs = []
    for candidate in bank.candidates:
        a, b = (
            indexed.get((candidate.candidate_id, "structured")),
            indexed.get((candidate.candidate_id, "bev")),
        )
        if a is not None and b is not None:
            left, right = a.score.moving_mean_error_m, b.score.moving_mean_error_m
            if left is not None and right is not None:
                pairs.append(
                    PairedMeasurement(
                        pair_id=candidate.candidate_id, group_id=record.group_id, a=left, b=right
                    )
                )
    return BankJob(
        run_hash=run_hash,
        references=tuple(references),
        forecasts=tuple(forecasts),
        selection_scores=selections,
        paired_moving_error=summarize_paired(tuple(pairs), seed=config.seed) if pairs else None,
    )


def baseline_dry_run(
    source: Path,
    output: Path,
    config: EngineeringRunConfig,
    *,
    limit_jobs: int | None = None,
) -> dict[str, object]:
    if limit_jobs is not None and not 1 <= limit_jobs <= 4:
        raise ValueError("An engineering invocation permits a limit of 1-4 completed jobs")
    with source.open() as stream:
        line = stream.readline()
    if not line:
        raise ValueError("Engineering diagnostics require an existing development record")
    record = ShotRecord.model_validate_json(line)
    simulator = SnookerSimulator(record.physics)
    manifest = {
        "config": config.model_dump(mode="json"),
        "record_hash": fingerprint(record.model_dump(mode="json")),
        "source_sha256": file_sha256(source),
        "source_code_hash": source_fingerprint(),
        "resolved_physics": simulator.metadata(),
        "runtime_versions": {
            name: version(name) for name in ("numpy", "scipy", "pydantic", "pooltool-billiards")
        },
        "existing_layouts": 1,
        "actual_model_inference": False,
        "formal_evaluation": False,
    }
    run_hash = fingerprint(manifest)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError(
                    "Changed source/config/code requires a new versioned run directory"
                )
        else:
            if any(path.name != ".writer.lock" for path in output.iterdir()):
                raise ValueError("A new engineering run requires an empty output directory")
            write_json(manifest_path, manifest)
        limits = BudgetLimits(
            max_model_requests=2 * config.candidate_count,
            max_physics_evaluations=3 * config.objective_budget + 3 + config.candidate_count,
        )
        goal = make_study_case(record, simulator).goal_input()
        ledger = TrialLedger(output / "ledger.sqlite", limits)
        completed = []
        new_jobs = 0
        try:
            for method in ("random", "sobol", "geometry_powell", "fixed_bank"):
                path = output / f"{method}.json"
                if path.exists():
                    envelope = JobEnvelope.model_validate_json(path.read_text())
                    if envelope.run_hash != run_hash:
                        raise ValueError("Completed artifact belongs to another frozen run")
                    job = (
                        BankJob.model_validate(envelope.payload)
                        if method == "fixed_bank"
                        else SearchJob.model_validate(envelope.payload)
                    )
                    if job.run_hash != run_hash:
                        raise ValueError("Completed artifact belongs to another frozen run")
                elif limit_jobs is not None and new_jobs >= limit_jobs:
                    break
                elif method == "fixed_bank":
                    job = _bank_job(record, goal, simulator, ledger, output, config, run_hash)
                    _write_job(path, job)
                    new_jobs += 1
                else:
                    search_config = SearchConfig.model_validate(
                        {
                            "method": method,
                            "seed": config.seed,
                            "max_objective_calls": config.objective_budget,
                        }
                    )
                    result = search_goal(
                        goal,
                        search_config,
                        simulator,
                        ledger,
                        namespace=run_hash + "/search/" + method,
                    )
                    judged = None
                    judge_failure = None
                    if result.selected_action is not None:
                        try:
                            replay, _ = recorded_replay(
                                goal.initial,
                                result.selected_action,
                                simulator,
                                ledger,
                                namespace=run_hash + "/hidden-judge/" + method,
                                role="hidden_judge",
                            )
                        except SimulationFailure:
                            judge_failure = "reference_simulation_failure"
                        else:
                            judged = score_endpoint(
                                replay.goal,
                                goal.goal,
                                goal.tolerance_radii * goal.physics.table.ball_radius_m,
                                no_scratch=goal.no_scratch,
                            )
                    job = SearchJob(
                        run_hash=run_hash,
                        search=result,
                        hidden_judge=judged,
                        judge_failure=judge_failure,
                    )
                    _write_job(path, job)
                    new_jobs += 1
                completed.append(method)
            report: dict[str, object] = {
                "status": "complete_engineering_dry_run" if len(completed) == 4 else "partial",
                "run_hash": run_hash,
                "completed_jobs": completed,
                "new_jobs_this_invocation": new_jobs,
                "existing_development_layouts": 1,
                "actual_model_inference": False,
                "paid_api_requests": 0,
                "new_benchmark_tasks_generated": 0,
                "formal_evaluation": False,
                "fixture_prediction_source": "private_reference_for_engineering_only",
                "model_performance_claim": False,
                "counts": ledger.counts(),
                "physics_usage_by_role": ledger.physics_usage_by_role(),
                "usage": ledger.usage_totals(),
                "artifacts": {
                    method: file_sha256(output / f"{method}.json") for method in completed
                },
            }
            write_json(output / "validation.json", report)
            return report
        finally:
            ledger.close()
