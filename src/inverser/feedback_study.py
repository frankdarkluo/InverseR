import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Literal, Self

import httpx
import numpy as np
from pydantic import Field, model_validator
from scipy.stats import binomtest

from inverser.azure_study import MODELS
from inverser.baselines import SearchConfig, unit_action
from inverser.diagnostic_panel import FAMILIES, load_synthetic_panel_call, validate_diagnostic_panel
from inverser.diagnostics import PairedMeasurement, summarize_paired
from inverser.evaluation import BudgetLimits, TrialLedger, recorded_replay
from inverser.metrics import score_endpoint
from inverser.model_adapters import ModelReply
from inverser.physics import SimulationFailure, SnookerSimulator
from inverser.planning_pilot import PlanningPilotConfig, select_planning_cases
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.schemas import CueAction, PhysicsConfig, Record
from inverser.study import (
    GoalInput,
    GoalSubmission,
    SubmittedAction,
    public_physics,
    semantic_contacts,
)
from inverser.study_calls import (
    TOLERANCE_RADII,
    AttemptFeedback,
    CredentialUnavailable,
    FeedbackGoalInput,
    ReasoningEffort,
    SimulatedOutcome,
    StudyAdapter,
    StudyCall,
    azure_model_backends,
    credential_ready,
    endpoint_tolerance_profile,
    guarded_request,
    load_job,
    lock_run,
    producer_hashes,
    request_study_model,
    save_job,
    usage_summary,
)

LOGGER = logging.getLogger(__name__)
Condition = Literal["genuine", "shuffled", "reflection"]
CONDITION_MEANING = {
    "genuine": "simulated outcome of the submitted action",
    "shuffled": "simulated outcome of an independent seeded random action, shown as the attempt's",
    "reflection": "no execution; earlier actions only",
}


class FeedbackStudyConfig(Record):
    schema_version: Literal["feedback-study-v1"] = "feedback-study-v1"
    models: tuple[str, ...] = Field(default=MODELS, min_length=1, max_length=8)
    layouts: int = Field(default=32, ge=4, le=200)
    case_seed: int = Field(default=20261017, ge=0)
    max_attempts: int = Field(default=8, ge=2, le=16)
    conditions: tuple[Condition, ...] = ("genuine", "shuffled", "reflection")
    shuffle_seed: int = Field(default=20261019, ge=0)
    reasoning_effort: ReasoningEffort = "minimal"
    max_model_requests: int = Field(default=1408, ge=1, le=4000)
    max_total_tokens: int = Field(default=20_000_000, ge=1, le=40_000_000)
    max_output_tokens: int = Field(default=4096, ge=256, le=32768)
    request_token_reservation: int = Field(default=16384, ge=8192)
    timeout_s: float = Field(default=900, gt=0, le=900)
    min_request_interval_s: float = Field(default=0, ge=0, le=60)

    @model_validator(mode="after")
    def validate_design(self) -> Self:
        if self.layouts % 4:
            raise ValueError("Feedback layouts must be balanced across four source families")
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise ValueError("Feedback conditions must be distinct and nonempty")
        if len(set(self.models)) != len(self.models):
            raise ValueError("Feedback models must be distinct")
        if self.request_token_reservation < self.max_output_tokens:
            raise ValueError("Each token reservation must cover its completion cap")
        if self.max_model_requests < self.max_requests():
            raise ValueError("Request allocation must cover every attempt of every episode")
        return self

    def max_requests(self) -> int:
        # The first attempt is one shared request per layout and model.
        later = (self.max_attempts - 1) * len(self.conditions)
        return self.layouts * (1 + later) * len(self.models)


class FeedbackCase(Record):
    case_id: str
    source_group: str
    family: str
    goal: GoalInput
    physics: PhysicsConfig


def development_feedback_cases(
    panel: Path, native_run: Path, config: FeedbackStudyConfig
) -> tuple[tuple[FeedbackCase, ...], dict[str, object]]:
    """The planning pilot's fixed development cases, so classical results match case for case."""
    validation = validate_diagnostic_panel(panel)
    cases = []
    for case in select_planning_cases(
        panel, native_run, PlanningPilotConfig(layouts=config.layouts, seed=config.case_seed)
    ):
        goal = load_synthetic_panel_call(panel, case.request).input
        if not isinstance(goal, GoalInput):
            raise ValueError("Feedback episodes require a public goal contract")
        cases.append(
            FeedbackCase(
                case_id=case.request.case_id,
                source_group=case.source_group,
                family=case.family,
                goal=goal,
                physics=case.physics,
            )
        )
    return tuple(cases), {
        "panel_manifest_sha256": validation["manifest_sha256"],
        "native_manifest_sha256": file_sha256(native_run / "manifest.json"),
    }


class JudgedAttempt(Record):
    success: bool
    max_position_error_radii: float | None
    failure: str | None
    tolerances: dict[str, bool]


class AttemptRow(Record):
    attempt: int = Field(ge=1)
    reply: ModelReply | None
    error_code: str | None
    action: SubmittedAction | None
    feedback_action: CueAction | None
    shown_outcome: SimulatedOutcome | None
    judged: JudgedAttempt | None


class EpisodeRow(Record):
    case_id: str
    source_group: str
    family: str
    model: str
    condition: Condition
    stop_reason: Literal[
        "feedback_goal_reached", "max_attempts", "invalid_submission", "service_error"
    ]
    attempts: tuple[AttemptRow, ...] = Field(min_length=1)
    final_success: bool
    elapsed_s: float = Field(ge=0)


def shuffled_action(seed: int, case_id: str, attempt: int) -> CueAction:
    unit = np.random.default_rng(int(fingerprint((seed, case_id, attempt))[:8], 16)).random(4)
    return unit_action(unit, SearchConfig(method="random", seed=0, max_objective_calls=0))


def simulated_outcome(
    goal: GoalInput,
    action: CueAction,
    simulator: SnookerSimulator,
    ledger: TrialLedger,
    *,
    namespace: str,
    role: Literal["agent_search", "hidden_judge"],
) -> SimulatedOutcome:
    try:
        replay, _ = recorded_replay(
            goal.initial, action, simulator, ledger, namespace=namespace, role=role
        )
    except SimulationFailure:
        return SimulatedOutcome(
            resting_state=None, goal_reached=False, failure="simulation_failure"
        )
    radius = goal.physics.table.ball_radius_m
    score = score_endpoint(
        replay.goal, goal.goal, goal.tolerance_radii * radius, no_scratch=goal.no_scratch
    )
    return SimulatedOutcome.model_validate(
        {
            "resting_state": replay.goal,
            "events": semantic_contacts(goal.initial, replay.events),
            "goal_reached": score.success,
            "max_position_error_radii": score.max_position_error_m / radius
            if score.max_position_error_m is not None
            else None,
            "failure": score.failure,
        }
    )


def judge_attempt(
    goal: GoalInput,
    action: CueAction,
    simulator: SnookerSimulator,
    ledger: TrialLedger,
    run_hash: str,
) -> JudgedAttempt:
    radius = goal.physics.table.ball_radius_m
    try:
        replay, _ = recorded_replay(
            goal.initial,
            action,
            simulator,
            ledger,
            namespace=f"{run_hash}/{goal.case_id}/hidden-judge",
            role="hidden_judge",
        )
    except SimulationFailure:
        return JudgedAttempt(
            success=False,
            max_position_error_radii=None,
            failure="simulation_failure",
            tolerances={str(k): False for k in TOLERANCE_RADII},
        )
    score = score_endpoint(
        replay.goal, goal.goal, goal.tolerance_radii * radius, no_scratch=goal.no_scratch
    )
    return JudgedAttempt(
        success=score.success,
        max_position_error_radii=score.max_position_error_m / radius
        if score.max_position_error_m is not None
        else None,
        failure=score.failure,
        tolerances=endpoint_tolerance_profile(score, radius),
    )


def run_episode(
    case: FeedbackCase,
    goal: GoalInput,
    model: str,
    condition: Condition,
    adapter: StudyAdapter,
    simulator: SnookerSimulator,
    ledger: TrialLedger,
    config: FeedbackStudyConfig,
    run_hash: str,
    token_provider: Callable[[], str],
) -> tuple[EpisodeRow, str | None]:
    started = monotonic()
    history: list[AttemptFeedback] = []
    attempts: list[AttemptRow] = []
    stop_reason: Literal[
        "feedback_goal_reached", "max_attempts", "invalid_submission", "service_error"
    ] = "max_attempts"
    stop_error = None
    for attempt in range(1, config.max_attempts + 1):
        call = StudyCall(
            view="structured",
            input=FeedbackGoalInput(
                goal=goal,
                attempt=attempt,
                max_attempts=config.max_attempts,
                simulator_feedback=None if attempt == 1 else condition != "reflection",
                history=tuple(history),
            ),
        )
        if not credential_ready(token_provider):
            # Earlier attempts are cached in the ledger, so the episode replays on resume.
            raise CredentialUnavailable("Credential refresh failed before an episode request")
        outcome = guarded_request(partial(request_study_model, adapter, call, ledger))
        if outcome.reply is None or outcome.error_code is not None:
            attempts.append(
                AttemptRow(
                    attempt=attempt,
                    reply=outcome.reply,
                    error_code=outcome.error_code,
                    action=None,
                    feedback_action=None,
                    shown_outcome=None,
                    judged=None,
                )
            )
            stop_error = outcome.stop_error
            stop_reason = "service_error" if stop_error else "invalid_submission"
            break
        action = GoalSubmission.model_validate_json(outcome.reply.content).action
        judged = judge_attempt(goal, action, simulator, ledger, run_hash)
        feedback_action = (
            action
            if condition == "genuine"
            else shuffled_action(config.shuffle_seed, goal.case_id, attempt)
            if condition == "shuffled"
            else None
        )
        shown = (
            simulated_outcome(
                goal,
                feedback_action,
                simulator,
                ledger,
                namespace=f"{run_hash}/{goal.case_id}/agent-feedback",
                role="agent_search",
            )
            if feedback_action is not None
            else None
        )
        if condition == "genuine" and shown is not None and shown.goal_reached != judged.success:
            raise ValueError("Independent hidden judge disagrees with agent-visible feedback")
        attempts.append(
            AttemptRow(
                attempt=attempt,
                reply=outcome.reply,
                error_code=None,
                action=action,
                feedback_action=feedback_action,
                shown_outcome=shown,
                judged=judged,
            )
        )
        history.append(AttemptFeedback(attempt=attempt, action=action, outcome=shown))
        if shown is not None and shown.goal_reached:
            stop_reason = "feedback_goal_reached"
            break
    last = attempts[-1]
    return (
        EpisodeRow(
            case_id=goal.case_id,
            source_group=case.source_group,
            family=case.family,
            model=model,
            condition=condition,
            stop_reason=stop_reason,
            attempts=tuple(attempts),
            final_success=last.judged is not None and last.judged.success,
            elapsed_s=monotonic() - started,
        ),
        stop_error,
    )


def summarize_feedback(rows: list[EpisodeRow], config: FeedbackStudyConfig) -> dict[str, object]:
    groups: dict[tuple[str, str], list[EpisodeRow]] = defaultdict(list)
    for row in rows:
        groups[row.model, row.condition].append(row)
    arms = []
    for (model, condition), group in sorted(groups.items()):
        successes = sum(row.final_success for row in group)
        interval = binomtest(successes, len(group)).proportion_ci()
        capped = {}
        anytime = {}
        for cap in range(1, config.max_attempts + 1):
            capped[str(cap)] = sum(
                row.final_success
                if cap >= len(row.attempts)
                else bool((judged := row.attempts[cap - 1].judged) and judged.success)
                for row in group
            )
            anytime[str(cap)] = sum(
                any(a.judged is not None and a.judged.success for a in row.attempts[:cap])
                for row in group
            )
        finals = [row.attempts[-1].judged for row in group]
        arms.append(
            {
                "model": model,
                "condition": condition,
                "feedback_shown": CONDITION_MEANING[condition],
                "episodes": len(group),
                "final_successes": successes,
                "final_success_rate_ci95": [interval.low, interval.high],
                "final_success_by_tolerance_radii": {
                    str(k): sum(bool(j and j.tolerances[str(k)]) for j in finals)
                    for k in TOLERANCE_RADII
                },
                "success_if_capped_at_attempt": capped,
                "any_attempt_success_by_attempt": anytime,
                "stop_reasons": dict(Counter(row.stop_reason for row in group)),
                "mean_attempts_per_episode": float(np.mean([len(row.attempts) for row in group])),
                "final_success_by_family": {
                    family: sum(row.final_success for row in group if row.family == family)
                    for family in FAMILIES
                },
                "usage": usage_summary(
                    [a.reply for row in group for a in row.attempts if a.reply is not None]
                ),
            }
        )
    comparisons = []
    for model in config.models:
        for left, right in (
            ("genuine", "shuffled"),
            ("genuine", "reflection"),
            ("shuffled", "reflection"),
        ):
            a = {row.case_id: row for row in groups.get((model, left), [])}
            b = {row.case_id: row for row in groups.get((model, right), [])}
            common = sorted(a.keys() & b.keys())
            if not common:
                continue
            comparisons.append(
                {
                    "model": model,
                    "condition_a": left,
                    "condition_b": right,
                    "final_success_difference": summarize_paired(
                        tuple(
                            PairedMeasurement(
                                pair_id=case_id,
                                group_id=a[case_id].source_group,
                                a=float(a[case_id].final_success),
                                b=float(b[case_id].final_success),
                            )
                            for case_id in common
                        ),
                        seed=20261019,
                    ).model_dump(mode="json"),
                }
            )
    return {"by_model_condition": arms, "paired_condition_comparisons": comparisons}


def run_feedback_study(
    cases: tuple[FeedbackCase, ...],
    case_source: dict[str, object],
    source: Path,
    output: Path,
    config: FeedbackStudyConfig,
    *,
    endpoint_index: int = 4,
    execute: bool = False,
    token_provider: Callable[[], str] | None = None,
    transport: httpx.BaseTransport | None = None,
    limit_episodes: int | None = None,
) -> dict[str, object]:
    if len(cases) != config.layouts or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Feedback cases must be the declared number of distinct layouts")
    backends = azure_model_backends(
        source,
        endpoint_index,
        config.models,
        reasoning_effort=config.reasoning_effort,
        max_output_tokens=config.max_output_tokens,
        request_token_reservation=config.request_token_reservation,
        timeout_s=config.timeout_s,
    )
    models = config.models
    agent_rollouts = config.layouts * config.max_attempts * (len(models) + 1)
    judge_rollouts = config.layouts * config.max_attempts * len(config.conditions) * len(models)
    limits = BudgetLimits(
        max_model_requests=config.max_model_requests,
        max_total_tokens=config.max_total_tokens,
        max_physics_evaluations=agent_rollouts + judge_rollouts,
    )
    protocol: dict[str, object] = {
        **case_source,
        "azure_inventory_sha256": file_sha256(source),
        "endpoint_hash": fingerprint(backends[0].base_url),
        "config": config.model_dump(mode="json"),
        "backends": [backend.model_dump(mode="json") for backend in backends],
        "case_ids": [case.case_id for case in cases],
        "feedback_conditions": {name: CONDITION_MEANING[name] for name in config.conditions},
        "limits": limits.model_dump(mode="json"),
        "producer": producer_hashes(
            "feedback_study.py",
            "study_calls.py",
            "planning_pilot.py",
            "baselines.py",
            "model_adapters.py",
            "evaluation.py",
            "metrics.py",
            "study.py",
            "physics.py",
        ),
        "real_data_sent": False,
        "formal_evaluation": False,
    }
    run_hash = lock_run(output, protocol, reserve_bytes=200_000_000)
    planned = len(cases) * len(models) * len(config.conditions)
    if not execute:
        return {
            "status": "planned_no_calls",
            "run_hash": run_hash,
            "planned_episodes": planned,
            "maximum_requests": config.max_requests(),
        }
    if token_provider is None:
        raise ValueError("Azure execution requires a runtime bearer-token provider")
    token_provider()
    rows: list[EpisodeRow] = []
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
            for adapter in adapters.values():
                adapter.min_request_interval_s = config.min_request_interval_s
            for case in cases:
                goal = case.goal
                key = fingerprint(case.physics.model_dump(mode="json"))
                for model in models:
                    for condition in config.conditions:
                        path = output / "episodes" / model / condition / f"{goal.case_id}.json"
                        cached = load_job(path, run_hash)
                        if cached is not None:
                            row = EpisodeRow.model_validate(cached)
                            if (row.case_id, row.model, row.condition) != (
                                goal.case_id,
                                model,
                                condition,
                            ):
                                raise ValueError("Cached feedback episode identity changed")
                            rows.append(row)
                            continue
                        if stop_error or (
                            limit_episodes is not None and completed >= limit_episodes
                        ):
                            continue
                        if key not in simulators:
                            simulators[key] = SnookerSimulator(case.physics)
                            if goal.physics != public_physics(simulators[key]):
                                raise ValueError("Feedback runs require the declared known physics")
                        try:
                            row, stop_error = run_episode(
                                case,
                                goal,
                                model,
                                condition,
                                adapters[model],
                                simulators[key],
                                ledger,
                                config,
                                run_hash,
                                token_provider,
                            )
                        except CredentialUnavailable:
                            stop_error = "credential_unavailable"
                            continue
                        save_job(path, run_hash, row.model_dump(mode="json"))
                        rows.append(row)
                        completed += 1
                        LOGGER.info(
                            "Feedback episode %d/%d: %s/%s", len(rows), planned, model, condition
                        )
        report: dict[str, object] = {
            "status": "stopped_service_error"
            if stop_error
            else ("completed" if len(rows) == planned else "partial"),
            "run_hash": run_hash,
            "stop_error": stop_error,
            "independent_layouts": len(cases),
            "planned_episodes": planned,
            "completed_episodes": len(rows),
            "new_model_requests": ledger.counts()["model_requests"] - before,
            "resource_counts": ledger.counts(),
            "physics_usage_by_role": ledger.physics_usage_by_role(),
            "usage": ledger.usage_totals(),
            **summarize_feedback(rows, config),
            "real_data_sent": False,
            "formal_evaluation": False,
        }
        write_json(output / "summary.json", report)
        return report
    finally:
        ledger.close()
