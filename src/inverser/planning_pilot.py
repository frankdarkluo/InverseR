import fcntl
import json
import logging
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from math import hypot
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import binomtest

from inverser.azure_study import MODELS, PilotRow
from inverser.baselines import (
    BaselineMethod,
    SearchConfig,
    SearchResult,
    geometry_actions,
    search_goal,
)
from inverser.diagnostic_panel import (
    FAMILIES,
    PanelRequest,
    load_synthetic_panel_call,
    validate_diagnostic_panel,
)
from inverser.diagnostics import PairedMeasurement, summarize_paired
from inverser.evaluation import BudgetLimits, TrialLedger, recorded_replay
from inverser.metrics import EndpointScore, endpoint_loss, score_endpoint
from inverser.model_adapters import ModelAdapter
from inverser.physics import SimulationFailure, SnookerSimulator
from inverser.runtime import PROJECT_ROOT, check_storage, file_sha256, fingerprint, write_json
from inverser.schemas import CueAction, PhysicsConfig, Record
from inverser.study import GoalInput, GoalSubmission
from inverser.study_runner import JobEnvelope

LOGGER = logging.getLogger(__name__)
METHODS = (
    "geometry_direct",
    "gpt-5-mini_direct",
    "gpt-5_direct",
    "random",
    "sobol",
    "geometry_powell",
    "gpt-5-mini_powell",
    "gpt-5_powell",
)


class PlanningPilotConfig(Record):
    schema_version: Literal["cpu-planning-pilot-v1"] = "cpu-planning-pilot-v1"
    layouts: int = Field(default=32, ge=4, le=32)
    seed: int = Field(default=20261017, ge=0)
    checkpoints: tuple[int, ...] = (8, 32, 128)
    workers: int = Field(default=4, ge=1, le=4)

    @model_validator(mode="after")
    def validate_design(self) -> Self:
        if self.layouts % 4:
            raise ValueError("Planning layouts must be balanced across four source families")
        if not self.checkpoints or any(not 1 <= item <= 128 for item in self.checkpoints):
            raise ValueError("Planning checkpoints must be bounded by 1-128 search queries")
        if tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("Planning checkpoints must be distinct and increasing")
        return self


class NativeSeed(Record):
    model: str
    action: CueAction | None
    original_score: EndpointScore | None
    error_code: str | None
    source_file: str
    source_sha256: str


class PlanningCase(Record):
    request: PanelRequest
    source_group: str
    family: str
    initial_hash: str
    physics: PhysicsConfig
    native: tuple[NativeSeed, ...]


class PlanningCheckpoint(Record):
    query_budget: int = Field(ge=0)
    objective_calls: int = Field(ge=0)
    selected_action: CueAction | None
    loss_radii: float | None
    score: EndpointScore | None
    error_code: str | None


class PlanningJob(Record):
    case_id: str
    method: str
    search: SearchResult | None = None
    checkpoints: tuple[PlanningCheckpoint, ...]


class CaseRun(Record):
    case_id: str
    jobs: tuple[PlanningJob, ...]
    counts: dict[str, int]
    physics_usage_by_role: dict[str, dict[str, int]]
    new_physics_evaluations: int
    new_jobs: int


def select_planning_cases(
    panel: Path, native_run: Path, config: PlanningPilotConfig
) -> tuple[PlanningCase, ...]:
    manifest = json.loads((panel / "manifest.json").read_text())
    native_manifest = json.loads((native_run / "manifest.json").read_text())
    if native_manifest["panel_manifest_sha256"] != file_sha256(panel / "manifest.json"):
        raise ValueError("Cached native proposals must use the identical public panel")
    if native_manifest["config"]["design"] != "goal_structured":
        raise ValueError("Cached native proposals must come from the structured-goal design")
    requests = {
        request.case_id: request
        for request in (
            PanelRequest.model_validate_json(line)
            for line in (panel / "request_index.jsonl").open()
        )
        if request.task == "synthetic_goal" and request.view == "structured"
    }
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in json.loads((panel / "private/selection.json").read_text())["synthetic"]:
        groups[item["family"]].append(item)
    chosen = []
    for family in FAMILIES:
        pool = sorted(
            groups[family],
            key=lambda item: fingerprint((config.seed, item["source_group"], item["case_id"])),
        )
        if len(pool) < config.layouts // 4:
            raise ValueError("Too few distinct source groups for the balanced planning pilot")
        chosen.extend(pool[: config.layouts // 4])
    if any(
        len({item[name] for item in chosen}) != config.layouts
        for name in ("case_id", "source_group", "initial_hash")
    ):
        raise ValueError("Planning cases must have distinct layout hashes and source groups")
    physics = {
        fingerprint(item["config"]): PhysicsConfig.model_validate(item["config"])
        for item in manifest["resolved_physics"]
    }
    result = []
    for item in chosen:
        request = requests[item["case_id"]]
        if request.local_only or not request.model_adapter_ready:
            raise ValueError("Planning pilots require executable synthetic public goals")
        call = load_synthetic_panel_call(panel, request)
        native = []
        for model in MODELS:
            source = native_run / "jobs" / model / f"{request.request_id}.json"
            envelope = JobEnvelope.model_validate_json(source.read_text())
            row = PilotRow.model_validate(envelope.payload)
            if envelope.run_hash != native_manifest["run_hash"] or (
                row.case_id,
                row.request_id,
                row.model,
                row.task,
                row.view,
            ) != (request.case_id, request.request_id, model, "synthetic_goal", "structured"):
                raise ValueError("Cached proposal identity does not match the selected goal")
            action = None
            if row.valid_output:
                if row.reply is None or row.error_code is not None:
                    raise ValueError("A valid cached native proposal must contain a billed reply")
                ModelAdapter.validate_submission(call, row.reply.content)
                action = GoalSubmission.model_validate_json(row.reply.content).action
            elif row.error_code != "invalid_submission":
                raise ValueError("Unresolved service errors cannot be treated as native failures")
            native.append(
                NativeSeed(
                    model=model,
                    action=action,
                    original_score=row.goal_score,
                    error_code=row.error_code,
                    source_file=str(source.resolve()),
                    source_sha256=file_sha256(source),
                )
            )
        if len(physics) != 1:
            raise ValueError("This bounded pilot requires one declared known-physics configuration")
        result.append(
            PlanningCase(
                request=request,
                source_group=item["source_group"],
                family=item["family"],
                initial_hash=item["initial_hash"],
                physics=next(iter(physics.values())),
                native=tuple(native),
            )
        )
    return tuple(result)


def _judge_checkpoint(
    goal: GoalInput,
    action: CueAction | None,
    simulator: SnookerSimulator,
    ledger: TrialLedger,
    namespace: str,
    budget: int,
    calls: int,
) -> PlanningCheckpoint:
    score = None
    loss = None
    error = "no_selected_action" if action is None else None
    if action is not None:
        try:
            replay, _ = recorded_replay(
                goal.initial,
                action,
                simulator,
                ledger,
                namespace=namespace,
                role="hidden_judge",
            )
        except SimulationFailure:
            error = "hidden_judge_simulation_failure"
            LOGGER.error("Planning hidden judge failed for %s", goal.case_id)
        else:
            radius = goal.physics.table.ball_radius_m
            score = score_endpoint(
                replay.goal, goal.goal, goal.tolerance_radii * radius, no_scratch=goal.no_scratch
            )
            loss = endpoint_loss(
                replay.goal,
                goal.goal,
                ball_radius_m=radius,
                table_diagonal_m=hypot(goal.physics.table.width_m, goal.physics.table.length_m),
                no_scratch=goal.no_scratch,
            )
    return PlanningCheckpoint(
        query_budget=budget,
        objective_calls=calls,
        selected_action=action,
        loss_radii=loss,
        score=score,
        error_code=error,
    )


def _run_case(
    panel: Path, output: Path, case: PlanningCase, config: PlanningPilotConfig, run_hash: str
) -> CaseRun:
    folder = output / "cases" / case.request.case_id
    max_budget = config.checkpoints[-1]
    limits = BudgetLimits(
        max_model_requests=0,
        max_physics_evaluations=5 * max_budget + 5 * len(config.checkpoints) + 3,
    )
    ledger = TrialLedger(folder / "ledger.sqlite", limits)
    before = ledger.counts()["physics_evaluations"]
    simulator = None
    jobs = []
    new_jobs = 0
    try:
        call = load_synthetic_panel_call(panel, case.request)
        if not isinstance(call.input, GoalInput):
            raise ValueError("Planning pilot requires a public goal contract")
        goal = call.input
        native = {seed.model: seed for seed in case.native}
        for method in METHODS:
            path = folder / f"{method}.json"
            if path.exists():
                envelope = JobEnvelope.model_validate_json(path.read_text())
                job = PlanningJob.model_validate(envelope.payload)
                if envelope.run_hash != run_hash or (job.case_id, job.method) != (
                    goal.case_id,
                    method,
                ):
                    raise ValueError("Cached planning job identity changed")
                jobs.append(job)
                continue
            if simulator is None:
                simulator = SnookerSimulator(case.physics)
            source = native.get(method.removesuffix("_direct").removesuffix("_powell"))
            if source is not None and file_sha256(Path(source.source_file)) != source.source_sha256:
                raise ValueError("Cached native proposal changed during the planning run")
            initial = (source.action,) if source is not None and source.action is not None else ()
            seed = int(fingerprint((config.seed, case.source_group))[:8], 16)
            baseline: BaselineMethod = (
                "random"
                if method == "random"
                else "sobol"
                if method == "sobol"
                else "geometry_powell"
            )
            search_config = SearchConfig(method=baseline, seed=seed, max_objective_calls=max_budget)
            namespace = run_hash + "/" + goal.case_id
            search = None
            if source is not None and source.action is None:
                checkpoints = tuple(
                    PlanningCheckpoint(
                        query_budget=budget,
                        objective_calls=0,
                        selected_action=None,
                        loss_radii=None,
                        score=None,
                        error_code="invalid_native_seed",
                    )
                    for budget in ((0,) if method.endswith("_direct") else config.checkpoints)
                )
                LOGGER.warning("Retaining invalid native seed: %s/%s", goal.case_id, method)
            elif method.endswith("_direct"):
                actions = initial if source is not None else geometry_actions(goal, search_config)
                point = _judge_checkpoint(
                    goal,
                    actions[0] if actions else None,
                    simulator,
                    ledger,
                    namespace + "/hidden-judge",
                    0,
                    0,
                )
                if source is not None and point.score != source.original_score:
                    raise ValueError("Rejudged native proposal differs from its original score")
                checkpoints = (point,)
            else:
                search = search_goal(
                    goal,
                    search_config,
                    simulator,
                    ledger,
                    namespace=namespace + "/" + method,
                    initial_actions=initial,
                )
                points = []
                for budget in config.checkpoints:
                    prefix = search.attempts[:budget]
                    valid = [attempt for attempt in prefix if attempt.loss_radii is not None]
                    best = (
                        min(
                            valid,
                            key=lambda attempt: (
                                attempt.loss_radii
                                if attempt.loss_radii is not None
                                else float("inf")
                            ),
                        )
                        if valid
                        else None
                    )
                    point = _judge_checkpoint(
                        goal,
                        best.action if best else None,
                        simulator,
                        ledger,
                        namespace + "/hidden-judge",
                        budget,
                        len(prefix),
                    )
                    if best is not None and (
                        point.loss_radii != best.loss_radii
                        or (point.score is not None and point.score.success != best.goal_success)
                    ):
                        raise ValueError("Independent hidden judge disagrees with search feedback")
                    points.append(point)
                checkpoints = tuple(points)
            job = PlanningJob(
                case_id=goal.case_id, method=method, search=search, checkpoints=checkpoints
            )
            payload = job.model_dump(mode="json")
            write_json(
                path,
                JobEnvelope(
                    run_hash=run_hash, payload=payload, payload_sha256=fingerprint(payload)
                ).model_dump(mode="json"),
            )
            jobs.append(job)
            new_jobs += 1
        return CaseRun(
            case_id=goal.case_id,
            jobs=tuple(jobs),
            counts=ledger.counts(),
            physics_usage_by_role=ledger.physics_usage_by_role(),
            new_physics_evaluations=ledger.counts()["physics_evaluations"] - before,
            new_jobs=new_jobs,
        )
    finally:
        ledger.close()


def summarize_planning(
    cases: tuple[PlanningCase, ...], jobs: list[PlanningJob], config: PlanningPilotConfig
) -> dict[str, object]:
    grouped = {case.request.case_id: case for case in cases}
    summaries = []
    points: dict[tuple[str, int], dict[str, PlanningCheckpoint]] = defaultdict(dict)
    for job in jobs:
        for point in job.checkpoints:
            points[job.method, point.query_budget][job.case_id] = point
    for (method, budget), by_case in points.items():
        for family in (None, *FAMILIES):
            selected = {
                case_id: point
                for case_id, point in by_case.items()
                if family is None or grouped[case_id].family == family
            }
            successes = sum(
                point.score is not None and point.score.success for point in selected.values()
            )
            losses = [
                point.loss_radii for point in selected.values() if point.loss_radii is not None
            ]
            interval = binomtest(successes, len(selected)).proportion_ci() if selected else None
            summaries.append(
                {
                    "method": method,
                    "query_budget": budget,
                    "family": family,
                    "layouts": len(selected),
                    "goal_successes": successes,
                    "success_rate": successes / len(selected) if selected else None,
                    "success_rate_ci95": [interval.low, interval.high] if interval else None,
                    "mean_objective_calls": float(
                        np.mean([point.objective_calls for point in selected.values()])
                    )
                    if selected
                    else None,
                    "loss_scored_layouts": len(losses),
                    "median_loss_radii_scored_only": float(np.median(losses)) if losses else None,
                    "outcomes": dict(
                        Counter(
                            point.error_code
                            or (
                                point.score.failure or "success"
                                if point.score is not None
                                else "unscored"
                            )
                            for point in selected.values()
                        )
                    ),
                }
            )
    comparisons = []
    for budget in config.checkpoints:
        for method in ("gpt-5-mini_powell", "gpt-5_powell", "sobol", "random"):
            left, right = points[method, budget], points["geometry_powell", budget]
            pairs = []
            for case_id in sorted(left.keys() & right.keys()):
                a, b = left[case_id].score, right[case_id].score
                pairs.append(
                    PairedMeasurement(
                        pair_id=case_id,
                        group_id=grouped[case_id].source_group,
                        a=float(a is not None and a.success),
                        b=float(b is not None and b.success),
                    )
                )
            if pairs:
                comparisons.append(
                    {
                        "method_a": method,
                        "method_b": "geometry_powell",
                        "query_budget": budget,
                        "success_difference": summarize_paired(
                            tuple(pairs), seed=config.seed
                        ).model_dump(mode="json"),
                    }
                )
    return {"by_method_budget_family": summaries, "paired_success_comparisons": comparisons}


def run_planning_pilot(
    panel: Path,
    native_run: Path,
    output: Path,
    config: PlanningPilotConfig,
    *,
    limit_cases: int | None = None,
) -> dict[str, object]:
    validation = validate_diagnostic_panel(panel)
    cases = select_planning_cases(panel, native_run, config)
    if limit_cases is not None and not 1 <= limit_cases <= config.layouts:
        raise ValueError("A case limit must be inside the declared fixed panel")
    if output.is_symlink() or not output.resolve().is_relative_to(
        (PROJECT_ROOT / "data").resolve()
    ):
        raise ValueError("Planning pilot outputs must stay inside project data")
    protocol = {
        "config": config.model_dump(mode="json"),
        "panel_manifest_sha256": validation["manifest_sha256"],
        "native_manifest_sha256": file_sha256(native_run / "manifest.json"),
        "cases": [case.model_dump(mode="json") for case in cases],
        "runtime_versions": {
            name: version(name) for name in ("numpy", "scipy", "pydantic", "pooltool-billiards")
        },
        "producer_hashes": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "planning_pilot.py",
                "baselines.py",
                "evaluation.py",
                "metrics.py",
                "study.py",
                "physics.py",
                "diagnostics.py",
                "model_adapters.py",
            )
        },
        "new_model_requests": 0,
        "candidate_goal_witness_access": False,
        "anytime_search_horizon": config.checkpoints[-1],
        "formal_evaluation": False,
    }
    run_hash = fingerprint(protocol)
    manifest = {**protocol, "run_hash": run_hash}
    if not output.exists():
        check_storage(2_000_000_000)
        output.mkdir(parents=True)
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "manifest.json").exists():
            if json.loads((output / "manifest.json").read_text()) != manifest:
                raise ValueError("Changed planning inputs/code require a new run version")
        else:
            if any(path.name != ".writer.lock" for path in output.iterdir()):
                raise ValueError("A new planning run requires an empty output directory")
            write_json(output / "manifest.json", manifest)
        selected = cases if limit_cases is None else cases[:limit_cases]
        if config.workers == 1:
            results = [_run_case(panel, output, case, config, run_hash) for case in selected]
        else:
            with ProcessPoolExecutor(max_workers=config.workers) as executor:
                futures = [
                    executor.submit(_run_case, panel, output, case, config, run_hash)
                    for case in selected
                ]
                results = [future.result() for future in futures]
        jobs = [job for case_result in results for job in case_result.jobs]
        roles: dict[str, Counter[str]] = defaultdict(Counter)
        for case_result in results:
            for role, usage in case_result.physics_usage_by_role.items():
                roles[role].update(usage)
        report: dict[str, object] = {
            "status": "completed" if len(selected) == config.layouts else "partial",
            "run_hash": run_hash,
            "independent_layouts": len(selected),
            "family_counts": dict(Counter(case.family for case in selected)),
            "methods": METHODS,
            "completed_jobs": len(jobs),
            "new_jobs": sum(case_result.new_jobs for case_result in results),
            "new_physics_evaluations": sum(
                case_result.new_physics_evaluations for case_result in results
            ),
            "physics_usage_by_role": {role: dict(usage) for role, usage in roles.items()},
            "new_model_requests": 0,
            "new_model_tokens": 0,
            "historical_model_requests_per_layout": 2,
            "original_action_space": "cue speed/azimuth/contact disk; elevation zero",
            "anytime_search_horizon": config.checkpoints[-1],
            "formal_evaluation": False,
            **summarize_planning(cases, jobs, config),
        }
        write_json(output / "summary.json", report)
        return report
