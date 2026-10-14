import logging
from math import atan2, ceil, degrees, hypot, pi, sqrt
from time import perf_counter
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import qmc

from inverser.evaluation import (
    BudgetExhausted,
    CachedSimulationFailure,
    TrialLedger,
    recorded_replay,
)
from inverser.metrics import endpoint_loss, score_endpoint
from inverser.physics import SimulationFailure, SnookerSimulator
from inverser.runtime import fingerprint
from inverser.schemas import CueAction, Record
from inverser.study import ActionBank, CandidateAction, GoalInput, public_physics

LOGGER = logging.getLogger(__name__)
BaselineMethod = Literal["random", "sobol", "geometry_powell"]


class SearchConfig(Record):
    method: BaselineMethod
    seed: int = Field(ge=0)
    max_objective_calls: int = Field(ge=0, le=512)
    min_cue_speed_mps: float = Field(default=0.05, gt=0, le=3)
    max_cue_speed_mps: float = Field(default=3.0, gt=0, le=3)
    max_starts: int = Field(default=4, ge=1, le=16)

    @model_validator(mode="after")
    def validate_speed(self) -> Self:
        if self.min_cue_speed_mps >= self.max_cue_speed_mps:
            raise ValueError("Search speed bounds must form a nonempty interval")
        return self


class SearchAttempt(Record):
    objective_call: int = Field(ge=1)
    action: CueAction
    status: Literal["evaluated", "simulation_failure"]
    loss_radii: float | None
    goal_success: bool
    max_position_error_m: float | None
    cache_hit: bool
    elapsed_s: float = Field(ge=0)


class SearchResult(Record):
    method: BaselineMethod
    seed: int
    stop_reason: Literal[
        "goal_reached", "objective_budget_exhausted", "ledger_budget_exhausted", "zero_budget"
    ]
    selected_action: CueAction | None
    selected_loss_radii: float | None
    selected_goal_success: bool
    attempts: tuple[SearchAttempt, ...]
    actual_new_physics_evaluations: int = Field(ge=0)
    elapsed_s: float = Field(ge=0)


class _SearchStop(Exception):
    pass


def unit_action(unit: NDArray[np.float64], config: SearchConfig) -> CueAction:
    if unit.shape != (4,) or not np.isfinite(unit).all() or np.any((unit < 0) | (unit > 1)):
        raise ValueError("Search coordinates must be a finite point in the four-dimensional cube")
    radius = 0.5 * sqrt(float(unit[2]))
    angle = 2 * pi * float(unit[3])
    return CueAction(
        azimuth_deg=(360 * float(unit[0])) % 360,
        cue_speed_mps=config.min_cue_speed_mps
        + float(unit[1]) * (config.max_cue_speed_mps - config.min_cue_speed_mps),
        tip_side=radius * float(np.cos(angle)),
        tip_vertical=radius * float(np.sin(angle)),
        elevation_deg=0,
    )


def action_unit(action: CueAction, config: SearchConfig) -> NDArray[np.float64]:
    return np.asarray(
        (
            action.azimuth_deg / 360,
            (action.cue_speed_mps - config.min_cue_speed_mps)
            / (config.max_cue_speed_mps - config.min_cue_speed_mps),
            4 * (action.tip_side**2 + action.tip_vertical**2),
            (atan2(action.tip_vertical, action.tip_side) % (2 * pi)) / (2 * pi),
        ),
        dtype=np.float64,
    )


def sampled_actions(config: SearchConfig, count: int) -> tuple[CueAction, ...]:
    if not 0 <= count <= 512:
        raise ValueError("A bounded search proposal batch permits 0-512 candidates")
    if not count:
        return ()
    if config.method == "random":
        units = np.random.default_rng(config.seed).random((count, 4))
    else:
        units = qmc.Sobol(d=4, scramble=True, seed=config.seed).random_base2(
            (count - 1).bit_length()
        )[:count]
    return tuple(unit_action(unit, config) for unit in units)


def make_action_bank(
    case_id: str, *, count: int, seed: int, min_speed_mps: float = 0.05, max_speed_mps: float = 3.0
) -> ActionBank:
    if not 1 <= count <= 32:
        raise ValueError("A fixed candidate bank permits 1-32 independent-prior actions")
    config = SearchConfig(
        method="sobol",
        seed=seed,
        max_objective_calls=count,
        min_cue_speed_mps=min_speed_mps,
        max_cue_speed_mps=max_speed_mps,
    )
    candidates = tuple(
        CandidateAction(
            candidate_id="candidate-"
            + fingerprint({"seed": seed, "index": index, "action": action.model_dump(mode="json")})[
                :16
            ],
            action=action,
        )
        for index, action in enumerate(sampled_actions(config, count))
    )
    order = np.random.default_rng(seed).permutation(count)
    return ActionBank(
        case_id=case_id,
        seed=seed,
        min_cue_speed_mps=min_speed_mps,
        max_cue_speed_mps=max_speed_mps,
        candidates=tuple(candidates[int(index)] for index in order),
    )


def geometry_actions(goal: GoalInput, config: SearchConfig) -> tuple[CueAction, ...]:
    cue = next(ball for ball in goal.initial.balls if ball.kind == "cue")
    if cue.xy_m is None or any(ball.xy_m is None for ball in goal.initial.balls):
        raise ValueError("Geometric planning requires a complete on-table initial state")
    physics = goal.physics.parameters
    if goal.physics.withheld_parameters:
        raise ValueError(
            "This baseline requires declared known physics, not hidden true parameters"
        )
    coefficient = (
        12 / (49 * physics["sliding_friction"]) + 25 / (98 * physics["rolling_friction"])
    ) / physics["gravity_mps2"]
    transfer = 2 * goal.physics.cue.mass_kg / (goal.physics.cue.mass_kg + physics["ball_mass_kg"])
    actions = []

    def propose(point: tuple[float, float], travel_m: float) -> None:
        if travel_m <= 1e-10:
            return
        if cue.xy_m is None:
            raise ValueError("Cue position is unavailable")
        dx, dy = point[0] - cue.xy_m[0], point[1] - cue.xy_m[1]
        if hypot(dx, dy) <= 1e-10:
            return
        speed = sqrt(travel_m / coefficient) / transfer
        actions.append(
            CueAction(
                azimuth_deg=degrees(atan2(dy, dx)) % 360,
                cue_speed_mps=min(config.max_cue_speed_mps, max(config.min_cue_speed_mps, speed)),
                tip_side=0,
                tip_vertical=0,
                elevation_deg=0,
            )
        )

    target_cue = next(ball for ball in goal.goal.balls if ball.kind == "cue")
    if target_cue.xy_m is not None:
        propose(target_cue.xy_m, hypot(*(np.subtract(target_cue.xy_m, cue.xy_m))))
    for kind in sorted({ball.kind for ball in goal.initial.balls} - {"cue"}):
        sources = [ball for ball in goal.initial.balls if ball.kind == kind]
        targets = [ball for ball in goal.goal.balls if ball.kind == kind and ball.xy_m is not None]
        if targets:
            source_points = [ball.xy_m for ball in sources if ball.xy_m is not None]
            target_points = [ball.xy_m for ball in targets if ball.xy_m is not None]
            if len(source_points) != len(sources) or len(target_points) != len(targets):
                raise ValueError("Matched geometry has missing coordinates")
            costs = np.asarray(
                [[np.linalg.norm(np.subtract(a, b)) for b in target_points] for a in source_points]
            )
            rows, columns = linear_sum_assignment(costs)
            for i, j in zip(rows, columns, strict=True):
                source, target = sources[i].xy_m, targets[j].xy_m
                if source is None or target is None:
                    raise ValueError("Matched geometry has missing coordinates")
                dx, dy = target[0] - source[0], target[1] - source[1]
                distance = hypot(dx, dy)
                if distance <= 1e-10:
                    continue
                radius = goal.physics.table.ball_radius_m
                ghost = (
                    source[0] - 2 * radius * dx / distance,
                    source[1] - 2 * radius * dy / distance,
                )
                propose(ghost, hypot(*(np.subtract(ghost, cue.xy_m))) + distance)
        if any(ball.kind == kind and ball.status == "pocketed" for ball in goal.goal.balls):
            for source_ball in sources:
                source = source_ball.xy_m
                if source is None:
                    raise ValueError("Object-ball position is unavailable")
                for pocket in goal.physics.table.pockets:
                    dx, dy = pocket.xy_m[0] - source[0], pocket.xy_m[1] - source[1]
                    distance = hypot(dx, dy)
                    if distance <= 1e-10:
                        continue
                    radius = goal.physics.table.ball_radius_m
                    ghost = (
                        source[0] - 2 * radius * dx / distance,
                        source[1] - 2 * radius * dy / distance,
                    )
                    propose(ghost, hypot(*(np.subtract(ghost, cue.xy_m))) + distance)
    return tuple(actions)


def search_goal(
    goal: GoalInput,
    config: SearchConfig,
    simulator: SnookerSimulator,
    ledger: TrialLedger,
    *,
    namespace: str,
) -> SearchResult:
    if goal.physics.withheld_parameters or goal.physics != public_physics(simulator):
        raise ValueError("Known-physics search must use the declared public engine configuration")
    start_time = perf_counter()
    before = ledger.counts()["physics_evaluations"]
    attempts: list[SearchAttempt] = []
    best: SearchAttempt | None = None
    stop_reason: Literal[
        "goal_reached", "objective_budget_exhausted", "ledger_budget_exhausted", "zero_budget"
    ] = "objective_budget_exhausted"
    diagonal = hypot(goal.physics.table.width_m, goal.physics.table.length_m)
    radius = goal.physics.table.ball_radius_m
    run_namespace = (
        namespace
        + "/"
        + fingerprint(
            {"goal": goal.model_dump(mode="json"), "config": config.model_dump(mode="json")}
        )
    )

    def evaluate(action: CueAction) -> float:
        nonlocal best
        if len(attempts) >= config.max_objective_calls or (best is not None and best.goal_success):
            raise _SearchStop
        call_start = perf_counter()
        try:
            replay, cached = recorded_replay(
                goal.initial,
                action,
                simulator,
                ledger,
                namespace=run_namespace,
                role="agent_search",
            )
        except SimulationFailure as error:
            LOGGER.error("Baseline %s candidate failed to simulate", config.method)
            attempt = SearchAttempt(
                objective_call=len(attempts) + 1,
                action=action,
                status="simulation_failure",
                loss_radii=None,
                goal_success=False,
                max_position_error_m=None,
                cache_hit=isinstance(error, CachedSimulationFailure),
                elapsed_s=perf_counter() - call_start,
            )
            attempts.append(attempt)
            return 3 * (diagonal / radius + 1)
        score = score_endpoint(
            replay.goal, goal.goal, goal.tolerance_radii * radius, no_scratch=goal.no_scratch
        )
        loss = endpoint_loss(
            replay.goal,
            goal.goal,
            ball_radius_m=radius,
            table_diagonal_m=diagonal,
            no_scratch=goal.no_scratch,
        )
        attempt = SearchAttempt(
            objective_call=len(attempts) + 1,
            action=action,
            status="evaluated",
            loss_radii=loss,
            goal_success=score.success,
            max_position_error_m=score.max_position_error_m,
            cache_hit=cached,
            elapsed_s=perf_counter() - call_start,
        )
        attempts.append(attempt)
        if best is None or best.loss_radii is None or loss < best.loss_radii:
            best = attempt
        return loss

    try:
        if config.max_objective_calls == 0:
            stop_reason = "zero_budget"
        elif config.method in ("random", "sobol"):
            for action in sampled_actions(config, config.max_objective_calls):
                evaluate(action)
        else:
            seeds = (geometry_actions(goal, config) + sampled_actions(config, config.max_starts))[
                : config.max_starts
            ]
            for index, seed in enumerate(seeds):
                remaining = config.max_objective_calls - len(attempts)
                if remaining <= 0:
                    break
                allocation = ceil(remaining / (len(seeds) - index))
                minimize(
                    lambda unit: evaluate(unit_action(unit, config)),
                    action_unit(seed, config),
                    method="Powell",
                    bounds=[(0.0, 1.0)] * 4,
                    options={
                        "maxfev": allocation,
                        "maxiter": allocation,
                        "xtol": 0.01,
                        "ftol": 0.01,
                    },
                )
    except _SearchStop:
        pass
    except BudgetExhausted:
        LOGGER.warning("Baseline %s stopped at the persistent ledger cap", config.method)
        stop_reason = "ledger_budget_exhausted"
    if best is not None and best.goal_success:
        stop_reason = "goal_reached"
    return SearchResult(
        method=config.method,
        seed=config.seed,
        stop_reason=stop_reason,
        selected_action=best.action if best else None,
        selected_loss_radii=best.loss_radii if best else None,
        selected_goal_success=best.goal_success if best else False,
        attempts=tuple(attempts),
        actual_new_physics_evaluations=ledger.counts()["physics_evaluations"] - before,
        elapsed_s=perf_counter() - start_time,
    )
