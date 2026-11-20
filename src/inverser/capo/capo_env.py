"""Reward and feedback logic shared by CAPO rollout and offline evaluation."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from math import dist, hypot, isfinite
from typing import Self

from PIL import Image
from pydantic import Field, model_validator

from inverser.metrics import endpoint_loss, score_endpoint
from inverser.physics import Replay, SimulationFailure, SnookerSimulator
from inverser.preview import render_vlm_goal_pair
from inverser.schemas import CueAction, Layout, Record
from inverser.study import public_physics
from inverser.training_env import goal_reward
from inverser.vlm_suite import (
    Position,
    VLMForwardOutput,
    VLMGoalInput,
    forward_scores,
    parse_forward_answer,
    parse_goal_answer,
)

CAUSAL_PROMPT = (
    "In the reasoning segment, first identify ball positions, aiming geometry, and the likely "
    "ordered contacts; then consider sliding, rolling, and any 3D tip-offset spin deflection; "
    "finally use those physical laws to calibrate the numeric prediction or shot. Keep reasoning "
    "within 1024 tokens inside <think>...</think>, then output only the requested "
    "<answer>...</answer> block."
)


@dataclass(frozen=True)
class ForwardReward:
    event: float
    coordinate: float
    gated: float
    overrun_penalty: float

    @property
    def total(self) -> float:
        return self.event + self.gated - self.overrun_penalty


def forward_reward(
    prediction: VLMForwardOutput,
    truth: VLMForwardOutput,
    radius_m: float,
    *,
    alpha: float = 0.25,
    reasoning_tokens: int | None = None,
) -> ForwardReward:
    if not 0 <= alpha <= 1 or (reasoning_tokens is not None and reasoning_tokens < 0):
        raise ValueError("Invalid event gate or reasoning length")
    scores = forward_scores(prediction, truth, radius_m)
    left = {i for i, value in enumerate(prediction.object_collisions) if value}
    right = {i for i, value in enumerate(truth.object_collisions) if value}
    f1 = 2 * len(left & right) / (len(left) + len(right)) if left or right else 1.0
    event = 0.5 * (f1 + float(set(prediction.cushion_set) == set(truth.cushion_set)))
    distances = [
        dist(p, t) / radius_m
        if isinstance(p, tuple) and isinstance(t, tuple)
        else 0.0
        if p == t
        else 15.0
        for observed, target in zip(
            (*prediction.frames, prediction.resting_state),
            (*truth.frames, truth.resting_state),
            strict=True,
        )
        for p, t in zip(observed, target, strict=True)
    ]
    coordinate = float(scores["T3"]) + 0.5 * max(0.0, 1 - sum(distances) / len(distances) / 15)
    penalty = min(0.25, 0.25 * max(0, (reasoning_tokens or 0) - 1024) / 1024)
    return ForwardReward(event, coordinate, (alpha + (1 - alpha) * event) * coordinate, penalty)


def _shift(before: Position, after: Position, radius_m: float) -> tuple[float, float, float]:
    if isinstance(before, tuple) and isinstance(after, tuple):
        return (after[0] - before[0]) / radius_m, (after[1] - before[1]) / radius_m, 0
    # Pocket transitions are signed status changes, not invented pocket coordinates.
    return 0, 0, 2 * (int(after == "pocketed") - int(before == "pocketed"))


def twin_contrast(
    predictions: tuple[VLMForwardOutput, VLMForwardOutput],
    truths: tuple[VLMForwardOutput, VLMForwardOutput],
    radius_m: float,
) -> float:
    """Reward correctly oriented and calibrated action-induced terminal changes."""
    inventory = {len(row.resting_state) for row in (*predictions, *truths)}
    if len(inventory) != 1 or not isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Twin contrasts require matched slots and positive finite radius")
    predicted = [
        component
        for a, b in zip(*(row.resting_state for row in predictions), strict=True)
        for component in _shift(a, b, radius_m)
    ]
    actual = [
        component
        for a, b in zip(*(row.resting_state for row in truths), strict=True)
        for component in _shift(a, b, radius_m)
    ]
    p = hypot(*predicted)
    t = hypot(*actual)
    if not p or not t:
        return 0.0
    cosine = sum(a * b for a, b in zip(predicted, actual, strict=True)) / (p * t)
    return min(1.0, max(0.0, cosine)) * min(p, t) / max(p, t)


def varied_group(rewards: list[float], *, size: int = 8) -> bool:
    """DAPO accepts only a complete group with non-zero reward variance."""
    if size < 2 or len(rewards) != size or not all(isfinite(value) for value in rewards):
        raise ValueError("DAPO needs a complete, finite rollout group")
    return min(rewards) != max(rewards)


def parse_forward_reward(
    text: str,
    truth: VLMForwardOutput,
    radius_m: float,
    *,
    alpha: float = 0.25,
    reasoning_tokens: int | None = None,
) -> float:
    try:
        prediction = parse_forward_answer(
            text, ball_count=len(truth.resting_state), allow_json_fallback=False
        )
    except ValueError:
        return -1.0
    return forward_reward(
        prediction,
        truth,
        radius_m,
        alpha=alpha,
        reasoning_tokens=reasoning_tokens,
    ).total


class GoalAttempt(Record):
    ordinal: int = Field(ge=1, le=6)
    action: CueAction | None
    outcome: Layout | None
    loss_radii: float | None = Field(ge=0)
    success: bool
    failure: str | None

    @model_validator(mode="after")
    def check_attempt(self) -> Self:
        if (self.outcome is None) != (self.loss_radii is None):
            raise ValueError("A measured loss requires a simulated outcome")
        if self.success and (self.failure is not None or self.outcome is None):
            raise ValueError("Success requires a valid outcome")
        return self

    @property
    def reward(self) -> float:
        return goal_reward(self.loss_radii, self.success)


class GoalEpisode:
    """A fixed goal/table per episode, with public observations and private simulation."""

    def __init__(
        self, task: VLMGoalInput, initial: Layout, target: Layout, simulator: SnookerSimulator
    ) -> None:
        visible = public_physics(simulator, withheld_parameters=task.withheld_parameters)
        if (
            visible.parameters != task.parameters
            or visible.table.ball_radius_m != task.ball_radius_m
            or (visible.table.width_m, visible.table.length_m) != task.table_metres
            or len(initial.balls) != task.ball_count
            or len(target.balls) != task.ball_count
        ):
            raise ValueError("Episode physics and inventory must agree with the public task")
        self.task = task
        self.initial = initial
        self.target = target
        self.simulator = simulator
        self.table = visible.table
        self.attempts: list[GoalAttempt] = []

    @property
    def done(self) -> bool:
        return len(self.attempts) >= self.task.max_attempts or (
            bool(self.attempts) and self.attempts[-1].success
        )

    def step(
        self,
        text: str,
        *,
        replay_action: Callable[[CueAction], Replay] | None = None,
    ) -> GoalAttempt:
        if self.done:
            raise ValueError("Episode already terminated")
        ordinal = len(self.attempts) + 1
        try:
            action = parse_goal_answer(text, allow_json_fallback=True).cue_action()
        except ValueError:
            result = GoalAttempt(
                ordinal=ordinal,
                action=None,
                outcome=None,
                loss_radii=None,
                success=False,
                failure="invalid_submission",
            )
        else:
            try:
                replay = (
                    replay_action(action)
                    if replay_action
                    else self.simulator.replay(self.initial, action)
                )
            except SimulationFailure:
                result = GoalAttempt(
                    ordinal=ordinal,
                    action=action,
                    outcome=None,
                    loss_radii=None,
                    success=False,
                    failure="simulation_failure",
                )
            else:
                endpoint = score_endpoint(replay.goal, self.target, self.table.ball_radius_m)
                result = GoalAttempt(
                    ordinal=ordinal,
                    action=action,
                    outcome=replay.goal.endpoint_view(),
                    loss_radii=endpoint_loss(
                        replay.goal,
                        self.target,
                        ball_radius_m=self.table.ball_radius_m,
                        table_diagonal_m=hypot(*self.task.table_metres),
                    ),
                    success=endpoint.success,
                    failure=endpoint.failure,
                )
        self.attempts.append(result)
        return result

    def feedback(self, attempt: GoalAttempt) -> tuple[str, Image.Image | None]:
        if attempt not in self.attempts:
            raise ValueError("Feedback can only report an attempt executed by this episode")
        image = None
        residual = None
        if attempt.outcome is not None:
            image = render_vlm_goal_pair(self.initial, attempt.outcome, self.table)[1]
            cue = next(ball.xy_m for ball in attempt.outcome.balls if ball.kind == "cue")
            target = next(ball.xy_m for ball in self.target.balls if ball.kind == "cue")
            if cue is not None and target is not None:
                residual = [round(target[0] - cue[0], 4), round(target[1] - cue[1], 4)]
        payload = attempt.model_dump(mode="json", exclude={"outcome"}) | {
            "cue_residual_to_goal_m": residual,
            "outcome": attempt.outcome.model_dump(mode="json") if attempt.outcome else None,
        }
        return json.dumps(payload, separators=(",", ":"), allow_nan=False), image
