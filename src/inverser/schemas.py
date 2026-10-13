from collections import Counter
from math import hypot
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

BallKind = Literal["cue", "red", "yellow", "green", "brown", "blue", "pink", "black"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class BallPosition(Record):
    ball_id: str = Field(min_length=1)
    kind: BallKind
    status: Literal["on_table", "pocketed", "unobserved"] = "on_table"
    xy_m: tuple[float, float] | None = None

    @model_validator(mode="after")
    def check_position(self) -> Self:
        if (self.status == "on_table") != (self.xy_m is not None):
            raise ValueError("Only an on-table ball has a known table position")
        return self


class Layout(Record):
    balls: tuple[BallPosition, ...] = Field(min_length=1, max_length=22)

    @model_validator(mode="after")
    def check_inventory(self) -> Self:
        if len({ball.ball_id for ball in self.balls}) != len(self.balls):
            raise ValueError("Ball IDs must be unique within a layout")
        counts = Counter(ball.kind for ball in self.balls)
        if counts["cue"] != 1 or counts["red"] > 15:
            raise ValueError("A snooker layout requires one cue ball and at most 15 reds")
        if any(count > 1 for kind, count in counts.items() if kind not in ("cue", "red")):
            raise ValueError("Coloured snooker balls must be unique")
        return self

    def endpoint_view(self) -> Self:
        """Give reds endpoint-local IDs rather than historical tracking identities."""
        reds = sorted(
            (ball for ball in self.balls if ball.kind == "red"),
            key=lambda ball: (ball.status, ball.xy_m or (0.0, 0.0)),
        )
        other = tuple(ball for ball in self.balls if ball.kind != "red")
        return self.model_copy(
            update={
                "balls": other
                + tuple(
                    ball.model_copy(update={"ball_id": f"red_{index}"})
                    for index, ball in enumerate(reds)
                )
            }
        )


class CueAction(Record):
    azimuth_deg: float = Field(ge=0, lt=360)
    cue_speed_mps: float = Field(gt=0, le=3)
    tip_side: float = Field(ge=-0.5, le=0.5)
    tip_vertical: float = Field(ge=-0.5, le=0.5)
    elevation_deg: float = Field(default=0.0, ge=0, le=0)

    @model_validator(mode="after")
    def check_contact_disk(self) -> Self:
        if hypot(self.tip_side, self.tip_vertical) > 0.5 + 1e-12:
            raise ValueError("Pilot cue contact must lie within radius 0.5")
        return self


class PhysicsConfig(Record):
    simulator: Literal["pooltool-billiards"] = "pooltool-billiards"
    simulator_version: Literal["0.6.0"] = "0.6.0"
    resolver: Literal["pooltool-0.6.0-default-v9"] = "pooltool-0.6.0-default-v9"
    table_width_m: float = Field(default=1.778, gt=0)
    table_length_m: float = Field(default=3.569, gt=0)
    ball_radius_m: float = Field(default=0.02619375, gt=0)
    ball_mass_kg: float = Field(default=0.140, gt=0)
    sliding_friction: float = Field(default=0.5, gt=0)
    rolling_friction: float = Field(default=0.01, gt=0)
    spin_friction_proportionality: float = Field(default=10 * 2 / 5 / 9, gt=0)
    ball_friction: float = Field(default=0.05, gt=0)
    ball_restitution: float = Field(default=0.95, gt=0, le=1)
    cushion_restitution: float = Field(default=0.85, gt=0, le=1)
    cushion_friction: float = Field(default=0.5, gt=0)
    gravity_mps2: float = Field(default=9.81, gt=0)
    max_simulated_seconds: float = Field(default=120, gt=0)
    max_events: int = Field(default=1000, ge=10)


class TaskInput(Record):
    task_id: str
    initial: Layout
    goal: Layout
    physics: PhysicsConfig
    tolerance_radii: float = Field(default=1.0, gt=0)
    red_identity: Literal["endpoint_local_exchangeable"] = "endpoint_local_exchangeable"
    no_scratch: bool = True


class ContactEvent(Record):
    time_s: float = Field(ge=0)
    event_type: str
    agents: tuple[str, ...]


class MotionSample(Record):
    ball_id: str
    time_s: float = Field(ge=0)
    motion_state: int
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    angular_velocity_radps: tuple[float, float, float]


class ShotRecord(Record):
    schema_version: Literal["pilot-v0"] = "pilot-v0"
    task_id: str
    group_id: str
    source_kind: Literal["simulated"] = "simulated"
    label_tier: Literal["simulated_truth"] = "simulated_truth"
    game: Literal["snooker"] = "snooker"
    partition: Literal["development"] = "development"
    family: Literal["free_roll", "center", "vertical", "side"]
    physics: PhysicsConfig
    initial: Layout
    goal: Layout
    witness_action: CueAction
    duration_s: float = Field(ge=0)
    events: tuple[ContactEvent, ...]
    event_sampled_motion: tuple[MotionSample, ...]
    cue_initial_velocity_mps: tuple[float, float, float]
    cue_initial_angular_velocity_radps: tuple[float, float, float]

    def public_input(self) -> TaskInput:
        return TaskInput(
            task_id=self.task_id,
            initial=self.initial.endpoint_view(),
            goal=self.goal.endpoint_view(),
            physics=self.physics,
        )
