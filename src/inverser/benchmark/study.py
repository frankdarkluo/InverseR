from typing import Literal, Self

from pydantic import Field, model_validator

from inverser.physics import Replay, SnookerSimulator, sample_event_motion
from inverser.runtime import fingerprint
from inverser.schemas import (
    BallPosition,
    ContactEvent,
    CueAction,
    Layout,
    MotionSample,
    PhysicsConfig,
    Record,
    ShotRecord,
)

DYNAMIC_PARAMETERS = (
    "ball_mass_kg",
    "sliding_friction",
    "rolling_friction",
    "spin_friction_proportionality",
    "ball_friction",
    "ball_restitution",
    "cushion_restitution",
    "cushion_friction",
    "gravity_mps2",
)


class CircleGeometry(Record):
    geometry_id: str
    xy_m: tuple[float, float]
    radius_m: float = Field(gt=0)


class LinearGeometry(Record):
    geometry_id: str
    start_m: tuple[float, float]
    end_m: tuple[float, float]


class TableGeometry(Record):
    width_m: float = Field(gt=0)
    length_m: float = Field(gt=0)
    ball_radius_m: float = Field(gt=0)
    pockets: tuple[CircleGeometry, ...] = Field(min_length=6, max_length=6)
    linear_cushions: tuple[LinearGeometry, ...] = Field(min_length=1)
    circular_cushions: tuple[CircleGeometry, ...] = Field(min_length=1)
    coordinate_convention: Literal["x_right_y_up_metres"] = "x_right_y_up_metres"


class CueSpecifications(Record):
    mass_kg: float = Field(gt=0)
    length_m: float = Field(gt=0)
    tip_radius_m: float = Field(gt=0)
    shaft_radius_at_tip_m: float = Field(gt=0)
    shaft_radius_at_butt_m: float = Field(gt=0)
    end_mass_kg: float = Field(gt=0)
    impact_model: str


class PublicPhysics(Record):
    simulator: Literal["pooltool-billiards"] = "pooltool-billiards"
    simulator_version: Literal["0.6.0"] = "0.6.0"
    table: TableGeometry
    cue: CueSpecifications
    parameters: dict[str, float]
    withheld_parameters: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        visible, hidden = set(self.parameters), set(self.withheld_parameters)
        if len(hidden) != len(self.withheld_parameters):
            raise ValueError("Withheld parameter names must be unique")
        if visible & hidden or visible | hidden != set(DYNAMIC_PARAMETERS):
            raise ValueError("Dynamics parameters must be partitioned into visible and withheld")
        if any(value <= 0 for value in self.parameters.values()):
            raise ValueError("Visible dynamics parameters must be positive")
        return self


class ForwardInput(Record):
    schema_version: Literal["study-forward-v1"] = "study-forward-v1"
    case_id: str
    source_kind: Literal["simulated"] = "simulated"
    initial: Layout
    physics: PublicPhysics
    action: CueAction
    sample_times_s: tuple[float, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_times(self) -> Self:
        if any(time < 0 for time in self.sample_times_s) or any(
            a >= b for a, b in zip(self.sample_times_s, self.sample_times_s[1:])
        ):
            raise ValueError("Sampling times must be nonnegative and strictly increasing")
        return self


class GoalInput(Record):
    schema_version: Literal["study-goal-v1"] = "study-goal-v1"
    case_id: str
    source_kind: Literal["simulated"] = "simulated"
    initial: Layout
    goal: Layout
    physics: PublicPhysics
    tolerance_radii: float = Field(default=1.0, gt=0)
    no_scratch: bool = True
    red_identity: Literal["endpoint_local_exchangeable"] = "endpoint_local_exchangeable"


class SubmittedAction(CueAction):
    elevation_deg: float = Field(ge=0, le=0)


class GoalSubmission(Record):
    action: SubmittedAction


class PredictedFrame(Record):
    time_s: float = Field(ge=0)
    state: Layout


class SemanticContact(Record):
    event_type: Literal["ball_ball", "cushion", "pocket"]
    time_s: float = Field(ge=0)
    participants: tuple[str, str]


class ForwardSubmission(Record):
    frames: tuple[PredictedFrame, ...] = Field(min_length=1, max_length=8)
    resting_state: Layout
    events: tuple[SemanticContact, ...]

    @model_validator(mode="after")
    def validate_times(self) -> Self:
        if any(a.time_s >= b.time_s for a, b in zip(self.frames, self.frames[1:])):
            raise ValueError("Predicted frame times must strictly increase")
        return self


class CandidateAction(Record):
    candidate_id: str = Field(min_length=1)
    action: CueAction


class ActionBank(Record):
    schema_version: Literal["study-action-bank-v1"] = "study-action-bank-v1"
    case_id: str
    seed: int = Field(ge=0)
    sampling_prior: Literal["sobol_uniform_azimuth_speed_contact_disk"] = (
        "sobol_uniform_azimuth_speed_contact_disk"
    )
    min_cue_speed_mps: float = Field(gt=0)
    max_cue_speed_mps: float = Field(gt=0, le=3)
    candidates: tuple[CandidateAction, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_bank(self) -> Self:
        if self.min_cue_speed_mps >= self.max_cue_speed_mps:
            raise ValueError("Bank speed bounds must form a nonempty interval")
        if len({candidate.candidate_id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("Candidate IDs must be unique")
        if any(
            not self.min_cue_speed_mps <= candidate.action.cue_speed_mps <= self.max_cue_speed_mps
            for candidate in self.candidates
        ):
            raise ValueError("Candidate action violates the declared bank speed prior")
        return self


def _initial_slots(initial: Layout) -> dict[str, str]:
    public = initial.endpoint_view()
    return {
        ball.ball_id: next(
            slot.ball_id
            for slot in public.balls
            if (slot.kind, slot.status, slot.xy_m) == (ball.kind, ball.status, ball.xy_m)
        )
        for ball in initial.balls
    }


def semantic_contacts(
    initial: Layout, events: tuple[ContactEvent, ...]
) -> tuple[SemanticContact, ...]:
    types = {
        "ball_ball": "ball_ball",
        "ball_linear_cushion": "cushion",
        "ball_circular_cushion": "cushion",
        "ball_pocket": "pocket",
    }
    excluded = {
        "none",
        "stick_ball",
        "spinning_stationary",
        "rolling_stationary",
        "rolling_spinning",
        "sliding_rolling",
    }
    slots = _initial_slots(initial)
    contacts = []
    for event in events:
        if event.event_type in excluded:
            continue
        if event.event_type not in types or len(event.agents) != 2:
            raise ValueError("Unsupported raw event cannot be silently treated as contact GT")
        first, second = event.agents
        if first not in slots:
            raise ValueError("Contact event has no initial ball identity")
        if event.event_type == "ball_ball" and second not in slots:
            raise ValueError("Ball contact has an unknown second ball")
        participants = (
            (slots[first], slots[second])
            if event.event_type == "ball_ball"
            else (slots[first], second)
        )
        contacts.append(
            SemanticContact.model_validate(
                {
                    "event_type": types[event.event_type],
                    "time_s": event.time_s,
                    "participants": participants,
                }
            )
        )
    return tuple(contacts)


def replay_forward_truth(
    initial: Layout, replay: Replay, config: PhysicsConfig, sample_times_s: tuple[float, ...]
) -> ForwardSubmission:
    slots = _initial_slots(initial)
    kinds = {slots[ball.ball_id]: ball.kind for ball in initial.balls}
    samples = sample_event_motion(replay.motion, config, sample_times_s)
    return ForwardSubmission(
        frames=tuple(
            PredictedFrame(
                time_s=time,
                state=Layout(
                    balls=tuple(
                        BallPosition(
                            ball_id=slots[point.ball_id],
                            kind=kinds[slots[point.ball_id]],
                            status="pocketed" if point.motion_state == 4 else "on_table",
                            xy_m=None if point.motion_state == 4 else point.position_m[:2],
                        )
                        for point in samples
                        if point.time_s == time
                    )
                ),
            )
            for time in sample_times_s
        ),
        resting_state=Layout(
            balls=tuple(
                ball.model_copy(update={"ball_id": slots[ball.ball_id]})
                for ball in replay.goal.balls
            )
        ),
        events=semantic_contacts(initial, replay.events),
    )


def translating_ball_ids(initial: Layout, motion: tuple[MotionSample, ...]) -> tuple[str, ...]:
    slots = _initial_slots(initial)
    return tuple(
        sorted(
            {
                slots[point.ball_id]
                for point in motion
                if any(abs(component) > 1e-9 for component in point.velocity_mps[:2])
                or point.motion_state == 4
            }
        )
    )


class StudyCase(Record):
    schema_version: Literal["study-case-v1"] = "study-case-v1"
    partition: Literal["development"] = "development"
    case_id: str
    public_physics: PublicPhysics
    private_record: ShotRecord

    def forward_input(self, action: CueAction, sample_times_s: tuple[float, ...]) -> ForwardInput:
        return ForwardInput(
            case_id=self.case_id,
            initial=self.private_record.initial.endpoint_view(),
            physics=self.public_physics,
            action=action,
            sample_times_s=sample_times_s,
        )

    def goal_input(self) -> GoalInput:
        return GoalInput(
            case_id=self.case_id,
            initial=self.private_record.initial.endpoint_view(),
            goal=self.private_record.goal.endpoint_view(),
            physics=self.public_physics,
        )

    def sampled_truth(self, sample_times_s: tuple[float, ...]) -> tuple[MotionSample, ...]:
        motion = sample_event_motion(
            self.private_record.event_sampled_motion,
            self.private_record.physics,
            sample_times_s,
        )
        slots = _initial_slots(self.private_record.initial)
        return tuple(point.model_copy(update={"ball_id": slots[point.ball_id]}) for point in motion)

    def forward_truth(self, sample_times_s: tuple[float, ...]) -> ForwardSubmission:
        record = self.private_record
        replay = Replay(
            record.goal,
            record.duration_s,
            record.events,
            record.event_sampled_motion,
            record.cue_initial_velocity_mps,
            record.cue_initial_angular_velocity_radps,
        )
        return replay_forward_truth(record.initial, replay, record.physics, sample_times_s)

    def moving_ball_ids(self) -> tuple[str, ...]:
        return translating_ball_ids(
            self.private_record.initial, self.private_record.event_sampled_motion
        )

    def semantic_contacts(self) -> tuple[SemanticContact, ...]:
        return semantic_contacts(self.private_record.initial, self.private_record.events)


def public_physics(
    simulator: SnookerSimulator, *, withheld_parameters: tuple[str, ...] = ()
) -> PublicPhysics:
    if set(withheld_parameters) - set(DYNAMIC_PARAMETERS):
        raise ValueError("Unknown withheld dynamics parameter")
    table = simulator.pt.Table.from_table_specs(simulator.specs)
    cue = simulator.pt.Cue().specs
    config = simulator.config
    return PublicPhysics(
        table=TableGeometry(
            width_m=config.table_width_m,
            length_m=config.table_length_m,
            ball_radius_m=config.ball_radius_m,
            pockets=tuple(
                CircleGeometry(
                    geometry_id=key,
                    xy_m=(float(pocket.center[0]), float(pocket.center[1])),
                    radius_m=float(pocket.radius),
                )
                for key, pocket in sorted(table.pockets.items())
            ),
            linear_cushions=tuple(
                LinearGeometry(
                    geometry_id=key,
                    start_m=(float(cushion.p1[0]), float(cushion.p1[1])),
                    end_m=(float(cushion.p2[0]), float(cushion.p2[1])),
                )
                for key, cushion in sorted(table.cushion_segments.linear.items())
            ),
            circular_cushions=tuple(
                CircleGeometry(
                    geometry_id=key,
                    xy_m=(float(cushion.center[0]), float(cushion.center[1])),
                    radius_m=float(cushion.radius),
                )
                for key, cushion in sorted(table.cushion_segments.circular.items())
            ),
        ),
        cue=CueSpecifications(
            mass_kg=cue.M,
            length_m=cue.length,
            tip_radius_m=cue.tip_radius,
            shaft_radius_at_tip_m=cue.shaft_radius_at_tip,
            shaft_radius_at_butt_m=cue.shaft_radius_at_butt,
            end_mass_kg=cue.end_mass,
            impact_model=type(simulator.engine.resolver.stick_ball).__name__,
        ),
        parameters={
            name: float(getattr(config, name))
            for name in DYNAMIC_PARAMETERS
            if name not in withheld_parameters
        },
        withheld_parameters=withheld_parameters,
    )


def make_study_case(
    record: ShotRecord, simulator: SnookerSimulator, *, withheld_parameters: tuple[str, ...] = ()
) -> StudyCase:
    if record.physics != simulator.config:
        raise ValueError("Study metadata and record physics disagree")
    return StudyCase(
        case_id="case-" + fingerprint((record.task_id, record.group_id))[:20],
        public_physics=public_physics(simulator, withheld_parameters=withheld_parameters),
        private_record=record,
    )
