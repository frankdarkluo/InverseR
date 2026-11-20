from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

from inverser.runtime import configure_physics_runtime
from inverser.schemas import (
    BallPosition,
    ContactEvent,
    CueAction,
    Layout,
    MotionSample,
    PhysicsConfig,
)

if TYPE_CHECKING:
    import numpy as np
    import pooltool as pt
    from numpy.typing import NDArray


class SimulationFailure(RuntimeError):
    pass


def _vector3(row: "NDArray[np.float64]") -> tuple[float, float, float]:
    return float(row[0]), float(row[1]), float(row[2])


@dataclass(frozen=True)
class Replay:
    goal: Layout
    duration_s: float
    events: tuple[ContactEvent, ...]
    motion: tuple[MotionSample, ...]
    cue_velocity: tuple[float, float, float]
    cue_omega: tuple[float, float, float]


def sample_event_motion(
    motion: tuple[MotionSample, ...], config: PhysicsConfig, sample_times_s: tuple[float, ...]
) -> tuple[MotionSample, ...]:
    if not sample_times_s or any(not isfinite(time) or time < 0 for time in sample_times_s):
        raise ValueError("Sampling times must be finite and nonnegative")
    if any(a >= b for a, b in zip(sample_times_s, sample_times_s[1:])):
        raise ValueError("Sampling times must strictly increase")
    if not motion:
        raise ValueError("Cannot sample an empty event history")
    configure_physics_runtime()
    import numpy as np
    from pooltool.physics.evolve import evolve_ball_motion

    histories: dict[str, list[MotionSample]] = {}
    for point in motion:
        histories.setdefault(point.ball_id, []).append(point)
    sampled = []
    for ball_id, history in sorted(histories.items()):
        if history[0].time_s != 0 or any(a.time_s > b.time_s for a, b in zip(history, history[1:])):
            raise ValueError("Event history must start at zero with nondecreasing times")
        if history[-1].motion_state not in (0, 4):
            raise ValueError("Ground-truth sampling requires a complete resting/pocketed history")
        times = [point.time_s for point in history]
        for time in sample_times_s:
            # Duplicate collision timestamps contain pre/post states; use the right limit.
            point = history[bisect_right(times, time) - 1]
            rvw = np.asarray(
                (point.position_m, point.velocity_mps, point.angular_velocity_radps),
                dtype=np.float64,
            )
            state, mode = evolve_ball_motion(
                point.motion_state,
                rvw,
                config.ball_radius_m,
                config.ball_mass_kg,
                config.sliding_friction,
                config.spin_friction_proportionality * config.ball_radius_m,
                config.rolling_friction,
                config.gravity_mps2,
                time - point.time_s,
            )
            if not np.isfinite(state).all():
                raise SimulationFailure("Physical sampling produced non-finite state")
            sampled.append(
                MotionSample(
                    ball_id=ball_id,
                    time_s=time,
                    motion_state=int(mode),
                    position_m=_vector3(state[0]),
                    velocity_mps=_vector3(state[1]),
                    angular_velocity_radps=_vector3(state[2]),
                )
            )
    return tuple(sampled)


class SnookerSimulator:
    def __init__(self, config: PhysicsConfig) -> None:
        configure_physics_runtime()
        import attrs
        import pooltool as pt
        from pooltool.objects.table.collection import default_specs_from_game_type
        from pooltool.objects.table.specs import SnookerTableSpecs
        from pooltool.physics import PhysicsEngine
        from pooltool.physics.resolve.resolver import default_resolver

        if pt.__version__ != config.simulator_version:
            raise RuntimeError("Installed simulator version does not match the task contract")
        self.config = config
        self.pt = pt
        preset = default_specs_from_game_type(pt.GameType.SNOOKER)
        if not isinstance(preset, SnookerTableSpecs):
            raise RuntimeError("Expected a snooker-specific table preset")
        self.specs = attrs.evolve(
            preset,
            w=config.table_width_m,
            l=config.table_length_m,
        )
        self.engine = PhysicsEngine(resolver=default_resolver())
        self.ball_parameters = {
            "m": config.ball_mass_kg,
            "R": config.ball_radius_m,
            "u_s": config.sliding_friction,
            "u_r": config.rolling_friction,
            "u_sp_proportionality": config.spin_friction_proportionality,
            "u_b": config.ball_friction,
            "e_b": config.ball_restitution,
            "e_c": config.cushion_restitution,
            "f_c": config.cushion_friction,
            "g": config.gravity_mps2,
        }
        self.ball_params = pt.BallParams(**self.ball_parameters)
        self.rollouts = 0

    def metadata(self) -> dict[str, object]:
        import attrs

        return {
            "config": self.config.model_dump(mode="json"),
            "table_specs": attrs.asdict(self.specs),
            "cue_specs": attrs.asdict(self.pt.Cue().specs),
            "resolver_classes": {
                field.name: type(getattr(self.engine.resolver, field.name)).__name__
                for field in attrs.fields(type(self.engine.resolver))
            },
            "resolver_parameters": attrs.asdict(self.engine.resolver),
            "ball_parameters": self.ball_parameters,
        }

    def _system(self, initial: Layout, action: CueAction) -> "pt.System":
        from math import dist

        radius = self.config.ball_radius_m
        positions: list[tuple[float, float]] = []
        cue_id = ""
        balls = {}
        for ball in initial.balls:
            if ball.status != "on_table" or ball.xy_m is None:
                raise ValueError("Pilot initial states must be fully observed and on-table")
            x, y = ball.xy_m
            if not (
                radius <= x <= self.config.table_width_m - radius
                and radius <= y <= self.config.table_length_m - radius
            ):
                raise ValueError("Initial ball lies outside the valid table-center bounds")
            if any(dist(ball.xy_m, other) < 2 * radius for other in positions):
                raise ValueError("Initial balls overlap")
            positions.append(ball.xy_m)
            physical_ball = self.pt.Ball(id=ball.ball_id, params=self.ball_params)
            physical_ball.state.rvw[0] = [x, y, radius]
            balls[ball.ball_id] = physical_ball
            if ball.kind == "cue":
                cue_id = ball.ball_id
        return self.pt.System(
            cue=self.pt.Cue(
                V0=action.cue_speed_mps,
                phi=action.azimuth_deg,
                theta=action.elevation_deg,
                a=action.tip_side,
                b=action.tip_vertical,
                cue_ball_id=cue_id,
            ),
            table=self.pt.Table.from_table_specs(self.specs),
            balls=balls,
        )

    def replay(self, initial: Layout, action: CueAction) -> Replay:
        import numpy as np

        shot = self._system(initial, action)
        self.rollouts += 1
        result = self.pt.simulate(
            shot,
            engine=self.engine,
            max_events=self.config.max_events,
            t_final=self.config.max_simulated_seconds,
        )
        # Pooltool sets balls stationary at its event cap; that is not a valid endpoint.
        if len(result.events) >= self.config.max_events:
            raise SimulationFailure("Simulation reached the event cap")
        if result.t >= self.config.max_simulated_seconds or not np.isfinite(result.t):
            raise SimulationFailure("Simulation exceeded its finite duration limit")
        goal = []
        motion = []
        for observed in initial.balls:
            ball = result.balls[observed.ball_id]
            if not np.isfinite(ball.state.rvw).all():
                raise SimulationFailure("Simulation produced non-finite ball state")
            if ball.state.s not in (self.pt.constants.stationary, self.pt.constants.pocketed):
                raise SimulationFailure("Simulation stopped with a moving ball")
            pocketed = ball.state.s == self.pt.constants.pocketed
            x, y = map(float, ball.state.rvw[0, :2])
            if not pocketed and not (
                0 <= x <= self.config.table_width_m and 0 <= y <= self.config.table_length_m
            ):
                raise SimulationFailure("Non-pocketed terminal ball is outside the table")
            goal.append(
                BallPosition(
                    ball_id=observed.ball_id,
                    kind=observed.kind,
                    status="pocketed" if pocketed else "on_table",
                    xy_m=None if pocketed else (x, y),
                )
            )
            for state in ball.history:
                motion.append(
                    MotionSample(
                        ball_id=ball.id,
                        time_s=float(state.t),
                        motion_state=int(state.s),
                        position_m=_vector3(state.rvw[0]),
                        velocity_mps=_vector3(state.rvw[1]),
                        angular_velocity_radps=_vector3(state.rvw[2]),
                    )
                )
        cue_history = result.balls[result.cue.cue_ball_id].history
        post_impact = next(
            (state for state in cue_history if state.t == 0 and np.linalg.norm(state.rvw[1]) > 0),
            None,
        )
        if post_impact is None:
            raise SimulationFailure("No measured simulator post-impact cue-ball state")
        return Replay(
            goal=Layout(balls=tuple(goal)),
            duration_s=float(result.t),
            events=tuple(
                ContactEvent(
                    time_s=float(event.time),
                    event_type=event.event_type.value,
                    agents=event.ids,
                )
                for event in result.events
            ),
            motion=tuple(motion),
            cue_velocity=_vector3(post_impact.rvw[1]),
            cue_omega=_vector3(post_impact.rvw[2]),
        )
