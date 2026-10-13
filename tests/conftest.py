import pytest

from inverser.physics import SnookerSimulator
from inverser.schemas import BallPosition, CueAction, Layout, PhysicsConfig, ShotRecord
from inverser.study import make_study_case


@pytest.fixture
def study_fixture():
    simulator = SnookerSimulator(PhysicsConfig())
    initial = Layout(balls=(BallPosition(ball_id="cue", kind="cue", xy_m=(0.6, 0.8)),))
    action = CueAction(azimuth_deg=90, cue_speed_mps=0.3, tip_side=0, tip_vertical=0)
    replay = simulator.replay(initial, action)
    record = ShotRecord(
        task_id="fixture-task",
        group_id="fixture-layout",
        family="free_roll",
        physics=simulator.config,
        initial=initial,
        goal=replay.goal,
        witness_action=action,
        duration_s=replay.duration_s,
        events=replay.events,
        event_sampled_motion=replay.motion,
        cue_initial_velocity_mps=replay.cue_velocity,
        cue_initial_angular_velocity_radps=replay.cue_omega,
    )
    return simulator, record, make_study_case(record, simulator)
