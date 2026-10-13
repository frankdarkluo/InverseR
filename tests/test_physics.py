import math
from types import SimpleNamespace

import pytest

from inverser.physics import SimulationFailure, SnookerSimulator
from inverser.schemas import BallPosition, CueAction, Layout, PhysicsConfig


@pytest.fixture
def simulator():
    return SnookerSimulator(PhysicsConfig())


def test_cue_speed_is_not_ball_speed_and_replay_is_deterministic(simulator):
    initial = Layout(balls=(BallPosition(ball_id="cue", kind="cue", xy_m=(0.6, 0.8)),))
    action = CueAction(azimuth_deg=90, cue_speed_mps=0.3, tip_side=0, tip_vertical=0)
    first = simulator.replay(initial, action)
    second = simulator.replay(initial, action)
    assert first == second
    assert math.hypot(*first.cue_velocity[:2]) != pytest.approx(action.cue_speed_mps)
    assert first.cue_omega == (0, 0, 0)
    assert simulator.rollouts == 2
    assert initial.balls[0].xy_m == (0.6, 0.8)


def test_overlap_is_rejected_before_a_rollout(simulator):
    initial = Layout(
        balls=(
            BallPosition(ball_id="cue", kind="cue", xy_m=(0.6, 0.8)),
            BallPosition(ball_id="red", kind="red", xy_m=(0.6, 0.8)),
        )
    )
    with pytest.raises(ValueError, match="overlap"):
        simulator.replay(
            initial, CueAction(azimuth_deg=0, cue_speed_mps=0.5, tip_side=0, tip_vertical=0)
        )
    assert simulator.rollouts == 0


def test_center_hit_matches_analytical_sliding_then_rolling(simulator):
    initial = Layout(balls=(BallPosition(ball_id="cue", kind="cue", xy_m=(0.6, 0.8)),))
    result = simulator.replay(
        initial, CueAction(azimuth_deg=90, cue_speed_mps=0.3, tip_side=0, tip_vertical=0)
    )
    speed = math.hypot(*result.cue_velocity[:2])
    config = simulator.config
    sliding_time = 2 * speed / (7 * config.sliding_friction * config.gravity_mps2)
    rolling_speed = speed * 5 / 7
    distance = (
        speed * sliding_time
        - 0.5 * config.sliding_friction * config.gravity_mps2 * sliding_time**2
        + rolling_speed**2 / (2 * config.rolling_friction * config.gravity_mps2)
    )
    assert result.goal.balls[0].xy_m == pytest.approx((0.6, 0.8 + distance), abs=1e-8)


def test_event_cap_cannot_be_mistaken_for_a_resting_goal(simulator, monkeypatch):
    monkeypatch.setattr(
        simulator.pt,
        "simulate",
        lambda *args, **kwargs: SimpleNamespace(events=[None] * simulator.config.max_events, t=0),
    )
    initial = Layout(balls=(BallPosition(ball_id="cue", kind="cue", xy_m=(0.6, 0.8)),))
    with pytest.raises(SimulationFailure, match="event cap"):
        simulator.replay(
            initial, CueAction(azimuth_deg=90, cue_speed_mps=0.3, tip_side=0, tip_vertical=0)
        )
