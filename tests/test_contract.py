import itertools
import math

import pytest
from pydantic import ValidationError

from inverser.metrics import bottleneck_distance, score_endpoint
from inverser.schemas import BallPosition, CueAction, Layout


def layout(reds=((0.4, 0.6), (0.8, 1.2))):
    return Layout(
        balls=(BallPosition(ball_id="cue", kind="cue", xy_m=(0.2, 0.3)),)
        + tuple(BallPosition(ball_id=f"red_{i}", kind="red", xy_m=xy) for i, xy in enumerate(reds))
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"azimuth_deg": 360},
        {"cue_speed_mps": 0},
        {"cue_speed_mps": float("nan")},
        {"tip_side": 0.5, "tip_vertical": 0.5},
        {"elevation_deg": 10},
        {"force_newtons": 3},
    ],
)
def test_invalid_actions_are_rejected(changes):
    values = {"azimuth_deg": 90, "cue_speed_mps": 1, "tip_side": 0, "tip_vertical": 0}
    with pytest.raises(ValidationError):
        CueAction(**(values | changes))


def test_red_identity_is_exchangeable():
    assert score_endpoint(layout(), layout(((0.8, 1.2), (0.4, 0.6))), 1e-8).success


def test_endpoint_ids_do_not_reveal_track_identity():
    assert layout().endpoint_view() == layout(((0.8, 1.2), (0.4, 0.6))).endpoint_view()


def test_pocketed_is_not_unobserved():
    unknown = layout().model_copy(
        update={"balls": (BallPosition(ball_id="cue", kind="cue", status="unobserved"),)}
    )
    with pytest.raises(ValueError, match="Unobserved"):
        score_endpoint(unknown, layout(), 0.1)


def test_maximum_error_is_not_diluted_by_stationary_balls():
    result = score_endpoint(layout(((0.4, 0.6), (0.8, 1.4))), layout(), 0.1)
    assert not result.success
    assert result.max_position_error_m == pytest.approx(0.2)


def test_pocket_status_mismatch_fails():
    target = layout()
    changed = target.model_copy(
        update={
            "balls": target.balls[:2]
            + (BallPosition(ball_id="red_1", kind="red", status="pocketed"),)
        }
    )
    assert not score_endpoint(changed, target, 100).success


def test_red_matching_minimizes_maximum_not_average_error():
    predicted = [(0.0, 0.0), (0.2, 0.8), (0.8, 0.1), (0.5, 0.3)]
    target = [(0.9, 0.9), (0.7, 0.4), (0.1, 0.7), (0.2, 0.1)]
    expected = min(
        max(math.dist(a, b) for a, b in zip(predicted, permutation, strict=True))
        for permutation in itertools.permutations(target)
    )
    assert bottleneck_distance(predicted, target) == pytest.approx(expected)
