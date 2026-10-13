from dataclasses import replace

import pytest

from inverser.processing_audit import cue_displacement, parse_frame
from inverser.vision import Detection, FrameObservation


def frame(t, x):
    return FrameObservation(
        t,
        (Detection("cue", (x, 1), (10, 20), 1, 1, "observed_cue"),),
        0.99,
        0,
        0,
        True,
        True,
        0,
        0,
        False,
        False,
        0,
        1,
    )


def test_roundtrip_preserves_only_observed_coordinates():
    first = parse_frame(frame(0, 0.5).json())
    last = parse_frame(frame(2, 0.8).json())
    assert cue_displacement(first, last) == pytest.approx(0.3)
    assert not hasattr(first.detections[0], "angular_velocity")


def test_missing_cue_is_unknown_not_success_or_pocketed():
    first = frame(0, 0.5)
    last = replace(frame(2, 0.8), detections=())
    assert cue_displacement(first, last) is None


def test_nonfinite_observed_coordinates_fail_validation():
    data = frame(0, float("nan")).json()
    with pytest.raises(ValueError):
        parse_frame(data)
