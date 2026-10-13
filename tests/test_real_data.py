from dataclasses import replace

import cv2
import numpy as np
import pytest

from inverser.real_data import (
    EpisodeMedia,
    SnookerEpisode,
    SourceProvenance,
    calibration_record,
    export_episode,
    make_episode,
)
from inverser.table_detection import locate_table
from inverser.vision import Detection, FrameObservation, TableProjection


def frame(time, x):
    return FrameObservation(
        time,
        (Detection("cue", (x, 1), (100, 200), 1, 1, "track_a"),),
        0.95,
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


def episode():
    projection = TableProjection([[0, 0], [511, 0], [511, 1027], [0, 1027]], 512, 1028)
    return make_episode(
        SourceProvenance(
            collection="hf_snooker_videos",
            recording_id="video",
            original_file="video.mp4",
            original_sha256="a" * 64,
            group_id="video",
            media_license="unverified",
        ),
        calibration_record(projection, "supplied_corners", "nominal_regulation"),
        (frame(0, 0.5), frame(0.2, 0.7), frame(0.4, 0.8)),
        0.2,
        5,
        EpisodeMedia(
            initial_source="initial.jpg",
            terminal_source="terminal.jpg",
            initial_bev="initial_bev.jpg",
            terminal_bev="terminal_bev.jpg",
            clean_table_pair="table_pair.png",
            source_clip="clip.mp4",
            source_clip_start_s=0,
            source_clip_end_s=0.4,
        ),
    )


def test_common_record_roundtrip_and_no_invented_action():
    record = episode()
    assert SnookerEpisode.model_validate_json(record.model_dump_json()) == record
    assert record.cue_control_ground_truth is None
    assert record.quality.benchmark_accepted is False
    assert record.trajectories[0].interpolated_points == 0
    assert all(not event.verified for event in record.estimated_events)


def test_endpoint_export_does_not_leak_interior_identity(tmp_path):
    import json

    record = episode()
    export_episode(tmp_path / "structured.json", record)
    endpoints = json.loads((tmp_path / "endpoints.json").read_text())
    assert "trajectories" not in endpoints and "estimated_motion_onset_s" not in endpoints
    assert "track_id" not in endpoints["initial"]["balls"][0]


def test_missing_track_is_not_automatically_pocketed():
    original = episode()
    middle = replace(frame(0.2, 0.7), detections=())
    record = make_episode(
        original.source,
        original.calibration,
        (frame(0, 0.5), middle, frame(0.4, 0.8)),
        0.2,
        5,
        original.media,
    )
    assert record.trajectories[0].missing_intervals == 1
    assert not any(event.kind == "observation_disappearance" for event in record.estimated_events)


def test_table_detector_can_find_horizontal_overhead_and_reject_no_table():
    image = np.zeros((480, 640, 3), np.uint8)
    cv2.rectangle(image, (35, 75), (605, 405), (30, 120, 40), -1)
    location = locate_table(image)
    assert location is not None and location.long_axis_horizontal
    assert len(location.points) == 4
    assert locate_table(np.zeros_like(image)) is None
    projection = TableProjection(
        [list(point) for point in location.points],
        640,
        480,
        corner_order="ordered_short_edge",
    )
    assert projection.corner_order == "ordered_short_edge"
    with pytest.raises(ValueError, match="outside"):
        TableProjection(
            [[-5, 0], [-5, 400], [500, 400], [500, 0]],
            640,
            480,
            corner_order="ordered_short_edge",
        )


def test_nonfinite_real_positions_are_rejected():
    record = episode().model_dump(mode="json")
    record["initial"]["balls"][0]["xy_m"][0] = float("nan")
    with pytest.raises(ValueError):
        SnookerEpisode.model_validate(record)


def test_perspective_trapezoid_is_not_rotated_like_a_horizontal_overhead_view():
    image = np.zeros((600, 960, 3), np.uint8)
    points = np.asarray([[345, 185], [655, 195], [835, 443], [120, 450]], np.int32)
    cv2.fillConvexPoly(image, points, (30, 120, 40))
    location = locate_table(image)
    assert location is not None
    assert not location.long_axis_horizontal
    assert location.points[0][1] < location.points[2][1]


def test_clipped_table_quad_is_not_claimed_as_complete_automatic_calibration():
    image = np.zeros((480, 640, 3), np.uint8)
    cv2.rectangle(image, (0, 0), (639, 420), (30, 120, 40), -1)
    assert locate_table(image) is None


def test_unexpected_blackball_colour_hypotheses_are_flagged():
    source = episode().source.model_copy(update={"collection": "billiard_dataset"})
    first = frame(0, 0.5)
    second = frame(0.4, 0.8)
    blue = Detection("blue", (1, 1), (200, 200), 1, 1, "blue")
    result = make_episode(
        source,
        episode().calibration,
        (
            replace(first, detections=first.detections + (blue,)),
            replace(second, detections=second.detections + (blue,)),
        ),
        0.2,
        5,
        episode().media,
        game="blackball",
    )
    assert result.game == "blackball"
    assert "unexpected_colour_hypothesis_for_blackball" in result.quality.flags
    assert result.quality.benchmark_accepted is False
