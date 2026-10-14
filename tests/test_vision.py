from dataclasses import replace

import cv2
import numpy as np
import pytest

from inverser.processing import ShotWindowDetector, Snapshot
from inverser.vision import (
    BallAssociator,
    Detection,
    FrameObservation,
    GapAwareBallAssociator,
    TableProjection,
    classify_author_snooker_hsv,
    classify_hsv,
    detect_balls,
    representative_hsv,
)


def projection():
    return TableProjection([[0, 0], [511, 0], [511, 1027], [0, 1027]], 512, 1028)


def test_projection_respects_metric_axes():
    table = projection()
    assert table.xy_m(0, 0) == (0, 3.569)
    assert table.xy_m(table.width - 1, table.height - 1) == (1.778, 0)
    assert table.metadata()["ball_center_height_corrected"] is False
    with pytest.raises(ValueError, match="order"):
        TableProjection([[511, 0], [0, 0], [0, 1027], [511, 1027]], 512, 1028)


def test_isolated_ball_detection_and_cluster_uncertainty():
    table = projection()
    image = np.zeros((table.height, table.width, 3), np.uint8)
    image[:] = cv2.cvtColor(np.uint8([[[51, 132, 179]]]), cv2.COLOR_HSV2BGR)[0, 0]
    for center, hsv in [((120, 200), (0, 0, 240)), ((240, 500), (0, 220, 180))]:
        color = tuple(map(int, cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0]))
        cv2.circle(image, center, 8, color, -1)
    balls, felt, clutter, unresolved = detect_balls(image, table)
    assert {ball.kind for ball in balls} == {"cue", "red"}
    assert felt > 0.95
    assert clutter == 0 and unresolved == 0
    cv2.rectangle(image, (300, 600), (360, 660), (0, 0, 120), -1)
    _, _, clutter, unresolved = detect_balls(image, table)
    assert clutter > 0 and unresolved > 0


def test_native_detection_handles_spheres_above_a_perspective_table():
    table = TableProjection([[450, 50], [950, 50], [1250, 750], [150, 750]], 1400, 800)
    image = np.zeros((800, 1400, 3), np.uint8)
    felt = tuple(map(int, cv2.cvtColor(np.uint8([[[51, 132, 179]]]), cv2.COLOR_HSV2BGR)[0, 0]))
    cv2.fillConvexPoly(image, table.points.astype(np.int32), felt)
    for center, hsv in [((700, 200), (0, 0, 240)), ((550, 550), (0, 220, 180))]:
        color = tuple(map(int, cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0]))
        radius = round(table.source_ball_radius(*center))
        cv2.circle(image, center, radius, color, -1)
    balls, _, _, _ = detect_balls(image, table, native_image=True)
    assert {ball.kind for ball in balls} == {"cue", "red"}
    cue = next(ball for ball in balls if ball.kind == "cue")
    assert cue.xy_m == pytest.approx(table.xy_m(*table.source_to_bev(700, 200)), abs=0.006)


def test_tracker_retains_observed_identity_without_inventing_missing_balls():
    tracker = BallAssociator()
    cue = Detection("cue", (0.5, 1), (10, 20), 1, 1)
    red = Detection("red", (0.8, 1), (20, 20), 1, 1)
    first, _, _, _ = tracker.update((cue, red), 0)
    moved, maximum, _, changed = tracker.update((replace(cue, xy_m=(0.6, 1)), red), 0.2)
    assert not changed
    assert moved[0].track_id == first[0].track_id
    assert maximum == pytest.approx(0.1)
    missing, _, _, changed = tracker.update((moved[0],), 0.4)
    assert changed and len(missing) == 1
    with pytest.raises(ValueError, match="increase"):
        tracker.update((cue,), 0.3)


def test_red_hue_wrap_is_not_misclassified_as_yellow_or_blue():
    samples = np.asarray([[179, 220, 180], [0, 220, 180], [1, 220, 180]], np.uint8)
    h, s, v = representative_hsv(samples)
    assert h < 2 or h > 178
    assert s == 220 and v == 180


def test_source_specific_colours_are_opt_in_and_preserve_unknowns():
    assert classify_hsv(94, 200, 110) == "blue"
    assert classify_author_snooker_hsv(94, 200, 110) == "green"
    assert classify_hsv(155, 220, 185) == "unknown"
    assert classify_author_snooker_hsv(155, 220, 185) == "red"
    assert classify_author_snooker_hsv(159, 140, 254) == "pink"
    assert classify_author_snooker_hsv(70, 120, 140) == "unknown"
    assert classify_hsv(70, 120, 140) == "green"


def test_background_can_reveal_near_cloth_ball_without_inventory_forcing():
    table = projection()
    background = np.full((table.height, table.width, 3), (40, 110, 45), np.uint8)
    image = background.copy()
    colour = tuple(map(int, cv2.cvtColor(np.uint8([[[88, 200, 100]]]), cv2.COLOR_HSV2BGR)[0, 0]))
    cv2.circle(image, (240, 500), 8, colour, -1)
    detected, _, _, _ = detect_balls(
        image, table, colour_profile="author_snooker", background=background
    )
    assert len(detected) == 1 and detected[0].kind == "green"
    with pytest.raises(ValueError, match="coordinate canvas"):
        detect_balls(image, table, background=background[:10])
    with pytest.raises(ValueError, match="profile"):
        detect_balls(image, table, colour_profile="invented")


def test_compact_cue_recovers_visible_ball_connected_to_thin_shaft():
    table = projection()
    image = np.full((table.height, table.width, 3), (40, 110, 45), np.uint8)
    cv2.line(image, (240, 500), (230, 640), (230, 230, 230), 2)
    cv2.circle(image, (240, 500), 8, (250, 250, 250), -1)
    old, _, _, _ = detect_balls(image, table, native_image=True, colour_profile="author_snooker")
    assert not any(ball.kind == "cue" for ball in old)
    recovered, _, _, _ = detect_balls(
        image,
        table,
        native_image=True,
        colour_profile="author_snooker",
        compact_cue_recovery=True,
    )
    cues = [ball for ball in recovered if ball.kind == "cue"]
    assert len(cues) == 1
    assert cues[0].center_px == pytest.approx((240, 500), abs=2)
    assert cues[0].shape_score <= 0.75


def test_compact_cue_handles_contact_with_a_coloured_ball_without_filling_it():
    table = projection()
    image = np.full((table.height, table.width, 3), (40, 110, 45), np.uint8)
    cv2.circle(image, (240, 500), 8, (250, 250, 250), -1)
    cv2.circle(image, (256, 500), 8, (20, 20, 200), -1)
    recovered, _, _, _ = detect_balls(
        image,
        table,
        native_image=True,
        colour_profile="author_snooker",
        compact_cue_recovery=True,
    )
    assert len([ball for ball in recovered if ball.kind == "cue"]) == 1
    assert len(recovered) <= 2
    assert not any(ball.kind in ("pink", "black") for ball in recovered)


def test_compact_cue_rejects_thin_lines_and_large_white_occluders():
    table = projection()
    image = np.full((table.height, table.width, 3), (40, 110, 45), np.uint8)
    cv2.line(image, (200, 300), (230, 600), (250, 250, 250), 2)
    cv2.circle(image, (300, 500), 40, (250, 250, 250), -1)
    recovered, _, _, _ = detect_balls(
        image,
        table,
        native_image=True,
        colour_profile="author_snooker",
        compact_cue_recovery=True,
    )
    assert not any(ball.kind == "cue" for ball in recovered)
    with pytest.raises(ValueError, match="native author"):
        detect_balls(image, table, compact_cue_recovery=True)


def test_compact_cue_does_not_force_unique_inventory_or_duplicate_existing_cue():
    table = projection()
    image = np.full((table.height, table.width, 3), (40, 110, 45), np.uint8)
    for center in ((200, 300), (350, 500)):
        cv2.circle(image, center, 8, (250, 250, 250), -1)
    recovered, _, _, _ = detect_balls(
        image,
        table,
        native_image=True,
        colour_profile="author_snooker",
        compact_cue_recovery=True,
    )
    assert len([ball for ball in recovered if ball.kind == "cue"]) == 2


def test_gap_association_keeps_observed_points_without_filling_missing_frames():
    tracker = GapAwareBallAssociator()
    cue = Detection("cue", (0.5, 1), (10, 20), 1, 1)
    initial, _, _, _ = tracker.update((cue,), 0)
    missing, _, _, _ = tracker.update((), 0.05)
    assert missing == ()
    observed, _, _, _ = tracker.update((replace(cue, xy_m=(0.51, 1)),), 0.1)
    assert observed[0].track_id == initial[0].track_id
    assert observed[0].kind == "cue"
    tracker.update((), 0.2)
    later, _, _, _ = tracker.update((cue,), 0.3)
    assert later[0].track_id != initial[0].track_id


def test_gap_association_preserves_colour_abstention_and_flags_ambiguous_matches():
    tracker = GapAwareBallAssociator()
    cue = Detection("cue", (0.5, 1), (10, 20), 1, 1)
    initial, _, _, _ = tracker.update((cue,), 0)
    uncertain, _, _, _ = tracker.update((replace(cue, kind="unknown"),), 0.05)
    assert uncertain[0].track_id == initial[0].track_id
    assert uncertain[0].kind == "unknown"
    ambiguous = GapAwareBallAssociator()
    left = Detection("red", (0.5, 1), (10, 20), 1, 1)
    right = Detection("red", (0.52, 1), (20, 20), 1, 1)
    old, _, _, _ = ambiguous.update((left, right), 0)
    new, _, _, _ = ambiguous.update((replace(left, xy_m=(0.51, 1)),), 0.05)
    assert ambiguous.ambiguous_matches == 1
    assert new[0].track_id not in {ball.track_id for ball in old}


def test_subpixel_ball_core_is_flagged_not_allowed_to_abort_video(caplog):
    table = TableProjection(
        [[0, 0], [511, 0], [511, 1027], [0, 1027]],
        512,
        1028,
        ball_radius_m=1.1659600130884529 * 1.778 / 511,
    )
    image = np.zeros((table.height, table.width, 3), np.uint8)
    image[:] = cv2.cvtColor(np.uint8([[[51, 132, 179]]]), cv2.COLOR_HSV2BGR)[0, 0]
    cv2.rectangle(image, (100, 100), (103, 103), (0, 0, 180), -1)
    with caplog.at_level("DEBUG", logger="inverser.vision"):
        balls, felt, clutter, unresolved = detect_balls(image, table)
    assert not balls
    assert felt > 0.99
    assert clutter == 0
    assert unresolved == 1
    assert "under-resolved ball component" in caplog.text


def observation(t, motion, x=0.5):
    return FrameObservation(
        t,
        (Detection("cue", (x, 1), (10, 20), 1, 1, "cue"),),
        0.99,
        0,
        0,
        True,
        True,
        motion,
        0,
        False,
        False,
        motion,
        1,
    )


def test_stationary_motion_stationary_candidate_and_view_change():
    detector = ShotWindowDetector(0.026)
    frame = np.zeros((10, 10, 3), np.uint8)
    for i in range(8):
        assert detector.update(Snapshot(observation(i / 5, 0), frame, frame)) is None
    assert detector.update(Snapshot(observation(1.6, 0.1, 0.6), frame, frame)) is None
    candidate = None
    for i in range(9, 17):
        result = detector.update(Snapshot(observation(i / 5, 0, 0.6), frame, frame))
        candidate = result or candidate
    assert candidate is not None
    assert candidate.initial.observation.detections[0].xy_m == (0.5, 1)
    assert candidate.goal.observation.detections[0].xy_m == (0.6, 1)
    assert (
        detector.update(Snapshot(replace(observation(4, 0.1), view_valid=False), frame, frame))
        is None
    )


def test_stationary_cue_does_not_end_a_still_moving_object_ball_window():
    detector = ShotWindowDetector(0.026)
    frame = np.zeros((10, 10, 3), np.uint8)
    for i in range(8):
        detector.update(Snapshot(observation(i / 5, 0), frame, frame))
    detector.update(Snapshot(observation(1.6, 0.1, 0.6), frame, frame))
    for i in range(9, 18):
        moving = replace(observation(i / 5, 0.1, 0.6), cue_motion_m=0)
        assert detector.update(Snapshot(moving, frame, frame)) is None
