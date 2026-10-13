import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

import inverser.quality_screening as screening
from inverser.real_data import (
    EpisodeMedia,
    SourceProvenance,
    calibration_record,
    make_episode,
)
from inverser.vision import Detection, FrameObservation, TableProjection


def make_record(directory: Path, recording: str = "fixture"):
    directory.mkdir(parents=True)
    projection = TableProjection([[0, 0], [511, 0], [511, 1027], [0, 1027]], 512, 1028)
    frames = []
    for index in range(13):
        x = 0.5 if index < 2 else min(0.9, 0.5 + (index - 1) * 0.1)
        frames.append(
            FrameObservation(
                index / 5,
                (
                    Detection(
                        "cue",
                        (x, 1.5),
                        ((x / 1.778) * 511, (1 - 1.5 / 3.569) * 1027),
                        0.9,
                        1,
                        "cue",
                    ),
                    Detection(
                        "red",
                        (0.8, 2.5),
                        (0.8 / 1.778 * 511, (1 - 2.5 / 3.569) * 1027),
                        0.9,
                        1,
                        "red",
                    ),
                ),
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
                2,
            )
        )
    media = EpisodeMedia(
        initial_source=str(directory / "initial.jpg"),
        terminal_source=str(directory / "terminal.jpg"),
        initial_bev=str(directory / "first_bev.jpg"),
        terminal_bev=str(directory / "last_bev.jpg"),
        clean_table_pair=str(directory / "table.png"),
        source_clip=str(directory / "clip.mp4"),
        source_clip_start_s=0,
        source_clip_end_s=2.4,
    )
    episode = make_episode(
        SourceProvenance(
            collection="hf_snooker_videos",
            recording_id=recording,
            original_file="original.mp4",
            original_sha256=recording[0] * 64,
            group_id="group",
            media_license="unverified",
        ),
        calibration_record(projection, "supplied_corners", "nominal_regulation"),
        tuple(frames),
        0.4,
        5,
        media,
    )
    color = cv2.cvtColor(np.uint8([[[51, 132, 179]]]), cv2.COLOR_HSV2BGR)[0, 0]
    image = np.empty((1028, 512, 3), np.uint8)
    image[:] = color
    for filename in ("initial.jpg", "terminal.jpg", "first_bev.jpg", "last_bev.jpg", "table.png"):
        assert cv2.imwrite(str(directory / filename), image)
    (directory / "clip.mp4").write_bytes(b"fixture")
    record_path = directory / "structured.json"
    record_path.write_text(episode.model_dump_json())
    return episode, record_path


def test_metadata_checks_motion_and_rejects_pixel_metric_disagreement(tmp_path):
    episode, _ = make_record(tmp_path / "record")
    policy = screening.ScreeningPolicy()
    metrics = screening.metadata_metrics(episode, policy)
    assert metrics.connected_cue_track
    assert metrics.cue_excursion_radii > 10
    assert metrics.cue_motion_bursts == 1
    assert metrics.cue_terminal_still_s >= 0.8
    assert screening.decide_quality(metrics, policy)[0] == "needs_reprocessing"
    bad = episode.model_dump(mode="json")
    bad["initial"]["balls"][0]["bev_xy_px"][0] += 50
    broken = type(episode).model_validate(bad)
    with pytest.raises(ValueError, match="pixel"):
        screening.metadata_metrics(broken, policy)


def test_connected_return_shot_is_not_rejected_for_small_endpoint_shift(tmp_path):
    episode, _ = make_record(tmp_path / "record")
    metrics = screening.metadata_metrics(episode, screening.ScreeningPolicy())
    metrics = metrics.model_copy(
        update={"cue_endpoint_shift_m": 0, "camera_corner_shift_fraction": 0}
    )
    assert screening.decide_quality(metrics, screening.ScreeningPolicy()) == ("priority_review", ())


def test_low_motion_and_broken_identities_have_explicit_outcomes(tmp_path):
    episode, _ = make_record(tmp_path / "record")
    metrics = screening.metadata_metrics(episode, screening.ScreeningPolicy())
    no_motion = metrics.model_copy(update={"cue_excursion_radii": 0.1})
    assert screening.decide_quality(no_motion, screening.ScreeningPolicy())[0] == "not_suitable"
    broken = metrics.model_copy(
        update={"connected_cue_track": False, "duplicate_unique_colours": True}
    )
    kind, reasons = screening.decide_quality(broken, screening.ScreeningPolicy())
    assert kind == "needs_reprocessing"
    assert "cue_identity_or_continuity_unresolved" in reasons
    assert "duplicate_unique_ball_colours" in reasons
    moving_object = metrics.model_copy(
        update={"retained_object_balls_at_rest": False, "camera_corner_shift_fraction": 0}
    )
    assert (
        "terminal_object_ball_rest_unconfirmed"
        in screening.decide_quality(moving_object, screening.ScreeningPolicy())[1]
    )


def test_multiple_motion_bursts_require_boundary_review(tmp_path):
    episode, _ = make_record(tmp_path / "record")
    metrics = screening.metadata_metrics(episode, screening.ScreeningPolicy())
    repeated = metrics.model_copy(
        update={"cue_motion_bursts": 2, "camera_corner_shift_fraction": 0}
    )
    disposition, reasons = screening.decide_quality(repeated, screening.ScreeningPolicy())
    assert disposition == "needs_reprocessing"
    assert "single_cue_motion_burst_unresolved" in reasons


def test_camera_shift_and_unavailable_features_are_not_success_shaped():
    policy = screening.ScreeningPolicy()
    image = np.random.default_rng(42).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    corners = np.asarray([[120, 100], [280, 100], [280, 220], [120, 220]], dtype=np.float32)
    still = screening.compare_camera(image, image.copy(), corners, policy)
    assert still.corner_shift_fraction == pytest.approx(0, abs=1e-6)
    shifted = cv2.warpAffine(
        image, np.asarray([[1, 0, 20], [0, 1, 0]], dtype=np.float32), (400, 300)
    )
    moving = screening.compare_camera(image, shifted, corners, policy)
    assert moving.corner_shift_fraction > policy.max_camera_corner_shift_fraction
    unavailable = screening.compare_camera(
        np.zeros_like(image), np.zeros_like(image), corners, policy
    )
    assert unavailable.corner_shift_fraction is None
    assert unavailable.reason == "camera_background_features_unavailable"


def test_full_screening_is_readonly_repeatable_and_duplicate_aware(tmp_path, monkeypatch):
    monkeypatch.setattr(screening, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(screening, "check_storage", lambda *args: None)
    monkeypatch.setattr(
        screening,
        "compare_camera",
        lambda *args: screening.CameraComparison(
            matches=30, inlier_fraction=1, corner_shift_fraction=0
        ),
    )
    monkeypatch.setattr(
        screening,
        "probe_video",
        lambda *args: type("Probe", (), {"duration_s": 2.4, "fps": 30})(),
    )
    data = tmp_path / "data"
    source = data / "source"
    source.mkdir(parents=True)
    episodes = [
        make_record(source / "record_a", "a"),
        make_record(source / "record_b", "b"),
        make_record(source / "record_c", "c"),
    ]
    first_bev = Path(episodes[0][0].media.initial_bev)
    different_bev = cv2.imread(str(first_bev))
    cv2.rectangle(different_bev, (80, 80), (230, 160), (0, 0, 180), -1)
    assert cv2.imwrite(str(first_bev), different_bev)
    database = source / "collection.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE episodes(id TEXT,payload TEXT,path TEXT)")
        connection.executemany(
            "INSERT INTO episodes VALUES (?,?,?)",
            [
                (episode.episode_id, episode.model_dump_json(), str(path))
                for episode, path in episodes
            ],
        )
    (source / "collection_state.json").write_text(
        json.dumps({"status": "complete", "runner_active": False, "raw_candidates": 3})
    )
    (source / "collection_manifest.json").write_text('{"fixture":true}')
    original = database.read_bytes()
    output = data / "screened"
    first = screening.screen_corpus(source, output, workers=1)
    assert first["category_counts"] == {"priority_review": 1, "probable_duplicate": 2}
    assert first["accepted_benchmark_tasks"] == 0
    assert database.read_bytes() == original
    assert screening.screen_corpus(source, output, workers=1) == first
    rows = [
        screening.ScreeningDecision.model_validate_json(line)
        for line in (output / "decisions.jsonl").read_text().splitlines()
    ]
    keep = next(row for row in rows if row.disposition == "priority_review")
    assert all(
        row.duplicate_of == keep.episode_id
        for row in rows
        if row.disposition == "probable_duplicate"
    )
    assert all(row.benchmark_accepted is False for row in rows)
    original_image = cv2.imread(episodes[0][0].media.initial_source)
    decision = next(row for row in rows if row.episode_id == episodes[0][0].episode_id)
    assert decision.initial_perceptual_hash == screening.perceptual_hash(original_image)
    assert decision.initial_perceptual_hash != screening.perceptual_hash(different_bev)
    with pytest.raises(ValueError, match="changed"):
        screening.screen_corpus(
            source,
            output,
            workers=1,
            policy=screening.ScreeningPolicy(min_visible_fraction=0.99),
        )
