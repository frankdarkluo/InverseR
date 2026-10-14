import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from inverser.canvas_audit import (
    RegistrationPolicy,
    audit_author_canvases,
    audit_registered_points,
    register_source_crop,
    rendered_body,
)
from inverser.real_data import (
    EpisodeMedia,
    OriginalAnnotation,
    SourceProvenance,
    calibration_record,
    make_episode,
)
from inverser.runtime import file_sha256
from inverser.vision import FrameObservation, TableProjection


def textured_source(seed=7):
    rng = np.random.default_rng(seed)
    image = np.empty((180, 240, 3), np.uint8)
    image[:] = (45, 125, 50)
    noise = rng.integers(-18, 19, image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_registration_uses_background_pixels_not_manual_path_shapes():
    source = textured_source()
    patch = source[35:135, 25:205].copy()
    cv2.circle(patch, (50, 50), 10, (250, 250, 250), -1)
    registration = register_source_crop(source, patch, RegistrationPolicy())
    assert registration.source_offset_px == (25, 35)
    assert registration.normalized_residual < 1e-6
    assert registration.separated_peak_margin > 0.001
    assert registration.background_fraction > 0.9
    assert registration.passed_policy


def test_crop_origins_may_change_between_recordings():
    source = textured_source()
    policy = RegistrationPolicy()
    first = register_source_crop(source, source[35:135, 25:205].copy(), policy)
    second = register_source_crop(source, source[30:130, 25:205].copy(), policy)
    assert first.passed_policy and second.passed_policy
    assert first.source_offset_px != second.source_offset_px


def test_uniform_background_is_rejected_as_ambiguous():
    source = np.full((180, 240, 3), (45, 125, 50), np.uint8)
    registration = register_source_crop(source, source[35:135, 25:205].copy(), RegistrationPolicy())
    assert not registration.passed_policy
    assert registration.separated_peak_margin == pytest.approx(0, abs=1e-6)


def test_registration_rejects_no_cloth_and_larger_or_wrong_typed_patch():
    source = textured_source()
    for patch in (
        np.zeros((100, 180, 3), np.uint8),
        np.ones((200, 300, 3), np.uint8),
        source.astype(np.float32),
    ):
        with pytest.raises(ValueError):
            register_source_crop(source, patch, RegistrationPolicy())


def test_render_geometry_uses_actual_native_canvas_and_excludes_headers():
    patch = textured_source()[:100, :180]
    frame = np.zeros((240, 360, 3), np.uint8)
    frame[140:240, 180:360] = patch
    assert np.array_equal(rendered_body(frame, (180, 100), RegistrationPolicy()), patch)
    with pytest.raises(ValueError, match="native panel"):
        rendered_body(frame, (180, 101), RegistrationPolicy())


def write_source_fixture(root: Path, index: int, *, shifted_validation=False):
    directory = root / f"Snooker_Manual_{index}"
    tracking = directory / "TrackingResults"
    tracking.mkdir(parents=True)
    source = textured_source(seed=7 + index)
    cv2.circle(source, (75, 85), 4, (240, 240, 240), -1)
    raw = directory / "BGR.dat"
    raw.write_bytes(source.tobytes() * 6)
    manual = directory / "manual_track.png"
    path_image = np.full((100, 180, 3), 255, np.uint8)
    cv2.line(path_image, (5, 5), (170, 95), (0, 0, 0), 3)
    cv2.circle(path_image, (50, 50), 4, (0, 0, 0), -1)
    assert cv2.imwrite(str(manual), path_image)
    render = tracking / "MOLT-VACHASPATI-LEGG.avi"
    writer = cv2.VideoWriter(str(render), cv2.VideoWriter_fourcc(*"MJPG"), 20, (360, 240))
    if not writer.isOpened():
        raise RuntimeError("Fixture tracking render could not be opened")
    for frame_index in range(6):
        origin_y = 35 + (4 if shifted_validation and frame_index >= 3 else 0)
        body = source[origin_y : origin_y + 100, 25:205]
        frame = np.zeros((240, 360, 3), np.uint8)
        frame[140:240, 180:360] = body
        writer.write(frame)
    writer.release()
    projection = TableProjection([[5, 5], [234, 5], [234, 174], [5, 174]], 240, 180)
    observations = tuple(
        FrameObservation(time, (), 0.99, 0, 0, True, True, 0, 0, False, False) for time in (0, 0.25)
    )
    episode = make_episode(
        SourceProvenance(
            collection="billiard_dataset",
            recording_id=f"source-{index}",
            original_file=str(raw),
            original_sha256=file_sha256(raw),
            group_id=f"source-group-{index}",
            media_license="test-fixture",
        ),
        calibration_record(projection, "supplied_corners", "paper_baize_plus_cushion"),
        observations,
        None,
        20,
        EpisodeMedia(
            initial_source="unused",
            terminal_source="unused",
            initial_bev="unused",
            terminal_bev="unused",
            clean_table_pair="unused",
            source_clip=None,
            source_clip_start_s=0,
            source_clip_end_s=0.3,
        ),
        source_annotations=(
            OriginalAnnotation(
                kind="manual_path_image",
                path=str(manual),
                sha256=file_sha256(manual),
            ),
        ),
    )
    stat = raw.stat()
    asset = {
        "path": str(raw.relative_to(root)),
        "game": "snooker",
        "role": "raw_capture",
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "stable": True,
        "sha256": file_sha256(raw),
    }
    return episode, asset


def test_complete_audit_reports_real_boundaries_and_rejects_temporal_crop_shift(tmp_path):
    source = tmp_path / "sources"
    fixtures = [
        write_source_fixture(source, index, shifted_validation=index == 2) for index in range(3)
    ]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": "author-direct-upload-v1",
                "source": str(source),
                "assets": [asset for _, asset in fixtures],
            }
        )
    )
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text("\n".join(episode.model_dump_json() for episode, _ in fixtures) + "\n")
    output = tmp_path / "audit"
    result = audit_author_canvases(episodes, inventory, output, limit_recordings=3)
    assert result.source_frames_read == result.source_render_frames_decoded == 6
    assert result.inputs.episodes_sha256 == file_sha256(episodes)
    assert result.inputs.inventory_sha256 == file_sha256(inventory)
    assert result.support_recordings == 2 and result.held_out_recordings == 1
    assert result.validated_recordings == 2
    assert result.recordings[2].held_out_frame.frozen_origin_error_px == pytest.approx(4)
    assert not result.recordings[2].temporal_mapping_consistent
    assert not result.g1_approved
    assert all(not row.physical_benchmark_accepted for row in result.recordings)
    assert all(not row.manual_pixel_paths_used_for_registration for row in result.recordings)
    assert all(not row.metric_world_accuracy_verified for row in result.recordings)
    assert all(not row.fresh_full_raw_checksum_computed for row in result.recordings)
    assert (output / "audit.json").is_file()
    with pytest.raises(FileExistsError):
        audit_author_canvases(episodes, inventory, output, limit_recordings=3)
    with pytest.raises(ValueError, match="2-16"):
        audit_author_canvases(episodes, inventory, tmp_path / "large", limit_recordings=100)


def test_point_compatibility_never_refits_to_query_paths_and_does_not_measure_recall(
    tmp_path, monkeypatch
):
    from inverser import canvas_audit

    source = tmp_path / "sources"
    fixtures = [write_source_fixture(source, index) for index in range(3)]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": "author-direct-upload-v1",
                "source": str(source),
                "assets": [asset for _, asset in fixtures],
            }
        )
    )
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text("\n".join(episode.model_dump_json() for episode, _ in fixtures) + "\n")
    mapping = tmp_path / "mapping"
    audit_author_canvases(episodes, inventory, mapping, limit_recordings=3)

    def forbidden_refit(*args, **kwargs):
        raise AssertionError("Query path must not trigger registration or transform fitting")

    monkeypatch.setattr(canvas_audit, "register_source_crop", forbidden_refit)
    result = audit_registered_points(mapping / "audit.json", episodes, tmp_path / "points.json")
    assert result.excluded_support_recordings == 2
    assert result.recording_groups == 1 and result.sampled_frames == 2
    assert result.detected_points == result.points_within_3px == 2
    assert result.conditional_detected_point_coverage == 1
    assert not result.transform_refitted_using_manual_paths
    assert not result.complete_inventory_recall_measured
    assert not result.timed_trajectory_accuracy_measured
    assert not result.g1_approved
    with pytest.raises(FileExistsError):
        audit_registered_points(mapping / "audit.json", episodes, tmp_path / "points.json")
    episodes.write_text(episodes.read_text() + "\n")
    with pytest.raises(ValueError, match="Observation index changed"):
        audit_registered_points(mapping / "audit.json", episodes, tmp_path / "changed_index.json")


def test_point_audit_rejects_mutated_sample_pixels(tmp_path):
    source = tmp_path / "sources"
    fixtures = [write_source_fixture(source, index) for index in range(3)]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": "author-direct-upload-v1",
                "source": str(source),
                "assets": [asset for _, asset in fixtures],
            }
        )
    )
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text("\n".join(episode.model_dump_json() for episode, _ in fixtures) + "\n")
    mapping = tmp_path / "mapping"
    audit_author_canvases(episodes, inventory, mapping, limit_recordings=3)
    raw = source / "Snooker_Manual_2/BGR.dat"
    with raw.open("r+b") as stream:
        stream.write(b"\xff\xff\xff")
    with pytest.raises(ValueError, match="source pixels changed"):
        audit_registered_points(mapping / "audit.json", episodes, tmp_path / "changed.json")
