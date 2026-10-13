import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

import inverser.author_uploads as uploads
from inverser.real_data import BlackballEpisode, SnookerEpisode
from inverser.sources import VideoInfo


@pytest.mark.parametrize(
    "path,game,role",
    [
        ("blackball/BlackBall_0/BGR.avi", "blackball", "original_video"),
        ("blackball/BlackBall_0/BGR.dat", "blackball", "raw_capture"),
        ("Snooker_Manual/nooker_0/manual_track.png", "snooker", "manual_path_image"),
        ("blackball/background.png", "blackball", "background"),
        ("carom/shot/TrackingResults/MOLT.avi", "carom", "tracking_video"),
        (
            "blackball/shot/AnimationsResults/animation_MOLT.x3d",
            "blackball",
            "reconstructed_animation",
        ),
        ("blackball/shot/AnimationsResults/BGR.avi", "blackball", "reconstruction_video"),
        ("snooker/shot/TrackingResults/Molt_track.png", "snooker", "tracking_image"),
    ],
)
def test_asset_roles_and_game_domains_are_not_conflated(path, game, role):
    assert uploads.source_game(path) == game
    assert uploads.asset_role(path) == role


def animation(path, sample_count=3):
    keys = " ".join(str(i / sample_count) for i in range(sample_count))
    points = " ".join(f"0 {i} 2" for i in range(sample_count))
    path.write_text(
        f'<X3D><Scene><TimeSensor DEF="CLOCK" cycleInterval="2"/>'
        f'<PositionInterpolator DEF="BALL" key="{keys}" keyValue="{points}"/>'
        '<ROUTE fromNode="CLOCK" fromField="fraction_changed" toNode="BALL" '
        'toField="set_fraction"/></Scene></X3D>'
    )


def test_reconstructed_tracks_keep_scene_time_and_do_not_become_measured_gt(tmp_path):
    path = tmp_path / "animation_MOLT.x3d"
    animation(path)
    parsed = uploads.read_reconstructed_animation(path)
    assert parsed.track_count == 1 and parsed.track_sample_counts == (3,)
    assert parsed.tracks[0].animation_cycle_s == 2
    assert parsed.tracks[0].time_alignment_to_capture_verified is False
    assert parsed.cue_control_ground_truth is None


def test_malformed_or_entity_xml_is_explicitly_rejected(tmp_path):
    path = tmp_path / "animation_MOLT.x3d"
    path.write_text('<!DOCTYPE X3D [<!ENTITY x "unsafe">]><X3D/>')
    with pytest.raises(ValueError, match="entity"):
        uploads.read_reconstructed_animation(path)
    path.write_text('<X3D><PositionInterpolator DEF="x" key="0 .5" keyValue="0 0 0"/></X3D>')
    with pytest.raises(ValueError, match="Malformed"):
        uploads.read_reconstructed_animation(path)


def test_files_changing_during_snapshot_are_deferred(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "data" / "source"
    shot = source / "blackball" / "shot"
    shot.mkdir(parents=True)
    original = shot / "BGR.avi"
    original.write_bytes(b"uploading")
    monkeypatch.setattr(
        uploads.time, "sleep", lambda seconds: original.write_bytes(b"growing upload")
    )
    output = tmp_path / "data" / "snapshot"
    report = uploads.snapshot_uploads(source, output, stable_seconds=0, observe_seconds=1)
    assert report["stable_files"] == 0
    raw = json.loads((output / "inventory.json").read_text())
    assert raw["assets"][0]["deferred_reason"] == "file_changed_during_observation"
    assert raw["main_snooker_original_videos"] == 0
    with pytest.raises(ValueError, match="immutable"):
        uploads.snapshot_uploads(source, output, stable_seconds=0, observe_seconds=0)


def test_raw_frames_need_integral_size_and_unchanged_signature(tmp_path):
    path = tmp_path / "BGR.dat"
    path.write_bytes(b"truncated")
    info = VideoInfo(width=64, height=64, duration_s=1, fps=20, codec="mjpeg")
    with pytest.raises(ValueError, match="whole"):
        list(uploads.decode_raw_capture(path, info, uploads._signature(path)))


def test_direct_blackball_pilot_uses_raw_capture_and_preserves_snooker_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(uploads, "check_storage", lambda reserve: None)
    source = tmp_path / "data" / "author"
    shot = source / "blackball" / "BlackBall_0"
    shot.mkdir(parents=True)
    background = np.zeros((240, 320, 3), np.uint8)
    cv2.rectangle(background, (20, 35), (300, 205), (30, 120, 40), -1)
    frames = []
    for index in range(40):
        frame = background.copy()
        cv2.circle(frame, (150 + min(index, 15), 130), 5, (240, 240, 240), -1)
        cv2.circle(frame, (100, 100), 5, (0, 0, 180), -1)
        cv2.circle(frame, (200, 100), 5, (0, 220, 220), -1)
        frames.append(frame)
    raw = shot / "BGR.dat"
    raw.write_bytes(b"".join(frame.tobytes() for frame in frames))
    video = shot / "BGR.avi"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            "320x240",
            "-framerate",
            "20",
            "-i",
            "-",
            "-an",
            "-c:v",
            "mjpeg",
            "-threads",
            "1",
            str(video),
        ],
        input=raw.read_bytes() + np.zeros_like(background).tobytes(),
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert cv2.imwrite(str(shot.parent / "background.png"), background)
    scenes = shot / "AnimationsResults"
    scenes.mkdir()
    animation(scenes / "animation_MOLT.x3d", sample_count=41)
    snapshot = tmp_path / "data" / "snapshot"
    uploads.snapshot_uploads(source, snapshot, stable_seconds=0, observe_seconds=0)
    output = tmp_path / "data" / "pilot"
    result = uploads.convert_upload_pilot(snapshot / "inventory.json", output, game="blackball")
    assert result["converted_clips"] == 1
    assert result["auxiliary_only"] is True and result["main_snooker_tasks_added"] == 0
    report = result["reports"][0]
    assert report["raw_complete_frames"] == report["observed_frames"] == 40
    assert report["container_duration_frames_estimate"] == 41
    assert len(report["weak_algorithm_exports"]) == 1
    episode = BlackballEpisode.model_validate_json(Path(report["structured_record"]).read_text())
    assert episode.game == "blackball"
    assert episode.calibration.ball_radius_m == 0.0285
    assert episode.timestamp_basis == "capture_frame_index_at_declared_fps"
    assert episode.source.original_file.endswith("BGR.dat")
    assert episode.quality.benchmark_accepted is False
    with pytest.raises(ValueError):
        SnookerEpisode.model_validate(episode.model_dump())
    public = json.loads(Path(report["structured_record"]).with_name("endpoints.json").read_text())
    assert public["game"] == "blackball" and "trajectories" not in public
    assert (
        uploads.convert_upload_pilot(snapshot / "inventory.json", output, game="blackball")
        == result
    )
    with pytest.raises(ValueError, match="No stable original snooker"):
        uploads.convert_upload_pilot(
            snapshot / "inventory.json", tmp_path / "data" / "snooker-empty", game="snooker"
        )
