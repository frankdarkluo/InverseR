import pytest

from inverser.author_data import is_snooker_source_clip, select_author_clips


@pytest.mark.parametrize(
    "path,expected",
    [
        ("snooker/snooker_1/BGR.avi", True),
        ("Snooker_Manual/nooker_Manual_0/BGR.avi", True),
        ("snooker/snooker_1/AnimationsResults/reconstructed.avi", False),
        ("blackball/blackball_0/BGR.avi", False),
        ("snooker/snooker_1/TrackingResults/BGR.avi", False),
    ],
)
def test_source_game_and_generated_media_are_not_conflated(path, expected):
    assert is_snooker_source_clip(path) is expected


def test_same_manual_and_regular_recording_is_not_counted_twice():
    inventory = {
        "files": [
            {"path": "snooker/snooker_1/BGR.avi", "sha256": "a" * 64},
            {"path": "Snooker_Manual/snooker_1/BGR.avi", "sha256": "a" * 64},
            {"path": "snooker/snooker_2/BGR.avi", "sha256": "b" * 64},
        ]
    }
    clips, duplicates = select_author_clips(inventory)
    assert len(clips) == 2 and len(duplicates) == 1
    assert "Manual" in clips[0]["path"]


def test_author_adapter_uses_paper_dimensions_and_preserves_annotation_types(tmp_path, monkeypatch):
    import json
    import subprocess
    from pathlib import Path

    import cv2
    import numpy as np

    import inverser.author_data as adapter
    from inverser.real_data import SnookerEpisode
    from inverser.runtime import file_sha256

    monkeypatch.setattr(adapter, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(adapter, "check_storage", lambda reserve: None)
    source = tmp_path / "data" / "author"
    clip = source / "files" / "Snooker_Manual" / "snooker_0" / "BGR.avi"
    clip.parent.mkdir(parents=True)
    background = np.zeros((240, 320, 3), np.uint8)
    cv2.rectangle(background, (20, 35), (300, 205), (30, 120, 40), -1)
    frame = background.copy()
    cv2.circle(frame, (160, 120), 5, (240, 240, 240), -1)
    cv2.circle(frame, (100, 100), 5, (0, 0, 180), -1)
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
            str(clip),
        ],
        input=frame.tobytes() * 40,
        check=True,
        capture_output=True,
        timeout=30,
    )
    manual = clip.parent / "manual_track.png"
    cv2.imwrite(str(manual), background)
    bg = clip.parent.parent / "background.png"
    cv2.imwrite(str(bg), background)
    inventory = {
        "status": "complete",
        "files": [
            {
                "path": str(path.relative_to(source / "files")),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (clip, manual, bg)
        ],
    }
    (source / "_import").mkdir()
    (source / "_import" / "inventory.json").write_text(json.dumps(inventory))
    output = tmp_path / "data" / "structured"
    report = adapter.process_author_dataset(source, output)
    assert report["structured_unverified_clips"] == 1
    structured = report["reports"][0]["structured_record"]
    episode = SnookerEpisode.model_validate_json(Path(structured).read_text())
    assert episode.source.collection == "billiard_dataset"
    assert episode.calibration.table_size_m == (0.935, 1.854)
    assert episode.calibration.ball_radius_m == 0.0255
    assert episode.calibration.dimension_basis == "paper_baize_plus_cushion"
    assert episode.source_annotations[0].kind == "manual_path_image"
    assert episode.source_annotations[0].measured_action_ground_truth is False
    assert episode.quality.benchmark_accepted is False
    assert adapter.process_author_dataset(source, output) == report
