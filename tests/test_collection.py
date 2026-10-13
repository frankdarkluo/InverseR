import json
import subprocess
from concurrent.futures import Future
from pathlib import Path

import cv2
import numpy as np
import pytest

import inverser.collection as collection
from inverser.real_data import SnookerEpisode
from inverser.runtime import file_sha256


def job(path: Path, output: Path, start: float = 0, end: float = 6):
    info = path.stat()
    return collection.CollectionJob(
        job_id="fixture",
        video_path=str(path),
        recording_id="fixture",
        source_sha256=file_sha256(path),
        source_size_bytes=info.st_size,
        source_mtime_ns=info.st_mtime_ns,
        group_id="fixture-group",
        original_width=320,
        original_height=640,
        duration_s=6,
        owned_start_s=start,
        owned_end_s=end,
        scan_start_s=0,
        scan_end_s=6,
        corner_file=str(path.with_suffix(".json")),
        output=str(output),
        sample_hz=5,
    )


def test_window_ownership_covers_the_full_recording_once():
    windows = collection.ownership_windows(1423.5)
    assert sum(end - start for start, end, _, _ in windows) == pytest.approx(1423.5)
    assert windows[0][0] == 0 and windows[-1][1] == 1423.5
    for left, right in zip(windows, windows[1:]):
        assert left[1] == right[0]
        assert left[3] > right[2]
    assert all(end - start <= 900 for _, _, start, end in windows)


def test_context_overlap_does_not_own_a_shot_twice(tmp_path):
    path = tmp_path / "fixture.mp4"
    path.write_bytes(b"fixture")
    first = job(path, tmp_path / "first", 0, 3)
    second = job(path, tmp_path / "second", 3, 6)
    assert not collection.owns_candidate(first, 3)
    assert collection.owns_candidate(second, 3)


def test_worker_exports_common_bev_schema_and_keeps_raw_quality_flags(tmp_path):
    frames = []
    color = cv2.cvtColor(np.uint8([[[51, 132, 179]]]), cv2.COLOR_HSV2BGR)[0, 0]
    for i in range(30):
        frame = np.empty((640, 320, 3), np.uint8)
        frame[:] = color
        x = 120 if i < 9 else min(180, 120 + (i - 8) * 12)
        cv2.circle(frame, (x, 350), 5, (240, 240, 240), -1)
        cv2.circle(frame, (70, 200), 5, (0, 0, 180), -1)
        frames.append(frame)
    video = tmp_path / "fixture.mp4"
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
            "320x640",
            "-framerate",
            "5",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        input=b"".join(frame.tobytes() for frame in frames),
        check=True,
        capture_output=True,
        timeout=30,
    )
    video.with_suffix(".json").write_text(json.dumps([[0, 0], [319, 0], [319, 639], [0, 639]]))
    result = collection.collect_window(job(video, tmp_path / "output"), collection.code_hashes())
    assert result["status"] == "complete"
    assert result["sampled_frames"] == 30
    assert result["raw_candidate_records"] == 1
    episode = SnookerEpisode.model_validate_json(Path(result["records"][0]).read_text())
    assert episode.source.collection == "hf_snooker_videos"
    assert episode.quality.benchmark_accepted is False
    assert episode.quality.cue_endpoint_shift_m > 0.1
    assert Path(episode.media.clean_table_pair).is_file()
    assert Path(episode.media.source_clip).is_file()
    public = json.loads(Path(result["records"][0]).with_name("endpoints.json").read_text())
    assert "trajectories" not in public


def test_corpus_job_limit_pauses_and_resumes_without_repeating_complete_jobs(tmp_path, monkeypatch):
    source_file = tmp_path / "source.mp4"
    source_file.write_bytes(b"fixture")
    first = job(source_file, tmp_path / "first")
    second = first.model_copy(update={"job_id": "second", "output": str(tmp_path / "second")})
    monkeypatch.setattr(
        collection, "prepare_jobs", lambda *args, **kwargs: ([first, second], {"code_hashes": {}})
    )
    monkeypatch.setattr(collection, "code_hashes", lambda: {})
    monkeypatch.setattr(collection, "check_storage", lambda *args: None)
    calls = []

    class ImmediateExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, item, expected_code, attempt):
            calls.append(item.job_id)
            result = Future()
            result.set_result(
                {
                    "status": "complete",
                    "sampled_frames": 5,
                    "records": [],
                    "raw_candidate_records": 0,
                }
            )
            return result

    monkeypatch.setattr(collection, "ProcessPoolExecutor", ImmediateExecutor)
    output = tmp_path / "corpus"
    limited = collection.collect_corpus(tmp_path, output, workers=1, limit_jobs=1)
    assert limited["status"] == "paused" and limited["runner_active"] is False
    completed = collection.collect_corpus(tmp_path, output, workers=1)
    assert completed["status"] == "complete"
    assert calls == ["fixture", "second"]
    assert completed["jobs"] == {"complete": 2}


def test_failed_job_recovery_preserves_parent_and_skips_completed_jobs(tmp_path, monkeypatch):
    source = tmp_path / "data" / "source"
    source.mkdir(parents=True)
    source_file = source / "fixture.mp4"
    source_file.write_bytes(b"fixture")
    inventory = source / "inventory.json"
    inventory.write_text('{"fixture": true}\n')
    parent = tmp_path / "data" / "parent"
    recovered = tmp_path / "data" / "recovered"
    monkeypatch.setattr(collection, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(collection, "check_storage", lambda *args: None)
    phase = {"code": "before"}
    monkeypatch.setattr(collection, "code_hashes", lambda: {"vision.py": phase["code"]})

    def prepared(source_path, output_path, **kwargs):
        first = job(source_file, output_path / "fixture")
        second = first.model_copy(
            update={"job_id": "second", "output": str(output_path / "second")}
        )
        return [first, second], {
            "source": str(source_path.resolve()),
            "output": str(output_path.resolve()),
            "inventory_sha256": file_sha256(inventory),
            "sample_hz": 5,
            "code_hashes": collection.code_hashes(),
        }

    monkeypatch.setattr(collection, "prepare_jobs", prepared)
    calls = []

    class ImmediateExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, item, expected_code, attempt):
            calls.append((phase["code"], item.job_id))
            result = Future()
            if item.job_id == "second" and phase["code"] == "before":
                result.set_exception(ValueError("fixture empty-core error"))
            else:
                result.set_result(
                    {
                        "status": "complete",
                        "sampled_frames": 5,
                        "records": [],
                        "raw_candidate_records": 0,
                    }
                )
            return result

    monkeypatch.setattr(collection, "ProcessPoolExecutor", ImmediateExecutor)
    with pytest.raises(RuntimeError, match="Some corpus jobs failed"):
        collection.collect_corpus(source, parent, workers=1)
    original_database = (parent / "collection.sqlite").read_bytes()
    original_manifest = (parent / "collection_manifest.json").read_bytes()
    phase["code"] = "after"
    result = collection.recover_failed_corpus(source, parent, recovered, workers=1)
    assert result["status"] == "complete"
    assert calls == [("before", "fixture"), ("before", "second"), ("after", "second")]
    assert (parent / "collection.sqlite").read_bytes() == original_database
    assert (parent / "collection_manifest.json").read_bytes() == original_manifest
    provenance = json.loads((recovered / "recovery_manifest.json").read_text())
    assert provenance["inherited_complete_jobs"] == 1
    assert len(provenance["failed_jobs"]) == 1
    assert provenance["parent_code_hashes"] != provenance["recovery_code_hashes"]
    assert (
        collection.recover_failed_corpus(source, parent, recovered, workers=1)["status"]
        == "complete"
    )
    assert len(calls) == 3
