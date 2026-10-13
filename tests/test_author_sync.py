import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

import inverser.author_sync as sync
import inverser.author_uploads as uploads
from inverser.real_data import SnookerEpisode


def asset(path, role, sha="a" * 64, stable=True):
    return uploads.UploadedAsset(
        path=path,
        role=role,
        game="snooker",
        bytes=10,
        mtime_ns=1,
        stable=stable,
        sha256=sha if stable else None,
        deferred_reason=None if stable else "uploading",
    )


def test_duplicate_capture_and_manual_aliases_form_one_group():
    paths = [
        asset("snooker/Snooker_0/BGR.avi", "original_video"),
        asset("snooker/Snooker_0/BGR.dat", "raw_capture", "b" * 64),
        asset("snooker/background.png", "background", "c" * 64),
        asset("Snooker_Manual/Snooker_Manual_8/BGR.avi", "original_video"),
        asset("Snooker_Manual/Snooker_Manual_8/BGR.dat", "raw_capture", "b" * 64),
        asset("Snooker_Manual/background.png", "background", "c" * 64),
        asset("Snooker_Manual/Snooker_Manual_8/manual_track.png", "manual_path_image", "d" * 64),
    ]
    plans = sync.plan_recordings({row.path: row for row in paths})
    assert len(plans) == 1 and plans[0].ready
    assert plans[0].group_id == "b" * 64
    assert len(plans[0].aliases) == 2
    assert "Manual" in plans[0].selected_original
    assert plans[0].manual_path_images == ("Snooker_Manual/Snooker_Manual_8/manual_track.png",)


def test_unfinished_raw_capture_defers_conversion_instead_of_avi_fallback():
    rows = [
        asset("snooker/shot/BGR.avi", "original_video"),
        asset("snooker/shot/BGR.dat", "raw_capture", stable=False),
        asset("snooker/background.png", "background", "c" * 64),
    ]
    plan = sync.plan_recordings({row.path: row for row in rows})[0]
    assert not plan.ready and plan.raw_capture is None
    assert "complete_stable_raw_capture_not_available" in plan.deferred_reasons


def test_conflicting_raw_exports_do_not_silently_merge():
    rows = [
        asset("snooker/a/BGR.avi", "original_video"),
        asset("snooker/a/BGR.dat", "raw_capture", "b" * 64),
        asset("snooker/b/BGR.avi", "original_video"),
        asset("snooker/b/BGR.dat", "raw_capture", "c" * 64),
        asset("snooker/background.png", "background", "d" * 64),
    ]
    plan = sync.plan_recordings({row.path: row for row in rows})[0]
    assert not plan.ready
    assert "identical_video_aliases_have_conflicting_raw_captures" in plan.deferred_reasons


def test_hash_cache_reuses_unchanged_files_but_detects_same_size_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "source"
    path.write_bytes(b"aaaa")
    cache = tmp_path / "cache.sqlite"
    first, reused = uploads.cached_asset_sha256(path, cache)
    assert not reused
    assert uploads.cached_asset_sha256(path, cache) == (first, True)
    before = path.stat()
    path.write_bytes(b"bbbb")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    changed, reused = uploads.cached_asset_sha256(path, cache)
    assert not reused and changed != first


def test_snapshot_filters_auxiliary_games_and_reuses_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "data" / "source"
    for game in ("snooker", "blackball"):
        path = source / game / "shot" / "BGR.avi"
        path.parent.mkdir(parents=True)
        path.write_bytes(game.encode())
    rendered = source / "snooker" / "shot" / "TrackingResults" / "MOLT.avi"
    rendered.parent.mkdir()
    rendered.write_bytes(b"not an independent capture")
    cache = tmp_path / "cache.sqlite"
    first = uploads.snapshot_uploads(
        source,
        tmp_path / "data" / "first",
        stable_seconds=0,
        observe_seconds=0,
        games=("snooker",),
        hash_cache=cache,
        essential_only=True,
    )
    second = uploads.snapshot_uploads(
        source,
        tmp_path / "data" / "second",
        stable_seconds=0,
        observe_seconds=0,
        games=("snooker",),
        hash_cache=cache,
        essential_only=True,
    )
    assert first["observed_files"] == second["observed_files"] == 1
    assert first["files_hashed_this_run"] == 1
    assert second["hashes_reused_this_run"] == 1 and second["bytes_hashed_this_run"] == 0
    assert second["original_videos_by_game"] == {"snooker": 1}


def build_source(source: Path):
    shot = source / "snooker" / "Snooker_0"
    shot.mkdir(parents=True)
    background = np.zeros((240, 320, 3), np.uint8)
    cv2.rectangle(background, (20, 35), (300, 205), (30, 120, 40), -1)
    frames = []
    for index in range(40):
        frame = background.copy()
        cv2.circle(frame, (140 + min(index, 15), 130), 5, (240, 240, 240), -1)
        cv2.circle(frame, (90, 90), 5, (0, 0, 180), -1)
        frames.append(frame)
    raw = shot / "BGR.dat"
    raw.write_bytes(b"".join(frame.tobytes() for frame in frames))
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
            str(shot / "BGR.avi"),
        ],
        input=raw.read_bytes(),
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert cv2.imwrite(str(shot.parent / "background.png"), background)
    return shot, background


def test_sync_reuses_conversion_and_attaches_later_manual_paths_without_duplicate_shots(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(uploads, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sync, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sync, "check_storage", lambda *args: None)
    source = tmp_path / "data" / "source"
    original, background = build_source(source)
    cache = tmp_path / "hashes.sqlite"
    first_inventory = tmp_path / "data" / "inventory_1"
    uploads.snapshot_uploads(
        source,
        first_inventory,
        stable_seconds=0,
        observe_seconds=0,
        games=("snooker",),
        hash_cache=cache,
    )
    output = tmp_path / "data" / "structured"
    first = sync.sync_snooker_uploads(first_inventory / "inventory.json", output, hash_cache=cache)
    assert first["snapshot_records"] == first["converted_this_run"] == 1
    repeated = sync.sync_snooker_uploads(
        first_inventory / "inventory.json", output, hash_cache=cache
    )
    assert repeated["converted_this_run"] == 0 and repeated["reused_this_run"] == 1
    old_index = Path(first["episodes_jsonl"]).read_text()
    manual = source / "Snooker_Manual" / "Snooker_Manual_4"
    manual.mkdir(parents=True)
    for name in ("BGR.dat", "BGR.avi"):
        shutil.copyfile(original / name, manual / name)
    assert cv2.imwrite(str(manual.parent / "background.png"), background)
    raster = np.full((100, 200), 255, np.uint8)
    cv2.line(raster, (20, 30), (160, 70), 0, 4)
    assert cv2.imwrite(str(manual / "manual_track.png"), raster)
    (manual / "TrackingResults").mkdir()
    assert cv2.imwrite(str(manual / "TrackingResults" / "Molt_track.png"), raster)
    (manual / "Jaccard_Coefficient.txt").write_text("MOLT method: 1.0\n")
    second_inventory = tmp_path / "data" / "inventory_2"
    uploads.snapshot_uploads(
        source,
        second_inventory,
        stable_seconds=0,
        observe_seconds=0,
        games=("snooker",),
        hash_cache=cache,
    )
    second = sync.sync_snooker_uploads(
        second_inventory / "inventory.json", output, hash_cache=cache
    )
    assert second["original_snooker_video_files"] == 2 and second["content_groups"] == 1
    assert second["snapshot_records"] == 1 and second["records_with_manual_paths"] == 1
    assert second["converted_this_run"] == 1
    assert Path(first["episodes_jsonl"]).read_text() == old_index
    episode = SnookerEpisode.model_validate_json(Path(second["episodes_jsonl"]).read_text().strip())
    assert any(annotation.kind == "manual_path_image" for annotation in episode.source_annotations)
    assert episode.quality.benchmark_accepted is False
    metrics = second["manual_path_audit"]
    assert metrics["native_raster_comparisons"] == 1
    assert metrics["method_mean_path_iou"] == {"MOLT": 1.0}
    assert metrics["source_score_discrepancies_over_1e_5"] == 0
    assert metrics["resizing_or_alignment_fitted"] is False
    endpoint_path = Path(list(Path(output / "records").rglob("endpoints.json"))[-1])
    public = json.loads(endpoint_path.read_text())
    assert "source_annotations" not in public and "trajectories" not in public


def test_native_manual_canvas_mismatch_is_skipped_not_resized(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "data" / "source"
    folder = source / "Snooker_Manual" / "shot"
    (folder / "TrackingResults").mkdir(parents=True)
    manual = np.full((10, 10), 255, np.uint8)
    manual[3, 3] = 0
    assert cv2.imwrite(str(folder / "manual_track.png"), manual)
    assert cv2.imwrite(str(folder / "TrackingResults" / "Molt_track.png"), manual[:8])
    snapshot = tmp_path / "data" / "snapshot"
    uploads.snapshot_uploads(source, snapshot, stable_seconds=0, observe_seconds=0)
    raw = json.loads((snapshot / "inventory.json").read_text())
    assets = {row["path"]: uploads.UploadedAsset.model_validate(row) for row in raw["assets"]}
    report = sync.audit_manual_paths(source, assets, snapshot, None)
    assert report["native_raster_comparisons"] == 0
    assert report["skipped"][0]["reason"] == "native_canvas_shape_mismatch_no_resizing_applied"
