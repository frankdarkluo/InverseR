import csv
import fcntl
import gzip
import json
import logging
import math
import sqlite3
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, Self

import cv2
import numpy as np
from pydantic import Field, model_validator

from inverser.processing import decode_window, write_image
from inverser.real_data import (
    EpisodeMedia,
    OriginalAnnotation,
    SourceProvenance,
    calibration_record,
    export_episode,
    make_episode,
    render_clean_table_pair,
)
from inverser.runtime import PROJECT_ROOT, check_storage, file_sha256, write_json
from inverser.schemas import Record
from inverser.sources import VideoInfo, probe_video
from inverser.table_detection import locate_table
from inverser.vision import (
    BallAssociator,
    FrameObservation,
    ImageArray,
    TableProjection,
    observe_frame,
)

LOGGER = logging.getLogger(__name__)
Game = Literal["snooker", "blackball", "carom", "unknown"]
AssetRole = Literal[
    "original_video",
    "raw_capture",
    "background",
    "manual_path_image",
    "tracking_image",
    "tracking_video",
    "reconstructed_animation",
    "reconstruction_video",
    "scene_asset",
    "other",
]
PAPER_URL = "https://doi.org/10.1007/s10489-023-04542-3"


class UploadedAsset(Record):
    path: str
    game: Game
    role: AssetRole
    bytes: int = Field(ge=0)
    mtime_ns: int
    stable: bool
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    deferred_reason: str | None = None

    @model_validator(mode="after")
    def check_contract(self) -> Self:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.path or str(path) != self.path:
            raise ValueError("Uploaded asset path must remain inside its source root")
        if self.stable != (self.sha256 is not None):
            raise ValueError("Only unchanged, hashed files may be marked stable")
        if not self.stable and self.deferred_reason is None:
            raise ValueError("An unfinished asset needs an explicit reason")
        return self


class ReconstructedTrack(Record):
    source_track_id: str
    normalised_time_keys: tuple[float, ...]
    animation_cycle_s: float = Field(gt=0)
    scene_xyz: tuple[tuple[float, float, float], ...]
    units: str = "original scene coordinates; not verified table metres"
    time_alignment_to_capture_verified: Literal[False] = False
    algorithm_output_not_measured_ground_truth: Literal[True] = True


class ReconstructedAnimation(Record):
    schema_version: Literal["author-reconstruction-tracks-v1"] = "author-reconstruction-tracks-v1"
    algorithm: str
    source_file: str
    source_sha256: str
    tracks: tuple[ReconstructedTrack, ...]
    track_count: int
    track_sample_counts: tuple[int, ...]
    cue_control_ground_truth: None = None
    angular_velocity_ground_truth: None = None
    scene_units_and_capture_times_verified: Literal[False] = False


def source_game(path: str) -> Game:
    for part in PurePosixPath(path).parts:
        name = part.lower()
        if name.startswith(("snooker", "nooker_manual")):
            return "snooker"
        if name.startswith("blackball"):
            return "blackball"
        if name.startswith("carom"):
            return "carom"
    return "unknown"


def asset_role(path: str) -> AssetRole:
    relative = PurePosixPath(path)
    parts = [part.lower() for part in relative.parts]
    name = relative.name.lower()
    if name == "bgr.avi" and not any(
        part in ("trackingresults", "animationsresults") for part in parts
    ):
        return "original_video"
    if name == "bgr.dat":
        return "raw_capture"
    if name == "background.png":
        return "background"
    if name == "manual_track.png":
        return "manual_path_image"
    if relative.suffix.lower() == ".avi":
        if "trackingresults" in parts:
            return "tracking_video"
        if "animationsresults" in parts:
            return "reconstruction_video"
    if name.endswith("_track.png"):
        return "tracking_image"
    if relative.suffix.lower() == ".x3d":
        return "reconstructed_animation" if name.startswith("animation_") else "scene_asset"
    return "other"


def _signature(path: Path) -> tuple[int, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Uploaded inputs must be ordinary files, not links")
    info = path.stat()
    return info.st_size, info.st_mtime_ns


class UploadChangedError(RuntimeError):
    pass


def cached_asset_sha256(path: Path, cache: Path | None = None) -> tuple[str, bool]:
    _signature(path)
    info = path.stat()
    signature = (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino)
    if cache is None:
        digest = file_sha256(path)
        reused = False
    else:
        cache = cache.resolve()
        if not cache.is_relative_to(PROJECT_ROOT.resolve()):
            raise ValueError("Upload hash cache must remain inside the project")
        cache.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(cache, timeout=30)) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS hashes ("
                "path TEXT PRIMARY KEY,size INTEGER,mtime INTEGER,ctime INTEGER,"
                "device INTEGER,inode INTEGER,sha256 TEXT NOT NULL)"
            )
            previous = connection.execute(
                "SELECT size,mtime,ctime,device,inode,sha256 FROM hashes WHERE path=?",
                (str(path.resolve()),),
            ).fetchone()
            reused = previous is not None and tuple(previous[:5]) == signature
            digest = previous[5] if reused else file_sha256(path)
            after = path.stat()
            if (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_dev,
                after.st_ino,
            ) != signature:
                raise UploadChangedError(f"Source changed while checking its hash: {path.name}")
            with connection:
                connection.execute(
                    "INSERT OR REPLACE INTO hashes VALUES (?,?,?,?,?,?,?)",
                    (str(path.resolve()), *signature, digest),
                )
    after = path.stat()
    if (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_dev,
        after.st_ino,
    ) != signature:
        raise UploadChangedError(f"Source changed while checking its hash: {path.name}")
    return digest, reused


def snapshot_uploads(
    source: Path,
    output: Path,
    *,
    stable_seconds: float = 120,
    observe_seconds: float = 2,
    games: tuple[Game, ...] | None = None,
    hash_cache: Path | None = None,
    essential_only: bool = False,
) -> dict[str, object]:
    source, output = source.resolve(), output.resolve()
    data_root = (PROJECT_ROOT / "data").resolve()
    if (
        not source.is_relative_to(data_root)
        or not output.is_relative_to(data_root)
        or output == source
        or output.is_relative_to(source)
    ):
        raise ValueError(
            "Keep uploaded inputs and their inventory in separate project-data directories"
        )
    if stable_seconds < 0 or not 0 <= observe_seconds <= 30:
        raise ValueError(
            "Use a nonnegative age threshold and an observation interval of 0-30 seconds"
        )
    if not source.is_dir():
        raise ValueError("Direct-upload source directory does not exist")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".inventory.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "inventory.json").exists():
            raise ValueError("Inventories are immutable snapshots; use a new output version")
        files = [
            path
            for path in sorted(source.rglob("*"))
            if path.is_file()
            and not path.is_symlink()
            and (games is None or source_game(str(path.relative_to(source))) in games)
            and (
                not essential_only
                or asset_role(str(path.relative_to(source)))
                not in ("tracking_video", "reconstruction_video", "scene_asset")
            )
        ]
        initial = {str(path.relative_to(source)): _signature(path) for path in files}
        time.sleep(observe_seconds)
        assets = []
        files_hashed = hashes_reused = bytes_hashed = 0
        for name, before in initial.items():
            path = source / name
            reason = None
            digest = None
            try:
                after = _signature(path)
                if before != after:
                    reason = "file_changed_during_observation"
                elif time.time() - after[1] / 1e9 < stable_seconds:
                    reason = "recent_file_may_still_be_uploading"
                else:
                    digest, reused = cached_asset_sha256(path, hash_cache)
                    if reused:
                        hashes_reused += 1
                    else:
                        files_hashed += 1
                        bytes_hashed += before[0]
                    if _signature(path) != before:
                        digest = None
                        reason = "file_changed_while_hashing"
            except FileNotFoundError:
                reason = "file_disappeared_during_inventory"
                digest = None
            except UploadChangedError:
                reason = "file_changed_while_hashing"
                digest = None
            if reason:
                LOGGER.info("Deferred upload %s: %s", name, reason)
            assets.append(
                UploadedAsset(
                    path=name,
                    game=source_game(name),
                    role=asset_role(name),
                    bytes=before[0],
                    mtime_ns=before[1],
                    stable=digest is not None,
                    sha256=digest,
                    deferred_reason=reason,
                )
            )
        original = [asset for asset in assets if asset.role == "original_video"]
        summary = {
            "status": "snapshot_complete_not_entire_upload_complete",
            "source": str(source),
            "checked_utc": datetime.now(UTC).isoformat(),
            "stable_seconds": stable_seconds,
            "observe_seconds": observe_seconds,
            "game_filter": list(games) if games is not None else None,
            "essential_assets_only": essential_only,
            "hash_cache": str(hash_cache.resolve()) if hash_cache is not None else None,
            "files_hashed_this_run": files_hashed,
            "hashes_reused_this_run": hashes_reused,
            "bytes_hashed_this_run": bytes_hashed,
            "cache_signature": (
                "path,size,mtime,ctime,device,inode; operational cache not new full audit"
            ),
            "observed_files": len(assets),
            "observed_bytes": sum(asset.bytes for asset in assets),
            "stable_files": sum(asset.stable for asset in assets),
            "role_counts": dict(Counter(asset.role for asset in assets)),
            "original_videos_by_game": dict(Counter(asset.game for asset in original)),
            "stable_original_videos_by_game": dict(
                Counter(asset.game for asset in original if asset.stable)
            ),
            "main_snooker_original_videos": sum(
                asset.stable and asset.game == "snooker" for asset in original
            ),
            "manual_path_images": sum(asset.role == "manual_path_image" for asset in assets),
            "source_license": "CC-BY-SA-4.0 according to the author repository",
            "independent_label_accuracy_verified": False,
        }
        write_json(
            output / "inventory.json",
            {
                "schema_version": "author-direct-upload-v1",
                **summary,
                "assets": [asset.model_dump(mode="json") for asset in assets],
            },
        )
        write_json(output / "summary.json", summary)
        with (output / "assets.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["path", "game", "role", "bytes", "stable", "sha256", "deferred_reason"],
            )
            writer.writeheader()
            for asset in assets:
                writer.writerow({key: asset.model_dump()[key] for key in writer.fieldnames or ()})
        return summary


def verify_uploaded_asset(
    source: Path, asset: UploadedAsset, hash_cache: Path | None = None
) -> Path:
    path = source / asset.path
    if not path.resolve().is_relative_to(source.resolve()) or not asset.stable:
        raise ValueError("Asset is outside its source or has not completed stability checks")
    if _signature(path) != (asset.bytes, asset.mtime_ns):
        raise ValueError(f"Upload changed since its snapshot: {asset.path}")
    if cached_asset_sha256(path, hash_cache)[0] != asset.sha256:
        raise ValueError(f"Upload hash differs from its snapshot: {asset.path}")
    if _signature(path) != (asset.bytes, asset.mtime_ns):
        raise ValueError(f"Upload changed during verification: {asset.path}")
    return path


def decode_raw_capture(
    path: Path, info: VideoInfo, expected_signature: tuple[int, int]
) -> Iterator[tuple[float, ImageArray]]:
    frame_bytes = info.width * info.height * 3
    if expected_signature[0] % frame_bytes or not expected_signature[0]:
        raise ValueError("Raw BGR capture must contain whole nonempty image frames")
    with path.open("rb") as stream:
        for index in range(expected_signature[0] // frame_bytes):
            chunk = stream.read(frame_bytes)
            if len(chunk) != frame_bytes:
                raise ValueError("Raw capture became truncated while decoding")
            yield (
                index / info.fps,
                np.frombuffer(chunk, np.uint8).reshape(info.height, info.width, 3),
            )
        if stream.read(1) or _signature(path) != expected_signature:
            raise ValueError("Raw capture changed during decoding")


def read_reconstructed_animation(path: Path) -> ReconstructedAnimation:
    if path.stat().st_size > 20_000_000:
        raise ValueError("Animation XML exceeds the bounded parser input size")
    content = path.read_bytes()
    if b"<!ENTITY" in content.upper() or b"<![CDATA[" in content.upper():
        raise ValueError("Unsupported entity or executable-content declaration in animation")
    root = ET.fromstring(content)
    if root.tag != "X3D":
        raise ValueError("Expected an XML X3D animation")
    timers = {node.get("DEF"): node.get("cycleInterval") for node in root.iter("TimeSensor")}
    routes = [dict(node.attrib) for node in root.iter("ROUTE")]
    tracks: list[ReconstructedTrack] = []
    for node in root.iter("PositionInterpolator"):
        identifier = node.get("DEF")
        keys = [float(value) for value in node.get("key", "").replace(",", " ").split()]
        values = [float(value) for value in node.get("keyValue", "").replace(",", " ").split()]
        if (
            not identifier
            or not keys
            or len(values) != 3 * len(keys)
            or not all(math.isfinite(value) for value in (*keys, *values))
            or any(a >= b for a, b in zip(keys, keys[1:]))
            or min(keys) < 0
            or max(keys) > 1
        ):
            raise ValueError("Malformed reconstructed position samples or normalised times")
        timer_routes = [
            route
            for route in routes
            if route.get("toNode") == identifier and route.get("toField") == "set_fraction"
        ]
        if len(timer_routes) != 1:
            raise ValueError("PositionInterpolator must have one unambiguous source clock")
        cycle_text = timers.get(timer_routes[0].get("fromNode"))
        if cycle_text is None:
            raise ValueError("Reconstructed position samples have no declared animation clock")
        cycle = float(cycle_text)
        if not math.isfinite(cycle) or cycle <= 0:
            raise ValueError("Invalid reconstructed-animation cycle duration")
        tracks.append(
            ReconstructedTrack(
                source_track_id=identifier,
                normalised_time_keys=tuple(keys),
                animation_cycle_s=cycle,
                scene_xyz=tuple(
                    (values[index], values[index + 1], values[index + 2])
                    for index in range(0, len(values), 3)
                ),
            )
        )
    if not tracks:
        raise ValueError("Source animation contains no position tracks")
    return ReconstructedAnimation(
        algorithm=path.stem.removeprefix("animation_"),
        source_file=str(path),
        source_sha256=file_sha256(path),
        tracks=tuple(tracks),
        track_count=len(tracks),
        track_sample_counts=tuple(sorted({len(track.normalised_time_keys) for track in tracks})),
    )


def _clip_background(
    source: Path, clip: UploadedAsset, assets: dict[str, UploadedAsset]
) -> UploadedAsset | None:
    directory = PurePosixPath(clip.path).parent
    for parent in (directory, *directory.parents):
        name = str(parent / "background.png")
        asset = assets.get(name)
        if asset is not None and asset.stable:
            return asset
    return None


def convert_uploaded_clip(
    source: Path,
    clip: UploadedAsset,
    assets: dict[str, UploadedAsset],
    output: Path,
    *,
    hash_cache: Path | None = None,
    annotation_paths: tuple[str, ...] | None = None,
    group_id: str | None = None,
) -> dict[str, object]:
    if clip.game not in ("snooker", "blackball") or clip.role != "original_video":
        raise ValueError("Only original snooker or auxiliary blackball captures can be converted")
    video = verify_uploaded_asset(source, clip, hash_cache)
    info = probe_video(video)
    if info.duration_s > 300 or min(info.width, info.height) < 64:
        raise ValueError("Capture geometry/duration is outside this per-shot adapter")
    output.mkdir(parents=True, exist_ok=False)
    background_asset = _clip_background(source, clip, assets)
    if background_asset is None:
        raise ValueError(f"No stable source background available for {clip.path}")
    background = cv2.imread(str(verify_uploaded_asset(source, background_asset, hash_cache)))
    if background is None or background.shape[:2] != (info.height, info.width):
        raise ValueError("Background is not decodable at the capture resolution")
    location = locate_table(np.asarray(background, dtype=np.uint8))
    if location is None:
        raise ValueError("Source background does not support an automatic table quadrilateral")
    projection = TableProjection(
        [list(point) for point in location.points],
        info.width,
        info.height,
        table_width_m=0.935,
        table_length_m=1.854,
        ball_radius_m=0.0255 if clip.game == "snooker" else 0.0285,
        corner_order="ordered_short_edge",
    )
    raw_asset = assets.get(str(PurePosixPath(clip.path).with_suffix(".dat")))
    clock: Literal["capture_frame_index_at_declared_fps", "ffmpeg_resampling_grid"]
    if raw_asset is not None and raw_asset.stable:
        raw_path = verify_uploaded_asset(source, raw_asset, hash_cache)
        frames = decode_raw_capture(raw_path, info, (raw_asset.bytes, raw_asset.mtime_ns))
        primary = raw_asset
        clock = "capture_frame_index_at_declared_fps"
        raw_frame_count = raw_asset.bytes // (info.width * info.height * 3)
    else:
        frames = decode_window(
            video,
            start_s=0,
            duration_s=info.duration_s,
            sample_hz=info.fps,
            width=info.width,
            height=info.height,
        )
        primary = clip
        clock = "ffmpeg_resampling_grid"
        raw_frame_count = None
    original_annotations = []
    weak_exports = []
    prefix = str(PurePosixPath(clip.path).parent) + "/"
    for asset in assets.values():
        selected_annotation = (
            asset.path in annotation_paths
            if annotation_paths is not None
            else asset.path.startswith(prefix)
        )
        if not asset.stable or not selected_annotation:
            continue
        if asset.role not in ("manual_path_image", "tracking_image", "reconstructed_animation"):
            continue
        path = verify_uploaded_asset(source, asset, hash_cache)
        kind: Literal["manual_path_image", "tracking_output", "reconstructed_x3d"]
        if asset.role == "manual_path_image":
            kind = "manual_path_image"
        elif asset.role == "tracking_image":
            kind = "tracking_output"
        else:
            kind = "reconstructed_x3d"
            reconstructed = read_reconstructed_animation(path)
            weak_path = output / "algorithm_tracks" / f"{path.stem}.json.gz"
            weak_path.parent.mkdir(exist_ok=True)
            with gzip.open(weak_path, "wt", encoding="utf-8") as stream:
                stream.write(reconstructed.model_dump_json())
            verify_uploaded_asset(source, asset, hash_cache)
            weak_exports.append(
                {
                    "algorithm": reconstructed.algorithm,
                    "path": str(weak_path),
                    "tracks": reconstructed.track_count,
                    "sample_counts": list(reconstructed.track_sample_counts),
                    "animation_cycles_s": sorted(
                        {track.animation_cycle_s for track in reconstructed.tracks}
                    ),
                    "capture_clock_match_verified": False,
                }
            )
        if asset.sha256 is None:
            raise ValueError("Stable source annotation has no checksum")
        original_annotations.append(
            OriginalAnnotation(kind=kind, path=str(path), sha256=asset.sha256)
        )
    observations: list[FrameObservation] = []
    tracker = BallAssociator()
    onset: float | None = None
    first_source = last_source = first_bev = last_bev = None
    with gzip.open(output / "observations.jsonl.gz", "wt", encoding="utf-8") as stream:
        for timestamp, image in frames:
            observation, rectified = observe_frame(image, projection, tracker, timestamp)
            observations.append(observation)
            stream.write(json.dumps(observation.json(), allow_nan=False) + "\n")
            if first_source is None:
                first_source, first_bev = image.copy(), rectified.copy()
            last_source, last_bev = image.copy(), rectified.copy()
            if onset is None and observation.cue_motion_m is not None:
                if observation.cue_motion_m > 0.4 * projection.physics.ball_radius_m:
                    onset = timestamp
    if (
        len(observations) < 2
        or first_source is None
        or last_source is None
        or first_bev is None
        or last_bev is None
        or primary.sha256 is None
    ):
        raise ValueError("Source clip does not provide sufficient observed endpoint frames")
    verify_uploaded_asset(source, clip, hash_cache)
    if raw_asset is not None and raw_asset.stable:
        verify_uploaded_asset(source, raw_asset, hash_cache)
    verify_uploaded_asset(source, background_asset, hash_cache)
    if annotation_paths:
        for name in annotation_paths:
            verify_uploaded_asset(source, assets[name], hash_cache)
    media = EpisodeMedia(
        initial_source=str(output / "initial_source.jpg"),
        terminal_source=str(output / "terminal_source.jpg"),
        initial_bev=str(output / "initial_bev.jpg"),
        terminal_bev=str(output / "terminal_bev.jpg"),
        clean_table_pair=str(output / "clean_table_pair.png"),
        source_clip=str(video),
        source_clip_start_s=0,
        source_clip_end_s=observations[-1].time_s + 1 / info.fps,
    )
    episode = make_episode(
        SourceProvenance(
            collection="billiard_dataset",
            recording_id=PurePosixPath(clip.path).parent.name,
            original_file=str(source / primary.path),
            original_sha256=primary.sha256,
            group_id=group_id or primary.sha256,
            media_license="CC-BY-SA-4.0",
            source_publication=PAPER_URL,
        ),
        calibration_record(projection, "automatic_background_quad", "paper_baize_plus_cushion"),
        tuple(observations),
        onset,
        info.fps,
        media,
        tuple(original_annotations),
        game=clip.game,
        timestamp_basis=clock,
    )
    export_episode(output / "structured.json", episode)
    render_clean_table_pair(output / "clean_table_pair.png", episode)
    for media_path, image in (
        (media.initial_source, first_source),
        (media.terminal_source, last_source),
        (media.initial_bev, first_bev),
        (media.terminal_bev, last_bev),
    ):
        write_image(Path(media_path), image)
    report: dict[str, object] = {
        "status": "converted_unverified",
        "game": clip.game,
        "use_in_main_snooker_counts": clip.game == "snooker",
        "source": clip.path,
        "source_video_sha256": clip.sha256,
        "video": info.model_dump(mode="json"),
        "raw_complete_frames": raw_frame_count,
        "container_duration_frames_estimate": round(info.duration_s * info.fps),
        "observed_frames": len(observations),
        "frame_clock": clock,
        "raw_video_frame_alignment_verified": False,
        "initial_detected_balls": len(episode.initial.balls),
        "terminal_detected_balls": len(episode.terminal.balls),
        "quality_flags": list(episode.quality.flags),
        "original_manual_path_images": sum(
            label.kind == "manual_path_image" for label in original_annotations
        ),
        "weak_algorithm_exports": weak_exports,
        "structured_record": str(output / "structured.json"),
        "cue_or_spin_ground_truth": None,
        "benchmark_accepted": False,
    }
    write_json(output / "report.json", report)
    return report


def convert_upload_pilot(
    inventory: Path, output: Path, *, game: Literal["snooker", "blackball"], limit: int = 5
) -> dict[str, object]:
    if not 1 <= limit <= 20:
        raise ValueError("A direct-upload conversion pilot uses 1-20 original clips")
    raw = json.loads(inventory.read_text())
    if raw.get("schema_version") != "author-direct-upload-v1":
        raise ValueError("Expected a stability-checked direct-upload inventory")
    source = Path(raw["source"]).resolve()
    output = output.resolve()
    data_root = (PROJECT_ROOT / "data").resolve()
    if (
        not source.is_relative_to(data_root)
        or not output.is_relative_to(data_root)
        or output == source
        or output.is_relative_to(source)
    ):
        raise ValueError("Use a separate in-project structured output")
    assets = {
        asset.path: asset for asset in (UploadedAsset.model_validate(row) for row in raw["assets"])
    }
    eligible = [
        asset
        for asset in assets.values()
        if asset.game == game and asset.role == "original_video" and asset.stable
    ]
    unique: dict[str, UploadedAsset] = {}
    for asset in sorted(eligible, key=lambda row: row.path):
        if asset.sha256 is None:
            raise ValueError("Stable original video is missing a hash")
        unique.setdefault(asset.sha256, asset)
    chosen = list(unique.values())[:limit]
    if not chosen:
        raise ValueError(f"No stable original {game} clips in this upload snapshot")
    check_storage(1_000_000_000)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "inventory_sha256": file_sha256(inventory),
        "game": game,
        "limit": limit,
        "code_hashes": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "author_uploads.py",
                "real_data.py",
                "vision.py",
                "table_detection.py",
                "processing.py",
            )
        },
    }
    with (output / ".pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "config.json").exists():
            if json.loads((output / "config.json").read_text()) != config:
                raise ValueError("Direct-upload pilot source/code changed; use a new version")
            if (output / "summary.json").exists():
                return json.loads((output / "summary.json").read_text())
        elif any(path.name != ".pilot.lock" for path in output.iterdir()):
            raise ValueError("Refusing to overwrite an untracked pilot directory")
        write_json(output / "config.json", config)
        cv2.setNumThreads(1)
        reports = []
        for index, asset in enumerate(chosen):
            directory = output / f"{index:03d}_{PurePosixPath(asset.path).parent.name}"
            try:
                report = convert_uploaded_clip(source, asset, assets, directory)
                reports.append(report)
            except (
                OSError,
                ValueError,
                ET.ParseError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as error:
                LOGGER.error("Direct-upload clip failed %s: %s", asset.path, error)
                reports.append({"status": "failed", "source": asset.path, "error": str(error)})
            write_json(
                output / "pilot_state.json",
                {"status": "processing", "clips_examined": len(reports), "requested": len(chosen)},
            )
        summary: dict[str, object] = {
            "status": "pilot_complete"
            if all(row["status"] == "converted_unverified" for row in reports)
            else "pilot_with_failures",
            "game": game,
            "stable_originals_in_snapshot": len(eligible),
            "distinct_original_video_hashes": len(unique),
            "converted_clips": sum(row["status"] == "converted_unverified" for row in reports),
            "failed_clips": sum(row["status"] != "converted_unverified" for row in reports),
            "reports": reports,
            "main_snooker_tasks_added": 0,
            "auxiliary_only": game != "snooker",
            "accepted_benchmark_tasks": 0,
            "model_or_api_calls": 0,
            "new_uploads_are_not_automatically_included": True,
        }
        write_json(output / "summary.json", summary)
        write_json(
            output / "pilot_state.json",
            {key: value for key, value in summary.items() if key != "reports"},
        )
        if summary["failed_clips"]:
            raise RuntimeError(
                "Direct-upload pilot had explicit failures; inspect summary before expansion"
            )
        return summary
