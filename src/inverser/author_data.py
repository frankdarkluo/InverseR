import csv
import fcntl
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Literal, TypedDict

import cv2
import numpy as np

from inverser.processing import decode_window, write_image
from inverser.real_data import (
    EpisodeMedia,
    OriginalAnnotation,
    SnookerEpisode,
    SourceProvenance,
    calibration_record,
    export_episode,
    make_episode,
    render_clean_table_pair,
)
from inverser.runtime import PROJECT_ROOT, check_storage, file_sha256, fingerprint, write_json
from inverser.sources import probe_video
from inverser.table_detection import locate_table
from inverser.vision import BallAssociator, FrameObservation, TableProjection, observe_frame

LOGGER = logging.getLogger(__name__)
PAPER_URL = "https://doi.org/10.1007/s10489-023-04542-3"


class DuplicateClipGroup(TypedDict):
    sha256: str
    files: list[str]


def is_snooker_source_clip(path: str) -> bool:
    parts = Path(path).parts
    return (
        Path(path).name.lower() == "bgr.avi"
        and any(part.lower().startswith(("snooker", "nooker_manual")) for part in parts)
        and not any(part.lower() in ("trackingresults", "animationsresults") for part in parts)
    )


def select_author_clips(inventory: dict) -> tuple[list[dict], list[DuplicateClipGroup]]:
    selected = [row for row in inventory["files"] if is_snooker_source_clip(row["path"])]
    by_hash: dict[str, list[dict]] = {}
    for row in selected:
        by_hash.setdefault(row["sha256"], []).append(row)
    clips = []
    aliases: list[DuplicateClipGroup] = []
    for sha256, rows in by_hash.items():
        rows.sort(key=lambda row: ("manual" not in row["path"].lower(), row["path"]))
        clips.append(rows[0])
        if len(rows) > 1:
            aliases.append({"sha256": sha256, "files": [row["path"] for row in rows]})
    return clips, aliases


def _annotations(
    root: Path, clip: Path, inventory_by_path: dict[str, dict]
) -> tuple[OriginalAnnotation, ...]:
    labels = []
    kind: Literal["manual_path_image", "tracking_output", "reconstructed_x3d", "source_notes"]
    for path in sorted(clip.parent.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        row = inventory_by_path.get(relative)
        if row is None:
            raise ValueError("Source annotation is not in the extracted-file inventory")
        if path.name.lower() == "manual_track.png":
            kind = "manual_path_image"
        elif path.suffix.lower() == ".x3d":
            kind = "reconstructed_x3d"
        elif path.suffix.lower() == ".png" and "trackingresults" in relative.lower():
            kind = "tracking_output"
        else:
            continue
        labels.append(OriginalAnnotation(kind=kind, path=str(path), sha256=row["sha256"]))
    return tuple(labels)


def _background_for(clip: Path, root: Path) -> Path | None:
    for parent in (clip.parent, *clip.parents):
        if not parent.is_relative_to(root):
            break
        for name in ("background.png", "Background.png"):
            background = parent / name
            if background.is_file():
                return background
    return None


def process_author_clip(
    root: Path, row: dict, inventory_by_path: dict[str, dict], output: Path
) -> dict[str, object]:
    source = root / row["path"]
    if source.stat().st_size != row["bytes"] or file_sha256(source) != row["sha256"]:
        raise ValueError("Author clip differs from its CRC/SHA-verified upload inventory")
    info = probe_video(source)
    if info.duration_s > 300:
        raise ValueError("Author source exceeds the per-shot length safety limit")
    output.mkdir(parents=True, exist_ok=True)
    old_report = output / "report.json"
    if old_report.exists():
        previous = json.loads(old_report.read_text())
        if (
            previous.get("source_sha256") == row["sha256"]
            and previous["status"] == "structured_unverified"
        ):
            structured = previous["structured_record"]
            if not isinstance(structured, str):
                raise ValueError("Invalid existing author record reference")
            SnookerEpisode.model_validate_json(Path(structured).read_text())
            return previous
    frames: list[FrameObservation] = []
    initial_image = terminal_image = initial_bev = terminal_bev = None
    projection: TableProjection | None = None
    method: Literal["automatic_felt_quad", "automatic_background_quad"] = "automatic_felt_quad"
    background = _background_for(source, root)
    if background is not None:
        decoded_background = cv2.imread(str(background))
        if decoded_background is None or decoded_background.shape[:2] != (info.height, info.width):
            raise ValueError("Source background image cannot be decoded at the video dimensions")
        background_image = np.asarray(decoded_background, dtype=np.uint8)
        location = locate_table(background_image)
        if location is not None:
            projection = TableProjection(
                [list(point) for point in location.points],
                info.width,
                info.height,
                table_width_m=0.935,
                table_length_m=1.854,
                ball_radius_m=0.0255,
                corner_order="ordered_short_edge",
            )
            method = "automatic_background_quad"
    tracker = BallAssociator()
    hz = min(info.fps, 20.0)
    onset: float | None = None
    for timestamp, image in decode_window(
        source,
        start_s=0,
        duration_s=info.duration_s,
        sample_hz=hz,
        width=info.width,
        height=info.height,
    ):
        if projection is None:
            location = locate_table(image)
            if location is None:
                continue
            projection = TableProjection(
                [list(point) for point in location.points],
                info.width,
                info.height,
                table_width_m=0.935,
                table_length_m=1.854,
                ball_radius_m=0.0255,
                corner_order="ordered_short_edge",
            )
        observation, rectified = observe_frame(image, projection, tracker, timestamp)
        frames.append(observation)
        if initial_image is None:
            initial_image, initial_bev = image.copy(), rectified.copy()
        terminal_image, terminal_bev = image, rectified
        if onset is None and observation.cue_motion_m is not None:
            if observation.cue_motion_m > 0.4 * projection.physics.ball_radius_m:
                onset = timestamp
    if (
        projection is None
        or len(frames) < 2
        or initial_image is None
        or terminal_image is None
        or initial_bev is None
        or terminal_bev is None
    ):
        report = {
            "status": "unqualified",
            "source": row["path"],
            "reason": "No valid table projection or sufficient observed frames",
            "benchmark_accepted": False,
        }
        write_json(output / "report.json", report)
        return report
    provenance = SourceProvenance(
        collection="billiard_dataset",
        recording_id=source.parent.name,
        original_file=str(source),
        original_sha256=row["sha256"],
        group_id=row["sha256"],
        media_license="CC-BY-SA-4.0",
        source_publication=PAPER_URL,
    )
    media = EpisodeMedia(
        initial_source=str(output / "initial_source.jpg"),
        terminal_source=str(output / "terminal_source.jpg"),
        initial_bev=str(output / "initial_bev.jpg"),
        terminal_bev=str(output / "terminal_bev.jpg"),
        clean_table_pair=str(output / "clean_table_pair.png"),
        source_clip=str(source),
        source_clip_start_s=0,
        source_clip_end_s=info.duration_s,
    )
    annotations = _annotations(root, source, inventory_by_path)
    episode = make_episode(
        provenance,
        calibration_record(projection, method, "paper_baize_plus_cushion"),
        tuple(frames),
        onset,
        hz,
        media,
        annotations,
    )
    export_episode(output / "structured.json", episode)
    render_clean_table_pair(output / "clean_table_pair.png", episode)
    write_image(output / "initial_source.jpg", initial_image)
    write_image(output / "terminal_source.jpg", terminal_image)
    write_image(output / "initial_bev.jpg", initial_bev)
    write_image(output / "terminal_bev.jpg", terminal_bev)
    with (output / "observations.jsonl").open("w") as stream:
        for observation in frames:
            stream.write(json.dumps(observation.json(), allow_nan=False) + "\n")
    report = {
        "status": "structured_unverified",
        "source": row["path"],
        "source_sha256": row["sha256"],
        "observed_frames": len(frames),
        "sample_hz": hz,
        "probed_video": info.model_dump(mode="json"),
        "capture_time_scale_verified": False,
        "structured_record": str(output / "structured.json"),
        "original_manual_path_images": sum(
            label.kind == "manual_path_image" for label in annotations
        ),
        "original_reconstruction_files": sum(
            label.kind == "reconstructed_x3d" for label in annotations
        ),
        "benchmark_accepted": False,
        "dimension_note": (
            "Paper reports 1.854x0.935m baize-plus-cushion region, not measured playing bounds. "
            "51mm ball diameter; projection requires validation."
        ),
    }
    write_json(output / "report.json", report)
    return report


def _process_author_dataset(
    source: Path, output: Path, *, wait_ready_seconds: float = 0
) -> dict[str, object]:
    output = output.resolve()
    if not output.is_relative_to((PROJECT_ROOT / "data").resolve()):
        raise ValueError("Author processing outputs must stay in InverseR/data")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    output.mkdir(parents=True, exist_ok=True)
    inventory_path = source / "_import" / "inventory.json"
    start = time.monotonic()
    while not inventory_path.exists():
        write_json(
            output / "author_state.json",
            {
                "status": "waiting_for_archive_import",
                "source": str(source),
                "model_or_api_calls": 0,
            },
        )
        if time.monotonic() - start >= wait_ready_seconds:
            raise TimeoutError("Author archive import is not ready")
        time.sleep(15)
    inventory = json.loads(inventory_path.read_text())
    if inventory["status"] != "complete":
        raise ValueError("Cannot process an incomplete archive extraction")
    config = {
        "source_inventory_sha256": file_sha256(inventory_path),
        "schema_version": "snooker-observation-v1",
        "code_hashes": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "author_data.py",
                "real_data.py",
                "vision.py",
                "processing.py",
                "table_detection.py",
                "runtime.py",
            )
        },
    }
    config["hash"] = fingerprint(config)
    config_path = output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Author adapter code/source changed; use a new output version")
    write_json(config_path, config)
    previous_summary = output / "summary.json"
    if previous_summary.exists():
        previous = json.loads(previous_summary.read_text())
        if previous["status"] == "complete":
            return previous
    root = (source / "files").resolve()
    by_path = {row["path"]: row for row in inventory["files"]}
    clips, duplicate_groups = select_author_clips(inventory)
    if not clips:
        write_json(
            output / "author_state.json", {"status": "no_snooker_source_clips", "accepted_tasks": 0}
        )
        raise ValueError("Uploaded archive has no original snooker BGR.avi clips")
    check_storage(5_000_000_000)
    cv2.setNumThreads(1)
    reports = []
    with (output / "episodes.jsonl").open("w") as records:
        for index, row in enumerate(clips):
            directory = output / "clips" / f"{index:04d}_{Path(row['path']).parent.name}"
            result = process_author_clip(root, row, by_path, directory)
            reports.append(result)
            if result["status"] == "structured_unverified":
                structured_path = result["structured_record"]
                if not isinstance(structured_path, str):
                    raise ValueError("Author adapter did not return a record path")
                record = Path(structured_path).read_text()
                records.write(json.dumps(json.loads(record), separators=(",", ":")) + "\n")
                records.flush()
            write_json(
                output / "author_state.json",
                {
                    "status": "processing",
                    "processed_source_clips": len(reports),
                    "source_clips": len(clips),
                },
            )
            LOGGER.info(
                "Processed author clip %s/%s: %s", len(reports), len(clips), result["status"]
            )
    summary = {
        "status": "complete",
        "source_files_inventory": str(inventory_path),
        "source_snooker_clip_files": sum(len(group["files"]) for group in duplicate_groups)
        + len(clips)
        - len(duplicate_groups),
        "distinct_video_byte_hashes": len(clips),
        "duplicate_clip_groups": duplicate_groups,
        "structured_unverified_clips": sum(
            row["status"] == "structured_unverified" for row in reports
        ),
        "unqualified_clips": sum(row["status"] != "structured_unverified" for row in reports),
        "accepted_benchmark_tasks": 0,
        "reports": reports,
        "schema_version": "snooker-observation-v1",
        "paper_sensor": {
            "resolution_px": [640, 480],
            "fps": 20,
            "focal_length_px": 525,
            "height_m": 1.853,
        },
        "paper_dimensions": {
            "baize_plus_cushion_m": [1.854, 0.935],
            "snooker_ball_diameter_m": 0.051,
        },
        "reconstruction_is_not_action_gt": True,
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "author_state.json",
        {key: value for key, value in summary.items() if key != "reports"},
    )
    with (output / "episode_index.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "status", "structured_record"])
        for row in reports:
            writer.writerow([row["source"], row["status"], row.get("structured_record", "")])
    return summary


def process_author_dataset(
    source: Path, output: Path, *, wait_ready_seconds: float = 0
) -> dict[str, object]:
    output = output.resolve()
    if not output.is_relative_to((PROJECT_ROOT / "data").resolve()) or wait_ready_seconds < 0:
        raise ValueError("Author processing needs an in-project output and nonnegative wait budget")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return _process_author_dataset(source, output, wait_ready_seconds=wait_ready_seconds)
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as error:
            write_json(
                output / "author_state.json",
                {"status": "failed", "error": type(error).__name__, "message": str(error)},
            )
            raise
