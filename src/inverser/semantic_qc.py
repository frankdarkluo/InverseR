import csv
import fcntl
import gzip
import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from typing import Literal, Self

import cv2
import numpy as np
from pydantic import Field, model_validator

from inverser.canvas_audit import CanvasAuditResult, RecordingMappingAudit
from inverser.diagnostics import PairedMeasurement, summarize_paired
from inverser.real_data import SnookerEpisode
from inverser.runtime import PROJECT_ROOT, check_storage, file_sha256, fingerprint, write_json
from inverser.schemas import Record
from inverser.vision import (
    BallAssociator,
    Detection,
    GapAwareBallAssociator,
    TableProjection,
    detect_balls,
)

LOGGER = logging.getLogger(__name__)
MAX_RECORDING_OUTPUT_BYTES = 80_000_000


@dataclass(frozen=True)
class Pipeline:
    name: str
    source_colours: bool
    use_background: bool
    temporal: bool = False
    compact_cue: bool = False


PIPELINES = (
    Pipeline("legacy_hsv", False, False),
    Pipeline("source_colour", True, False),
    Pipeline("background_only", False, True),
    Pipeline("source_background", True, True),
    Pipeline("source_background_temporal", True, True, True),
)

CUE_GAP_PIPELINES = (
    Pipeline("source_colour", True, False),
    Pipeline("source_colour_temporal", True, False, True),
    Pipeline("source_colour_compact_cue", True, False, False, True),
    Pipeline("source_colour_compact_cue_temporal", True, False, True, True),
)


class SemanticQCConfig(Record):
    schema_version: Literal["author-semantic-qc-v1"] = "author-semantic-qc-v1"
    workers: int = Field(default=4, ge=1, le=4)
    max_new_output_bytes: int = Field(default=20_000_000_000, gt=0, le=20_000_000_000)
    background_threshold: int = Field(default=24, ge=1, le=255)
    path_radius_px: Literal[7] = 7
    bootstrap_seed: int = Field(default=20261014, ge=0)
    profile_suite: Literal["initial_ablation", "cue_gap_no_background"] = "initial_ablation"


class PipelineQuality(Record):
    pipeline: str
    frames: int = Field(ge=1)
    mean_detected_balls: float
    minimum_detected_balls: int
    maximum_detected_balls: int
    unknown_detection_fraction: float
    cue_unresolved_frame_fraction: float
    unique_colour_conflict_frame_fraction: float
    ontology_conflict_frame_fraction: float
    unknown_frame_fraction: float
    unresolved_component_frame_fraction: float
    new_tracks_per_100_frames: float
    matched_observation_fraction: float
    ambiguous_temporal_matches: int
    path_iou: float
    path_raster_precision: float
    path_raster_coverage: float
    path_iou_radius_6: float
    path_iou_radius_8: float
    manual_foreground_pixels: int
    predicted_foreground_pixels: int


class RecordingQuality(Record):
    run_hash: str
    episode_id: str
    source_group: str
    source_name: str
    role: Literal["support", "held_out"]
    frames: int
    declared_fps: float
    original_raw_sha256: str
    manual_raster_sha256: str
    background_sha256: str
    profiles: tuple[PipelineQuality, ...]
    output_bytes: int
    artifacts_sha256: dict[str, str]
    source_payloads_modified: Literal[False] = False
    measured_cue_action_available: Literal[False] = False
    detection_accuracy_measured: Literal[False] = False
    complete_inventory_recall_measured: Literal[False] = False
    identity_accuracy_measured: Literal[False] = False
    benchmark_accepted: Literal[False] = False


class RecordingCommit(Record):
    report: RecordingQuality
    report_sha256: str

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if fingerprint(self.report.model_dump(mode="json")) != self.report_sha256:
            raise ValueError("Committed recording report checksum differs")
        return self


def _signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino


def _path_scores(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    intersection = int(np.count_nonzero(predicted & target))
    predicted_count, target_count = int(np.count_nonzero(predicted)), int(np.count_nonzero(target))
    if not target_count:
        raise ValueError("Manual path raster must contain a nonempty spatial reference")
    union = predicted_count + target_count - intersection
    return (
        intersection / union,
        intersection / predicted_count if predicted_count else 0.0,
        intersection / target_count,
    )


def _image_write(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"QC image could not be written: {path}")


def _recording_job(
    episode: SnookerEpisode,
    mapping: RecordingMappingAudit,
    config: SemanticQCConfig,
    run_hash: str,
    output: Path,
) -> RecordingQuality:
    cv2.setNumThreads(1)
    if (
        episode.source.group_id != mapping.source_group
        or episode.source.original_file != mapping.raw_capture
        or episode.source.original_sha256 != mapping.raw_snapshot_sha256
        or not 0 < episode.sample_hz <= 60
    ):
        raise ValueError("QC source group, raw capture or declared clock disagrees with the audit")
    if output.exists():
        raise FileExistsError("A recording QC job needs a new immutable directory")
    output.mkdir(parents=True)
    raw, manual = Path(mapping.raw_capture), Path(mapping.manual_raster)
    candidates = (
        raw.parent / "background.png",
        raw.parents[1] / "background.png",
    )
    background_path = next((path for path in candidates if path.is_file()), None)
    if background_path is None:
        raise ValueError("Author source has no declared local background image")
    signatures = {path: _signature(path) for path in (raw, manual, background_path)}
    if file_sha256(raw) != mapping.raw_snapshot_sha256:
        raise ValueError("Raw capture differs from the frozen source inventory")
    if file_sha256(manual) != mapping.manual_raster_sha256:
        raise ValueError("Manual raster differs from the frozen native mapping")
    background_hash = file_sha256(background_path)
    background = cv2.imread(str(background_path))
    target = cv2.imread(str(manual), cv2.IMREAD_GRAYSCALE)
    c = episode.calibration
    width, height = c.image_size_px
    if background is None or background.shape != (height, width, 3):
        raise ValueError("Native background is not available at the source-frame resolution")
    background_image = np.asarray(background, dtype=np.uint8)
    if target is None or (target.shape[1], target.shape[0]) != mapping.native_canvas_px:
        raise ValueError("Manual reference must remain in its original native canvas")
    manual_mask = target < 128
    if not np.any(manual_mask):
        raise ValueError("Manual path reference is empty")
    frame_bytes = width * height * 3
    if raw.stat().st_size != mapping.raw_frame_count * frame_bytes:
        raise ValueError("Native raw frame count differs from the spatial audit")
    projection = TableProjection(
        [list(point) for point in c.source_corners_px],
        width,
        height,
        output_width=c.bev_size_px[0],
        table_width_m=c.table_size_m[0],
        table_length_m=c.table_size_m[1],
        ball_radius_m=c.ball_radius_m,
        corner_order="ordered_short_edge",
    )
    inverse = np.linalg.inv(projection.matrix)
    offset = mapping.calibration.registration.source_offset_px
    pipelines = PIPELINES if config.profile_suite == "initial_ablation" else CUE_GAP_PIPELINES
    trackers: dict[str, BallAssociator] = {
        pipeline.name: GapAwareBallAssociator() if pipeline.temporal else BallAssociator()
        for pipeline in pipelines
    }
    path_masks = {
        pipeline.name: {radius: np.zeros_like(target, np.uint8) for radius in (6, 7, 8)}
        for pipeline in pipelines
    }
    counts: dict[str, list[int]] = {pipeline.name: [] for pipeline in pipelines}
    statistics: dict[str, Counter[str]] = {pipeline.name: Counter() for pipeline in pipelines}
    track_ids: dict[str, set[str]] = {pipeline.name: set() for pipeline in pipelines}
    last_native = None
    with ExitStack() as stack:
        streams = {
            pipeline.name: stack.enter_context(
                gzip.open(output / f"{pipeline.name}_observations.jsonl.gz", "wt", encoding="utf-8")
            )
            for pipeline in pipelines
        }
        raw_stream = stack.enter_context(raw.open("rb"))
        for frame_index in range(mapping.raw_frame_count):
            data = raw_stream.read(frame_bytes)
            if len(data) != frame_bytes:
                raise ValueError("Source raw capture truncated during the bounded pilot")
            image = np.frombuffer(data, np.uint8).reshape(height, width, 3)
            last_native = image
            cached_detections: dict[tuple[bool, bool, bool], tuple[tuple[Detection, ...], int]] = {}
            for pipeline in pipelines:
                key = pipeline.source_colours, pipeline.use_background, pipeline.compact_cue
                if key not in cached_detections:
                    detected, _, _, unresolved = detect_balls(
                        image,
                        projection,
                        native_image=True,
                        colour_profile="author_snooker" if pipeline.source_colours else "default",
                        background=background_image if pipeline.use_background else None,
                        background_threshold=config.background_threshold,
                        compact_cue_recovery=pipeline.compact_cue,
                    )
                    cached_detections[key] = detected, unresolved
                detections, unresolved = cached_detections[key]
                tracker = trackers[pipeline.name]
                observed, maximum, median, changed = tracker.update(
                    detections, frame_index / episode.sample_hz
                )
                colour_counts: Counter[str] = Counter(ball.kind for ball in observed)
                unknown = colour_counts["unknown"] > 0
                cue_unresolved = colour_counts["cue"] != 1
                unique_conflict = any(
                    colour_counts[kind] > 1
                    for kind in ("yellow", "green", "brown", "blue", "pink", "black")
                )
                ontology_conflict = (
                    unknown
                    or cue_unresolved
                    or unique_conflict
                    or colour_counts["red"] > 15
                    or len(observed) > 22
                )
                stats = statistics[pipeline.name]
                stats.update(
                    {
                        "unknown_detections": colour_counts["unknown"],
                        "detections": len(observed),
                        "unknown_frames": int(unknown),
                        "cue_unresolved": int(cue_unresolved),
                        "unique_conflict": int(unique_conflict),
                        "ontology_conflict": int(ontology_conflict),
                        "unresolved_components": int(unresolved > 0),
                        "matched_observations": tracker.matched_ball_count,
                        "ambiguous_matches": (
                            tracker.ambiguous_matches
                            if isinstance(tracker, GapAwareBallAssociator)
                            else 0
                        ),
                    }
                )
                counts[pipeline.name].append(len(observed))
                native_points = []
                for ball in observed:
                    if ball.track_id is None:
                        raise RuntimeError("An observed point has no explicit track hypothesis")
                    track_ids[pipeline.name].add(ball.track_id)
                    homogeneous = inverse @ np.asarray([*ball.center_px, 1.0])
                    if not np.isfinite(homogeneous).all() or abs(homogeneous[2]) < 1e-12:
                        raise ValueError("Detected point has no finite native coordinate")
                    x, y = homogeneous[:2] / homogeneous[2]
                    native_points.append((float(x), float(y)))
                    if (
                        0 <= x - offset[0] < target.shape[1]
                        and 0 <= y - offset[1] < target.shape[0]
                    ):
                        center = round(float(x - offset[0])), round(float(y - offset[1]))
                        for radius, mask in path_masks[pipeline.name].items():
                            cv2.circle(mask, center, radius, 255, -1)
                payload = {
                    "frame_index": frame_index,
                    "time_s": frame_index / episode.sample_hz,
                    "detections": [ball.json() for ball in observed],
                    "native_points_px": native_points,
                    "unresolved_components": unresolved,
                    "ontology_conflict": ontology_conflict,
                    "max_observed_motion_m": maximum,
                    "median_observed_motion_m": median,
                    "inventory_changed": changed,
                    "ambiguous_temporal_matches": (
                        tracker.ambiguous_matches
                        if isinstance(tracker, GapAwareBallAssociator)
                        else 0
                    ),
                }
                encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
                if len(encoded) > 262_144:
                    raise ValueError(
                        "Unexpectedly large QC frame record exceeds the output safeguard"
                    )
                streams[pipeline.name].write(encoded + "\n")
                if frame_index in (0, mapping.raw_frame_count - 1):
                    overlay = image.copy()
                    for ball, point in zip(observed, native_points, strict=True):
                        center = round(point[0]), round(point[1])
                        cv2.circle(overlay, center, 8, (0, 0, 255), 1)
                        cv2.putText(
                            overlay,
                            ball.kind,
                            (center[0] + 6, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (255, 255, 255),
                            1,
                        )
                    label = "initial" if frame_index == 0 else "terminal"
                    _image_write(output / f"{pipeline.name}_{label}.png", overlay)
            if frame_index % 32 == 0:
                for stream in streams.values():
                    stream.flush()
                if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > (
                    MAX_RECORDING_OUTPUT_BYTES
                ):
                    raise RuntimeError("Recording output reached its fixed storage guard")
        if raw_stream.read(1):
            raise ValueError("Source raw capture grew during the pilot")
    if last_native is None:
        raise ValueError("Native source did not provide any frames")
    for path, original_signature in signatures.items():
        if _signature(path) != original_signature:
            raise ValueError("Source payload changed while the pilot was running")
    profiles = []
    for pipeline in pipelines:
        name = pipeline.name
        stats = statistics[name]
        frames, detection_count = mapping.raw_frame_count, stats["detections"]
        if not detection_count:
            LOGGER.error("Pipeline %s returned zero detections for %s", name, episode.episode_id)
            raise ValueError("Empty detections cannot be presented as clean inventory results")
        mask = path_masks[name][7]
        iou, precision, coverage = _path_scores(mask > 0, manual_mask)
        _image_write(output / f"{name}_path.png", 255 - mask)
        profiles.append(
            PipelineQuality(
                pipeline=name,
                frames=frames,
                mean_detected_balls=float(np.mean(counts[name])),
                minimum_detected_balls=min(counts[name]),
                maximum_detected_balls=max(counts[name]),
                unknown_detection_fraction=stats["unknown_detections"] / detection_count,
                cue_unresolved_frame_fraction=stats["cue_unresolved"] / frames,
                unique_colour_conflict_frame_fraction=stats["unique_conflict"] / frames,
                ontology_conflict_frame_fraction=stats["ontology_conflict"] / frames,
                unknown_frame_fraction=stats["unknown_frames"] / frames,
                unresolved_component_frame_fraction=stats["unresolved_components"] / frames,
                new_tracks_per_100_frames=100 * len(track_ids[name]) / frames,
                matched_observation_fraction=stats["matched_observations"] / detection_count,
                ambiguous_temporal_matches=stats["ambiguous_matches"],
                path_iou=iou,
                path_raster_precision=precision,
                path_raster_coverage=coverage,
                path_iou_radius_6=_path_scores(path_masks[name][6] > 0, manual_mask)[0],
                path_iou_radius_8=_path_scores(path_masks[name][8] > 0, manual_mask)[0],
                manual_foreground_pixels=int(np.count_nonzero(manual_mask)),
                predicted_foreground_pixels=int(np.count_nonzero(mask)),
            )
        )
    artifacts = {path.name: file_sha256(path) for path in output.iterdir() if path.is_file()}
    report = RecordingQuality(
        run_hash=run_hash,
        episode_id=episode.episode_id,
        source_group=episode.source.group_id,
        source_name=raw.parent.name,
        role=mapping.role,
        frames=mapping.raw_frame_count,
        declared_fps=episode.sample_hz,
        original_raw_sha256=mapping.raw_snapshot_sha256,
        manual_raster_sha256=mapping.manual_raster_sha256,
        background_sha256=background_hash,
        profiles=tuple(profiles),
        output_bytes=sum(path.stat().st_size for path in output.iterdir() if path.is_file()),
        artifacts_sha256=artifacts,
    )
    if report.output_bytes > MAX_RECORDING_OUTPUT_BYTES:
        raise RuntimeError("Completed recording output exceeds its reserved storage limit")
    commit = RecordingCommit(
        report=report, report_sha256=fingerprint(report.model_dump(mode="json"))
    )
    write_json(output / "report.json", commit.model_dump(mode="json"))
    LOGGER.info(
        "Completed source %s: %d frames, %d fixed profiles",
        raw.parent.name,
        report.frames,
        len(pipelines),
    )
    return report


def run_semantic_qc(
    mapping_path: Path,
    episodes_path: Path,
    output: Path,
    config: SemanticQCConfig,
    *,
    limit_recordings: int | None = None,
) -> dict[str, object]:
    if limit_recordings is not None and not 1 <= limit_recordings <= 16:
        raise ValueError("The author semantic pilot permits 1-16 recordings")
    mapping = CanvasAuditResult.model_validate_json(mapping_path.read_text())
    if file_sha256(episodes_path) != mapping.inputs.episodes_sha256:
        raise ValueError("Observation source index changed since native mapping validation")
    if any(not row.spatial_registration_validated for row in mapping.recordings):
        raise ValueError("Every selected native canvas must pass mapping validation")
    indexed = {
        episode.episode_id: episode
        for episode in (SnookerEpisode.model_validate_json(line) for line in episodes_path.open())
    }
    pipelines = PIPELINES if config.profile_suite == "initial_ablation" else CUE_GAP_PIPELINES
    reference_pipeline = pipelines[0].name
    manifest = {
        "schema_version": "author-semantic-pilot-v1",
        "config": config.model_dump(mode="json"),
        "mapping_sha256": file_sha256(mapping_path),
        "episodes_sha256": file_sha256(episodes_path),
        "source_code_hash": fingerprint(
            {
                name: file_sha256(Path(__file__).with_name(name))
                for name in (
                    "semantic_qc.py",
                    "vision.py",
                    "schemas.py",
                    "real_data.py",
                    "canvas_audit.py",
                    "diagnostics.py",
                    "runtime.py",
                )
            }
        ),
        "runtime_versions": {
            name: version(name) for name in ("numpy", "opencv-python-headless", "pydantic", "scipy")
        },
        "pipelines": [pipeline.__dict__ for pipeline in pipelines],
        "reference_pipeline": reference_pipeline,
        "support_groups": [row.source_group for row in mapping.recordings if row.role == "support"],
        "evaluation_groups": [
            row.source_group for row in mapping.recordings if row.role == "held_out"
        ],
        "actual_model_requests": 0,
        "human_annotations_added": 0,
        "g1_approved": False,
    }
    run_hash = fingerprint(manifest)
    if not output.resolve().is_relative_to((PROJECT_ROOT / "data").resolve()):
        raise ValueError("Real-data pilot outputs must remain inside project data")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Changed code/data/protocol needs a new immutable pilot version")
        else:
            if any(path.name != ".writer.lock" for path in output.iterdir()):
                raise ValueError("A new semantic pilot needs an empty output directory")
            write_json(manifest_path, manifest)
        rows: list[RecordingQuality] = []
        pending: list[RecordingMappingAudit] = []
        for mapping_row in mapping.recordings:
            directory = output / mapping_row.episode_id
            report_path = directory / "report.json"
            if report_path.is_file():
                report = RecordingCommit.model_validate_json(report_path.read_text()).report
                if report.run_hash != run_hash:
                    raise ValueError("Completed recording belongs to another frozen run")
                if any(
                    file_sha256(directory / name) != checksum
                    for name, checksum in report.artifacts_sha256.items()
                ):
                    raise ValueError("Completed QC artifacts changed since their committed report")
                rows.append(report)
            elif directory.exists():
                raise ValueError("Incomplete recording requires a new explicit recovery output")
            else:
                pending.append(mapping_row)
        if limit_recordings is not None:
            pending = pending[:limit_recordings]
        reserve = len(pending) * MAX_RECORDING_OUTPUT_BYTES
        used = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if reserve + used > config.max_new_output_bytes:
            raise RuntimeError("Semantic pilot would exceed its approved new-output allocation")
        check_storage(reserve)
        failure_rows = []
        with ProcessPoolExecutor(
            max_workers=config.workers, mp_context=get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(
                    _recording_job,
                    indexed[row.episode_id],
                    row,
                    config,
                    run_hash,
                    output / row.episode_id,
                ): row
                for row in pending
            }
            for future in as_completed(futures):
                mapping_row = futures[future]
                try:
                    report = future.result()
                except (ValueError, OSError, RuntimeError, cv2.error) as error:
                    LOGGER.error(
                        "Recording %s failed explicitly: %s", mapping_row.episode_id, error
                    )
                    failure_rows.append(
                        {
                            "episode_id": mapping_row.episode_id,
                            "error_type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                else:
                    rows.append(report)
        rows.sort(key=lambda row: row.source_name)
        flat = [
            {
                "episode_id": row.episode_id,
                "source_group": row.source_group,
                "source_name": row.source_name,
                "role": row.role,
                **profile.model_dump(mode="json"),
            }
            for row in rows
            for profile in row.profiles
        ]
        if flat:
            with (output / "per_recording.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
                writer.writeheader()
                writer.writerows(flat)
        held_out = [row for row in rows if row.role == "held_out"]
        aggregate = []
        fields = tuple(PipelineQuality.model_fields)
        for pipeline in pipelines:
            profiles = [
                profile
                for row in held_out
                for profile in row.profiles
                if profile.pipeline == pipeline.name
            ]
            if not profiles:
                continue
            aggregate.append(
                {
                    "pipeline": pipeline.name,
                    "recording_groups": len(profiles),
                    "frames": sum(profile.frames for profile in profiles),
                    **{
                        field: float(np.mean([getattr(profile, field) for profile in profiles]))
                        for field in fields
                        if field
                        not in (
                            "pipeline",
                            "frames",
                            "minimum_detected_balls",
                            "maximum_detected_balls",
                            "manual_foreground_pixels",
                            "predicted_foreground_pixels",
                        )
                    },
                }
            )
        paired = {}
        for pipeline in pipelines[1:]:
            pairs = []
            for row in held_out:
                profile_by_name = {profile.pipeline: profile for profile in row.profiles}
                pairs.append(
                    PairedMeasurement(
                        pair_id=row.episode_id,
                        group_id=row.source_group,
                        a=profile_by_name[pipeline.name].path_iou,
                        b=profile_by_name[reference_pipeline].path_iou,
                    )
                )
            if pairs:
                paired[pipeline.name] = summarize_paired(
                    tuple(pairs), seed=config.bootstrap_seed
                ).model_dump(mode="json")
        write_json(output / "failures.json", failure_rows)
        output_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if output_bytes > config.max_new_output_bytes:
            raise RuntimeError("New output allocation was exhausted")
        summary: dict[str, object] = {
            "status": "failed"
            if failure_rows
            else "complete"
            if len(rows) == len(mapping.recordings)
            else "partial",
            "run_hash": run_hash,
            "completed_recordings": len(rows),
            "failed_recordings": len(failure_rows),
            "new_recordings_this_invocation": len(pending) - len(failure_rows),
            "support_recordings": sum(row.role == "support" for row in rows),
            "evaluation_recordings": len(held_out),
            "independent_fps_clock_validation": False,
            "evaluation_original_frames": sum(row.frames for row in held_out),
            "new_output_bytes": output_bytes,
            "actual_model_requests": 0,
            "human_annotations_added": 0,
            "g1_approved": False,
            "metrics_interpretation": (
                "Spatial raster and automatic-consistency diagnostics, "
                "not detection/identity accuracy or measured cue actions."
            ),
            "aggregate_macro_by_recording": aggregate,
            "paired_path_iou_differences": paired,
            "source_payloads_modified": False,
        }
        write_json(output / "summary.json", summary)
        return summary
