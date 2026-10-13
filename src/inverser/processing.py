import csv
import json
import logging
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from inverser.runtime import PROJECT_ROOT, check_storage, file_sha256, fingerprint, write_json
from inverser.schemas import Record
from inverser.sources import probe_video
from inverser.vision import (
    BallAssociator,
    FrameObservation,
    ImageArray,
    TableProjection,
    draw_observations,
    observe_frame,
)

LOGGER = logging.getLogger(__name__)


class CandidateSummary(Record):
    candidate_id: str
    start_s: float
    end_s: float
    observations: str
    clip: str
    warnings: list[str]
    initial_detected_balls: int
    goal_detected_balls: int
    benchmark_accepted: Literal[False] = False


class WindowReport(Record):
    status: Literal["complete", "failed"]
    config_hash: str
    source: str
    window_start_s: float
    window_duration_s: float
    sample_hz: float
    sampled_frames: int
    frame_quality_counts: dict[str, int]
    ball_detection_counts: dict[str, int]
    candidate_exports: int
    candidate_index: list[CandidateSummary]
    segment_rejections: dict[str, int]
    unfinished_motion_window: bool
    label_tier: Literal["silver_unverified"] = "silver_unverified"
    accepted_benchmark_tasks: Literal[0] = 0
    full_inventory_validation: str = "not performed"
    calibration_error: str = "not measured; planar projection only"
    model_or_api_calls: Literal[0] = 0


def decode_window(
    video: Path,
    *,
    start_s: float,
    duration_s: float,
    sample_hz: float,
    width: int,
    height: int,
    scale_width: int | None = None,
) -> Iterator[tuple[float, ImageArray]]:
    filters = f"fps={sample_hz}"
    if scale_width is not None:
        if not 256 <= scale_width <= width:
            raise ValueError("Decode scale width must be 256 pixels up to the source width")
        height = 2 * round(height * scale_width / width / 2)
        width = scale_width
        filters += f",scale={width}:{height}"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        "2",
        "-filter_threads",
        "1",
        "-ss",
        str(start_s),
        "-noautorotate",
        "-i",
        str(video),
        "-t",
        str(duration_s),
        "-an",
        "-vf",
        filters,
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-",
    ]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        if process.stdout is None:
            raise RuntimeError("FFmpeg did not expose decoded frames")
        frame_bytes = width * height * 3
        index = 0
        try:
            while True:
                chunk = process.stdout.read(frame_bytes)
                if not chunk:
                    break
                if len(chunk) != frame_bytes:
                    raise ValueError("FFmpeg returned a truncated raw frame")
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3))
                yield start_s + index / sample_hz, frame
                index += 1
            result = process.wait(timeout=60)
            errors.seek(0)
            if result:
                raise RuntimeError(
                    f"Video decoding failed: {errors.read(4000).decode(errors='replace')}"
                )
            if index < duration_s * sample_hz * 0.9:
                raise ValueError("Decoded window is shorter than its declared sampling interval")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


@dataclass
class Snapshot:
    observation: FrameObservation
    source_frame: ImageArray
    rectified: ImageArray


@dataclass
class Candidate:
    initial: Snapshot
    goal: Snapshot
    trace: tuple[FrameObservation, ...]
    motion_onset_s: float | None = None


class ShotWindowDetector:
    def __init__(
        self, ball_radius_m: float, *, rest_seconds: float = 1.0, cue_trigger_only: bool = False
    ) -> None:
        self.radius = ball_radius_m
        self.rest_seconds = rest_seconds
        self.still_since: float | None = None
        self.last_still_time: float | None = None
        self.anchor: Snapshot | None = None
        self.active: Snapshot | None = None
        self.trace: list[FrameObservation] = []
        self.rejections: Counter[str] = Counter()
        self.motion_onset_s: float | None = None
        self.cue_trigger_only = cue_trigger_only

    def update(self, snapshot: Snapshot) -> Candidate | None:
        observation = snapshot.observation
        time_s = observation.time_s
        if not observation.view_valid:
            if self.active:
                self.rejections["view_or_camera_change_during_motion"] += 1
            self.active = self.anchor = None
            self.trace = []
            self.still_since = None
            self.last_still_time = None
            return None
        motion = observation.max_matched_motion_m
        stationary = (
            observation.visibility_clear
            and observation.cue_motion_m is not None
            and observation.cue_motion_m <= self.radius * 0.25
            and observation.median_matched_motion_m is not None
            and observation.median_matched_motion_m <= self.radius * 0.25
            and motion is not None
            and motion <= self.radius * 0.5
            and observation.matched_ball_count >= min(3, len(observation.detections))
        )
        if stationary:
            if self.still_since is None:
                self.still_since = time_s
            self.last_still_time = time_s
        elif observation.visibility_clear or (
            self.last_still_time is None or time_s - self.last_still_time > 0.4
        ):
            self.still_since = None
        if self.active is not None:
            self.trace.append(observation)
            if time_s - self.active.observation.time_s > 60:
                self.rejections["motion_window_exceeds_60_seconds"] += 1
                self.active = self.anchor = None
                self.trace = []
                return None
            if (
                stationary
                and self.still_since is not None
                and time_s - self.still_since >= self.rest_seconds
            ):
                candidate = Candidate(self.active, snapshot, tuple(self.trace), self.motion_onset_s)
                self.active = None
                self.trace = []
                self.anchor = snapshot
                return candidate
        elif (
            stationary
            and self.still_since is not None
            and time_s - self.still_since >= self.rest_seconds
        ):
            self.anchor = snapshot
        elif (
            self.anchor is not None
            and observation.visibility_clear
            and motion is not None
            and motion > self.radius * 0.4
            and (
                not self.cue_trigger_only
                or (
                    observation.cue_motion_m is not None
                    and observation.cue_motion_m > self.radius * 0.4
                )
            )
            and time_s - self.anchor.observation.time_s <= 15
        ):
            self.active = self.anchor
            self.motion_onset_s = time_s
            self.trace = [self.anchor.observation, observation]
        return None


def write_image(path: Path, image: ImageArray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Image export failed: {path}")


def export_candidate(
    candidate: Candidate,
    video: Path,
    output: Path,
    index: int,
    source_hash: str,
    *,
    source_duration_s: float | None = None,
) -> CandidateSummary:
    name = f"candidate_{index:04d}"
    initial, goal = candidate.initial.observation, candidate.goal.observation
    if any(ball.track_id is None for ball in (*initial.detections, *goal.detections)):
        raise ValueError("Endpoint detection lacks an observed local track identity")
    initial_ids = {ball.track_id for ball in initial.detections if ball.track_id is not None}
    final_ids = {ball.track_id for ball in goal.detections if ball.track_id is not None}
    missing = initial_ids - final_ids
    introduced = final_ids - initial_ids
    warnings = ["automatic_colour_and_shape_detector_not_independently_validated"]
    if missing or introduced:
        warnings.append("identity_or_visibility_changes_not_automatically_classified_as_potting")
    if initial.unresolved_components or goal.unresolved_components:
        warnings.append("unresolved_foreground_or_touching_ball_clusters_at_endpoint")
    if any(not observation.visibility_clear for observation in candidate.trace):
        warnings.append("intermediate_occlusion_or_detection_quality_gap")
    tracks: dict[str, dict[str, object]] = {}
    for observation in candidate.trace:
        for ball in observation.detections:
            if ball.track_id is None:
                raise ValueError("Tracked candidate observation has no local identity")
            if ball.track_id not in tracks:
                tracks[ball.track_id] = {"kind": ball.kind, "samples": []}
            samples = tracks[ball.track_id]["samples"]
            if not isinstance(samples, list):
                raise ValueError("Invalid internal track samples")
            samples.append({"time_s": observation.time_s, "xy_m": ball.xy_m})
    record = {
        "schema_version": "real-shot-candidate-v0",
        "candidate_id": f"{video.stem}-{name}",
        "source_file": str(video),
        "source_sha256": source_hash,
        "source_kind": "real_video",
        "label_tier": "silver_unverified",
        "partition": "development",
        "start_s": initial.time_s,
        "end_s": goal.time_s,
        "initial": initial.json(),
        "goal": goal.json(),
        "tracks": tracks,
        "missing_initial_track_ids": sorted(missing),
        "new_terminal_track_ids": sorted(introduced),
        "warnings": warnings,
        "full_ball_inventory_verified": False,
        "single_cue_strike_verified": False,
        "benchmark_accepted": False,
        "timestamps": "FFmpeg fps-filter sample grid; not measured cue-impact timestamps",
        "projection": "planar homography; ball-height and lens effects uncorrected",
        "angular_velocity_or_cue_force_labels": None,
    }
    directory = output / "candidates" / name
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "observations.json", record)
    write_image(directory / "initial_source.jpg", candidate.initial.source_frame)
    write_image(directory / "goal_source.jpg", candidate.goal.source_frame)
    pair = np.hstack(
        (
            draw_observations(candidate.initial.rectified, initial),
            draw_observations(candidate.goal.rectified, goal),
        )
    )
    write_image(directory / "endpoint_pair.jpg", pair)
    clip_start = max(0.0, initial.time_s - 0.5)
    clip_end = goal.time_s + 0.5
    if source_duration_s is not None:
        clip_end = min(clip_end, source_duration_s)
    clip_duration = clip_end - clip_start
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "2",
            "-filter_threads",
            "1",
            "-ss",
            str(clip_start),
            "-i",
            str(video),
            "-t",
            str(clip_duration),
            "-an",
            "-vf",
            "scale='min(1280,iw)':-2",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-threads",
            "2",
            "-movflags",
            "+faststart",
            "-y",
            str(directory / "clip.mp4"),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    clip = probe_video(directory / "clip.mp4")
    if abs(clip.duration_s - clip_duration) > 0.2:
        raise ValueError("Candidate clip duration does not match its requested source interval")
    write_json(
        directory / "clip_reference.json",
        {
            "source_start_s": clip_start,
            "source_duration_s": clip_duration,
            "source_end_s": clip_end,
            "audio_included": False,
            "video": clip.model_dump(mode="json"),
        },
    )
    return CandidateSummary(
        candidate_id=f"{video.stem}-{name}",
        start_s=initial.time_s,
        end_s=goal.time_s,
        observations=str(directory / "observations.json"),
        clip=str(directory / "clip.mp4"),
        warnings=warnings,
        initial_detected_balls=len(initial.detections),
        goal_detected_balls=len(goal.detections),
    )


def process_video_window(
    source: Path,
    filename: str,
    output: Path,
    *,
    duration_s: float = 600,
    start_fraction: float = 0.05,
    sample_hz: float = 5,
    max_candidates: int = 12,
) -> WindowReport:
    if not (0 < duration_s <= 900 and 0 <= start_fraction < 1 and 2 <= sample_hz <= 15):
        raise ValueError("Pilot requires <=900 seconds per window and 2-15 sampled fps")
    if not 1 <= max_candidates <= 50:
        raise ValueError("Pilot candidate export limit must be 1-50 per video")
    if Path(filename).name != filename or not filename.endswith(".mp4"):
        raise ValueError("Expected a video filename, not an arbitrary path")
    source = source.resolve()
    output = output.resolve()
    if not output.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Processing output must stay in the project")
    check_storage(3_000_000_000)
    video = source / "files" / filename
    corners = source / "files" / f"{video.stem}_corners.json"
    if not corners.is_file():
        raise ValueError(f"No source corner sidecar for {filename}")
    inventory = json.loads((source / "inventory.json").read_text())["files"]
    record = next((item for item in inventory if item["path"] == filename), None)
    if record is None or video.stat().st_size != record["bytes"] or not record["checksum_verified"]:
        raise ValueError("Video does not match the verified acquisition inventory")
    video_info = probe_video(video)
    if sample_hz > video_info.fps:
        raise ValueError("Sample rate cannot exceed source frame rate")
    start_s = video_info.duration_s * start_fraction
    duration_s = min(duration_s, video_info.duration_s - start_s)
    projection = TableProjection(
        json.loads(corners.read_text()), video_info.width, video_info.height
    )
    source_files = ("vision.py", "processing.py", "runtime.py")
    config = {
        "source": str(video),
        "source_sha256": record["sha256"],
        "source_stat": [video.stat().st_size, video.stat().st_mtime_ns],
        "corners_sha256": file_sha256(corners),
        "start_s": start_s,
        "duration_s": duration_s,
        "sample_hz": sample_hz,
        "max_candidates": max_candidates,
        "opencv": cv2.__version__,
        "source_code_hashes": {
            name: file_sha256(Path(__file__).with_name(name)) for name in source_files
        },
        "calibration": projection.metadata(),
        "local_processing_scope": "researcher-requested bounded quality pilot, no training",
    }
    config_hash = fingerprint(config)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists():
        previous = WindowReport.model_validate_json(report_path.read_text())
        if previous.status == "complete" and previous.config_hash == config_hash:
            return previous
        raise ValueError(
            "Existing processing run differs or is incomplete; use a new output version"
        )
    write_json(output / "config.json", config)
    cv2.setNumThreads(1)
    associator = BallAssociator()
    detector = ShotWindowDetector(projection.physics.ball_radius_m)
    flags: Counter[str] = Counter()
    detection_counts: Counter[str] = Counter()
    candidates: list[CandidateSummary] = []
    frame_count = 0
    observation_path = output / "frame_observations.jsonl"
    if observation_path.exists():
        raise ValueError("Existing partial observation stream; use a new run directory")
    complete = False
    try:
        with observation_path.open("w") as observations:
            for time_s, frame in decode_window(
                video,
                start_s=start_s,
                duration_s=duration_s,
                sample_hz=sample_hz,
                width=video_info.width,
                height=video_info.height,
            ):
                observation, rectified = observe_frame(frame, projection, associator, time_s)
                observations.write(json.dumps(observation.json(), allow_nan=False) + "\n")
                frame_count += 1
                flags["table_view_valid"] += observation.view_valid
                flags["visibility_clear"] += observation.visibility_clear
                flags["possible_camera_motion"] += observation.possible_camera_motion
                flags["unresolved_components"] += observation.unresolved_components > 0
                detection_counts.update(ball.kind for ball in observation.detections)
                if frame_count == 1:
                    write_image(
                        output / "calibration_preview.jpg",
                        draw_observations(rectified, observation),
                    )
                snapshot = Snapshot(observation, frame, rectified)
                candidate = detector.update(snapshot)
                if candidate is not None and len(candidates) < max_candidates:
                    candidates.append(
                        export_candidate(
                            candidate, video, output, len(candidates), record["sha256"]
                        )
                    )
                if frame_count % 100 == 0:
                    write_json(
                        output / "processing_state.json",
                        {
                            "status": "scanning",
                            "frames": frame_count,
                            "source_time_s": time_s,
                            "candidate_exports": len(candidates),
                        },
                    )
            complete = True
    finally:
        report = WindowReport(
            status="complete" if complete else "failed",
            config_hash=config_hash,
            source=filename,
            window_start_s=start_s,
            window_duration_s=duration_s,
            sample_hz=sample_hz,
            sampled_frames=frame_count,
            frame_quality_counts=dict(flags),
            ball_detection_counts=dict(detection_counts),
            candidate_exports=len(candidates),
            candidate_index=candidates,
            segment_rejections=dict(detector.rejections),
            unfinished_motion_window=detector.active is not None,
        )
        write_json(report_path, report.model_dump(mode="json"))
        write_json(output / "processing_state.json", report.model_dump(mode="json"))
    return report


def process_video_pilot(
    source: Path,
    output: Path,
    filenames: list[str],
    *,
    duration_s: float = 600,
    start_fraction: float = 0.05,
    sample_hz: float = 5,
    max_candidates: int = 12,
) -> dict[str, object]:
    if not filenames or len(filenames) > 3 or len(set(filenames)) != len(filenames):
        raise ValueError("This G1 pilot accepts 1-3 distinct videos")
    output.mkdir(parents=True, exist_ok=True)
    reports: list[WindowReport] = []
    for filename in filenames:
        LOGGER.info("Scanning %s for candidate stationary/motion/stationary intervals", filename)
        reports.append(
            process_video_window(
                source,
                filename,
                output / Path(filename).stem,
                duration_s=duration_s,
                start_fraction=start_fraction,
                sample_hz=sample_hz,
                max_candidates=max_candidates,
            )
        )
    summary: dict[str, object] = {
        "status": "complete",
        "videos": len(reports),
        "sampled_seconds": sum(report.window_duration_s for report in reports),
        "candidate_exports": sum(report.candidate_exports for report in reports),
        "accepted_benchmark_tasks": 0,
        "label_tier": "silver_unverified",
        "reports": [report.model_dump(mode="json") for report in reports],
        "g1": "requires evidence review; no automatic acceptance",
    }
    write_json(output / "summary.json", summary)
    with (output / "candidate_index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["video", "candidate_id", "start_s", "end_s", "observations", "clip"],
        )
        writer.writeheader()
        for report in reports:
            for candidate in report.candidate_index:
                values = candidate.model_dump(mode="json")
                writer.writerow(
                    {"video": report.source}
                    | {key: values[key] for key in writer.fieldnames or () if key != "video"}
                )
    return summary
