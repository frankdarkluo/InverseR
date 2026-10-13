import csv
import fcntl
import json
import logging
import math
import sqlite3
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw

from inverser.real_data import ObservedState, SnookerEpisode
from inverser.runtime import PROJECT_ROOT, check_storage, file_sha256, fingerprint, write_json
from inverser.schemas import Record
from inverser.sources import probe_video

LOGGER = logging.getLogger(__name__)
Disposition = Literal[
    "priority_review", "needs_reprocessing", "not_suitable", "probable_duplicate", "invalid"
]


class ScreeningPolicy(Record):
    version: Literal["automatic-quality-v2"] = "automatic-quality-v2"
    min_duration_s: float = 2.0
    max_priority_duration_s: float = 45.0
    min_visible_fraction: float = 0.90
    min_endpoint_shape_score: float = 0.55
    min_track_retention: float = 0.80
    min_cue_excursion_radii: float = 2.0
    min_cue_peak_speed_mps: float = 0.20
    max_terminal_step_radii: float = 0.50
    required_terminal_still_s: float = 0.80
    min_motion_burst_s: float = 0.40
    motion_burst_separation_s: float = 1.0
    min_felt_fraction: float = 0.70
    max_camera_corner_shift_fraction: float = 0.01
    min_camera_matches: int = 12
    min_camera_inlier_fraction: float = 0.50
    duplicate_time_tolerance_s: float = 1.0
    duplicate_image_hash_distance: int = 4


class ScreeningMetrics(Record):
    duration_s: float
    visible_frame_fraction: float
    endpoint_min_balls: int
    duplicate_unique_colours: bool
    unresolved_endpoint_components: int
    endpoint_shape_min: float
    endpoint_track_retention: float
    connected_cue_track: bool
    cue_endpoint_shift_m: float | None
    cue_excursion_radii: float | None
    cue_peak_speed_mps: float | None
    cue_motion_bursts: int | None
    cue_terminal_still_s: float | None
    retained_object_balls_at_rest: bool
    calibration_internal_error_px: float
    endpoint_felt_fraction: float | None = None
    camera_matches: int = 0
    camera_inlier_fraction: float | None = None
    camera_corner_shift_fraction: float | None = None
    clip_duration_error_s: float | None = None


class ScreeningDecision(Record):
    episode_id: str
    recording_id: str
    group_id: str
    record_path: str
    record_sha256: str
    start_s: float
    onset_s: float | None
    terminal_s: float
    disposition: Disposition
    reasons: tuple[str, ...]
    metrics: ScreeningMetrics | None
    initial_image: str
    terminal_image: str
    initial_bev: str
    terminal_bev: str
    source_clip: str | None
    initial_perceptual_hash: str | None = None
    terminal_perceptual_hash: str | None = None
    duplicate_of: str | None = None
    benchmark_accepted: Literal[False] = False
    independent_accuracy_verified: Literal[False] = False


class CameraComparison(Record):
    matches: int = 0
    inlier_fraction: float | None = None
    corner_shift_fraction: float | None = None
    reason: str | None = None


def compare_camera(
    initial: np.ndarray, terminal: np.ndarray, corners: np.ndarray, policy: ScreeningPolicy
) -> CameraComparison:
    if initial.shape != terminal.shape:
        return CameraComparison(reason="source_image_dimensions_changed")
    height, width = initial.shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 0)
    mask[round(height * 0.88) :] = 0
    orb = cv2.ORB.create(nfeatures=600)
    key_a, descriptors_a = orb.detectAndCompute(initial, mask)
    key_b, descriptors_b = orb.detectAndCompute(terminal, mask)
    if descriptors_a is None or descriptors_b is None:
        return CameraComparison(reason="camera_background_features_unavailable")
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors_a, descriptors_b, k=2)
    good = [
        pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good) < policy.min_camera_matches:
        return CameraComparison(matches=len(good), reason="too_few_camera_background_matches")
    source = np.asarray([key_a[item.queryIdx].pt for item in good], dtype=np.float32)
    target = np.asarray([key_b[item.trainIdx].pt for item in good], dtype=np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=1000
    )
    if matrix is None or inliers is None:
        return CameraComparison(matches=len(good), reason="camera_transform_unresolved")
    fraction = float(np.mean(inliers))
    if fraction < policy.min_camera_inlier_fraction:
        return CameraComparison(
            matches=len(good),
            inlier_fraction=fraction,
            reason="camera_transform_low_inlier_fraction",
        )
    transformed = corners @ matrix[:, :2].T + matrix[:, 2]
    shift = float(np.max(np.linalg.norm(transformed - corners, axis=1)) / math.hypot(width, height))
    return CameraComparison(
        matches=len(good), inlier_fraction=fraction, corner_shift_fraction=shift
    )


def perceptual_hash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(small)[:8, :8].ravel()[1:]
    bits = coefficients > np.median(coefficients)
    return f"{sum(int(value) << index for index, value in enumerate(bits)):016x}"


def _endpoint_counts(state: ObservedState) -> Counter[str]:
    return Counter(ball.kind for ball in state.balls)


def _motion_metrics(
    episode: SnookerEpisode, policy: ScreeningPolicy
) -> tuple[bool, float | None, float | None, float | None, int | None, float | None]:
    first = [ball for ball in episode.initial.balls if ball.kind == "cue"]
    last = [ball for ball in episode.terminal.balls if ball.kind == "cue"]
    if len(first) != 1 or len(last) != 1:
        return False, None, None, None, None, None
    shift = math.dist(first[0].xy_m, last[0].xy_m)
    if first[0].track_id is None or first[0].track_id != last[0].track_id:
        return False, shift, None, None, None, None
    trajectory = next(
        (track for track in episode.trajectories if track.track_id == first[0].track_id), None
    )
    if trajectory is None or len(trajectory.points) < 2:
        return False, shift, None, None, None, None
    points = trajectory.points
    if (
        points[0].time_s != episode.start_s
        or points[-1].time_s != episode.terminal_s
        or any(b.time_s - a.time_s > 1.5 / episode.sample_hz for a, b in zip(points, points[1:]))
    ):
        return False, shift, None, None, None, None
    radius = episode.calibration.ball_radius_m
    excursion = max(math.dist(point.xy_m, first[0].xy_m) for point in points) / radius
    steps = [(a.time_s, b.time_s, math.dist(a.xy_m, b.xy_m)) for a, b in zip(points, points[1:])]
    speeds = [distance / (end - start) for start, end, distance in steps]
    bursts = 0
    active_start: float | None = None
    active_end: float | None = None
    for (start, end, _), speed in zip(steps, speeds, strict=True):
        if speed >= policy.min_cue_peak_speed_mps:
            if active_start is None:
                active_start = start
            elif active_end is not None and start - active_end >= policy.motion_burst_separation_s:
                bursts += active_end - active_start >= policy.min_motion_burst_s - 1e-8
                active_start = start
            active_end = end
    if active_start is not None and active_end is not None:
        bursts += active_end - active_start >= policy.min_motion_burst_s - 1e-8
    tail = 0.0
    for start, end, distance in reversed(steps):
        if distance > policy.max_terminal_step_radii * radius:
            break
        tail += end - start
    return True, shift, excursion, max(speeds, default=0), bursts, tail


def metadata_metrics(episode: SnookerEpisode, policy: ScreeningPolicy) -> ScreeningMetrics:
    width, length = episode.calibration.table_size_m
    radius = episode.calibration.ball_radius_m
    if min(width, length, radius) <= 0:
        raise ValueError("Invalid physical table or ball dimensions")
    pixel_w, pixel_h = episode.calibration.bev_size_px
    if min(pixel_w, pixel_h) < 2:
        raise ValueError("Invalid BEV image dimensions")
    matrix = np.asarray(episode.calibration.homography, dtype=np.float64)
    corners = np.asarray(episode.calibration.source_corners_px, dtype=np.float64)
    projected = np.column_stack((corners, np.ones(4))) @ matrix.T
    if np.any(np.abs(projected[:, 2]) < 1e-12):
        raise ValueError("Invalid calibration homogeneous coordinates")
    projected = projected[:, :2] / projected[:, 2, None]
    target = np.asarray([[0, 0], [pixel_w - 1, 0], [pixel_w - 1, pixel_h - 1], [0, pixel_h - 1]])
    calibration_error = float(np.max(np.linalg.norm(projected - target, axis=1)))
    if calibration_error > 0.05:
        raise ValueError("Stored calibration does not reproduce its own corner contract")
    for state in (episode.initial, episode.terminal):
        if len(state.balls) > 22:
            raise ValueError("Endpoint inventory exceeds the snooker maximum")
        for ball in state.balls:
            x, y = ball.xy_m
            if not -radius <= x <= width + radius or not -radius <= y <= length + radius:
                raise ValueError("Observed endpoint position lies outside the table")
            expected = (
                ball.bev_xy_px[0] / (pixel_w - 1) * width,
                (1 - ball.bev_xy_px[1] / (pixel_h - 1)) * length,
            )
            if math.dist(expected, ball.xy_m) > 1e-6:
                raise ValueError("Stored metric and pixel endpoint positions disagree")
            if ball.track_id is not None:
                track = next(
                    (item for item in episode.trajectories if item.track_id == ball.track_id), None
                )
                if track is None or not any(
                    abs(point.time_s - state.time_s) < 1e-8
                    and math.dist(point.xy_m, ball.xy_m) < 1e-8
                    for point in track.points
                ):
                    raise ValueError("Endpoint observation is absent from its declared track")
    first_ids = {ball.track_id for ball in episode.initial.balls if ball.track_id is not None}
    last_ids = {ball.track_id for ball in episode.terminal.balls if ball.track_id is not None}
    duplicates = any(
        any(
            count > 1
            for kind, count in _endpoint_counts(state).items()
            if kind not in ("red", "unknown")
        )
        for state in (episode.initial, episode.terminal)
    )
    connected, shift, excursion, speed, bursts, tail = _motion_metrics(episode, policy)
    object_rest = True
    for ball in episode.terminal.balls:
        if ball.kind == "cue" or ball.track_id not in first_ids:
            continue
        trajectory = next(
            (track for track in episode.trajectories if track.track_id == ball.track_id), None
        )
        if trajectory is None:
            raise ValueError("Terminal object ball has no declared observed trajectory")
        recent = [
            point
            for point in trajectory.points
            if point.time_s >= episode.terminal_s - policy.required_terminal_still_s - 1e-8
        ]
        if (
            len(recent) < 2
            or episode.terminal_s - recent[0].time_s < policy.required_terminal_still_s - 1e-8
            or any(
                b.time_s - a.time_s > 1.5 / episode.sample_hz
                or math.dist(a.xy_m, b.xy_m) > policy.max_terminal_step_radii * radius
                for a, b in zip(recent, recent[1:])
            )
        ):
            object_rest = False
    return ScreeningMetrics(
        duration_s=episode.terminal_s - episode.start_s,
        visible_frame_fraction=episode.quality.visible_frame_fraction,
        endpoint_min_balls=min(len(episode.initial.balls), len(episode.terminal.balls)),
        duplicate_unique_colours=duplicates,
        unresolved_endpoint_components=episode.initial.unresolved_components
        + episode.terminal.unresolved_components,
        endpoint_shape_min=min(
            (
                ball.heuristic_shape_score
                for ball in (*episode.initial.balls, *episode.terminal.balls)
            ),
            default=0,
        ),
        endpoint_track_retention=len(first_ids & last_ids) / max(len(first_ids | last_ids), 1),
        connected_cue_track=connected,
        cue_endpoint_shift_m=shift,
        cue_excursion_radii=excursion,
        cue_peak_speed_mps=speed,
        cue_motion_bursts=bursts,
        cue_terminal_still_s=tail,
        retained_object_balls_at_rest=object_rest,
        calibration_internal_error_px=calibration_error,
    )


def decide_quality(
    metrics: ScreeningMetrics, policy: ScreeningPolicy
) -> tuple[Disposition, tuple[str, ...]]:
    if metrics.duration_s < policy.min_duration_s:
        return "not_suitable", ("too_short_for_this_single_shot_protocol",)
    if metrics.connected_cue_track and (
        metrics.cue_excursion_radii is None
        or metrics.cue_excursion_radii < policy.min_cue_excursion_radii
        or metrics.cue_peak_speed_mps is None
        or metrics.cue_peak_speed_mps < policy.min_cue_peak_speed_mps
    ):
        return "not_suitable", ("no_resolved_cue_motion_above_noise_floor",)
    reasons = []
    if not metrics.connected_cue_track:
        reasons.append("cue_identity_or_continuity_unresolved")
    if metrics.duplicate_unique_colours:
        reasons.append("duplicate_unique_ball_colours")
    if metrics.unresolved_endpoint_components:
        reasons.append("unresolved_endpoint_components")
    if metrics.endpoint_shape_min < policy.min_endpoint_shape_score:
        reasons.append("weak_endpoint_shape_evidence")
    if metrics.endpoint_track_retention < policy.min_track_retention:
        reasons.append("endpoint_identity_churn")
    if metrics.visible_frame_fraction < policy.min_visible_fraction:
        reasons.append("insufficient_visible_observations")
    if metrics.duration_s > policy.max_priority_duration_s:
        reasons.append("long_window_needs_multiple_shot_or_intervention_check")
    if (
        metrics.cue_terminal_still_s is None
        or metrics.cue_terminal_still_s < policy.required_terminal_still_s - 1e-8
    ):
        reasons.append("terminal_cue_rest_unconfirmed")
    if metrics.cue_motion_bursts != 1:
        reasons.append("single_cue_motion_burst_unresolved")
    if not metrics.retained_object_balls_at_rest:
        reasons.append("terminal_object_ball_rest_unconfirmed")
    if (
        metrics.endpoint_felt_fraction is not None
        and metrics.endpoint_felt_fraction < policy.min_felt_fraction
    ):
        reasons.append("endpoint_table_view_mismatch")
    if metrics.camera_corner_shift_fraction is None:
        reasons.append("camera_background_consistency_unresolved")
    elif metrics.camera_corner_shift_fraction > policy.max_camera_corner_shift_fraction:
        reasons.append("camera_moved_between_endpoints")
    return ("needs_reprocessing", tuple(reasons)) if reasons else ("priority_review", ())


def _read_image(path: str, root: Path) -> np.ndarray:
    image_path = Path(path)
    if not image_path.resolve().is_relative_to(root) or image_path.is_symlink():
        raise ValueError("Media reference escapes project data or is a symlink")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Image cannot be decoded: {image_path.name}")
    return np.asarray(image, dtype=np.uint8)


def _felt_fraction(image: np.ndarray, radius_px: int) -> float:
    border = max(2, radius_px)
    interior = image[border:-border, border:-border]
    if not interior.size:
        raise ValueError("BEV image has no valid table interior")
    hsv = cv2.cvtColor(interior, cv2.COLOR_BGR2HSV)
    return float(
        np.mean(
            (hsv[:, :, 0] >= 30)
            & (hsv[:, :, 0] <= 100)
            & (hsv[:, :, 1] >= 40)
            & (hsv[:, :, 2] >= 35)
        )
    )


def screen_episode(
    payload: str, path: str, policy: ScreeningPolicy, data_root: Path
) -> ScreeningDecision:
    episode = SnookerEpisode.model_validate_json(payload)
    record = Path(path)
    if not record.resolve().is_relative_to(data_root) or record.is_symlink():
        raise ValueError("Structured record escapes project data")
    disk = SnookerEpisode.model_validate_json(record.read_text())
    if disk != episode:
        raise ValueError("Structured file differs from frozen SQLite payload")
    values = {
        "episode_id": episode.episode_id,
        "recording_id": episode.source.recording_id,
        "group_id": episode.source.group_id,
        "record_path": path,
        "record_sha256": file_sha256(record),
        "start_s": episode.start_s,
        "onset_s": episode.estimated_motion_onset_s,
        "terminal_s": episode.terminal_s,
        "initial_image": episode.media.initial_source,
        "terminal_image": episode.media.terminal_source,
        "initial_bev": episode.media.initial_bev,
        "terminal_bev": episode.media.terminal_bev,
        "source_clip": episode.media.source_clip,
    }
    try:
        metrics = metadata_metrics(episode, policy)
        initial = _read_image(episode.media.initial_source, data_root)
        terminal = _read_image(episode.media.terminal_source, data_root)
        width, height = episode.calibration.image_size_px
        if initial.shape[:2] != (height, width) or terminal.shape[:2] != (height, width):
            raise ValueError("Endpoint source images disagree with calibration resolution")
        bev_first = _read_image(episode.media.initial_bev, data_root)
        bev_last = _read_image(episode.media.terminal_bev, data_root)
        pixel_w, pixel_h = episode.calibration.bev_size_px
        if bev_first.shape[:2] != (pixel_h, pixel_w) or bev_last.shape[:2] != (pixel_h, pixel_w):
            raise ValueError("BEV media disagree with calibration resolution")
        clip = episode.media.source_clip
        if (
            clip is None
            or not Path(clip).resolve().is_relative_to(data_root)
            or Path(clip).is_symlink()
        ):
            raise ValueError("Missing or invalid source clip")
        video_info = probe_video(Path(clip))
        expected_duration = episode.media.source_clip_end_s - episode.media.source_clip_start_s
        duration_error = abs(video_info.duration_s - expected_duration)
        if duration_error > max(0.2, 2 / video_info.fps):
            raise ValueError("Clip duration does not match its declared source interval")
        camera = compare_camera(
            initial, terminal, np.asarray(episode.calibration.source_corners_px), policy
        )
        radius_px = round(
            episode.calibration.ball_radius_m / episode.calibration.table_size_m[0] * pixel_w
        )
        metrics = metrics.model_copy(
            update={
                "endpoint_felt_fraction": min(
                    _felt_fraction(bev_first, radius_px), _felt_fraction(bev_last, radius_px)
                ),
                "camera_matches": camera.matches,
                "camera_inlier_fraction": camera.inlier_fraction,
                "camera_corner_shift_fraction": camera.corner_shift_fraction,
                "clip_duration_error_s": duration_error,
            }
        )
        disposition, reasons = decide_quality(metrics, policy)
        if camera.reason and "camera_background_consistency_unresolved" in reasons:
            reasons += (camera.reason,)
        return ScreeningDecision.model_validate(
            values
            | {
                "disposition": disposition,
                "reasons": reasons,
                "metrics": metrics,
                "initial_perceptual_hash": perceptual_hash(initial),
                "terminal_perceptual_hash": perceptual_hash(terminal),
            }
        )
    except (
        OSError,
        ValueError,
        cv2.error,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        LOGGER.warning("Invalid episode %s: %s", episode.episode_id, error)
        return ScreeningDecision.model_validate(
            values
            | {
                "disposition": "invalid",
                "reasons": (f"{type(error).__name__}: {error}",),
                "metrics": None,
            }
        )


def mark_probable_duplicates(
    decisions: list[ScreeningDecision], policy: ScreeningPolicy
) -> list[ScreeningDecision]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(decisions):
        if row.disposition != "invalid" and row.onset_s is not None:
            by_group[row.group_id].append(index)
    parent = list(range(len(decisions)))

    def component(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    rank = {"priority_review": 0, "needs_reprocessing": 1, "not_suitable": 2, "invalid": 3}
    for indices in by_group.values():
        indices.sort(key=lambda index: (decisions[index].onset_s or 0, decisions[index].episode_id))
        for position, index in enumerate(indices):
            current = decisions[index]
            for other_index in indices[position + 1 :]:
                other = decisions[other_index]
                if (
                    abs((other.onset_s or 0) - (current.onset_s or 0))
                    > policy.duplicate_time_tolerance_s
                ):
                    break
                if (
                    current.recording_id == other.recording_id
                    or abs(current.start_s - other.start_s) > policy.duplicate_time_tolerance_s
                    or abs(current.terminal_s - other.terminal_s)
                    > policy.duplicate_time_tolerance_s
                    or not all(
                        value is not None
                        for value in (
                            current.initial_perceptual_hash,
                            current.terminal_perceptual_hash,
                            other.initial_perceptual_hash,
                            other.terminal_perceptual_hash,
                        )
                    )
                ):
                    continue
                first_distance = (
                    int(current.initial_perceptual_hash or "", 16)
                    ^ int(other.initial_perceptual_hash or "", 16)
                ).bit_count()
                last_distance = (
                    int(current.terminal_perceptual_hash or "", 16)
                    ^ int(other.terminal_perceptual_hash or "", 16)
                ).bit_count()
                if max(first_distance, last_distance) > policy.duplicate_image_hash_distance:
                    continue
                parent[component(other_index)] = component(index)
    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(decisions)):
        clusters[component(index)].append(index)
    representatives = list(decisions)

    def representative_score(index: int) -> tuple[int, float, str]:
        decision = decisions[index]
        visible_fraction = decision.metrics.visible_frame_fraction if decision.metrics else 0.0
        return rank[decision.disposition], -visible_fraction, decision.episode_id

    for indices in clusters.values():
        if len(indices) < 2:
            continue
        keep = min(indices, key=representative_score)
        for index in indices:
            if index == keep:
                continue
            representatives[index] = decisions[index].model_copy(
                update={
                    "disposition": "probable_duplicate",
                    "duplicate_of": decisions[keep].episode_id,
                    "reasons": decisions[index].reasons
                    + ("synchronised_endpoint_images_in_shared_source_group",),
                }
            )
    return representatives


def _export_screening(output: Path, decisions: list[ScreeningDecision]) -> None:
    with (output / "decisions.jsonl").open("w") as stream:
        for row in decisions:
            stream.write(row.model_dump_json() + "\n")
    fields = [
        "episode_id",
        "recording_id",
        "group_id",
        "disposition",
        "start_s",
        "terminal_s",
        "duration_s",
        "visible_frame_fraction",
        "cue_excursion_radii",
        "cue_peak_speed_mps",
        "endpoint_track_retention",
        "unresolved_endpoint_components",
        "duplicate_unique_colours",
        "cue_motion_bursts",
        "camera_corner_shift_fraction",
        "duplicate_of",
        "reasons",
        "record_path",
        "source_clip",
    ]
    for disposition in (
        "all",
        "priority_review",
        "needs_reprocessing",
        "not_suitable",
        "probable_duplicate",
        "invalid",
    ):
        with (output / f"{disposition}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in decisions:
                if disposition != "all" and row.disposition != disposition:
                    continue
                values = row.model_dump(mode="json")
                if row.metrics:
                    values.update(row.metrics.model_dump(mode="json"))
                writer.writerow(
                    {
                        field: ";".join(row.reasons)
                        if field == "reasons"
                        else values.get(field, "")
                        for field in fields
                    }
                )
    with (output / "priority_episodes.jsonl").open("w") as stream:
        for row in decisions:
            if row.disposition == "priority_review":
                episode = SnookerEpisode.model_validate_json(Path(row.record_path).read_text())
                stream.write(episode.model_dump_json() + "\n")


def screen_corpus(
    source: Path, output: Path, *, workers: int = 4, policy: ScreeningPolicy | None = None
) -> dict[str, object]:
    if not 1 <= workers <= 4:
        raise ValueError("Screen with 1-4 workers within the approved CPU budget")
    policy = policy or ScreeningPolicy()
    source, output = source.resolve(), output.resolve()
    data_root = (PROJECT_ROOT / "data").resolve()
    if (
        not source.is_relative_to(data_root)
        or not output.is_relative_to(data_root)
        or source == output
        or output.is_relative_to(source)
    ):
        raise ValueError("Use a separate in-project quality-output directory")
    state = json.loads((source / "collection_state.json").read_text())
    if state["status"] != "complete" or state["runner_active"]:
        raise ValueError("Quality screening requires a finished, inactive raw collection")
    database = source / "collection.sqlite"
    check_storage(2_000_000_000)
    cv2.setNumThreads(1)
    config = {
        "source": str(source),
        "database_sha256": file_sha256(database),
        "source_manifest_sha256": file_sha256(source / "collection_manifest.json"),
        "policy": policy.model_dump(mode="json"),
        "screening_code_sha256": file_sha256(Path(__file__)),
        "input_schema_code_sha256": file_sha256(Path(__file__).with_name("real_data.py")),
        "opencv": cv2.__version__,
        "workers": workers,
        "selection_is_ground_truth": False,
        "duplicate_hash_basis": "native_source_endpoint_images_not_bev",
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".screening.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "run_config.json").exists():
            if json.loads((output / "run_config.json").read_text()) != config:
                raise ValueError("Screening policy/code/input changed; use a new screening version")
            if (output / "summary.json").exists():
                return json.loads((output / "summary.json").read_text())
        elif any(path.name != ".screening.lock" for path in output.iterdir()):
            raise ValueError("Refusing to overwrite an untracked screening directory")
        write_json(output / "run_config.json", config)
        decisions = []
        with (
            closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection,
            ThreadPoolExecutor(max_workers=workers) as executor,
            (output / "raw_decisions.jsonl").open("w") as raw_stream,
        ):
            rows = list(connection.execute("SELECT payload,path FROM episodes ORDER BY id"))
            if len(rows) != state["raw_candidates"]:
                raise ValueError("Frozen source state and raw episode count disagree")

            def evaluate(row: tuple[str, str]) -> ScreeningDecision:
                return screen_episode(row[0], row[1], policy, data_root)

            for index, result in enumerate(executor.map(evaluate, rows), 1):
                decisions.append(result)
                raw_stream.write(result.model_dump_json() + "\n")
                if index % 100 == 0:
                    raw_stream.flush()
                    LOGGER.info("Screened %s/%s raw episodes", index, len(rows))
                    write_json(
                        output / "screening_state.json",
                        {"status": "screening", "screened": index, "expected": len(rows)},
                    )
        decisions = mark_probable_duplicates(decisions, policy)
        if len({row.episode_id for row in decisions}) != len(decisions):
            raise ValueError("Raw collection contains repeated episode IDs")
        _export_screening(output, decisions)
        categories = Counter(row.disposition for row in decisions)
        reasons = Counter(reason for row in decisions for reason in row.reasons)
        by_source: dict[str, Counter[str]] = defaultdict(Counter)
        for row in decisions:
            by_source[row.recording_id][row.disposition] += 1
        summary = {
            "status": "complete_automatic_screening",
            "input_episodes": len(decisions),
            "category_counts": dict(categories),
            "reason_counts": dict(reasons),
            "by_recording": {name: dict(counts) for name, counts in sorted(by_source.items())},
            "priority_review_recordings": len(
                {row.recording_id for row in decisions if row.disposition == "priority_review"}
            ),
            "policy_hash": fingerprint(policy.model_dump(mode="json")),
            "original_records_modified": False,
            "accepted_benchmark_tasks": 0,
            "independent_label_accuracy_measured": False,
            "model_or_api_calls": 0,
            "interpretation": (
                "Automatic triage: priority_review is a shortlist, not Gold or calibrated physics, "
                "a verified complete inventory or proof of one strike. Excluded low-motion windows "
                "can include legitimate slow/return shots and are preserved."
            ),
        }
        write_json(output / "summary.json", summary)
        write_json(output / "screening_state.json", summary)
        return summary


def screening_gallery(
    output: Path, *, per_category: int = 8, seed: int = 20261011
) -> dict[str, object]:
    if not 1 <= per_category <= 24:
        raise ValueError("Sample 1-24 episodes per screening category")
    decisions = [
        ScreeningDecision.model_validate_json(line)
        for line in (output / "decisions.jsonl").read_text().splitlines()
    ]
    rng = np.random.default_rng(seed)
    selected = []
    for category in ("priority_review", "needs_reprocessing", "not_suitable", "probable_duplicate"):
        pool = [row for row in decisions if row.disposition == category]
        if not pool:
            continue
        indices = rng.choice(len(pool), min(per_category, len(pool)), replace=False)
        picked = [pool[int(index)] for index in indices]
        for row in picked:
            selected.append(row.model_dump(mode="json"))
        columns, width, height = 4, 384, 260
        canvas = Image.new(
            "RGB", (columns * width, math.ceil(len(picked) / columns) * height), "#eeeeee"
        )
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(picked):
            left, top = index % columns * width, index // columns * height
            for side, path in enumerate((row.initial_image, row.terminal_image)):
                with Image.open(path) as original:
                    image = original.convert("RGB")
                    image.thumbnail((width - 8, (height - 45) // 2))
                    canvas.paste(image, (left + 4, top + 45 + side * 105))
            draw.text(
                (left + 4, top + 3), f"{row.recording_id} / {row.episode_id[:12]}", fill="#202020"
            )
            draw.text(
                (left + 4, top + 16), f"{row.start_s:.1f}-{row.terminal_s:.1f}s", fill="#202020"
            )
            draw.text((left + 4, top + 29), category, fill="#902020")
        canvas.save(output / f"{category}_sample.jpg")
    manifest = {
        "seed": seed,
        "sampling": "uniform random without replacement within automatic category",
        "selected": selected,
        "sample_count": len(selected),
        "human_gold_review": False,
        "representative_of_other_categories": False,
    }
    write_json(output / "spotcheck_sample.json", manifest)
    return {"sampled_episodes": len(selected), "galleries": str(output)}
