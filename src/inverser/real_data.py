import math
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from inverser.runtime import fingerprint, write_json
from inverser.schemas import Record
from inverser.vision import FrameObservation, ObservedKind, TableProjection


class SourceProvenance(Record):
    collection: Literal["hf_snooker_videos", "billiard_dataset"]
    recording_id: str
    original_file: str
    original_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    group_id: str
    media_license: str
    source_publication: str | None = None
    content_duplicate_group: str | None = None


class TableCalibration(Record):
    method: Literal["supplied_corners", "automatic_felt_quad", "automatic_background_quad"]
    source_corners_px: tuple[tuple[float, float], ...] = Field(min_length=4, max_length=4)
    image_size_px: tuple[int, int]
    homography: tuple[tuple[float, float, float], ...] = Field(min_length=3, max_length=3)
    table_size_m: tuple[float, float]
    ball_radius_m: float = Field(gt=0)
    dimension_basis: Literal["nominal_regulation", "paper_baize_plus_cushion"]
    bev_size_px: tuple[int, int]
    lens_distortion_corrected: bool = False
    ball_height_corrected: bool = False
    metric_error_measured: bool = False
    orientation_rule_verified: bool = False
    flags: tuple[str, ...] = ()


class ObservedBall(Record):
    observation_id: str
    track_id: str | None
    kind: ObservedKind
    status: Literal["observed"] = "observed"
    xy_m: tuple[float, float]
    bev_xy_px: tuple[float, float]
    heuristic_shape_score: float = Field(ge=0, le=1)
    position_uncertainty_m: float | None = None
    colour_independently_verified: bool = False


class ObservedState(Record):
    time_s: float = Field(ge=0)
    balls: tuple[ObservedBall, ...]
    complete_inventory_verified: bool = False
    view_valid: bool
    visibility_clear: bool
    unresolved_components: int = Field(ge=0)


class TrajectoryPoint(Record):
    time_s: float = Field(ge=0)
    xy_m: tuple[float, float]
    evidence: Literal["detected"] = "detected"


class ObservedTrajectory(Record):
    track_id: str
    kind: ObservedKind
    points: tuple[TrajectoryPoint, ...]
    missing_intervals: int = Field(ge=0)
    identity_verified: bool = False
    interpolated_points: Literal[0] = 0

    @model_validator(mode="after")
    def increasing_times(self) -> Self:
        if any(a.time_s >= b.time_s for a, b in zip(self.points, self.points[1:])):
            raise ValueError("Observed trajectory times must strictly increase")
        return self


class EstimatedEvent(Record):
    kind: Literal[
        "cue_motion_onset", "ball_ball_proximity", "cushion_proximity", "observation_disappearance"
    ]
    time_s: float = Field(ge=0)
    participants: tuple[str, ...]
    evidence: Literal["heuristic_candidate"] = "heuristic_candidate"
    verified: Literal[False] = False
    temporal_resolution_s: float = Field(gt=0)


class OriginalAnnotation(Record):
    kind: Literal["manual_path_image", "tracking_output", "reconstructed_x3d", "source_notes"]
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    measured_action_ground_truth: Literal[False] = False
    time_indexed_observations_verified: bool = False


class EpisodeMedia(Record):
    initial_source: str
    terminal_source: str
    initial_bev: str
    terminal_bev: str
    clean_table_pair: str
    source_clip: str | None
    source_clip_start_s: float = Field(ge=0)
    source_clip_end_s: float = Field(gt=0)


class EpisodeQuality(Record):
    tier: Literal["raw_candidate", "silver_unverified"] = "raw_candidate"
    flags: tuple[str, ...]
    cue_endpoint_shift_m: float | None
    missing_track_ids: tuple[str, ...]
    introduced_track_ids: tuple[str, ...]
    observed_frames: int = Field(ge=2)
    visible_frame_fraction: float = Field(ge=0, le=1)
    exactly_one_strike_verified: Literal[False] = False
    independent_annotation_validation: Literal[False] = False
    benchmark_accepted: Literal[False] = False


class ObservedEpisode(Record):
    episode_id: str
    source: SourceProvenance
    calibration: TableCalibration
    start_s: float = Field(ge=0)
    terminal_s: float = Field(gt=0)
    estimated_motion_onset_s: float | None = Field(ge=0)
    sample_hz: float = Field(gt=0)
    timestamp_basis: Literal["ffmpeg_resampling_grid", "capture_frame_index_at_declared_fps"] = (
        "ffmpeg_resampling_grid"
    )
    initial: ObservedState
    terminal: ObservedState
    trajectories: tuple[ObservedTrajectory, ...]
    estimated_events: tuple[EstimatedEvent, ...]
    source_annotations: tuple[OriginalAnnotation, ...] = ()
    media: EpisodeMedia
    quality: EpisodeQuality
    cue_control_ground_truth: None = None
    angular_velocity_ground_truth: None = None
    candidate_partition: Literal["unassigned"] = "unassigned"

    @model_validator(mode="after")
    def check_interval(self) -> Self:
        if (
            self.estimated_motion_onset_s is not None
            and not self.start_s <= self.estimated_motion_onset_s <= self.terminal_s
        ):
            raise ValueError("Motion onset must lie in the observation interval")
        if self.initial.time_s != self.start_s or self.terminal.time_s != self.terminal_s:
            raise ValueError("Episode endpoints disagree with observation timestamps")
        if self.terminal_s <= self.start_s:
            raise ValueError("Episode duration must be positive")
        for trajectory in self.trajectories:
            if any(
                not self.start_s <= point.time_s <= self.terminal_s for point in trajectory.points
            ):
                raise ValueError("Trajectory observation lies outside the episode")
        if any(
            not self.start_s <= event.time_s <= self.terminal_s for event in self.estimated_events
        ):
            raise ValueError("Estimated event lies outside the episode")
        return self


class SnookerEpisode(ObservedEpisode):
    schema_version: Literal["snooker-observation-v1"] = "snooker-observation-v1"
    game: Literal["snooker"] = "snooker"


class BlackballEpisode(ObservedEpisode):
    schema_version: Literal["blackball-observation-v1"] = "blackball-observation-v1"
    game: Literal["blackball"] = "blackball"


def calibration_record(
    projection: TableProjection,
    method: Literal["supplied_corners", "automatic_felt_quad", "automatic_background_quad"],
    basis: Literal["nominal_regulation", "paper_baize_plus_cushion"],
) -> TableCalibration:
    return TableCalibration(
        method=method,
        source_corners_px=tuple((float(point[0]), float(point[1])) for point in projection.points),
        image_size_px=(projection.source_width, projection.source_height),
        homography=tuple(
            (float(row[0]), float(row[1]), float(row[2])) for row in projection.matrix
        ),
        table_size_m=(projection.physics.table_width_m, projection.physics.table_length_m),
        ball_radius_m=projection.physics.ball_radius_m,
        dimension_basis=basis,
        bev_size_px=(projection.width, projection.height),
        flags=(
            "ball_image_centers_projected_as_table_plane",
            "nominal_or_paper_dimensions_not_measured_per_recording",
        ),
    )


def observed_state(frame: FrameObservation) -> ObservedState:
    return ObservedState(
        time_s=frame.time_s,
        balls=tuple(
            ObservedBall(
                observation_id=f"ball_{index}",
                track_id=ball.track_id,
                kind=ball.kind,
                xy_m=ball.xy_m,
                bev_xy_px=ball.center_px,
                heuristic_shape_score=ball.shape_score,
            )
            for index, ball in enumerate(frame.detections)
        ),
        view_valid=frame.view_valid,
        visibility_clear=frame.visibility_clear,
        unresolved_components=frame.unresolved_components,
    )


def observed_events(
    frames: tuple[FrameObservation, ...],
    radius_m: float,
    sample_hz: float,
    motion_onset_s: float | None,
) -> tuple[EstimatedEvent, ...]:
    events = (
        [
            EstimatedEvent(
                kind="cue_motion_onset",
                time_s=motion_onset_s,
                participants=("cue",),
                temporal_resolution_s=1 / sample_hz,
            )
        ]
        if motion_onset_s is not None
        else []
    )
    last_proximity: dict[tuple[str, ...], float] = {}
    for frame in frames:
        balls = [ball for ball in frame.detections if ball.track_id is not None]
        for index, left in enumerate(balls):
            for right in balls[index + 1 :]:
                if left.track_id is None or right.track_id is None:
                    continue
                participants = tuple(sorted((left.track_id, right.track_id)))
                if math.dist(left.xy_m, right.xy_m) <= 2.6 * radius_m:
                    previous = last_proximity.get(participants)
                    if previous is None or frame.time_s - previous > 1.0:
                        events.append(
                            EstimatedEvent(
                                kind="ball_ball_proximity",
                                time_s=frame.time_s,
                                participants=participants,
                                temporal_resolution_s=1 / sample_hz,
                            )
                        )
                    last_proximity[participants] = frame.time_s
    return tuple(events)


def make_episode(
    provenance: SourceProvenance,
    calibration: TableCalibration,
    frames: tuple[FrameObservation, ...],
    motion_onset_s: float | None,
    sample_hz: float,
    media: EpisodeMedia,
    source_annotations: tuple[OriginalAnnotation, ...] = (),
    *,
    game: Literal["snooker", "blackball"] = "snooker",
    timestamp_basis: Literal[
        "ffmpeg_resampling_grid", "capture_frame_index_at_declared_fps"
    ] = "ffmpeg_resampling_grid",
) -> SnookerEpisode | BlackballEpisode:
    if len(frames) < 2:
        raise ValueError("An episode needs at least two observations")
    initial, terminal = observed_state(frames[0]), observed_state(frames[-1])
    kinds: dict[str, ObservedKind] = {}
    points: dict[str, list[TrajectoryPoint]] = {}
    for frame in frames:
        for ball in frame.detections:
            if ball.track_id is None:
                raise ValueError("Observed track ID is absent")
            kinds.setdefault(ball.track_id, ball.kind)
            points.setdefault(ball.track_id, []).append(
                TrajectoryPoint(time_s=frame.time_s, xy_m=ball.xy_m)
            )
    trajectories = tuple(
        ObservedTrajectory(
            track_id=track_id,
            kind=kinds[track_id],
            points=tuple(samples),
            missing_intervals=sum(
                second.time_s - first.time_s > 1.5 / sample_hz
                for first, second in zip(samples, samples[1:])
            ),
        )
        for track_id, samples in sorted(points.items())
    )
    first_ids = {ball.track_id for ball in initial.balls if ball.track_id is not None}
    last_ids = {ball.track_id for ball in terminal.balls if ball.track_id is not None}
    first_cue = [ball for ball in initial.balls if ball.kind == "cue"]
    last_cue = [ball for ball in terminal.balls if ball.kind == "cue"]
    shift = (
        math.dist(first_cue[0].xy_m, last_cue[0].xy_m)
        if len(first_cue) == len(last_cue) == 1
        else None
    )
    flags = ["automatic_detector_not_independently_validated", "complete_inventory_unknown"]
    if game == "blackball" and any(
        ball.kind not in ("cue", "red", "yellow", "black", "unknown")
        for state in (initial, terminal)
        for ball in state.balls
    ):
        flags.append("unexpected_colour_hypothesis_for_blackball")
    if motion_onset_s is None:
        flags.append("motion_onset_unresolved")
    if first_ids != last_ids:
        flags.append("identity_changes_or_missing_observations_not_pot_labels")
    if initial.unresolved_components or terminal.unresolved_components:
        flags.append("unresolved_endpoint_components")
    if any(not frame.visibility_clear for frame in frames):
        flags.append("intermediate_visibility_gaps")
    if shift is None or shift < 2 * calibration.ball_radius_m:
        flags.append("small_or_unknown_cue_endpoint_shift")
    episode_type = SnookerEpisode if game == "snooker" else BlackballEpisode
    return episode_type(
        episode_id=fingerprint(
            {
                "source_group": provenance.group_id,
                "original_sha256": provenance.original_sha256,
                "start_sample": round(frames[0].time_s * sample_hz),
                "onset_sample": (
                    round(motion_onset_s * sample_hz) if motion_onset_s is not None else None
                ),
                "end_sample": round(frames[-1].time_s * sample_hz),
            }
        ),
        source=provenance,
        calibration=calibration,
        start_s=frames[0].time_s,
        terminal_s=frames[-1].time_s,
        estimated_motion_onset_s=motion_onset_s,
        sample_hz=sample_hz,
        timestamp_basis=timestamp_basis,
        initial=initial,
        terminal=terminal,
        trajectories=trajectories,
        estimated_events=observed_events(
            frames, calibration.ball_radius_m, sample_hz, motion_onset_s
        ),
        source_annotations=source_annotations,
        media=media,
        quality=EpisodeQuality(
            flags=tuple(flags),
            cue_endpoint_shift_m=shift,
            missing_track_ids=tuple(sorted(first_ids - last_ids)),
            introduced_track_ids=tuple(sorted(last_ids - first_ids)),
            observed_frames=len(frames),
            visible_frame_fraction=sum(frame.visibility_clear for frame in frames) / len(frames),
        ),
    )


def export_episode(path: Path, episode: SnookerEpisode | BlackballEpisode) -> None:
    write_json(path, episode.model_dump(mode="json"))

    # Endpoint-only views do not expose interior track IDs, motion onset or action evidence.
    def endpoints(state: ObservedState) -> dict[str, object]:
        data = state.model_dump(mode="json")
        data["balls"] = [
            {key: value for key, value in ball.items() if key != "track_id"}
            for ball in data["balls"]
        ]
        return data

    write_json(
        path.with_name("endpoints.json"),
        {
            "schema_version": f"{episode.game}-endpoints-v1",
            "game": episode.game,
            "episode_id": episode.episode_id,
            "initial": endpoints(episode.initial),
            "terminal": endpoints(episode.terminal),
            "calibration": episode.calibration.model_dump(mode="json"),
            "benchmark_accepted": False,
        },
    )


def render_clean_table_pair(path: Path, episode: SnookerEpisode | BlackballEpisode) -> None:
    from PIL import Image, ImageDraw

    from inverser.preview import COLORS

    width, height, margin = 256, 520, 16
    canvas = Image.new("RGB", (2 * width, height), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    table_w, table_l = episode.calibration.table_size_m
    radius = episode.calibration.ball_radius_m / table_w * (width - 2 * margin)
    for index, (name, state) in enumerate(
        (("Initial estimate", episode.initial), ("Terminal estimate", episode.terminal))
    ):
        left, top = index * width + margin, 38
        right, bottom = (index + 1) * width - margin, height - margin
        draw.text((left, 10), name, fill="#202020")
        draw.rectangle((left, top, right, bottom), fill="#226b47")
        for ball in state.balls:
            x = left + ball.xy_m[0] / table_w * (right - left)
            y = bottom - ball.xy_m[1] / table_l * (bottom - top)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=COLORS.get(ball.kind, "#999999"),
                outline="#202020",
            )
    canvas.save(path)
