import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import Field

from inverser.author_uploads import UploadedAsset
from inverser.real_data import SnookerEpisode
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.schemas import Record
from inverser.vision import ImageArray, TableProjection, detect_balls

LOGGER = logging.getLogger(__name__)


class RegistrationPolicy(Record):
    max_normalized_residual: float = Field(default=0.02, gt=0, lt=1)
    min_separated_peak_margin: float = Field(default=0.001, gt=0)
    min_background_fraction: float = Field(default=0.5, gt=0, le=1)
    validation_tolerance_px: float = Field(default=1.0, gt=0)
    source_panel: Literal["bottom_right_body"] = "bottom_right_body"
    header_height_px: Literal[20] = 20


class PixelRegistration(Record):
    method: Literal["native_background_pixel_crop_registration"] = (
        "native_background_pixel_crop_registration"
    )
    source_offset_px: tuple[int, int]
    normalized_residual: float = Field(ge=0)
    separated_peak_margin: float = Field(ge=0)
    background_fraction: float = Field(gt=0, le=1)
    passed_policy: bool


class FrameMappingAudit(Record):
    frame_index: int = Field(ge=0)
    source_frame_sha256: str
    rendered_patch_sha256: str
    registration: PixelRegistration
    frozen_origin_error_px: float = Field(ge=0)


class RecordingMappingAudit(Record):
    episode_id: str
    source_group: str
    role: Literal["support", "held_out"]
    raw_capture: str
    source_render: str
    manual_raster: str
    raw_snapshot_sha256: str
    raw_signature_unchanged: Literal[True] = True
    fresh_full_raw_checksum_computed: Literal[False] = False
    manual_raster_sha256: str
    native_canvas_px: tuple[int, int]
    source_render_frame_count: int
    raw_frame_count: int
    crop_source_to_native: tuple[tuple[float, float, float], ...]
    calibration: FrameMappingAudit
    held_out_frame: FrameMappingAudit
    temporal_mapping_consistent: bool
    spatial_registration_validated: bool
    manual_pixel_paths_used_for_registration: Literal[False] = False
    frame_timing_physically_verified: Literal[False] = False
    metric_world_accuracy_verified: Literal[False] = False
    measured_cue_action_available: Literal[False] = False
    physical_benchmark_accepted: Literal[False] = False


class CanvasInputReceipts(Record):
    episodes_file: str
    episodes_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory_file: str
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_source_groups: tuple[str, ...] = Field(min_length=2, max_length=16)


class CanvasAuditResult(Record):
    schema_version: Literal["source-canvas-audit-v2"] = "source-canvas-audit-v2"
    status: Literal["bounded_source_audit_complete"] = "bounded_source_audit_complete"
    input_fingerprint: str
    inputs: CanvasInputReceipts
    policy: RegistrationPolicy
    recordings: tuple[RecordingMappingAudit, ...] = Field(min_length=2, max_length=16)
    support_recordings: int
    held_out_recordings: int
    native_canvas_counts: dict[str, int]
    source_offset_counts: dict[str, int]
    validated_recordings: int
    source_frames_read: int
    source_render_frames_decoded: int
    full_videos_reprocessed: Literal[0] = 0
    actual_model_requests: Literal[0] = 0
    shared_single_crop_supported: bool
    mapping_to_manual_pixels_requires_shared_roi_provenance: Literal[True] = True
    g1_approved: Literal[False] = False


class SampledPointCompatibility(Record):
    episode_id: str
    source_group: str
    frame_index: int
    detected_points: int
    outside_canvas_points: int
    points_within_3px: int
    shared_crop_points_within_3px: int
    mean_in_canvas_distance_px: float | None
    maximum_in_canvas_distance_px: float | None
    random_canvas_coverage_within_3px: float
    colour_counts: dict[str, int]
    inventory_conflicts: tuple[str, ...]
    unresolved_components: int


class PointCompatibilityResult(Record):
    schema_version: Literal["native-point-compatibility-v1"] = "native-point-compatibility-v1"
    mapping_audit_sha256: str
    implementation_sha256: str
    protocol: Literal["held_out_sampled_points_against_native_path_union"] = (
        "held_out_sampled_points_against_native_path_union"
    )
    excluded_support_recordings: int
    excluded_mapping_failures: int
    recording_groups: int
    sampled_frames: int
    detected_points: int
    points_within_3px: int
    shared_crop_points_within_3px: int
    conditional_detected_point_coverage: float | None
    frames_with_inventory_conflicts: int
    rows: tuple[SampledPointCompatibility, ...]
    transform_refitted_using_manual_paths: Literal[False] = False
    complete_inventory_recall_measured: Literal[False] = False
    timed_trajectory_accuracy_measured: Literal[False] = False
    metric_world_accuracy_verified: Literal[False] = False
    g1_approved: Literal[False] = False


def register_source_crop(
    source: ImageArray,
    rendered_patch: ImageArray,
    policy: RegistrationPolicy,
) -> PixelRegistration:
    if (
        source.dtype != np.uint8
        or rendered_patch.dtype != np.uint8
        or source.ndim != 3
        or rendered_patch.ndim != 3
        or source.shape[2] != 3
        or rendered_patch.shape[2] != 3
        or any(a < b for a, b in zip(source.shape[:2], rendered_patch.shape[:2], strict=True))
    ):
        raise ValueError("Registration requires native BGR images with an in-source crop")
    hsv = cv2.cvtColor(rendered_patch, cv2.COLOR_BGR2HSV)
    cloth = cv2.inRange(hsv, np.asarray([30, 40, 35]), np.asarray([100, 255, 255]))
    cloth = cv2.erode(cloth, np.ones((9, 9), np.uint8))
    fraction = float(np.count_nonzero(cloth) / cloth.size)
    if fraction < policy.min_background_fraction:
        raise ValueError("Render does not contain enough independent cloth pixels for registration")
    residuals = cv2.matchTemplate(source, rendered_patch, cv2.TM_SQDIFF_NORMED, mask=cloth)
    if not np.isfinite(residuals).all():
        raise ValueError("Pixel registration produced non-finite comparison values")
    best, _, offset, _ = cv2.minMaxLoc(residuals)
    x, y = offset
    separated = residuals.copy()
    separated[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = np.inf
    alternatives = separated[np.isfinite(separated)]
    if not len(alternatives):
        raise ValueError("Registration has no separated offsets to test ambiguity")
    margin = max(0.0, float(np.min(alternatives)) - best)
    passed = best <= policy.max_normalized_residual and margin >= policy.min_separated_peak_margin
    if not passed:
        LOGGER.warning("Pixel registration rejected: residual %.6f, margin %.6f", best, margin)
    return PixelRegistration(
        source_offset_px=(x, y),
        normalized_residual=best,
        separated_peak_margin=margin,
        background_fraction=fraction,
        passed_policy=passed,
    )


def rendered_body(
    frame: ImageArray, canvas_px: tuple[int, int], policy: RegistrationPolicy
) -> ImageArray:
    width, height = canvas_px
    stride = height + policy.header_height_px
    if frame.shape != (2 * stride, 2 * width, 3):
        raise ValueError("Source rendering does not match the declared 2x2 native panel layout")
    return frame[stride + policy.header_height_px : 2 * stride, width : 2 * width].copy()


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino


def _source_path(root: Path, path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    if (
        resolved.is_symlink()
        or not resolved.is_file()
        or not resolved.resolve().is_relative_to(root)
    ):
        raise ValueError("Audit inputs must be regular files inside the declared source root")
    return resolved


def _sample_raw_frame(path: Path, image_size_px: tuple[int, int], frame_index: int) -> ImageArray:
    width, height = image_size_px
    frame_bytes = width * height * 3
    if not 0 <= frame_index < path.stat().st_size // frame_bytes:
        raise ValueError("Requested source frame is outside the raw capture")
    with path.open("rb") as stream:
        stream.seek(frame_index * frame_bytes)
        data = stream.read(frame_bytes)
    if len(data) != frame_bytes:
        raise ValueError("Raw capture changed during bounded sampling")
    return np.frombuffer(data, np.uint8).reshape(height, width, 3)


def _point_distance(distance_map: NDArray[np.float32], point: tuple[float, float]) -> float | None:
    x, y = point
    height, width = distance_map.shape
    if not 0 <= x <= width - 1 or not 0 <= y <= height - 1:
        return None
    return float(cv2.getRectSubPix(distance_map, (1, 1), (x, y))[0, 0])


def audit_registered_points(
    mapping_path: Path, episodes_path: Path, output: Path
) -> PointCompatibilityResult:
    if output.exists():
        raise FileExistsError("A point-compatibility audit must not overwrite prior evidence")
    mapping = CanvasAuditResult.model_validate_json(mapping_path.read_text())
    if file_sha256(episodes_path) != mapping.inputs.episodes_sha256:
        raise ValueError("Observation index changed since the frozen spatial audit")
    episodes = {
        episode.episode_id: episode
        for episode in (SnookerEpisode.model_validate_json(line) for line in episodes_path.open())
    }
    support = [row for row in mapping.recordings if row.role == "support"]
    if not support:
        raise ValueError("Shared-crop control requires a declared support recording")
    shared_offset = support[0].calibration.registration.source_offset_px
    rows = []
    excluded = 0
    groups = set()
    for recording in mapping.recordings:
        if recording.role == "support":
            continue
        if not recording.spatial_registration_validated:
            excluded += 1
            LOGGER.warning("Excluded unvalidated mapping for %s", recording.episode_id)
            continue
        episode = episodes.get(recording.episode_id)
        if episode is None or episode.source.group_id != recording.source_group:
            raise ValueError("Mapping and observation recording groups disagree")
        if episode.source.original_file != recording.raw_capture:
            raise ValueError("Mapping refers to a different raw capture than the observation")
        manual = Path(recording.manual_raster)
        if file_sha256(manual) != recording.manual_raster_sha256:
            raise ValueError("Manual spatial reference changed since the frozen mapping audit")
        image = cv2.imread(str(manual), cv2.IMREAD_GRAYSCALE)
        if image is None or (image.shape[1], image.shape[0]) != recording.native_canvas_px:
            raise ValueError("Manual spatial reference is not decodable in its native canvas")
        path_pixels = image < 128
        if not np.any(path_pixels):
            raise ValueError("An empty manual path cannot provide point compatibility evidence")
        distance_map = np.asarray(
            cv2.distanceTransform(
                (~path_pixels).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            ),
            dtype=np.float32,
        )
        calibration = episode.calibration
        projection = TableProjection(
            [list(point) for point in calibration.source_corners_px],
            *calibration.image_size_px,
            output_width=calibration.bev_size_px[0],
            table_width_m=calibration.table_size_m[0],
            table_length_m=calibration.table_size_m[1],
            ball_radius_m=calibration.ball_radius_m,
            corner_order="ordered_short_edge",
        )
        inverse = np.linalg.inv(projection.matrix)
        offset = recording.calibration.registration.source_offset_px
        groups.add(recording.source_group)
        for frame_record in (recording.calibration, recording.held_out_frame):
            native = _sample_raw_frame(
                Path(recording.raw_capture), calibration.image_size_px, frame_record.frame_index
            )
            if hashlib.sha256(native.tobytes()).hexdigest() != frame_record.source_frame_sha256:
                raise ValueError("Sampled source pixels changed since the frozen mapping audit")
            detections, _, _, unresolved = detect_balls(native, projection, native_image=True)
            counts: Counter[str] = Counter(ball.kind for ball in detections)
            conflicts = [
                f"multiple_{kind}_hypotheses"
                for kind in ("yellow", "green", "brown", "blue", "pink", "black")
                if counts[kind] > 1
            ]
            if counts["cue"] != 1:
                conflicts.append("cue_inventory_unresolved")
            if counts["unknown"]:
                conflicts.append("unknown_colour_hypotheses")
            distances = []
            outside = matched = shared_matched = 0
            for detection in detections:
                homogeneous = inverse @ np.asarray([*detection.center_px, 1.0])
                if not np.isfinite(homogeneous).all() or abs(homogeneous[2]) < 1e-12:
                    raise ValueError("Detected point has no finite native image coordinate")
                x, y = homogeneous[:2] / homogeneous[2]
                distance = _point_distance(
                    distance_map, (float(x - offset[0]), float(y - offset[1]))
                )
                shared_distance = _point_distance(
                    distance_map, (float(x - shared_offset[0]), float(y - shared_offset[1]))
                )
                if distance is None:
                    outside += 1
                else:
                    distances.append(distance)
                    matched += distance <= 3
                shared_matched += shared_distance is not None and shared_distance <= 3
            rows.append(
                SampledPointCompatibility(
                    episode_id=recording.episode_id,
                    source_group=recording.source_group,
                    frame_index=frame_record.frame_index,
                    detected_points=len(detections),
                    outside_canvas_points=outside,
                    points_within_3px=matched,
                    shared_crop_points_within_3px=shared_matched,
                    mean_in_canvas_distance_px=float(np.mean(distances)) if distances else None,
                    maximum_in_canvas_distance_px=max(distances) if distances else None,
                    random_canvas_coverage_within_3px=float(np.mean(distance_map <= 3)),
                    colour_counts=dict(sorted(counts.items())),
                    inventory_conflicts=tuple(conflicts),
                    unresolved_components=unresolved,
                )
            )
    detected = sum(row.detected_points for row in rows)
    matched = sum(row.points_within_3px for row in rows)
    if not detected:
        LOGGER.warning("No sampled ball detections provide spatial compatibility evidence")
    report = PointCompatibilityResult(
        mapping_audit_sha256=file_sha256(mapping_path),
        implementation_sha256=file_sha256(Path(__file__)),
        excluded_support_recordings=len(support),
        excluded_mapping_failures=excluded,
        recording_groups=len(groups),
        sampled_frames=len(rows),
        detected_points=detected,
        points_within_3px=matched,
        shared_crop_points_within_3px=sum(row.shared_crop_points_within_3px for row in rows),
        conditional_detected_point_coverage=matched / detected if detected else None,
        frames_with_inventory_conflicts=sum(bool(row.inventory_conflicts) for row in rows),
        rows=tuple(rows),
    )
    write_json(output, report.model_dump(mode="json"))
    return report


def audit_author_canvases(
    episodes_path: Path,
    inventory_path: Path,
    output: Path,
    *,
    limit_recordings: int = 16,
    policy: RegistrationPolicy | None = None,
) -> CanvasAuditResult:
    if not 2 <= limit_recordings <= 16:
        raise ValueError("A bounded source audit permits 2-16 manual-linked recordings")
    policy = policy or RegistrationPolicy()
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("schema_version") != "author-direct-upload-v1":
        raise ValueError("Native canvas audit requires a versioned author-upload inventory")
    source = Path(inventory["source"]).resolve()
    assets = {
        asset.path: asset
        for asset in (UploadedAsset.model_validate(row) for row in inventory["assets"])
    }
    selected = []
    for line in episodes_path.open():
        episode = SnookerEpisode.model_validate_json(line)
        manuals = [a for a in episode.source_annotations if a.kind == "manual_path_image"]
        if len(manuals) > 1:
            raise ValueError("Source audit needs one unambiguous manual reference per recording")
        if manuals:
            selected.append((episode, manuals[0]))
    selected.sort(key=lambda pair: Path(pair[1].path).parent.name)
    selected = selected[:limit_recordings]
    if len(selected) < 2 or len({pair[0].source.group_id for pair in selected}) != len(selected):
        raise ValueError("Source audit requires distinct recording groups, not manual duplicates")
    inputs = CanvasInputReceipts(
        episodes_file=str(episodes_path.resolve()),
        episodes_sha256=file_sha256(episodes_path),
        inventory_file=str(inventory_path.resolve()),
        inventory_sha256=file_sha256(inventory_path),
        implementation_sha256=file_sha256(Path(__file__)),
        selected_source_groups=tuple(episode.source.group_id for episode, _ in selected),
    )
    input_hash = fingerprint(
        {"inputs": inputs.model_dump(mode="json"), "policy": policy.model_dump(mode="json")}
    )
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (episode, manual) in enumerate(selected):
        raw_path = _source_path(source, episode.source.original_file)
        if raw_path.suffix != ".dat":
            raise ValueError("Bounded source registration requires original raw BGR capture")
        asset = assets.get(str(raw_path.relative_to(source)))
        if (
            asset is None
            or not asset.stable
            or asset.sha256 is None
            or asset.sha256 != episode.source.original_sha256
        ):
            raise ValueError("Raw capture and immutable inventory provenance disagree")
        signature = _file_signature(raw_path)
        if signature[:2] != (asset.bytes, asset.mtime_ns):
            raise ValueError("Raw capture changed since the immutable source snapshot")
        manual_path = _source_path(source, manual.path)
        if file_sha256(manual_path) != manual.sha256:
            raise ValueError("Manual raster changed since its source snapshot")
        with Image.open(manual_path) as image:
            canvas = image.size
        render_path = _source_path(
            source, str(manual_path.parent / "TrackingResults/MOLT-VACHASPATI-LEGG.avi")
        )
        render_signature = _file_signature(render_path)
        width, height = episode.calibration.image_size_px
        frame_bytes = width * height * 3
        if not asset.bytes or asset.bytes % frame_bytes:
            raise ValueError("Raw capture does not contain integral complete source frames")
        raw_count = asset.bytes // frame_bytes
        frame_indices = (0, raw_count // 2)
        capture = cv2.VideoCapture(str(render_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or frame_count < raw_count:
            capture.release()
            raise ValueError("Source tracking render cannot supply the declared audit frames")
        frame_reports = []
        first_offset: tuple[int, int] | None = None
        try:
            with raw_path.open("rb") as stream:
                for frame_index in frame_indices:
                    stream.seek(frame_index * frame_bytes)
                    data = stream.read(frame_bytes)
                    if len(data) != frame_bytes:
                        raise ValueError("Raw capture changed during bounded sampling")
                    native = np.frombuffer(data, np.uint8).reshape(height, width, 3)
                    if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                        raise ValueError(
                            "Source tracking render does not support declared frame seek"
                        )
                    decoded, rendered = capture.read()
                    if not decoded:
                        raise ValueError("Source tracking render frame is not decodable")
                    patch = rendered_body(np.asarray(rendered, dtype=np.uint8), canvas, policy)
                    registration = register_source_crop(native, patch, policy)
                    if first_offset is None:
                        first_offset = registration.source_offset_px
                    discrepancy = float(
                        np.linalg.norm(np.subtract(first_offset, registration.source_offset_px))
                    )
                    frame_reports.append(
                        FrameMappingAudit(
                            frame_index=frame_index,
                            source_frame_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
                            rendered_patch_sha256=hashlib.sha256(patch.tobytes()).hexdigest(),
                            registration=registration,
                            frozen_origin_error_px=discrepancy,
                        )
                    )
        finally:
            capture.release()
        if (
            _file_signature(raw_path) != signature
            or _file_signature(render_path) != render_signature
        ):
            raise ValueError("Source assets changed during spatial registration")
        if first_offset is None:
            raise RuntimeError("No calibration frame was registered")
        calibration, held_out = frame_reports
        x, y = first_offset
        consistent = (
            held_out.registration.passed_policy
            and held_out.frozen_origin_error_px <= policy.validation_tolerance_px
        )
        row = RecordingMappingAudit(
            episode_id=episode.episode_id,
            source_group=episode.source.group_id,
            role="support" if index < 2 else "held_out",
            raw_capture=str(raw_path),
            source_render=str(render_path),
            manual_raster=str(manual_path),
            raw_snapshot_sha256=asset.sha256,
            manual_raster_sha256=manual.sha256,
            native_canvas_px=canvas,
            source_render_frame_count=frame_count,
            raw_frame_count=raw_count,
            crop_source_to_native=((1.0, 0.0, float(-x)), (0.0, 1.0, float(-y)), (0.0, 0.0, 1.0)),
            calibration=calibration,
            held_out_frame=held_out,
            temporal_mapping_consistent=consistent,
            spatial_registration_validated=calibration.registration.passed_policy and consistent,
        )
        rows.append(row)
        write_json(output / f"recording_{index:02d}.json", row.model_dump(mode="json"))
    result = CanvasAuditResult(
        input_fingerprint=input_hash,
        inputs=inputs,
        policy=policy,
        recordings=tuple(rows),
        support_recordings=min(2, len(rows)),
        held_out_recordings=max(0, len(rows) - 2),
        native_canvas_counts=dict(
            Counter(f"{row.native_canvas_px[0]}x{row.native_canvas_px[1]}" for row in rows)
        ),
        source_offset_counts=dict(
            Counter(str(row.calibration.registration.source_offset_px) for row in rows)
        ),
        validated_recordings=sum(row.spatial_registration_validated for row in rows),
        source_frames_read=2 * len(rows),
        source_render_frames_decoded=2 * len(rows),
        shared_single_crop_supported=len(
            {(row.native_canvas_px, row.calibration.registration.source_offset_px) for row in rows}
        )
        == 1,
    )
    write_json(output / "audit.json", result.model_dump(mode="json"))
    return result
