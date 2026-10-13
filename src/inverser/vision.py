import logging
import math
from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from inverser.schemas import BallKind, PhysicsConfig

ImageArray = NDArray[np.uint8]
ObservedKind = BallKind | Literal["unknown"]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    kind: ObservedKind
    xy_m: tuple[float, float]
    center_px: tuple[float, float]
    shape_score: float
    radius_ratio: float
    track_id: str | None = None

    def json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FrameObservation:
    time_s: float
    detections: tuple[Detection, ...]
    felt_fraction: float
    large_foreground_fraction: float
    unresolved_components: int
    view_valid: bool
    visibility_clear: bool
    max_matched_motion_m: float | None
    median_matched_motion_m: float | None
    inventory_changed: bool
    possible_camera_motion: bool
    cue_motion_m: float | None = None
    matched_ball_count: int = 0

    def json(self) -> dict[str, object]:
        return asdict(self)


class TableProjection:
    def __init__(
        self,
        points: list[list[float]],
        source_width: int,
        source_height: int,
        *,
        output_width: int = 512,
        table_width_m: float = 1.778,
        table_length_m: float = 3.569,
        ball_radius_m: float = 0.02619375,
        corner_order: Literal["far_left_first", "ordered_short_edge"] = "far_left_first",
    ) -> None:
        corners = np.asarray(points, dtype=np.float32)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError("Table calibration requires four finite image-coordinate pairs")
        if not (
            np.all(corners[:, 0] >= 0)
            and np.all(corners[:, 0] < source_width)
            and np.all(corners[:, 1] >= 0)
            and np.all(corners[:, 1] < source_height)
        ):
            raise ValueError("Corner coordinates are outside the source image")
        if not cv2.isContourConvex(corners.astype(np.int32)):
            raise ValueError("Table corners must form a convex quadrilateral")
        if corner_order == "far_left_first" and not (
            corners[0, 0] < corners[1, 0]
            and corners[3, 0] < corners[2, 0]
            and corners[:2, 1].mean() < corners[2:, 1].mean()
        ):
            raise ValueError("Expected supplied far-left, far-right, near-right, near-left order")
        if output_width < 256 or output_width > 1024:
            raise ValueError("Projection width must be 256-1024 pixels")
        self.physics = PhysicsConfig(
            table_width_m=table_width_m,
            table_length_m=table_length_m,
            ball_radius_m=ball_radius_m,
        )
        self.corner_order = corner_order
        self.width = output_width
        self.height = round(output_width * self.physics.table_length_m / self.physics.table_width_m)
        self.radius_px = self.physics.ball_radius_m / self.physics.table_width_m * (self.width - 1)
        self.source_width = source_width
        self.source_height = source_height
        self.points = corners
        self.matrix = cv2.getPerspectiveTransform(
            corners,
            np.asarray(
                [
                    [0, 0],
                    [self.width - 1, 0],
                    [self.width - 1, self.height - 1],
                    [0, self.height - 1],
                ],
                np.float32,
            ),
        )
        self.interior = np.zeros((self.height, self.width), dtype=np.uint8)
        border = round(self.radius_px * 1.3)
        self.interior[border:-border, border:-border] = 1
        self.source_interior = cv2.warpPerspective(
            self.interior,
            np.linalg.inv(self.matrix),
            (source_width, source_height),
            flags=cv2.INTER_NEAREST,
        )
        roi_y, roi_x = np.nonzero(self.source_interior)
        if not len(roi_x):
            raise ValueError("Table projection produced an empty source-image ROI")
        self.source_bounds = (
            max(0, int(roi_x.min()) - 3),
            max(0, int(roi_y.min()) - 3),
            min(source_width, int(roi_x.max()) + 4),
            min(source_height, int(roi_y.max()) + 4),
        )

    def warp(self, frame: ImageArray) -> ImageArray:
        return np.asarray(
            cv2.warpPerspective(frame, self.matrix, (self.width, self.height)), dtype=np.uint8
        )

    def xy_m(self, x: float, y: float) -> tuple[float, float]:
        return (
            x / (self.width - 1) * self.physics.table_width_m,
            (1 - y / (self.height - 1)) * self.physics.table_length_m,
        )

    def source_to_bev(self, x: float, y: float) -> tuple[float, float]:
        point = self.matrix @ np.asarray([x, y, 1.0])
        if abs(point[2]) < 1e-12:
            raise ValueError("Image point projects to an invalid homogeneous coordinate")
        return float(point[0] / point[2]), float(point[1] / point[2])

    def source_ball_radius(self, x: float, y: float) -> float:
        first = self.xy_m(*self.source_to_bev(x, y))
        second = self.xy_m(*self.source_to_bev(x + 1, y))
        metres_per_pixel = math.dist(first, second)
        if metres_per_pixel <= 0:
            raise ValueError("Invalid local image-to-table scale")
        return self.physics.ball_radius_m / metres_per_pixel

    def metadata(self) -> dict[str, object]:
        return {
            "source_corners": self.points.tolist(),
            "homography": self.matrix.tolist(),
            "bev_pixels": [self.width, self.height],
            "table_m": [self.physics.table_width_m, self.physics.table_length_m],
            "projection_model": "planar table homography applied to detected ball-image centers",
            "source_corner_order": self.corner_order,
            "lens_distortion_corrected": False,
            "ball_center_height_corrected": False,
            "physical_error_calibrated": False,
        }


def classify_hsv(h: float, s: float, v: float) -> ObservedKind:
    if v < 75:
        return "black"
    if s < 55 and v > 145:
        return "cue"
    if (h < 13 or h > 170) and s >= 150:
        return "red"
    if (h > 145 or h < 13) and 55 <= s < 150 and v > 140:
        return "pink"
    if 13 <= h <= 35:
        return "yellow" if v > 150 and h >= 18 else "brown"
    if 35 < h < 90:
        return "green"
    if 90 <= h <= 140:
        return "blue"
    return "unknown"


def representative_hsv(colors: NDArray[np.uint8]) -> tuple[float, float, float]:
    saturation = float(np.median(colors[:, 1]))
    value = float(np.median(colors[:, 2]))
    if saturation < 55 or value < 60:
        return 0.0, saturation, value
    chromatic = colors[(colors[:, 1] >= 70) & (colors[:, 2] >= 60)]
    if not len(chromatic):
        return float(np.median(colors[:, 0])), saturation, value
    angles = chromatic[:, 0].astype(float) * math.pi / 90
    weights = chromatic[:, 1].astype(float)
    hue = math.atan2(
        float(np.sum(np.sin(angles) * weights)),
        float(np.sum(np.cos(angles) * weights)),
    ) % (2 * math.pi)
    return hue * 90 / math.pi, saturation, value


def detect_balls(
    rectified: ImageArray, projection: TableProjection, *, native_image: bool = False
) -> tuple[tuple[Detection, ...], float, float, int]:
    offset_x = offset_y = 0
    mask = projection.source_interior if native_image else projection.interior
    if native_image:
        offset_x, offset_y, right, bottom = projection.source_bounds
        rectified = rectified[offset_y:bottom, offset_x:right]
        mask = mask[offset_y:bottom, offset_x:right]
    hsv = np.asarray(cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV), dtype=np.uint8)
    h, s, v = (hsv[:, :, index].astype(np.float32) for index in range(3))
    interior = mask.astype(bool)
    felt = (h >= 35) & (h <= 95) & (s >= 40) & (v >= 35) & interior
    felt_fraction = float(np.count_nonzero(felt) / np.count_nonzero(interior))
    if not np.any(felt):
        return (), felt_fraction, 1.0, 0
    median = np.median(hsv[felt], axis=0)
    hue_distance = np.minimum(np.abs(h - median[0]), 180 - np.abs(h - median[0]))
    foreground = (
        (hue_distance > 12) | (s < 60) | (v < 0.25 * median[2]) | (s > median[1] + 50)
    ) & interior
    binary = foreground.astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    large_area = 0.0
    unresolved = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        local_radius = (
            projection.source_ball_radius(x + width / 2 + offset_x, y + height / 2 + offset_y)
            if native_image
            else projection.radius_px
        )
        reference_area = math.pi * local_radius**2
        if area < reference_area * 0.3:
            continue
        if area > reference_area * 2.3:
            large_area += area
            unresolved += 1
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * math.pi * area / perimeter**2
        if circularity < 0.45 or max(width, height) / min(width, height) > 1.8:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center_x = float(moments["m10"] / moments["m00"])
        center_y = float(moments["m01"] / moments["m00"])
        yy, xx = np.ogrid[y : y + height, x : x + width]
        core = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= (0.6 * local_radius) ** 2
        colors = hsv[y : y + height, x : x + width][core]
        if not len(colors):
            unresolved += 1
            LOGGER.debug(
                "Rejected under-resolved ball component at (%.2f, %.2f), radius %.3f pixels",
                center_x + offset_x,
                center_y + offset_y,
                local_radius,
            )
            continue
        kind = classify_hsv(*representative_hsv(colors))
        bev_x, bev_y = (
            projection.source_to_bev(center_x + offset_x, center_y + offset_y)
            if native_image
            else (center_x, center_y)
        )
        detections.append(
            Detection(
                kind=kind,
                xy_m=projection.xy_m(bev_x, bev_y),
                center_px=(bev_x, bev_y),
                shape_score=float(min(circularity, 1.0)),
                radius_ratio=float(math.sqrt(area / reference_area)),
            )
        )
    return (
        tuple(sorted(detections, key=lambda detection: detection.center_px)),
        felt_fraction,
        float(large_area / np.count_nonzero(interior)),
        unresolved,
    )


class BallAssociator:
    def __init__(self) -> None:
        self.previous: tuple[Detection, ...] = ()
        self.previous_time: float | None = None
        self.next_id = 0
        self.cue_motion_m: float | None = None
        self.matched_ball_count = 0

    def update(
        self, detections: tuple[Detection, ...], time_s: float
    ) -> tuple[tuple[Detection, ...], float | None, float | None, bool]:
        from dataclasses import replace

        if self.previous_time is not None and time_s <= self.previous_time:
            raise ValueError("Observation timestamps must increase")
        dt = time_s - self.previous_time if self.previous_time is not None else 0
        matches: dict[int, str] = {}
        distances: list[float] = []
        self.cue_motion_m = None
        if self.previous and detections and 0 < dt <= 0.5:
            cost = np.asarray(
                [
                    [
                        math.dist(old.xy_m, new.xy_m) if old.kind == new.kind else 1e6
                        for new in detections
                    ]
                    for old in self.previous
                ]
            )
            old_indices, new_indices = linear_sum_assignment(cost)
            for old_index, new_index in zip(old_indices, new_indices, strict=True):
                distance = float(cost[old_index, new_index])
                old = self.previous[old_index]
                if distance <= min(0.6, 3 * dt + 0.01) and old.track_id is not None:
                    matches[int(new_index)] = old.track_id
                    distances.append(distance)
                    if old.kind == "cue":
                        self.cue_motion_m = distance
        tracked = []
        for index, detection in enumerate(detections):
            track_id = matches.get(index)
            if track_id is None:
                track_id = f"track_{self.next_id:05d}"
                self.next_id += 1
            tracked.append(replace(detection, track_id=track_id))
        changed = (
            len(matches) != len(self.previous)
            or len(matches) != len(detections)
            or self.previous_time is None
        )
        self.previous = tuple(tracked)
        self.previous_time = time_s
        self.matched_ball_count = len(matches)
        return (
            self.previous,
            max(distances) if distances else None,
            float(np.median(distances)) if distances else None,
            changed,
        )


def observe_frame(
    frame: ImageArray,
    projection: TableProjection,
    associator: BallAssociator,
    time_s: float,
) -> tuple[FrameObservation, ImageArray]:
    rectified = projection.warp(frame)
    # A plane homography stretches sphere silhouettes; detect in the source image first.
    balls, felt, clutter, unresolved = detect_balls(frame, projection, native_image=True)
    tracked, maximum, median, changed = associator.update(balls, time_s)
    camera_motion = (
        median is not None and len(balls) >= 5 and median > 0.5 * projection.physics.ball_radius_m
    )
    view_valid = felt >= 0.70 and not camera_motion
    clear = (
        view_valid
        and clutter < 0.025
        and len(balls) <= 22
        and sum(ball.kind == "cue" for ball in balls) == 1
        and not any(ball.kind == "unknown" for ball in balls)
    )
    return (
        FrameObservation(
            time_s,
            tracked,
            felt,
            clutter,
            unresolved,
            view_valid,
            clear,
            maximum,
            median,
            changed,
            camera_motion,
            associator.cue_motion_m,
            associator.matched_ball_count,
        ),
        rectified,
    )


def draw_observations(rectified: ImageArray, observation: FrameObservation) -> ImageArray:
    image = rectified.copy()
    for ball in observation.detections:
        x, y = map(round, ball.center_px)
        cv2.circle(image, (x, y), 9, (0, 255, 255), 1)
        cv2.putText(
            image,
            f"{ball.kind}:{(ball.track_id or '').removeprefix('track_')}",
            (x + 9, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1,
        )
    cv2.putText(
        image,
        f"t={observation.time_s:.2f} felt={observation.felt_fraction:.2f}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    return image
