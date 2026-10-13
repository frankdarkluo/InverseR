from typing import Literal

import cv2
import numpy as np
from pydantic import Field

from inverser.schemas import Record


class CalibrationLandmark(Record):
    source_group: str = Field(min_length=1)
    source_xy_px: tuple[float, float]
    target_xy_px: tuple[float, float]
    evidence: Literal["independent_calibration_geometry"]


class SpatialMappingAudit(Record):
    matrix: tuple[tuple[float, float, float], ...]
    calibration_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    validation_errors_px: tuple[float, ...]
    max_validation_error_px: float
    tolerance_px: float
    spatial_mapping_validated: bool
    metric_world_accuracy_verified: Literal[False] = False
    timed_manual_trajectory_verified: Literal[False] = False
    measured_cue_action_available: Literal[False] = False
    physical_benchmark_accepted: Literal[False] = False


def validate_spatial_mapping(
    calibration: tuple[CalibrationLandmark, ...],
    validation: tuple[CalibrationLandmark, ...],
    *,
    tolerance_px: float,
) -> SpatialMappingAudit:
    if len(calibration) < 4 or len(validation) < 2:
        raise ValueError("Mapping needs at least four calibration and two held-out landmarks")
    if not np.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("Mapping tolerance must be positive and finite")
    calibration_groups = {point.source_group for point in calibration}
    validation_groups = {point.source_group for point in validation}
    if calibration_groups & validation_groups:
        raise ValueError("Calibration and validation source groups must be disjoint")
    source = np.asarray([point.source_xy_px for point in calibration], dtype=np.float64)
    target = np.asarray([point.target_xy_px for point in calibration], dtype=np.float64)
    matrix, _ = cv2.findHomography(source, target, method=0)
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Independent landmarks do not determine a nonsingular mapping")
    held_source = np.asarray([(*point.source_xy_px, 1.0) for point in validation], dtype=np.float64)
    projected = held_source @ matrix.T
    if np.any(np.abs(projected[:, 2]) < 1e-12):
        raise ValueError("Held-out landmarks project to infinity")
    projected = projected[:, :2] / projected[:, 2:]
    held_target = np.asarray([point.target_xy_px for point in validation], dtype=np.float64)
    errors = np.linalg.norm(projected - held_target, axis=1)
    if not np.isfinite(errors).all():
        raise ValueError("Mapping validation produced non-finite error")
    maximum = float(np.max(errors))
    return SpatialMappingAudit(
        matrix=tuple(
            (float(row[0]), float(row[1]), float(row[2]))
            for row in matrix
        ),
        calibration_groups=tuple(sorted(calibration_groups)),
        validation_groups=tuple(sorted(validation_groups)),
        validation_errors_px=tuple(float(value) for value in errors),
        max_validation_error_px=maximum,
        tolerance_px=tolerance_px,
        spatial_mapping_validated=maximum <= tolerance_px,
    )
