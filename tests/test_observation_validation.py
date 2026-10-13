import pytest
from pydantic import ValidationError

from inverser.observation_validation import CalibrationLandmark, validate_spatial_mapping


def landmark(x, y, group="calibration", shift=0):
    return CalibrationLandmark(
        source_group=group,
        source_xy_px=(x, y),
        target_xy_px=(2 * x + 10 + shift, 3 * y + 20),
        evidence="independent_calibration_geometry",
    )


def test_mapping_uses_held_out_sources_and_does_not_create_physical_gt():
    calibration = tuple(landmark(x, y) for x, y in ((0, 0), (100, 0), (100, 100), (0, 100)))
    validation = (landmark(20, 30, "query-a"), landmark(70, 60, "query-b"))
    report = validate_spatial_mapping(calibration, validation, tolerance_px=1)
    assert report.spatial_mapping_validated
    assert report.max_validation_error_px < 1e-6
    assert not report.physical_benchmark_accepted
    assert not report.metric_world_accuracy_verified
    assert not report.timed_manual_trajectory_verified
    wrong = (landmark(20, 30, "query-a", shift=2), landmark(70, 60, "query-b", shift=2))
    rejected = validate_spatial_mapping(calibration, wrong, tolerance_px=1)
    assert not rejected.spatial_mapping_validated
    assert rejected.max_validation_error_px == pytest.approx(2)


def test_calibration_rejects_target_leakage_and_degenerate_geometry():
    calibration = tuple(landmark(x, y) for x, y in ((0, 0), (100, 0), (100, 100), (0, 100)))
    with pytest.raises(ValueError, match="disjoint"):
        validate_spatial_mapping(calibration, (landmark(20, 30), landmark(70, 60)), tolerance_px=1)
    degenerate = tuple(landmark(x, 0) for x in (0, 10, 20, 30))
    with pytest.raises(ValueError, match="nonsingular"):
        validate_spatial_mapping(
            degenerate, (landmark(20, 30, "held"), landmark(70, 60, "held")), tolerance_px=1
        )
    with pytest.raises(ValidationError, match="evidence"):
        CalibrationLandmark(
            source_group="query",
            source_xy_px=(0, 0),
            target_xy_px=(0, 0),
            evidence="fitted_to_query_manual_path",
        )
