from pathlib import Path

import pytest

from inverser.analytics_data import (
    annotation_time_seconds,
    audit_analytics_data,
    read_analytics_layout,
    read_analytics_trajectory,
    spreadsheet_xml_rows,
)


def workbook(path: Path, rows: tuple[tuple[str, ...], ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    row_xml = "".join(
        "<Row>" + "".join(f"<Cell><Data>{value}</Data></Cell>" for value in row) + "</Row>"
        for row in rows
    )
    path.write_text(
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" '
        'xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">'
        '<Worksheet ss:Name="Source video 2019"><Table>'
        + row_xml
        + "</Table></Worksheet></Workbook>"
    )


def test_source_video_clock_is_not_a_cue_action_or_fps_label():
    assert annotation_time_seconds("4:52.19") == pytest.approx(292.19)
    assert annotation_time_seconds("1:04:52.19") == pytest.approx(3892.19)
    assert annotation_time_seconds("") is None
    assert annotation_time_seconds("26.56", allow_seconds_only=True) == pytest.approx(26.56)
    assert annotation_time_seconds("1899-12-31T01:01:17.040") == pytest.approx(3677.04)
    with pytest.raises(ValueError):
        annotation_time_seconds("26.56")
    with pytest.raises(ValueError):
        annotation_time_seconds("2026-10-14T01:01:17.040")
    for text in ("4:70.20", "1:70:00", "12frames", "NaN"):
        with pytest.raises(ValueError):
            annotation_time_seconds(text)


def test_layout_keeps_units_and_ball_identity_unverified(tmp_path):
    path = tmp_path / "Coordinates/2019 tournament/match/Frame 1.xml"
    workbook(
        path,
        (
            ("Points",),
            ("Name", "X", "Y", "Time"),
            ("Marker 1", "21.75", "67.11", "4:52.19"),
            ("Marker 2", "-4.5", "90", "4:52.19"),
        ),
    )
    layout = read_analytics_layout(path, tmp_path / "Coordinates")
    assert layout.coordinate_basis == "source_native_units_unverified"
    assert not layout.marker_to_ball_identity_verified
    assert not layout.main_snooker_benchmark_eligible
    assert not layout.measured_cue_action_available
    assert "negative_source_coordinate_status_unverified" in layout.flags
    assert layout.points[0].annotation_video_time_s == pytest.approx(292.19)


def test_actual_track_schema_supports_factual_time_and_cm_but_not_interventions(tmp_path):
    path = tmp_path / "trajectories/tournament/round1/object2/cue ball.xml"
    workbook(
        path,
        (
            ("Track",),
            ("Label :", "cue ball"),
            ("Coords (x,y:cm; t:time)",),
            ("x", "y", "t"),
            ("21.75", "67.11", "4:52.19"),
            ("22.98", "71.92", "4:52.22"),
        ),
    )
    track = read_analytics_trajectory(path, tmp_path / "trajectories")
    assert track.timestamps_strictly_increasing
    assert track.coordinate_basis == "source_declared_centimetres"
    assert track.samples[0].xy_cm == (21.75, 67.11)
    assert not track.complete_scene_inventory_available
    assert not track.action_conditioned_intervention_ground_truth
    assert not track.source_spatial_calibration_independently_verified


def test_missing_units_and_entity_xml_are_explicit_failures(tmp_path):
    path = tmp_path / "track.xml"
    workbook(
        path,
        (
            ("Label :", "cue ball"),
            ("x", "y", "t"),
            ("21", "67", "4:52.19"),
            ("22", "70", "4:52.22"),
        ),
    )
    with pytest.raises(ValueError, match="centimetre"):
        read_analytics_trajectory(path, tmp_path)
    path.write_text('<!DOCTYPE Workbook [<!ENTITY x "unsafe">]><Workbook/>')
    with pytest.raises(ValueError, match="entity"):
        spreadsheet_xml_rows(path)


def test_office_lock_is_not_a_data_workbook_and_unverified_clock_is_preserved(tmp_path):
    from openpyxl import Workbook

    dataset = tmp_path / "dataset"
    path = dataset / "Coordinates/2019 tournament/match/Frame 1.xml"
    workbook(path, (("Name", "X", "Y", "Time"), ("Marker 1", "21", "67", "0.00")))
    variables = dataset / "Variables/match"
    variables.mkdir(parents=True)
    source = Workbook()
    source.active.append(["Frame", "Who BREAK"])
    source.active.append([1, "source-player"])
    source.save(variables / "Variables.xlsx")
    source.close()
    (variables / "~$Variables.xlsx").write_bytes(b"source-office-lock")
    output = tmp_path / "audit"
    summary = audit_analytics_data(dataset, output)
    assert summary["parsed_layout_records"] == 1
    assert summary["variable_workbooks"] == 1 and summary["ignored_office_lock_files"] == 1
    layout = read_analytics_layout(path, dataset / "Coordinates")
    assert layout.points[0].time_text == "0.00"
    assert layout.points[0].annotation_video_time_s is None
    assert "unparsed_source_clock" in layout.flags
