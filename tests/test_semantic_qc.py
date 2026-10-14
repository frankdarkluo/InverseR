import json
from pathlib import Path

import cv2
import pytest
from test_canvas_audit import textured_source, write_source_fixture

from inverser import semantic_qc
from inverser.canvas_audit import audit_author_canvases
from inverser.semantic_qc import (
    RecordingCommit,
    SemanticQCConfig,
    run_semantic_qc,
)


def inputs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(semantic_qc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(semantic_qc, "check_storage", lambda reserve: None)
    source = tmp_path / "sources"
    fixtures = [write_source_fixture(source, index) for index in range(3)]
    for index in range(3):
        assert cv2.imwrite(
            str(source / f"Snooker_Manual_{index}/background.png"),
            textured_source(seed=7 + index),
        )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": "author-direct-upload-v1",
                "source": str(source),
                "assets": [asset for _, asset in fixtures],
            }
        )
    )
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text("\n".join(episode.model_dump_json() for episode, _ in fixtures) + "\n")
    mapping = tmp_path / "mapping"
    audit_author_canvases(episodes, inventory, mapping, limit_recordings=3)
    return mapping / "audit.json", episodes


def test_bounded_pilot_records_ablations_and_does_not_promote_accuracy(tmp_path, monkeypatch):
    mapping, episodes = inputs(tmp_path, monkeypatch)
    output = tmp_path / "data/qc"
    config = SemanticQCConfig(workers=1)
    result = run_semantic_qc(mapping, episodes, output, config)
    assert result["status"] == "complete"
    assert result["completed_recordings"] == 3 and result["evaluation_recordings"] == 1
    assert result["evaluation_original_frames"] == 6
    assert result["actual_model_requests"] == result["human_annotations_added"] == 0
    assert result["g1_approved"] is False
    assert len(result["aggregate_macro_by_recording"]) == 5
    assert result["new_output_bytes"] < config.max_new_output_bytes
    for report_path in output.glob("*/report.json"):
        report = RecordingCommit.model_validate_json(report_path.read_text()).report
        assert not report.detection_accuracy_measured
        assert not report.complete_inventory_recall_measured
        assert not report.identity_accuracy_measured
        assert not report.benchmark_accepted
        profiles = {profile.pipeline: profile for profile in report.profiles}
        assert profiles["source_background"].path_iou == (
            profiles["source_background_temporal"].path_iou
        )
        assert profiles["source_background"].ontology_conflict_frame_fraction == (
            profiles["source_background_temporal"].ontology_conflict_frame_fraction
        )
    resumed = run_semantic_qc(mapping, episodes, output, config)
    assert resumed["new_recordings_this_invocation"] == 0
    assert resumed["aggregate_macro_by_recording"] == result["aggregate_macro_by_recording"]
    assert resumed["paired_path_iou_differences"] == result["paired_path_iou_differences"]


def test_support_only_partial_run_does_not_emit_evaluation_metrics(tmp_path, monkeypatch):
    mapping, episodes = inputs(tmp_path, monkeypatch)
    output = tmp_path / "data/partial"
    config = SemanticQCConfig(workers=1)
    result = run_semantic_qc(mapping, episodes, output, config, limit_recordings=2)
    assert result["status"] == "partial"
    assert result["support_recordings"] == 2 and result["evaluation_recordings"] == 0
    assert result["aggregate_macro_by_recording"] == []
    complete = run_semantic_qc(mapping, episodes, output, config)
    assert complete["status"] == "complete" and complete["new_recordings_this_invocation"] == 1
    assert (output / "per_recording.csv").is_file()


def test_allocation_and_completed_report_integrity_are_enforced(tmp_path, monkeypatch):
    mapping, episodes = inputs(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="allocation"):
        run_semantic_qc(
            mapping,
            episodes,
            tmp_path / "data/no_quota",
            SemanticQCConfig(workers=1, max_new_output_bytes=1),
        )
    output = tmp_path / "data/checksum"
    config = SemanticQCConfig(workers=1)
    run_semantic_qc(mapping, episodes, output, config, limit_recordings=1)
    report_path = next(output.glob("*/report.json"))
    data = json.loads(report_path.read_text())
    data["report"]["profiles"][0]["path_iou"] = 0.123456
    report_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        run_semantic_qc(mapping, episodes, output, config)
    for overrides in ({"workers": 5}, {"max_new_output_bytes": 20_000_000_001}):
        with pytest.raises(ValueError):
            SemanticQCConfig(**overrides)


def test_no_background_cue_gap_suite_keeps_fixed_source_colour_reference(tmp_path, monkeypatch):
    mapping, episodes = inputs(tmp_path, monkeypatch)
    output = tmp_path / "data/cue_gap"
    result = run_semantic_qc(
        mapping,
        episodes,
        output,
        SemanticQCConfig(workers=1, profile_suite="cue_gap_no_background"),
    )
    assert result["status"] == "complete"
    rows = {row["pipeline"]: row for row in result["aggregate_macro_by_recording"]}
    assert set(rows) == {
        "source_colour",
        "source_colour_temporal",
        "source_colour_compact_cue",
        "source_colour_compact_cue_temporal",
    }
    assert rows["source_colour"]["path_iou"] == rows["source_colour_temporal"]["path_iou"]
    assert (
        rows["source_colour_compact_cue"]["path_iou"]
        == (rows["source_colour_compact_cue_temporal"]["path_iou"])
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["reference_pipeline"] == "source_colour"
    assert not any(profile["use_background"] for profile in manifest["pipelines"])
