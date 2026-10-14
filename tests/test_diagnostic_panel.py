import json

import pytest
from pydantic import ValidationError

from inverser import diagnostic_panel
from inverser.diagnostic_panel import (
    DiagnosticPanelConfig,
    PanelRequest,
    RealForecastLabels,
    intervention_action,
    load_synthetic_panel_call,
    prepare_diagnostic_panel,
    select_synthetic_records,
)
from inverser.schemas import BallPosition, Layout
from inverser.world_model_data import FactualForecastInput, PrefixPoint


def source_inputs(tmp_path, study_fixture):
    _, record, _ = study_fixture
    synthetic = tmp_path / "synthetic.jsonl"
    records = []
    for index, family in enumerate(("free_roll", "center", "vertical", "side")):
        initial = Layout(
            balls=(BallPosition(ball_id="cue", kind="cue", xy_m=(0.6, 0.8 + index * 0.1)),)
        )
        shifted_goal = record.goal.model_copy(
            update={
                "balls": tuple(
                    ball.model_copy(update={"xy_m": (ball.xy_m[0], ball.xy_m[1] + index * 0.1)})
                    for ball in record.goal.balls
                )
            }
        )
        records.append(
            record.model_copy(
                update={
                    "initial": initial,
                    "goal": shifted_goal,
                    "task_id": f"source-{index}",
                    "group_id": f"group-{index}",
                    "family": family,
                }
            )
        )
    synthetic.write_text("\n".join(r.model_dump_json() for r in records) + "\n")
    public = tmp_path / "prefix.jsonl"
    private = tmp_path / "future.jsonl"
    public_rows = []
    private_rows = []
    for index in range(6):
        data = FactualForecastInput(
            case_id=f"real-{index}",
            observed_prefix=tuple(PrefixPoint(time_s=i * 0.1, xy_cm=(i, i)) for i in range(10)),
            query_times_s=(1.2, 1.5, 1.9),
        )
        labels = RealForecastLabels(
            case_id=data.case_id,
            source_file="original.xml",
            source_sha256="a" * 64,
            source_group=f"round-{index}",
            source_label="cue ball",
            future=tuple(PrefixPoint(time_s=t, xy_cm=(20, 20)) for t in data.query_times_s),
        )
        public_rows.append(data.model_dump_json())
        private_rows.append(labels.model_dump_json())
    public.write_text("\n".join(public_rows) + "\n")
    private.write_text("\n".join(private_rows) + "\n")
    return synthetic, public, private


def test_small_panel_exact_request_shape_and_blind_input_isolation(
    tmp_path, monkeypatch, study_fixture
):
    monkeypatch.setattr(diagnostic_panel, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(diagnostic_panel, "check_storage", lambda reserve: None)
    sources = source_inputs(tmp_path, study_fixture)
    output = tmp_path / "data/panel"
    config = DiagnosticPanelConfig(synthetic_layouts=4, real_round_cases=4)
    report = prepare_diagnostic_panel(*sources, output, config)
    assert report["request_records"] == 28
    assert report["synthetic_request_records"] == 24
    assert report["real_local_only_request_records"] == 4
    assert report["actual_model_calls"] == 0 and not report["formal_test_freeze"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["synthetic_family_counts"] == dict.fromkeys(
        ("free_roll", "center", "vertical", "side"), 1
    )
    assert manifest["physics_replays"] == 8
    selection = json.loads((output / "private/selection.json").read_text())
    assert len({row["initial_hash"] for row in selection["synthetic"]}) == 4
    assert len({row["source_group"] for row in selection["real"]}) == 4
    for line in (output / "request_index.jsonl").open():
        request = PanelRequest.model_validate_json(line)
        payload = json.loads((output / request.payload_file).read_text())
        if request.task == "real_factual_forecast":
            assert "future" not in payload and request.local_only
            with pytest.raises(ValueError, match="local-only adapter"):
                load_synthetic_panel_call(output, request)
        else:
            call = load_synthetic_panel_call(output, request)
            assert call.seed == 0
            if request.task == "synthetic_forward":
                assert "goal" not in payload["task"]
                assert not any("goal.png" in name for name in request.image_files)
            if request.view == "bev":
                assert "initial" not in payload["task"]
            assert "witness_action" not in str(payload)
    resumed = prepare_diagnostic_panel(*sources, output, config)
    assert resumed == report


def test_panel_integrity_and_version_lock(tmp_path, monkeypatch, study_fixture):
    monkeypatch.setattr(diagnostic_panel, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(diagnostic_panel, "check_storage", lambda reserve: None)
    sources = source_inputs(tmp_path, study_fixture)
    output = tmp_path / "data/panel"
    config = DiagnosticPanelConfig(synthetic_layouts=4, real_round_cases=4)
    prepare_diagnostic_panel(*sources, output, config)
    with pytest.raises(ValueError, match="new diagnostic-panel"):
        prepare_diagnostic_panel(*sources, output, config.model_copy(update={"seed": 3}))
    path = next((output / "public").rglob("*_structured.json"))
    path.write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        prepare_diagnostic_panel(*sources, output, config)


def test_interventions_change_one_declared_cue_control(study_fixture):
    _, record, _ = study_fixture
    for family in ("free_roll", "center", "vertical", "side"):
        value = record.model_copy(update={"family": family})
        action, name = intervention_action(value)
        changed = {
            key
            for key, value in action.model_dump().items()
            if value != record.witness_action.model_dump()[key]
        }
        assert changed == {name}


def test_panel_size_and_source_selection_are_explicit(tmp_path, study_fixture):
    sources = source_inputs(tmp_path, study_fixture)
    config = DiagnosticPanelConfig(synthetic_layouts=4, real_round_cases=4)
    assert len(select_synthetic_records(sources[0], config)) == 4
    with pytest.raises(ValueError, match="too few"):
        select_synthetic_records(sources[0], config.model_copy(update={"synthetic_layouts": 8}))
    with pytest.raises(ValidationError):
        DiagnosticPanelConfig(synthetic_layouts=5)
    with pytest.raises(ValidationError, match="contained public"):
        PanelRequest(
            request_id="bad",
            case_id="bad",
            task="synthetic_forward",
            condition="base_action",
            view="structured",
            payload_file="private/answer.json",
            local_only=False,
            model_adapter_ready=True,
            api_common_subset=True,
        )
