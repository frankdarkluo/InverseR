import json

import pytest
from test_diagnostic_panel import source_inputs

from inverser import diagnostic_panel, planning_pilot
from inverser.azure_study import PilotRow
from inverser.diagnostic_panel import (
    DiagnosticPanelConfig,
    PanelRequest,
    load_synthetic_panel_call,
    prepare_diagnostic_panel,
)
from inverser.metrics import score_endpoint
from inverser.model_adapters import ModelReply, TokenUsage
from inverser.planning_pilot import PlanningPilotConfig, run_planning_pilot, select_planning_cases
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.study import GoalSubmission
from inverser.study_runner import JobEnvelope


def planning_inputs(tmp_path, monkeypatch, study_fixture):
    monkeypatch.setattr(diagnostic_panel, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(diagnostic_panel, "check_storage", lambda reserve: None)
    monkeypatch.setattr(planning_pilot, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(planning_pilot, "check_storage", lambda reserve: None)
    panel = tmp_path / "data/panel"
    prepare_diagnostic_panel(
        *source_inputs(tmp_path, study_fixture),
        panel,
        DiagnosticPanelConfig(synthetic_layouts=4, real_round_cases=4),
    )
    simulator = study_fixture[0]
    native = tmp_path / "data/native"
    run_hash = fingerprint("oracle-native-fixture-for-engineering-only")
    write_json(
        native / "manifest.json",
        {
            "panel_manifest_sha256": file_sha256(panel / "manifest.json"),
            "config": {"design": "goal_structured"},
            "run_hash": run_hash,
        },
    )
    for request in (
        PanelRequest.model_validate_json(line) for line in (panel / "request_index.jsonl").open()
    ):
        if request.task != "synthetic_goal" or request.view != "structured":
            continue
        goal = load_synthetic_panel_call(panel, request).input
        label = json.loads((panel / "private" / f"{request.case_id}.json").read_text())
        submission = GoalSubmission(action=label["record"]["witness_action"])
        score = score_endpoint(
            simulator.replay(goal.initial, submission.action).goal,
            goal.goal,
            goal.physics.table.ball_radius_m,
        )
        for model in ("gpt-5-mini", "gpt-5"):
            invalid = model == "gpt-5-mini" and label["family"] == "side"
            row = PilotRow(
                model=model,
                request_id=request.request_id,
                case_id=request.case_id,
                task=request.task,
                condition="goal",
                view="structured",
                valid_output=not invalid,
                elapsed_s=0,
                reply=ModelReply(
                    backend_alias=model,
                    model_id=model,
                    content=('{"invalid":1}' if invalid else submission.model_dump_json()),
                    usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                ),
                goal_score=None if invalid else score,
                error_code="invalid_submission" if invalid else None,
            )
            payload = row.model_dump(mode="json")
            write_json(
                native / "jobs" / model / f"{request.request_id}.json",
                {
                    "run_hash": run_hash,
                    "payload": payload,
                    "payload_sha256": fingerprint(payload),
                },
            )
    return panel, native


def test_planning_protocol_bounds_and_public_source_matching(tmp_path, monkeypatch, study_fixture):
    panel, native = planning_inputs(tmp_path, monkeypatch, study_fixture)
    config = PlanningPilotConfig(layouts=4, workers=1, checkpoints=(2, 4))
    cases = select_planning_cases(panel, native, config)
    assert len(cases) == len({case.initial_hash for case in cases}) == 4
    assert len({case.source_group for case in cases}) == 4
    assert set(case.family for case in cases) == set(diagnostic_panel.FAMILIES)
    assert cases == select_planning_cases(panel, native, config)
    assert all(case.request.view == "structured" and not case.request.local_only for case in cases)
    with pytest.raises(ValueError, match="balanced"):
        PlanningPilotConfig(layouts=5)
    with pytest.raises(ValueError, match="increasing"):
        PlanningPilotConfig(checkpoints=(4, 2))
    source = native / "manifest.json"
    value = json.loads(source.read_text())
    value["panel_manifest_sha256"] = "0" * 64
    write_json(source, value)
    with pytest.raises(ValueError, match="identical public panel"):
        select_planning_cases(panel, native, config)


def test_planning_budget_failures_resume_and_hidden_judge_isolation(
    tmp_path, monkeypatch, study_fixture
):
    panel, native = planning_inputs(tmp_path, monkeypatch, study_fixture)
    config = PlanningPilotConfig(layouts=4, workers=1, checkpoints=(2, 4))
    output = tmp_path / "data/planning"
    partial = run_planning_pilot(panel, native, output, config, limit_cases=1)
    assert partial["status"] == "partial" and partial["completed_jobs"] == 8
    report = run_planning_pilot(panel, native, output, config)
    assert report["status"] == "completed" and report["completed_jobs"] == 32
    assert report["new_model_requests"] == report["new_model_tokens"] == 0
    assert set(report["physics_usage_by_role"]) == {"agent_search", "hidden_judge"}
    for row in report["by_method_budget_family"]:
        if row["family"] is None:
            assert row["layouts"] == 4
        if row["method"] == "gpt-5-mini_powell" and row["family"] == "side":
            assert row["outcomes"] == {"invalid_native_seed": 1}
    for path in (output / "cases").rglob("*.json"):
        job = JobEnvelope.model_validate_json(path.read_text()).payload
        if job["search"] is not None:
            assert len(job["search"]["attempts"]) <= 4
        assert all(
            point["objective_calls"] <= point["query_budget"] for point in job["checkpoints"]
        )
    monkeypatch.setattr(
        planning_pilot,
        "SnookerSimulator",
        lambda *args: pytest.fail("Completed planning cases must not construct a simulator"),
    )
    resumed = run_planning_pilot(panel, native, output, config)
    assert resumed["new_jobs"] == resumed["new_physics_evaluations"] == 0
    assert resumed["physics_usage_by_role"] == report["physics_usage_by_role"]
    with pytest.raises(ValueError, match="new run version"):
        run_planning_pilot(panel, native, output, config.model_copy(update={"seed": 9}))


def test_planning_rejects_completed_job_corruption(tmp_path, monkeypatch, study_fixture):
    panel, native = planning_inputs(tmp_path, monkeypatch, study_fixture)
    config = PlanningPilotConfig(layouts=4, workers=1, checkpoints=(1,))
    output = tmp_path / "data/planning"
    run_planning_pilot(panel, native, output, config, limit_cases=1)
    path = next((output / "cases").rglob("geometry_direct.json"))
    value = json.loads(path.read_text())
    value["payload"]["method"] = "corrupted"
    write_json(path, value)
    with pytest.raises(ValueError, match="checksum mismatch"):
        run_planning_pilot(panel, native, output, config, limit_cases=1)
