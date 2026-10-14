import json
import sqlite3

import pytest

from inverser import study_runner
from inverser.study_runner import EngineeringRunConfig, baseline_dry_run


def source_record(tmp_path, study_fixture):
    _, record, _ = study_fixture
    source = tmp_path / "privileged.jsonl"
    source.write_text(record.model_dump_json() + "\n")
    return source


def test_engineering_runner_is_complete_versioned_and_resumable(tmp_path, study_fixture):
    source = source_record(tmp_path, study_fixture)
    output = tmp_path / "run"
    config = EngineeringRunConfig(seed=7, objective_budget=2, candidate_count=2)
    first = baseline_dry_run(source, output, config)
    assert first["status"] == "complete_engineering_dry_run"
    assert first["completed_jobs"] == ["random", "sobol", "geometry_powell", "fixed_bank"]
    assert first["new_jobs_this_invocation"] == 4
    assert first["actual_model_inference"] is False
    assert first["model_performance_claim"] is False
    assert first["formal_evaluation"] is False
    assert first["paid_api_requests"] == first["new_benchmark_tasks_generated"] == 0
    assert first["existing_development_layouts"] == 1
    second = baseline_dry_run(source, output, config)
    assert second["new_jobs_this_invocation"] == 0
    assert second["counts"] == first["counts"]
    assert second["physics_usage_by_role"] == first["physics_usage_by_role"]
    assert set(first["physics_usage_by_role"]) == {
        "agent_search",
        "hidden_judge",
        "offline_reference",
    }
    assert second["artifacts"] == first["artifacts"]
    bank = json.loads((output / "fixed_bank.json").read_text())["payload"]
    assert bank["fixture"] == "oracle_forecast_for_engineering_only"
    assert bank["paired_moving_error"]["independent_groups"] == 1
    assert bank["paired_moving_error"]["confidence_interval"] is None
    for input_file in output.glob("candidate-*_public_input.json"):
        task = json.loads(input_file.read_text())["task"]
        assert "goal" not in task and "witness_action" not in task
    with pytest.raises(ValueError, match="new versioned"):
        baseline_dry_run(source, output, config.model_copy(update={"seed": 8}))


def test_job_limit_returns_explicit_partial_then_continues(tmp_path, study_fixture):
    source = source_record(tmp_path, study_fixture)
    config = EngineeringRunConfig(seed=9, objective_budget=2, candidate_count=2)
    output = tmp_path / "partial"
    partial = baseline_dry_run(source, output, config, limit_jobs=1)
    assert partial["status"] == "partial"
    assert partial["completed_jobs"] == ["random"]
    assert partial["counts"]["model_requests"] == 0
    complete = baseline_dry_run(source, output, config)
    assert complete["status"] == "complete_engineering_dry_run"
    assert complete["new_jobs_this_invocation"] == 3
    assert complete["counts"]["physics_evaluations"] <= 11


def test_interruption_after_replay_before_artifact_does_not_repeat_physics(
    tmp_path, study_fixture, monkeypatch
):
    source = source_record(tmp_path, study_fixture)
    output = tmp_path / "interrupted"
    config = EngineeringRunConfig(seed=9, objective_budget=2, candidate_count=2)
    original = study_runner._write_job

    def interrupted(path, job):
        raise RuntimeError("fixture interruption before job commit")

    monkeypatch.setattr(study_runner, "_write_job", interrupted)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        baseline_dry_run(source, output, config, limit_jobs=1)
    with sqlite3.connect(output / "ledger.sqlite") as connection:
        before = connection.execute("SELECT SUM(evaluations) FROM physics").fetchone()[0]
    monkeypatch.setattr(study_runner, "_write_job", original)
    resumed = baseline_dry_run(source, output, config, limit_jobs=1)
    assert resumed["counts"]["physics_evaluations"] == before
    result = json.loads((output / "random.json").read_text())["payload"]["search"]
    assert result["actual_new_physics_evaluations"] == 0
    assert all(attempt["cache_hit"] for attempt in result["attempts"])


def test_modified_completed_artifacts_are_not_silently_reused(tmp_path, study_fixture):
    source = source_record(tmp_path, study_fixture)
    output = tmp_path / "checksum"
    config = EngineeringRunConfig(seed=9, objective_budget=1, candidate_count=2)
    baseline_dry_run(source, output, config, limit_jobs=1)
    path = output / "random.json"
    artifact = json.loads(path.read_text())
    artifact["payload"]["search"]["selected_goal_success"] = not (
        artifact["payload"]["search"]["selected_goal_success"]
    )
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="checksum"):
        baseline_dry_run(source, output, config)


def test_engineering_bounds_are_not_an_unguarded_large_sweep():
    for values in (
        {"candidate_count": 100},
        {"objective_budget": 100},
        {"sample_times_s": (-1, 1)},
        {"sample_times_s": (1, 1)},
    ):
        with pytest.raises(ValueError):
            EngineeringRunConfig(**values)
