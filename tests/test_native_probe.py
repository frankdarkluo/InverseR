import json

import httpx
import pytest
from test_azure_study import azure_inventory
from test_diagnostic_panel import source_inputs

from inverser import diagnostic_panel, native_probe, study_calls
from inverser.diagnostic_panel import DiagnosticPanelConfig, prepare_diagnostic_panel
from inverser.native_probe import NativeProbeConfig, run_native_probe
from inverser.study import GoalSubmission


def probe_panel(tmp_path, monkeypatch, study_fixture):
    for module in (diagnostic_panel, study_calls):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(module, "check_storage", lambda reserve: None)
    panel = tmp_path / "data/panel"
    prepare_diagnostic_panel(
        *source_inputs(tmp_path, study_fixture),
        panel,
        DiagnosticPanelConfig(synthetic_layouts=4, real_round_cases=4),
    )
    return panel


def prompt_task(request):
    return json.loads(json.loads(request.content)["messages"][1]["content"][0]["text"])["task"]


def truth_handler(panel, calls):
    def handler(request):
        calls.append(request)
        body = json.loads(request.content)
        task = prompt_task(request)
        labels = json.loads((panel / "private" / f"{task['case_id']}.json").read_text())
        if "action" in task:
            content = json.dumps(
                next(
                    item["truth"]
                    for item in labels["forward_labels"].values()
                    if item["action"] == task["action"]
                )
            )
        else:
            content = GoalSubmission(action=labels["record"]["witness_action"]).model_dump_json()
        output = 1000 if body["reasoning_effort"] == "high" else 10
        return httpx.Response(
            200,
            json={
                "model": request.url.path.split("/")[-3] + "-2025-08-07",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": output,
                    "total_tokens": 5 + output,
                },
            },
        )

    return handler


def execute_kwargs(panel, calls):
    return {
        "endpoint_index": 0,
        "execute": True,
        "token_provider": lambda: "fixture-token",
        "transport": httpx.MockTransport(truth_handler(panel, calls)),
    }


def test_effort_probe_plans_first_then_bills_each_effort_once(tmp_path, monkeypatch, study_fixture):
    panel = probe_panel(tmp_path, monkeypatch, study_fixture)
    source = azure_inventory(tmp_path)
    output = tmp_path / "data/probe"
    calls = []
    config = NativeProbeConfig(
        layouts=4, reasoning_efforts=("minimal", "high"), max_model_requests=48
    )
    plan = run_native_probe(panel, source, output, config, endpoint_index=0)
    assert plan["status"] == "planned_no_calls" and plan["planned_requests"] == 48 and not calls
    kwargs = execute_kwargs(panel, calls)
    partial = run_native_probe(panel, source, output, config, limit_jobs=3, **kwargs)
    assert partial["status"] == "partial" and partial["new_model_requests"] == 3
    report = run_native_probe(panel, source, output, config, **kwargs)
    assert report["status"] == "completed" and len(calls) == 48
    bodies = [json.loads(request.content) for request in calls]
    assert {body["reasoning_effort"] for body in bodies} == {"minimal", "high"}
    assert all(body["max_completion_tokens"] == 32768 for body in bodies)
    assert not any("image_url" in json.dumps(body) for body in bodies)
    assert len(report["by_arm"]) == 4
    for arm in report["by_arm"]:
        assert arm["valid_outputs"] == arm["attempted_requests"] == 12
        assert arm["goal_success_by_tolerance_radii"]["1.0"] == 4
        assert arm["resting_success_by_tolerance_radii"]["0.25"] == 8
        expected = 1000 if arm["reasoning_effort"] == "high" else 10
        assert arm["usage"]["output_tokens"]["max"] == expected
    assert all(item["joint_positions_successes"] == 4 for item in report["interventions"])
    assert report["physics_usage_by_role"]["hidden_judge"]["evaluations"] == 4
    repeated = run_native_probe(panel, source, output, config, **kwargs)
    assert repeated["new_model_requests"] == 0 and len(calls) == 48
    changed = config.model_copy(update={"reasoning_efforts": ("minimal",)})
    with pytest.raises(ValueError, match="new output directory"):
        run_native_probe(panel, source, output, changed, **kwargs)


def test_forward_intervention_adds_replayed_oracle_kinematics(tmp_path, monkeypatch, study_fixture):
    panel = probe_panel(tmp_path, monkeypatch, study_fixture)
    source = azure_inventory(tmp_path)
    output = tmp_path / "data/forward"
    calls = []
    config = NativeProbeConfig(
        design="forward_intervention",
        layouts=4,
        reasoning_efforts=("medium",),
        max_model_requests=48,
    )
    kwargs = execute_kwargs(panel, calls)
    report = run_native_probe(panel, source, output, config, **kwargs)
    assert report["status"] == "completed" and len(calls) == 48
    tasks = [prompt_task(request) for request in calls]
    oracle = [task for task in tasks if "cue_ball_post_impact_state" in task]
    assert len(oracle) == 16 and all("goal" not in task for task in tasks)
    for task in oracle:
        labels = json.loads((panel / "private" / f"{task['case_id']}.json").read_text())
        if task["action"] == labels["forward_labels"]["base_action"]["action"]:
            state = task["cue_ball_post_impact_state"]
            assert state["velocity_mps"] == labels["record"]["cue_initial_velocity_mps"]
    assert sum("image_url" in json.dumps(json.loads(r.content)) for r in calls) == 16
    arms = {(arm["information"], arm["view"], arm["model"]): arm for arm in report["by_arm"]}
    assert len(arms) == 6
    assert all(arm["frames_success_by_tolerance_radii"]["1.0"] == 8 for arm in arms.values())
    assert report["physics_usage_by_role"]["offline_reference"]["evaluations"] == 8
    assert {item["information"] for item in report["interventions"]} == {
        "native",
        "oracle_post_impact_kinematics",
    }
    monkeypatch.setattr(
        native_probe,
        "SnookerSimulator",
        lambda *args: pytest.fail("Completed probe jobs must not replay oracle kinematics"),
    )
    repeated = run_native_probe(panel, source, output, config, **kwargs)
    assert repeated["new_model_requests"] == 0 and len(calls) == 48


def test_probe_config_keeps_the_declared_design_bounds():
    assert NativeProbeConfig().planned_requests() == 192
    assert (
        NativeProbeConfig(
            design="forward_intervention",
            layouts=64,
            reasoning_efforts=("minimal",),
            max_model_requests=768,
        ).planned_requests()
        == 768
    )
    with pytest.raises(ValueError, match="eight-layout"):
        NativeProbeConfig(layouts=12, max_model_requests=4000)
    with pytest.raises(ValueError, match="single effort"):
        NativeProbeConfig(design="forward_intervention", layouts=4, max_model_requests=4000)
    with pytest.raises(ValueError, match="cover every planned"):
        NativeProbeConfig(max_model_requests=191)
    with pytest.raises(ValueError, match="completion cap"):
        NativeProbeConfig(request_token_reservation=16384)
