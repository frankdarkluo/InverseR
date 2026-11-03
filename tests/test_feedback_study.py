import json

import httpx
import pytest
from test_azure_study import azure_inventory
from test_native_probe import prompt_task
from test_planning_pilot import planning_inputs

from inverser import study_calls
from inverser.feedback_study import (
    FeedbackStudyConfig,
    development_feedback_cases,
    run_feedback_study,
)
from inverser.study_calls import REFLECTION_NOTE


def run_dev(panel, native, source, output, config, **kwargs):
    cases, case_source = development_feedback_cases(panel, native, config)
    return run_feedback_study(cases, case_source, source, output, config, **kwargs)


def feedback_inputs(tmp_path, monkeypatch, study_fixture):
    panel, native = planning_inputs(tmp_path, monkeypatch, study_fixture)
    monkeypatch.setattr(study_calls, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(study_calls, "check_storage", lambda reserve: None)
    return panel, native


def corrective_handler(panel, calls, invalid_model=None):
    # Wrong first attempt, then the hidden witness: only genuine feedback can stop early.
    def handler(request):
        calls.append(request)
        task = prompt_task(request)
        model = request.url.path.split("/")[-3]
        labels = json.loads((panel / "private" / f"{task['case_id']}.json").read_text())
        witness = labels["record"]["witness_action"]
        if model == invalid_model:
            content = '{"bad":1}'
        elif task["attempt"] == 1:
            turned = {**witness, "azimuth_deg": (witness["azimuth_deg"] + 90) % 360}
            content = json.dumps({"action": turned})
        else:
            content = json.dumps({"action": witness})
        return httpx.Response(
            200,
            json={
                "model": model + "-2025-08-07",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            },
        )

    return handler


def kwargs_for(handler):
    return {
        "endpoint_index": 0,
        "execute": True,
        "token_provider": lambda: "fixture-token",
        "transport": httpx.MockTransport(handler),
    }


def test_feedback_conditions_attribute_what_the_agent_was_shown(
    tmp_path, monkeypatch, study_fixture
):
    panel, native = feedback_inputs(tmp_path, monkeypatch, study_fixture)
    source = azure_inventory(tmp_path)
    output = tmp_path / "data/feedback"
    calls = []
    config = FeedbackStudyConfig(layouts=4, max_attempts=3, max_model_requests=72)
    plan = run_dev(panel, native, source, output, config, endpoint_index=0)
    assert plan["planned_episodes"] == 24 and plan["maximum_requests"] == 56 and not calls
    kwargs = kwargs_for(corrective_handler(panel, calls))
    partial = run_dev(panel, native, source, output, config, limit_episodes=2, **kwargs)
    assert partial["status"] == "partial" and partial["completed_episodes"] == 2
    report = run_dev(panel, native, source, output, config, **kwargs)
    # One shared opening request, then 1 genuine and 2+2 shuffled/reflection revisions.
    assert report["status"] == "completed" and len(calls) == 4 * (1 + 1 + 2 + 2) * 2
    arms = {(arm["model"], arm["condition"]): arm for arm in report["by_model_condition"]}
    for model in ("gpt-5-mini", "gpt-5"):
        assert arms[model, "genuine"]["stop_reasons"] == {"feedback_goal_reached": 4}
        assert arms[model, "genuine"]["mean_attempts_per_episode"] == 2
        for condition in ("shuffled", "reflection"):
            assert arms[model, condition]["stop_reasons"] == {"max_attempts": 4}
        for condition in ("genuine", "shuffled", "reflection"):
            arm = arms[model, condition]
            assert arm["final_successes"] == 4
            assert arm["success_if_capped_at_attempt"]["1"] == 0
            assert arm["any_attempt_success_by_attempt"]["2"] == 4
    second = [task for task in map(prompt_task, calls) if task["attempt"] == 2]
    reflected = [t["history"][0] for t in second if t["feedback_protocol"] == REFLECTION_NOTE]
    assert len(reflected) == 8 and all(item["outcome"] is None for item in reflected)
    shown = [
        t["history"][0]["outcome"] for t in second if t["feedback_protocol"] != REFLECTION_NOTE
    ]
    assert len(shown) == 16 and not any(outcome["goal_reached"] for outcome in shown)
    for path in (output / "episodes").rglob("*.json"):
        first = json.loads(path.read_text())["payload"]["attempts"][0]
        if path.parent.name == "shuffled":
            assert first["feedback_action"] != first["action"]
        elif path.parent.name == "genuine":
            assert first["feedback_action"] == first["action"]
    roles = report["physics_usage_by_role"]
    assert roles["hidden_judge"]["evaluations"] == 8
    assert roles["agent_search"]["evaluations"] == 8 + 12
    assert {item["condition_b"] for item in report["paired_condition_comparisons"]} == {
        "shuffled",
        "reflection",
    }
    repeated = run_dev(panel, native, source, output, config, **kwargs)
    assert repeated["new_model_requests"] == 0 and len(calls) == 48
    openings = [task for task in map(prompt_task, calls) if task["attempt"] == 1]
    # One opening request per layout and model, identical across the three conditions.
    assert len(openings) == 8 and len({json.dumps(t, sort_keys=True) for t in openings}) == 4


def test_invalid_feedback_reply_ends_the_episode_without_repair(
    tmp_path, monkeypatch, study_fixture
):
    panel, native = feedback_inputs(tmp_path, monkeypatch, study_fixture)
    calls = []
    config = FeedbackStudyConfig(layouts=4, max_attempts=2, max_model_requests=32)
    report = run_dev(
        panel,
        native,
        azure_inventory(tmp_path),
        tmp_path / "data/invalid",
        config,
        **kwargs_for(corrective_handler(panel, calls, invalid_model="gpt-5")),
    )
    arms = {(arm["model"], arm["condition"]): arm for arm in report["by_model_condition"]}
    # The rejected opening is shared, so every condition stops on it without a new request.
    for condition in ("genuine", "shuffled", "reflection"):
        assert arms["gpt-5", condition]["stop_reasons"] == {"invalid_submission": 4}
        assert arms["gpt-5", condition]["final_successes"] == 0
        assert arms["gpt-5", condition]["mean_attempts_per_episode"] == 1
        assert arms["gpt-5-mini", condition]["final_successes"] == 4
    assert len(calls) == 4 + 4 * (1 + 3)


def test_feedback_config_requires_room_for_every_attempt():
    assert FeedbackStudyConfig().max_requests() == 32 * (1 + 7 * 3) * 2 == 1408
    with pytest.raises(ValueError, match="cover every attempt"):
        FeedbackStudyConfig(max_model_requests=1407)
    with pytest.raises(ValueError, match="distinct"):
        FeedbackStudyConfig(conditions=("genuine", "genuine"))


def test_credential_failure_mid_episode_is_replayed_from_the_ledger(
    tmp_path, monkeypatch, study_fixture
):
    panel, native = feedback_inputs(tmp_path, monkeypatch, study_fixture)
    calls = []
    handler = corrective_handler(panel, calls)
    broken = {"value": False}

    def counting(request):
        response = handler(request)
        broken["value"] = len(calls) == 2
        return response

    def token():
        if broken["value"]:
            raise RuntimeError("expired login")
        return "fixture-token"

    config = FeedbackStudyConfig(
        layouts=4, max_attempts=3, conditions=("reflection",), max_model_requests=24
    )
    output = tmp_path / "data/credential"
    arguments = (panel, native, azure_inventory(tmp_path), output, config)
    stopped = run_dev(
        *arguments,
        endpoint_index=0,
        execute=True,
        token_provider=token,
        transport=httpx.MockTransport(counting),
    )
    assert stopped["stop_error"] == "credential_unavailable"
    assert stopped["completed_episodes"] == 0 and len(calls) == 2
    broken["value"] = False
    resumed = run_dev(*arguments, **kwargs_for(handler))
    assert resumed["status"] == "completed"
    # The interrupted episode replays its two billed attempts from the ledger.
    assert resumed["new_model_requests"] == 4 * 3 * 2 - 2 and len(calls) == 24
