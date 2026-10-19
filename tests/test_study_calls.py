import json

import httpx
import pytest
from pydantic import ValidationError
from test_azure_study import azure_inventory

from inverser.baselines import make_action_bank
from inverser.evaluation import BudgetLimits, TrialLedger, UnresolvedRequest, request_model
from inverser.model_adapters import BackendConfig, ModelCall, rendered_asset
from inverser.study import GoalSubmission
from inverser.study_calls import (
    FEEDBACK_NOTE,
    FIRST_ATTEMPT_NOTE,
    REFLECTION_NOTE,
    AttemptFeedback,
    BankSelectionInput,
    FeedbackGoalInput,
    OracleForwardInput,
    PostImpactKinematics,
    SimulatedOutcome,
    StudyAdapter,
    StudyCall,
    guarded_request,
    request_study_model,
    study_azure_backends,
)

TIMES = (0.25, 0.5, 1.0)


def backends(tmp_path, effort="high"):
    return study_azure_backends(
        azure_inventory(tmp_path),
        0,
        reasoning_effort=effort,
        max_output_tokens=32768,
        request_token_reservation=40960,
        timeout_s=300,
    )


def task(call):
    return json.loads(call.prompt())["task"]


def test_study_backends_admit_long_reasoning_without_changing_inventory_identity(tmp_path):
    mini, full = backends(tmp_path)
    assert (mini.model_id, full.model_id) == ("gpt-5-mini-2025-08-07", "gpt-5-2025-08-07")
    assert mini.max_output_tokens == 32768 and mini.reasoning_effort == "high"
    assert mini.base_url == "https://fixture.openai.azure.com/"
    with pytest.raises(ValidationError):
        BackendConfig.model_validate({**mini.model_dump(), "base_url": mini.base_url})


def test_study_prompts_extend_frozen_tasks_with_only_declared_fields(study_fixture):
    _, record, case = study_fixture
    forward = case.forward_input(record.witness_action, TIMES)
    oracle = StudyCall(
        view="structured",
        input=OracleForwardInput(
            forward=forward,
            cue_ball_post_impact=PostImpactKinematics(
                velocity_mps=record.cue_initial_velocity_mps,
                angular_velocity_radps=record.cue_initial_angular_velocity_radps,
            ),
        ),
    )
    base = task(ModelCall(view="structured", input=forward))
    extended = task(oracle)
    assert set(extended) - set(base) == {"cue_ball_post_impact_state"}
    assert {k: v for k, v in extended.items() if k in base and k != "schema_version"} == {
        k: v for k, v in base.items() if k != "schema_version"
    }
    oracle.validate_reply(case.forward_truth(TIMES).model_dump_json())
    with pytest.raises(ValueError):
        oracle.validate_reply('{"bad":1}')

    goal = case.goal_input()
    bank = make_action_bank(case.case_id, count=3, seed=1)
    selection = StudyCall(
        view="structured", input=BankSelectionInput(goal=goal, candidates=bank.candidates)
    )
    assert len(task(selection)["candidate_actions"]) == 3 and "goal" in task(selection)
    selection.validate_reply(json.dumps({"candidate_id": bank.candidates[1].candidate_id}))
    with pytest.raises(ValueError, match="not in the declared bank"):
        selection.validate_reply('{"candidate_id":"unknown"}')
    image = StudyCall(
        view="bev",
        input=BankSelectionInput(goal=goal, candidates=bank.candidates),
        images=tuple(
            rendered_asset(goal, role, goal.physics.table) for role in ("initial", "goal")
        ),
    )
    assert "goal" not in task(image) and "initial" not in task(image)
    assert "goal_inventory" in task(image) and len(json.loads(image.prompt())) == 2
    with pytest.raises(ValueError, match="image variant"):
        StudyCall(view="bev", input=oracle.input)


def test_feedback_history_is_consecutive_and_matches_the_feedback_condition(study_fixture):
    _, record, case = study_fixture
    goal = case.goal_input()
    outcome = SimulatedOutcome(
        resting_state=goal.initial,
        goal_reached=False,
        max_position_error_radii=3,
        failure="position",
    )
    first = AttemptFeedback(attempt=1, action=record.witness_action.model_dump(), outcome=outcome)
    call = StudyCall(
        view="structured",
        input=FeedbackGoalInput(
            goal=goal, attempt=2, max_attempts=3, simulator_feedback=True, history=(first,)
        ),
    )
    assert task(call)["feedback_protocol"] == FEEDBACK_NOTE
    assert task(call)["history"][0]["outcome"]["failure"] == "position"
    reflection = FeedbackGoalInput(
        goal=goal,
        attempt=2,
        max_attempts=3,
        simulator_feedback=False,
        history=(first.model_copy(update={"outcome": None}),),
    )
    assert task(StudyCall(view="structured", input=reflection))["feedback_protocol"] == (
        REFLECTION_NOTE
    )
    with pytest.raises(ValidationError, match="exactly when feedback"):
        FeedbackGoalInput(
            goal=goal, attempt=2, max_attempts=3, simulator_feedback=False, history=(first,)
        )
    with pytest.raises(ValidationError, match="follow the recorded history"):
        FeedbackGoalInput(goal=goal, attempt=3, max_attempts=3, simulator_feedback=True)
    opening = FeedbackGoalInput(goal=goal, attempt=1, max_attempts=3, simulator_feedback=None)
    assert task(StudyCall(view="structured", input=opening))["feedback_protocol"] == (
        FIRST_ATTEMPT_NOTE
    )
    with pytest.raises(ValidationError, match="shared first attempt"):
        FeedbackGoalInput(goal=goal, attempt=1, max_attempts=3, simulator_feedback=False)
    reached = outcome.model_copy(
        update={"goal_reached": True, "failure": None, "max_position_error_radii": 0}
    )
    with pytest.raises(ValidationError, match="ends the episode"):
        FeedbackGoalInput(
            goal=goal,
            attempt=2,
            max_attempts=3,
            simulator_feedback=True,
            history=(first.model_copy(update={"outcome": reached}),),
        )
    with pytest.raises(ValidationError, match="disagree"):
        SimulatedOutcome(resting_state=goal.initial, goal_reached=True, failure="position")


def test_study_requests_share_the_frozen_adapter_shape_and_bill_once(tmp_path, study_fixture):
    _, record, case = study_fixture
    goal = case.goal_input()
    bank = make_action_bank(case.case_id, count=3, seed=1)
    seen = []

    def handler(request):
        seen.append(request)
        body = json.loads(request.content)
        content = json.loads(body["messages"][1]["content"][0]["text"])["task"]
        reply = (
            {"candidate_id": "unknown" if len(seen) > 2 else bank.candidates[0].candidate_id}
            if "candidate_actions" in content
            else GoalSubmission(action=record.witness_action.model_dump()).model_dump(mode="json")
        )
        return httpx.Response(
            200,
            json={
                "model": "gpt-5-mini-2025-08-07",
                "choices": [{"message": {"content": json.dumps(reply)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16},
            },
        )

    ledger = TrialLedger(
        tmp_path / "ledger.sqlite",
        BudgetLimits(max_model_requests=3, max_physics_evaluations=0, max_total_tokens=200000),
    )
    selection = StudyCall(
        view="structured", input=BankSelectionInput(goal=goal, candidates=bank.candidates)
    )
    with StudyAdapter(
        backends(tmp_path)[0].model_copy(update={"execution_authorized": True}),
        transport=httpx.MockTransport(handler),
        token_provider=lambda: "fixture-token",
    ) as adapter:
        request_model(adapter, ModelCall(view="structured", input=goal), ledger)
        request_study_model(adapter, selection, ledger)
        assert request_study_model(adapter, selection, ledger).cached
        frozen, study = (json.loads(request.content) for request in seen)
        assert frozen["max_completion_tokens"] == study["max_completion_tokens"] == 32768
        for body in (frozen, study):
            body["messages"][1]["content"][0]["text"] = "<prompt>"
        assert frozen == study
        assert seen[0].url == seen[1].url
        assert seen[0].headers["Authorization"] == seen[1].headers["Authorization"]
        changed = StudyCall(
            view="structured",
            input=BankSelectionInput(goal=goal, candidates=tuple(reversed(bank.candidates))),
        )
        outcome = guarded_request(lambda: request_study_model(adapter, changed, ledger))
        assert (outcome.error_code, outcome.stop_error) == ("invalid_submission", None)
        assert outcome.reply is not None and outcome.reply.usage.total_tokens == 16
        with pytest.raises(UnresolvedRequest):
            request_study_model(adapter, changed, ledger)
    assert len(seen) == 3 and ledger.usage_totals()["total_tokens"] == 48
    ledger.close()


def test_service_errors_stop_without_retry(tmp_path, study_fixture):
    _, _, case = study_fixture
    ledger = TrialLedger(
        tmp_path / "ledger.sqlite",
        BudgetLimits(max_model_requests=1, max_physics_evaluations=0, max_total_tokens=200000),
    )
    call = StudyCall(
        view="structured",
        input=BankSelectionInput(
            goal=case.goal_input(),
            candidates=make_action_bank(case.case_id, count=2, seed=3).candidates,
        ),
    )
    with StudyAdapter(
        backends(tmp_path)[0].model_copy(update={"execution_authorized": True}),
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        token_provider=lambda: "fixture-token",
    ) as adapter:
        outcome = guarded_request(lambda: request_study_model(adapter, call, ledger))
        assert outcome.stop_error == outcome.error_code == "http_500" and outcome.reply is None
        with pytest.raises(UnresolvedRequest):
            request_study_model(adapter, call, ledger)
    ledger.close()
