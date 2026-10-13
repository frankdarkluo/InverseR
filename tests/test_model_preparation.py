import json

import httpx
import pytest
from pydantic import ValidationError

from inverser.evaluation import (
    BudgetExhausted,
    BudgetLimits,
    TrialLedger,
    UnresolvedRequest,
    evaluate_submission,
    request_model,
    study_dry_run,
)
from inverser.model_adapters import (
    BackendConfig,
    ModelAdapter,
    ModelCall,
    rendered_asset,
)
from inverser.study import GoalSubmission


def test_bev_removes_ball_coordinates_and_disallows_forward_goal_image(study_fixture):
    _, record, case = study_fixture
    goal = case.goal_input()
    images = (
        rendered_asset(goal, "initial", goal.physics.table),
        rendered_asset(goal, "goal", goal.physics.table),
    )
    call = ModelCall(view="bev", input=goal, images=images)
    task = json.loads(call.prompt())["task"]
    assert "initial" not in task and "goal" not in task
    assert "xy_m" not in json.dumps(task["initial_inventory"])
    forward = case.forward_input(record.witness_action, (0.25,))
    with pytest.raises(ValueError, match="goal image"):
        rendered_asset(forward, "goal", forward.physics.table)
    with pytest.raises(ValidationError, match="permitted"):
        ModelCall(view="bev", input=forward, images=images)
    with pytest.raises(ValidationError, match="source_kind"):
        ModelCall.model_validate(
            {
                "view": "structured",
                "input": {**goal.model_dump(), "source_kind": "real_observation"},
            }
        )


def test_actual_backends_require_authorization_and_local_loopback():
    with pytest.raises(RuntimeError, match="authorization"):
        ModelAdapter(
            BackendConfig(
                kind="api",
                alias="unallocated",
                model_id="not-probed",
                base_url="https://example.invalid/v1",
            ),
            transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        )
    with pytest.raises(ValidationError, match="loopback"):
        BackendConfig(
            kind="local_vllm",
            alias="unsafe",
            model_id="local",
            base_url="http://10.0.0.1:8000/v1",
        )
    with pytest.raises(ValidationError, match="secrets"):
        BackendConfig(
            kind="api",
            alias="unsafe",
            model_id="remote",
            base_url="https://example.invalid/v1?api_key=not-a-real-key",
        )


def test_request_cache_and_quota_persist_across_restart(tmp_path, study_fixture):
    _, record, case = study_fixture
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(
            200,
            json={
                "model": "fixture",
                "choices": [
                    {
                        "message": {
                            "content": GoalSubmission(
                                action=record.witness_action.model_dump()
                            ).model_dump_json()
                        }
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
            },
        )

    path = tmp_path / "ledger.sqlite"
    limits = BudgetLimits(max_model_requests=1, max_physics_evaluations=1)
    ledger = TrialLedger(path, limits)
    call = ModelCall(view="structured", input=case.goal_input())
    with ModelAdapter(
        BackendConfig(
            kind="mock", alias="test", model_id="fixture", base_url="https://fixture.invalid/v1"
        ),
        transport=httpx.MockTransport(handler),
    ) as adapter:
        first = request_model(adapter, call, ledger)
        assert first.usage.total_tokens == 10 and not first.cached
        ledger.close()
        ledger = TrialLedger(path, limits)
        assert request_model(adapter, call, ledger).cached
        with pytest.raises(BudgetExhausted):
            request_model(adapter, call.model_copy(update={"seed": 1}), ledger)
    assert len(sent) == 1
    assert ledger.counts()["model_requests"] == 1
    ledger.close()
    with pytest.raises(ValueError, match="new ledger"):
        TrialLedger(path, BudgetLimits(max_model_requests=2, max_physics_evaluations=1))


def test_unknown_response_is_not_rebilled_automatically(tmp_path, study_fixture):
    _, _, case = study_fixture

    def handler(request):
        raise httpx.ReadTimeout("fixture timeout", request=request)

    ledger = TrialLedger(
        tmp_path / "ledger.sqlite", BudgetLimits(max_model_requests=2, max_physics_evaluations=0)
    )
    with ModelAdapter(
        BackendConfig(
            kind="mock", alias="test", model_id="fixture", base_url="https://fixture.invalid/v1"
        ),
        transport=httpx.MockTransport(handler),
    ) as adapter:
        call = ModelCall(view="structured", input=case.goal_input())
        with pytest.raises(httpx.ReadTimeout):
            request_model(adapter, call, ledger)
        with pytest.raises(UnresolvedRequest, match="resubmitted"):
            request_model(adapter, call, ledger)
    assert ledger.counts()["model_requests"] == 1
    assert (
        ledger.connection.execute("SELECT status FROM requests").fetchone()[0] == "unknown_response"
    )
    ledger.close()


def test_wrong_model_and_invalid_format_remain_recorded_failures(tmp_path, study_fixture):
    _, _, case = study_fixture
    ledger = TrialLedger(
        tmp_path / "ledger.sqlite", BudgetLimits(max_model_requests=1, max_physics_evaluations=0)
    )
    with ModelAdapter(
        BackendConfig(
            kind="mock", alias="test", model_id="fixture", base_url="https://fixture.invalid/v1"
        ),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"model": "wrong"})),
    ) as adapter:
        with pytest.raises(ValueError, match="identity"):
            request_model(adapter, ModelCall(view="structured", input=case.goal_input()), ledger)
    assert ledger.counts()["model_requests"] == 1
    assert ledger.connection.execute("SELECT status FROM requests").fetchone()[0] == "known_error"
    ledger.close()


def test_hidden_judge_and_physics_quota(tmp_path, study_fixture):
    simulator, record, _ = study_fixture
    ledger = TrialLedger(
        tmp_path / "ledger.sqlite", BudgetLimits(max_model_requests=0, max_physics_evaluations=1)
    )
    assert evaluate_submission(record, record.witness_action, simulator, ledger).success
    assert ledger.counts()["physics_evaluations"] == 1
    with pytest.raises(BudgetExhausted, match="Physics"):
        evaluate_submission(record, record.witness_action, simulator, ledger)
    ledger.close()


def test_dry_run_has_no_model_performance_claim(tmp_path, study_fixture):
    _, record, _ = study_fixture
    source = tmp_path / "privileged.jsonl"
    source.write_text(record.model_dump_json() + "\n")
    report = study_dry_run(source, tmp_path / "dry")
    assert report["status"] == "engineering_mock_dry_run"
    assert report["actual_model_inference"] is False
    assert report["paid_api_requests"] == 0
    assert report["model_performance_claim"] is False
    assert report["counts"] == {
        "model_requests": 2,
        "physics_attempts": 2,
        "physics_evaluations": 2,
    }
    assert all(row["success"] and row["second_request_cached"] for row in report["results"])
    bev_input = json.loads((tmp_path / "dry/bev_input.json").read_text())
    assert "initial" not in bev_input["task"] and "goal" not in bev_input["task"]


def test_reserved_physics_cannot_overspend_across_connections(tmp_path):
    path = tmp_path / "ledger.sqlite"
    limits = BudgetLimits(max_model_requests=0, max_physics_evaluations=1)
    first, second = TrialLedger(path, limits), TrialLedger(path, limits)
    reservation = first.reserve_physics()
    with pytest.raises(BudgetExhausted, match="Physics"):
        second.reserve_physics()
    first.record_physics(reservation, 1)
    assert second.counts()["physics_evaluations"] == 1
    first.close()
    second.close()


def test_endpoint_and_revision_are_part_of_request_identity(study_fixture):
    _, _, case = study_fixture
    config = BackendConfig(
        kind="mock", alias="same-alias", model_id="same-model", base_url="https://one.invalid/v1"
    )
    call = ModelCall(view="structured", input=case.goal_input())
    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    with (
        ModelAdapter(config, transport=transport) as first,
        ModelAdapter(
            config.model_copy(update={"base_url": "https://two.invalid/v1"}), transport=transport
        ) as second,
    ):
        assert first.request_key(call) != second.request_key(call)
