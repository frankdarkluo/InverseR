import json

import httpx
import pytest
from test_diagnostic_panel import source_inputs

from inverser import azure_study, diagnostic_panel
from inverser.azure_study import AzurePilotConfig, load_azure_backends, run_azure_panel
from inverser.diagnostic_panel import DiagnosticPanelConfig, prepare_diagnostic_panel
from inverser.evaluation import BudgetLimits, TrialLedger, UnresolvedRequest, request_model
from inverser.model_adapters import BackendConfig, InvalidModelReply, ModelAdapter, ModelCall
from inverser.study import GoalSubmission


def azure_inventory(tmp_path):
    path = tmp_path / "azure-inventory.json"
    path.write_text(
        json.dumps(
            {
                "azure_openai_endpoints": [
                    {
                        "endpoint": "https://fixture.openai.azure.com/",
                        "api_version": "2024-12-01-preview",
                        "deployments": [
                            {
                                "model": model,
                                "deployment_name": model,
                                "sku": "GlobalStandard",
                                "version": "2025-08-07",
                            }
                            for model in ("gpt-5-mini", "gpt-5")
                        ],
                    }
                ],
            }
        )
    )
    return path


def test_azure_request_uses_deployment_revision_and_supported_reasoning_parameters(
    tmp_path, study_fixture
):
    _, record, case = study_fixture
    backend = load_azure_backends(azure_inventory(tmp_path), 0, AzurePilotConfig())[0]
    calls = []

    def handler(request):
        calls.append(request)
        body = json.loads(request.content)
        assert body["max_completion_tokens"] == 4096
        assert body["reasoning_effort"] == "minimal"
        assert body["response_format"] == {"type": "json_object"}
        assert not {"temperature", "top_p", "seed", "max_tokens"} & set(body)
        assert request.headers["Authorization"] == "Bearer fixture-token"
        assert request.url.path == "/openai/deployments/gpt-5-mini/chat/completions"
        assert request.url.params["api-version"] == "2024-12-01-preview"
        return httpx.Response(
            200,
            json={
                "model": "gpt-5-mini-2025-08-07",
                "choices": [
                    {
                        "message": {
                            "content": GoalSubmission(
                                action=record.witness_action.model_dump()
                            ).model_dump_json()
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16},
            },
        )

    ledger = TrialLedger(
        tmp_path / "ledger.sqlite",
        BudgetLimits(
            max_model_requests=1,
            max_physics_evaluations=0,
            max_total_tokens=20000,
        ),
    )
    with ModelAdapter(
        backend.model_copy(update={"execution_authorized": True}),
        transport=httpx.MockTransport(handler),
        token_provider=lambda: "fixture-token",
    ) as adapter:
        call = ModelCall(view="structured", input=case.goal_input())
        reply = request_model(adapter, call, ledger)
        assert reply.finish_reason == "stop"
        assert request_model(adapter, call, ledger).cached
    assert len(calls) == 1 and ledger.usage_totals()["total_tokens"] == 16
    ledger.close()


def test_invalid_billed_reply_preserves_usage_and_is_never_retried(tmp_path, study_fixture):
    _, _, case = study_fixture
    ledger = TrialLedger(
        tmp_path / "ledger.sqlite",
        BudgetLimits(
            max_model_requests=2,
            max_physics_evaluations=0,
        ),
    )
    with ModelAdapter(
        BackendConfig(
            kind="mock", alias="test", model_id="fixture", base_url="https://fixture.invalid/v1"
        ),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "fixture",
                    "choices": [{"message": {"content": '{"wrong":1}'}}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14},
                },
            )
        ),
    ) as adapter:
        call = ModelCall(view="structured", input=case.goal_input())
        with pytest.raises(InvalidModelReply):
            request_model(adapter, call, ledger)
        with pytest.raises(UnresolvedRequest):
            request_model(adapter, call, ledger)
    assert ledger.counts()["model_requests"] == 1
    assert ledger.usage_totals()["total_tokens"] == 14
    assert ledger.connection.execute("SELECT status FROM requests").fetchone()[0] == "known_error"
    ledger.close()


def test_api_pilot_pairs_models_isolates_real_data_and_resumes(
    tmp_path, monkeypatch, study_fixture
):
    monkeypatch.setattr(diagnostic_panel, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(diagnostic_panel, "check_storage", lambda reserve: None)
    monkeypatch.setattr(azure_study, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(azure_study, "check_storage", lambda reserve: None)
    sources = source_inputs(tmp_path, study_fixture)
    panel = tmp_path / "data/panel"
    prepare_diagnostic_panel(
        *sources, panel, DiagnosticPanelConfig(synthetic_layouts=4, real_round_cases=4)
    )
    calls = []

    def handler(request):
        calls.append(request)
        task = json.loads(json.loads(request.content)["messages"][1]["content"][0]["text"])["task"]
        assert task["source_kind"] == "simulated"
        assert "private" not in json.dumps(task) and "witness_action" not in json.dumps(task)
        if "action" in task:
            labels = json.loads((panel / "private" / f"{task['case_id']}.json").read_text())
            truth = next(
                item["truth"]
                for item in labels["forward_labels"].values()
                if item["action"] == task["action"]
            )
            content = json.dumps(truth)
        else:
            labels = json.loads((panel / "private" / f"{task['case_id']}.json").read_text())
            content = GoalSubmission(action=labels["record"]["witness_action"]).model_dump_json()
        model = request.url.path.split("/")[-3]
        return httpx.Response(
            200,
            json={
                "model": model + "-2025-08-07",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            },
        )

    output = tmp_path / "data/api"
    source = azure_inventory(tmp_path)
    config = AzurePilotConfig()
    plan = run_azure_panel(panel, source, output, config, endpoint_index=0)
    assert plan["status"] == "planned_no_calls" and not calls
    kwargs = {
        "endpoint_index": 0,
        "execute": True,
        "token_provider": lambda: "fixture-token",
        "transport": httpx.MockTransport(handler),
    }
    partial = run_azure_panel(panel, source, output, config, limit_jobs=2, **kwargs)
    assert partial["status"] == "partial" and partial["new_model_requests"] == 2
    complete = run_azure_panel(panel, source, output, config, **kwargs)
    assert complete["status"] == "completed"
    assert complete["resource_counts"]["model_requests"] == 24
    assert len(calls) == 24 and not complete["real_data_sent"]
    assert len(complete["interventions"]) == 8
    assert all(item["goal_successes"] == 2 for item in complete["by_model_view"])
    repeated = run_azure_panel(panel, source, output, config, **kwargs)
    assert repeated["new_model_requests"] == 0 and len(calls) == 24
    with pytest.raises(ValueError, match="new output"):
        run_azure_panel(
            panel, source, output, config.model_copy(update={"reasoning_effort": "low"}), **kwargs
        )
