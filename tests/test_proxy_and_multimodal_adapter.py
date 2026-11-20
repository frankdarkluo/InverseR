import json
import os
from contextlib import closing

import httpx
import pytest
from pydantic import ValidationError

from inverser.evaluation import BudgetLimits, TrialLedger, request_model
from inverser.model_adapters import (
    InvalidModelReply,
    ModelAdapter,
    ModelCall,
    ModelReply,
    rendered_asset,
)
from inverser.study import GoalSubmission
from inverser.study_calls import (
    StudyAdapter,
    StudyBackendConfig,
    local_vllm_backends,
    openai_proxy_backends,
)


@pytest.fixture(autouse=True)
def proxy_environment(monkeypatch):
    monkeypatch.setenv("INVERSE_PROXY_BASE_URL", "http://127.0.0.1:8536/v1")
    monkeypatch.delenv("INVERSE_PROXY_API_KEY", raising=False)


def backend(*, local=False, thinking=False, url=None):
    kwargs = dict(
        model_revision="fixture-revision",
        reasoning_effort="high" if thinking else "none",
        max_output_tokens=4096,
        request_token_reservation=16384,
        timeout_s=120,
    )
    if local:
        return local_vllm_backends(
            url or "http://127.0.0.1:8000/v1",
            ("Qwen3.5-9B",),
            enable_thinking=thinking,
            thinking_token_budget=1024 if thinking else None,
            **kwargs,
        )[0]
    os.environ["INVERSE_PROXY_BASE_URL"] = url or "http://127.0.0.1:8536/v1"
    return openai_proxy_backends(None, ("fixture-model",), **kwargs)[0]


@pytest.mark.parametrize("local,thinking", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("adapter_type", [ModelAdapter, StudyAdapter])
def test_multimodal_reply_keeps_reasoning_usage_across_cache(
    tmp_path, study_fixture, local, thinking, inline, adapter_type
):
    _, record, case = study_fixture
    config = backend(local=local, thinking=thinking)
    answer = GoalSubmission(action=record.witness_action.model_dump()).model_dump_json()
    seen = []

    def handler(request):
        seen.append(request)
        message = {"content": answer}
        if inline and thinking:
            message["content"] = f"<think>Contact before rest.</think>\n{answer}"
        else:
            message["reasoning_content"] = "Contact before rest."
        return httpx.Response(
            200,
            json={
                "model": config.model_id,
                "choices": [{"message": message, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 12,
                    "total_tokens": 20,
                    "completion_tokens_details": {"reasoning_tokens": 5},
                },
            },
        )

    call = ModelCall(
        view="bev",
        input=case.goal_input(),
        images=tuple(
            rendered_asset(case.goal_input(), role, case.public_physics.table)
            for role in ("initial", "goal")
        ),
    )
    with (
        closing(
            TrialLedger(
                tmp_path / "ledger.sqlite",
                BudgetLimits(
                    max_model_requests=1, max_physics_evaluations=0, max_total_tokens=16384
                ),
            )
        ) as ledger,
        adapter_type(
            config.model_copy(update={"execution_authorized": True}),
            transport=httpx.MockTransport(handler),
        ) as adapter,
    ):
        reply = request_model(adapter, call, ledger)
        cached = request_model(adapter, call, ledger)
        assert reply.content == cached.content == answer
        assert reply.reasoning_content == cached.reasoning_content == "Contact before rest."
        assert cached.cached and cached.usage.reasoning_tokens == 5
        assert ledger.usage_totals()["reasoning_tokens"] == 5
        assert ledger.usage_totals()["total_tokens"] == 20
    assert len(seen) == 1
    payload = json.loads(seen[0].content)
    assert len(payload["messages"][1]["content"]) == 3
    assert "initial_inventory" in payload["messages"][1]["content"][0]["text"]
    assert all(
        item["image_url"]["url"].startswith("data:image/png;base64,")
        for item in payload["messages"][1]["content"][1:]
    )
    if local:
        assert payload["chat_template_kwargs"]["enable_thinking"] is thinking
        assert payload["response_format"] == {"type": "json_object"}
        assert payload.get("thinking_token_budget") == (1024 if thinking else None)
    else:
        assert payload["reasoning_effort"] == "none"
    assert "authorization" not in seen[0].headers


@pytest.mark.parametrize(
    "url",
    ["http://10.0.0.1:8000/v1", "http://example.test/v1", "https://user:secret@example.test"],
)
def test_proxy_rejects_plaintext_remote_and_url_credentials(url):
    with pytest.raises(ValidationError):
        backend(url=url)


def test_proxy_execution_requires_authorization_revision_and_remote_credentials():
    with pytest.raises(RuntimeError, match="authorization"):
        StudyAdapter(backend())
    with pytest.raises(ValueError, match="credentials"):
        StudyAdapter(
            backend(url="https://fixture.test/v1").model_copy(update={"execution_authorized": True})
        )
    with pytest.raises(ValueError, match="revision"):
        StudyAdapter(
            backend().model_copy(
                update={"execution_authorized": True, "model_revision": "unverified"}
            )
        )


@pytest.mark.parametrize("content", ["<think>Unfinished", "<think>x<think>y</think>{}", "{}"])
def test_malformed_answers_remain_billed_invalid(tmp_path, study_fixture, content):
    _, _, case = study_fixture
    config = backend(local=True, thinking=True)

    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": config.model_id,
                "choices": [{"message": {"content": content}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
            },
        )

    with (
        closing(
            TrialLedger(
                tmp_path / "ledger.sqlite",
                BudgetLimits(
                    max_model_requests=1, max_physics_evaluations=0, max_total_tokens=16384
                ),
            )
        ) as ledger,
        StudyAdapter(
            config.model_copy(update={"execution_authorized": True}),
            transport=httpx.MockTransport(handler),
        ) as adapter,
    ):
        with pytest.raises(InvalidModelReply):
            request_model(adapter, ModelCall(view="structured", input=case.goal_input()), ledger)
        assert ledger.usage_totals()["total_tokens"] == 20
        assert "reasoning_tokens" not in ledger.usage_totals()


def test_old_cached_reply_needs_no_new_fields():
    old = {
        "backend_alias": "fixture",
        "model_id": "fixture",
        "content": "{}",
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        "cached": False,
        "finish_reason": "stop",
    }
    assert ModelReply.model_validate(old).model_dump(mode="json") == old


@pytest.mark.parametrize("budget", [0, -1, 4096])
def test_thinking_budget_must_be_positive_and_leave_answer_space(budget):
    config = backend(local=True, thinking=True)
    with pytest.raises(ValidationError):
        StudyBackendConfig.model_validate(
            {
                **config.model_dump(),
                "base_url": config.base_url,
                "thinking_token_budget": budget,
            }
        )
    with pytest.raises(ValidationError, match="thinking"):
        StudyBackendConfig.model_validate(
            {
                **config.model_dump(),
                "base_url": config.base_url,
                "enable_thinking": False,
            }
        )


@pytest.mark.parametrize("routed", [False, True])
def test_routed_model_identity_is_opt_in_and_never_relabelled(study_fixture, routed):
    _, record, case = study_fixture
    config = backend().model_copy(
        update={"execution_authorized": True, "allow_routed_model_id": routed}
    )
    answer = GoalSubmission(action=record.witness_action.model_dump()).model_dump_json()

    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "actual-model-2026-11-19",
                "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
            },
        )

    with StudyAdapter(config, transport=httpx.MockTransport(handler)) as adapter:
        call = ModelCall(view="structured", input=case.goal_input())
        if not routed:
            with pytest.raises(ValueError, match="identity"):
                adapter._complete(call, "fixture")
        else:
            reply = adapter._complete(call, "fixture")
            assert reply.model_id == "actual-model-2026-11-19"
            assert reply.requested_model_id == "fixture-model"
            assert reply.usage.reasoning_tokens is None


def test_proxy_aliases_and_credentials_are_only_loaded_at_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("INVERSE_PROXY_API_KEY", "fixture-secret")
    aliases = tmp_path / "aliases.json"
    aliases.write_text('{"GPT-5.6 Sol":"gpt-5.6-sol"}')
    config = openai_proxy_backends(
        aliases,
        ("GPT-5.6 Sol",),
        model_revision="catalog-sha256-fixture",
        reasoning_effort="low",
        max_output_tokens=1024,
        request_token_reservation=8192,
        timeout_s=120,
    )[0]
    assert config.alias == "GPT-5.6 Sol" and config.model_id == "gpt-5.6-sol"
    assert "fixture-secret" not in config.model_dump_json()
    seen_payloads = []

    def error_handler(request):
        seen_payloads.append(json.loads(request.content))
        return httpx.Response(400, text='{"error":"unsupported parameter"}')

    with StudyAdapter(
        config.model_copy(update={"execution_authorized": True}),
        transport=httpx.MockTransport(error_handler),
    ) as adapter:
        with pytest.raises(httpx.HTTPStatusError):
            adapter._exchange("fixture", (), "fixture", 0, answer_tag=True)
    sent = seen_payloads[0]
    assert sent["max_completion_tokens"] == 1024
    assert sent["reasoning_effort"] == "low"
    assert sent["stop"] == ["</answer>"]
    assert not {"temperature", "max_tokens", "seed", "response_format"} & set(sent)
