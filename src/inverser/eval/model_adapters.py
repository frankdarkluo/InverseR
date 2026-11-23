import base64
import hashlib
import io
import json
import logging
from collections.abc import Callable
from ipaddress import ip_address
from types import TracebackType
from typing import Literal, Self
from urllib.parse import urlsplit

import httpx
from PIL import Image
from pydantic import Field, model_validator

from inverser.preview import render_layout
from inverser.runtime import fingerprint
from inverser.schemas import Record
from inverser.study import (
    ForwardInput,
    ForwardSubmission,
    GoalInput,
    GoalSubmission,
    TableGeometry,
)

LOGGER = logging.getLogger(__name__)
VLM_SYSTEM_PROMPT = (
    "Solve the task using the provided images and physical parameters. "
    "Return one JSON answer that conforms to output_schema. "
    "Do not return the schema, the task, or a copy of the input."
)
# The one system prompt for every <answer>-tag request: SFT, RL rollouts and evaluation.
CAUSAL_PROMPT = (
    "In the reasoning segment, first identify ball positions, aiming geometry, and the likely "
    "ordered contacts; then consider sliding, rolling, and any 3D tip-offset spin deflection; "
    "finally use those physical laws to calibrate the numeric prediction or shot. Keep reasoning "
    "within 1024 tokens inside <think>...</think>, then output only the requested "
    "<answer>...</answer> block."
)


class RenderedAsset(Record):
    case_id: str
    source_kind: Literal["simulated"] = "simulated"
    role: Literal["initial", "goal"]
    png_base64: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_image(self) -> Self:
        data = base64.b64decode(self.png_base64, validate=True)
        if hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError("Rendered asset checksum mismatch")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.size != (self.width_px, self.height_px):
                raise ValueError("Rendered asset format or size mismatch")
            image.verify()
        return self


class ModelCall(Record):
    view: Literal["structured", "bev"]
    input: ForwardInput | GoalInput = Field(discriminator="schema_version")
    images: tuple[RenderedAsset, ...] = ()
    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_images(self) -> Self:
        expected = ("initial",) if isinstance(self.input, ForwardInput) else ("initial", "goal")
        if self.view == "structured":
            if self.images:
                raise ValueError("Structured calls must not include images")
        elif tuple(image.role for image in self.images) != expected:
            raise ValueError("BEV call requires exactly the permitted initial/goal images")
        if any(image.case_id != self.input.case_id for image in self.images):
            raise ValueError("Image and input case IDs disagree")
        return self

    def prompt(self) -> str:
        data = self.input.model_dump(mode="json")
        if self.view == "bev":
            initial = self.input.initial
            data.pop("initial")
            data["initial_inventory"] = [
                {"ball_id": ball.ball_id, "kind": ball.kind, "status": ball.status}
                for ball in initial.balls
            ]
            data["initial_red_slot_order"] = "increasing x, then increasing y"
            if isinstance(self.input, GoalInput):
                data.pop("goal")
                data["goal_inventory"] = [
                    {"kind": ball.kind, "status": ball.status} for ball in self.input.goal.balls
                ]
        schema = (
            ForwardSubmission.model_json_schema()
            if isinstance(self.input, ForwardInput)
            else GoalSubmission.model_json_schema()
        )
        return json.dumps(
            {"task": data, "output_schema": schema},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class BackendConfig(Record):
    kind: Literal["mock", "api", "local_vllm", "openai_proxy"]
    alias: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(default="unverified", min_length=1)
    base_url: str = Field(exclude=True, repr=False)
    execution_authorized: bool = False
    max_output_tokens: int = Field(default=1024, gt=0, le=8192)
    request_token_reservation: int = Field(default=16384, gt=0)
    provider: Literal["openai", "azure_openai"] = "openai"
    deployment_name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]+$")
    api_version: str | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    timeout_s: float = Field(default=30, gt=0, le=300)
    allow_routed_model_id: bool = Field(default=False, exclude_if=lambda value: not value)
    temperature: float = Field(default=0, ge=0, le=2, exclude_if=lambda value: value == 0)
    enable_thinking: bool | None = Field(default=None, exclude_if=lambda value: value is None)
    thinking_token_budget: int | None = Field(
        default=None, ge=1, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
        if self.enable_thinking is not None and self.kind != "local_vllm":
            raise ValueError("enable_thinking is a local vLLM chat-template option")
        if self.enable_thinking is True and self.reasoning_effort == "none":
            raise ValueError("Thinking cannot be enabled with reasoning_effort none")
        if self.thinking_token_budget is not None and (
            self.kind != "local_vllm"
            or self.enable_thinking is not True
            or self.thinking_token_budget >= self.max_output_tokens
        ):
            raise ValueError(
                "A local thinking budget needs thinking on and room for the JSON answer"
            )
        endpoint = urlsplit(self.base_url)
        if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            raise ValueError("Authentication/query secrets must not be embedded in endpoint URLs")
        if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
            raise ValueError("Backend endpoint must be an HTTP(S) URL")
        if self.kind == "api" and endpoint.scheme != "https":
            raise ValueError("External API endpoints must use HTTPS")
        if self.provider == "azure_openai":
            if self.kind != "api" or not self.deployment_name or not self.api_version:
                raise ValueError("Azure requires an API backend, deployment name and API version")
            if not self.reasoning_effort:
                raise ValueError("Azure reasoning experiments require an explicit effort")
        if self.kind in ("local_vllm", "openai_proxy"):
            hostname = endpoint.hostname
            loopback = hostname == "localhost"
            if not loopback:
                try:
                    loopback = ip_address(hostname).is_loopback
                except ValueError:
                    loopback = False
            if self.kind == "local_vllm" and not loopback:
                raise ValueError("Local vLLM endpoint must use a loopback address")
            if self.kind == "openai_proxy":
                if self.provider != "openai":
                    raise ValueError("OpenAI-compatible proxies require provider openai")
                if not loopback and endpoint.scheme != "https":
                    raise ValueError("Remote proxy endpoints must use HTTPS")
        return self


class TokenUsage(Record):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0, exclude_if=lambda value: value is None)


class ModelReply(Record):
    backend_alias: str
    model_id: str
    content: str
    usage: TokenUsage
    cached: bool = False
    finish_reason: str | None = None
    reasoning_content: str | None = Field(default=None, exclude_if=lambda value: value is None)
    requested_model_id: str | None = Field(default=None, exclude_if=lambda value: value is None)


class InvalidModelReply(ValueError):
    def __init__(self, reply: ModelReply, message: str) -> None:
        super().__init__(message)
        self.reply = reply


def rendered_asset(
    data: ForwardInput | GoalInput, role: Literal["initial", "goal"], geometry: TableGeometry
) -> RenderedAsset:
    if role == "initial":
        layout = data.initial
    elif isinstance(data, GoalInput):
        layout = data.goal
    else:
        raise ValueError("Forward prediction may not receive a goal image")
    image, _ = render_layout(layout, geometry)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    payload = stream.getvalue()
    return RenderedAsset(
        case_id=data.case_id,
        role=role,
        png_base64=base64.b64encode(payload).decode("ascii"),
        sha256=hashlib.sha256(payload).hexdigest(),
        width_px=image.width,
        height_px=image.height,
    )


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Model response {name} must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"Model response {name} keys must be strings")
        result[key] = item
    return result


def _token(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Model usage fields must be nonnegative integers")
    return value


def split_reasoning(text: str) -> tuple[str | None, str]:
    """Only unwrap one complete leading think block; malformed output stays invalid."""
    stripped = text.lstrip()
    if stripped.startswith("<think>"):
        trace, separator, answer = stripped[len("<think>") :].partition("</think>")
        if separator and "<think>" not in trace and "</think>" not in answer:
            return trace, answer.strip()
    return None, text


class ModelAdapter:
    def __init__(
        self,
        config: BackendConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        api_key: str | None = None,
        token_provider: Callable[[], str] | None = None,
    ) -> None:
        if config.kind != "mock" and not config.execution_authorized:
            raise RuntimeError("Actual backend execution needs explicit resource authorization")
        needs_credentials = config.kind == "api" or (
            config.kind == "openai_proxy" and urlsplit(config.base_url).scheme == "https"
        )
        if needs_credentials and not api_key and token_provider is None:
            raise ValueError("API credentials must be supplied at runtime")
        if api_key and token_provider is not None:
            raise ValueError("Supply exactly one runtime credential source")
        if config.kind != "mock" and config.model_revision == "unverified":
            raise ValueError("Actual backend execution needs a declared model revision")
        if config.request_token_reservation < config.max_output_tokens:
            raise ValueError("Token reservation must cover the configured output cap")
        if config.kind == "mock" and not isinstance(transport, httpx.MockTransport):
            raise ValueError("Mock backend requires an explicit mock transport")
        self.config = config
        self._client = httpx.Client(transport=transport, timeout=config.timeout_s)
        self._api_key = api_key
        self._token_provider = token_provider

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._client.close()

    def request_key(self, call: ModelCall) -> str:
        return fingerprint(
            {
                "backend": self.config.model_dump(mode="json"),
                "endpoint_hash": fingerprint(self.config.base_url),
                "call": call.model_dump(mode="json"),
                "prompt": call.prompt(),
            }
        )

    def _complete(self, call: ModelCall, request_id: str) -> ModelReply:
        content: list[dict[str, object]] = [{"type": "text", "text": call.prompt()}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + image.png_base64},
            }
            for image in call.images
        )
        headers = {"X-Request-ID": request_id}
        token = self._token_provider() if self._token_provider is not None else self._api_key
        if token is not None:
            headers["Authorization"] = "Bearer " + token
        system = "Return only the requested numerical JSON."
        if self.config.kind in ("local_vllm", "openai_proxy"):
            system = VLM_SYSTEM_PROMPT
        if self.config.enable_thinking:
            system = "Reason about the physical events first. " + system
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        url = self.config.base_url.rstrip("/")
        params = None
        if self.config.provider == "azure_openai":
            url += f"/openai/deployments/{self.config.deployment_name}/chat/completions"
            params = {"api-version": self.config.api_version}
            request_payload = {
                "messages": messages,
                "max_completion_tokens": self.config.max_output_tokens,
                "reasoning_effort": self.config.reasoning_effort,
                "response_format": {"type": "json_object"},
            }
        else:
            url += "/chat/completions"
            request_payload = {
                "model": self.config.model_id,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_output_tokens,
                "seed": call.seed,
            }
            if self.config.kind == "local_vllm" and self.config.enable_thinking is not None:
                request_payload["chat_template_kwargs"] = {
                    "enable_thinking": self.config.enable_thinking
                }
                request_payload["response_format"] = {"type": "json_object"}
            if self.config.kind == "openai_proxy":
                request_payload["reasoning_effort"] = self.config.reasoning_effort
            if self.config.thinking_token_budget is not None:
                request_payload["thinking_token_budget"] = self.config.thinking_token_budget
        response = self._client.post(url, headers=headers, params=params, json=request_payload)
        response.raise_for_status()
        payload: object = response.json()
        body = _object(payload, "body")
        returned_model = body.get("model")
        if not isinstance(returned_model, str) or not returned_model.strip():
            raise ValueError("Returned model identity must be nonempty text")
        if returned_model != self.config.model_id and not self.config.allow_routed_model_id:
            raise ValueError("Returned model identity does not match the configured deployment")
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Exactly one model response choice is required")
        message = _object(_object(choices[0], "choice").get("message"), "message")
        text = message.get("content")
        if not isinstance(text, str) or not text:
            raise ValueError("A nonempty textual model response is required")
        inline_reasoning, text = split_reasoning(text)
        reasoning = message.get("reasoning_content", message.get("reasoning"))
        if reasoning is None:
            reasoning = inline_reasoning
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("Model reasoning content must be text")
        usage_data = body.get("usage")
        if usage_data is None:
            LOGGER.warning(
                "Usage unavailable for backend %s request %s", self.config.alias, request_id
            )
            usage = TokenUsage()
        else:
            fields = _object(usage_data, "usage")
            details = fields.get("completion_tokens_details")
            usage = TokenUsage(
                input_tokens=_token(fields.get("prompt_tokens")),
                output_tokens=_token(fields.get("completion_tokens")),
                total_tokens=_token(fields.get("total_tokens")),
                reasoning_tokens=(
                    _token(_object(details, "completion_tokens_details").get("reasoning_tokens"))
                    if details is not None
                    else None
                ),
            )
        finish_reason = _object(choices[0], "choice").get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ValueError("Provider finish_reason must be a string")
        reply = ModelReply(
            backend_alias=self.config.alias,
            model_id=returned_model,
            content=text,
            usage=usage,
            finish_reason=finish_reason,
            reasoning_content=reasoning,
            requested_model_id=self.config.model_id if self.config.allow_routed_model_id else None,
        )
        try:
            self.validate_submission(call, text)
        except ValueError as error:
            raise InvalidModelReply(reply, str(error)) from error
        return reply

    @staticmethod
    def validate_submission(call: ModelCall, text: str) -> None:
        _, text = split_reasoning(text)
        if isinstance(call.input, GoalInput):
            GoalSubmission.model_validate_json(text)
        else:
            submission = ForwardSubmission.model_validate_json(text)
            if tuple(frame.time_s for frame in submission.frames) != call.input.sample_times_s:
                raise ValueError("Returned sampling times do not match the requested times")
            expected_inventory = {ball.ball_id: ball.kind for ball in call.input.initial.balls}
            for frame in submission.frames:
                if {
                    ball.ball_id: ball.kind for ball in frame.state.balls
                } != expected_inventory or any(
                    ball.status == "unobserved" for ball in frame.state.balls
                ):
                    raise ValueError("Predicted frames must preserve every initial ball slot")
            if {
                ball.ball_id: ball.kind for ball in submission.resting_state.balls
            } != expected_inventory or any(
                ball.status == "unobserved" for ball in submission.resting_state.balls
            ):
                raise ValueError("Forward resting state must preserve every initial ball slot")
            pockets = {circle.geometry_id for circle in call.input.physics.table.pockets}
            cushions = {
                segment.geometry_id for segment in call.input.physics.table.linear_cushions
            } | {circle.geometry_id for circle in call.input.physics.table.circular_cushions}
            for event in submission.events:
                first, second = event.participants
                if first not in expected_inventory:
                    raise ValueError("Contact prediction has no initial ball slot")
                if event.event_type == "ball_ball":
                    valid = second in expected_inventory and first != second
                else:
                    valid = second in (pockets if event.event_type == "pocket" else cushions)
                if not valid:
                    raise ValueError("Contact prediction has unavailable ball/geometry")
