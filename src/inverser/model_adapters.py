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
    kind: Literal["mock", "api", "local_vllm"]
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
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    timeout_s: float = Field(default=30, gt=0, le=300)

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
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
        if self.kind == "local_vllm":
            hostname = endpoint.hostname
            if hostname != "localhost":
                try:
                    loopback = ip_address(hostname).is_loopback
                except ValueError as error:
                    raise ValueError("Local vLLM endpoint must use a loopback address") from error
                if not loopback:
                    raise ValueError("Local vLLM endpoint must use a loopback address")
        return self


class TokenUsage(Record):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ModelReply(Record):
    backend_alias: str
    model_id: str
    content: str
    usage: TokenUsage
    cached: bool = False
    finish_reason: str | None = None


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
        if config.kind == "api" and not api_key and token_provider is None:
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
        messages = [
            {"role": "system", "content": "Return only the requested numerical JSON."},
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
                "temperature": 0,
                "max_tokens": self.config.max_output_tokens,
                "seed": call.seed,
            }
        response = self._client.post(url, headers=headers, params=params, json=request_payload)
        response.raise_for_status()
        payload: object = response.json()
        body = _object(payload, "body")
        if body.get("model") != self.config.model_id:
            raise ValueError("Returned model identity does not match the configured deployment")
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Exactly one model response choice is required")
        message = _object(_object(choices[0], "choice").get("message"), "message")
        text = message.get("content")
        if not isinstance(text, str) or not text:
            raise ValueError("A nonempty textual model response is required")
        usage_data = body.get("usage")
        if usage_data is None:
            LOGGER.warning(
                "Usage unavailable for backend %s request %s", self.config.alias, request_id
            )
            usage = TokenUsage()
        else:
            fields = _object(usage_data, "usage")
            usage = TokenUsage(
                input_tokens=_token(fields.get("prompt_tokens")),
                output_tokens=_token(fields.get("completion_tokens")),
                total_tokens=_token(fields.get("total_tokens")),
            )
        finish_reason = _object(choices[0], "choice").get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ValueError("Provider finish_reason must be a string")
        reply = ModelReply(
            backend_alias=self.config.alias,
            model_id=self.config.model_id,
            content=text,
            usage=usage,
            finish_reason=finish_reason,
        )
        try:
            self.validate_submission(call, text)
        except ValueError as error:
            raise InvalidModelReply(reply, str(error)) from error
        return reply

    @staticmethod
    def validate_submission(call: ModelCall, text: str) -> None:
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
