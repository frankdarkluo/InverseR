import json
import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self

import httpx
import numpy as np
from pydantic import Field, model_validator

from inverser.azure_study import AzurePilotConfig, load_azure_backends
from inverser.diagnostics import ForwardScore
from inverser.evaluation import TrialLedger
from inverser.metrics import EndpointScore
from inverser.model_adapters import (
    BackendConfig,
    InvalidModelReply,
    ModelAdapter,
    ModelCall,
    ModelReply,
    RenderedAsset,
    TokenUsage,
    _object,
    _token,
)
from inverser.runtime import PROJECT_ROOT, check_storage, file_sha256, fingerprint, write_json
from inverser.schemas import Layout, Record
from inverser.study import (
    CandidateAction,
    ForwardInput,
    ForwardSubmission,
    GoalInput,
    GoalSubmission,
    SemanticContact,
    SubmittedAction,
)
from inverser.study_runner import JobEnvelope

LOGGER = logging.getLogger(__name__)
ReasoningEffort = Literal["minimal", "low", "medium", "high"]
# Pre-registered sensitivity tolerances; the 1-radius primary threshold is unchanged.
TOLERANCE_RADII = (0.25, 0.5, 1.0, 2.0, 5.0)
KINEMATICS_NOTE = (
    "Simulator-measured cue-ball linear and angular velocity immediately after the cue impact; "
    "table frame x right, y up, z vertical."
)
SELECTION_NOTE = (
    "Choose exactly one candidate_id whose single execution from the initial state best reaches "
    "the goal state."
)
FEEDBACK_NOTE = (
    "Each earlier attempt was executed once from the same initial state in the declared "
    "simulator and its simulated outcome is listed. An outcome with goal_reached=true ends the "
    "episode; otherwise the action submitted at attempt max_attempts is final."
)
REFLECTION_NOTE = (
    "Earlier attempts were not executed and no simulated outcome is available. The action "
    "submitted at attempt max_attempts is final."
)
FIRST_ATTEMPT_NOTE = (
    "This is the first of max_attempts attempts. Submit one cue action; later attempts may "
    "revise it."
)


class StudyBackendConfig(BackendConfig):
    max_output_tokens: int = Field(default=1024, gt=0, le=32768)
    timeout_s: float = Field(default=30, gt=0, le=900)


def study_azure_backends(
    source: Path,
    endpoint_index: int,
    *,
    reasoning_effort: ReasoningEffort,
    max_output_tokens: int,
    request_token_reservation: int,
    timeout_s: float,
) -> tuple[StudyBackendConfig, ...]:
    return tuple(
        StudyBackendConfig.model_validate(
            {
                **backend.model_dump(),
                "base_url": backend.base_url,
                "reasoning_effort": reasoning_effort,
                "max_output_tokens": max_output_tokens,
                "request_token_reservation": request_token_reservation,
                "timeout_s": timeout_s,
            }
        )
        for backend in load_azure_backends(source, endpoint_index, AzurePilotConfig())
    )


class PostImpactKinematics(Record):
    velocity_mps: tuple[float, float, float]
    angular_velocity_radps: tuple[float, float, float]


class OracleForwardInput(Record):
    schema_version: Literal["study-forward-oracle-kinematics-v1"] = (
        "study-forward-oracle-kinematics-v1"
    )
    forward: ForwardInput
    cue_ball_post_impact: PostImpactKinematics


class BankSelectionInput(Record):
    schema_version: Literal["study-bank-selection-v1"] = "study-bank-selection-v1"
    goal: GoalInput
    candidates: tuple[CandidateAction, ...] = Field(min_length=2, max_length=32)

    @model_validator(mode="after")
    def validate_candidates(self) -> Self:
        if len({candidate.candidate_id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("Candidate IDs must be unique")
        return self


class BankSelectionSubmission(Record):
    candidate_id: str = Field(min_length=1)


class SimulatedOutcome(Record):
    resting_state: Layout | None
    events: tuple[SemanticContact, ...] = ()
    goal_reached: bool
    max_position_error_radii: float | None = Field(default=None, ge=0)
    failure: (
        Literal["scratch", "inventory_or_pocket_status_mismatch", "position", "simulation_failure"]
        | None
    )

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if (self.resting_state is None) != (self.failure == "simulation_failure"):
            raise ValueError("Only a failed simulation lacks a resting state")
        if self.goal_reached != (self.failure is None):
            raise ValueError("Goal success and the failure category disagree")
        return self


class AttemptFeedback(Record):
    attempt: int = Field(ge=1)
    action: SubmittedAction
    outcome: SimulatedOutcome | None


class FeedbackGoalInput(Record):
    schema_version: Literal["study-goal-feedback-v1"] = "study-goal-feedback-v1"
    goal: GoalInput
    attempt: int = Field(ge=1)
    max_attempts: int = Field(ge=1, le=16)
    # None on the first attempt, so every feedback condition shares one opening proposal.
    simulator_feedback: bool | None
    history: tuple[AttemptFeedback, ...] = ()

    @model_validator(mode="after")
    def validate_history(self) -> Self:
        if self.attempt != len(self.history) + 1 or self.attempt > self.max_attempts:
            raise ValueError("Attempt number must follow the recorded history within the cap")
        if (self.attempt == 1) != (self.simulator_feedback is None):
            raise ValueError("Only the shared first attempt leaves the feedback condition unset")
        if [item.attempt for item in self.history] != list(range(1, self.attempt)):
            raise ValueError("History attempts must be numbered consecutively from one")
        if any((item.outcome is None) == self.simulator_feedback for item in self.history):
            raise ValueError("Every earlier attempt has an outcome exactly when feedback is on")
        if any(item.outcome is not None and item.outcome.goal_reached for item in self.history):
            raise ValueError("A reached goal ends the episode before another attempt")
        return self


class StudyCall(Record):
    view: Literal["structured", "bev"]
    input: OracleForwardInput | BankSelectionInput | FeedbackGoalInput = Field(
        discriminator="schema_version"
    )
    images: tuple[RenderedAsset, ...] = ()

    @model_validator(mode="after")
    def validate_view(self) -> Self:
        if self.view == "bev" and not isinstance(self.input, BankSelectionInput):
            raise ValueError("Only bank selection has an image variant in this study")
        self.base_call()
        return self

    def base_call(self) -> ModelCall:
        if isinstance(self.input, OracleForwardInput):
            return ModelCall(view="structured", input=self.input.forward, images=self.images)
        return ModelCall(view=self.view, input=self.input.goal, images=self.images)

    def prompt(self) -> str:
        data = json.loads(self.base_call().prompt())["task"]
        data["schema_version"] = self.input.schema_version
        schema: dict[str, object]
        if isinstance(self.input, OracleForwardInput):
            data["cue_ball_post_impact_state"] = {
                **self.input.cue_ball_post_impact.model_dump(mode="json"),
                "meaning": KINEMATICS_NOTE,
            }
            schema = ForwardSubmission.model_json_schema()
        elif isinstance(self.input, BankSelectionInput):
            data["candidate_actions"] = [
                candidate.model_dump(mode="json") for candidate in self.input.candidates
            ]
            data["selection_rule"] = SELECTION_NOTE
            schema = BankSelectionSubmission.model_json_schema()
        else:
            data["attempt"] = self.input.attempt
            data["max_attempts"] = self.input.max_attempts
            data["history"] = [item.model_dump(mode="json") for item in self.input.history]
            data["feedback_protocol"] = (
                FIRST_ATTEMPT_NOTE
                if self.input.simulator_feedback is None
                else FEEDBACK_NOTE
                if self.input.simulator_feedback
                else REFLECTION_NOTE
            )
            schema = GoalSubmission.model_json_schema()
        return json.dumps(
            {"task": data, "output_schema": schema},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def validate_reply(self, text: str) -> None:
        if isinstance(self.input, OracleForwardInput):
            ModelAdapter.validate_submission(self.base_call(), text)
        elif isinstance(self.input, BankSelectionInput):
            choice = BankSelectionSubmission.model_validate_json(text)
            if choice.candidate_id not in {c.candidate_id for c in self.input.candidates}:
                raise ValueError("Selected candidate is not in the declared bank")
        else:
            GoalSubmission.model_validate_json(text)


class StudyAdapter(ModelAdapter):
    """Mirrors the frozen ModelAdapter HTTP exchange for ModelCall and StudyCall prompts.

    Unlike the frozen adapter, an empty completion (e.g. reasoning exhausted the cap) is kept as
    a billed invalid reply with its usage instead of aborting the run.
    """

    def study_request_key(self, call: StudyCall) -> str:
        return fingerprint(
            {
                "backend": self.config.model_dump(mode="json"),
                "endpoint_hash": fingerprint(self.config.base_url),
                "call": call.model_dump(mode="json"),
                "prompt": call.prompt(),
            }
        )

    def _exchange(
        self, prompt: str, images: tuple[RenderedAsset, ...], request_id: str, seed: int
    ) -> ModelReply:
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + image.png_base64},
            }
            for image in images
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
                "seed": seed,
            }
        response = self._client.post(url, headers=headers, params=params, json=request_payload)
        response.raise_for_status()
        body = _object(response.json(), "body")
        if body.get("model") != self.config.model_id:
            raise ValueError("Returned model identity does not match the configured deployment")
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Exactly one model response choice is required")
        choice = _object(choices[0], "choice")
        text = _object(choice.get("message"), "message").get("content")
        if text is not None and not isinstance(text, str):
            raise ValueError("Model response content must be text")
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
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ValueError("Provider finish_reason must be a string")
        reply = ModelReply(
            backend_alias=self.config.alias,
            model_id=self.config.model_id,
            content=text or "",
            usage=usage,
            finish_reason=finish_reason,
        )
        if not reply.content:
            raise InvalidModelReply(reply, f"Empty completion (finish_reason={finish_reason})")
        return reply

    def _complete(self, call: ModelCall, request_id: str) -> ModelReply:
        reply = self._exchange(call.prompt(), call.images, request_id, call.seed)
        try:
            self.validate_submission(call, reply.content)
        except ValueError as error:
            raise InvalidModelReply(reply, str(error)) from error
        return reply

    def complete_study(self, call: StudyCall, request_id: str) -> ModelReply:
        reply = self._exchange(call.prompt(), call.images, request_id, 0)
        try:
            call.validate_reply(reply.content)
        except ValueError as error:
            raise InvalidModelReply(reply, str(error)) from error
        return reply


def request_study_model(adapter: StudyAdapter, call: StudyCall, ledger: TrialLedger) -> ModelReply:
    key = adapter.study_request_key(call)
    reservation = 0 if adapter.config.kind == "mock" else adapter.config.request_token_reservation
    cached = ledger.reserve_request(key, reservation)
    if cached is not None:
        return cached
    try:
        reply = adapter.complete_study(call, key)
    except InvalidModelReply as error:
        ledger.finish_request(key, error.reply)
        ledger.reject_reply(key)
        raise
    except httpx.RequestError:
        ledger.fail_request(key, "transport_response_unknown", unknown_response=True)
        raise
    except httpx.HTTPStatusError:
        ledger.fail_request(key, "http_status_error", unknown_response=False)
        raise
    except ValueError:
        ledger.fail_request(key, "response_identity_format_or_usage_error", unknown_response=False)
        raise
    ledger.finish_request(key, reply)
    return reply


class RequestOutcome(Record):
    reply: ModelReply | None
    error_code: str | None
    stop_error: str | None


def guarded_request(send: Callable[[], ModelReply]) -> RequestOutcome:
    """Map one billed request to a row outcome; service errors stop the run without retries."""
    try:
        return RequestOutcome(reply=send(), error_code=None, stop_error=None)
    except InvalidModelReply as error:
        return RequestOutcome(reply=error.reply, error_code="invalid_submission", stop_error=None)
    except httpx.HTTPStatusError as error:
        code = f"http_{error.response.status_code}"
        LOGGER.error("API service error %s; stopping without retries", code)
        return RequestOutcome(reply=None, error_code=code, stop_error=code)
    except httpx.RequestError:
        LOGGER.error("API transport failed; stopping without rebilling")
        code = "transport_response_unknown"
        return RequestOutcome(reply=None, error_code=code, stop_error=code)


def producer_hashes(*names: str) -> dict[str, str]:
    return {name: file_sha256(Path(__file__).with_name(name)) for name in names}


def require_data_path(output: Path) -> None:
    if output.is_symlink() or not output.resolve().is_relative_to(
        (PROJECT_ROOT / "data").resolve()
    ):
        raise ValueError("Study artifacts must remain inside the project data directory")


def create_artifact_dir(output: Path, reserve_bytes: int) -> None:
    require_data_path(output)
    if output.exists():
        raise ValueError("A new frozen study artifact requires a new output directory")
    check_storage(reserve_bytes)
    output.mkdir(parents=True)


def lock_run(output: Path, protocol: dict[str, object], *, reserve_bytes: int) -> str:
    require_data_path(output)
    run_hash = fingerprint(protocol)
    manifest = {**protocol, "run_hash": run_hash}
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Changed study protocol requires a new output directory")
        return run_hash
    if output.exists() and any(output.iterdir()):
        raise ValueError("A new study run requires an empty output directory")
    check_storage(reserve_bytes)
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    return run_hash


def load_job(path: Path, run_hash: str) -> dict[str, object] | None:
    if not path.exists():
        return None
    envelope = JobEnvelope.model_validate_json(path.read_text())
    if envelope.run_hash != run_hash:
        raise ValueError("Cached study job belongs to another run")
    return envelope.payload


def save_job(path: Path, run_hash: str, payload: dict[str, object]) -> None:
    write_json(
        path,
        JobEnvelope(
            run_hash=run_hash, payload=payload, payload_sha256=fingerprint(payload)
        ).model_dump(mode="json"),
    )


def usage_summary(rows: list[ModelReply]) -> dict[str, object]:
    def stats(values: list[int]) -> dict[str, float | int] | None:
        if not values:
            return None
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "max": max(values),
        }

    replies = [reply for reply in rows if not reply.cached]
    return {
        "billed_replies": len(replies),
        "shared_cached_replies": len(rows) - len(replies),
        "input_tokens": stats(
            [r.usage.input_tokens for r in replies if r.usage.input_tokens is not None]
        ),
        "output_tokens": stats(
            [r.usage.output_tokens for r in replies if r.usage.output_tokens is not None]
        ),
        "usage_unavailable": sum(r.usage.total_tokens is None for r in replies),
        "finish_reasons": dict(Counter(str(r.finish_reason) for r in replies)),
    }


def endpoint_tolerance_profile(score: EndpointScore, radius_m: float) -> dict[str, bool]:
    return {
        str(k): score.failure in (None, "position")
        and score.max_position_error_m is not None
        and score.max_position_error_m <= k * radius_m
        for k in TOLERANCE_RADII
    }


def forward_tolerance_profile(
    prediction: ForwardSubmission, truth: ForwardSubmission, score: ForwardScore, radius_m: float
) -> dict[str, dict[str, bool]]:
    statuses = {ball.ball_id: ball.status for ball in truth.resting_state.balls}
    resting_status_match = all(
        ball.status == statuses[ball.ball_id] for ball in prediction.resting_state.balls
    )
    frames_error = score.maximum_position_error_m
    resting_error = score.resting_max_position_error_m
    return {
        "frames": {
            str(k): score.status_mismatches == 0
            and (frames_error is None or frames_error <= k * radius_m)
            for k in TOLERANCE_RADII
        },
        "resting": {
            str(k): resting_status_match
            and (resting_error is None or resting_error <= k * radius_m)
            for k in TOLERANCE_RADII
        },
    }
