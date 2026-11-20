"""Ledgered VLM evaluation using separately constructed forward and goal requests."""

import base64
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import islice
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from inverser.blind_protocol import verify_blind_protocol
from inverser.capo_env import GoalEpisode
from inverser.evaluation import (
    BudgetExhausted,
    BudgetLimits,
    TrialLedger,
    UnresolvedRequest,
    recorded_replay,
)
from inverser.model_adapters import InvalidModelReply, ModelReply, RenderedAsset
from inverser.physics import Replay, SnookerSimulator
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.schemas import CueAction, Layout, PhysicsConfig
from inverser.study_calls import (
    StudyAdapter,
    StudyBackendConfig,
    load_job,
    lock_run,
    producer_hashes,
    save_job,
    usage_summary,
)
from inverser.vlm_suite import (
    VLMForwardInput,
    VLMForwardOutput,
    VLMGoalInput,
    forward_scores,
    invalid_forward_scores,
    parse_forward_answer,
    parse_goal_answer,
)


def _asset(public: Path, name: str, scene_id: str, *, initial: bool = True) -> RenderedAsset:
    path = (public / name).resolve()
    if not path.is_relative_to(public.resolve()):
        raise ValueError("Image path leaves the public split")
    return _png_asset(path.read_bytes(), scene_id, initial=initial)


def _png_asset(data: bytes, scene_id: str, *, initial: bool = False) -> RenderedAsset:
    with Image.open(io.BytesIO(data)) as png:
        width, height = png.size
    return RenderedAsset(
        case_id=scene_id,
        role="initial" if initial else "goal",
        png_base64=base64.b64encode(data).decode("ascii"),
        sha256=sha256(data).hexdigest(),
        width_px=width,
        height_px=height,
    )


def feedback_image_pair(goal: RenderedAsset, outcome: Image.Image) -> RenderedAsset:
    """Keep INIT at native scale and combine goal/latest outcome for two-image endpoints."""
    with Image.open(io.BytesIO(base64.b64decode(goal.png_base64))) as blueprint:
        image = Image.new("RGB", (blueprint.width + 12 + outcome.width, blueprint.height), "white")
        image.paste(blueprint, (0, 0))
        image.paste(outcome, (blueprint.width + 12, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return _png_asset(buffer.getvalue(), goal.case_id)


def visual_request(
    adapter: StudyAdapter,
    ledger: TrialLedger,
    prompt: str,
    images: tuple[RenderedAsset, ...],
    *,
    episode_key: str,
    seed: int,
    output_schema: dict[str, object] | None = None,
    answer_tag: bool = False,
) -> ModelReply:
    key_body: dict[str, object] = {
        "episode_key": episode_key,
        "backend": adapter.config.model_dump(mode="json"),
        "endpoint": fingerprint(adapter.config.base_url),
        "prompt": prompt,
        "images": [item.sha256 for item in images],
        "seed": seed,
        "output_schema": output_schema,
    }
    if answer_tag:
        key_body["answer_tag"] = True
    key = fingerprint(key_body)
    cached = ledger.reserve_request(
        key, 0 if adapter.config.kind == "mock" else adapter.config.request_token_reservation
    )
    if cached is not None:
        return cached
    try:
        reply = adapter._exchange(
            prompt, images, key, seed, output_schema=output_schema, answer_tag=answer_tag
        )
    except InvalidModelReply as error:
        # A billed empty answer is a valid observation of model failure.
        reply = error.reply
    except httpx.RequestError:
        ledger.fail_request(key, "transport_response_unknown", unknown_response=True)
        raise
    except httpx.HTTPStatusError as error:
        ledger.fail_request(key, f"http_{error.response.status_code}", unknown_response=False)
        raise
    except ValueError:
        ledger.fail_request(key, "response_identity_format_or_usage_error", unknown_response=False)
        raise
    ledger.finish_request(key, reply)
    return reply


def run_vlm_eval(
    split: Path,
    output: Path,
    backend: StudyBackendConfig,
    *,
    tasks: tuple[str, ...],
    count: int | None,
    budget_tokens: int,
    execute: bool,
    seed: int = 0,
    constrain_json: bool = True,
    blind_protocol: Path | None = None,
    arm: str | None = None,
    transport: httpx.BaseTransport | None = None,
    workers: int = 1,
) -> dict[str, object]:
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - {"forward", "T4", "T5"}:
        raise ValueError("Choose distinct forward, T4 and/or T5 evaluation arms")
    if count is not None and count <= 0:
        raise ValueError("The scene limit must be positive")
    if workers < 1 or workers > 64:
        raise ValueError("Evaluation workers must be between 1 and 64")
    manifest_path = split / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    blind = None
    if manifest["split"] == "test":
        if blind_protocol is None or arm is None or count is not None or not constrain_json:
            raise ValueError(
                "Blind test must wait for a locked evaluation protocol and checkpoints"
            )
        if set(tasks) != {"forward", "T4", "T5"}:
            raise ValueError("Blind test must run the full task inventory")
        blind = verify_blind_protocol(blind_protocol, manifest_path, backend, arm, seed)
    elif blind_protocol is not None:
        raise ValueError("A blind freeze protocol cannot authorize development runs")
    public = split / "public"
    private = split / "private"
    sources = (public / "scenes.jsonl", private / "labels.jsonl", private / "physics.json")
    for path in sources:
        if file_sha256(path) != manifest["artifact_sha256"][str(path.relative_to(split))]:
            raise ValueError(f"Dataset hash mismatch: {path.name}")
    with sources[0].open() as stream:
        inputs = [json.loads(line) for line in islice(stream, count)]
    with sources[1].open() as stream:
        labels = [json.loads(line) for line in islice(stream, count)]
    if not inputs or len(inputs) != len(labels):
        raise ValueError("Public and private row counts disagree")
    if any(row["scene_id"] != label["scene_id"] for row, label in zip(inputs, labels, strict=True)):
        raise ValueError("Public and private scene ordering disagree")
    costs = {"forward": 1, "T4": 1, "T5": 6}
    planned_jobs = [
        (scene, label, task)
        for scene, label in zip(inputs, labels, strict=True)
        for task in tasks
        if label["regime"] == "standard" or task == "T5"
    ]
    request_cap = sum(costs[task] for _, _, task in planned_jobs)
    limits = BudgetLimits(
        max_model_requests=request_cap,
        max_total_tokens=budget_tokens,
        max_physics_evaluations=sum(
            costs[task] for _, _, task in planned_jobs if task != "forward"
        ),
    )
    run_hash = lock_run(
        output,
        {
            "schema_version": "inverser-vlm-base-eval-v1",
            "answer_contract": "answer-tag-v1",
            "data": file_sha256(manifest_path),
            "tasks": list(tasks),
            "count": len(inputs),
            "seed": seed,
            "constrain_json": constrain_json,
            "backend": backend.model_dump(mode="json"),
            "endpoint_hash": fingerprint(backend.base_url),
            "limits": limits.model_dump(mode="json"),
            "producer": producer_hashes(
                "vlm_eval.py", "vlm_suite.py", "capo_env.py", "study_calls.py"
            ),
            **({"blind_protocol_sha256": blind, "arm": arm} if blind else {}),
        },
        reserve_bytes=20_000_000,
    )
    if not execute:
        return {"status": "planned_no_calls", "run_hash": run_hash, "request_cap": request_cap}
    local_state = threading.local()
    sim_lock = threading.Lock()
    all_simulators: list[SnookerSimulator] = []
    physics = json.loads(sources[2].read_text())
    ledger = TrialLedger(output / "ledger.sqlite", limits)
    rows: list[dict[str, Any]] = []
    stop_error = None
    abort = threading.Event()

    def get_simulator(regime: str) -> SnookerSimulator:
        sims: dict[str, SnookerSimulator] = getattr(local_state, "sims", {})
        if not hasattr(local_state, "sims"):
            local_state.sims = sims
        if regime not in sims:
            sim = SnookerSimulator(PhysicsConfig.model_validate(physics[regime]["config"]))
            sims[regime] = sim
            with sim_lock:
                all_simulators.append(sim)
        return sims[regime]

    def checked_asset(name: str, scene_id: str, *, initial: bool = True) -> RenderedAsset:
        asset = _asset(public, name, scene_id, initial=initial)
        if asset.sha256 != manifest["artifact_sha256"][f"public/{name}"]:
            raise ValueError(f"Dataset PNG hash mismatch: {name}")
        return asset

    try:
        with StudyAdapter(
            backend.model_copy(update={"execution_authorized": True}), transport=transport
        ) as adapter:

            def evaluate_job(
                item: tuple[dict[str, Any], dict[str, Any], str],
            ) -> dict[str, Any] | None:
                if abort.is_set():
                    return None
                scene, label, task = item
                scene_id = scene["scene_id"]
                job = output / "jobs" / f"{scene_id}-{task}.json"
                if (saved := load_job(job, run_hash)) is not None:
                    return saved
                info = {
                    "scene_id": scene_id,
                    "group_id": label["group_id"],
                    "family": label["family"],
                    "ood": label["regime"] != "standard",
                    "task": task,
                }

                def send(
                    ordinal: int,
                    prompt: str,
                    images: tuple[RenderedAsset, ...],
                ) -> ModelReply:
                    return visual_request(
                        adapter,
                        ledger,
                        prompt,
                        images,
                        episode_key=f"{scene_id}/{task}/{ordinal}",
                        seed=seed,
                        output_schema=None,
                        answer_tag=constrain_json,
                    )

                if task == "forward":
                    forward = VLMForwardInput.model_validate(scene["forward"])
                    reply = send(
                        1,
                        forward.prompt(),
                        (checked_asset(forward.image, scene_id),),
                    )
                    truth = VLMForwardOutput.model_validate(label["truth"])
                    try:
                        predicted = parse_forward_answer(
                            reply.content, ball_count=forward.ball_count
                        )
                        scores = forward_scores(predicted, truth, forward.ball_radius_m)
                    except ValueError:
                        predicted = None
                        scores = invalid_forward_scores()
                    row = info | {
                        "valid": predicted is not None,
                        "scores": scores,
                        "true_object_collision": any(truth.object_collisions),
                        "predicted_no_object_collision": (
                            predicted is not None and not any(predicted.object_collisions)
                        ),
                        "replies": [reply.model_dump(mode="json")],
                    }
                else:
                    query = VLMGoalInput.model_validate(
                        next(
                            entry
                            for entry in scene["goals"]
                            if entry["task_id"].startswith(f"{task}_")
                        )
                    )
                    sim = get_simulator(label["regime"])
                    env = GoalEpisode(
                        query,
                        Layout.model_validate(label["initial"]),
                        Layout.model_validate(label["goal"]),
                        sim,
                    )
                    initial_images = (
                        checked_asset(query.initial_image, scene_id),
                        checked_asset(query.goal_image, scene_id, initial=False),
                    )
                    images: tuple[RenderedAsset, ...] = initial_images
                    prompt = query.prompt() + "\nThe images in order are INITIAL and GOAL."
                    history = []
                    replies: list[dict[str, Any]] = []
                    while not env.done:
                        ordinal = len(replies) + 1
                        reply = send(ordinal, prompt, images)
                        replies.append(reply.model_dump(mode="json"))

                        def judge(action: CueAction) -> Replay:
                            return recorded_replay(
                                env.initial,
                                action,
                                env.simulator,
                                ledger,
                                namespace=f"{scene_id}/{task}/{ordinal}",
                                role="hidden_judge",
                            )[0]

                        try:
                            normalized = parse_goal_answer(reply.content).model_dump_json()
                        except ValueError:
                            normalized = ""
                        attempt = env.step(normalized, replay_action=judge)
                        if not env.done:
                            feedback, image = env.feedback(attempt)
                            history.append(feedback)
                            prompt = (
                                query.prompt()
                                + "\nAttempt observations in order:\n"
                                + ("\n".join(history))
                            )
                            if image is not None:
                                images = (
                                    initial_images[0],
                                    feedback_image_pair(initial_images[1], image),
                                )
                                prompt += (
                                    "\nImages: INITIAL; GOAL (left), latest outcome (right)."
                                )
                            else:
                                images = initial_images
                                prompt += "\nImages: INITIAL, GOAL."
                    row = info | {
                        "success": env.attempts[-1].success,
                        "attempts": [entry.model_dump(mode="json") for entry in env.attempts],
                        "replies": replies,
                    }
                save_job(job, run_hash, row)
                return row

            if workers == 1:
                for item in planned_jobs:
                    if (done_row := evaluate_job(item)) is not None:
                        rows.append(done_row)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(evaluate_job, item) for item in planned_jobs]
                    for future in futures:
                        try:
                            done_row = future.result()
                            if done_row is not None:
                                rows.append(done_row)
                        except (
                            httpx.HTTPError,
                            BudgetExhausted,
                            UnresolvedRequest,
                            ValueError,
                        ) as error:
                            abort.set()
                            if stop_error is None:
                                stop_error = type(error).__name__
    except (httpx.HTTPError, BudgetExhausted, UnresolvedRequest, ValueError) as error:
        stop_error = type(error).__name__
    finally:
        with ledger._lock:
            all_replies = [
                ModelReply.model_validate_json(payload)
                for (payload,) in ledger.connection.execute(
                    "SELECT reply_json FROM requests WHERE reply_json IS NOT NULL"
                )
            ]
            req_status = dict(
                ledger.connection.execute("SELECT status, COUNT(*) FROM requests GROUP BY status")
            )
        done_ok = len(rows) == len(planned_jobs) and stop_error is None
        report: dict[str, object] = {
            "status": "complete" if done_ok else "stopped",
            "jobs": len(rows),
            "run_hash": run_hash,
            "stop_error": stop_error,
            "usage": ledger.usage_totals(),
            "counts": ledger.counts(),
            "usage_by_reply": usage_summary(all_replies),
            "request_status": req_status,
            "simulator_rollouts_this_process": sum(sim.rollouts for sim in all_simulators),
        }
        write_json(output / "summary.json", report)
        ledger.close()
    return report


def forward_schema(ball_count: int) -> dict[str, object]:
    """Constrain dynamic slot counts at decode time, not just in postvalidation."""
    schema = VLMForwardOutput.model_json_schema()
    props = schema["properties"]
    props["object_collisions"].update(minItems=ball_count - 1, maxItems=ball_count - 1)
    if ball_count == 1:
        props["first_contact"] = {"type": "null"}
    else:
        props["first_contact"]["anyOf"][0]["maximum"] = ball_count - 1
    props["frames"]["items"].update(minItems=ball_count, maxItems=ball_count)
    props["resting_state"].update(minItems=ball_count, maxItems=ball_count)
    return schema
