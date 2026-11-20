"""Budget-matched adaptations of CueTip's event objective and DeepPHY's WM prompting."""

import io
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from pydantic import Field
from scipy.optimize import dual_annealing

from inverser.capo_env import GoalAttempt, GoalEpisode
from inverser.evaluation import BudgetExhausted, BudgetLimits, TrialLedger, recorded_replay
from inverser.model_adapters import ModelReply
from inverser.physics import Replay, SnookerSimulator
from inverser.preview import render_vlm_forward_scene
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.schemas import CueAction, Layout, PhysicsConfig
from inverser.self_verify import select_by_forecast
from inverser.study import semantic_contacts
from inverser.study_calls import (
    StudyAdapter,
    StudyBackendConfig,
    load_job,
    lock_run,
    producer_hashes,
    save_job,
)
from inverser.vlm_eval import (
    _asset,
    _png_asset,
    feedback_image_pair,
    visual_request,
)
from inverser.vlm_suite import (
    VLMForwardInput,
    VLMForwardOutput,
    VLMGoalInput,
    VLMGoalOutput,
    _extract_answer_tags,
    parse_forward_answer,
    parse_goal_answer,
)

SEARCH_CALLS = 6
SELF_PROPOSALS = 8
EVENT_TYPES = frozenset({"ball_ball", "cushion", "pocket"})
PROPOSAL_TAGS = (
    "prediction",
    "desired_events",
    "azimuth_deg",
    "cue_speed_mps",
    "tip_side",
    "tip_vertical",
)


class EventProposal(VLMGoalOutput):
    prediction: str = Field(min_length=1, max_length=400)
    desired_events: tuple[str, ...] = Field(max_length=8)

    def cue_action(self) -> CueAction:
        return CueAction(**self.model_dump(exclude={"prediction", "desired_events"}))


def parse_event_proposal(text: str) -> EventProposal:
    stripped = text.strip()
    if stripped.startswith("{"):
        parsed = EventProposal.model_validate_json(stripped)
        if set(parsed.desired_events) - EVENT_TYPES:
            raise ValueError("Invalid desired_events entry")
        return parsed
    tags = _extract_answer_tags(text, PROPOSAL_TAGS)
    raw_events = tags["desired_events"].strip()
    if raw_events.lower() in ("none", "[]"):
        events: tuple[str, ...] = ()
    else:
        events = tuple(part.strip().lower() for part in raw_events.split(",") if part.strip())
    if not events and raw_events.lower() not in ("none", "[]"):
        raise ValueError("Empty desired_events must use 'none'")
    if set(events) - EVENT_TYPES:
        raise ValueError("Only ball_ball, cushion, pocket are valid desired_events")
    return EventProposal(
        prediction=tags["prediction"][:400],
        desired_events=events,
        azimuth_deg=float(tags["azimuth_deg"]),
        cue_speed_mps=float(tags["cue_speed_mps"]),
        tip_side=float(tags["tip_side"]),
        tip_vertical=float(tags["tip_vertical"]),
    )


class SearchLimit(Exception):
    """Local annealing limit, not a provider/ledger budget failure."""


def event_distance(predicted: tuple[str, ...], observed: tuple[str, ...]) -> float:
    if not predicted:
        return 0.0
    j = 0
    for event in observed:
        if j < len(predicted) and predicted[j] == event:
            j += 1
    return 1 - j / len(predicted)


def event_guided_annealing(
    proposal: EventProposal,
    score: Callable[[CueAction], tuple[GoalAttempt, tuple[str, ...]]],
    seed: int,
) -> tuple[GoalAttempt, list[GoalAttempt]]:
    """At most six evaluator calls, even if SciPy's optimizer changes its stop convention."""
    desired = proposal.desired_events
    if any(item not in EVENT_TYPES for item in desired):
        raise ValueError("Only normalized contact types may guide the optimizer")
    history: list[GoalAttempt] = []
    values = []
    x0 = np.asarray(
        [
            proposal.azimuth_deg,
            max(0.001, proposal.cue_speed_mps),
            proposal.tip_side,
            proposal.tip_vertical,
        ]
    )

    def objective(x: np.ndarray) -> float:
        if len(history) >= SEARCH_CALLS:
            raise SearchLimit
        azimuth, speed, side, vertical = map(float, x)
        ratio = max(1.0, float(np.hypot(side, vertical)) / 0.5)
        action = CueAction(
            azimuth_deg=azimuth % 360,
            cue_speed_mps=speed,
            tip_side=side / ratio,
            tip_vertical=vertical / ratio,
        )
        attempt, events = score(action)
        history.append(attempt)
        # A feasible goal always beats an unsuccessful shot, independently of event agreement.
        loss = min(attempt.loss_radii, 10_000) if attempt.loss_radii is not None else 10_000.0
        value = (
            (0 if attempt.success else 1) * 20_000 + loss + 0.5 * event_distance(desired, events)
        )
        values.append(value)
        return value

    try:
        dual_annealing(
            objective,
            [(0, 360), (0.001, 3), (-0.5, 0.5), (-0.5, 0.5)],
            x0=x0,
            maxfun=SEARCH_CALLS,
            no_local_search=True,
            rng=np.random.default_rng(seed),
        )
    except SearchLimit:
        pass
    return history[int(np.argmin(values))], history


def run_agentic_baseline(
    split: Path,
    output: Path,
    backend: StudyBackendConfig,
    *,
    method: str,
    budget_tokens: int,
    execute: bool,
    count: int | None,
    transport: httpx.BaseTransport | None = None,
    workers: int = 1,
) -> dict[str, object]:
    if method not in ("cuetip", "wm_reflection", "self_verify") or count is not None and count < 1:
        raise ValueError("Choose a supported adaptation and a positive optional limit")
    if workers < 1 or workers > 64:
        raise ValueError("Agentic baseline workers must be between 1 and 64")
    if method == "self_verify" and backend.temperature <= 0:
        raise ValueError("Self-verification must sample distinct-seeded proposals")
    manifest_path = split / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["split"] != "valid":
        raise ValueError("Agentic baseline development is restricted to valid")
    paths = [split / name for name in ("public/scenes.jsonl", "private/labels.jsonl")]
    for path in paths:
        if file_sha256(path) != manifest["artifact_sha256"][str(path.relative_to(split))]:
            raise ValueError("Frozen split row hash mismatch")
    queries, labels = [
        [json.loads(line) for line in path.read_text().splitlines()][:count] for path in paths
    ]
    model_cap = {"cuetip": 1, "wm_reflection": SEARCH_CALLS, "self_verify": 2 * SELF_PROPOSALS}
    limits = BudgetLimits(
        max_model_requests=len(queries) * model_cap[method],
        max_total_tokens=budget_tokens,
        max_physics_evaluations=len(queries) * (1 if method == "self_verify" else SEARCH_CALLS + 1),
    )
    run_hash = lock_run(
        output,
        {
            "schema_version": "agentic-vlm-adaptations-v1",
            "method": method,
            "split_sha256": file_sha256(manifest_path),
            "count": len(queries),
            "backend": backend.model_dump(mode="json"),
            "endpoint": fingerprint(backend.base_url),
            "limits": limits.model_dump(mode="json"),
            "producer": producer_hashes("agentic_baselines.py", "capo_env.py", "vlm_eval.py"),
            "upstream": (
                "CAPO internal best-of-N"
                if method == "self_verify"
                else "CueTip dual_annealing no_local_search; endpoint and event-sequence objective"
                if method == "cuetip"
                else "DeepPHY predict before action and reflect on failure"
            ),
            "oracle_boundary": (
                "eight inverse proposals and up to eight internal forecasts; only final judge"
                if method == "self_verify"
                else "equal six search attempts + one independent selected-action judge"
            ),
        },
        reserve_bytes=20_000_000,
    )
    if not execute:
        return {"status": "planned_no_calls", "request_cap": limits.max_model_requests}
    local_state = threading.local()
    ledger = TrialLedger(output / "ledger.sqlite", limits)
    completed: list[dict[str, Any]] = []
    stop = None
    abort = threading.Event()

    def get_simulator() -> SnookerSimulator:
        sim = getattr(local_state, "sim", None)
        if sim is None:
            sim = SnookerSimulator(PhysicsConfig())
            local_state.sim = sim
        return sim

    try:
        with StudyAdapter(
            backend.model_copy(update={"execution_authorized": True}), transport=transport
        ) as client:

            def evaluate_scene(
                item: tuple[int, tuple[dict[str, Any], dict[str, Any]]],
            ) -> dict[str, Any] | None:
                if abort.is_set():
                    return None
                i, (query, label) = item
                key = label["scene_id"]
                if key != query["scene_id"]:
                    raise ValueError("Public and private row ordering mismatch")
                path = output / "jobs" / f"{i:04d}.json"
                if (row := load_job(path, run_hash)) is not None:
                    return row
                simulator = get_simulator()
                initial = Layout.model_validate(label["initial"])
                goal = Layout.model_validate(label["goal"])
                task = VLMGoalInput.model_validate(query["goals"][-1])
                env = GoalEpisode(task, initial, goal, simulator)
                initial_images = (
                    _asset(split / "public", task.initial_image, key),
                    _asset(split / "public", task.goal_image, key, initial=False),
                )
                for asset, name in zip(
                    initial_images, (task.initial_image, task.goal_image), strict=True
                ):
                    if asset.sha256 != manifest["artifact_sha256"][f"public/{name}"]:
                        raise ValueError("Frozen split image hash mismatch")
                history: list[GoalAttempt] = []
                selected = None
                chosen_action = None
                model_replies: list[ModelReply] = []
                images = initial_images

                def propose(turn: int, notes: list[str]) -> EventProposal | None:
                    meta = json.dumps(
                        task.model_dump(
                            mode="json",
                            exclude={"initial_image", "goal_image", "episode_id"},
                        ),
                        separators=(",", ":"),
                    )
                    prompt = (
                        "Propose one shot from the INITIAL image to reach the GOAL image without "
                        "scratching (x right, y up; azimuth 0 along +x). State a brief physical "
                        "contact/motion forecast in <prediction>, list expected ordered contacts "
                        "in <desired_events> using ball_ball, cushion, pocket (or 'none'), and "
                        "output the four continuous shot controls in <answer>:\n"
                        "<answer>\n"
                        "<prediction>Cue hits object 1 into pocket and stops.</prediction>\n"
                        "<desired_events>ball_ball, pocket</desired_events>\n"
                        "<azimuth_deg>62.45</azimuth_deg>\n"
                        "<cue_speed_mps>1.38</cue_speed_mps>\n"
                        "<tip_side>0.00</tip_side>\n"
                        "<tip_vertical>0.25</tip_vertical>\n"
                        "</answer>\n"
                        f"{meta}\n"
                        "Images are INITIAL and GOAL (left in feedback panel); any right panel "
                        "is the last observed outcome."
                    )
                    if notes:
                        prompt += "\nReflect on differences between prior forecast and actual "
                        prompt += "outcome before proposing the next move:\n" + "\n".join(notes)
                    reply = visual_request(
                        client,
                        ledger,
                        prompt,
                        images,
                        episode_key=f"{key}/{method}/{turn}",
                        seed=20261119 + i + turn,
                        output_schema=None,
                        answer_tag=True,
                    )
                    model_replies.append(reply)
                    try:
                        return parse_event_proposal(reply.content)
                    except ValueError:
                        return None

                def execute_proposal(
                    action: CueAction, ordinal: int
                ) -> tuple[GoalAttempt, tuple[str, ...]]:
                    observed: tuple[str, ...] = ()

                    def judge(actual: CueAction) -> Replay:
                        nonlocal observed
                        replay, _ = recorded_replay(
                            initial,
                            actual,
                            simulator,
                            ledger,
                            namespace=f"{key}/{method}/search/{ordinal}",
                            role="agent_search",
                        )
                        observed = tuple(
                            event.event_type for event in semantic_contacts(initial, replay.events)
                        )
                        return replay

                    worker = (
                        GoalEpisode(task, initial, goal, simulator) if method == "cuetip" else env
                    )
                    action_json = action.model_dump_json(exclude={"elevation_deg"})
                    return worker.step(action_json, replay_action=judge), observed

                observed_notes: list[str] = []
                if method == "self_verify":
                    single_shot = VLMGoalInput.model_validate(query["goals"][0])
                    proposals = []
                    for turn in range(SELF_PROPOSALS):
                        reply = visual_request(
                            client,
                            ledger,
                            single_shot.prompt() + "\nImages: INITIAL and GOAL.",
                            initial_images,
                            episode_key=f"{key}/self/propose/{turn}",
                            seed=20261119 + i * SELF_PROPOSALS + turn,
                            output_schema=None,
                            answer_tag=True,
                        )
                        model_replies.append(reply)
                        try:
                            proposals.append(parse_goal_answer(reply.content).cue_action())
                        except ValueError:
                            continue
                    forward_task = VLMForwardInput.model_validate(query["forward"])
                    forecast_index = 0

                    def forecast(action: CueAction) -> VLMForwardOutput | None:
                        nonlocal forecast_index
                        forecast_index += 1
                        here = forward_task.model_copy(update={"action": action})
                        image = render_vlm_forward_scene(initial, env.table, action)
                        blob = io.BytesIO()
                        image.save(blob, format="PNG")
                        reply = visual_request(
                            client,
                            ledger,
                            here.prompt(),
                            (_png_asset(blob.getvalue(), key, initial=True),),
                            episode_key=f"{key}/self/forecast/{forecast_index}",
                            seed=20261119,
                            output_schema=None,
                            answer_tag=True,
                        )
                        model_replies.append(reply)
                        try:
                            return parse_forward_answer(reply.content, ball_count=here.ball_count)
                        except ValueError:
                            return None

                    choice = (
                        select_by_forecast(
                            proposals,
                            forecast,
                            initial,
                            goal,
                            ball_radius_m=task.ball_radius_m,
                            table_metres=task.table_metres,
                        )
                        if proposals
                        else None
                    )
                    chosen_action = choice[1] if choice else None
                elif method == "cuetip":
                    proposal = propose(1, [])
                    if proposal:
                        counter = iter(range(1, SEARCH_CALLS + 1))
                        selected, history = event_guided_annealing(
                            proposal,
                            lambda action: execute_proposal(action, next(counter)),
                            seed=20261119 + i,
                        )
                else:
                    for turn in range(1, SEARCH_CALLS + 1):
                        proposal = propose(turn, observed_notes)
                        if proposal is None:
                            observed_notes.append(
                                "Last proposal was invalid JSON or out of bounds."
                            )
                            continue
                        attempt, events = execute_proposal(proposal.cue_action(), turn)
                        history.append(attempt)
                        feedback, image = env.feedback(attempt)
                        observed_notes.append(
                            json.dumps(
                                {
                                    "predicted": proposal.prediction,
                                    "actual_events": events,
                                    "outcome": json.loads(feedback),
                                },
                                separators=(",", ":"),
                            )
                        )
                        if image is not None:
                            images = (
                                initial_images[0],
                                feedback_image_pair(initial_images[1], image),
                            )
                        selected = attempt
                        if attempt.success:
                            break
                judged = None
                action = (
                    chosen_action
                    if method == "self_verify"
                    else (selected.action if selected is not None else None)
                )
                if action is not None:
                    final = GoalEpisode(task, initial, goal, simulator)
                    judged = final.step(
                        action.model_dump_json(exclude={"elevation_deg"}),
                        replay_action=lambda actual: recorded_replay(
                            initial,
                            actual,
                            simulator,
                            ledger,
                            namespace=f"{key}/{method}/judge",
                            role="hidden_judge",
                        )[0],
                    )
                row = {field: label[field] for field in ("scene_id", "group_id", "family")} | {
                    "task": "T4",
                    "success": bool(judged and judged.success),
                    "search_attempts": [attempt.model_dump(mode="json") for attempt in history],
                    "judge": judged.model_dump(mode="json") if judged else None,
                    "replies": [reply.model_dump(mode="json") for reply in model_replies],
                }
                save_job(path, run_hash, row)
                return row

            items = list(enumerate(zip(queries, labels, strict=True)))
            if workers == 1:
                for item in items:
                    if (done_row := evaluate_scene(item)) is not None:
                        completed.append(done_row)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(evaluate_scene, item) for item in items]
                    for future in futures:
                        try:
                            done_row = future.result()
                            if done_row is not None:
                                completed.append(done_row)
                        except (httpx.HTTPError, BudgetExhausted, ValueError, RuntimeError) as err:
                            abort.set()
                            if stop is None:
                                stop = type(err).__name__
    except (httpx.HTTPError, BudgetExhausted, ValueError, RuntimeError) as error:
        stop = type(error).__name__
    finally:
        report: dict[str, object] = {
            "status": "complete" if stop is None and len(completed) == len(queries) else "stopped",
            "run_hash": run_hash,
            "jobs": len(completed),
            "stop_error": stop,
            "counts": ledger.counts(),
            "usage": ledger.usage_totals(),
            "successes": sum(bool(item["success"]) for item in completed),
        }
        write_json(output / "summary.json", report)
        ledger.close()
    return report
