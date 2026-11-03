import base64
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from importlib.metadata import version
from math import cos, dist, hypot, radians, sin
from pathlib import Path
from time import monotonic
from typing import Literal, Self

import httpx
import numpy as np
from PIL import Image
from pydantic import Field, model_validator
from scipy.stats import binomtest

from inverser.azure_study import MODELS, select_pilot_requests
from inverser.baselines import make_action_bank
from inverser.diagnostic_panel import validate_diagnostic_panel
from inverser.diagnostics import (
    CandidateOutcome,
    ForwardScore,
    PairedMeasurement,
    score_bank_selection,
    score_forward,
    summarize_paired,
)
from inverser.evaluation import BudgetLimits, TrialLedger, request_model
from inverser.metrics import endpoint_loss, score_endpoint
from inverser.model_adapters import ModelCall, ModelReply, RenderedAsset, rendered_asset
from inverser.physics import Replay, SimulationFailure, SnookerSimulator
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.schemas import CueAction, Layout, PhysicsConfig, Record, ShotRecord
from inverser.study import (
    ActionBank,
    ForwardInput,
    ForwardSubmission,
    GoalInput,
    PredictedFrame,
    PublicPhysics,
    make_study_case,
    replay_forward_truth,
    translating_ball_ids,
)
from inverser.study_calls import (
    TOLERANCE_RADII,
    BankSelectionInput,
    BankSelectionSubmission,
    ReasoningEffort,
    StudyAdapter,
    StudyCall,
    create_artifact_dir,
    credential_ready,
    forward_tolerance_profile,
    guarded_request,
    load_job,
    lock_run,
    producer_hashes,
    request_study_model,
    save_job,
    study_azure_backends,
    usage_summary,
)

LOGGER = logging.getLogger(__name__)
SELECTION_METHODS = ("forecast_selector", "direct_selection")


class BankPanelConfig(Record):
    schema_version: Literal["development-bank-panel-v1"] = "development-bank-panel-v1"
    layouts: int = Field(default=64, ge=4, le=64)
    candidates: int = Field(default=8, ge=2, le=32)
    bev_layouts: int = Field(default=16, ge=0, le=64)
    seed: int = Field(default=20261019, ge=0)
    sample_times_s: tuple[float, float, float] = (0.25, 0.5, 1.0)
    max_bank_draws: int = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.layouts % 4 or self.bev_layouts % 4 or self.bev_layouts > self.layouts:
            raise ValueError("Bank and image subsets must be family-balanced and nested")
        if not 0 <= self.sample_times_s[0] < self.sample_times_s[1] < self.sample_times_s[2]:
            raise ValueError("Bank sampling times must be nonnegative and strictly increasing")
        return self


class BankRequest(Record):
    request_id: str
    case_id: str
    kind: Literal["forecast", "selection"]
    view: Literal["structured", "bev"]
    candidate_id: str | None
    input_file: str
    image_files: tuple[str, ...] = ()
    prompt_sha256: str

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if any(
            not path.startswith("public/") or ".." in Path(path).parts
            for path in (self.input_file, *self.image_files)
        ):
            raise ValueError("Bank requests may reference only contained public artifacts")
        if (self.candidate_id is None) != (self.kind == "selection"):
            raise ValueError("Exactly the forecast requests name one bank candidate")
        return self


def _asset(panel: Path, case_id: str, name: str) -> RenderedAsset:
    path = panel / name
    with Image.open(path) as image:
        width, height = image.size
    return RenderedAsset.model_validate(
        {
            "case_id": case_id,
            "role": path.stem,
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
            "sha256": file_sha256(path),
            "width_px": width,
            "height_px": height,
        }
    )


def load_bank_call(
    panel: Path, request: BankRequest, *, check: bool = True
) -> ModelCall | StudyCall:
    data = json.loads((panel / request.input_file).read_text())
    images = tuple(_asset(panel, request.case_id, name) for name in request.image_files)
    call: ModelCall | StudyCall
    if request.kind == "forecast":
        call = ModelCall(view=request.view, input=ForwardInput.model_validate(data), images=images)
    else:
        call = StudyCall(
            view=request.view, input=BankSelectionInput.model_validate(data), images=images
        )
    if check and fingerprint(call.prompt()) != request.prompt_sha256:
        raise ValueError("Loaded bank call differs from the frozen visible prompt")
    return call


def _bank_requests(panel: Path) -> tuple[BankRequest, ...]:
    return tuple(
        BankRequest.model_validate_json(line) for line in (panel / "request_index.jsonl").open()
    )


def validate_bank_panel(output: Path) -> dict[str, object]:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, checksum in manifest["artifact_hashes"].items():
        path = output / name
        if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
            raise ValueError("Frozen bank panel contains an out-of-scope artifact")
        if file_sha256(path) != checksum:
            raise ValueError(f"Frozen bank artifact changed: {name}")
    requests = _bank_requests(output)
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("Bank request IDs must be unique")
    for request in requests:
        task = json.loads(load_bank_call(output, request).prompt())["task"]
        if request.kind == "forecast" and "goal" in task:
            raise ValueError("A blind bank forecast contains the goal")
    if any("witness" in path.read_text() for path in (output / "public").rglob("*.json")):
        raise ValueError("A model-visible bank payload names the goal witness")
    config = BankPanelConfig.model_validate(manifest["config"])
    expected = config.layouts * (config.candidates + 1) + config.bev_layouts * (
        config.candidates + 1
    )
    if len(requests) != expected:
        raise ValueError("Bank panel does not have the declared request shape")
    return {
        "status": "frozen_bank_panel_validated",
        "manifest_sha256": file_sha256(output / "manifest.json"),
        "layouts": config.layouts,
        "bev_layouts": config.bev_layouts,
        "candidates_per_layout": config.candidates,
        "request_records": len(requests),
        "request_kinds": dict(Counter(f"{r.kind}/{r.view}" for r in requests)),
        "actual_model_calls": 0,
        "formal_test_freeze": False,
    }


def _scratch(layout: Layout) -> bool:
    return any(ball.kind == "cue" and ball.status == "pocketed" for ball in layout.balls)


@dataclass(frozen=True)
class DrawnBank:
    bank: ActionBank
    draw: int
    witness: str
    goal: GoalInput
    outcomes: tuple[CandidateOutcome, ...]
    truths: dict[str, object]


def draw_bank(
    case_id: str,
    initial: Layout,
    simulator: SnookerSimulator,
    public: PublicPhysics,
    physics: PhysicsConfig,
    *,
    seed: int,
    candidates: int,
    max_draws: int,
    sample_times_s: tuple[float, ...],
) -> DrawnBank:
    """Goal-independent Sobol bank; the goal comes from one valid, non-scratch bank member."""
    for draw in range(max_draws):
        bank = make_action_bank(
            case_id, count=candidates, seed=int(fingerprint((seed, case_id, draw))[:8], 16)
        )
        replays: dict[str, Replay | None] = {}
        for candidate in bank.candidates:
            try:
                replays[candidate.candidate_id] = simulator.replay(initial, candidate.action)
            except SimulationFailure:
                replays[candidate.candidate_id] = None
        eligible = [
            candidate.candidate_id
            for candidate in bank.candidates
            if (replay := replays[candidate.candidate_id]) is not None and not _scratch(replay.goal)
        ]
        if eligible:
            break
    else:
        raise ValueError("No valid non-scratch goal witness within the bounded bank draws")
    rng = np.random.default_rng(int(fingerprint((seed, case_id, draw, "goal"))[:8], 16))
    witness = eligible[int(rng.integers(len(eligible)))]
    witness_replay = replays[witness]
    if witness_replay is None:
        raise RuntimeError("Eligible witness lost its replay")
    goal = GoalInput(
        case_id=case_id, initial=initial, goal=witness_replay.goal.endpoint_view(), physics=public
    )
    radius = goal.physics.table.ball_radius_m
    diagonal = hypot(goal.physics.table.width_m, goal.physics.table.length_m)
    outcomes = []
    truths: dict[str, object] = {}
    for candidate in bank.candidates:
        replay = replays[candidate.candidate_id]
        if replay is None:
            outcomes.append(
                CandidateOutcome(
                    candidate_id=candidate.candidate_id,
                    goal_success=False,
                    failure="simulation_failure",
                )
            )
            truths[candidate.candidate_id] = None
            continue
        outcomes.append(
            CandidateOutcome(
                candidate_id=candidate.candidate_id,
                loss_radii=endpoint_loss(
                    replay.goal,
                    goal.goal,
                    ball_radius_m=radius,
                    table_diagonal_m=diagonal,
                    no_scratch=goal.no_scratch,
                ),
                goal_success=score_endpoint(
                    replay.goal,
                    goal.goal,
                    goal.tolerance_radii * radius,
                    no_scratch=goal.no_scratch,
                ).success,
            )
        )
        truths[candidate.candidate_id] = {
            "truth": replay_forward_truth(initial, replay, physics, sample_times_s).model_dump(
                mode="json"
            ),
            "moving_ball_ids": translating_ball_ids(initial, replay.motion),
        }
    if not next(o for o in outcomes if o.candidate_id == witness).goal_success:
        raise ValueError("Bank witness does not reproduce its own goal")
    return DrawnBank(bank, draw, witness, goal, tuple(outcomes), truths)


def prepare_bank_panel(
    source_panel: Path, output: Path, config: BankPanelConfig
) -> dict[str, object]:
    validation = validate_diagnostic_panel(source_panel)
    protocol = {
        "config": config.model_dump(mode="json"),
        "source_panel_manifest_sha256": validation["manifest_sha256"],
        "producer_hashes": producer_hashes(
            "bank_study.py",
            "study_calls.py",
            "baselines.py",
            "study.py",
            "physics.py",
            "metrics.py",
            "model_adapters.py",
            "preview.py",
        ),
        "runtime_versions": {
            name: version(name)
            for name in ("numpy", "scipy", "pydantic", "pooltool-billiards", "Pillow")
        },
        "partition": "development",
        "bank_prior": "goal_independent_sobol_action_bank",
        "goal_source": "uniform_valid_non_scratch_bank_candidate",
        "actual_model_inference": False,
        "formal_g2_test_freeze": False,
    }
    protocol_hash = fingerprint(protocol)
    if (output / "manifest.json").exists():
        if json.loads((output / "manifest.json").read_text())["protocol_hash"] != protocol_hash:
            raise ValueError("Changed sources/code/protocol require a new bank-panel version")
        return validate_bank_panel(output)
    case_ids = [
        request.case_id
        for request in select_pilot_requests(source_panel, config.layouts, design="goal_structured")
    ]
    create_artifact_dir(output, 300_000_000)
    (output / "public").mkdir()
    (output / "private").mkdir()
    requests = []
    selection = []
    simulators: dict[str, SnookerSimulator] = {}
    draws: Counter[int] = Counter()
    for index, case_id in enumerate(case_ids):
        labels = json.loads((source_panel / "private" / f"{case_id}.json").read_text())
        record = ShotRecord.model_validate(labels["record"])
        key = fingerprint(record.physics.model_dump(mode="json"))
        if key not in simulators:
            simulators[key] = SnookerSimulator(record.physics)
        simulator = simulators[key]
        study_case = make_study_case(record, simulator)
        if study_case.case_id != case_id:
            raise ValueError("Bank layouts must keep the source panel case identity")
        initial = record.initial.endpoint_view()
        drawn = draw_bank(
            case_id,
            initial,
            simulator,
            study_case.public_physics,
            record.physics,
            seed=config.seed,
            candidates=config.candidates,
            max_draws=config.max_bank_draws,
            sample_times_s=config.sample_times_s,
        )
        bank, draw, witness, goal = drawn.bank, drawn.draw, drawn.witness, drawn.goal
        outcomes, truths = drawn.outcomes, drawn.truths
        draws[draw] += 1
        case_dir = output / "public" / case_id
        case_dir.mkdir()
        bev = index < config.bev_layouts
        views: tuple[Literal["structured", "bev"], ...] = (
            ("structured", "bev") if bev else ("structured",)
        )
        images: dict[str, str] = {}
        if bev:
            for role in ("initial", "goal"):
                asset = rendered_asset(goal, role, goal.physics.table)
                path = case_dir / f"{role}.png"
                path.write_bytes(base64.b64decode(asset.png_base64))
                images[role] = str(path.relative_to(output))
        planned: list[tuple[Literal["forecast", "selection"], str | None, Path, Record]] = []
        for candidate in bank.candidates:
            forward = ForwardInput(
                case_id=case_id,
                initial=initial,
                physics=goal.physics,
                action=candidate.action,
                sample_times_s=config.sample_times_s,
            )
            planned.append(
                (
                    "forecast",
                    candidate.candidate_id,
                    case_dir / f"forecast_{candidate.candidate_id}.json",
                    forward,
                )
            )
        planned.append(
            (
                "selection",
                None,
                case_dir / "selection.json",
                BankSelectionInput(goal=goal, candidates=bank.candidates),
            )
        )
        for kind, candidate_id, path, data in planned:
            write_json(path, data.model_dump(mode="json"))
            for view in views:
                image_files: tuple[str, ...] = ()
                if view == "bev":
                    image_files = (
                        (images["initial"],)
                        if kind == "forecast"
                        else (images["initial"], images["goal"])
                    )
                draft = BankRequest(
                    request_id=fingerprint((case_id, kind, candidate_id, view))[:24],
                    case_id=case_id,
                    kind=kind,
                    view=view,
                    candidate_id=candidate_id,
                    input_file=str(path.relative_to(output)),
                    image_files=image_files,
                    prompt_sha256="",
                )
                prompt = load_bank_call(output, draft, check=False).prompt()
                requests.append(draft.model_copy(update={"prompt_sha256": fingerprint(prompt)}))
        write_json(
            output / "private" / f"{case_id}.json",
            {
                "source_task_id": record.task_id,
                "source_group": record.group_id,
                "family": record.family,
                "bank": bank.model_dump(mode="json"),
                "bank_draw": draw,
                "witness_candidate_id": witness,
                "goal": goal.model_dump(mode="json"),
                "outcomes": [outcome.model_dump(mode="json") for outcome in outcomes],
                "forecast_truth": truths,
            },
        )
        selection.append(
            {
                "case_id": case_id,
                "source_group": record.group_id,
                "family": record.family,
                "bev": bev,
                "feasible_candidates": sum(outcome.goal_success for outcome in outcomes),
                "failed_candidates": sum(outcome.failure is not None for outcome in outcomes),
            }
        )
    with (output / "request_index.jsonl").open("w", encoding="utf-8") as stream:
        for request in requests:
            stream.write(request.model_dump_json() + "\n")
    write_json(output / "private/selection.json", {"synthetic": selection})
    write_json(
        output / "manifest.json",
        {
            **protocol,
            "protocol_hash": protocol_hash,
            "family_counts": dict(Counter(item["family"] for item in selection)),
            "bank_draw_counts": {str(k): v for k, v in sorted(draws.items())},
            "feasible_candidate_counts": dict(
                Counter(str(item["feasible_candidates"]) for item in selection)
            ),
            "physics_replays": sum(simulator.rollouts for simulator in simulators.values()),
            "resolved_physics": [simulator.metadata() for simulator in simulators.values()],
            "artifact_hashes": {
                str(path.relative_to(output)): file_sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    report = validate_bank_panel(output)
    write_json(output / "validation.json", report)
    return report


def free_roll_resting_state(initial: Layout, action: CueAction, physics: PublicPhysics) -> Layout:
    # Same straight sliding-then-rolling travel model as the geometry baseline; no collisions/spin.
    parameters = physics.parameters
    coefficient = (
        12 / (49 * parameters["sliding_friction"]) + 25 / (98 * parameters["rolling_friction"])
    ) / parameters["gravity_mps2"]
    transfer = 2 * physics.cue.mass_kg / (physics.cue.mass_kg + parameters["ball_mass_kg"])
    travel = coefficient * (transfer * action.cue_speed_mps) ** 2
    radius = physics.table.ball_radius_m
    angle = radians(action.azimuth_deg)
    balls = []
    for ball in initial.balls:
        if ball.kind == "cue" and ball.xy_m is not None:
            x, y = ball.xy_m[0] + travel * cos(angle), ball.xy_m[1] + travel * sin(angle)
            ball = ball.model_copy(
                update={
                    "xy_m": (
                        min(max(x, radius), physics.table.width_m - radius),
                        min(max(y, radius), physics.table.length_m - radius),
                    )
                }
            )
        balls.append(ball)
    return Layout(balls=tuple(balls))


def trivial_forward_baselines(
    forward: ForwardInput, truth: ForwardSubmission, moving_ball_ids: tuple[str, ...]
) -> dict[str, float | None]:
    """Resting max error (radii) of two trivial forecasters: nothing moves, straight free roll."""
    radius = forward.physics.table.ball_radius_m
    static = score_forward(
        ForwardSubmission(
            frames=tuple(
                PredictedFrame(time_s=time, state=forward.initial)
                for time in forward.sample_times_s
            ),
            resting_state=forward.initial,
            events=(),
        ),
        truth,
        initial=forward.initial,
        moving_ball_ids=moving_ball_ids,
        position_tolerance_m=radius,
        event_time_tolerance_s=0.02,
    ).resting_max_position_error_m
    rest = {ball.ball_id: ball for ball in truth.resting_state.balls}
    free = [
        dist(ball.xy_m, reference)
        for ball in free_roll_resting_state(forward.initial, forward.action, forward.physics).balls
        if ball.xy_m is not None and (reference := rest[ball.ball_id].xy_m) is not None
    ]
    return {
        "static": static / radius if static is not None else None,
        "free_roll": max(free) / radius if free else None,
    }


class BankStudyConfig(Record):
    schema_version: Literal["bank-study-v1"] = "bank-study-v1"
    reasoning_effort: ReasoningEffort = "minimal"
    max_model_requests: int = Field(default=1440, ge=1, le=4000)
    max_total_tokens: int = Field(default=20_000_000, ge=1, le=40_000_000)
    max_output_tokens: int = Field(default=4096, ge=256, le=32768)
    request_token_reservation: int = Field(default=16384, ge=8192)
    timeout_s: float = Field(default=900, gt=0, le=900)

    @model_validator(mode="after")
    def validate_reservation(self) -> Self:
        if self.request_token_reservation < self.max_output_tokens:
            raise ValueError("Each token reservation must cover its completion cap")
        return self


class BankRow(Record):
    model: str
    request_id: str
    case_id: str
    kind: Literal["forecast", "selection"]
    view: Literal["structured", "bev"]
    candidate_id: str | None
    valid_output: bool
    elapsed_s: float = Field(ge=0)
    reply: ModelReply | None = None
    forecast_score: ForwardScore | None = None
    forecast_tolerances: dict[str, dict[str, bool]] | None = None
    error_code: str | None = None


class SelectionResult(Record):
    source_group: str
    success: bool
    regret_radii: float | None
    rank_correlation: float | None
    abstained: bool


def _pick(losses: dict[str, float | None], order: list[str]) -> str | None:
    ranked = [(loss, index) for index, cid in enumerate(order) if (loss := losses[cid]) is not None]
    return order[min(ranked)[1]] if ranked else None


def _select(
    outcomes: tuple[CandidateOutcome, ...],
    losses: dict[str, float | None],
    selected: str | None,
    source_group: str,
) -> SelectionResult:
    score = score_bank_selection(outcomes, losses, selected_candidate_id=selected)
    return SelectionResult(
        source_group=source_group,
        success=score.selected_goal_success,
        regret_radii=score.selected_regret_radii,
        rank_correlation=score.rank_correlation,
        abstained=score.selected_candidate_id is None,
    )


def _method_summary(results: dict[str, SelectionResult]) -> dict[str, object]:
    successes = sum(result.success for result in results.values())
    interval = binomtest(successes, len(results)).proportion_ci()
    regrets = [r.regret_radii for r in results.values() if r.regret_radii is not None]
    correlations = [r.rank_correlation for r in results.values() if r.rank_correlation is not None]
    return {
        "layouts": len(results),
        "selected_goal_successes": successes,
        "success_rate": successes / len(results),
        "success_rate_ci95": [interval.low, interval.high],
        "abstentions": sum(result.abstained for result in results.values()),
        "median_regret_radii": float(np.median(regrets)) if regrets else None,
        "ranked_layouts": len(correlations),
        "mean_rank_correlation": float(np.mean(correlations)) if correlations else None,
    }


def summarize_bank(panel: Path, rows: list[BankRow]) -> dict[str, object]:
    requests = _bank_requests(panel)
    private = {
        case_id: json.loads((panel / "private" / f"{case_id}.json").read_text())
        for case_id in sorted({request.case_id for request in requests})
    }
    by_request = {(row.model, row.request_id): row for row in rows}
    indexed = {
        (request.case_id, request.kind, request.candidate_id, request.view): request
        for request in requests
    }
    results: dict[tuple[str, str, str], dict[str, SelectionResult]] = defaultdict(dict)
    random_success: dict[str, float] = {}
    for case_id, labels in private.items():
        goal = GoalInput.model_validate(labels["goal"])
        outcomes = tuple(CandidateOutcome.model_validate(item) for item in labels["outcomes"])
        actions = {
            item["candidate_id"]: CueAction.model_validate(item["action"])
            for item in labels["bank"]["candidates"]
        }
        order = list(actions)
        group = labels["source_group"]
        radius = goal.physics.table.ball_radius_m
        diagonal = hypot(goal.physics.table.width_m, goal.physics.table.length_m)
        random_success[case_id] = sum(o.goal_success for o in outcomes) / len(outcomes)

        def loss(layout: Layout) -> float:
            return endpoint_loss(
                layout,
                goal.goal,
                ball_radius_m=radius,
                table_diagonal_m=diagonal,
                no_scratch=goal.no_scratch,
            )

        free_roll: dict[str, float | None] = {
            cid: loss(free_roll_resting_state(goal.initial, actions[cid], goal.physics))
            for cid in order
        }
        oracle: dict[str, float | None] = {o.candidate_id: o.loss_radii for o in outcomes}
        for method, losses in (("free_roll_selector", free_roll), ("oracle", oracle)):
            results["reference", "structured", method][case_id] = _select(
                outcomes, losses, _pick(losses, order), group
            )
        for model in MODELS:
            for view in ("structured", "bev"):
                if (case_id, "selection", None, view) not in indexed:
                    continue
                forecasts = [
                    by_request.get((model, indexed[case_id, "forecast", cid, view].request_id))
                    for cid in order
                ]
                choice = by_request.get(
                    (model, indexed[case_id, "selection", None, view].request_id)
                )
                if choice is None or any(row is None for row in forecasts):
                    continue
                predicted: dict[str, float | None] = {
                    cid: loss(
                        ForwardSubmission.model_validate_json(row.reply.content).resting_state
                    )
                    if row is not None and row.valid_output and row.reply is not None
                    else None
                    for cid, row in zip(order, forecasts, strict=True)
                }
                direct = (
                    BankSelectionSubmission.model_validate_json(choice.reply.content).candidate_id
                    if choice.valid_output and choice.reply is not None
                    else None
                )
                results[model, view, "forecast_selector"][case_id] = _select(
                    outcomes, predicted, _pick(predicted, order), group
                )
                results[model, view, "direct_selection"][case_id] = _select(
                    outcomes, dict.fromkeys(order), direct, group
                )
    comparisons = []
    for model in MODELS:
        for view in ("structured", "bev"):
            arms = {
                method: {case_id: float(r.success) for case_id, r in results[key].items()}
                for method in SELECTION_METHODS
                if (key := (model, view, method)) in results
            }
            if not arms:
                continue
            arms["random_expected"] = random_success
            arms["free_roll_selector"] = {
                case_id: float(r.success)
                for case_id, r in results["reference", "structured", "free_roll_selector"].items()
            }
            for left, right in (
                ("forecast_selector", "direct_selection"),
                ("forecast_selector", "random_expected"),
                ("direct_selection", "random_expected"),
                ("forecast_selector", "free_roll_selector"),
            ):
                common = sorted(arms[left].keys() & arms[right].keys())
                comparisons.append(
                    {
                        "model": model,
                        "view": view,
                        "method_a": left,
                        "method_b": right,
                        "success_difference": summarize_paired(
                            tuple(
                                PairedMeasurement(
                                    pair_id=case_id,
                                    group_id=private[case_id]["source_group"],
                                    a=arms[left][case_id],
                                    b=arms[right][case_id],
                                )
                                for case_id in common
                            ),
                            seed=20261019,
                        ).model_dump(mode="json"),
                    }
                )
    forecast_summary = []
    for model in MODELS:
        for view in ("structured", "bev"):
            group_rows = [r for r in rows if (r.model, r.view, r.kind) == (model, view, "forecast")]
            if not group_rows:
                continue
            scores = [r.forecast_score for r in group_rows if r.forecast_score is not None]
            radius = GoalInput.model_validate(
                private[group_rows[0].case_id]["goal"]
            ).physics.table.ball_radius_m
            resting = [
                s.resting_max_position_error_m / radius
                for s in scores
                if s.resting_max_position_error_m is not None
            ]
            forecast_summary.append(
                {
                    "model": model,
                    "view": view,
                    "attempted_forecasts": len(group_rows),
                    "valid_forecasts": sum(r.valid_output for r in group_rows),
                    "scored_forecasts": len(scores),
                    "median_resting_error_radii_scored_only": float(np.median(resting))
                    if resting
                    else None,
                    "resting_success_by_tolerance_radii": {
                        str(k): sum(
                            r.forecast_tolerances["resting"][str(k)]
                            for r in group_rows
                            if r.forecast_tolerances
                        )
                        for k in TOLERANCE_RADII
                    },
                    "contact_f1_scored_only": float(np.mean([s.contacts.f1 for s in scores]))
                    if scores
                    else None,
                    "usage": usage_summary([r.reply for r in group_rows if r.reply is not None]),
                }
            )
    return {
        "forecast_accuracy": forecast_summary,
        "selection": [
            {"model": model, "view": view, "method": method, **_method_summary(by_case)}
            for (model, view, method), by_case in sorted(results.items())
            if by_case
        ],
        "random_expected_success_rate": float(np.mean(list(random_success.values()))),
        "paired_selection_comparisons": comparisons,
    }


def run_bank_study(
    panel: Path,
    source: Path,
    output: Path,
    config: BankStudyConfig,
    *,
    endpoint_index: int = 4,
    execute: bool = False,
    token_provider: Callable[[], str] | None = None,
    transport: httpx.BaseTransport | None = None,
    limit_jobs: int | None = None,
) -> dict[str, object]:
    validation = validate_bank_panel(panel)
    requests = _bank_requests(panel)
    planned = len(requests) * len(MODELS)
    if config.max_model_requests < planned:
        raise ValueError("Request allocation must cover every bank request for both models")
    backends = study_azure_backends(
        source,
        endpoint_index,
        reasoning_effort=config.reasoning_effort,
        max_output_tokens=config.max_output_tokens,
        request_token_reservation=config.request_token_reservation,
        timeout_s=config.timeout_s,
    )
    limits = BudgetLimits(
        max_model_requests=config.max_model_requests,
        max_total_tokens=config.max_total_tokens,
        max_physics_evaluations=0,
    )
    protocol: dict[str, object] = {
        "bank_panel_manifest_sha256": validation["manifest_sha256"],
        "azure_inventory_sha256": file_sha256(source),
        "endpoint_hash": fingerprint(backends[0].base_url),
        "config": config.model_dump(mode="json"),
        "backends": [backend.model_dump(mode="json") for backend in backends],
        "request_ids": [request.request_id for request in requests],
        "limits": limits.model_dump(mode="json"),
        "producer": producer_hashes(
            "bank_study.py",
            "study_calls.py",
            "model_adapters.py",
            "evaluation.py",
            "diagnostics.py",
            "metrics.py",
        ),
        "real_data_sent": False,
        "agent_tool_evaluations": 0,
        "formal_evaluation": False,
    }
    run_hash = lock_run(output, protocol, reserve_bytes=200_000_000)
    if not execute:
        return {"status": "planned_no_calls", "run_hash": run_hash, "planned_requests": planned}
    if token_provider is None:
        raise ValueError("Azure execution requires a runtime bearer-token provider")
    token_provider()
    rows: list[BankRow] = []
    labels: dict[str, dict[str, object]] = {}
    ledger = TrialLedger(output / "ledger.sqlite", limits)
    before = ledger.counts()["model_requests"]
    stop_error = None
    completed = 0
    try:
        with ExitStack() as stack:
            adapters = {
                backend.alias: stack.enter_context(
                    StudyAdapter(
                        backend.model_copy(update={"execution_authorized": True}),
                        token_provider=token_provider,
                        transport=transport,
                    )
                )
                for backend in backends
            }
            for request in requests:
                for model in MODELS:
                    path = output / "jobs" / model / f"{request.request_id}.json"
                    cached = load_job(path, run_hash)
                    if cached is not None:
                        row = BankRow.model_validate(cached)
                        if (row.model, row.request_id) != (model, request.request_id):
                            raise ValueError("Cached bank job identity changed")
                        rows.append(row)
                        continue
                    if stop_error or (limit_jobs is not None and completed >= limit_jobs):
                        continue
                    if not credential_ready(token_provider):
                        stop_error = "credential_unavailable"
                        continue
                    started = monotonic()
                    call = load_bank_call(panel, request)
                    outcome = guarded_request(
                        partial(request_model, adapters[model], call, ledger)
                        if isinstance(call, ModelCall)
                        else partial(request_study_model, adapters[model], call, ledger)
                    )
                    stop_error = outcome.stop_error
                    scored: dict[str, object] = {}
                    if (
                        isinstance(call, ModelCall)
                        and outcome.reply is not None
                        and outcome.error_code is None
                        and request.candidate_id is not None
                    ):
                        if request.case_id not in labels:
                            labels[request.case_id] = json.loads(
                                (panel / "private" / f"{request.case_id}.json").read_text()
                            )
                        truths = labels[request.case_id]["forecast_truth"]
                        if not isinstance(truths, dict):
                            raise ValueError("Bank forecast truth must be keyed by candidate")
                        entry = truths[request.candidate_id]
                        if entry is not None:
                            prediction = ForwardSubmission.model_validate_json(
                                outcome.reply.content
                            )
                            truth = ForwardSubmission.model_validate(entry["truth"])
                            radius = call.input.physics.table.ball_radius_m
                            score = score_forward(
                                prediction,
                                truth,
                                initial=call.input.initial,
                                moving_ball_ids=tuple(entry["moving_ball_ids"]),
                                position_tolerance_m=radius,
                                event_time_tolerance_s=0.02,
                            )
                            scored = {
                                "forecast_score": score,
                                "forecast_tolerances": forward_tolerance_profile(
                                    prediction, truth, score, radius
                                ),
                            }
                    row = BankRow.model_validate(
                        {
                            "model": model,
                            "request_id": request.request_id,
                            "case_id": request.case_id,
                            "kind": request.kind,
                            "view": request.view,
                            "candidate_id": request.candidate_id,
                            "valid_output": outcome.error_code is None,
                            "elapsed_s": monotonic() - started,
                            "reply": outcome.reply,
                            "error_code": outcome.error_code,
                            **scored,
                        }
                    )
                    save_job(path, run_hash, row.model_dump(mode="json"))
                    rows.append(row)
                    completed += 1
                    LOGGER.info("Bank job %d/%d: %s/%s", len(rows), planned, model, request.kind)
        report: dict[str, object] = {
            "status": "stopped_service_error"
            if stop_error
            else ("completed" if len(rows) == planned else "partial"),
            "run_hash": run_hash,
            "stop_error": stop_error,
            "planned_requests": planned,
            "completed_jobs": len(rows),
            "new_model_requests": ledger.counts()["model_requests"] - before,
            "resource_counts": ledger.counts(),
            "usage": ledger.usage_totals(),
            **summarize_bank(panel, rows),
            "real_data_sent": False,
            "formal_evaluation": False,
        }
        write_json(output / "summary.json", report)
        return report
    finally:
        ledger.close()
