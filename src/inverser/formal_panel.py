import base64
import json
import logging
from collections import Counter
from importlib.metadata import version
from math import dist
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator

from inverser.bank_study import _asset, draw_bank, trivial_forward_baselines
from inverser.diagnostic_panel import FAMILIES
from inverser.metrics import score_endpoint
from inverser.model_adapters import ModelCall, rendered_asset
from inverser.physics import Replay, SimulationFailure, SnookerSimulator
from inverser.pilot import MAX_CANDIDATES, sample_case
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.schemas import CueAction, PhysicsConfig, Record, ShotRecord
from inverser.study import (
    ForwardInput,
    GoalInput,
    make_study_case,
    replay_forward_truth,
    translating_ball_ids,
)
from inverser.study_calls import (
    BankRankingInput,
    ForwardQuery,
    PostImpactKinematics,
    StudyCall,
    create_artifact_dir,
    producer_hashes,
)

LOGGER = logging.getLogger(__name__)
Kind = Literal["forward", "goal", "bank_forecast", "bank_ranking"]
# Single-parameter interventions per family, tried in order. The first that moves some ball's
# resting position by the declared effect is used; selection sees forward truth only, no goal.
INTERVENTION_LADDERS: dict[str, tuple[tuple[str, str, float], ...]] = {
    "free_roll": (("cue_speed_mps", "scale", 1.5), ("cue_speed_mps", "scale", 0.6)),
    "center": tuple(("azimuth_deg", "add", step) for step in (3.0, -3.0, 6.0, -6.0, 10.0, -10.0)),
    "vertical": (("tip_vertical", "add", 0.25), ("tip_vertical", "add", -0.25)),
    "side": (("tip_side", "add", 0.25), ("tip_side", "add", -0.25)),
}


class FormalTestConfig(Record):
    schema_version: Literal["formal-test-panel-v1"] = "formal-test-panel-v1"
    seed: int = Field(default=20261021, ge=0)
    per_family: int = Field(default=50, ge=1, le=100)
    bev_per_family: int = Field(default=12, ge=0, le=100)
    feedback_per_family: int = Field(default=16, ge=0, le=100)
    candidates: int = Field(default=8, ge=2, le=32)
    sample_times_s: tuple[float, float, float] = (0.25, 0.5, 1.0)
    min_intervention_effect_radii: float = Field(default=2.0, gt=0)
    max_bank_draws: int = Field(default=5, ge=1, le=10)
    max_index: int = Field(default=5000, ge=1, le=20000)

    @model_validator(mode="after")
    def validate_design(self) -> Self:
        if max(self.bev_per_family, self.feedback_per_family) > self.per_family:
            raise ValueError("Image and feedback subsets must be nested in each family")
        if not 0 <= self.sample_times_s[0] < self.sample_times_s[1] < self.sample_times_s[2]:
            raise ValueError("Sampling times must be nonnegative and strictly increasing")
        return self


class FormalRequest(Record):
    request_id: str
    case_id: str
    kind: Kind
    view: Literal["structured", "bev"]
    condition: Literal["base_action", "intervened_action"] | None = None
    information: Literal["native", "oracle"] | None = None
    candidate_id: str | None = None
    input_file: str
    image_files: tuple[str, ...] = ()
    prompt_sha256: str

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if any(
            not path.startswith("public/") or ".." in Path(path).parts
            for path in (self.input_file, *self.image_files)
        ):
            raise ValueError("Formal requests may reference only contained public artifacts")
        if (self.kind == "forward") != (
            self.condition is not None and self.information is not None
        ):
            raise ValueError("Exactly the forward requests carry a condition and information arm")
        if (self.kind == "bank_forecast") != (self.candidate_id is not None):
            raise ValueError("Exactly the bank forecasts name one candidate")
        if self.information == "oracle" and self.view != "structured":
            raise ValueError("Oracle kinematics are given only with the structured view")
        return self


def generate_test_records(
    config: FormalTestConfig, excluded_groups: set[str]
) -> tuple[tuple[ShotRecord, ...], dict[str, int]]:
    """Fresh layouts from the pilot sampler under a new seed, balanced per family."""
    physics = PhysicsConfig()
    simulator = SnookerSimulator(physics)
    chosen: dict[str, list[ShotRecord]] = {family: [] for family in FAMILIES}
    groups: set[str] = set()
    stats: Counter[str] = Counter()
    for index in range(config.max_index):
        if all(len(chosen[family]) >= config.per_family for family in FAMILIES):
            break
        if len(chosen[sample_case(config.seed, index, 0, physics)[2]]) >= config.per_family:
            continue
        for candidate in range(MAX_CANDIDATES):
            initial, action, family = sample_case(config.seed, index, candidate, physics)
            try:
                replay = simulator.replay(initial, action)
            except SimulationFailure:
                stats["rejected_simulation_failure"] += 1
                continue
            if any(ball.kind == "cue" and ball.status == "pocketed" for ball in replay.goal.balls):
                stats["rejected_scratch"] += 1
                continue
            group = fingerprint(
                {
                    "initial": initial.endpoint_view().model_dump(mode="json"),
                    "physics": physics.model_dump(mode="json"),
                }
            )
            if group in excluded_groups or group in groups:
                stats["skipped_overlap"] += 1
                break
            groups.add(group)
            chosen[family].append(
                ShotRecord(
                    task_id=f"formal-test-{config.seed}-{index:06d}",
                    group_id=group,
                    family=family,
                    physics=physics,
                    initial=initial,
                    goal=replay.goal,
                    witness_action=action,
                    duration_s=replay.duration_s,
                    events=replay.events,
                    event_sampled_motion=replay.motion,
                    cue_initial_velocity_mps=replay.cue_velocity,
                    cue_initial_angular_velocity_radps=replay.cue_omega,
                )
            )
            break
        else:
            stats["indices_without_acceptable_candidate"] += 1
    if any(len(chosen[family]) < config.per_family for family in FAMILIES):
        raise ValueError("Too few fresh layouts within the declared index range")
    stats["generation_replays"] = simulator.rollouts
    return tuple(record for family in FAMILIES for record in chosen[family]), dict(stats)


def _effect_radii(base: Replay, other: Replay, radius_m: float) -> float | None:
    """Largest resting displacement between two outcomes; None means a ball changed status."""
    reference = {ball.ball_id: ball for ball in base.goal.balls}
    worst = 0.0
    for ball in other.goal.balls:
        if ball.status != reference[ball.ball_id].status:
            return None
        if ball.xy_m is not None and (xy := reference[ball.ball_id].xy_m) is not None:
            worst = max(worst, dist(ball.xy_m, xy) / radius_m)
    return worst


def strong_intervention(
    record: ShotRecord, base: Replay, simulator: SnookerSimulator, threshold_radii: float
) -> tuple[CueAction, Replay, dict[str, object]]:
    radius = record.physics.ball_radius_m
    best: tuple[CueAction, Replay, float, dict[str, object]] | None = None
    for name, operation, amount in INTERVENTION_LADDERS[record.family]:
        values = record.witness_action.model_dump(mode="json")
        original = values[name]
        changed = original * amount if operation == "scale" else original + amount
        if name == "azimuth_deg":
            changed %= 360
        elif name == "cue_speed_mps":
            changed = min(max(changed, 0.05), 3.0)
        else:
            changed = min(max(changed, -0.5), 0.5)
        if changed == original:
            continue
        try:
            action = CueAction.model_validate({**values, name: changed})
            replay = simulator.replay(record.initial, action)
        except (ValueError, SimulationFailure):
            continue
        effect = _effect_radii(base, replay, radius)
        info: dict[str, object] = {
            "parameter": name,
            "operation": operation,
            "amount": amount,
            "effect_radii": effect,
            "status_change": effect is None,
        }
        if effect is None or effect >= threshold_radii:
            return action, replay, {**info, "weak": False}
        if best is None or effect > best[2]:
            best = action, replay, effect, info
    if best is None:
        raise ValueError(f"No valid intervention on the declared ladder for {record.task_id}")
    return best[0], best[1], {**best[3], "weak": True}


def load_formal_call(
    panel: Path, request: FormalRequest, *, check: bool = True
) -> ModelCall | StudyCall:
    data = json.loads((panel / request.input_file).read_text())
    images = tuple(_asset(panel, request.case_id, name) for name in request.image_files)
    call: ModelCall | StudyCall
    if request.kind == "goal":
        call = ModelCall(view="structured", input=GoalInput.model_validate(data))
    elif request.kind == "bank_ranking":
        call = StudyCall(
            view=request.view, input=BankRankingInput.model_validate(data), images=images
        )
    else:
        call = StudyCall(view=request.view, input=ForwardQuery.model_validate(data), images=images)
    if check and fingerprint(call.prompt()) != request.prompt_sha256:
        raise ValueError("Loaded formal call differs from the frozen visible prompt")
    return call


def formal_requests(panel: Path) -> tuple[FormalRequest, ...]:
    return tuple(
        FormalRequest.model_validate_json(line) for line in (panel / "request_index.jsonl").open()
    )


def validate_formal_test(output: Path) -> dict[str, object]:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, checksum in manifest["artifact_hashes"].items():
        path = output / name
        if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
            raise ValueError("Frozen formal test contains an out-of-scope artifact")
        if file_sha256(path) != checksum:
            raise ValueError(f"Frozen formal test artifact changed: {name}")
    requests = formal_requests(output)
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("Formal request IDs must be unique")
    for request in requests:
        task = json.loads(load_formal_call(output, request).prompt())["task"]
        if request.kind in ("forward", "bank_forecast") and "goal" in task:
            raise ValueError("A blind forecast contains the goal")
    if any("witness" in path.read_text() for path in (output / "public").rglob("*.json")):
        raise ValueError("A model-visible payload names a goal witness")
    config = FormalTestConfig.model_validate(manifest["config"])
    layouts = 4 * config.per_family
    images = 4 * config.bev_per_family
    expected = layouts * (4 + 1 + config.candidates + 1) + images * (2 + config.candidates + 1)
    if len(requests) != expected:
        raise ValueError("Formal test does not have the declared request shape")
    return {
        "status": "frozen_formal_test_validated",
        "manifest_sha256": file_sha256(output / "manifest.json"),
        "layouts": layouts,
        "bev_layouts": images,
        "feedback_layouts": 4 * config.feedback_per_family,
        "request_records": len(requests),
        "request_kinds": dict(Counter(f"{r.kind}/{r.view}" for r in requests)),
        "actual_model_calls": 0,
        "formal_test_freeze": True,
    }


def prepare_formal_test(
    dev_source: Path, output: Path, config: FormalTestConfig
) -> dict[str, object]:
    protocol = {
        "config": config.model_dump(mode="json"),
        "dev_source_sha256": file_sha256(dev_source),
        "generator": "inverser.pilot.sample_case with a fresh seed; scratches/failures rejected",
        "intervention_ladders": {
            family: [list(step) for step in ladder]
            for family, ladder in INTERVENTION_LADDERS.items()
        },
        "producer_hashes": producer_hashes(
            "formal_panel.py",
            "bank_study.py",
            "study_calls.py",
            "pilot.py",
            "physics.py",
            "study.py",
            "metrics.py",
            "model_adapters.py",
            "preview.py",
            "baselines.py",
            "diagnostics.py",
        ),
        "runtime_versions": {
            name: version(name)
            for name in ("numpy", "scipy", "pydantic", "pooltool-billiards", "Pillow")
        },
        "partition": "formal_test",
        "formal_test_freeze": True,
        "never_used_for_development": True,
        "actual_model_inference": False,
    }
    protocol_hash = fingerprint(protocol)
    if (output / "manifest.json").exists():
        if json.loads((output / "manifest.json").read_text())["protocol_hash"] != protocol_hash:
            raise ValueError("Changed sources/code/protocol require a new formal-test version")
        return validate_formal_test(output)
    excluded = {json.loads(line)["group_id"] for line in dev_source.open()}
    records, generation = generate_test_records(config, excluded)
    create_artifact_dir(output, 2_000_000_000)
    (output / "public").mkdir()
    (output / "private").mkdir()
    simulator = SnookerSimulator(PhysicsConfig())
    requests: list[FormalRequest] = []
    selection = []
    rank_in_family: Counter[str] = Counter()

    def add(draft: FormalRequest) -> None:
        prompt = load_formal_call(output, draft, check=False).prompt()
        requests.append(draft.model_copy(update={"prompt_sha256": fingerprint(prompt)}))

    def image(path: Path, data: ForwardInput | GoalInput, role: Literal["initial", "goal"]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            base64.b64decode(rendered_asset(data, role, data.physics.table).png_base64)
        )
        return str(path.relative_to(output))

    for record in records:
        position = rank_in_family[record.family]
        rank_in_family[record.family] += 1
        bev = position < config.bev_per_family
        feedback = position < config.feedback_per_family
        case = make_study_case(record, simulator)
        case_id = case.case_id
        base = simulator.replay(record.initial, record.witness_action)
        if not score_endpoint(base.goal, record.goal, 1e-8).success or base.events != record.events:
            raise ValueError("A generated witness no longer reproduces its endpoint and events")
        altered_action, altered, intervention = strong_intervention(
            record, base, simulator, config.min_intervention_effect_radii
        )
        case_dir = output / "public" / case_id
        case_dir.mkdir()
        goal = case.goal_input()
        initial_png = image(case_dir / "initial.png", goal, "initial")
        labels: dict[str, object] = {}
        for condition, action, replay in (
            ("base_action", record.witness_action, base),
            ("intervened_action", altered_action, altered),
        ):
            forward = case.forward_input(action, config.sample_times_s)
            truth = replay_forward_truth(
                record.initial, replay, record.physics, forward.sample_times_s
            )
            moving = translating_ball_ids(record.initial, replay.motion)
            kinematics = PostImpactKinematics(
                velocity_mps=replay.cue_velocity, angular_velocity_radps=replay.cue_omega
            )
            labels[condition] = {
                "action": action.model_dump(mode="json"),
                "truth": truth.model_dump(mode="json"),
                "moving_ball_ids": moving,
                "baselines": trivial_forward_baselines(forward, truth, moving),
            }
            for information, query in (
                ("native", ForwardQuery(forward=forward)),
                ("oracle", ForwardQuery(forward=forward, cue_ball_post_impact=kinematics)),
            ):
                path = case_dir / f"forward_{condition}_{information}.json"
                write_json(path, query.model_dump(mode="json"))
                views: tuple[Literal["structured", "bev"], ...] = (
                    ("structured", "bev") if bev and information == "native" else ("structured",)
                )
                for view in views:
                    add(
                        FormalRequest.model_validate(
                            {
                                "request_id": fingerprint((case_id, condition, information, view))[
                                    :24
                                ],
                                "case_id": case_id,
                                "kind": "forward",
                                "view": view,
                                "condition": condition,
                                "information": information,
                                "input_file": str(path.relative_to(output)),
                                "image_files": (initial_png,) if view == "bev" else (),
                                "prompt_sha256": "",
                            }
                        )
                    )
        write_json(case_dir / "goal.json", goal.model_dump(mode="json"))
        add(
            FormalRequest(
                request_id=fingerprint((case_id, "goal", "structured"))[:24],
                case_id=case_id,
                kind="goal",
                view="structured",
                input_file=str((case_dir / "goal.json").relative_to(output)),
                prompt_sha256="",
            )
        )
        initial = record.initial.endpoint_view()
        drawn = draw_bank(
            case_id,
            initial,
            simulator,
            case.public_physics,
            record.physics,
            seed=config.seed,
            candidates=config.candidates,
            max_draws=config.max_bank_draws,
            sample_times_s=config.sample_times_s,
        )
        bank_views: tuple[Literal["structured", "bev"], ...] = (
            ("structured", "bev") if bev else ("structured",)
        )
        bank_goal_png = image(case_dir / "bank" / "goal.png", drawn.goal, "goal") if bev else ""
        for candidate in drawn.bank.candidates:
            path = case_dir / "bank" / f"forecast_{candidate.candidate_id}.json"
            path.parent.mkdir(exist_ok=True)
            query = ForwardQuery(
                forward=ForwardInput(
                    case_id=case_id,
                    initial=initial,
                    physics=case.public_physics,
                    action=candidate.action,
                    sample_times_s=config.sample_times_s,
                )
            )
            write_json(path, query.model_dump(mode="json"))
            for view in bank_views:
                add(
                    FormalRequest(
                        request_id=fingerprint((case_id, "bank", candidate.candidate_id, view))[
                            :24
                        ],
                        case_id=case_id,
                        kind="bank_forecast",
                        view=view,
                        candidate_id=candidate.candidate_id,
                        input_file=str(path.relative_to(output)),
                        image_files=(initial_png,) if view == "bev" else (),
                        prompt_sha256="",
                    )
                )
        path = case_dir / "bank" / "ranking.json"
        write_json(
            path,
            BankRankingInput(goal=drawn.goal, candidates=drawn.bank.candidates).model_dump(
                mode="json"
            ),
        )
        for view in bank_views:
            add(
                FormalRequest(
                    request_id=fingerprint((case_id, "ranking", view))[:24],
                    case_id=case_id,
                    kind="bank_ranking",
                    view=view,
                    input_file=str(path.relative_to(output)),
                    image_files=(initial_png, bank_goal_png) if view == "bev" else (),
                    prompt_sha256="",
                )
            )
        write_json(
            output / "private" / f"{case_id}.json",
            {
                "source_task_id": record.task_id,
                "source_group": record.group_id,
                "family": record.family,
                "record": record.model_dump(mode="json"),
                "intervention": intervention,
                "forward_labels": labels,
                "bank": {
                    "bank": drawn.bank.model_dump(mode="json"),
                    "bank_draw": drawn.draw,
                    "witness_candidate_id": drawn.witness,
                    "goal": drawn.goal.model_dump(mode="json"),
                    "outcomes": [outcome.model_dump(mode="json") for outcome in drawn.outcomes],
                    "forecast_truth": drawn.truths,
                },
            },
        )
        selection.append(
            {
                "case_id": case_id,
                "source_group": record.group_id,
                "family": record.family,
                "bev": bev,
                "feedback": feedback,
                "weak_intervention": intervention["weak"],
                "bank_feasible_candidates": sum(outcome.goal_success for outcome in drawn.outcomes),
            }
        )
    with (output / "request_index.jsonl").open("w", encoding="utf-8") as stream:
        for request in requests:
            stream.write(request.model_dump_json() + "\n")
    write_json(output / "private/selection.json", {"formal_test": selection})
    effects: dict[str, list[float]] = {family: [] for family in FAMILIES}
    for path in (output / "private").glob("case-*.json"):
        labels_file = json.loads(path.read_text())
        value = labels_file["intervention"]["effect_radii"]
        if value is not None:
            effects[labels_file["family"]].append(value)
    write_json(
        output / "manifest.json",
        {
            **protocol,
            "protocol_hash": protocol_hash,
            "generation": generation,
            "dev_overlapping_groups": 0,
            "family_counts": dict(Counter(item["family"] for item in selection)),
            "weak_interventions": dict(
                Counter(item["family"] for item in selection if item["weak_intervention"])
            ),
            "median_intervention_effect_radii": {
                family: float(np.median(values)) if values else None
                for family, values in effects.items()
            },
            "bank_feasible_candidate_counts": dict(
                Counter(str(item["bank_feasible_candidates"]) for item in selection)
            ),
            "physics_replays": simulator.rollouts,
            "resolved_physics": [simulator.metadata()],
            "artifact_hashes": {
                str(path.relative_to(output)): file_sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    report = validate_formal_test(output)
    write_json(output / "validation.json", report)
    LOGGER.info("Frozen formal test: %d layouts, %d requests", len(records), len(requests))
    return report


def formal_selection(panel: Path) -> list[dict[str, object]]:
    value: list[dict[str, object]] = json.loads((panel / "private/selection.json").read_text())[
        "formal_test"
    ]
    return value
