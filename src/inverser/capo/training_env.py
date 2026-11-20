"""Fresh training layouts and a single-turn goal environment for post-training.

The prompt, output contract and hidden scoring are those of the frozen formal-suite goal task;
only the layouts differ: a new seed, zero overlap with the development and formal-test groups.
No model is called here; rewards come from the pinned simulator.
"""

import json
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Literal

from pydantic import Field

from inverser.metrics import endpoint_loss, score_endpoint
from inverser.model_adapters import VLM_SYSTEM_PROMPT, ModelCall
from inverser.physics import SimulationFailure, SnookerSimulator
from inverser.pilot import FAMILIES, MAX_CANDIDATES, sample_case
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.schemas import CueAction, PhysicsConfig, Record, ShotRecord
from inverser.study import GoalInput, GoalSubmission, make_study_case, public_physics
from inverser.study_calls import create_artifact_dir, producer_hashes

# The system message every adapter sends; test_training_env checks it against a real request.
SYSTEM_PROMPT = VLM_SYSTEM_PROMPT
THINK_BLOCK = re.compile(r"\A\s*<think>.*?</think>", re.DOTALL)


class TrainingSetConfig(Record):
    schema_version: Literal["training-set-v1"] = "training-set-v1"
    seed: int = Field(default=20261103, ge=0)
    per_family: int = Field(default=1000, ge=1, le=20000)
    max_index: int = Field(default=200_000, ge=1)


class TrainingGoal(Record):
    """What the environment needs per layout; the full record stays in privileged.jsonl."""

    task_id: str
    family: Literal["free_roll", "center", "vertical", "side"]
    input: GoalInput
    witness_action: CueAction


def training_records(
    config: TrainingSetConfig, excluded_groups: set[str], stats: Counter[str] | None = None
) -> Iterator[ShotRecord]:
    """Fresh layouts from the pilot sampler, balanced per family, yielded lazily.

    Mirrors formal_panel.generate_test_records (same sampler, rejections and group fingerprint)
    but is kept apart so the frozen formal test's producer files stay untouched.
    """
    physics = PhysicsConfig()
    simulator = SnookerSimulator(physics)
    tally = Counter[str]() if stats is None else stats
    counts: Counter[str] = Counter()
    groups: set[str] = set()
    for index in range(config.max_index):
        if all(counts[family] >= config.per_family for family in FAMILIES):
            tally["generation_replays"] = simulator.rollouts
            return
        if counts[sample_case(config.seed, index, 0, physics)[2]] >= config.per_family:
            continue
        for candidate in range(MAX_CANDIDATES):
            initial, action, family = sample_case(config.seed, index, candidate, physics)
            try:
                replay = simulator.replay(initial, action)
            except SimulationFailure:
                tally["rejected_simulation_failure"] += 1
                continue
            if any(ball.kind == "cue" and ball.status == "pocketed" for ball in replay.goal.balls):
                tally["rejected_scratch"] += 1
                continue
            group = fingerprint(
                {
                    "initial": initial.endpoint_view().model_dump(mode="json"),
                    "physics": physics.model_dump(mode="json"),
                }
            )
            if group in excluded_groups or group in groups:
                tally["skipped_overlap"] += 1
                break
            groups.add(group)
            counts[family] += 1
            yield ShotRecord(
                task_id=f"train-{config.seed}-{index:06d}",
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
            break
        else:
            tally["indices_without_acceptable_candidate"] += 1
    raise ValueError("Too few fresh layouts within the declared index range")


def training_goals(
    records: Sequence[ShotRecord], simulator: SnookerSimulator
) -> tuple[TrainingGoal, ...]:
    return tuple(
        TrainingGoal(
            task_id=record.task_id,
            family=record.family,
            input=make_study_case(record, simulator).goal_input(),
            witness_action=record.witness_action,
        )
        for record in records
    )


def goal_reward(loss_radii: float | None, success: bool, *, scale_radii: float = 20.0) -> float:
    """0 for an unusable submission; up to 1 as the resting error (ball radii, with the status,
    pocket and scratch penalties of endpoint_loss) shrinks within ``scale_radii``; plus 1 on
    success at the task tolerance. GRPO uses group-relative advantages, so only the monotone
    shape matters; ``scale_radii`` is the one pre-registered knob (see training_protocol.md)."""
    if loss_radii is None:
        return 0.0
    return float(success) + max(0.0, 1.0 - loss_radii / scale_radii)


def parse_submission(text: str) -> CueAction | None:
    """The formal goal contract: the reply is the GoalSubmission JSON and nothing else, except
    that a leading reasoning block, which thinking models emit before the answer, is dropped."""
    try:
        return GoalSubmission.model_validate_json(THINK_BLOCK.sub("", text, count=1).strip()).action
    except ValueError:
        return None


@dataclass(frozen=True)
class StepResult:
    action: CueAction | None
    success: bool
    loss_radii: float | None
    reward: float
    failure: str | None


class GoalEnvironment:
    """Single turn: one goal prompt per layout, one submission, one hidden replay, one reward."""

    def __init__(self, goals: Sequence[TrainingGoal], physics: PhysicsConfig | None = None) -> None:
        if not goals:
            raise ValueError("An environment needs at least one goal")
        self.goals = tuple(goals)
        self.simulator = SnookerSimulator(physics or PhysicsConfig())
        expected = public_physics(self.simulator)
        if any(goal.input.physics != expected for goal in self.goals):
            raise ValueError("Goal physics must match the environment simulator")

    def __len__(self) -> int:
        return len(self.goals)

    def goal_input(self, index: int) -> GoalInput:
        return self.goals[index].input

    def messages(self, index: int) -> list[dict[str, str]]:
        """The chat messages the study adapter sends for a structured goal request."""
        prompt = ModelCall(view="structured", input=self.goal_input(index)).prompt()
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

    def rows(self) -> list[dict[str, object]]:
        """Prompt rows for a conversational TRL dataset; ``case_index`` routes rewards back."""
        return [
            {
                "prompt": self.messages(index),
                "case_index": index,
                "case_id": goal.input.case_id,
                "family": goal.family,
            }
            for index, goal in enumerate(self.goals)
        ]

    def step(self, index: int, text: str) -> StepResult:
        action = parse_submission(text)
        if action is None:
            return StepResult(None, False, None, 0.0, "invalid_submission")
        goal = self.goal_input(index)
        try:
            replay = self.simulator.replay(goal.initial, action)
        except SimulationFailure:
            return StepResult(action, False, None, 0.0, "simulation_failure")
        table = goal.physics.table
        tolerance = goal.tolerance_radii * table.ball_radius_m
        score = score_endpoint(replay.goal, goal.goal, tolerance, no_scratch=goal.no_scratch)
        loss = endpoint_loss(
            replay.goal,
            goal.goal,
            ball_radius_m=table.ball_radius_m,
            table_diagonal_m=hypot(table.width_m, table.length_m),
            no_scratch=goal.no_scratch,
        )
        return StepResult(
            action, score.success, loss, goal_reward(loss, score.success), score.failure
        )

    def rewards(
        self, completions: Sequence[str | list[dict[str, str]]], case_index: Sequence[int]
    ) -> list[float]:
        """TRL reward-function shape: one reward per completion, conversational or plain."""
        texts = [item if isinstance(item, str) else item[-1]["content"] for item in completions]
        return [
            self.step(index, text).reward for index, text in zip(case_index, texts, strict=True)
        ]


def load_training_goals(path: Path) -> tuple[TrainingGoal, ...]:
    with (path / "goals.jsonl").open(encoding="utf-8") as stream:
        return tuple(TrainingGoal.model_validate_json(line) for line in stream)


def environment_from_training_set(path: Path) -> GoalEnvironment:
    manifest = json.loads((path / "manifest.json").read_text())
    return GoalEnvironment(
        load_training_goals(path), PhysicsConfig.model_validate(manifest["physics"])
    )


def exclusion_sources(
    records: Sequence[Path], panels: Sequence[Path]
) -> list[tuple[str, str, set[str]]]:
    """Group fingerprints never to train on: record files (development pilots, other training
    sets) and frozen formal panels, each with the hash of the file that lists them."""
    sources = []
    for path in records:
        with path.open(encoding="utf-8") as stream:
            groups = {json.loads(line)["group_id"] for line in stream}
        sources.append((str(path), file_sha256(path), groups))
    for panel in panels:
        selection = json.loads((panel / "private" / "selection.json").read_text())
        groups = {item["source_group"] for item in selection["formal_test"]}
        sources.append((str(panel), file_sha256(panel / "manifest.json"), groups))
    return sources


def prepare_training_set(
    output: Path,
    config: TrainingSetConfig,
    *,
    exclude_records: Sequence[Path] = (),
    exclude_panels: Sequence[Path] = (),
) -> dict[str, object]:
    """Freeze fresh training layouts with their goal views and a zero-overlap audit."""
    sources = exclusion_sources(exclude_records, exclude_panels)
    excluded = set[str]().union(*(groups for _, _, groups in sources))
    physics = PhysicsConfig()
    protocol = {
        "config": config.model_dump(mode="json"),
        "physics": physics.model_dump(mode="json"),
        "generator": "inverser.training_env.training_records: pilot sampler under a fresh seed; "
        "scratches, simulation failures and excluded groups rejected",
        "exclusions": [
            {"source": name, "sha256": digest, "groups": len(groups)}
            for name, digest, groups in sources
        ],
        "producer_hashes": producer_hashes(
            "training_env.py",
            "pilot.py",
            "physics.py",
            "study.py",
            "metrics.py",
            "model_adapters.py",
            "schemas.py",
        ),
        "partition": "training",
        "actual_model_inference": False,
    }
    create_artifact_dir(output, 500_000_000)
    simulator = SnookerSimulator(physics)
    tally: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    groups: set[str] = set()
    with (
        (output / "privileged.jsonl").open("w", encoding="utf-8") as records,
        (output / "goals.jsonl").open("w", encoding="utf-8") as goals,
    ):
        for record in training_records(config, excluded, tally):
            records.write(record.model_dump_json() + "\n")
            goals.write(training_goals((record,), simulator)[0].model_dump_json() + "\n")
            counts[record.family] += 1
            groups.add(record.group_id)
    if groups & excluded or len(groups) != sum(counts.values()):
        raise ValueError("Training layouts must be distinct and disjoint from every excluded group")
    loaded = load_training_goals(output)
    sample = loaded[:: max(1, len(loaded) // 20)][:20]
    environment = GoalEnvironment(sample, physics)
    for index, goal in enumerate(sample):
        witness = GoalSubmission.model_validate({"action": goal.witness_action.model_dump()})
        if not environment.step(index, witness.model_dump_json()).success:
            raise ValueError(f"The witness of {goal.task_id} no longer reaches its goal")
    write_json(
        output / "manifest.json",
        {
            **protocol,
            "protocol_hash": fingerprint(protocol),
            "generation": dict(tally),
            "family_counts": dict(counts),
            "layouts": len(loaded),
            "excluded_overlapping_groups": 0,
            "witness_replays_checked": len(sample),
            "resolved_physics": [simulator.metadata()],
            "artifact_hashes": {
                name: file_sha256(output / name) for name in ("privileged.jsonl", "goals.jsonl")
            },
        },
    )
    return {
        "status": "training_set_frozen",
        "layouts": len(loaded),
        "family_counts": dict(counts),
        "generation": dict(tally),
        "excluded_groups": len(excluded),
        "excluded_overlapping_groups": 0,
        "witness_replays_checked": len(sample),
        "manifest_sha256": file_sha256(output / "manifest.json"),
        "actual_model_calls": 0,
    }
