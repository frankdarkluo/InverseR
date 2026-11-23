"""Versioned InverseR-VLM contract, offline scoring and Pooltool-grounded corpus."""

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterator
from math import dist, hypot
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO, Literal, Self

from PIL import Image
from pydantic import Field, StrictBool, model_validator

from inverser.metrics import score_endpoint
from inverser.model_adapters import split_reasoning
from inverser.physics import Replay, SimulationFailure, SnookerSimulator
from inverser.pilot import Family, sample_case
from inverser.preview import render_vlm_forward_scene, render_vlm_goal_pair
from inverser.runtime import check_storage, file_sha256, fingerprint, write_json
from inverser.schemas import CueAction, Layout, PhysicsConfig, Record, ShotRecord
from inverser.study import (
    DYNAMIC_PARAMETERS,
    PublicPhysics,
    TableGeometry,
    public_physics,
    replay_forward_truth,
)
from inverser.study_calls import producer_hashes, require_data_path

TIMES = (1.0, 2.0, 3.0, 4.0, 5.0)
HORIZON_TAGS = ("t1s", "t2s", "t3s", "t4s", "t5s", "resting")
EVENT_TAGS = ("collisions", "first_contact", "cushions")
FORWARD_TAGS = (*EVENT_TAGS, *HORIZON_TAGS)
GOAL_TAGS = ("azimuth_deg", "cue_speed_mps", "tip_side", "tip_vertical")
_EVENT_SPEC = (
    "each object ball's participation in any ball-ball contact (use 'none' when ball_count is "
    "1), the object slot the cue ball hits first ('none' if it hits no object ball; never a "
    "cushion name), and the ordered boundaries hit by any ball from LEFT, RIGHT, BOTTOM, TOP "
    "including repeats ('none' if no cushion bounce)"
)
BOUNDARIES: set[str] = {"LEFT", "RIGHT", "BOTTOM", "TOP"}
TAG_PATTERN = re.compile(r"<([a-z0-9_]+)>(.*?)</\1>", re.DOTALL)
FAMILIES: tuple[Family, ...] = ("free_roll", "center", "vertical", "side")
Boundary = Literal["LEFT", "RIGHT", "BOTTOM", "TOP"]
Position = tuple[float, float] | Literal["pocketed"]
SUITE_DIR = "inverser-vlm-v1"
FAMILY_SLUGS: dict[Family, str] = {
    "free_roll": "roll",
    "center": "center",
    "vertical": "vert",
    "side": "side",
}
OOD_SLUGS = {"slow_friction": "slow", "fast_friction": "fast"}
WITHHELD = ("sliding_friction", "rolling_friction", "cushion_restitution")
SPLITS = ("train", "valid", "test")
SPLIT_PAIRS = (3000, 30, 75)
LOGGER = logging.getLogger(__name__)


def scene_naming(
    split: str, regime: str, family: Family, ball_count: int, ordinal: int, role: str
) -> tuple[str, str, str, str, str | None]:
    slug = FAMILY_SLUGS[family]
    if regime == "standard":
        stem = f"{slug}-{ball_count}ball-{ordinal:04d}"
        group_id = f"{split}-{stem}"
        scene_id = f"{group_id}-{role}"
        folder = f"images/{slug}"
        return (
            group_id,
            scene_id,
            f"{folder}/{stem}-init.png",
            f"{folder}/{stem}-{role}-goal.png",
            f"{folder}/{stem}-{role}-fwd.png",
        )
    stem = f"{OOD_SLUGS[regime]}-{slug}-{ball_count}ball-{ordinal:04d}"
    group_id = f"{split}-{stem}"
    return (
        group_id,
        group_id,
        f"images/ood/{stem}-init.png",
        f"images/ood/{stem}-goal.png",
        None,
    )


class VLMForwardInput(Record):
    task_ids: tuple[str, str, str] = ("T1_ball_collision", "T2_cushion_bounce", "T3_coordinates")
    scene_id: str
    image: str
    ball_count: int = Field(ge=1, le=4)
    sample_times_s: tuple[float, ...] = TIMES
    table_metres: tuple[float, float]
    ball_radius_m: float = Field(gt=0)
    parameters: dict[str, float]
    action: CueAction

    def prompt(self) -> str:
        slots_hint = " | ".join(
            f"{slot}: 0.500, 1.000" for slot in range(self.ball_count)
        )
        template = (
            "<answer>\n"
            + _event_template(self.ball_count)
            + "\n".join(f"<{tag}>{slots_hint}</{tag}>" for tag in HORIZON_TAGS)
            + "\n</answer>"
        )
        return (
            "Use numbered ball slots from the INITIAL image (0 = cue). The cyan arrow shows "
            "cue azimuth in x-right/y-up metres and encodes cue-stick speed at 0.16 seconds "
            "per ray; the inset shows horizontal and vertical tip offsets. The action fields "
            "give exact commands for calibration; read initial positions from the image. "
            f"Predict {_EVENT_SPEC}. Predict all numbered slots "
            f"0..{self.ball_count - 1} in order at 1s..5s and when stationary, using "
            "'slot: x, y' in metres or 'slot: pocketed'. Return only the <answer> block:\n"
            f"{template}\n"
            + json.dumps(
                self.model_dump(mode="json", exclude={"image", "scene_id"}),
                separators=(",", ":"),
            )
        )


class ContactForecast(Record):
    """The discrete contact events of one shot: forward T1/T2 targets and the T4/T5 forecast."""

    object_collisions: tuple[StrictBool, ...] = Field(max_length=3)
    first_contact: int | None = Field(ge=1, le=3)
    cushion_set: tuple[Boundary, ...] = Field(max_length=4)
    cushion_sequence: tuple[Boundary, ...] = Field(max_length=1000)

    @model_validator(mode="after")
    def validate_events(self) -> Self:
        if self.first_contact is not None and (
            self.first_contact > len(self.object_collisions)
            or not self.object_collisions[self.first_contact - 1]
        ):
            raise ValueError("First cue contact must identify a colliding object ball")
        if len(set(self.cushion_set)) != len(self.cushion_set) or set(self.cushion_set) != set(
            self.cushion_sequence
        ):
            raise ValueError("Cushion set and sequence disagree")
        return self


class VLMForwardOutput(ContactForecast):
    frames: tuple[tuple[Position, ...], ...] = Field(min_length=5, max_length=5)
    resting_state: tuple[Position, ...]

    @model_validator(mode="after")
    def validate_output(self) -> Self:
        size = len(self.object_collisions) + 1
        if any(len(frame) != size for frame in (*self.frames, self.resting_state)):
            raise ValueError("Coordinates must cover every numbered slot at every horizon")
        return self


class VLMGoalInput(Record):
    task_id: Literal["T4_visual_inverse", "T5_interactive_adaptation"]
    episode_id: str
    initial_image: str
    goal_image: str
    ball_count: int = Field(ge=1, le=4)
    table_metres: tuple[float, float]
    ball_radius_m: float = Field(gt=0)
    parameters: dict[str, float]
    withheld_parameters: tuple[str, ...] = ()
    max_attempts: Literal[1, 3, 6]  # T5: 6 in v1, 3 in inverser-v3

    @model_validator(mode="after")
    def validate_condition(self) -> Self:
        if (self.task_id == "T4_visual_inverse") != (self.max_attempts == 1):
            raise ValueError("T4 is single shot and T5 has a multi-attempt cap")
        hidden = set(self.withheld_parameters)
        visible = set(self.parameters)
        if hidden not in (set(), set(WITHHELD)) or visible != set(DYNAMIC_PARAMETERS) - hidden:
            raise ValueError("Invalid visible and withheld physics partition")
        if hidden and self.max_attempts == 1:
            raise ValueError("Withheld parameters are only used by T5")
        return self

    def prompt(self) -> str:
        return (
            "Propose one shot from the INITIAL image to reach the GOAL image without "
            "scratching. The coordinate system is x right, y up; azimuth 0 points along +x "
            "(0 <= azimuth_deg < 360, 0 < cue_speed_mps <= 3, -0.5 <= tip_side, tip_vertical "
            "<= 0.5 with tip_side^2 + tip_vertical^2 <= 0.25). Goal object-ball numbers are "
            "local; same-colour objects are exchangeable. For T5 each attempted shot starts "
            "again from INITIAL; use available outcome feedback to refine the next action. "
            f"Also forecast the contacts your shot will produce: {_EVENT_SPEC}. "
            "Return only the <answer> block:\n"
            "<answer>\n"
            + _event_template(self.ball_count)
            + "<azimuth_deg>62.45</azimuth_deg>\n"
            "<cue_speed_mps>1.38</cue_speed_mps>\n"
            "<tip_side>0.00</tip_side>\n"
            "<tip_vertical>0.25</tip_vertical>\n"
            "</answer>\n"
            + json.dumps(
                self.model_dump(mode="json", exclude={"initial_image", "goal_image", "episode_id"}),
                separators=(",", ":"),
            )
        )


class VLMGoalOutput(Record):
    azimuth_deg: float = Field(ge=0, lt=360)
    cue_speed_mps: float = Field(gt=0, le=3)
    tip_side: float = Field(ge=-0.5, le=0.5)
    tip_vertical: float = Field(ge=-0.5, le=0.5)

    @model_validator(mode="after")
    def validate_contact(self) -> Self:
        self.cue_action()
        return self

    def cue_action(self) -> CueAction:
        return CueAction(**self.model_dump())


def format_position_row(positions: tuple[Position, ...]) -> str:
    return " | ".join(
        f"{slot}: {point[0]:.3f}, {point[1]:.3f}"
        if isinstance(point, tuple)
        else f"{slot}: pocketed"
        for slot, point in enumerate(positions)
    )


def _event_template(ball_count: int) -> str:
    collisions = (
        "none"
        if ball_count == 1
        else " | ".join(
            f"{slot}:{'true' if slot == 1 else 'false'}" for slot in range(1, ball_count)
        )
    )
    first = "none" if ball_count == 1 else "1"
    return (
        f"<collisions>{collisions}</collisions>\n"
        f"<first_contact>{first}</first_contact>\n"
        "<cushions>LEFT, BOTTOM</cushions>\n"
    )


def _format_events(output: ContactForecast) -> list[str]:
    collisions = (
        "none"
        if not output.object_collisions
        else " | ".join(
            f"{slot}:{'true' if hit else 'false'}"
            for slot, hit in enumerate(output.object_collisions, start=1)
        )
    )
    first = "none" if output.first_contact is None else str(output.first_contact)
    cushions = "none" if not output.cushion_sequence else ", ".join(output.cushion_sequence)
    return [
        f"<collisions>{collisions}</collisions>",
        f"<first_contact>{first}</first_contact>",
        f"<cushions>{cushions}</cushions>",
    ]


def format_forward_answer(output: VLMForwardOutput) -> str:
    horizons = (*output.frames, output.resting_state)
    lines = [
        "<answer>",
        *_format_events(output),
        *(
            f"<{tag}>{format_position_row(row)}</{tag}>"
            for tag, row in zip(HORIZON_TAGS, horizons, strict=True)
        ),
        "</answer>",
    ]
    return "\n".join(lines)


def format_goal_answer(
    output: VLMGoalOutput | CueAction, forecast: ContactForecast | None = None
) -> str:
    return "\n".join(
        [
            "<answer>",
            *(_format_events(forecast) if forecast is not None else ()),
            f"<azimuth_deg>{output.azimuth_deg:.4f}</azimuth_deg>",
            f"<cue_speed_mps>{output.cue_speed_mps:.4f}</cue_speed_mps>",
            f"<tip_side>{output.tip_side:.4f}</tip_side>",
            f"<tip_vertical>{output.tip_vertical:.4f}</tip_vertical>",
            "</answer>",
        ]
    )


def _extract_answer_tags(
    text: str,
    required: tuple[str, ...],
    *,
    optional: tuple[str, ...] = (),
    lenient: bool = False,
) -> dict[str, str]:
    """Strict: the reply after reasoning is exactly one <answer> block (RL rewards). Lenient:
    text around that single block is ignored (evaluation). ``optional`` tags are all or none."""
    _, body = split_reasoning(text)
    stripped = body.strip()
    start, end = stripped.find("<answer>"), stripped.find("</answer>") + len("</answer>")
    if (
        stripped.count("<answer>") != 1
        or stripped.count("</answer>") != 1
        or end < start
        or not lenient
        and (start != 0 or end != len(stripped))
    ):
        raise ValueError("Reply must wrap the final prediction in a single <answer>...</answer>")
    inner = stripped[start + len("<answer>") : end - len("</answer>")]
    found: dict[str, str] = {}
    cursor = 0
    for match in TAG_PATTERN.finditer(inner):
        if inner[cursor : match.start()].strip():
            raise ValueError("Unexpected text between <answer> slot tags")
        name, value = match.group(1), match.group(2).strip()
        if name in found or name not in (*required, *optional):
            raise ValueError(f"Duplicate or unexpected slot tag: <{name}>")
        found[name] = value
        cursor = match.end()
    if inner[cursor:].strip() or set(found) not in (set(required), {*required, *optional}):
        raise ValueError("Missing required <answer> slot tags or trailing text")
    return found


def _parse_positions(raw: str, *, expected_balls: int | None) -> tuple[Position, ...]:
    if not raw:
        raise ValueError("Empty coordinate slot row")
    items = [part.strip() for part in raw.split("|")]
    if expected_balls is not None and len(items) != expected_balls:
        raise ValueError("Coordinate slot count disagrees with scene inventory")
    parsed: list[Position] = []
    for index, item in enumerate(items):
        slot_str, sep, value = item.partition(":")
        if not sep or not slot_str.strip().isdigit() or int(slot_str.strip()) != index:
            raise ValueError(f"Invalid numbered slot entry: {item!r}")
        rhs = value.strip()
        if rhs.lower() == "pocketed":
            parsed.append("pocketed")
            continue
        coords = [piece.strip() for piece in rhs.split(",")]
        if len(coords) != 2:
            raise ValueError(f"Coordinate slot requires two numbers or pocketed: {item!r}")
        x, y = float(coords[0]), float(coords[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("Predicted coordinates must be finite")
        parsed.append((x, y))
    return tuple(parsed)


def _tag_reply(stripped: str, lenient: bool) -> bool:
    return "<answer>" in stripped if lenient else stripped.startswith("<answer>")


def parse_forward_answer(
    text: str,
    *,
    ball_count: int | None = None,
    allow_json_fallback: bool = True,
    lenient: bool = False,
) -> VLMForwardOutput:
    _, cleaned = split_reasoning(text)
    stripped = cleaned.strip()
    if not _tag_reply(stripped, lenient):
        if not allow_json_fallback:
            raise ValueError("Forward reply must start with <answer>")
        parsed = VLMForwardOutput.model_validate_json(stripped)
        if ball_count is not None and len(parsed.resting_state) != ball_count:
            raise ValueError("Forward JSON slot count disagrees with scene inventory")
        return parsed
    tags = _extract_answer_tags(stripped, FORWARD_TAGS, lenient=lenient)
    horizons = tuple(_parse_positions(tags[tag], expected_balls=ball_count) for tag in HORIZON_TAGS)
    inferred_balls = ball_count if ball_count is not None else len(horizons[-1])
    if any(len(row) != inferred_balls for row in horizons):
        raise ValueError("All horizon rows must have matching ball counts")
    return VLMForwardOutput(
        **dict(_parse_events(tags, inferred_balls)),
        frames=horizons[:5],
        resting_state=horizons[5],
    )


def _parse_events(tags: dict[str, str], inferred_balls: int) -> ContactForecast:
    raw_collisions = tags["collisions"]
    if inferred_balls == 1:
        if raw_collisions.lower() != "none":
            raise ValueError("Single-ball scenes must report <collisions>none</collisions>")
        collisions: tuple[bool, ...] = ()
    else:
        if raw_collisions.lower() == "none":
            collisions = (False,) * (inferred_balls - 1)
        else:
            parts = [part.strip() for part in raw_collisions.split("|")]
            if len(parts) != inferred_balls - 1:
                raise ValueError("Object collision entries must match object ball slots")
            flags: list[bool] = []
            for slot, part in enumerate(parts, start=1):
                idx_str, sep, flag_str = part.partition(":")
                flag = flag_str.strip().lower()
                if not sep or not idx_str.strip().isdigit() or int(idx_str.strip()) != slot:
                    raise ValueError(f"Invalid collision slot label: {part!r}")
                if flag not in ("true", "false"):
                    raise ValueError(f"Invalid collision boolean: {part!r}")
                flags.append(flag == "true")
            collisions = tuple(flags)
    raw_first = tags["first_contact"].lower()
    if raw_first in ("none", "null"):
        first_contact = None
    elif raw_first.isdigit():
        first_contact = int(raw_first)
    else:
        raise ValueError(f"Invalid <first_contact>: {tags['first_contact']!r}")
    raw_cushions = tags["cushions"]
    if raw_cushions.lower() in ("none", ""):
        if raw_cushions == "":
            raise ValueError("Use 'none' for empty <cushions>")
        sequence: tuple[Boundary, ...] = ()
    else:
        tokens = [token.strip().upper() for token in raw_cushions.split(",")]
        if any(token not in BOUNDARIES for token in tokens):
            raise ValueError(f"Invalid boundary in <cushions>: {raw_cushions!r}")
        sequence = tuple(token for token in tokens)  # type: ignore[misc]
    return ContactForecast(
        object_collisions=collisions,
        first_contact=first_contact,
        cushion_set=tuple(dict.fromkeys(sequence)),
        cushion_sequence=sequence,
    )


def parse_goal_answer(
    text: str, *, allow_json_fallback: bool = True, lenient: bool = False
) -> VLMGoalOutput:
    """The shot of a 4-tag or 7-tag (contact forecast + shot) goal reply."""
    _, cleaned = split_reasoning(text)
    stripped = cleaned.strip()
    if not _tag_reply(stripped, lenient):
        if not allow_json_fallback:
            raise ValueError("Goal reply must start with <answer>")
        data = json.loads(stripped)
        if isinstance(data, dict) and isinstance(data.get("action"), dict) and len(data) == 1:
            data = data["action"]
        return VLMGoalOutput.model_validate(data)
    tags = _extract_answer_tags(stripped, GOAL_TAGS, optional=EVENT_TAGS, lenient=lenient)
    values: dict[str, float] = {}
    for key in GOAL_TAGS:
        number = float(tags[key])
        if not math.isfinite(number):
            raise ValueError(f"Non-finite goal parameter in <{key}>")
        values[key] = number
    return VLMGoalOutput.model_validate(values)


def parse_goal_forecast(
    text: str, ball_count: int, *, lenient: bool = False
) -> ContactForecast | None:
    """The contact forecast of a 7-tag goal reply; None for a 4-tag (shot only) reply."""
    _, cleaned = split_reasoning(text)
    tags = _extract_answer_tags(cleaned.strip(), GOAL_TAGS, optional=EVENT_TAGS, lenient=lenient)
    return _parse_events(tags, ball_count) if "collisions" in tags else None


def normalize_goal_reply(text: str) -> str:
    """Evaluation reading of a goal reply as GoalSubmission JSON; '' when no shot is usable."""
    try:
        return parse_goal_answer(text, lenient=True).model_dump_json()
    except ValueError:
        return ""


def invalid_forward_scores() -> dict[str, float | bool]:
    return {
        "T1": False,
        "T2": False,
        "T3": 0.0,
        "T3_all_slots": False,
        "S_total": 0.0,
        "first_contact_exact": False,
        "cushion_sequence_exact": False,
        "T1_per_ball": 0.0,
        "T2_cushion_set": False,
        "T3_1r": 0.0,
        "T3_2r": 0.0,
        "T3_3r": 0.0,
        "T3_5r": 0.0,
        "S_total_3r": 0.0,
    }


def forward_scores(
    prediction: VLMForwardOutput, truth: VLMForwardOutput, radius_m: float
) -> dict[str, float | bool]:
    if not radius_m > 0 or len(prediction.object_collisions) != len(truth.object_collisions):
        raise ValueError("Prediction inventory must match the truth and radius must be positive")
    first_exact = prediction.first_contact == truth.first_contact
    t1 = prediction.object_collisions == truth.object_collisions and first_exact
    t1_per_ball = (
        sum(
            left == right
            for left, right in zip(
                prediction.object_collisions, truth.object_collisions, strict=True
            )
        )
        / len(truth.object_collisions)
        if truth.object_collisions
        else 1.0
    )
    cushion_set_match = set(prediction.cushion_set) == set(truth.cushion_set)
    sequence_exact = prediction.cushion_sequence == truth.cushion_sequence
    t2 = cushion_set_match and sequence_exact
    distances = [
        dist(left, right) / radius_m
        if isinstance(left, tuple) and isinstance(right, tuple)
        else 0.0
        if left == right
        else float("inf")
        for observed, target in zip(
            (*prediction.frames, prediction.resting_state),
            (*truth.frames, truth.resting_state),
            strict=True,
        )
        for left, right in zip(observed, target, strict=True)
    ]
    by_radius = {k: sum(value <= k for value in distances) / len(distances) for k in (1, 2, 3, 5)}
    t3 = by_radius[1]
    t3_3r = by_radius[3]
    return {
        "T1": t1,
        "T2": t2,
        "T3": t3,
        "T3_all_slots": t3 == 1,
        "S_total": 0.3 * t1 + 0.3 * t2 + 0.4 * t3,
        "first_contact_exact": first_exact,
        "cushion_sequence_exact": sequence_exact,
        "T1_per_ball": t1_per_ball,
        "T2_cushion_set": cushion_set_match,
        "T3_1r": by_radius[1],
        "T3_2r": by_radius[2],
        "T3_3r": t3_3r,
        "T3_5r": by_radius[5],
        "S_total_3r": 0.3 * t1_per_ball + 0.3 * float(cushion_set_match) + 0.4 * t3_3r,
    }


def _boundary_map(table: TableGeometry) -> dict[str, Boundary]:
    def closest(x: float, y: float) -> Boundary:
        distances: dict[Boundary, float] = {
            "LEFT": abs(x),
            "RIGHT": abs(table.width_m - x),
            "BOTTOM": abs(y),
            "TOP": abs(table.length_m - y),
        }
        return min(distances, key=distances.__getitem__)

    return {
        face.geometry_id: closest(
            (face.start_m[0] + face.end_m[0]) / 2, (face.start_m[1] + face.end_m[1]) / 2
        )
        for face in table.linear_cushions
    } | {face.geometry_id: closest(*face.xy_m) for face in table.circular_cushions}


def forward_truth(
    initial: Layout, replay: Replay, physics: PhysicsConfig, table: TableGeometry
) -> VLMForwardOutput:
    labels = replay_forward_truth(initial, replay, physics, TIMES)
    slots = {ball.ball_id: index for index, ball in enumerate(initial.endpoint_view().balls)}
    boundary = _boundary_map(table)
    collided: set[int] = set()
    first = None
    sequence: list[Boundary] = []
    for event in labels.events:
        a, b = event.participants
        if event.event_type == "ball_ball":
            i, j = slots[a], slots[b]
            collided.update((i, j))
            if first is None and (i == 0 or j == 0):
                first = max(i, j)
        elif event.event_type == "cushion":
            sequence.append(boundary[b])

    def positions(layout: Layout) -> tuple[Position, ...]:
        return tuple(
            ball.xy_m if ball.xy_m is not None else "pocketed"
            for ball in sorted(layout.balls, key=lambda ball: slots[ball.ball_id])
        )

    return VLMForwardOutput(
        object_collisions=tuple(index in collided for index in range(1, len(slots))),
        first_contact=first,
        cushion_set=tuple(dict.fromkeys(sequence)),
        cushion_sequence=tuple(sequence),
        frames=tuple(positions(frame.state) for frame in labels.frames),
        resting_state=positions(labels.resting_state),
    )


def layout_group(initial: Layout) -> str:
    return fingerprint(initial.endpoint_view().model_dump(mode="json"))


def _legacy_groups(pilot: Path, formal: Path) -> tuple[set[str], dict[str, str]]:
    with pilot.open() as stream:
        groups = {layout_group(ShotRecord.model_validate_json(line).initial) for line in stream}
    selection = formal / "private/selection.json"
    entries = json.loads(selection.read_text())["formal_test"]
    sources = {"pilot": file_sha256(pilot), "formal_selection": file_sha256(selection)}
    formal_layouts = {}
    for entry in entries:
        case_id = entry["case_id"]
        row = json.loads((formal / "private" / f"{case_id}.json").read_text())
        group = layout_group(ShotRecord.model_validate(row["record"]).initial)
        formal_layouts[case_id] = group
    groups.update(formal_layouts.values())
    return groups, sources | {"formal_layouts": fingerprint(formal_layouts)}


def _counterfactuals(action: CueAction, family: str) -> Iterator[tuple[str, CueAction]]:
    edits = [("cue_speed_mps", action.cue_speed_mps * 0.55)]
    if family in ("vertical", "side"):
        key = "tip_vertical" if family == "vertical" else "tip_side"
        edits.insert(0, (key, -0.42 if getattr(action, key) > 0 else 0.42))
    edits.extend([("cue_speed_mps", 2.5), ("azimuth_deg", (action.azimuth_deg + 25) % 360)])
    for field, value in edits:
        yield field, CueAction.model_validate(action.model_dump() | {field: value})


def _terminal_shift(left: Layout, right: Layout, physics: PhysicsConfig) -> float:
    # Compare physical identities, not the exchangeable goal labels used by T4/T5.
    by_id = {ball.ball_id: ball for ball in right.balls}
    diagonal = hypot(physics.table_width_m, physics.table_length_m)
    return (
        max(
            dist(ball.xy_m, other.xy_m)
            if ball.xy_m is not None and other.xy_m is not None
            else diagonal
            if ball.status != other.status
            else 0.0
            for ball in left.balls
            for other in (by_id[ball.ball_id],)
        )
        / physics.ball_radius_m
    )


def requires_contact(initial: Layout, goal: Layout, radius_m: float) -> bool:
    """True when some object ball must move more than one radius, or be pocketed, to reach goal."""
    target = {ball.ball_id: ball for ball in goal.balls}
    return any(
        ball.kind != "cue"
        and (
            target[ball.ball_id].xy_m is None
            or dist(ball.xy_m, target[ball.ball_id].xy_m) > radius_m
        )
        for ball in initial.balls
    )


def prepare_vlm_suite(
    root: Path,
    pilot: Path,
    formal: Path,
    *,
    counts: tuple[int, int, int] = SPLIT_PAIRS,
    ood_per_family: int = 30,
) -> dict[str, object]:
    """Counts are pairs per family: train has 12k anchors; valid/test have 240/600 images."""
    require_data_path(root)
    if len(counts) != 3 or not all(0 < value <= 3000 for value in counts):
        raise ValueError("Provide three per-family counts between 1 and 3000")
    if not 0 <= ood_per_family <= 30 or ood_per_family % 2:
        raise ValueError("OOD per-family count must be even and between 0 and 30")
    suite_root = root / SUITE_DIR
    roots = [suite_root / split for split in SPLITS]
    if suite_root.exists() or suite_root.is_symlink():
        raise ValueError("Existing split roots are immutable; use a different parent")
    check_storage(2_000_000_000)
    groups, exclusions = _legacy_groups(pilot, formal)
    root.mkdir(parents=True, exist_ok=True)
    config = PhysicsConfig()
    simulators = {"standard": SnookerSimulator(config)}
    for name, scale, restitution in (("slow_friction", 0.5, 0.65), ("fast_friction", 2, 0.97)):
        simulators[name] = SnookerSimulator(
            PhysicsConfig.model_validate(
                config.model_dump()
                | {
                    "sliding_friction": config.sliding_friction * scale,
                    "rolling_friction": config.rolling_friction * scale,
                    "cushion_restitution": restitution,
                }
            )
        )
    observations = {
        name: public_physics(simulator, withheld_parameters=WITHHELD if name != "standard" else ())
        for name, simulator in simulators.items()
    }
    producer = producer_hashes(
        "vlm_suite.py",
        "preview.py",
        "physics.py",
        "study.py",
        "schemas.py",
        "pilot.py",
        "metrics.py",
    )
    summary: dict[str, object] = {}
    with TemporaryDirectory(prefix=".vlm-build-", dir=root) as temporary:
        staging = Path(temporary) / SUITE_DIR
        for number, (split, destination, n) in enumerate(zip(SPLITS, roots, counts, strict=True)):
            out = staging / split
            cases_path = out / "public"
            private_path = out / "private"
            cases_path.mkdir(parents=True)
            private_path.mkdir()
            used: set[str] = set()
            ordinals: Counter[tuple[str, Family, int]] = Counter()
            coverage: Counter[str] = Counter()
            ball_counts: Counter[str] = Counter()
            rejects: Counter[str] = Counter()
            interventions: Counter[str] = Counter()
            ood_counts: Counter[str] = Counter()
            with (
                (cases_path / "scenes.jsonl").open("w") as public,
                (private_path / "labels.jsonl").open("w") as private,
            ):
                for regime in (
                    ("standard", "slow_friction", "fast_friction")
                    if split == "test" and ood_per_family
                    else ("standard",)
                ):
                    target = n if regime == "standard" else ood_per_family // 2
                    family_count: Counter[str] = Counter()
                    simulator = simulators[regime]
                    visibility = observations[regime]
                    seed = 202611190 + 10 * number + tuple(simulators).index(regime)
                    for index in range(300_000):
                        if all(family_count[family] == target for family in FAMILIES):
                            break
                        family = FAMILIES[index % len(FAMILIES)]
                        if family_count[family] == target:
                            continue
                        ball_count = 1 if family == "free_roll" else 2 + (family_count[family] % 3)
                        initial, action, _ = sample_case(
                            seed,
                            index,
                            0,
                            simulator.config,
                            family_override=family,
                            ball_count=ball_count,
                        )
                        group = layout_group(initial)
                        if group in groups:
                            rejects["excluded_layout"] += 1
                            continue
                        try:
                            anchor = simulator.replay(initial, action)
                        except SimulationFailure:
                            rejects["simulation_failure"] += 1
                            continue
                        if _scratch(anchor):
                            rejects["anchor_scratch"] += 1
                            continue
                        variants = [("anchor", action, anchor)]
                        effect = None
                        intervention = None
                        if regime == "standard":
                            for field, twin_action in _counterfactuals(action, family):
                                try:
                                    twin = simulator.replay(initial, twin_action)
                                except SimulationFailure:
                                    rejects["twin_simulation_failure"] += 1
                                    continue
                                effect = _terminal_shift(anchor.goal, twin.goal, simulator.config)
                                if not _scratch(twin) and effect >= 2:
                                    variants.append(("twin", twin_action, twin))
                                    intervention = field
                                    interventions[field] += 1
                                    break
                            else:
                                rejects["no_eligible_twin"] += 1
                                continue
                        groups.add(group)
                        used.add(group)
                        family_count[family] += 1
                        ordinals[regime, family, len(initial.balls)] += 1
                        ordinal = ordinals[regime, family, len(initial.balls)]
                        if len(used) % 500 == 0:
                            LOGGER.info(
                                "%s: accepted %d independent layout groups", split, len(used)
                            )
                        if regime != "standard":
                            ood_counts[regime] += 1
                        coverage[f"{'standard' if regime == 'standard' else 'ood'}/{family}"] += 1
                        ball_counts[f"{regime}/{family}/{len(initial.balls)}"] += len(variants)
                        for member, submitted, replay in variants:
                            _export_scene(
                                out,
                                split,
                                public,
                                private,
                                group,
                                ordinal,
                                family,
                                regime,
                                member,
                                initial,
                                submitted,
                                replay,
                                simulator,
                                visibility,
                                effect,
                                intervention,
                            )
                    else:
                        raise RuntimeError(f"Unable to fill all {split}/{regime} family quotas")
            write_json(out / "groups.json", sorted(used))
            write_json(
                private_path / "physics.json",
                {name: sim.metadata() for name, sim in simulators.items()},
            )
            hashes = {
                str(path.relative_to(out)): file_sha256(path)
                for path in sorted(out.rglob("*"))
                if path.is_file()
            }
            manifest = {
                "schema_version": "inverser-vlm-v1",
                "split": split,
                "family_counts": coverage,
                "ball_count_by_regime_family": ball_counts,
                "anchor_count": 4 * n,
                "twin_count": 4 * n,
                "standard_scene_count": 8 * n,
                "ood_episode_count": sum(ood_counts.values()),
                "ood_regime_counts": ood_counts,
                "independent_layout_groups": len(used),
                "selection_rejections": rejects,
                "twin_interventions": interventions,
                "witness_replays_verified": 8 * n + sum(ood_counts.values()),
                "format": {
                    "times_s": TIMES,
                    "tolerance_radii": 1.0,
                    "minimum_twin_shift_radii": 2.0,
                    "ray_seconds": 0.16,
                    "boundary_map": _boundary_map(observations["standard"].table),
                    "ood_withheld_fields": WITHHELD,
                },
                "legacy_exclusions": exclusions,
                "producer": producer,
                "artifact_sha256": hashes,
            }
            write_json(out / "manifest.json", manifest)
            LOGGER.info(
                "%s: checked %d scenes, %d files",
                split,
                8 * n + sum(ood_counts.values()),
                len(hashes),
            )
            summary[split] = {
                "path": str(destination),
                "manifest_sha256": file_sha256(out / "manifest.json"),
                **{
                    key: manifest[key]
                    for key in (
                        "anchor_count",
                        "twin_count",
                        "standard_scene_count",
                        "ood_episode_count",
                        "independent_layout_groups",
                        "witness_replays_verified",
                    )
                },
            }
        staging.rename(suite_root)
    return {"splits": summary, "group_overlap": 0, "model_calls": 0}


def verify_vlm_suite(
    root: Path,
    pilot: Path,
    formal: Path,
    *,
    counts: tuple[int | dict[str, int], ...] = SPLIT_PAIRS,
    ood_per_family: int | dict[str, int] = 30,
    t5_attempts: int = 6,
    min_contact_share: float = 0.5,
) -> dict[str, object]:
    """Audit the published files against their manifests and independent membership invariants.

    Counts are pairs per family, either one number or a {family: pairs} mapping. In every regime at
    least `min_contact_share` of the goals must need an object ball to move (requires_contact).
    """
    excluded, provenance = _legacy_groups(pilot, formal)
    verified = {}
    for split, pairs in zip(SPLITS, counts, strict=True):
        path = root / SUITE_DIR / split
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["legacy_exclusions"] != provenance or manifest["split"] != split:
            raise ValueError(f"{split} exclusion provenance or split mismatch")
        for name, digest in manifest["artifact_sha256"].items():
            candidate = path / name
            if not candidate.resolve().is_relative_to(path.resolve()):
                raise ValueError(f"{split} unsafe artifact path: {name}")
            if file_sha256(candidate) != digest:
                raise ValueError(f"{split} checksum mismatch: {name}")
        membership: dict[str, dict[str, tuple[Layout, CueAction, str]]] = {}
        ordinals: Counter[tuple[str, Family, int]] = Counter()
        group_ordinals: dict[str, int] = {}
        families: Counter[tuple[str, str, str]] = Counter()
        contact: Counter[str] = Counter()
        scenes: Counter[str] = Counter()
        images: set[str] = set()
        with (
            (path / "private/labels.jsonl").open() as private,
            (path / "public/scenes.jsonl").open() as public,
        ):
            for label_line, scene_line in zip(private, public, strict=True):
                label, scene = json.loads(label_line), json.loads(scene_line)
                scenes[label["regime"]] += 1
                contact[label["regime"]] += requires_contact(
                    Layout.model_validate(label["initial"]),
                    Layout.model_validate(label["goal"]),
                    PhysicsConfig().ball_radius_m,
                )
                initial = Layout.model_validate(label["initial"])
                goal = Layout.model_validate(label["goal"])
                layout_sha = layout_group(initial)
                role, regime, family = (label[key] for key in ("twin_role", "regime", "family"))
                if role == "anchor":
                    if layout_sha in group_ordinals:
                        raise ValueError(f"{split} duplicate anchor group {layout_sha}")
                    ordinals[regime, family, len(initial.balls)] += 1
                    group_ordinals[layout_sha] = ordinals[regime, family, len(initial.balls)]
                elif layout_sha not in group_ordinals:
                    raise ValueError(f"{split} orphan twin record {layout_sha}")
                group_id, expected_scene, expected_init, expected_goal, expected_fwd = (
                    scene_naming(
                        split,
                        regime,
                        family,
                        len(initial.balls),
                        group_ordinals[layout_sha],
                        role,
                    )
                )
                if (
                    label.get("layout_sha256") != layout_sha
                    or label["group_id"] != group_id
                    or layout_sha in excluded
                    or label["scene_id"] != scene["scene_id"]
                    or scene["scene_id"] != expected_scene
                    or role in membership.setdefault(layout_sha, {})
                ):
                    raise ValueError(f"{split} inconsistent or overlapping group {group_id}")
                action = CueAction.model_validate(label["action"])
                membership[layout_sha][role] = (goal, action, regime)
                families[regime, family, role] += 1
                goals = [VLMGoalInput.model_validate(goal_input) for goal_input in scene["goals"]]
                if regime == "standard":
                    prompt = VLMForwardInput.model_validate(scene["forward"])
                    truth = VLMForwardOutput.model_validate(label["truth"])
                    if (
                        prompt.scene_id != expected_scene
                        or prompt.image != expected_fwd
                        or prompt.action != action
                        or prompt.ball_count != len(initial.balls)
                        or (prompt.ball_count != len(truth.resting_state))
                    ):
                        raise ValueError(f"{split} forward input/label mismatch {group_id}")
                    if forward_scores(truth, truth, prompt.ball_radius_m)["S_total"] != 1.0:
                        raise ValueError(f"{split} perfect-witness scoring mismatch {group_id}")
                    if [item.max_attempts for item in goals] != [1, t5_attempts]:
                        raise ValueError(f"{split} missing standard inverse tasks {group_id}")
                    images.add(prompt.image)
                elif (
                    scene["forward"] is not None
                    or label["truth"] is not None
                    or len(goals) != 1
                    or goals[0].max_attempts != t5_attempts
                    or goals[0].withheld_parameters != WITHHELD
                ):
                    raise ValueError(f"{split} invalid OOD task or visibility {group_id}")
                for item in goals:
                    if (
                        item.episode_id != scene["scene_id"]
                        or item.initial_image != expected_init
                        or item.goal_image != expected_goal
                        or item.ball_count != len(initial.balls)
                    ):
                        raise ValueError(f"{split} inverse input/label mismatch {group_id}")
                    images.update((item.initial_image, item.goal_image))
        def per_family(value: int | dict[str, int], family: str) -> int:
            return value[family] if isinstance(value, dict) else value

        expected = {
            ("standard", family, role): per_family(pairs, family)
            for family in FAMILIES
            for role in ("anchor", "twin")
        }
        if split == "test" and ood_per_family:
            expected.update(
                {
                    (regime, family, "anchor"): per_family(ood_per_family, family) // 2
                    for regime in ("slow_friction", "fast_friction")
                    for family in FAMILIES
                }
            )
        if any(contact[regime] < min_contact_share * n for regime, n in scenes.items()):
            raise ValueError(f"{split}: too few goals need an object ball to move")
        if families != expected or set(membership) != set(
            json.loads((path / "groups.json").read_text())
        ):
            raise ValueError(f"{split} quotas or layout-group list mismatch")
        if manifest["witness_replays_verified"] != sum(families.values()):
            raise ValueError(f"{split} witness replay accounting mismatch")
        for members in membership.values():
            ordinary, a, regime = members["anchor"]
            if regime != "standard":
                if len(members) != 1:
                    raise ValueError(f"{split} OOD layout has unrelated paired records")
                continue
            twin, b, twin_regime = members["twin"]
            changed = sum(a.model_dump()[key] != value for key, value in b.model_dump().items())
            if (
                changed != 1
                or twin_regime != regime
                or _terminal_shift(ordinary, twin, PhysicsConfig()) < 2
            ):
                raise ValueError(f"{split} invalid counterfactual pair")
        for name in sorted(images):
            if f"public/{name}" not in manifest["artifact_sha256"]:
                raise ValueError(f"{split} unhashed image {name}")
            with Image.open(path / "public" / name) as image:
                if image.format != "PNG" or max(image.size) != 1024:
                    raise ValueError(f"{split} invalid BEV image {name}")
                image.verify()
        excluded.update(membership)
        verified[split] = {
            "manifest_sha256": file_sha256(manifest_path),
            "layout_groups": len(membership),
            "scenes": sum(families.values()),
            "verified_pngs": len(images),
        }
    return {"status": "verified", "group_overlap": 0, "splits": verified, "model_calls": 0}


def _scratch(replay: Replay) -> bool:
    return any(ball.kind == "cue" and ball.status == "pocketed" for ball in replay.goal.balls)


def _export_scene(
    out: Path,
    split: str,
    public: IO[str],
    private: IO[str],
    layout_sha: str,
    ordinal: int,
    family: Family,
    regime: str,
    member: str,
    initial: Layout,
    action: CueAction,
    replay: Replay,
    simulator: SnookerSimulator,
    visibility: PublicPhysics,
    effect: float | None,
    intervention: str | None,
    t5_attempts: Literal[3, 6] = 6,
) -> None:
    group_id, scene_id, initial_name, goal_name, forward_name = scene_naming(
        split, regime, family, len(initial.balls), ordinal, member
    )
    (out / "public" / initial_name).parent.mkdir(parents=True, exist_ok=True)
    initial_image, goal_image = render_vlm_goal_pair(
        initial.endpoint_view(), replay.goal.endpoint_view(), visibility.table
    )
    if member == "anchor":
        initial_image.save(out / "public" / initial_name)
    goal_image.save(out / "public" / goal_name)
    parameters = visibility.parameters
    forward = None
    truth = None
    if regime == "standard":
        assert forward_name is not None
        render_vlm_forward_scene(initial.endpoint_view(), visibility.table, action).save(
            out / "public" / forward_name
        )
        forward = VLMForwardInput(
            scene_id=scene_id,
            image=forward_name,
            ball_count=len(initial.balls),
            parameters=parameters,
            action=action,
            table_metres=(visibility.table.width_m, visibility.table.length_m),
            ball_radius_m=visibility.table.ball_radius_m,
        ).model_dump(mode="json")
        truth = forward_truth(initial, replay, simulator.config, visibility.table)
    tasks: tuple[
        tuple[Literal["T4_visual_inverse", "T5_interactive_adaptation"], Literal[1, 3, 6]], ...
    ] = (("T4_visual_inverse", 1), ("T5_interactive_adaptation", t5_attempts))
    goals = [
        VLMGoalInput(
            task_id=task_id,
            episode_id=scene_id,
            initial_image=initial_name,
            goal_image=goal_name,
            ball_count=len(initial.balls),
            parameters=parameters,
            withheld_parameters=visibility.withheld_parameters,
            max_attempts=attempts,
            table_metres=(visibility.table.width_m, visibility.table.length_m),
            ball_radius_m=visibility.table.ball_radius_m,
        ).model_dump(mode="json")
        for task_id, attempts in tasks
        if regime == "standard" or task_id == "T5_interactive_adaptation"
    ]
    witness = simulator.replay(initial, action)
    if (
        replay != witness
        or not score_endpoint(witness.goal, replay.goal, simulator.config.ball_radius_m).success
    ):
        raise RuntimeError(f"Witness replay changed for {scene_id}")
    public.write(json.dumps({"scene_id": scene_id, "forward": forward, "goals": goals}) + "\n")
    private.write(
        json.dumps(
            {
                "scene_id": scene_id,
                "group_id": group_id,
                "layout_sha256": layout_sha,
                "family": family,
                "regime": regime,
                "twin_role": member,
                "twin_effect_radii": effect,
                "twin_intervention": intervention,
                "initial": initial.model_dump(mode="json"),
                "goal": replay.goal.model_dump(mode="json"),
                "action": action.model_dump(mode="json"),
                "events": [event.model_dump(mode="json") for event in replay.events],
                "truth": truth.model_dump(mode="json") if truth else None,
            }
        )
        + "\n"
    )
