import csv
import json
import math
from collections import Counter
from dataclasses import fields
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import TypeAdapter

from inverser.processing import WindowReport
from inverser.runtime import file_sha256, write_json
from inverser.schemas import PhysicsConfig, Record
from inverser.sources import probe_video
from inverser.vision import FrameObservation

FRAME_ADAPTER = TypeAdapter(FrameObservation)


class CandidateQuality(Record):
    candidate_id: str
    video: str
    start_s: float
    end_s: float
    observed_cue_endpoint_shift_m: float | None
    nontrivial_cue_endpoint_candidate: bool
    clip_duration_s: float
    observations_sha256: str
    flags: list[str]
    clip: str
    endpoint_pair: str
    benchmark_accepted: bool = False


def parse_frame(raw: object) -> FrameObservation:
    if not isinstance(raw, dict) or set(raw) - {field.name for field in fields(FrameObservation)}:
        raise ValueError("Invalid or unexpected frame-observation fields")
    frame = FRAME_ADAPTER.validate_python(raw)
    numbers = [frame.time_s, frame.felt_fraction, frame.large_foreground_fraction]
    for ball in frame.detections:
        numbers.extend((*ball.xy_m, *ball.center_px, ball.shape_score, ball.radius_ratio))
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError("Non-finite observed coordinates or frame statistics")
    return frame


def cue_displacement(initial: FrameObservation, goal: FrameObservation) -> float | None:
    first = [ball for ball in initial.detections if ball.kind == "cue"]
    last = [ball for ball in goal.detections if ball.kind == "cue"]
    if len(first) != 1 or len(last) != 1:
        return None
    return math.dist(first[0].xy_m, last[0].xy_m)


def audit_processing(output: Path) -> dict[str, object]:
    root = output.resolve()
    summary = json.loads((root / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Cannot audit an incomplete processing run")
    reports = [WindowReport.model_validate(row) for row in summary["reports"]]
    radius = PhysicsConfig().ball_radius_m
    results: list[CandidateQuality] = []
    warnings: Counter[str] = Counter()
    for report in reports:
        for entry in report.candidate_index:
            path = Path(entry.observations)
            clip = Path(entry.clip)
            if not path.resolve().is_relative_to(root) or not clip.resolve().is_relative_to(root):
                raise ValueError("Candidate reference escapes its processing run")
            raw = json.loads(path.read_text())
            if raw["benchmark_accepted"] or raw["label_tier"] != "silver_unverified":
                raise ValueError(
                    "Unverified observations were incorrectly promoted to benchmark truth"
                )
            initial = parse_frame(raw["initial"])
            goal = parse_frame(raw["goal"])
            if (
                initial.time_s != entry.start_s
                or goal.time_s != entry.end_s
                or goal.time_s <= initial.time_s
            ):
                raise ValueError("Endpoint timestamps disagree with candidate index")
            for track in raw["tracks"].values():
                previous_time = float("-inf")
                for sample in track["samples"]:
                    time_s = float(sample["time_s"])
                    xy = sample["xy_m"]
                    if (
                        not initial.time_s <= time_s <= goal.time_s
                        or time_s <= previous_time
                        or len(xy) != 2
                        or not all(math.isfinite(value) for value in xy)
                    ):
                        raise ValueError("Invalid or non-increasing observed trajectory sample")
                    previous_time = time_s
            clip_info = probe_video(clip)
            flags = list(entry.warnings)
            displacement = cue_displacement(initial, goal)
            if displacement is None:
                flags.append("cue_endpoint_not_unique")
            elif displacement < 2 * radius:
                flags.append("small_cue_endpoint_shift_requires_shot_verification")
            if raw["single_cue_strike_verified"] or raw["full_ball_inventory_verified"]:
                raise ValueError(
                    "Automatic pilot unexpectedly claims manual/complete shot validation"
                )
            promising = displacement is not None and displacement >= 2 * radius
            warnings.update(flags)
            results.append(
                CandidateQuality(
                    candidate_id=entry.candidate_id,
                    video=report.source,
                    start_s=entry.start_s,
                    end_s=entry.end_s,
                    observed_cue_endpoint_shift_m=displacement,
                    nontrivial_cue_endpoint_candidate=promising,
                    clip_duration_s=clip_info.duration_s,
                    observations_sha256=file_sha256(path),
                    flags=flags,
                    clip=str(clip),
                    endpoint_pair=str(path.parent / "endpoint_pair.jpg"),
                )
            )
    review: dict[str, object] = {
        "status": "audited_with_unresolved_quality_flags",
        "source_videos": len(reports),
        "sampled_source_seconds": sum(item.window_duration_s for item in reports),
        "sampled_frames": sum(item.sampled_frames for item in reports),
        "candidate_clips": len(results),
        "nontrivial_cue_endpoint_candidates": sum(
            row.nontrivial_cue_endpoint_candidate for row in results
        ),
        "warning_counts": dict(warnings),
        "full_inventory_or_single_strike_verified": 0,
        "accepted_benchmark_tasks": 0,
        "candidate_reviews": [row.model_dump(mode="json") for row in results],
        "interpretation": (
            "Prioritisation for subsequent quality review, not physical-action ground truth. "
            "A small cue endpoint change can be a real shot or a false candidate."
        ),
    }
    write_json(root / "quality_review.json", review)
    with (root / "candidate_quality.csv").open("w", newline="") as stream:
        fields = [
            "candidate_id",
            "video",
            "start_s",
            "end_s",
            "observed_cue_endpoint_shift_m",
            "nontrivial_cue_endpoint_candidate",
            "flags",
            "clip",
            "endpoint_pair",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            values = row.model_dump(mode="json")
            writer.writerow(
                {key: ";".join(row.flags) if key == "flags" else values[key] for key in fields}
            )
    return review


def render_processing_overview(output: Path, *, count: int = 12) -> None:
    if not 1 <= count <= 24:
        raise ValueError("Overview is limited to 1-24 candidates")
    reviews = json.loads((output / "quality_review.json").read_text())["candidate_reviews"]
    selected = sorted(reviews, key=lambda row: not row["nontrivial_cue_endpoint_candidate"])[:count]
    if not selected:
        raise ValueError("No exported candidates to render")
    tile_width, tile_height, columns = 384, 418, 3
    canvas = Image.new(
        "RGB", (columns * tile_width, math.ceil(len(selected) / columns) * tile_height), "#eeeeee"
    )
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(selected):
        left = index % columns * tile_width
        top = index // columns * tile_height
        with Image.open(item["endpoint_pair"]) as opened:
            image = opened.convert("RGB")
            image.thumbnail((tile_width - 8, tile_height - 40))
            canvas.paste(image, (left + 4, top + 32))
        draw.text((left + 6, top + 4), item["candidate_id"], fill="#202020")
        draw.text((left + 6, top + 17), "AUTO / unverified / source->goal", fill="#802020")
    canvas.save(output / "pilot_overview.jpg")
