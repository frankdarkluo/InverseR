import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator

from inverser.analytics_data import AnalyticsTrajectory
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.schemas import Record

LOGGER = logging.getLogger(__name__)


class PrefixPoint(Record):
    time_s: float = Field(ge=0)
    xy_cm: tuple[float, float]


class FactualForecastInput(Record):
    schema_version: Literal["real-factual-forecast-v1"] = "real-factual-forecast-v1"
    case_id: str
    source_kind: Literal["published_real_annotation"] = "published_real_annotation"
    game: Literal["nine_ball_auxiliary"] = "nine_ball_auxiliary"
    coordinate_basis: Literal["source_declared_centimetres"] = "source_declared_centimetres"
    observed_prefix: tuple[PrefixPoint, ...] = Field(min_length=10, max_length=10)
    query_times_s: tuple[float, float, float]
    complete_scene_inventory_available: Literal[False] = False
    cue_action_available: Literal[False] = False
    spin_available: Literal[False] = False
    task_scope: Literal["factual_forecast_not_action_intervention"] = (
        "factual_forecast_not_action_intervention"
    )

    @model_validator(mode="after")
    def validate_time_boundary(self) -> Self:
        if self.observed_prefix[0].time_s != 0 or any(
            a.time_s >= b.time_s for a, b in zip(self.observed_prefix, self.observed_prefix[1:])
        ):
            raise ValueError(
                "Factual prefix must start at zero with strictly increasing observations"
            )
        if self.query_times_s[0] <= self.observed_prefix[-1].time_s or any(
            a >= b for a, b in zip(self.query_times_s, self.query_times_s[1:])
        ):
            raise ValueError("Query times must be strictly ordered after the entire visible prefix")
        return self


def build_factual_forecast_pilot(trajectories: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    tracks = [AnalyticsTrajectory.model_validate_json(line) for line in trajectories.open()]
    seen = set()
    failures: Counter[str] = Counter()
    cases = []
    scores: list[dict[str, str | float]] = []
    for track in tracks:
        content_key = fingerprint(
            {
                "label": track.source_label,
                "video": track.source_video_title,
                "samples": [point.model_dump(mode="json") for point in track.samples],
            }
        )
        if content_key in seen:
            failures["duplicate_annotation_trace"] += 1
            continue
        seen.add(content_key)
        if not track.timestamps_strictly_increasing or len(track.samples) < 13:
            failures["insufficient_or_non_strict_history"] += 1
            continue
        initial_time = track.samples[0].annotation_video_time_s
        prefix = tuple(
            PrefixPoint(time_s=point.annotation_video_time_s - initial_time, xy_cm=point.xy_cm)
            for point in track.samples[:10]
        )
        last = prefix[-1]
        future = []
        for horizon in (0.25, 0.5, 1.0):
            candidates = [
                point
                for point in track.samples[10:]
                if point.annotation_video_time_s - initial_time >= last.time_s + horizon
            ]
            if not candidates:
                break
            point = candidates[0]
            if point.annotation_video_time_s - initial_time - last.time_s - horizon > 0.06:
                break
            future.append(
                PrefixPoint(time_s=point.annotation_video_time_s - initial_time, xy_cm=point.xy_cm)
            )
        if len(future) != 3:
            failures["missing_requested_horizon_or_large_annotation_gap"] += 1
            continue
        case_id = "forecast-" + content_key[:20]
        data = FactualForecastInput(
            case_id=case_id,
            observed_prefix=prefix,
            query_times_s=(future[0].time_s, future[1].time_s, future[2].time_s),
        )
        local = np.asarray(
            [[point.time_s - last.time_s, 1] for point in prefix[-3:]], dtype=np.float64
        )
        positions = np.asarray([point.xy_cm for point in prefix[-3:]], dtype=np.float64)
        velocity = np.linalg.lstsq(local, positions, rcond=None)[0][0]
        target = np.asarray([point.xy_cm for point in future], dtype=np.float64)
        horizons = np.asarray([point.time_s - last.time_s for point in future])
        static = np.broadcast_to(last.xy_cm, target.shape)
        constant_velocity = np.asarray(last.xy_cm) + horizons[:, None] * velocity
        group = str(Path(track.source_file).parent.parent)
        score: dict[str, str | float] = {
            "case_id": case_id,
            "source_group": group,
            "last_position_mean_query_error_cm": float(
                np.mean(np.linalg.norm(static - target, axis=1))
            ),
            "constant_velocity_mean_query_error_cm": float(
                np.mean(np.linalg.norm(constant_velocity - target, axis=1))
            ),
        }
        scores.append(score)
        cases.append(
            (
                data,
                {
                    "case_id": case_id,
                    "source_file": track.source_file,
                    "source_sha256": track.source_sha256,
                    "source_group": group,
                    "source_label": track.source_label,
                    "future": [point.model_dump(mode="json") for point in future],
                    "measured_cue_action_available": False,
                    "source_calibration_independently_verified": False,
                },
            )
        )
    if not cases:
        LOGGER.error("No timed trajectory meets the declared factual-prefix pilot contract")
        raise ValueError("No eligible factual forecasts; do not fabricate model inputs")
    with (
        (output / "public_prefix.jsonl").open("w", encoding="utf-8") as public,
        (output / "privileged_future.jsonl").open("w", encoding="utf-8") as private,
    ):
        import json

        for data, privileged in cases:
            public.write(data.model_dump_json() + "\n")
            private.write(json.dumps(privileged, allow_nan=False) + "\n")
    grouped: dict[str, list[dict[str, str | float]]] = defaultdict(list)
    for score in scores:
        grouped[str(score["source_group"])].append(score)
    write_json(output / "reference_baselines.json", scores)
    report = {
        "status": "development_real_factual_forecast_pilot",
        "source_trajectories_sha256": file_sha256(trajectories),
        "source_trajectory_records": len(tracks),
        "eligible_unique_trace_cases": len(cases),
        "source_round_groups": len(grouped),
        "mean_query_error_cm_equal_round_groups": {
            metric: float(
                np.mean(
                    [np.mean([float(row[metric]) for row in rows]) for rows in grouped.values()]
                )
            )
            for metric in (
                "last_position_mean_query_error_cm",
                "constant_velocity_mean_query_error_cm",
            )
        },
        "excluded_counts": dict(failures),
        "prefix_points": 10,
        "requested_horizons_s": [0.25, 0.5, 1.0],
        "reference_sampling": (
            "First source annotation after requested horizon; max60ms gap; no interpolation."
        ),
        "actual_llm_world_model_inference": False,
        "learned_dynamics_trained": False,
        "action_conditioned_intervention_gt": False,
        "metric_semantics": (
            "Source-annotated factual position compatibility in declared cm, "
            "not measured physical-control success."
        ),
    }
    write_json(output / "summary.json", report)
    return report
