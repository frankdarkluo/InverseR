import json

import pytest

from inverser.analytics_data import AnalyticsTrajectory, TrajectorySample
from inverser.world_model_data import FactualForecastInput, build_factual_forecast_pilot


def test_factual_prefix_has_no_future_action_or_privileged_identity(tmp_path):
    samples = tuple(
        TrajectorySample(
            xy_cm=(index, 2 * index),
            time_text=f"0:{index * 0.1:05.2f}",
            annotation_video_time_s=index * 0.1,
        )
        for index in range(25)
    )
    track = AnalyticsTrajectory(
        sample_id="fixture",
        source_file=str(tmp_path / "tournament/round1/object1/cue.xml"),
        source_sha256="a" * 64,
        source_group="source-round-object",
        source_label="cue ball",
        source_video_title="source-video",
        samples=samples,
        timestamps_strictly_increasing=True,
    )
    source = tmp_path / "trajectories.jsonl"
    source.write_text(
        track.model_dump_json()
        + "\n"
        + track.model_copy(update={"sample_id": "duplicate-file"}).model_dump_json()
        + "\n"
    )
    report = build_factual_forecast_pilot(source, tmp_path / "pilot")
    assert report["eligible_unique_trace_cases"] == 1
    assert report["excluded_counts"] == {"duplicate_annotation_trace": 1}
    public = FactualForecastInput.model_validate_json(
        (tmp_path / "pilot/public_prefix.jsonl").read_text()
    )
    assert len(public.observed_prefix) == 10 and public.observed_prefix[0].time_s == 0
    assert max(p.time_s for p in public.observed_prefix) < min(public.query_times_s)
    payload = public.model_dump()
    assert {"future", "source_file", "source_sha256", "witness_action"}.isdisjoint(payload)
    assert not public.cue_action_available
    baseline = json.loads((tmp_path / "pilot/reference_baselines.json").read_text())[0]
    assert baseline["constant_velocity_mean_query_error_cm"] < 1e-10
    assert report["actual_llm_world_model_inference"] is False
    with pytest.raises(ValueError, match="after the entire"):
        FactualForecastInput.model_validate(
            {**payload, "query_times_s": (public.observed_prefix[-1].time_s, 1.5, 2.0)}
        )
