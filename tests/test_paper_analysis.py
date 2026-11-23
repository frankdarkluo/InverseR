import json

import pytest

from inverser import study_calls
from inverser.paper_analysis import analyze_runs, summarize_rows
from inverser.runtime import write_json
from inverser.study_calls import lock_run, save_job


def test_mechanism_denominators_keep_invalids_distinct_from_stasis():
    def row(*, valid, event, coords, stasis):
        return {
            "task": "forward",
            "family": "center",
            "valid": valid,
            "true_object_collision": True,
            "predicted_no_object_collision": stasis,
            "scores": {
                "T1": event,
                "T2": True,
                "T3": float(coords),
                "T3_all_slots": coords,
                "S_total": 0.3 + 0.3 * event + 0.4 * coords,
                "T1_per_ball": float(event),
                "T2_cushion_set": True,
                "T3_1r": float(coords),
                "T3_2r": float(coords),
                "T3_3r": 1.0 if valid else 0.0,
                "T3_5r": 1.0 if valid else 0.0,
                "S_total_3r": (0.3 * event + 0.3 + 0.4) if valid else 0.0,
            },
        }

    report = summarize_rows(
        [
            row(valid=True, event=True, coords=True, stasis=False),
            row(valid=True, event=False, coords=True, stasis=True),
            row(valid=False, event=False, coords=False, stasis=False),
            {"task": "T5", "success": True, "attempts": [{"success": False}, {"success": True}]},
        ]
    )
    counts = report["forward"]["all"]
    assert counts["joint_event_and_all_coordinates"] == {"n": 3, "count": 1, "rate": 1 / 3}
    assert counts["joint_normalized_event_and_3r"] == {"n": 3, "count": 1, "rate": 1 / 3}
    assert counts["mean"]["T3_3r"] == pytest.approx(2 / 3)
    assert counts["mean"]["S_total_3r"] == pytest.approx(1.7 / 3)
    assert counts["all_coordinates_given_wrong_event"] == {"n": 2, "count": 1, "rate": 0.5}
    assert counts["stasis_given_true_collision_and_valid"] == {"n": 2, "count": 1, "rate": 0.5}
    assert report["T5_by_attempt"]["1"]["rate"] == 0
    assert report["T5_by_attempt"]["6"]["rate"] == 1  # v1 rows: six attempts


def test_v3_goal_rows_report_their_attempt_cap_forecasts_and_the_no_shot_baseline():
    def attempt(loss, success=False):
        return {"loss_radii": loss, "success": success}

    def forecast(present, agreement=None):
        return {"forecast_present": present, "event_agreement": agreement}

    report = summarize_rows(
        [
            {
                "task": "T4",
                "success": False,
                "attempts": [attempt(30.0)],
                "baseline_radii": 40.0,
                "max_attempts": 1,
                "forecasts": [forecast(True, 1.0)],
            },
            {
                "task": "T4",
                "success": False,
                "attempts": [attempt(None)],
                "baseline_radii": 40.0,
                "max_attempts": 1,
                "forecasts": [forecast(False)],
            },
            {
                "task": "T5",
                "success": True,
                "attempts": [attempt(9.0), attempt(0.5, True)],
                "baseline_radii": 12.0,
                "max_attempts": 3,
                "forecasts": [forecast(True, 0.5), forecast(True, 1.0)],
            },
        ]
    )
    assert list(report["T5_by_attempt"]) == ["1", "2", "3"]
    assert report["T5_by_attempt"]["2"]["rate"] == 1
    assert report["T4_beats_no_shot"] == {"n": 2, "count": 1, "rate": 0.5}
    assert report["T4_forecast"]["coverage"]["rate"] == 0.5
    assert report["T4_forecast"]["event_agreement"] == {"n": 1, "rate": 1.0}
    assert report["T5_forecast"]["event_agreement"] == {"n": 2, "rate": 0.75}


def test_analysis_is_byte_reproducible_and_rejects_modified_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(study_calls, "PROJECT_ROOT", tmp_path)
    base = tmp_path / "data"
    runs = {name: base / name for name in ("left", "right")}
    for arm, path in runs.items():
        run_hash = lock_run(path, {"arm": arm}, reserve_bytes=0)
        write_json(
            path / "summary.json",
            {
                "status": "complete",
                "run_hash": run_hash,
                "completed": 2,
                "usage": {},
                "counts": {"model_requests": 2},
            },
        )
        for i in range(2):
            save_job(path / "jobs" / f"{i}.json", run_hash, {"idx": i, "correct": arm == "left"})
    first = analyze_runs(runs, base / "analysis-first")
    contrast = first["paired"]["left_minus_right"]["public_exact_answer"]
    assert contrast["confidence_interval"] == [1, 1]
    analyze_runs(runs, base / "analysis-second")
    for name in ("analysis.json", "table.csv"):
        assert (base / "analysis-first" / name).read_bytes() == (
            base / "analysis-second" / name
        ).read_bytes()
    target = runs["left"] / "jobs/0.json"
    payload = json.loads(target.read_text())
    payload["payload"]["correct"] = False
    write_json(target, payload)
    with pytest.raises(ValueError):
        analyze_runs(runs, base / "modified")
