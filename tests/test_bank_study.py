import json
from math import dist

import httpx
import pytest
from test_azure_study import azure_inventory
from test_native_probe import probe_panel, prompt_task

from inverser.bank_study import (
    BankPanelConfig,
    BankStudyConfig,
    free_roll_resting_state,
    prepare_bank_panel,
    run_bank_study,
    validate_bank_panel,
)
from inverser.runtime import write_json
from inverser.schemas import BallPosition, CueAction, Layout
from inverser.study import public_physics

CONFIG = BankPanelConfig(layouts=4, candidates=4, bev_layouts=4, seed=7)


def bank_panel(tmp_path, monkeypatch, study_fixture):
    source = probe_panel(tmp_path, monkeypatch, study_fixture)
    output = tmp_path / "data/bank"
    return source, output, prepare_bank_panel(source, output, CONFIG)


def test_bank_panel_freezes_goal_independent_banks_with_in_bank_witnesses(
    tmp_path, monkeypatch, study_fixture
):
    source, output, report = bank_panel(tmp_path, monkeypatch, study_fixture)
    assert report["request_records"] == 40
    assert report["request_kinds"] == {
        "forecast/structured": 16,
        "selection/structured": 4,
        "forecast/bev": 16,
        "selection/bev": 4,
    }
    for path in (output / "private").glob("case-*.json"):
        labels = json.loads(path.read_text())
        outcomes = {item["candidate_id"]: item for item in labels["outcomes"]}
        assert outcomes[labels["witness_candidate_id"]]["goal_success"]
        assert len(outcomes) == len(labels["bank"]["candidates"]) == 4
    assert all("witness" not in path.read_text() for path in (output / "public").rglob("*.json"))
    assert prepare_bank_panel(source, output, CONFIG)["status"] == "frozen_bank_panel_validated"
    with pytest.raises(ValueError, match="new bank-panel version"):
        prepare_bank_panel(source, output, CONFIG.model_copy(update={"seed": 8}))
    tampered = next((output / "public").rglob("selection.json"))
    value = json.loads(tampered.read_text())
    value["goal"]["tolerance_radii"] = 2.0
    write_json(tampered, value)
    with pytest.raises(ValueError, match="Frozen bank artifact changed"):
        validate_bank_panel(output)


def test_bank_study_scores_forecast_selection_direct_choice_and_references(
    tmp_path, monkeypatch, study_fixture
):
    _, panel, _ = bank_panel(tmp_path, monkeypatch, study_fixture)
    labels = {path.stem: json.loads(path.read_text()) for path in (panel / "private").glob("c*")}
    calls = []

    def handler(request):
        calls.append(request)
        task = prompt_task(request)
        case = labels[task["case_id"]]
        model = request.url.path.split("/")[-3]
        if "candidate_actions" in task:
            first = task["candidate_actions"][0]["candidate_id"]
            choice = case["witness_candidate_id"] if model == "gpt-5-mini" else first
            content = json.dumps({"candidate_id": choice})
        else:
            assert "goal" not in task
            candidate = next(
                item["candidate_id"]
                for item in case["bank"]["candidates"]
                if item["action"] == task["action"]
            )
            truths = case["forecast_truth"]
            content = json.dumps(
                (truths[candidate] or truths[case["witness_candidate_id"]])["truth"]
            )
        return httpx.Response(
            200,
            json={
                "model": model + "-2025-08-07",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            },
        )

    source = azure_inventory(tmp_path)
    output = tmp_path / "data/bank-run"
    config = BankStudyConfig(reasoning_effort="low", max_model_requests=80)
    plan = run_bank_study(panel, source, output, config, endpoint_index=0)
    assert plan["planned_requests"] == 80 and not calls
    kwargs = {
        "endpoint_index": 0,
        "execute": True,
        "token_provider": lambda: "fixture-token",
        "transport": httpx.MockTransport(handler),
    }
    report = run_bank_study(panel, source, output, config, **kwargs)
    assert report["status"] == "completed" and len(calls) == 80
    assert {json.loads(r.content)["reasoning_effort"] for r in calls} == {"low"}
    selection = {(r["model"], r["view"], r["method"]): r for r in report["selection"]}
    for view in ("structured", "bev"):
        for model in ("gpt-5-mini", "gpt-5"):
            assert selection[model, view, "forecast_selector"]["selected_goal_successes"] == 4
            assert selection[model, view, "forecast_selector"]["median_regret_radii"] == 0
        assert selection["gpt-5-mini", view, "direct_selection"]["selected_goal_successes"] == 4
    assert selection["reference", "structured", "oracle"]["selected_goal_successes"] == 4
    assert selection["reference", "structured", "free_roll_selector"]["layouts"] == 4
    assert 0 < report["random_expected_success_rate"] <= 1
    assert all(row["valid_forecasts"] == 16 for row in report["forecast_accuracy"])
    assert len(report["paired_selection_comparisons"]) == 16
    repeated = run_bank_study(panel, source, output, config, **kwargs)
    assert repeated["new_model_requests"] == 0 and len(calls) == 80
    with pytest.raises(ValueError, match="cover every bank request"):
        run_bank_study(
            panel,
            source,
            tmp_path / "data/short",
            config.model_copy(update={"max_model_requests": 79}),
            endpoint_index=0,
        )


def test_free_roll_reference_matches_engine_only_without_contacts(study_fixture):
    simulator = study_fixture[0]
    physics = public_physics(simulator)
    initial = Layout(balls=(BallPosition(ball_id="cue", kind="cue", xy_m=(0.9, 0.8)),))
    for speed, exact in ((0.4, True), (0.8, False)):
        action = CueAction(azimuth_deg=90, cue_speed_mps=speed, tip_side=0, tip_vertical=0)
        truth = simulator.replay(initial, action).goal.balls[0].xy_m
        predicted = free_roll_resting_state(initial, action, physics).balls[0].xy_m
        assert (dist(truth, predicted) < 1e-6) == exact
