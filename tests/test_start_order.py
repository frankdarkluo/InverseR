import pytest
from test_planning_pilot import planning_inputs

from inverser import start_order, study_calls
from inverser.planning_pilot import PlanningPilotConfig, run_planning_pilot
from inverser.start_order import StartOrderConfig, run_start_order_ablation


def test_start_order_reproduces_v1_traces_and_isolates_order(tmp_path, monkeypatch, study_fixture):
    panel, native = planning_inputs(tmp_path, monkeypatch, study_fixture)
    monkeypatch.setattr(study_calls, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(study_calls, "check_storage", lambda reserve: None)
    reference = tmp_path / "data/v1"
    run_planning_pilot(
        panel, native, reference, PlanningPilotConfig(layouts=4, workers=1, checkpoints=(2, 4))
    )
    config = StartOrderConfig(layouts=4, workers=1, checkpoints=(2, 4))
    output = tmp_path / "data/order"
    report = run_start_order_ablation(panel, native, reference, output, config)
    assert report["status"] == "completed"
    # geometry_powell plus both models' first-start arms, except mini's invalid side proposal.
    assert report["v1_reference_traces_checked"] == 4 + 4 + 3
    assert report["v1_reference_traces_reproduced"] == report["v1_reference_traces_checked"]
    arms = {row["arm"] for row in report["by_arm_budget_family"]}
    assert {"gpt-5_last", "gpt-5-mini_only", "geometry_g1_only", "sobol_s4_only"} <= arms
    invalid = [
        row
        for row in report["by_arm_budget_family"]
        if row["arm"] == "gpt-5-mini_last" and row["family"] == "side"
    ]
    assert all(row["successes"] == 0 for row in invalid)
    monkeypatch.setattr(start_order, "SnookerSimulator", lambda *a: pytest.fail("no new physics"))
    assert (
        run_start_order_ablation(panel, native, reference, output, config)["completed_jobs"]
        == report["completed_jobs"]
    )
