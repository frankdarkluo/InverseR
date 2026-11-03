import json

import httpx
import pytest
from test_azure_study import azure_inventory
from test_formal_panel import formal_panel
from test_native_probe import prompt_task

from inverser import formal_suite
from inverser.feedback_study import FeedbackStudyConfig, run_feedback_study
from inverser.formal_suite import (
    FormalClassicalConfig,
    FormalSuiteConfig,
    formal_feedback_cases,
    rescore_formal_runs,
    run_formal_classical,
    run_formal_suite,
)
from inverser.study import GoalSubmission


def labels_by_case(panel):
    return {path.stem: json.loads(path.read_text()) for path in (panel / "private").glob("c*.json")}


def oracle_handler(panel, calls):
    labels = labels_by_case(panel)

    def handler(request):
        calls.append(request)
        task = prompt_task(request)
        case = labels[task["case_id"]]
        if "candidate_actions" in task:
            witness = case["bank"]["witness_candidate_id"]
            others = [
                c["candidate_id"] for c in task["candidate_actions"] if c["candidate_id"] != witness
            ]
            content = json.dumps({"ranking": [witness, *others]})
        elif "action" in task and task["schema_version"] == "study-forward-v2":
            truths = [
                item["truth"]
                for item in case["forward_labels"].values()
                if item["action"] == task["action"]
            ]
            for candidate in case["bank"]["bank"]["candidates"]:
                if candidate["action"] == task["action"]:
                    truths.append(
                        case["bank"]["forecast_truth"][candidate["candidate_id"]]["truth"]
                    )
            content = json.dumps(truths[0])
        else:
            content = GoalSubmission(action=case["record"]["witness_action"]).model_dump_json()
        return httpx.Response(
            200,
            json={
                "model": request.url.path.split("/")[-3] + "-2025-08-07",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            },
        )

    return handler


def test_formal_suite_scores_forward_goal_and_bank_rankings(tmp_path, monkeypatch):
    _, panel, report = formal_panel(tmp_path, monkeypatch)
    planned_per_model = report["request_records"]
    config = FormalSuiteConfig(
        models=("gpt-5-mini", "gpt-5"),
        max_model_requests=2 * planned_per_model,
        max_total_tokens=10**6,
    )
    source = azure_inventory(tmp_path)
    output = tmp_path / "data/suite"
    plan = run_formal_suite(panel, source, output, config, endpoint_index=0)
    assert plan["planned_requests"] == 2 * planned_per_model
    calls = []
    kwargs = {
        "endpoint_index": 0,
        "execute": True,
        "token_provider": lambda: "fixture-token",
        "transport": httpx.MockTransport(oracle_handler(panel, calls)),
    }
    result = run_formal_suite(panel, source, output, config, **kwargs)
    assert result["status"] == "completed" and len(calls) == 2 * planned_per_model
    assert all(
        arm["valid_outputs"] == arm["requests"] == arm["beats_static_baseline"]
        for arm in result["forward"]
    )
    assert {c["contrast"] for c in result["paired_forward_contrasts"]} == {
        "oracle_kinematics",
        "bev_view",
    }
    assert all(
        item["no_change_baseline_median_response_error_radii"] > 0
        for item in result["interventions"]
    )
    assert all(goal["successes"] == goal["goals"] == 4 for goal in result["goal_direct"])
    bank = result["bank"]
    assert bank["chance"]["top1"] == 0.5
    for method in bank["methods"]:
        if method["model"] != "reference":
            assert method["top1"] == method["layouts"]
    assert {c["comparison"] for c in bank["paired_comparisons"]} >= {
        "forecast_minus_chance",
        "direct_minus_chance",
    }
    repeated = run_formal_suite(panel, source, output, config, **kwargs)
    assert repeated["new_model_requests"] == 0 and len(calls) == 2 * planned_per_model
    rescored = rescore_formal_runs(panel, (output,), tmp_path / "data/rescored.json")
    run = rescored["runs"][output.name]
    for key in ("forward", "paired_forward_contrasts", "interventions", "goal_direct", "bank"):
        assert json.loads(json.dumps(run[key])) == json.loads(json.dumps(result[key]))
    with pytest.raises(ValueError, match="cover every formal request"):
        run_formal_suite(
            panel,
            source,
            tmp_path / "data/short",
            config.model_copy(update={"max_model_requests": 1}),
            endpoint_index=0,
        )


def test_formal_classical_baselines_resume_without_new_physics(tmp_path, monkeypatch):
    _, panel, _ = formal_panel(tmp_path, monkeypatch)
    output = tmp_path / "data/classical"
    config = FormalClassicalConfig(checkpoints=(2, 4), workers=1)
    report = run_formal_classical(panel, output, config)
    assert report["status"] == "completed" and report["completed_jobs"] == 4 * 4
    overall = {
        (row["method"], row["query_budget"])
        for row in report["by_method_budget_family"]
        if row["family"] is None
    }
    assert overall == {
        ("geometry_direct", 0),
        ("random", 2),
        ("random", 4),
        ("sobol", 2),
        ("sobol", 4),
        ("geometry_powell", 2),
        ("geometry_powell", 4),
    }
    monkeypatch.setattr(
        formal_suite,
        "SnookerSimulator",
        lambda *a: pytest.fail("Completed cases must not simulate"),
    )
    assert run_formal_classical(panel, output, config)["completed_jobs"] == 16


def test_formal_feedback_cases_run_on_the_declared_subset(tmp_path, monkeypatch):
    _, panel, _ = formal_panel(tmp_path, monkeypatch)
    cases, case_source = formal_feedback_cases(panel)
    assert len(cases) == 4 and "formal_test_manifest_sha256" in case_source
    calls = []
    config = FeedbackStudyConfig(
        layouts=4, max_attempts=2, conditions=("genuine",), max_model_requests=16
    )
    report = run_feedback_study(
        cases,
        case_source,
        azure_inventory(tmp_path),
        tmp_path / "data/feedback",
        config,
        endpoint_index=0,
        execute=True,
        token_provider=lambda: "fixture-token",
        transport=httpx.MockTransport(oracle_handler(panel, calls)),
    )
    assert report["status"] == "completed"
    assert all(arm["final_successes"] == 4 for arm in report["by_model_condition"])
