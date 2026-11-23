import json
from collections import Counter

import httpx
import pytest

from inverser import study_calls
from inverser.model_adapters import ModelCall
from inverser.physics import SnookerSimulator
from inverser.pilot import FAMILIES
from inverser.runtime import write_json
from inverser.study import GoalSubmission
from inverser.study_calls import StudyAdapter, local_vllm_backends
from inverser.training_env import (
    GoalEnvironment,
    TrainingSetConfig,
    environment_from_training_set,
    goal_reward,
    load_training_goals,
    prepare_training_set,
    tail_reward,
    training_goals,
    training_records,
)

SMALL = TrainingSetConfig(seed=7, per_family=1, max_index=500)


def fresh_records():
    return list(training_records(SMALL, set()))


def witness_json(goal):
    return GoalSubmission(action=goal.witness_action.model_dump()).model_dump_json()


def test_training_records_are_balanced_distinct_and_skip_excluded_groups():
    tally = Counter()
    records = list(training_records(SMALL, set(), tally))
    assert sorted(record.family for record in records) == sorted(FAMILIES)
    assert len({record.group_id for record in records}) == 4
    assert all(record.task_id.startswith("train-7-") for record in records)
    assert tally["generation_replays"] >= 4
    again = list(training_records(SMALL, {records[0].group_id}))
    assert len(again) == 4 and records[0].group_id not in {record.group_id for record in again}
    with pytest.raises(ValueError, match="Too few"):
        list(training_records(SMALL.model_copy(update={"max_index": 2}), set()))


def test_goal_reward_is_monotone_with_a_success_bonus():
    assert goal_reward(None, False) == 0.0
    assert goal_reward(10.0, False) == 0.5 and goal_reward(100.0, False) == 0.0
    assert goal_reward(0.5, True) == 1.975 and goal_reward(10.0, False, scale_radii=40) == 0.75


def test_no_shot_baseline_leaves_a_tap_with_nothing_near_or_far():
    # Far goal (L0 = 50r): near keeps its 20r scale, the tail pays only for beating L0.
    assert goal_reward(10.0, False, baseline_radii=50.0) == 0.5
    assert tail_reward(25.0, 50.0) == pytest.approx(0.1)
    assert goal_reward(50.0, False, baseline_radii=50.0) == tail_reward(50.0, 50.0) == 0.0
    assert tail_reward(80.0, 50.0) == 0.0 and tail_reward(None, 50.0) == 0.0
    # Close goal (L0 = 8r): the near scale shrinks to L0, so a tap (L = L0) earns nothing.
    assert goal_reward(8.0, False, baseline_radii=8.0) == 0.0
    assert goal_reward(4.0, False, baseline_radii=8.0) == 0.5
    assert goal_reward(0.5, True, baseline_radii=8.0) == pytest.approx(1 + 1 - 0.5 / 8)


def test_goal_environment_scores_the_formal_goal_contract():
    records = fresh_records()
    goals = training_goals(records, SnookerSimulator(records[0].physics))
    env = GoalEnvironment(goals)
    assert len(env) == 4 and "witness" not in json.dumps(env.messages(0))
    hit = env.step(0, witness_json(goals[0]))
    assert hit.success and hit.failure is None and hit.reward > 1.9 and hit.loss_radii <= 1.0
    assert env.step(0, "<think>\naim first\n</think>\n" + witness_json(goals[0])) == hit
    bad = env.step(0, "not json")
    assert (bad.action, bad.reward, bad.failure) == (None, 0.0, "invalid_submission")
    away = {**goals[0].witness_action.model_dump()}
    away["azimuth_deg"] = (away["azimuth_deg"] + 180) % 360
    wrong = env.step(0, GoalSubmission(action=away).model_dump_json())
    assert not wrong.success and 0.0 <= wrong.reward < 1.0 and wrong.loss_radii > 1.0
    plain, chat = witness_json(goals[0]), [{"role": "assistant", "content": "x"}]
    assert env.rewards([plain, chat], [0, 0]) == [hit.reward, 0.0]
    assert env.simulator.rollouts == 4
    rows = env.rows()
    assert [row["case_index"] for row in rows] == [0, 1, 2, 3]
    assert rows[0]["prompt"] == env.messages(0) and rows[0]["family"] == goals[0].family
    with pytest.raises(ValueError, match="at least one goal"):
        GoalEnvironment(())


def test_environment_prompt_matches_the_study_adapter_request():
    records = fresh_records()[:1]
    env = GoalEnvironment(training_goals(records, SnookerSimulator(records[0].physics)))
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "local-model",
                "choices": [
                    {"message": {"content": witness_json(env.goals[0])}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    backend = local_vllm_backends(
        "http://127.0.0.1:8000/v1",
        ("local-model",),
        model_revision="rev",
        reasoning_effort="none",
        max_output_tokens=1024,
        request_token_reservation=16384,
        timeout_s=30,
    )[0]
    with StudyAdapter(
        backend.model_copy(update={"execution_authorized": True}),
        transport=httpx.MockTransport(handler),
    ) as adapter:
        adapter._complete(ModelCall(view="structured", input=env.goal_input(0)), "request-id")
    system, user = seen[0]["messages"]
    assert system == env.messages(0)[0]
    assert user["content"][0]["text"] == env.messages(0)[1]["content"]


def test_prepare_training_set_freezes_goals_disjoint_from_exclusions(tmp_path, monkeypatch):
    monkeypatch.setattr(study_calls, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(study_calls, "check_storage", lambda reserve: None)
    taken = fresh_records()[0]
    dev = tmp_path / "dev.jsonl"
    dev.write_text(json.dumps({"group_id": taken.group_id}) + "\n")
    panel = tmp_path / "formal"
    (panel / "private").mkdir(parents=True)
    write_json(panel / "manifest.json", {"frozen": True})
    write_json(panel / "private/selection.json", {"formal_test": [{"source_group": "other"}]})
    output = tmp_path / "data/train"
    report = prepare_training_set(output, SMALL, exclude_records=(dev,), exclude_panels=(panel,))
    assert report["layouts"] == 4 and report["excluded_groups"] == 2
    assert report["witness_replays_checked"] == 4 and report["actual_model_calls"] == 0
    groups = {json.loads(line)["group_id"] for line in (output / "privileged.jsonl").open()}
    assert taken.group_id not in groups and len(groups) == 4
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["partition"] == "training" and manifest["excluded_overlapping_groups"] == 0
    assert manifest["family_counts"] == dict.fromkeys(FAMILIES, 1)
    assert [item["groups"] for item in manifest["exclusions"]] == [1, 1]
    goals = load_training_goals(output)
    env = environment_from_training_set(output)
    assert len(env) == 4 and env.step(1, witness_json(goals[1])).success
    with pytest.raises(ValueError, match="new output directory"):
        prepare_training_set(output, SMALL)
