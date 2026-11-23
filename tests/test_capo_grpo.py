from collections import Counter

import pytest

from inverser.capo_env import GoalEpisode
from inverser.capo_grpo import (
    aligned_pair_rewards,
    curriculum_schedule,
    decode_completion,
    pair_shaped_rewards,
    rank_pair_keys,
    resample_informative_pair,
    score_forward_completion,
    score_goal_completion,
)
from inverser.study import public_physics
from inverser.vlm_suite import (
    VLMForwardOutput,
    VLMGoalInput,
    format_forward_answer,
    format_goal_answer,
    forward_truth,
    normalize_goal_reply,
)


class FixtureTokenizer:
    def convert_tokens_to_ids(self, token):
        return {"<think>": 1, "</think>": 2}[token]

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        return " ".join(str(item) for item in ids if item > 3)


def test_dapo_replaces_pair_when_either_group_is_flat():
    varying = [0.0] * 7 + [1.0]
    draws = iter(
        [
            ({"id": 0}, varying + [1.0] * 8),
            ({"id": 1}, [0.0] * 8 + varying),
            ({"id": 2}, varying + varying),
        ]
    )
    stats = Counter()
    assert resample_informative_pair(lambda: next(draws), stats) == {"id": 2}
    assert stats == {
        "accepted_pair_groups": 1,
        "accepted_completions": 16,
        "rejected_zero_variance_pair_groups": 2,
    }
    with pytest.raises(RuntimeError, match="exhausted eight"):
        resample_informative_pair(lambda: ({}, [0.0] * 16), Counter())


def test_one_schedule_for_every_arm_delays_interaction_until_spin_stage():
    assert curriculum_schedule((2, 2, 3)) == [
        (1, "forward"),
        (1, "T4"),
        (2, "forward"),
        (2, "T4"),
        (3, "forward"),
        (3, "T4"),
        (3, "T5"),
    ]
    with pytest.raises(ValueError):
        curriculum_schedule((2, 2, 2))


def test_rank_partition_and_gather_keep_twin_reward_credit_aligned():
    keys = ("anchor", "twin")
    assert rank_pair_keys(keys, 0, 1) == keys
    assert rank_pair_keys(keys, 0, 2) == ("anchor",)
    assert rank_pair_keys(keys, 1, 2) == ("twin",)
    assert rank_pair_keys((("a", 2), ("b", 3)), 1, 2) == (("b", 3),)  # T5 keys keep their turn
    truth_a = VLMForwardOutput(
        object_collisions=(),
        first_contact=None,
        cushion_set=(),
        cushion_sequence=(),
        frames=(((0.0, 0.0),),) * 5,
        resting_state=((0.0, 0.0),),
    )
    truth_b = truth_a.model_copy(update={"resting_state": ((1.0, 0.0),)})
    rank_a = [("anchor", 1.0, truth_a, truth_a, 0.1)] * 8
    rank_b = [("twin", 2.0, truth_b, truth_b, 0.1)] * 7
    rank_b += [("twin", -1.0, None, truth_b, 0.1)]
    gathered = aligned_pair_rewards(rank_a + rank_b)
    assert gathered == [1.5] * 7 + [1.0] + [2.5] * 7 + [-1.0]
    with pytest.raises(ValueError, match="rank order"):
        aligned_pair_rewards(rank_a + rank_a)


def test_qwen_implicit_opening_and_incomplete_reasoning():
    tokenizer = FixtureTokenizer()
    assert decode_completion(tokenizer, [7, 9, 2, 8, 3]) == ("7 9", "8", 2)
    assert decode_completion(tokenizer, [1, 7, 2, 8]) == ("7", "8", 1)
    for invalid in ([1, 7, 8], [7, 2, 8, 2], [8, 1, 7, 2, 9]):
        assert decode_completion(tokenizer, invalid) == (None, "", None)


def test_joint_twin_shaping_does_not_reward_invalid_predictions():
    a = VLMForwardOutput(
        object_collisions=(),
        first_contact=None,
        cushion_set=(),
        cushion_sequence=(),
        frames=(((0.0, 0.0),),) * 5,
        resting_state=((0.0, 0.0),),
    )
    b = a.model_copy(update={"resting_state": ((1.0, 0.0),)})
    receipt = score_forward_completion(
        format_forward_answer(a), a, 0.1, mode="capo", reasoning_tokens=0
    )
    assert receipt == (2.5, a)
    for mode in ("outcome", "additive", "capo"):
        assert score_forward_completion("{}", b, 0.1, mode=mode, reasoning_tokens=None) == (
            -1.0,
            None,
        )
    # Exact coordinates, wrong cushion set: E = 2/3; only the gate shrinks the coordinate term.
    bounced = a.model_copy(update={"cushion_set": ("LEFT",), "cushion_sequence": ("LEFT",)})
    shot = format_forward_answer(bounced)
    credit = {
        mode: score_forward_completion(shot, a, 0.1, mode=mode, reasoning_tokens=0)[0]
        for mode in ("outcome", "additive", "capo")
    }
    assert credit["outcome"] == 1.5
    assert credit["additive"] == pytest.approx(2 / 3 + 1.5)
    assert credit["capo"] == pytest.approx(2 / 3 + 0.75 * 1.5)
    overrun = score_forward_completion(shot, a, 0.1, mode="outcome", reasoning_tokens=2048)
    assert overrun[0] == 1.25  # the length penalty applies to every arm
    left, right = pair_shaped_rewards(
        [(2.5, a)] * 3, [(2.5, b), (1.0, a), (-1.0, None)], (a, b), 0.1
    )
    assert left == [3.0, 2.5, 2.5]
    assert right == [3.0, 1.0, -1.0]
    with pytest.raises(ValueError):
        pair_shaped_rewards([(2.5, a)], [], (a, b), 0.1)


def _goal_task(sim, kind: str) -> VLMGoalInput:
    physics = public_physics(sim)
    return VLMGoalInput(
        task_id="T4_visual_inverse" if kind == "T4" else "T5_interactive_adaptation",
        episode_id="case",
        initial_image="initial",
        goal_image="goal",
        ball_count=1,
        table_metres=(physics.table.width_m, physics.table.length_m),
        ball_radius_m=physics.table.ball_radius_m,
        parameters=physics.parameters,
        max_attempts=1 if kind == "T4" else 3,
    )


def test_goal_credit_differs_only_in_how_the_forecast_meets_the_goal(study_fixture):
    sim, record, _ = study_fixture
    task = _goal_task(sim, "T4")

    def credit(reply: str, mode: str, beta_e: float = 0.1):
        return score_goal_completion(
            reply, GoalEpisode(task, record.initial, record.goal, sim), mode=mode, beta_e=beta_e
        )

    table = public_physics(sim).table
    contacts = forward_truth(
        record.initial, sim.replay(record.initial, record.witness_action), sim.config, table
    )
    for mode in ("outcome", "additive", "capo"):
        # RL demands the forecast: a shot-only reply is unusable.
        assert credit(format_goal_answer(record.witness_action), mode) == (-1.0, None)
    right = format_goal_answer(record.witness_action, forecast=contacts)
    reward, measured = credit(right, "outcome")
    assert measured is not None and measured["success"] == 1.0 and measured["event"] == 1.0
    goal = measured["near"] + measured["tail"]
    assert goal > 2.0 and reward == goal
    assert credit(right, "capo")[0] == pytest.approx(goal)
    assert credit(right, "additive")[0] == pytest.approx(goal + 0.1)
    # A fluke: the same goal-reaching shot with a wrong cushion forecast (E = 2/3).
    fluke = contacts.model_copy(
        update={"cushion_set": ("LEFT",), "cushion_sequence": ("LEFT",)}
        if not contacts.cushion_sequence
        else {"cushion_set": (), "cushion_sequence": ()}
    )
    wrong = format_goal_answer(record.witness_action, forecast=fluke)
    tail = measured["tail"]
    assert credit(wrong, "outcome")[0] == pytest.approx(goal)
    assert credit(wrong, "additive")[0] == pytest.approx(goal + 0.1 * 2 / 3)
    assert credit(wrong, "capo")[0] == pytest.approx(tail + (goal - tail) * 0.75)
    # A tap away from the goal with a perfect "nothing happens" forecast.
    tap = record.witness_action.model_copy(update={"azimuth_deg": 270.0, "cue_speed_mps": 0.05})
    tap_contacts = forward_truth(record.initial, sim.replay(record.initial, tap), sim.config, table)
    tapped = format_goal_answer(tap, forecast=tap_contacts)
    reward, measured = credit(tapped, "capo")
    assert measured is not None and measured["zero_contact"] == 1.0
    assert measured["near"] == measured["tail"] == 0.0 and reward == 0.0
    assert credit(tapped, "outcome")[0] == 0.0
    assert credit(tapped, "additive")[0] == pytest.approx(0.1)  # the shortcut the gate removes


def test_t5_turn_credit_adds_the_same_improvement_bonus_to_every_arm(study_fixture):
    sim, record, _ = study_fixture
    task = _goal_task(sim, "T5")
    table = public_physics(sim).table
    contacts = forward_truth(
        record.initial, sim.replay(record.initial, record.witness_action), sim.config, table
    )
    miss = format_goal_answer(record.witness_action.model_copy(update={"azimuth_deg": 0.0}))
    right = format_goal_answer(record.witness_action, forecast=contacts)
    for mode in ("outcome", "additive", "capo"):
        fresh = GoalEpisode(task, record.initial, record.goal, sim)
        single, measured = score_goal_completion(right, fresh, mode=mode, beta_e=0.1)
        replayed = GoalEpisode(task, record.initial, record.goal, sim)
        replayed.step(normalize_goal_reply(f"<think>first try</think>\n{miss}"))
        assert len(replayed.attempts) == 1 and not replayed.attempts[0].success
        later, _ = score_goal_completion(right, replayed, mode=mode, beta_e=0.1)
        first = replayed.attempts[0]
        assert measured is not None and first.loss_radii is not None
        assert later > single  # the second turn beats the first miss
