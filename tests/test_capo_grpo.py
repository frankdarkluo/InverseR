from collections import Counter

import pytest

from inverser.capo_grpo import (
    aligned_pair_rewards,
    curriculum_schedule,
    decode_completion,
    pair_shaped_rewards,
    rank_pair_keys,
    resample_informative_pair,
    score_forward_completion,
)
from inverser.vlm_suite import (
    VLMForwardOutput,
    format_coordinate_answer,
    format_forward_answer,
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


def test_joint_schedule_interleaves_arms_and_delays_interaction_until_spin_stage():
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
    for mode in ("outcome", "uncoupled"):
        control = curriculum_schedule((2, 2, 3), mode)
        assert {stage for stage, _ in control} == {3}
        assert Counter(task for _, task in control) == {"forward": 3, "T4": 4}
    assert curriculum_schedule((2, 2, 3), "uncoupled")[:3] == [(3, "forward")] * 3


def test_rank_partition_and_gather_keep_twin_reward_credit_aligned():
    keys = ("anchor", "twin")
    assert rank_pair_keys(keys, 0, 1) == keys
    assert rank_pair_keys(keys, 0, 2) == ("anchor",)
    assert rank_pair_keys(keys, 1, 2) == ("twin",)
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
    assert score_forward_completion("{}", b, 0.1, mode="outcome", reasoning_tokens=None) == (
        -1.0,
        None,
    )
    coordinates = format_coordinate_answer(b)
    assert (
        score_forward_completion(coordinates, b, 0.1, mode="outcome", reasoning_tokens=0)[0] == 1.5
    )
    assert (
        score_forward_completion(coordinates, b, 0.1, mode="uncoupled", reasoning_tokens=0)[0]
        == 1.5
    )
    assert score_forward_completion(coordinates, b, 0.1, mode="capo", reasoning_tokens=0) == (
        -1.0,
        None,
    )
    left, right = pair_shaped_rewards(
        [(2.5, a)] * 3, [(2.5, b), (1.0, a), (-1.0, None)], (a, b), 0.1
    )
    assert left == [3.0, 2.5, 2.5]
    assert right == [3.0, 1.0, -1.0]
    with pytest.raises(ValueError):
        pair_shaped_rewards([(2.5, a)], [], (a, b), 0.1)
