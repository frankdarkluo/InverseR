import json

import pytest

from inverser.capo_env import (
    GoalEpisode,
    forward_reward,
    parse_forward_reward,
    twin_contrast,
    varied_group,
)
from inverser.pilot import sample_case
from inverser.schemas import PhysicsConfig
from inverser.self_verify import select_by_forecast
from inverser.study import public_physics
from inverser.vlm_suite import (
    WITHHELD,
    VLMForwardOutput,
    VLMGoalInput,
    format_forward_answer,
    format_goal_answer,
    forward_truth,
)


def test_causal_reward_requires_events_even_when_coordinates_are_correct():
    truth = VLMForwardOutput(
        object_collisions=(True,),
        first_contact=1,
        cushion_set=("LEFT",),
        cushion_sequence=("LEFT",),
        frames=(((0.0, 0.0), (0.0, 1.0)),) * 5,
        resting_state=((1.0, 0.0), (0.0, 1.0)),
    )
    wrong = VLMForwardOutput.model_validate(
        truth.model_dump()
        | {
            "object_collisions": (False,),
            "first_contact": None,
            "cushion_set": (),
            "cushion_sequence": (),
        }
    )
    full = forward_reward(truth, truth, 0.1)
    bad_event = forward_reward(wrong, truth, 0.1)
    assert full.total == 2.5
    assert bad_event.event == 0 and bad_event.coordinate == full.coordinate
    assert bad_event.gated == 0.25 * full.gated
    tagged = format_forward_answer(truth)
    assert parse_forward_reward(f"<think>contact</think>\n{tagged}", truth, 0.1) == full.total
    assert parse_forward_reward(truth.model_dump_json(), truth, 0.1) == -1
    assert parse_forward_reward(f"```xml\n{tagged}\n```", truth, 0.1) == -1
    assert forward_reward(truth, truth, 0.1, reasoning_tokens=2048).total == 2.25


def test_twin_contrast_rewards_change_but_not_stasis_wrong_direction_or_exaggeration():
    a = VLMForwardOutput(
        object_collisions=(),
        first_contact=None,
        cushion_set=(),
        cushion_sequence=(),
        frames=(((0.0, 0.0),),) * 5,
        resting_state=((0.0, 0.0),),
    )
    b = a.model_copy(update={"resting_state": ((1.0, 0.0),)})
    bigger = a.model_copy(update={"resting_state": ((2.0, 0.0),)})
    assert twin_contrast((a, b), (a, b), 0.125) == 1
    assert twin_contrast((a, a), (a, b), 0.125) == 0
    assert twin_contrast((b, a), (a, b), 0.125) == 0
    assert twin_contrast((a, bigger), (a, b), 0.125) == 0.5
    assert not varied_group([1.0] * 8) and varied_group([1.0] * 7 + [0.0])
    with pytest.raises(ValueError):
        varied_group([1.0] * 7)


def test_episode_resets_layout_and_feedback_does_not_reveal_physics(study_fixture):
    sim, record, _ = study_fixture
    physics = public_physics(sim, withheld_parameters=WITHHELD)
    task = VLMGoalInput(
        task_id="T5_interactive_adaptation",
        episode_id="case",
        initial_image="initial",
        goal_image="goal",
        ball_count=1,
        table_metres=(physics.table.width_m, physics.table.length_m),
        ball_radius_m=physics.table.ball_radius_m,
        parameters=physics.parameters,
        withheld_parameters=WITHHELD,
        max_attempts=6,
    )
    episode = GoalEpisode(task, record.initial, record.goal, sim)
    n = sim.rollouts
    assert episode.step("invalid").reward == 0 and sim.rollouts == n
    wrong = format_goal_answer(record.witness_action.model_copy(update={"azimuth_deg": 270.0}))
    miss = episode.step(wrong)
    feedback, image = episode.feedback(miss)
    assert image is not None and not miss.success and miss.ordinal == 2
    assert json.loads(feedback)["cue_residual_to_goal_m"] is not None
    assert all(field not in feedback for field in WITHHELD)
    hit = episode.step(f"<think>adjust aim</think>\n{format_goal_answer(record.witness_action)}")
    assert hit.success and episode.done and sim.rollouts == n + 2
    with pytest.raises(ValueError, match="terminated"):
        episode.step(format_goal_answer(record.witness_action))
    stalled = GoalEpisode(task, record.initial, record.goal, sim)
    for _ in range(6):
        stalled.step("invalid")
    assert stalled.done


def test_self_verify_ranks_only_model_forecasts(study_fixture):
    sim, record, _ = study_fixture
    physics = PhysicsConfig()
    sizing = dict(
        ball_radius_m=physics.ball_radius_m,
        table_metres=(physics.table_width_m, physics.table_length_m),
    )
    a = record.witness_action
    b = a.model_copy(update={"azimuth_deg": 180.0})
    table = public_physics(sim).table
    hit = forward_truth(record.initial, sim.replay(record.initial, a), physics, table)
    miss = hit.model_copy(update={"resting_state": (record.initial.balls[0].xy_m,)})
    count = sim.rollouts
    forecasts = {a.azimuth_deg: hit, b.azimuth_deg: miss}
    found = select_by_forecast(
        [b, a],
        lambda action: forecasts[action.azimuth_deg],
        record.initial,
        record.goal,
        **sizing,
    )
    assert found == (1, a) and sim.rollouts == count
    assert select_by_forecast([a], lambda _: None, record.initial, record.goal, **sizing) is None


def test_spin_families_can_have_four_total_balls():
    for family in ("center", "vertical", "side"):
        layout, _, actual = sample_case(
            4,
            1,
            0,
            PhysicsConfig(),
            family_override=family,
            ball_count=4,
        )
        assert actual == family and len(layout.balls) == 4
