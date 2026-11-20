import pytest

from inverser.agentic_baselines import (
    EventProposal,
    event_distance,
    event_guided_annealing,
    parse_event_proposal,
)
from inverser.capo_env import GoalAttempt
from inverser.evaluation import BudgetExhausted


def test_parse_event_proposal_accepts_answer_tags_and_rejects_bad_events():
    tagged = (
        "<think>Aim for cushion then object 1.</think>\n"
        "<answer>\n"
        "<prediction>Rail rebound into object ball.</prediction>\n"
        "<desired_events>cushion, ball_ball</desired_events>\n"
        "<azimuth_deg>120.5</azimuth_deg>\n"
        "<cue_speed_mps>1.45</cue_speed_mps>\n"
        "<tip_side>0.10</tip_side>\n"
        "<tip_vertical>-0.15</tip_vertical>\n"
        "</answer>"
    )
    parsed = parse_event_proposal(tagged)
    assert parsed.desired_events == ("cushion", "ball_ball")
    assert parsed.cue_action().azimuth_deg == 120.5
    with pytest.raises(ValueError):
        parse_event_proposal(tagged.replace("ball_ball", "jump_shot"))


def test_annealing_counts_simulator_calls_and_uses_event_order(study_fixture):
    _, record, _ = study_fixture
    proposal = EventProposal(
        azimuth_deg=359.0,
        cue_speed_mps=0.3,
        tip_side=0.0,
        tip_vertical=0.0,
        prediction="First wall then ball.",
        desired_events=("cushion", "ball_ball"),
    )
    assert event_distance(("cushion", "ball_ball"), ("ball_ball", "cushion")) == 0.5
    seen = []

    def judge(action):
        seen.append(action)
        attempt = GoalAttempt(
            ordinal=1,
            action=action,
            outcome=record.goal,
            loss_radii=0.0,
            success=True,
            failure=None,
        )
        return attempt, ("cushion", "ball_ball") if len(seen) == 3 else ("ball_ball",)

    selected, history = event_guided_annealing(proposal, judge, 5)
    assert 3 <= len(history) == len(seen) <= 6
    assert selected.action == seen[2]
    assert all(0 <= action.azimuth_deg < 360 for action in seen)

    def over_budget(action):
        raise BudgetExhausted("trial allocation")

    with pytest.raises(BudgetExhausted, match="trial allocation"):
        event_guided_annealing(proposal, over_budget, 5)
