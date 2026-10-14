import pytest

from inverser.diagnostics import (
    CandidateOutcome,
    PairedMeasurement,
    score_bank_selection,
    score_contacts,
    score_forward,
    score_intervention_pair,
    summarize_paired,
)
from inverser.metrics import endpoint_loss
from inverser.schemas import BallPosition, Layout
from inverser.study import ForwardSubmission, PredictedFrame, SemanticContact


def positions(cue=(0.3, 0.5), reds=((0.8, 1.0), (1.1, 2.0))):
    return Layout(
        balls=(BallPosition(ball_id="cue", kind="cue", xy_m=cue),)
        + tuple(
            BallPosition(ball_id=f"red_{index}", kind="red", xy_m=xy)
            for index, xy in enumerate(reds)
        )
    )


def test_loss_preserves_red_exchangeability_max_error_and_status():
    target = positions()
    assert (
        endpoint_loss(
            positions(reds=((1.1, 2.0), (0.8, 1.0))),
            target,
            ball_radius_m=0.05,
            table_diagonal_m=4,
        )
        == 0
    )
    assert endpoint_loss(
        positions(cue=(0.4, 0.5)), target, ball_radius_m=0.05, table_diagonal_m=4
    ) == pytest.approx(2)
    pocketed = target.model_copy(
        update={
            "balls": target.balls[:2]
            + (BallPosition(ball_id="red_1", kind="red", status="pocketed"),)
        }
    )
    assert endpoint_loss(pocketed, target, ball_radius_m=0.05, table_diagonal_m=4) == 81
    with pytest.raises(ValueError, match="same-type"):
        endpoint_loss(positions(reds=()), target, ball_radius_m=0.05, table_diagonal_m=4)


def test_forward_score_separates_moving_from_stationary_majority():
    truth_layout = positions()
    truth = ForwardSubmission(
        frames=(PredictedFrame(time_s=1, state=truth_layout),),
        resting_state=truth_layout,
        events=(),
    )
    predicted = truth.model_copy(
        update={"frames": (PredictedFrame(time_s=1, state=positions(cue=(0.5, 0.5))),)}
    )
    result = score_forward(
        predicted,
        truth,
        initial=truth_layout,
        moving_ball_ids=("cue",),
        position_tolerance_m=0.05,
        event_time_tolerance_s=0.1,
    )
    assert result.moving_mean_error_m == pytest.approx(0.2)
    assert result.static_mean_error_m == 0
    assert result.maximum_position_error_m == pytest.approx(0.2)
    assert not result.positions_success and result.resting_success
    assert result.moving_position_comparisons == 1 and result.static_position_comparisons == 2


def test_forward_slot_identity_is_not_rematched_at_each_time():
    layout = positions()
    truth = ForwardSubmission(
        frames=(PredictedFrame(time_s=1, state=layout),),
        resting_state=layout,
        events=(),
    )
    reversed_reds = positions(reds=((1.1, 2.0), (0.8, 1.0)))
    changed = truth.model_copy(update={"frames": (PredictedFrame(time_s=1, state=reversed_reds),)})
    result = score_forward(
        changed,
        truth,
        initial=layout,
        moving_ball_ids=("red_0", "red_1"),
        position_tolerance_m=0.01,
        event_time_tolerance_s=0.1,
    )
    assert not result.positions_success and result.resting_success
    swapped_rest = truth.model_copy(update={"resting_state": reversed_reds})
    result = score_forward(
        swapped_rest,
        truth,
        initial=layout,
        moving_ball_ids=("red_0", "red_1"),
        position_tolerance_m=0.01,
        event_time_tolerance_s=0.1,
    )
    assert result.positions_success and not result.resting_success
    with pytest.raises(ValueError, match="sampling times"):
        score_forward(
            truth.model_copy(update={"frames": (PredictedFrame(time_s=2, state=layout),)}),
            truth,
            initial=layout,
            moving_ball_ids=(),
            position_tolerance_m=0.01,
            event_time_tolerance_s=0.1,
        )


def contact(kind, time, participants):
    return SemanticContact(event_type=kind, time_s=time, participants=participants)


def test_event_matching_handles_ball_order_repeats_missing_and_false_events():
    truth = (
        contact("ball_ball", 0.2, ("cue", "red_0")),
        contact("ball_ball", 0.5, ("cue", "red_0")),
        contact("cushion", 1.0, ("cue", "3")),
    )
    prediction = (
        contact("ball_ball", 0.21, ("red_0", "cue")),
        contact("ball_ball", 0.52, ("cue", "red_0")),
        contact("pocket", 1.0, ("red_0", "lb")),
    )
    score = score_contacts(prediction, truth, time_tolerance_s=0.05)
    assert score.matched == 2 and score.false_positive == score.false_negative == 1
    assert score.f1 == pytest.approx(2 / 3)
    assert score.matched_order_correct and not score.exact_sequence


def test_event_order_respects_simultaneous_ties_without_ignoring_real_reversal():
    truth = (contact("cushion", 1.0, ("cue", "3")), contact("pocket", 1.1, ("red_0", "lb")))
    reversed_events = (
        contact("pocket", 1.0, ("red_0", "lb")),
        contact("cushion", 1.1, ("cue", "3")),
    )
    result = score_contacts(reversed_events, truth, time_tolerance_s=0.2)
    assert result.matched == 2 and not result.matched_order_correct
    tied = tuple(event.model_copy(update={"time_s": 1.0}) for event in truth)
    assert score_contacts(tuple(reversed(tied)), tied, time_tolerance_s=0.01).exact_sequence
    empty = score_contacts((), (), time_tolerance_s=0.01)
    assert empty.exact_sequence and empty.f1 == 1
    assert score_contacts((), truth, time_tolerance_s=0.1).f1 == 0


def test_bank_scores_the_submitted_choice_not_hidden_best_candidate():
    reference = (
        CandidateOutcome(candidate_id="a", loss_radii=0, goal_success=True),
        CandidateOutcome(candidate_id="b", loss_radii=3, goal_success=False),
        CandidateOutcome(candidate_id="c", loss_radii=7, goal_success=False),
    )
    losses = {"a": 7.0, "b": 3.0, "c": 0.0}
    result = score_bank_selection(reference, losses, selected_candidate_id="c")
    assert result.rank_correlation == pytest.approx(-1)
    assert result.selected_regret_radii == 7 and not result.selected_goal_success
    assert result.feasibility == "has_feasible"
    constant = score_bank_selection(
        reference, dict.fromkeys(losses, 1.0), selected_candidate_id="a"
    )
    assert constant.rank_correlation is None and constant.selected_regret_radii == 0
    with pytest.raises(ValueError, match="not in"):
        score_bank_selection(reference, losses, selected_candidate_id="outside")


def test_bank_simulation_failure_does_not_certify_unreachability():
    reference = (
        CandidateOutcome(candidate_id="a", loss_radii=3, goal_success=False),
        CandidateOutcome(candidate_id="b", goal_success=False, failure="simulation_failure"),
    )
    result = score_bank_selection(reference, {"a": 1.0, "b": None}, selected_candidate_id="b")
    assert result.feasibility == "unknown_due_to_failures"
    assert result.selected_regret_radii is None
    assert result.failed_reference_candidates == result.invalid_predictions == 1
    abstained = score_bank_selection(reference, {"a": 1.0, "b": None}, selected_candidate_id=None)
    assert abstained.selected_failure == "abstained"
    assert (
        score_bank_selection(reference[:1], {"a": 3.0}, selected_candidate_id=None).feasibility
        == "none_in_evaluated_bank"
    )


def test_paired_statistics_cluster_variants_and_are_reproducible():
    rows = (
        PairedMeasurement(pair_id="a-1", group_id="layout-a", a=4, b=0),
        PairedMeasurement(pair_id="a-2", group_id="layout-a", a=4, b=0),
        PairedMeasurement(pair_id="b-1", group_id="layout-b", a=0, b=0),
    )
    result = summarize_paired(rows, seed=9)
    assert result.paired_rows == 3 and result.independent_groups == 2
    assert result.mean_difference_a_minus_b == 2
    assert result == summarize_paired(tuple(reversed(rows)), seed=9)
    assert summarize_paired(rows[:2], seed=9).confidence_interval is None
    with pytest.raises(ValueError, match="distinct"):
        summarize_paired((rows[0], rows[0]), seed=9)


def test_exact_truth_codec_retains_initial_slots_at_rest(study_fixture):
    _, _, case = study_fixture
    truth = case.forward_truth((0.1, 0.5))
    score = score_forward(
        truth,
        truth,
        initial=case.forward_input(case.private_record.witness_action, (0.1, 0.5)).initial,
        moving_ball_ids=case.moving_ball_ids(),
        position_tolerance_m=0.01,
        event_time_tolerance_s=0.01,
    )
    assert score.positions_success and score.resting_success
    assert score.moving_mean_error_m == 0 and score.static_mean_error_m is None


def test_ignoring_action_change_fails_joint_intervention_even_with_perfect_factual():
    initial = positions()
    factual = ForwardSubmission(
        frames=(PredictedFrame(time_s=1, state=initial),), resting_state=initial, events=()
    )
    changed = positions(cue=(0.5, 0.5))
    intervened_truth = ForwardSubmission(
        frames=(PredictedFrame(time_s=1, state=changed),), resting_state=changed, events=()
    )
    result = score_intervention_pair(
        factual,
        factual,
        factual,
        intervened_truth,
        initial=initial,
        moving_ball_ids=("cue",),
        position_tolerance_m=0.05,
        event_time_tolerance_s=0.1,
    )
    assert result.factual.positions_success and not result.intervened.positions_success
    assert not result.joint_positions_success
    assert result.moving_response_mean_error_m == pytest.approx(0.2)
    assert result.maximum_response_error_m == pytest.approx(0.2)
    assert result.moving_response_comparisons == 1 and result.paired_ball_times == 3
