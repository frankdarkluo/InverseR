from collections import defaultdict
from math import dist
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from inverser.schemas import Layout, Record
from inverser.study import ForwardSubmission, SemanticContact


class ContactScore(Record):
    matched: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    matched_order_correct: bool
    exact_sequence: bool


def _event_key(event: SemanticContact) -> tuple[str, tuple[str, str]]:
    first, second = event.participants
    if event.event_type == "ball_ball":
        first, second = sorted((first, second))
    participants = first, second
    return event.event_type, participants


def score_contacts(
    predicted: tuple[SemanticContact, ...],
    target: tuple[SemanticContact, ...],
    *,
    time_tolerance_s: float,
    tie_tolerance_s: float = 1e-9,
) -> ContactScore:
    if not np.isfinite(time_tolerance_s) or time_tolerance_s <= 0:
        raise ValueError("Contact matching requires a positive finite temporal tolerance")
    if not np.isfinite(tie_tolerance_s) or tie_tolerance_s < 0:
        raise ValueError("Contact tie tolerance must be nonnegative and finite")
    pairs: list[tuple[int, int]] = []
    if predicted and target:
        gaps = np.asarray(
            [[abs(a.time_s - b.time_s) / time_tolerance_s for b in target] for a in predicted],
            dtype=np.float64,
        )
        compatible = np.asarray(
            [[_event_key(a) == _event_key(b) for b in target] for a in predicted]
        ) & (gaps <= 1)
        rows, columns = linear_sum_assignment(
            np.where(compatible, gaps, max(len(predicted), len(target)) + 1)
        )
        pairs = [(int(i), int(j)) for i, j in zip(rows, columns, strict=True) if compatible[i, j]]
    matched = len(pairs)
    precision = matched / len(predicted) if predicted else float(not target)
    recall = matched / len(target) if target else float(not predicted)
    order_correct = all(
        predicted[i].time_s < predicted[k].time_s
        for i, j in pairs
        for k, target_index in pairs
        if target[target_index].time_s - target[j].time_s > tie_tolerance_s
    )
    return ContactScore(
        matched=matched,
        false_positive=len(predicted) - matched,
        false_negative=len(target) - matched,
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0,
        matched_order_correct=order_correct,
        exact_sequence=matched == len(predicted) == len(target) and order_correct,
    )


class ForwardScore(Record):
    evaluated_ball_times: int = Field(ge=0)
    status_mismatches: int = Field(ge=0)
    moving_position_comparisons: int = Field(ge=0)
    static_position_comparisons: int = Field(ge=0)
    moving_mean_error_m: float | None
    static_mean_error_m: float | None
    maximum_position_error_m: float | None
    positions_success: bool
    resting_success: bool
    resting_max_position_error_m: float | None
    contacts: ContactScore


def score_forward(
    predicted: ForwardSubmission,
    target: ForwardSubmission,
    *,
    initial: Layout,
    moving_ball_ids: tuple[str, ...],
    position_tolerance_m: float,
    event_time_tolerance_s: float,
) -> ForwardScore:
    if not np.isfinite(position_tolerance_m) or position_tolerance_m <= 0:
        raise ValueError("Position tolerance must be positive and finite")
    if tuple(frame.time_s for frame in predicted.frames) != tuple(
        frame.time_s for frame in target.frames
    ):
        raise ValueError("Forward prediction must use exactly the requested truth sampling times")
    inventory = {ball.ball_id: ball.kind for ball in initial.balls}
    moving = set(moving_ball_ids)
    if len(moving) != len(moving_ball_ids) or moving - set(inventory):
        raise ValueError("Moving-ball strata must be distinct initial slots")
    errors: dict[str, list[float]] = {"moving": [], "static": []}
    status_mismatches = 0
    evaluated = 0
    for a, b in zip(predicted.frames, target.frames, strict=True):
        for layout in (a.state, b.state):
            if {ball.ball_id: ball.kind for ball in layout.balls} != inventory or any(
                ball.status == "unobserved" for ball in layout.balls
            ):
                raise ValueError("Forward frames must retain every initial slot without unknowns")
        truth = {ball.ball_id: ball for ball in b.state.balls}
        for ball in a.state.balls:
            reference = truth[ball.ball_id]
            evaluated += 1
            status_mismatches += ball.status != reference.status
            if ball.xy_m is not None and reference.xy_m is not None:
                key = "moving" if ball.ball_id in moving else "static"
                errors[key].append(dist(ball.xy_m, reference.xy_m))
    all_errors = errors["moving"] + errors["static"]
    maximum = max(all_errors) if all_errors else None
    for resting in (predicted.resting_state, target.resting_state):
        if {ball.ball_id: ball.kind for ball in resting.balls} != inventory or any(
            ball.status == "unobserved" for ball in resting.balls
        ):
            raise ValueError("Forward resting state must preserve every initial slot")
    resting_truth = {ball.ball_id: ball for ball in target.resting_state.balls}
    resting_status_mismatch = any(
        ball.status != resting_truth[ball.ball_id].status for ball in predicted.resting_state.balls
    )
    resting_errors = [
        dist(ball.xy_m, reference.xy_m)
        for ball in predicted.resting_state.balls
        for reference in (resting_truth[ball.ball_id],)
        if ball.xy_m is not None and reference.xy_m is not None
    ]
    resting_maximum = max(resting_errors) if resting_errors else None
    return ForwardScore(
        evaluated_ball_times=evaluated,
        status_mismatches=status_mismatches,
        moving_position_comparisons=len(errors["moving"]),
        static_position_comparisons=len(errors["static"]),
        moving_mean_error_m=float(np.mean(errors["moving"])) if errors["moving"] else None,
        static_mean_error_m=float(np.mean(errors["static"])) if errors["static"] else None,
        maximum_position_error_m=maximum,
        positions_success=status_mismatches == 0
        and (maximum is None or maximum <= position_tolerance_m),
        resting_success=not resting_status_mismatch
        and (resting_maximum is None or resting_maximum <= position_tolerance_m),
        resting_max_position_error_m=resting_maximum,
        contacts=score_contacts(
            predicted.events, target.events, time_tolerance_s=event_time_tolerance_s
        ),
    )


class CandidateOutcome(Record):
    candidate_id: str
    loss_radii: float | None = Field(default=None, ge=0)
    goal_success: bool
    failure: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if (self.loss_radii is None) != (self.failure is not None):
            raise ValueError("Candidate must have a finite scored loss or an explicit failure")
        if self.failure is not None and self.goal_success:
            raise ValueError("A failed candidate cannot be a successful shot")
        return self


class InterventionScore(Record):
    paired_ball_times: int
    moving_response_comparisons: int
    moving_response_mean_error_m: float | None
    maximum_response_error_m: float | None
    status_patterns_correct: bool
    factual: ForwardScore
    intervened: ForwardScore
    joint_positions_success: bool
    joint_events_success: bool


def score_intervention_pair(
    factual_prediction: ForwardSubmission,
    intervened_prediction: ForwardSubmission,
    factual_truth: ForwardSubmission,
    intervened_truth: ForwardSubmission,
    *,
    initial: Layout,
    moving_ball_ids: tuple[str, ...],
    position_tolerance_m: float,
    event_time_tolerance_s: float,
) -> InterventionScore:
    if tuple(frame.time_s for frame in factual_truth.frames) != tuple(
        frame.time_s for frame in intervened_truth.frames
    ):
        raise ValueError("Intervention responses require identical physical sampling times")
    scores = tuple(
        score_forward(
            prediction,
            truth,
            initial=initial,
            moving_ball_ids=moving_ball_ids,
            position_tolerance_m=position_tolerance_m,
            event_time_tolerance_s=event_time_tolerance_s,
        )
        for prediction, truth in (
            (factual_prediction, factual_truth),
            (intervened_prediction, intervened_truth),
        )
    )
    moving = set(moving_ball_ids)
    errors = []
    moving_errors = []
    patterns = True
    paired = 0
    for frames in zip(
        factual_prediction.frames,
        intervened_prediction.frames,
        factual_truth.frames,
        intervened_truth.frames,
        strict=True,
    ):
        indexed = tuple({ball.ball_id: ball for ball in frame.state.balls} for frame in frames)
        for ball in initial.balls:
            a, b, c, d = (states[ball.ball_id] for states in indexed)
            paired += 1
            patterns &= (a.status, b.status) == (c.status, d.status)
            if (
                a.xy_m is not None
                and b.xy_m is not None
                and c.xy_m is not None
                and d.xy_m is not None
            ):
                predicted_delta = np.subtract(b.xy_m, a.xy_m)
                truth_delta = np.subtract(d.xy_m, c.xy_m)
                error = float(np.linalg.norm(predicted_delta - truth_delta))
                errors.append(error)
                if ball.ball_id in moving:
                    moving_errors.append(error)
    return InterventionScore(
        paired_ball_times=paired,
        moving_response_comparisons=len(moving_errors),
        moving_response_mean_error_m=float(np.mean(moving_errors)) if moving_errors else None,
        maximum_response_error_m=max(errors) if errors else None,
        status_patterns_correct=patterns,
        factual=scores[0],
        intervened=scores[1],
        joint_positions_success=scores[0].positions_success and scores[1].positions_success,
        joint_events_success=scores[0].contacts.exact_sequence
        and scores[1].contacts.exact_sequence,
    )


class BankSelectionScore(Record):
    candidate_count: int
    failed_reference_candidates: int
    invalid_predictions: int
    ranked_candidate_count: int
    rank_correlation: float | None
    selected_candidate_id: str | None
    selected_goal_success: bool
    selected_failure: str | None
    oracle_loss_radii: float | None
    selected_regret_radii: float | None
    feasibility: Literal["has_feasible", "none_in_evaluated_bank", "unknown_due_to_failures"]


def score_bank_selection(
    outcomes: tuple[CandidateOutcome, ...],
    prediction_losses: dict[str, float | None],
    *,
    selected_candidate_id: str | None,
) -> BankSelectionScore:
    if not outcomes:
        raise ValueError("Bank scoring requires a nonempty candidate bank")
    indexed = {outcome.candidate_id: outcome for outcome in outcomes}
    if len(indexed) != len(outcomes) or set(indexed) != set(prediction_losses):
        raise ValueError("Reference and prediction banks must have identical distinct IDs")
    if selected_candidate_id is not None and selected_candidate_id not in indexed:
        raise ValueError("Selected action is not in the declared bank")
    if any(
        loss is not None and (not np.isfinite(loss) or loss < 0)
        for loss in prediction_losses.values()
    ):
        raise ValueError("Predicted losses must be nonnegative finite values or explicit unknowns")
    valid = [outcome for outcome in outcomes if outcome.loss_radii is not None]
    oracle = min(
        (outcome.loss_radii for outcome in valid if outcome.loss_radii is not None), default=None
    )
    selected = indexed[selected_candidate_id] if selected_candidate_id is not None else None
    paired = [
        (outcome.loss_radii, prediction_losses[outcome.candidate_id])
        for outcome in valid
        if prediction_losses[outcome.candidate_id] is not None
    ]
    correlation = None
    if len(paired) >= 2:
        values = np.asarray(paired, dtype=np.float64)
        if np.ptp(values[:, 0]) > 0 and np.ptp(values[:, 1]) > 0:
            correlation = float(spearmanr(values[:, 0], values[:, 1]).statistic)
    failed = sum(outcome.failure is not None for outcome in outcomes)
    return BankSelectionScore(
        candidate_count=len(outcomes),
        failed_reference_candidates=failed,
        invalid_predictions=sum(loss is None for loss in prediction_losses.values()),
        ranked_candidate_count=len(paired),
        rank_correlation=correlation,
        selected_candidate_id=selected_candidate_id,
        selected_goal_success=selected.goal_success if selected else False,
        selected_failure=selected.failure if selected else "abstained",
        oracle_loss_radii=oracle,
        selected_regret_radii=selected.loss_radii - oracle
        if selected is not None and selected.loss_radii is not None and oracle is not None
        else None,
        feasibility="has_feasible"
        if any(outcome.goal_success for outcome in outcomes)
        else "unknown_due_to_failures"
        if failed
        else "none_in_evaluated_bank",
    )


class PairedMeasurement(Record):
    pair_id: str
    group_id: str
    a: float
    b: float


class PairedSummary(Record):
    paired_rows: int
    independent_groups: int
    weighting: Literal["equal_source_layout_groups"] = "equal_source_layout_groups"
    mean_difference_a_minus_b: float
    confidence_level: float
    confidence_interval: tuple[float, float] | None
    bootstrap_seed: int
    bootstrap_repetitions: int


def summarize_paired(
    rows: tuple[PairedMeasurement, ...],
    *,
    seed: int,
    repetitions: int = 1000,
    confidence_level: float = 0.95,
) -> PairedSummary:
    if not rows or len({row.pair_id for row in rows}) != len(rows):
        raise ValueError("Paired statistics require nonempty distinct pair IDs")
    if seed < 0 or not 100 <= repetitions <= 20000 or not 0 < confidence_level < 1:
        raise ValueError("Invalid bootstrap seed, repetitions or confidence level")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.group_id].append(row.a - row.b)
    differences = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64)
    interval = None
    if len(differences) >= 2:
        rng = np.random.default_rng(seed)
        boot = [
            float(np.mean(differences[rng.integers(0, len(differences), size=len(differences))]))
            for _ in range(repetitions)
        ]
        tail = (1 - confidence_level) / 2
        lo, hi = np.quantile(boot, (tail, 1 - tail))
        interval = float(lo), float(hi)
    return PairedSummary(
        paired_rows=len(rows),
        independent_groups=len(grouped),
        mean_difference_a_minus_b=float(np.mean(differences)),
        confidence_level=confidence_level,
        confidence_interval=interval,
        bootstrap_seed=seed,
        bootstrap_repetitions=repetitions,
    )
