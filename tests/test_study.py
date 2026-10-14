import math

import pytest
from pydantic import ValidationError

from inverser.physics import sample_event_motion
from inverser.preview import render_layout
from inverser.schemas import BallPosition, ContactEvent, CueAction, Layout
from inverser.study import GoalSubmission, make_study_case


def test_forward_view_omits_goal_witness_and_source_labels(study_fixture):
    _, record, case = study_fixture
    view = case.forward_input(record.witness_action, (0.25, 1.0))
    payload = view.model_dump(mode="json")
    assert {"goal", "witness_action", "private_record", "group_id", "family"}.isdisjoint(payload)
    assert payload["case_id"].startswith("case-")
    assert payload["physics"]["cue"]["mass_kg"] == 0.567
    assert len(payload["physics"]["table"]["pockets"]) == 6
    assert "fixture-task" not in view.model_dump_json()
    with pytest.raises(ValidationError, match="Extra inputs"):
        type(view).model_validate({**payload, "goal": record.goal.model_dump(mode="json")})


def test_hidden_physics_does_not_fill_defaults_or_reveal_actual_values(study_fixture):
    simulator, record, _ = study_fixture
    case = make_study_case(
        record, simulator, withheld_parameters=("rolling_friction", "cushion_restitution")
    )
    physics = case.goal_input().physics
    assert "rolling_friction" not in physics.parameters
    assert "cushion_restitution" not in physics.parameters
    assert set(physics.withheld_parameters) == {"rolling_friction", "cushion_restitution"}
    with pytest.raises(ValueError, match="Unknown withheld"):
        make_study_case(record, simulator, withheld_parameters=("invented_parameter",))


def test_times_and_explicit_action_fields_are_validated(study_fixture):
    _, record, case = study_fixture
    for times in ((1.0, 0.5), (0.5, 0.5), (-0.1,), (math.nan,)):
        with pytest.raises(ValidationError):
            case.forward_input(record.witness_action, times)
    action = record.witness_action.model_dump(mode="json")
    assert GoalSubmission.model_validate({"action": action}).action.elevation_deg == 0
    action.pop("elevation_deg")
    with pytest.raises(ValidationError, match="elevation_deg"):
        GoalSubmission.model_validate({"action": action})


def test_regular_labels_follow_acceleration_not_linear_interpolation(study_fixture):
    _, record, case = study_fixture
    config = record.physics
    speed = record.cue_initial_velocity_mps[1]
    sliding_time = 2 * speed / (7 * config.sliding_friction * config.gravity_mps2)
    time = sliding_time / 2
    sampled = case.sampled_truth((0.0, time, record.duration_s + 1))
    assert sampled[0].velocity_mps[1] == pytest.approx(speed)
    expected_y = 0.8 + speed * time - 0.5 * config.sliding_friction * config.gravity_mps2 * time**2
    assert sampled[1].position_m[1] == pytest.approx(expected_y, abs=1e-10)
    assert sampled[-1].position_m[:2] == pytest.approx(record.goal.balls[0].xy_m, abs=1e-10)
    assert sampled[-1].velocity_mps == (0, 0, 0)
    with pytest.raises(ValueError, match="strictly"):
        sample_event_motion(record.event_sampled_motion, config, (1, 1))


def test_metric_renderer_has_identical_x_y_scale(study_fixture):
    _, _, case = study_fixture
    image, transform = render_layout(case.goal_input().initial, case.public_physics.table)
    origin = transform.point((0, 0))
    dx = transform.point((1, 0))
    dy = transform.point((0, 1))
    assert dx[0] - origin[0] == pytest.approx(origin[1] - dy[1], abs=1e-10)
    assert max(image.size) == 1024
    unknown = Layout(balls=(BallPosition(ball_id="cue", kind="cue", status="unobserved"),))
    with pytest.raises(ValueError, match="unobserved"):
        render_layout(unknown, case.public_physics.table)
    with pytest.raises(ValueError, match="256-2048"):
        render_layout(case.goal_input().initial, case.public_physics.table, long_edge_px=20)


def test_endpoint_red_ids_remain_local():
    initial = Layout(
        balls=(
            BallPosition(ball_id="cue", kind="cue", xy_m=(0.3, 0.5)),
            BallPosition(ball_id="red_tracking_17", kind="red", xy_m=(0.9, 1.5)),
            BallPosition(ball_id="red_tracking_23", kind="red", xy_m=(0.4, 1.5)),
        )
    ).endpoint_view()
    assert [(ball.ball_id, ball.xy_m) for ball in initial.balls if ball.kind == "red"] == [
        ("red_0", (0.4, 1.5)),
        ("red_1", (0.9, 1.5)),
    ]
    with pytest.raises(ValidationError):
        CueAction(azimuth_deg=90, cue_speed_mps=0.5, tip_side=0.5, tip_vertical=0.5)


def test_semantic_events_exclude_solver_transitions_and_surface_unknowns(study_fixture):
    _, record, case = study_fixture
    assert case.semantic_contacts() == ()
    contact = ContactEvent(time_s=0.4, event_type="ball_linear_cushion", agents=("cue", "3"))
    updated = case.model_copy(
        update={"private_record": record.model_copy(update={"events": (contact,)})}
    )
    assert updated.semantic_contacts()[0].participants == ("cue", "3")
    assert updated.semantic_contacts()[0].event_type == "cushion"
    bad = ContactEvent(time_s=0.4, event_type="unverified_event", agents=("cue", "3"))
    updated = case.model_copy(
        update={"private_record": record.model_copy(update={"events": (bad,)})}
    )
    with pytest.raises(ValueError, match="Unsupported raw event"):
        updated.semantic_contacts()
