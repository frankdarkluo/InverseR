import pytest

torch = pytest.importorskip("torch")

from inverser.learned_wm import (  # noqa: E402
    LatentDynamics,
    StateDynamics,
    action_features,
    cem,
    encode_action,
    read_split,
    state_cost,
    state_inputs,
    state_loss,
    state_prediction,
    state_targets,
)
from inverser.schemas import BallPosition, Layout, PhysicsConfig  # noqa: E402
from inverser.vlm_suite import VLMForwardOutput  # noqa: E402


def test_cem_keeps_valid_controls_and_improves_circular_objective():
    def objective(actions):
        desired = actions.new_tensor([0.994, 0.6, 0.25, -0.3])
        delta = actions - desired
        delta[:, 0] = (delta[:, 0] + 0.5).remainder(1) - 0.5
        return delta.square().sum(dim=1)

    action, cost = cem(objective, device="cpu", seed=9, population=256)
    assert cost < 0.004
    assert min(action.azimuth_deg, 360 - action.azimuth_deg) < 10
    assert 0 < action.cue_speed_mps <= 3
    assert action.tip_side**2 + action.tip_vertical**2 <= 0.25
    assert cem(objective, device="cpu", seed=9, population=256) == (action, cost)


def test_state_transition_loss_and_pocket_aware_red_matching():
    layout = Layout(
        balls=(
            BallPosition(ball_id="cue", kind="cue", xy_m=(0.50, 0.50)),
            BallPosition(ball_id="r1", kind="red", xy_m=(0.60, 0.65)),
            BallPosition(ball_id="r2", kind="red", xy_m=(0.65, 0.70)),
        )
    )
    config = PhysicsConfig()
    table = config.table_width_m, config.table_length_m
    positions = ((0.60, 0.50), (0.66, 0.70), "pocketed")
    truth = VLMForwardOutput(
        object_collisions=(True, False),
        first_contact=1,
        cushion_set=("LEFT",),
        cushion_sequence=("LEFT", "LEFT"),
        frames=(positions,) * 5,
        resting_state=positions,
    )
    balls, present = state_inputs(layout, table)
    target = {key: value[None] for key, value in state_targets(truth, table).items()}
    model = StateDynamics()
    prediction = model(balls[None], action_features(torch.zeros(1, 4)), present[None])
    loss = state_loss(prediction, target, present[None])
    assert torch.isfinite(loss) and prediction["xy"].shape == (1, 6, 4, 2)
    loss.backward()
    assert model.action.weight.grad is not None
    validated = state_prediction(prediction, table, 3)
    assert len(validated.object_collisions) == 2 and len(validated.frames) == 5

    goal = Layout(
        balls=(
            BallPosition(ball_id="cue", kind="cue", xy_m=positions[0]),
            BallPosition(ball_id="r1", kind="red", xy_m=positions[1]),
            BallPosition(ball_id="r2", kind="red", status="pocketed"),
        )
    )
    perfect = {"xy": target["xy"], "pocket": target["pocket"] * 100 - 50}
    permuted = {key: value[:, :, [0, 2, 1, 3]] for key, value in perfect.items()}
    assert state_cost(permuted, goal, config).item() < 1e-6
    assert state_cost({"xy": perfect["xy"], "pocket": -perfect["pocket"]}, goal, config) > 1


def test_latent_transition_updates_six_blocks_without_goal_as_input():
    model = LatentDynamics(6, channels=24, width=48)
    assert len(model.transition.layers) == 6
    patches = torch.randn(2, 6, 24)
    adam = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pred = model(patches, torch.randn(2, 5))
    torch.testing.assert_close(pred, patches)
    pred.square().sum().backward()
    adam.step()
    assert torch.count_nonzero(model.delta.weight)


def test_training_loader_refuses_frozen_test_before_reading_any_rows(tmp_path):
    (tmp_path / "manifest.json").write_text('{"split":"test"}')
    with pytest.raises(ValueError, match="refusing test"):
        read_split(tmp_path, "train")
    with pytest.raises(ValueError, match="refusing test"):
        read_split(tmp_path, "valid")


def test_action_encoding_contains_only_four_continuous_controls():
    action, _ = cem(lambda points: points.square().sum(dim=-1), device="cpu", seed=1)
    assert action_features(torch.tensor([encode_action(action)])).shape == (1, 5)
