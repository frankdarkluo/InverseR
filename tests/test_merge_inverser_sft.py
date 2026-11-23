import json
import runpy
from pathlib import Path

from inverser.physics import SnookerSimulator
from inverser.schemas import BallPosition, CueAction, Layout, PhysicsConfig


def test_legacy_replay_rejects_bad_witnesses_and_static_targets():
    script = Path(__file__).resolve().parents[1] / "scripts/merge_inverser_sft.py"
    replay_legacy = runpy.run_path(str(script))["_replay_legacy"]
    sim = SnookerSimulator(PhysicsConfig())
    initial = Layout(balls=(BallPosition(ball_id="cue", kind="cue", xy_m=(0.45, 0.8)),))
    action = CueAction(azimuth_deg=90, cue_speed_mps=0.6, tip_side=0, tip_vertical=0)
    target = sim.replay(initial, action).goal

    def label(group, goal, command):
        return json.dumps(
            {
                "group_id": group,
                "initial": initial.model_dump(),
                "goal": goal.model_dump(),
                "action": command.model_dump(),
            }
        )

    wrong = action.model_copy(update={"azimuth_deg": 270.0})
    lines = (label("good", target, action), label("miss", target, wrong))
    assert replay_legacy((*lines, label("static", initial, action))) == {
        "miss": "serialized_witness_miss",
        "static": "static_goal",
    }
