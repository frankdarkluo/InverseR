import json

import numpy as np
from PIL import Image

from inverser.capo.t5_feedback import (
    FEEDBACK_LONG_EDGE_PX,
    feedback_pair,
    t5_feedback,
    t5_turn_prompt,
)
from inverser.capo_env import GoalEpisode
from inverser.physics import SnookerSimulator
from inverser.schemas import BallPosition, CueAction, Layout, PhysicsConfig
from inverser.study import public_physics
from inverser.vlm_suite import VLMGoalInput, format_goal_answer

STRAIGHT_UP = CueAction(azimuth_deg=90, cue_speed_mps=1.5, tip_side=0, tip_vertical=0)


def _episode() -> GoalEpisode:
    sim = SnookerSimulator(PhysicsConfig())
    physics = public_physics(sim)
    initial = Layout(
        balls=(
            BallPosition(ball_id="cue", kind="cue", xy_m=(0.6, 0.8)),
            BallPosition(ball_id="red_0", kind="red", xy_m=(0.6, 1.2)),
        )
    )
    task = VLMGoalInput(
        task_id="T5_interactive_adaptation",
        episode_id="case",
        initial_image="initial",
        goal_image="goal",
        ball_count=2,
        table_metres=(physics.table.width_m, physics.table.length_m),
        ball_radius_m=physics.table.ball_radius_m,
        parameters=physics.parameters,
        withheld_parameters=(),
        max_attempts=6,
    )
    goal = sim.replay(initial, STRAIGHT_UP.model_copy(update={"cue_speed_mps": 0.8})).goal
    return GoalEpisode(task, initial, goal, sim)


def _try(episode: GoalEpisode, text: str):
    captured = []

    def judge(action):
        captured.append(episode.simulator.replay(episode.initial, action))
        return captured[-1]

    episode.step(text, replay_action=judge)
    return t5_feedback(episode, captured[-1] if captured else None)


def _pixels(image, low, high) -> int:
    rgb = np.asarray(image.convert("RGB")).astype(int)
    return int(((rgb >= low) & (rgb <= high)).all(axis=-1).sum())


def test_text_has_events_but_no_positions_residual_or_loss():
    episode = _episode()
    text, image = _try(episode, format_goal_answer(STRAIGHT_UP))
    fields = json.loads(text)
    assert set(fields) == {"ordinal", "success", "failure", "first_contact", "cushions"}
    assert fields["first_contact"] == 1 and "TOP" in fields["cushions"]
    assert not fields["success"] and fields["failure"] == "position"
    assert image is not None and max(image.size) == FEEDBACK_LONG_EDGE_PX


def test_image_keeps_goal_and_earlier_landings_through_invalid_turns():
    episode = _episode()
    text, image = _try(episode, "invalid")
    assert image is None and json.loads(text)["first_contact"] is None
    _, first = _try(episode, format_goal_answer(STRAIGHT_UP))
    _, after_invalid = _try(episode, "invalid")
    aside = STRAIGHT_UP.model_copy(update={"azimuth_deg": 95})
    _, second = _try(episode, format_goal_answer(aside))
    yellow, blue = ((200, 180, 0), (255, 255, 90)), ((100, 180, 220), (190, 235, 255))
    assert first is not None and after_invalid is not None and second is not None
    assert _pixels(first, *yellow) > 0 and _pixels(first, *blue) == 0
    assert after_invalid.size == first.size == second.size
    assert _pixels(second, *blue) > 0


def test_turn_prompt_and_image_pair_are_the_evaluation_layout():
    task = _episode().task
    first = t5_turn_prompt(task, [], with_outcome=False)
    assert first == task.prompt() + "\nThe images in order are INITIAL and GOAL."
    assert "<collisions>1:true</collisions>" in first and "never a cushion name" in first
    later = t5_turn_prompt(task, ["a", "b"], with_outcome=True)
    assert later.startswith(task.prompt() + "\nAttempt observations in order:\na\nb\n")
    assert later.endswith("numbered blue dots = earlier landings.")
    assert t5_turn_prompt(task, ["a"], with_outcome=False).endswith("\nImages: INITIAL, GOAL.")
    pair = feedback_pair(Image.new("RGBA", (30, 20)), Image.new("RGB", (10, 5)))
    assert pair.mode == "RGB" and pair.size == (52, 20)
