"""T5 feedback without numeric positions (D2 in docs/research/capo_rsi_skill_spec.md).

The text keeps what a player sees: attempt number, success, failure type, the object slot the cue
hit first and the cushions hit in order. Positions appear only in one image that overlays the goal
and every earlier landing, rendered at one size for both RL turns and evaluation.
"""

import json

from PIL import Image, ImageDraw, ImageFont

from inverser.capo_env import GoalEpisode
from inverser.physics import Replay
from inverser.preview import _render_vlm_table, render_layout
from inverser.vlm_suite import VLMGoalInput, forward_truth

# ~3.4 px per ball radius and ~150 visual tokens per feedback image.
FEEDBACK_LONG_EDGE_PX = 512
GOAL_RING = "#ffd400"
EARLIER = "#8fd3ff"


def t5_feedback(episode: GoalEpisode, replay: Replay | None) -> tuple[str, Image.Image | None]:
    """Describe the latest attempt; ``replay`` is that attempt's simulation, if it ran."""
    attempt = episode.attempts[-1]
    first, cushions = None, ()
    if attempt.outcome is not None and replay is not None:
        events = forward_truth(episode.initial, replay, episode.simulator.config, episode.table)
        first, cushions = events.first_contact, events.cushion_sequence
    text = json.dumps(
        {
            "ordinal": attempt.ordinal,
            "success": attempt.success,
            "failure": attempt.failure,
            "first_contact": first,
            "cushions": list(cushions),
        },
        separators=(",", ":"),
    )
    return text, _history_image(episode)


def t5_turn_prompt(query: VLMGoalInput, history: list[str], with_outcome: bool) -> str:
    """The user text of one goal turn; RL rollouts and evaluation must send identical bytes."""
    if not history:
        return query.prompt() + "\nThe images in order are INITIAL and GOAL."
    prompt = query.prompt() + "\nAttempt observations in order:\n" + "\n".join(history)
    if with_outcome:
        return prompt + (
            "\nImages: INITIAL; GOAL (left), latest outcome (right). "
            "Yellow rings = GOAL; numbered blue dots = earlier landings."
        )
    return prompt + "\nImages: INITIAL, GOAL."


def feedback_pair(goal: Image.Image, outcome: Image.Image) -> Image.Image:
    """GOAL at native scale beside the latest outcome: the second image of a feedback turn."""
    image = Image.new("RGB", (goal.width + 12 + outcome.width, goal.height), "white")
    image.paste(goal, (0, 0))
    image.paste(outcome, (goal.width + 12, 0))
    return image


def _history_image(episode: GoalEpisode) -> Image.Image | None:
    seen = [attempt for attempt in episode.attempts if attempt.outcome is not None]
    if not seen:
        return None
    latest = seen[-1]
    assert latest.outcome is not None
    # Draw in the INITIAL/GOAL style at their native scale, then shrink once to the shared size.
    # preview.py is frozen by the dataset producer hash, so its renderer is reused, not changed.
    image = _render_vlm_table(
        latest.outcome,
        episode.table,
        label=f"OUTCOME {latest.ordinal} | yellow rings = GOAL | blue = earlier attempts",
    )
    _, projection = render_layout(latest.outcome, episode.table)
    draw = ImageDraw.Draw(image)
    ring = episode.table.ball_radius_m * projection.pixels_per_m + 2
    for ball in episode.target.balls:
        if ball.xy_m is not None:
            x, y = projection.point(ball.xy_m)
            draw.ellipse((x - ring, y - ring, x + ring, y + ring), outline=GOAL_RING, width=3)
    font = ImageFont.load_default(size=22)
    for attempt in seen[:-1]:
        assert attempt.outcome is not None
        for ball in attempt.outcome.balls:
            if ball.xy_m is None:
                continue
            x, y = projection.point(ball.xy_m)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=EARLIER)
            if ball.kind == "cue":
                draw.text((x + 6, y - 6), str(attempt.ordinal), font=font, fill=EARLIER)
    scale = FEEDBACK_LONG_EDGE_PX / max(image.size)
    size = (round(image.width * scale), round(image.height * scale))
    return image.resize(size, Image.Resampling.LANCZOS)
