"""Rank inverse proposals with internal forecasts only; the judge is not an input."""

from collections.abc import Callable, Sequence
from math import hypot

from inverser.metrics import endpoint_loss
from inverser.schemas import BallPosition, CueAction, Layout
from inverser.vlm_suite import VLMForwardOutput


def select_by_forecast(
    candidates: Sequence[CueAction],
    forecast: Callable[[CueAction], VLMForwardOutput | None],
    initial: Layout,
    target: Layout,
    *,
    ball_radius_m: float,
    table_metres: tuple[float, float],
) -> tuple[int, CueAction] | None:
    """Return original proposal index and action; ties preserve proposal order."""
    if not 1 <= len(candidates) <= 8:
        raise ValueError("Self-verification consumes at most eight inverse proposals")
    slots = initial.endpoint_view().balls
    best: tuple[float, int, CueAction] | None = None
    for index, candidate in enumerate(candidates):
        predicted = forecast(candidate)
        if predicted is None or len(predicted.resting_state) != len(slots):
            continue
        layout = Layout(
            balls=tuple(
                BallPosition(
                    ball_id=ball.ball_id,
                    kind=ball.kind,
                    status="on_table" if isinstance(point, tuple) else "pocketed",
                    xy_m=point if isinstance(point, tuple) else None,
                )
                for ball, point in zip(slots, predicted.resting_state, strict=True)
            )
        )
        loss = endpoint_loss(
            layout, target, ball_radius_m=ball_radius_m, table_diagonal_m=hypot(*table_metres)
        )
        if best is None or loss < best[0]:
            best = (loss, index, candidate)
    return (best[1], best[2]) if best else None
