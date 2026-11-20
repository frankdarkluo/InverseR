from dataclasses import dataclass
from math import ceil, cos, radians, sin
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

from inverser.schemas import CueAction, Layout, TaskInput

if TYPE_CHECKING:
    from inverser.study import TableGeometry

COLORS = {
    "cue": "white",
    "red": "#da3030",
    "yellow": "#eeee44",
    "green": "#30a050",
    "brown": "#91512d",
    "blue": "#4080e0",
    "pink": "#f0a0c0",
    "black": "#111111",
}


@dataclass(frozen=True)
class MetricProjection:
    pixels_per_m: float
    left_px: float
    top_px: float
    maximum_y_m: float

    def point(self, xy_m: tuple[float, float]) -> tuple[float, float]:
        return (
            self.left_px + xy_m[0] * self.pixels_per_m,
            self.top_px + (self.maximum_y_m - xy_m[1]) * self.pixels_per_m,
        )


def render_layout(
    layout: Layout, geometry: "TableGeometry", *, long_edge_px: int = 1024
) -> tuple[Image.Image, MetricProjection]:
    if not 256 <= long_edge_px <= 2048:
        raise ValueError("Controlled BEV long edge must be 256-2048 pixels")
    if any(ball.status == "unobserved" for ball in layout.balls):
        raise ValueError("Controlled renderer cannot silently omit unobserved balls")
    margin = 40
    padding_m = max(circle.radius_m for circle in geometry.circular_cushions) * 2
    width_m = geometry.width_m + 2 * padding_m
    height_m = geometry.length_m + 2 * padding_m
    scale = (long_edge_px - 2 * margin) / max(width_m, height_m)
    size = (ceil(width_m * scale) + 2 * margin, ceil(height_m * scale) + 2 * margin)
    image = Image.new("RGB", size, "#eeeeee")
    draw = ImageDraw.Draw(image)
    projection = MetricProjection(
        scale, margin + padding_m * scale, margin, geometry.length_m + padding_m
    )
    lower = projection.point((0, geometry.length_m))
    upper = projection.point((geometry.width_m, 0))
    draw.rectangle((*lower, *upper), fill="#226b47")
    for cushion in geometry.linear_cushions:
        draw.line(
            (projection.point(cushion.start_m), projection.point(cushion.end_m)),
            fill="#735235",
            width=max(1, round(scale * 0.012)),
        )
    for circle in geometry.circular_cushions:
        x, y = projection.point(circle.xy_m)
        radius = circle.radius_m * scale
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="#735235")
    for pocket in geometry.pockets:
        x, y = projection.point(pocket.xy_m)
        radius = pocket.radius_m * scale
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#111111")
    radius = geometry.ball_radius_m * scale
    for ball in layout.balls:
        if ball.xy_m is None:
            continue
        x, y = projection.point(ball.xy_m)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=COLORS[ball.kind],
            outline="#202020",
        )
    draw.text(
        (margin, 8),
        f"x right / y up; table {geometry.width_m:g} x {geometry.length_m:g} m",
        fill="#202020",
    )
    return image, projection


def _render_vlm_table(
    layout: Layout, geometry: "TableGeometry", *, label: str, action: CueAction | None = None
) -> Image.Image:
    image, projection = render_layout(layout, geometry)
    draw = ImageDraw.Draw(image)
    width, length = geometry.width_m, geometry.length_m
    left, top = projection.point((0, length))
    right, bottom = projection.point((width, 0))
    font = ImageFont.load_default(size=14)
    side_y = top + 0.36 * (bottom - top)
    for (x, y), name, anchor in (
        (((left + right) / 2, top - 10), "TOP", "mm"),
        (((left + right) / 2, bottom + 26), "BOTTOM", "mm"),
        ((left - 14, side_y), "LEFT", "rm"),
        ((right + 14, side_y), "RIGHT", "lm"),
    ):
        draw.text((x, y), name, anchor=anchor, font=font, fill="#101010")
    draw.text((left, image.height - 19), label, font=font, fill="#101010")
    for maximum, horizontal in ((width, True), (length, False)):
        for index in range(int(maximum * 2) + 1):
            tick = index / 2
            x, y = projection.point((tick, 0) if horizontal else (0, tick))
            if horizontal:
                draw.line((x, y + 1, x, y + 5), fill="#101010")
                draw.text((x, y + 7), f"{tick:g}", anchor="mt", fill="#101010")
            else:
                draw.line((x - 5, y, x - 1, y), fill="#101010")
                draw.text((x - 7, y), f"{tick:g}", anchor="rm", fill="#101010")
    if action is not None:
        cue = next(ball for ball in layout.balls if ball.kind == "cue").xy_m
        if cue is None:
            raise ValueError("Forward cue ray requires an on-table cue ball")
        # A fixed 0.16 seconds-per-ray scale encodes cue-stick speed, not post-impact speed.
        angle = radians(action.azimuth_deg)
        dx, dy = 0.16 * action.cue_speed_mps * cos(angle), 0.16 * action.cue_speed_mps * sin(angle)
        start = projection.point(cue)
        tip = projection.point((cue[0] + dx, cue[1] + dy))
        draw.line((start, tip), fill="#63eeff", width=3)
        bx, by = tip
        draw.polygon(
            [
                tip,
                (bx - 7 * cos(angle - 0.48), by + 7 * sin(angle - 0.48)),
                (bx - 7 * cos(angle + 0.48), by + 7 * sin(angle + 0.48)),
            ],
            fill="#63eeff",
        )
        cx, cy = right - 22, 25
        draw.rectangle((cx - 24, cy - 19, cx + 24, cy + 19), fill="#17382e")
        draw.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), outline="white")
        draw.line((cx - 13, cy, cx + 13, cy), fill="#849d96")
        draw.line((cx, cy - 13, cx, cy + 13), fill="#849d96")
        u, v = cx + 26 * action.tip_side, cy - 26 * action.tip_vertical
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill="#63eeff")
        draw.text(
            (left, 29),
            f"{action.cue_speed_mps:.3f} m/s   tip "
            f"({action.tip_side:+.3f}, {action.tip_vertical:+.3f})",
            font=font,
            fill="#101010",
        )
    for index, ball in enumerate(layout.balls):
        if ball.xy_m is not None:
            draw.text(
                projection.point(ball.xy_m),
                str(index),
                anchor="mm",
                fill="#0b1511" if ball.kind == "cue" else "white",
            )
    return image


def render_vlm_forward_scene(
    initial: Layout, geometry: "TableGeometry", action: CueAction
) -> Image.Image:
    return _render_vlm_table(initial, geometry, label="INITIAL | ball 0 = cue", action=action)


def render_vlm_goal_pair(
    initial: Layout, goal: Layout, geometry: "TableGeometry"
) -> tuple[Image.Image, Image.Image]:
    return (
        _render_vlm_table(initial, geometry, label="INITIAL | ball 0 = cue"),
        _render_vlm_table(goal, geometry, label="GOAL | object-ball numbers are local"),
    )


def render_preview(public_path: Path, output_path: Path, count: int = 4) -> None:
    if not 1 <= count <= 8:
        raise ValueError("A preview contains 1-8 tasks")
    tasks = []
    with public_path.open() as stream:
        for _ in range(count):
            line = stream.readline()
            if not line:
                break
            tasks.append(TaskInput.model_validate_json(line))
    if not tasks:
        raise ValueError("No public task inputs to render")
    panel_w, panel_h, margin = 260, 510, 24
    image = Image.new("RGB", (2 * panel_w, len(tasks) * panel_h), "#eeeeee")
    draw = ImageDraw.Draw(image)
    for row, task in enumerate(tasks):
        for column, (label, layout) in enumerate((("Initial", task.initial), ("Goal", task.goal))):
            left, top = column * panel_w + margin, row * panel_h + 38
            width, height = panel_w - 2 * margin, panel_h - 70
            draw.text((left, row * panel_h + 12), f"{task.task_id} / {label}", fill="#202020")
            draw.rectangle((left, top, left + width, top + height), fill="#226b47")
            for px, py in ((0, 0), (1, 0), (0, 0.5), (1, 0.5), (0, 1), (1, 1)):
                pocket_x, pocket_y = left + px * width, top + py * height
                draw.ellipse(
                    (pocket_x - 6, pocket_y - 6, pocket_x + 6, pocket_y + 6), fill="#111111"
                )
            radius = task.physics.ball_radius_m / task.physics.table_width_m * width
            for ball in layout.balls:
                if ball.xy_m is None:
                    continue
                x = left + ball.xy_m[0] / task.physics.table_width_m * width
                y = top + (1 - ball.xy_m[1] / task.physics.table_length_m) * height
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=COLORS[ball.kind],
                    outline="#202020",
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
