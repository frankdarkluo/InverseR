"""DeepPHY nine-ball state transitions adapted to project-pinned Pooltool 0.6."""

import base64
import hashlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from inverser.physics import SnookerSimulator
from inverser.schemas import PhysicsConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data/external/DeepPHY/conf/env_config_04_pooltool.json"
COLORS = (
    "white",
    "yellow",
    "#598aff",
    "red",
    "#ad68df",
    "orange",
    "#60ad6a",
    "#ba5877",
    "#161616",
    "yellow",
)


def main():
    settings = json.loads(CONFIG.read_text())
    physics = SnookerSimulator(PhysicsConfig())
    pt = physics.pt
    table = pt.Table.default()
    for line in sys.stdin:
        request = json.loads(line)
        operation = request["op"]
        if operation == "list":
            result = {"tasks": [str(i) for i in range(100)], "pooltool": pt.__version__}
        else:
            game = int(request["game"])
            if not 0 <= game < 100:
                raise ValueError("Nine-ball game ID is outside the pinned 100-game protocol")
            state = request.get("state")
            if state is None:
                np.random.seed(20261119 + game)
                random.seed(20261119 + game)
                state = {
                    key: ball.xyz[:2].tolist()
                    for key, ball in pt.get_rack(pt.GameType.NINEBALL, table).items()
                }
            if "cue" not in state:
                state["cue"] = [table.w / 2, table.l / 4]
            if operation == "render":
                scale = 420.0
                margin = 30
                canvas = Image.new(
                    "RGB", (round(table.w * scale) + 60, round(table.l * scale) + 60), "#dfdfdf"
                )
                draw = ImageDraw.Draw(canvas)

                def point(x, y):
                    return margin + x * scale, margin + (table.l - y) * scale

                left, top = point(0, table.l)
                right, bottom = point(table.w, 0)
                draw.rectangle((left, top, right, bottom), fill="#086000")
                for pocket in table.pockets.values():
                    x, y = point(*pocket.center[:2])
                    r = pocket.radius * scale
                    draw.ellipse((x - r, y - r, x + r, y + r), fill="#565656")
                r = pt.BallParams().R * scale
                for key, xy in state.items():
                    x, y = point(*xy)
                    color = COLORS[0 if key == "cue" else int(key)]
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline="black")
                    draw.text(
                        (x, y),
                        key if key != "cue" else "0",
                        fill="white" if key == "8" else "black",
                        anchor="mm",
                    )
                buffer = io.BytesIO()
                canvas.save(buffer, format="PNG")
                png = buffer.getvalue()
                result = {
                    "png": base64.b64encode(png).decode(),
                    "png_sha256": hashlib.sha256(png).hexdigest(),
                    "state": state,
                }
            elif operation == "simulate":
                speeds, struck = settings["speed_mapping"], settings["strikespot_mapping"]
                speed, strike = request["speed"], request["strikespot"]
                balls = {key: pt.Ball.create(key, xy=xy) for key, xy in state.items()}
                system = pt.System(table=table, balls=balls, cue=pt.Cue(cue_ball_id="cue"))
                target = min((key for key in state if key != "cue"), key=int)
                system.cue.set_state(
                    V0=speeds[speed],
                    phi=pt.aim.at_ball(system, target),
                    **struck[strike],
                )
                shot = pt.simulate(system, max_events=1000, t_final=120)
                if len(shot.events) >= 1000 or shot.t >= 120:
                    result = {"status": "simulation_budget", "solved": False, "state": state}
                else:
                    potted = sorted(
                        {
                            event.agents[0].id
                            for event in shot.events
                            if event.event_type == pt.EventType.BALL_POCKET
                        }
                    )
                    result = {
                        "status": "9_BALL_POTTED"
                        if "9" in potted
                        else (
                            "CUE_BALL_POTTED"
                            if "cue" in potted
                            else "OTHER_BALLS_POTTED"
                            if potted
                            else "NO_BALLS_POTTED"
                        ),
                        "solved": "9" in potted,
                        "potted": potted,
                        "target_ball": target,
                        "top_speed": speed == max(speeds, key=speeds.get),
                        "state": {
                            key: ball.xyz[:2].tolist()
                            for key, ball in shot.balls.items()
                            if key not in potted
                        },
                    }
            else:
                raise ValueError("Unknown nine-ball operation")
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
