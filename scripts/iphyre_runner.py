"""Line-delimited JSON bridge for the pinned DeepPHY I-PHYRE simulator patch."""

import base64
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402
from iphyre.games import GAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEEP_PHY = ROOT / "data/external/DeepPHY"
PATCH = DEEP_PHY / "res/02_iphyre/simulator.py"
SETTINGS = json.loads((DEEP_PHY / "conf/env_config_02_iphyre.json").read_text())


def main() -> None:
    spec = importlib.util.spec_from_file_location("deepphy_annotated_simulator", PATCH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Pinned I-PHYRE renderer patch cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "list":
            result = {"tasks": list(GAMES), "fps": SETTINGS["fps"]}
        else:
            game = request["game"]
            if game not in GAMES:
                raise ValueError("Unknown I-PHYRE task")
            env = module.IPHYRE(game, fps=SETTINGS["fps"])
            env.reset()
            blocks = [point for point in env.get_action_space()[1:] if any(point)]
            if request["op"] == "render":
                env.init_screen()
                env.screen.fill((255, 255, 255))
                env.space.debug_draw(env.draw_options)
                env._draw_action_indices()
                image = io.BytesIO()
                pygame.image.save(env.screen, image, "scene.png")
                png = image.getvalue()
                result = {
                    "png": base64.b64encode(png).decode(),
                    "png_sha256": hashlib.sha256(png).hexdigest(),
                    "eliminable_blocks": len(blocks),
                }
            elif request["op"] == "simulate":
                schedule = []
                for action in request["actions"]:
                    index, time_s = action["index"], action["time"]
                    if (
                        type(index) is not int
                        or not 0 <= index < len(blocks)
                        or not math.isfinite(time_s)
                        or not 0 <= time_s <= 15
                    ):
                        raise ValueError("Elimination index or time out of bounds")
                    schedule.append((*blocks[index], time_s or 1 / SETTINGS["fps"]))
                schedule.sort(key=lambda row: row[2])
                reward = 0.0
                index = 0
                time_s = 0.0
                while time_s < 15:
                    point = [0.0, 0.0]
                    if index < len(schedule) and time_s >= schedule[index][2]:
                        point = list(schedule[index][:2])
                        index += 1
                    _, step_reward, done = env.step(point)
                    reward += step_reward
                    time_s += 1 / SETTINGS["fps"]
                    if done:
                        break
                result = {"reward": reward, "solved": reward > SETTINGS["solved_threshold"]}
            else:
                raise ValueError("Unknown runner operation")
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
