"""Use the isolated official PHYRE simulator with the DeepPHY discrete action grid."""

import base64
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import phyre
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data/external/DeepPHY/conf/env_config_01_phyre.json"


def main():
    settings = json.loads(CONFIG.read_text())
    height, width = settings["grid_size"]
    levels = settings["radius_levels"]
    _, _, test = phyre.get_fold("ball_within_template", 0)
    for line in sys.stdin:
        request = json.loads(line)
        operation = request["op"]
        if operation == "list":
            result = {"tasks": list(test), "fold": 0, "eval_setup": "ball_within_template"}
        else:
            game = request["game"]
            if game not in test:
                raise ValueError("PHYRE task is outside the pinned test fold")
            sim = phyre.initialize_simulator((game,), "ball")
            if operation == "render":
                rgb = phyre.observations_to_uint8_rgb(sim.initial_scenes[0])
                image = Image.fromarray(rgb)
                draw = ImageDraw.Draw(image)
                for iy in range(height):
                    for ix in range(width):
                        x, y = image.width * ix // width, image.height * iy // height
                        draw.line((x, 0, x, image.height), fill="#ac5858")
                        draw.line((0, y, image.width, y), fill="#ac5858")
                        draw.text((x + 2, y + 1), str(iy * width + ix + 1), fill="#303030")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                payload = buffer.getvalue()
                result = {
                    "png": base64.b64encode(payload).decode(),
                    "png_sha256": hashlib.sha256(payload).hexdigest(),
                }
            elif operation == "simulate":
                cell, radius = request["cell"], request["radius"]
                if (
                    type(cell) is not int
                    or type(radius) is not int
                    or not (1 <= cell <= height * width and 1 <= radius <= levels)
                ):
                    raise ValueError("Invalid discrete PHYRE action")
                y, x = divmod(cell - 1, width)
                coords = ((x + 0.5) / width, 1 - (y + 0.5) / height)
                relative_r = 0.0625 + (radius - 1) / (levels - 1) * (1 - 0.0625)
                replay = sim.simulate_action(0, np.array([*coords, relative_r]), need_images=False)
                result = {
                    "status": replay.status.name,
                    "solved": bool(replay.status.is_solved()),
                    "action": [*coords, relative_r],
                }
            else:
                raise ValueError("Unknown PHYRE operation")
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
