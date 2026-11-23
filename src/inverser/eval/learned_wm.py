"""GPU learned dynamics baselines on the frozen InverseR-VLM splits."""

import json
import logging
from collections.abc import Callable
from itertools import islice, permutations
from math import hypot
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from inverser.metrics import endpoint_loss, score_endpoint
from inverser.physics import SimulationFailure, SnookerSimulator
from inverser.runtime import file_sha256, write_json
from inverser.schemas import CueAction, Layout, PhysicsConfig
from inverser.study_calls import create_artifact_dir, producer_hashes, save_job
from inverser.vlm_suite import Boundary, Position, VLMForwardOutput, forward_scores

LOGGER = logging.getLogger(__name__)
BOUNDARIES: tuple[Boundary, ...] = ("LEFT", "RIGHT", "BOTTOM", "TOP")
EVENT_SLOTS = 12  # including EOS; reject training data exceeding this rather than truncating.
DINO_GRID = (252, 140)
DEFAULT_CEM = (5, 128, 16)


def action_features(actions: Tensor) -> Tensor:
    """Encode normalized azimuth/speed/tip as sin, cos, speed, side, vertical."""
    angle = actions[..., 0] * (2 * torch.pi)
    return torch.stack((angle.sin(), angle.cos(), *actions[..., 1:].unbind(-1)), dim=-1)


def encode_action(action: CueAction) -> tuple[float, float, float, float]:
    return (
        action.azimuth_deg / 360,
        action.cue_speed_mps / 3,
        action.tip_side * 2,
        (action.tip_vertical * 2),
    )


def project_actions(actions: Tensor) -> Tensor:
    angle = actions[..., :1].remainder(1)
    speed = actions[..., 1:2].clamp(0.003, 1)
    offsets = actions[..., 2:].clamp(-1, 1)
    offsets = offsets / offsets.norm(dim=-1, keepdim=True).clamp_min(1)
    return torch.cat((angle, speed, offsets), dim=-1)


def cem(
    costs: Callable[[Tensor], Tensor],
    *,
    device: str,
    seed: int,
    iterations: int = 5,
    population: int = 128,
    elite: int = 16,
) -> tuple[CueAction, float]:
    if not 0 < elite < population or iterations < 1:
        raise ValueError("Invalid CEM allocation")
    rng = torch.Generator(device=device).manual_seed(seed)
    mean = torch.tensor([0.0, 0.4, 0.0, 0.0], device=device)
    scale = torch.tensor([0.30, 0.24, 0.38, 0.38], device=device)
    floor = torch.tensor([0.004, 0.01, 0.02, 0.02], device=device)
    best: Tensor | None = None
    best_cost = float("inf")
    for iteration in range(iterations):
        points = torch.randn((population, 4), device=device, generator=rng) * scale + mean
        if iteration == 0:
            points[:, 0] = torch.rand(population, generator=rng, device=device)
        else:
            points[0] = mean
        points = project_actions(points)
        values = costs(points).float()
        if values.shape != (population,) or not bool(torch.isfinite(values).all()):
            raise ValueError("CEM needs one finite predicted cost per candidate")
        winners = values.topk(elite, largest=False).indices
        points = points[winners]
        candidate_cost = float(values[winners[0]])
        if candidate_cost < best_cost:
            best, best_cost = points[0].clone(), candidate_cost
        # Unwrap angles around the best elite, so 359° and 1° do not average to 180°.
        center = points[0, 0].clone()
        points[:, 0] = center + (points[:, 0] - center + 0.5).remainder(1) - 0.5
        mean = points.mean(dim=0)
        scale = points.std(dim=0, correction=0).maximum(floor)
    assert best is not None
    return decode_action(best), best_cost


def decode_action(point: Tensor) -> CueAction:
    angle, speed, side, vertical = map(float, point.tolist())
    # Float32 projection can overshoot the exact CueAction disk by a few ULPs.
    norm = max(1.0, hypot(side, vertical))
    return CueAction(
        azimuth_deg=angle * 360,
        cue_speed_mps=speed * 3,
        tip_side=side / (2 * norm),
        tip_vertical=vertical / (2 * norm),
    )


def transformer(width: int, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        width,
        6,
        dim_feedforward=width * 4,
        dropout=0.0,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False
    )


class LatentDynamics(nn.Module):
    """Residual six-block ViT over frozen DINOv2 patch tokens, conditioned on the shot."""

    def __init__(self, patches: int, channels: int = 384, width: int = 192) -> None:
        super().__init__()
        self.patch = nn.Linear(channels, width)
        self.action = nn.Linear(5, width)
        self.position = nn.Parameter(torch.zeros(1, patches, width))
        nn.init.normal_(self.position, std=0.02)
        self.transition = transformer(width, 6)
        self.delta = nn.Linear(width, channels)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, patches: Tensor, commands: Tensor) -> Tensor:
        x = self.patch(patches) + self.position + self.action(commands).unsqueeze(1)
        return patches + self.delta(self.transition(x))


class StateDynamics(nn.Module):
    """Object tokens for all six horizons with pocket and ordered contact predictions."""

    def __init__(self, width: int = 96) -> None:
        super().__init__()
        self.ball = nn.Linear(3, width)
        self.action = nn.Linear(5, width)
        self.slot = nn.Parameter(torch.randn(1, 4, width) * 0.02)
        self.transition = transformer(width, 4)
        self.frames = nn.Linear(width, 6 * 3)
        self.collision = nn.Linear(width, 1)
        self.first = nn.Linear(width, 4)
        self.wall = nn.Linear(width, EVENT_SLOTS * 5)

    def forward(self, balls: Tensor, commands: Tensor, present: Tensor) -> dict[str, Tensor]:
        h = self.ball(balls) + self.slot + self.action(commands).unsqueeze(1)
        h = self.transition(h, src_key_padding_mask=~present)
        frames = self.frames(h).reshape(-1, 4, 6, 3).transpose(1, 2)
        pooled = (h * present.unsqueeze(-1)).sum(dim=1) / present.sum(dim=1, keepdim=True)
        first = self.first(pooled).masked_fill(~present, -1e4)
        return {
            "xy": frames[..., :2].sigmoid(),
            "pocket": frames[..., 2],
            "collision": self.collision(h[:, 1:]).squeeze(-1),
            "first": first,
            "walls": self.wall(pooled).reshape(-1, EVENT_SLOTS, 5),
        }


def state_inputs(layout: Layout, table: tuple[float, float]) -> tuple[Tensor, Tensor]:
    balls = torch.zeros((4, 3))
    for index, ball in enumerate(layout.endpoint_view().balls):
        if ball.xy_m is None:
            raise ValueError("Initial state must be fully observed on the table")
        balls[index] = torch.tensor([ball.xy_m[0] / table[0], ball.xy_m[1] / table[1], index == 0])
    return balls, torch.arange(4) < len(layout.balls)


def state_targets(truth: VLMForwardOutput, table: tuple[float, float]) -> dict[str, Tensor]:
    xy = torch.zeros((6, 4, 2))
    pocket = torch.zeros((6, 4))
    for t, frame in enumerate((*truth.frames, truth.resting_state)):
        for i, point in enumerate(frame):
            if isinstance(point, tuple):
                xy[t, i] = torch.tensor([point[0] / table[0], point[1] / table[1]])
            else:
                pocket[t, i] = 1
    walls = [BOUNDARIES.index(side) + 1 for side in truth.cushion_sequence]
    if len(walls) >= EVENT_SLOTS:
        raise ValueError("Ordered wall label does not fit the baseline (including EOS)")
    return {
        "xy": xy,
        "pocket": pocket,
        "first": torch.tensor(truth.first_contact or 0),
        "collision": F.pad(
            torch.tensor(truth.object_collisions, dtype=torch.float32),
            (0, 3 - len(truth.object_collisions)),
        ),
        "walls": torch.tensor(walls + [0] + [-100] * (EVENT_SLOTS - len(walls) - 1)),
    }


def state_loss(prediction: dict[str, Tensor], target: dict[str, Tensor], present: Tensor) -> Tensor:
    mask = present[:, None].expand_as(target["pocket"])
    live = mask & (target["pocket"] == 0)
    coord = F.smooth_l1_loss(prediction["xy"][live], target["xy"][live], beta=0.02)
    pocket = F.binary_cross_entropy_with_logits(prediction["pocket"][mask], target["pocket"][mask])
    collision = F.binary_cross_entropy_with_logits(
        prediction["collision"], target["collision"], reduction="none"
    )
    collision = (collision * present[:, 1:]).sum() / present[:, 1:].sum().clamp_min(1)
    return (
        12 * coord
        + pocket
        + collision
        + F.cross_entropy(prediction["first"], target["first"])
        + F.cross_entropy(prediction["walls"].transpose(1, 2), target["walls"])
    )


def state_prediction(
    result: dict[str, Tensor], table: tuple[float, float], count: int
) -> VLMForwardOutput:
    collision = (result["collision"][0, : count - 1] > 0).tolist()
    first = int(result["first"][0, :count].argmax())
    if first:
        collision[first - 1] = True
    walls: list[Boundary] = []
    for label in result["walls"][0].argmax(dim=-1).tolist():
        if not label:
            break
        walls.append(BOUNDARIES[label - 1])
    xy = result["xy"][0, :, :count].tolist()
    pocket = (result["pocket"][0, :, :count] > 0).tolist()
    frames: tuple[tuple[Position, ...], ...] = tuple(
        tuple(
            "pocketed" if flag else (point[0] * table[0], point[1] * table[1])
            for point, flag in zip(points, flags, strict=True)
        )
        for points, flags in zip(xy, pocket, strict=True)
    )
    return VLMForwardOutput(
        object_collisions=tuple(collision),
        first_contact=first or None,
        cushion_sequence=tuple(walls),
        cushion_set=tuple(dict.fromkeys(walls)),
        frames=frames[:5],
        resting_state=frames[5],
    )


def state_cost(result: dict[str, Tensor], goal: Layout, config: PhysicsConfig) -> Tensor:
    """Match exchangeable reds; goal coordinates never enter the dynamics net."""
    slots = goal.endpoint_view().balls
    if slots[0].kind != "cue" or any(ball.kind != "red" for ball in slots[1:]):
        raise ValueError("CEM cost expects the frozen snooker cue/red inventory")
    xy = result["xy"][:, -1, : len(slots)] * result["xy"].new_tensor(
        [config.table_width_m, config.table_length_m]
    )
    pocket = result["pocket"][:, -1, : len(slots)].sigmoid()
    goal_xy = xy.new_tensor([ball.xy_m or (0, 0) for ball in slots])
    on = xy.new_tensor([ball.xy_m is not None for ball in slots])
    diagonal = hypot(config.table_width_m, config.table_length_m)
    pair = (
        torch.linalg.vector_norm(xy[:, :, None] - goal_xy, dim=-1) * on
        + (pocket[:, :, None] - (1 - on)).abs() * diagonal
    )
    best = torch.stack(
        [
            pair[:, range(len(slots)), (0, *order)].amax(1)
            for order in permutations(range(1, len(slots)))
        ]
    ).amin(0)
    return best + diagonal * pocket[:, 0]


def read_split(split: Path, expected: str, limit: int | None = None) -> tuple[dict, list[dict]]:
    if limit is not None and limit <= 0:
        raise ValueError("The scene limit must be positive")
    manifest = json.loads((split / "manifest.json").read_text())
    if manifest["split"] != expected:
        raise ValueError(f"Expected {expected}, refusing {manifest['split']} data")
    paths = [split / name for name in ("public/scenes.jsonl", "private/labels.jsonl")]
    for path in paths:
        if file_sha256(path) != manifest["artifact_sha256"][str(path.relative_to(split))]:
            raise ValueError("Frozen split row hash mismatch")
    rows = []
    with paths[0].open() as public, paths[1].open() as private:
        for p, q in islice(zip(public, private, strict=True), limit):
            visible, label = json.loads(p), json.loads(q)
            if visible["scene_id"] != label["scene_id"] or label["regime"] != "standard":
                raise ValueError("Expected matched, standard-physics data")
            rows.append({"public": visible, "label": label})
    if not rows:
        raise ValueError("Empty split")
    return manifest, rows


class PatchImages(Dataset):
    def __init__(self, split: Path, names: list[str], hashes: dict[str, str]) -> None:
        self.split, self.names, self.hashes = split, names, hashes

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Tensor:
        key = "public/" + self.names[index]
        path = self.split / key
        if not path.resolve().is_relative_to((self.split / "public").resolve()):
            raise ValueError("PNG path leaves the frozen public split")
        if file_sha256(path) != self.hashes[key]:
            raise ValueError(f"Frozen image hash mismatch: {key}")
        with Image.open(path) as image:
            # Preserve the entire table: resize directly to the patch grid, never center-crop.
            array = np.array(image.convert("RGB").resize(DINO_GRID[::-1], Image.Resampling.BICUBIC))
        return torch.from_numpy(array).permute(2, 0, 1)


@torch.no_grad()
def encode_patches(
    split: Path,
    rows: list[dict],
    hashes: dict[str, str],
    model_path: Path,
    device: str,
) -> tuple[Tensor, Tensor]:
    from transformers import AutoModel

    pairs = [
        (row["public"]["goals"][0]["initial_image"], row["public"]["goals"][0]["goal_image"])
        for row in rows
    ]
    indexed = {
        name: index
        for index, name in enumerate(dict.fromkeys(name for pair in pairs for name in pair))
    }
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    encoder.requires_grad_(False)
    processor = json.loads((model_path / "preprocessor_config.json").read_text())
    mean = torch.tensor(processor["image_mean"], device=device)[None, :, None, None]
    std = torch.tensor(processor["image_std"], device=device)[None, :, None, None]
    loader = DataLoader(
        PatchImages(split, list(indexed), hashes),
        batch_size=64,
        num_workers=4,
        pin_memory=True,
    )
    cache: Tensor | None = None
    cursor = 0
    for i, pixels in enumerate(loader):
        if i % 50 == 0:
            LOGGER.info("DINO patches: %d / %d images", cursor, len(indexed))
        x = (pixels.to(device, non_blocking=True).float() / 255 - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = encoder(pixel_values=x).last_hidden_state[:, 1:].to(torch.float16).cpu()
        if cache is None:
            cache = torch.empty((len(indexed), *tokens.shape[1:]), dtype=tokens.dtype)
        cache[cursor : cursor + len(tokens)] = tokens
        cursor += len(tokens)
    assert cache is not None
    mapped = torch.tensor([[indexed[a], indexed[b]] for a, b in pairs])
    return cache, mapped


def _valid(
    method: str,
    model: nn.Module,
    split: Path,
    output: Path,
    manifest: dict,
    rows: list[dict],
    config: PhysicsConfig,
    model_path: Path,
    device: str,
    seed: int,
    run_hash: str,
) -> dict[str, Any]:
    outcomes = []
    if method == "dino":
        features, mapped = encode_patches(
            split, rows, manifest["artifact_sha256"], model_path, device
        )
    simulator = SnookerSimulator(config)
    table = config.table_width_m, config.table_length_m
    model.eval()
    for index, row in enumerate(rows):
        label = row["label"]
        info = {key: label[key] for key in ("scene_id", "group_id", "family")}
        with torch.no_grad():
            if method == "state":
                initial = Layout.model_validate(label["initial"])
                goal = Layout.model_validate(label["goal"])
                balls, present = (value.to(device) for value in state_inputs(initial, table))
                recorded = torch.tensor(
                    [encode_action(CueAction.model_validate(label["action"]))], device=device
                )
                result = model(balls[None], action_features(recorded), present[None])
                truth = VLMForwardOutput.model_validate(label["truth"])
                pred = state_prediction(result, table, len(initial.balls))
                fwd = info | {
                    "task": "forward",
                    "valid": True,
                    "scores": forward_scores(pred, truth, config.ball_radius_m),
                    "true_object_collision": any(truth.object_collisions),
                    "predicted_no_object_collision": not any(pred.object_collisions),
                    "prediction": pred.model_dump(mode="json"),
                }
                save_job(output / "jobs" / f"{index:04d}-forward.json", run_hash, fwd)

                def cost(commands: Tensor) -> Tensor:
                    n = len(commands)
                    pred = model(
                        balls.expand(n, -1, -1), action_features(commands), present.expand(n, -1)
                    )
                    return state_cost(pred, goal, config)
            else:
                start, target = features[mapped[index]].to(device=device, dtype=torch.float32)

                def cost(commands: Tensor) -> Tensor:
                    pred = model(start.expand(len(commands), -1, -1), action_features(commands))
                    return (pred - target).square().mean(dim=(1, 2))

            action, predicted_cost = cem(cost, device=device, seed=seed + index)

        # DINO's selector reads only image features; structured truth is first used for judging.
        initial = Layout.model_validate(label["initial"])
        goal = Layout.model_validate(label["goal"])
        try:
            replay = simulator.replay(initial, action)
            judged = score_endpoint(replay.goal, goal, config.ball_radius_m)
            dist_r = endpoint_loss(
                replay.goal,
                goal,
                ball_radius_m=config.ball_radius_m,
                table_diagonal_m=hypot(*table),
            )
            success, failure = judged.success, judged.failure
        except SimulationFailure:
            success, failure, dist_r = False, "simulation_failure", None
        result_row = info | {
            "task": "T4",
            "success": success,
            "failure": failure,
            "loss_radii": dist_r,
            "action": action.model_dump(mode="json"),
            "predicted_cost": predicted_cost,
            "planner_simulator_calls": 0,
            "judge_simulator_calls": 1,
        }
        save_job(output / "jobs" / f"{index:04d}-T4.json", run_hash, result_row)
        outcomes.append(result_row)
    return {
        "successes": sum(row["success"] for row in outcomes),
        "scenes": len(rows),
        "judge_simulator_calls": simulator.rollouts,
        "planner_simulator_calls": 0,
    }


def train_world_model(
    method: str,
    train: Path,
    valid: Path,
    output: Path,
    dino_path: Path,
    *,
    epochs: int,
    seed: int,
    limit: int | None = None,
    valid_limit: int | None = None,
) -> dict[str, Any]:
    if method not in ("dino", "state") or epochs < 1 or not torch.cuda.is_available():
        raise ValueError(
            "A recognized learned baseline, positive epoch count and CUDA are required"
        )
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda:0"
    train_manifest, rows = read_split(train, "train", limit)
    valid_manifest, val_rows = read_split(valid, "valid", valid_limit)
    config = PhysicsConfig()
    table = config.table_width_m, config.table_length_m
    path = output / "manifest.json"
    create_artifact_dir(output, 400_000_000)
    encoded = torch.tensor(
        [encode_action(CueAction.model_validate(row["label"]["action"])) for row in rows],
        device=device,
    )
    protocol = {
        "schema_version": "learned-wm-v1",
        "baseline": method,
        "epochs": epochs,
        "seed": seed,
        "train_manifest_sha256": file_sha256(train / "manifest.json"),
        "valid_manifest_sha256": file_sha256(valid / "manifest.json"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "train_scenes": len(rows),
        "valid_scenes": len(val_rows),
        "dino_encoder": (
            {
                name: file_sha256(dino_path / name)
                for name in ("config.json", "model.safetensors", "preprocessor_config.json")
            }
            if method == "dino"
            else None
        ),
        "cem_iterations_population_elite": DEFAULT_CEM,
        "producer": producer_hashes("learned_wm.py", "vlm_suite.py", "physics.py", "metrics.py"),
    }
    from inverser.runtime import fingerprint

    run_hash = fingerprint(protocol)
    write_json(path, protocol | {"run_hash": run_hash})
    if method == "dino":
        cache, mapped = encode_patches(
            train, rows, train_manifest["artifact_sha256"], dino_path, device
        )
        model: nn.Module = LatentDynamics(cache.shape[1], cache.shape[2]).to(device)
        batch_size = 48
    else:
        states, targets = [], []
        for row in rows:
            label = row["label"]
            states.append(state_inputs(Layout.model_validate(label["initial"]), table))
            targets.append(state_targets(VLMForwardOutput.model_validate(label["truth"]), table))
        balls = torch.stack([state[0] for state in states]).to(device)
        present = torch.stack([state[1] for state in states]).to(device)
        all_targets = {
            key: torch.stack([row[key] for row in targets]).to(device) for key in targets[0]
        }
        model = StateDynamics().to(device)
        batch_size = 128
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    history = []
    steps = 0
    for epoch in range(epochs):
        batch_losses = []
        for indices in torch.randperm(len(rows)).split(batch_size):
            ii = indices.to(device)
            commands = action_features(encoded[ii])
            optimizer.zero_grad(set_to_none=True)
            if method == "dino":
                x, y = (
                    cache[mapped[indices, col]].to(device=device, dtype=torch.float32)
                    for col in (0, 1)
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = F.mse_loss(model(x, commands), y)
            else:
                loss = state_loss(
                    model(balls[ii], commands, present[ii]),
                    {key: value[ii] for key, value in all_targets.items()},
                    present[ii],
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            batch_losses.append(float(loss.detach()))
        history.append(sum(batch_losses) / len(batch_losses))
        LOGGER.info("%s: epoch %d/%d train_loss=%.6f", method, epoch + 1, epochs, history[-1])
        torch.save(model.state_dict(), output / "weights.pt")
        write_json(output / "training.json", {"steps": steps, "epoch_losses": history})
    if method == "dino":
        del cache
    result = _valid(
        method,
        model,
        valid,
        output,
        valid_manifest,
        val_rows,
        config,
        dino_path,
        device,
        seed,
        run_hash,
    )
    report = {
        "status": "complete",
        "run_hash": run_hash,
        "steps": steps,
        "epoch_losses": history,
        "weights_sha256": file_sha256(output / "weights.pt"),
        "jobs": len(val_rows) * (2 if method == "state" else 1),
        "counts": {"model_requests": 0},
        "usage": {},
        "valid": result,
    }
    write_json(output / "summary.json", report)
    return report


ORACLE_REFINE_EVALS = 1500
ORACLE_RESTARTS = 4


def _oracle_scene(item: tuple[int, dict, int, int, int]) -> list[dict[str, Any]]:
    """Ground-truth state + true Pooltool: CEM, then local Nelder-Mead refinement (upper bound)."""
    from scipy.optimize import minimize

    from inverser.study import public_physics
    from inverser.vlm_suite import forward_truth

    torch.set_num_threads(1)
    index, label, seed, refine_evals, max_attempts = item
    config = PhysicsConfig()
    simulator = SnookerSimulator(config)
    initial, goal = Layout.model_validate(label["initial"]), Layout.model_validate(label["goal"])
    diagonal = hypot(config.table_width_m, config.table_length_m)
    info = {key: label[key] for key in ("scene_id", "group_id", "family")}
    truth = VLMForwardOutput.model_validate(label["truth"])
    witness = simulator.replay(initial, CueAction.model_validate(label["action"]))
    replayed = forward_truth(initial, witness, config, public_physics(simulator).table)
    forward = info | {
        "task": "forward",
        "valid": True,
        "scores": forward_scores(replayed, truth, config.ball_radius_m),
        "true_object_collision": any(truth.object_collisions),
        "predicted_no_object_collision": not any(replayed.object_collisions),
    }

    def loss(point: Tensor) -> float:
        try:
            replay = simulator.replay(initial, decode_action(point))
            value = endpoint_loss(
                replay.goal, goal, ball_radius_m=config.ball_radius_m, table_diagonal_m=diagonal
            )
        except SimulationFailure:
            value = float("inf")
        return min(value, 1e6)

    def cost(points: Tensor) -> Tensor:
        return torch.tensor([loss(point) for point in points])

    def project(x: np.ndarray) -> Tensor:
        return project_actions(torch.tensor(x, dtype=torch.float64))

    def plan(cem_seed: int) -> tuple[CueAction, float]:
        action, value = cem(cost, device="cpu", seed=cem_seed)
        if not refine_evals:
            return action, value
        # CEM's 640 samples cannot resolve 1r on a 3.6 m table; polish locally with Pooltool.
        start = np.array(encode_action(action))
        simplex = start + np.vstack([np.zeros(4), np.diag([0.003, 0.02, 0.1, 0.1])])
        result = minimize(
            lambda x: loss(project(x)),
            start,
            method="Nelder-Mead",
            options={"initial_simplex": simplex, "maxfev": refine_evals, "fatol": 1e-3},
        )
        return decode_action(project(result.x)), float(result.fun)

    # Internal restarts are planning only; each executed attempt still fires a single shot.
    restarts = ORACLE_RESTARTS if refine_evals else 1
    attempts = []
    for attempt in range(max_attempts):  # the scene's T5 cap: 6 in v1, 3 in inverser-v3
        # Attempt 0 uses the learned-WM CEM seed, so the T4 rows differ only in the dynamics.
        action, best = plan(seed + index + 1000 * attempt)
        for restart in range(1, restarts):
            if best <= 1:  # endpoint_loss <= 1 radius is exactly a non-scratch success
                break
            candidate, value = plan(seed + index + 1000 * attempt + 100_000 * restart)
            if value < best:
                action, best = candidate, value
        replay = simulator.replay(initial, action)
        judged = score_endpoint(replay.goal, goal, config.ball_radius_m)
        attempts.append(
            {
                "ordinal": attempt + 1,
                "action": action.model_dump(mode="json"),
                "success": judged.success,
                "failure": judged.failure,
                "loss_radii": endpoint_loss(
                    replay.goal, goal, ball_radius_m=config.ball_radius_m, table_diagonal_m=diagonal
                ),
            }
        )
        if judged.success:
            break
    goal_rows = [
        info
        | {"task": task, "ood": False, "success": shots[-1]["success"], "attempts": shots}
        | {"max_attempts": cap}
        for task, shots, cap in (("T4", attempts[:1], 1), ("T5", attempts, max_attempts))
    ]
    return [forward, *goal_rows, {"simulator_rollouts": simulator.rollouts}]


def run_oracle_cem(
    valid: Path,
    output: Path,
    *,
    seed: int,
    workers: int,
    refine_evals: int = ORACLE_REFINE_EVALS,
    count: int | None = None,
) -> dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor

    from inverser.runtime import fingerprint

    _, rows = read_split(valid, "valid", count)
    caps = {row["public"]["goals"][-1]["max_attempts"] for row in rows}
    if len(caps) != 1:
        raise ValueError("One oracle run needs a single T5 attempt cap")
    create_artifact_dir(output, 50_000_000)
    protocol = {
        "schema_version": "oracle-cem-v1",
        "baseline": "ground-truth state + Pooltool cost, same CEM as learned WMs",
        "seed": seed,
        "attempts": caps.pop(),
        **({"count": count} if count is not None else {}),
        "valid_manifest_sha256": file_sha256(valid / "manifest.json"),
        "cem_iterations_population_elite": DEFAULT_CEM,
        # Omitted at 0 so the matched-budget diagnostic run keeps its original manifest.
        **(
            {"refine_nelder_mead_evals": refine_evals, "restarts_per_attempt": ORACLE_RESTARTS}
            if refine_evals
            else {}
        ),
    }
    run_hash = fingerprint(protocol)
    write_json(output / "manifest.json", protocol | {"run_hash": run_hash})
    cap = protocol["attempts"]
    items = [(i, r["label"], seed, refine_evals, cap) for i, r in enumerate(rows)]
    with ProcessPoolExecutor(workers) as pool:
        scenes = list(pool.map(_oracle_scene, items))
    rollouts = 0
    for index, scene in enumerate(scenes):
        *jobs, counter = scene
        rollouts += counter["simulator_rollouts"]
        for job in jobs:
            save_job(output / "jobs" / f"{index:04d}-{job['task']}.json", run_hash, job)
    report = {
        "status": "complete",
        "run_hash": run_hash,
        "jobs": 3 * len(rows),
        "counts": {"model_requests": 0},
        "usage": {},
        "simulator_rollouts": rollouts,
    }
    write_json(output / "summary.json", report)
    return report
