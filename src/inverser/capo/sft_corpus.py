"""Build a train-only, simulator-verified corpus in native CAPO SFT format."""

import argparse
import json
import logging
import os
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO, Any

from inverser.benchmark.vlm_suite import (
    FAMILIES,
    VLMForwardInput,
    VLMGoalInput,
    _counterfactuals,
    _legacy_groups,
    _scratch,
    format_forward_answer,
    format_goal_answer,
    forward_truth,
    layout_group,
    parse_goal_answer,
)
from inverser.capo.capo_trainer import TAG_CAUSAL_PROMPT
from inverser.metrics import endpoint_loss, score_endpoint
from inverser.physics import Replay, SimulationFailure, SnookerSimulator
from inverser.pilot import sample_case
from inverser.preview import render_vlm_forward_scene, render_vlm_goal_pair
from inverser.runtime import check_storage, file_sha256, fingerprint, write_json
from inverser.schemas import Layout, PhysicsConfig
from inverser.study import PublicPhysics, public_physics
from inverser.study_calls import producer_hashes, require_data_path

SCHEMA = "inverser-sft-v2"
LOGGER = logging.getLogger(__name__)
_SIM: SnookerSimulator
_VIS: PublicPhysics
_OUT: Path
_SEED: int


def _init_worker(out: Path, seed: int) -> None:
    global _SIM, _VIS, _OUT, _SEED
    _SIM = SnookerSimulator(PhysicsConfig())
    _VIS = public_physics(_SIM)
    _OUT, _SEED = out, seed


def _layout_bin(layout: Layout) -> str:
    # shortcut: 5 mm same-cell filter; use geometric neighbors for a broader near-duplicate policy.
    return fingerprint(
        sorted(
            (b.kind, round(b.xy_m[0] / 0.005), round(b.xy_m[1] / 0.005))
            for b in layout.balls
            if b.xy_m is not None
        )
    )


def _contact_count(replay: Replay) -> int:
    return len(
        {
            ball
            for event in replay.events
            if event.event_type == "ball_ball"
            for ball in event.agents
            if ball != "cue"
        }
    )


def _pair(slot: int, generation: int) -> tuple[dict[str, Any], list[dict], Counter[str]]:
    family = FAMILIES[slot % 4]
    index = slot // 4
    balls = 1 if family == "free_roll" else 2 + index % 3
    tier = (
        "roll"
        if balls == 1
        else "no_contact"
        if index // 3 % 10 == 0
        else "multi_contact"
        if balls >= 3 and index // 3 % 10 == 1
        else "contact"
    )
    config = _SIM.config
    radius = config.ball_radius_m
    table = _VIS.table
    diagonal = (table.width_m**2 + table.length_m**2) ** 0.5
    rejects: Counter[str] = Counter()
    for attempt in range(generation * 4_000, (generation + 1) * 4_000):
        initial, action, _ = sample_case(
            _SEED, slot, attempt, config, family_override=family, ball_count=balls
        )
        # Assign red IDs before replay in the same order displayed to the model.
        initial = initial.endpoint_view()
        try:
            anchor = _SIM.replay(initial, action)
            contacts = _contact_count(anchor)
            if _scratch(anchor):
                rejects["scratch_anchor"] += 1
                continue
            if (
                (tier == "no_contact" and contacts)
                or (tier == "contact" and not contacts)
                or (tier == "multi_contact" and contacts < 2)
            ):
                rejects["contact_stratum"] += 1
                continue
            if score_endpoint(anchor.goal, initial, radius).success:
                rejects["static_anchor"] += 1
                continue
            for field, twin_action in _counterfactuals(action, family):
                if family == "vertical" and field != "tip_vertical":
                    continue
                if family == "side" and field != "tip_side":
                    continue
                twin = _SIM.replay(initial, twin_action)
                if _scratch(twin) or score_endpoint(twin.goal, initial, radius).success:
                    continue
                separation = endpoint_loss(
                    anchor.goal, twin.goal, ball_radius_m=radius, table_diagonal_m=diagonal
                )
                if separation >= 2:
                    break
            else:
                rejects["no_eligible_twin"] += 1
                continue
            members = []
            for cmd, replay in ((action, anchor), (twin_action, twin)):
                answer = format_goal_answer(cmd)
                try:
                    parsed = parse_goal_answer(answer, allow_json_fallback=False)
                except ValueError:
                    break
                if (
                    _SIM.replay(initial, cmd) != replay
                    or not score_endpoint(
                        _SIM.replay(initial, parsed.cue_action()).goal,
                        replay.goal,
                        radius,
                    ).success
                ):
                    break
                truth = forward_truth(initial, replay, config, table)
                members.append((cmd, replay, truth, answer))
            if len(members) != 2:
                rejects["serialized_witness"] += 1
                continue
        except SimulationFailure:
            rejects["simulation_failure"] += 1
            continue
        break
    else:
        raise RuntimeError(f"No verified {family}/{tier} pair for slot {slot}: {rejects}")

    stem = f"images/{slot // 1000:03d}/{slot:06d}"
    init_name = f"{stem}-init.png"
    (_OUT / init_name).parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    image_hashes: dict[str, str] = {}
    for role, (cmd, replay, truth, answer) in zip(("anchor", "twin"), members, strict=True):
        goal_name, fwd_name = f"{stem}-{role}-goal.png", f"{stem}-{role}-fwd.png"
        initial_image, goal_image = render_vlm_goal_pair(
            initial, replay.goal.endpoint_view(), table
        )
        if role == "anchor":
            initial_image.save(_OUT / init_name)
            image_hashes[init_name] = file_sha256(_OUT / init_name)
        goal_image.save(_OUT / goal_name)
        render_vlm_forward_scene(initial, table, cmd).save(_OUT / fwd_name)
        for name in (goal_name, fwd_name):
            image_hashes[name] = file_sha256(_OUT / name)
        forward = VLMForwardInput(
            scene_id=f"{slot}-{role}",
            image=fwd_name,
            ball_count=balls,
            action=cmd,
            parameters=_VIS.parameters,
            table_metres=(table.width_m, table.length_m),
            ball_radius_m=radius,
        )
        inverse = VLMGoalInput(
            task_id="T4_visual_inverse",
            episode_id=f"{slot}-{role}",
            initial_image=init_name,
            goal_image=goal_name,
            ball_count=balls,
            parameters=_VIS.parameters,
            max_attempts=1,
            table_metres=(table.width_m, table.length_m),
            ball_radius_m=radius,
        )
        contact = "none" if truth.first_contact is None else str(truth.first_contact)
        rails = ", ".join(truth.cushion_sequence) or "none"
        pots = sum(ball.status == "pocketed" for ball in replay.goal.balls)
        rows.extend(
            (
                _row(
                    forward.prompt(),
                    [fwd_name],
                    f"The shot first contacts object slot {contact}; ordered rail contacts: "
                    f"{rails}. Track numbered positions through these events to rest.",
                    format_forward_answer(truth),
                ),
                _row(
                    inverse.prompt() + "\nImages: INITIAL, GOAL.",
                    [init_name, goal_name],
                    f"One valid shot has first cue-object contact slot {contact}, rail sequence "
                    f"{rails}, and {pots} object balls pocketed. It reaches the goal without "
                    "scratching; other actions may also work.",
                    answer,
                ),
            )
        )
    return (
        {
            "slot": slot,
            "generation": generation,
            "candidate": attempt,
            "layout_sha256": layout_group(initial),
            "layout_bin": _layout_bin(initial),
            "family": family,
            "balls": balls,
            "tier": tier,
            "intervention": field,
            "endpoint_shift_radii": separation,
            "contacted_object_counts": [_contact_count(row[1]) for row in members],
            "cushion_counts": [len(row[2].cushion_sequence) for row in members],
            "changed_contact_mask": members[0][2].object_collisions
            != members[1][2].object_collisions,
            "changed_cushion_sequence": members[0][2].cushion_sequence
            != members[1][2].cushion_sequence,
            "images": image_hashes,
        },
        rows,
        rejects,
    )


def _row(prompt: str, images: list[str], trace: str, answer: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": TAG_CAUSAL_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "completion": [{"role": "assistant", "content": f"<think>\n{trace}\n</think>\n{answer}"}],
        "image_paths": images,
    }


def _exclusions(suite: Path, pilot: Path, formal: Path) -> tuple:
    layouts, legacy = _legacy_groups(pilot, formal)
    bins: set[str] = set()
    images: set[str] = set()
    manifests = {}
    for name in ("train", "valid", "test"):
        split = suite / name
        manifest_path = split / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["legacy_exclusions"] != legacy or manifest["split"] != name:
            raise ValueError(f"{name}: exclusions do not match the v1 source hashes")
        manifests[name] = file_sha256(manifest_path)
        for rel in ("groups.json", "private/labels.jsonl"):
            if file_sha256(split / rel) != manifest["artifact_sha256"][rel]:
                raise ValueError(f"{name}: source {rel} does not match the frozen manifest")
        layouts.update(json.loads((split / "groups.json").read_text()))
        images.update(h for p, h in manifest["artifact_sha256"].items() if p.endswith("-init.png"))
        with (split / "private/labels.jsonl").open() as stream:
            for line in stream:
                bins.add(_layout_bin(Layout.model_validate(json.loads(line)["initial"])))
    return layouts, bins, images, {"v1_manifests": manifests, "legacy": legacy}


def _write_rows(stream: IO[str], rows: Iterator[dict]) -> None:
    for row in rows:
        stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def build_corpus(
    output: Path, suite: Path, pilot: Path, formal: Path, *, pairs: int, seed: int, workers: int
) -> dict:
    require_data_path(output)
    if output.exists() or pairs < 4 or pairs % 4 or seed < 0 or workers < 1:
        raise ValueError(
            "Use a new train path, a positive multiple of four pairs and valid seed/workers"
        )
    check_storage(pairs * 160_000)
    layouts, bins, images, sources = _exclusions(suite, pilot, formal)
    keys: set[str] = set()
    coverage: Counter[str] = Counter()
    rejects: Counter[str] = Counter()
    diagnostics: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        stage = Path(temp)
        with (
            (stage / "sft.jsonl").open("w") as data,
            (stage / "groups.jsonl").open("w") as groups,
            ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(stage, seed)) as pool,
        ):
            for start in range(0, pairs, 512):
                pending = list(range(start, min(start + 512, pairs)))
                for generation in range(32):
                    retry = []
                    for meta, rows, rejected in pool.map(
                        _pair, pending, [generation] * len(pending), chunksize=4
                    ):
                        rejects.update(rejected)
                        layout, layout_bin, paths = (
                            meta["layout_sha256"],
                            meta["layout_bin"],
                            meta["images"],
                        )
                        init_hash = next(iter(paths.values()))
                        row_keys = {
                            fingerprint([r["prompt"], [paths[p] for p in r["image_paths"]]])
                            for r in rows
                        }
                        if (
                            layout in layouts
                            or layout_bin in bins
                            or init_hash in images
                            or len(row_keys) != 4
                            or not row_keys.isdisjoint(keys)
                        ):
                            retry.append(meta["slot"])
                            rejects["duplicate_input_or_layout"] += 1
                            for p in paths:
                                (stage / p).unlink()
                            continue
                        layouts.add(layout)
                        bins.add(layout_bin)
                        images.add(init_hash)
                        keys.update(row_keys)
                        coverage[f"{meta['family']}/{meta['balls']}/{meta['tier']}"] += 1
                        diagnostics["pairs_with_changed_contact_mask"] += meta[
                            "changed_contact_mask"
                        ]
                        diagnostics["pairs_with_changed_cushion_sequence"] += meta[
                            "changed_cushion_sequence"
                        ]
                        diagnostics["shots_with_multiple_contacted_objects"] += sum(
                            n >= 2 for n in meta["contacted_object_counts"]
                        )
                        diagnostics["shots_with_multiple_cushion_contacts"] += sum(
                            n >= 2 for n in meta["cushion_counts"]
                        )
                        _write_rows(data, iter(rows))
                        _write_rows(groups, iter([meta]))
                    if not retry:
                        break
                    pending = retry
                else:
                    raise RuntimeError(f"Deduplication retry limit hit on slots {pending[:8]}")
                if start % 4096 == 0 or start + 512 >= pairs:
                    LOGGER.info(
                        "accepted %d/%d layout pairs; %d SFT rows", len(keys) // 4, pairs, len(keys)
                    )
        if len(keys) != pairs * 4 or sum(coverage.values()) != pairs:
            raise RuntimeError("The corpus did not reach its deduplicated quota")
        manifest = {
            "schema_version": SCHEMA,
            "split": "train",
            "seed": seed,
            "independent_layout_groups": pairs,
            "standard_scene_count": pairs * 2,
            "anchor_count": pairs,
            "twin_count": pairs,
            "witness_replays_verified": pairs * 2,
            "sft_examples": {"forward": pairs * 2, "inverse": pairs * 2, "total": pairs * 4},
            "coverage": dict(coverage),
            "rejections": dict(rejects),
            "diagnostics": dict(diagnostics),
            "exclusions": sources,
            "png_count": pairs * 5,
            "artifact_sha256": {
                name: file_sha256(stage / name) for name in ("sft.jsonl", "groups.jsonl")
            },
            "producer": producer_hashes(
                "sft_corpus.py",
                "vlm_suite.py",
                "capo_trainer.py",
                "pilot.py",
                "physics.py",
                "preview.py",
            ),
        }
        write_json(stage / "manifest.json", manifest)
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=Path("data/research/inverser-vlm-v1"))
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--formal", type=Path, default=Path("data/research/formal-test-v1"))
    parser.add_argument("--pairs", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(
        json.dumps(
            build_corpus(
                args.output,
                args.suite,
                args.pilot,
                args.formal,
                pairs=args.pairs,
                seed=args.seed,
                workers=args.workers,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
