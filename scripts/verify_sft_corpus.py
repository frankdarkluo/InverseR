"""Verify the frozen v2 SFT corpus, including a deterministic sample of physics replays."""

import argparse
import json
from collections import Counter
from itertools import batched
from pathlib import Path
from random import Random

from inverser.benchmark.vlm_suite import (
    FAMILIES,
    format_forward_answer,
    forward_truth,
    layout_group,
    parse_forward_answer,
    parse_goal_answer,
)
from inverser.capo.sft_corpus import SCHEMA, _exclusions, _layout_bin
from inverser.metrics import endpoint_loss, score_endpoint
from inverser.physics import SnookerSimulator
from inverser.pilot import sample_case
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.schemas import CueAction, PhysicsConfig
from inverser.study import public_physics


def verify_corpus(
    path: Path,
    suite: Path,
    pilot: Path,
    formal: Path,
    replay_pairs: int,
) -> dict:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["split"] != "train" or manifest["schema_version"] != SCHEMA:
        raise ValueError("Not a frozen v2 SFT train corpus")
    for rel in ("sft.jsonl", "groups.jsonl"):
        if file_sha256(path / rel) != manifest["artifact_sha256"][rel]:
            raise ValueError(f"Corpus hash mismatch: {rel}")
    layouts, bins, images, sources = _exclusions(suite, pilot, formal)
    if sources != manifest["exclusions"]:
        raise ValueError("Frozen source receipts have changed")
    n = manifest["independent_layout_groups"]
    if not 0 <= replay_pairs <= n:
        raise ValueError("Replay sample must fit the number of corpus pairs")
    replay_slots = set(Random(20261120).sample(range(n), replay_pairs))
    simulator = SnookerSimulator(PhysicsConfig())
    table = public_physics(simulator).table
    radius = table.ball_radius_m
    diagonal = (table.width_m**2 + table.length_m**2) ** 0.5
    seen_rows: set[str] = set()
    seen_slots: set[int] = set()
    row_counts: Counter[str] = Counter()
    with (path / "groups.jsonl").open() as groups, (path / "sft.jsonl").open() as rows:
        for line, records in zip(groups, batched(rows, 4), strict=True):
            if len(records) != 4:
                raise ValueError("Incomplete SFT group")
            meta = json.loads(line)
            slot, layout, layout_bin, digests = (
                meta["slot"],
                meta["layout_sha256"],
                meta["layout_bin"],
                meta["images"],
            )
            init = [name for name in digests if name.endswith("-init.png")]
            if (
                not 0 <= slot < n
                or slot in seen_slots
                or layout in layouts
                or layout_bin in bins
                or len(init) != 1
                or len(digests) != 5
                or digests[init[0]] in images
            ):
                raise ValueError(f"Duplicate or malformed pair: {slot}")
            if meta["endpoint_shift_radii"] < 2:
                raise ValueError(f"Twin endpoints too close: {slot}")
            for name, digest in digests.items():
                image = (path / name).resolve()
                if not image.is_relative_to(path.resolve()) or file_sha256(image) != digest:
                    raise ValueError(f"Invalid image artifact: {name}")
            seen_slots.add(slot)
            layouts.add(layout)
            bins.add(layout_bin)
            images.add(digests[init[0]])
            names_used: set[str] = set()
            decoded = [json.loads(record) for record in records]
            for i, row in enumerate(decoded):
                names = row["image_paths"]
                forward = i % 2 == 0
                role = "anchor" if i < 2 else "twin"
                stem = init[0].removesuffix("-init.png")
                completion = row["completion"][0]["content"]
                if forward:
                    if names != [f"{stem}-{role}-fwd.png"]:
                        raise ValueError(f"Incorrect forward input: {slot}")
                    parse_forward_answer(
                        completion, ball_count=meta["balls"], allow_json_fallback=False
                    )
                else:
                    prompt = json.loads(row["prompt"][1]["content"].rsplit("\n", 2)[-2])
                    if (
                        names != [init[0], f"{stem}-{role}-goal.png"]
                        or "action" in prompt
                        or prompt["task_id"] != "T4_visual_inverse"
                    ):
                        raise ValueError(f"Incorrect inverse input: {slot}")
                    parse_goal_answer(completion, allow_json_fallback=False)
                names_used.update(names)
                key = fingerprint([row["prompt"], [digests[name] for name in names]])
                if key in seen_rows:
                    raise ValueError(f"Duplicate model input: {slot}")
                seen_rows.add(key)
                row_counts["forward" if forward else "inverse"] += 1
            if names_used != set(digests):
                raise ValueError(f"Image receipt does not match model inputs: {slot}")
            if slot not in replay_slots:
                continue
            initial, _, _ = sample_case(
                manifest["seed"],
                slot,
                meta["candidate"],
                simulator.config,
                family_override=FAMILIES[slot % 4],
                ball_count=meta["balls"],
            )
            initial = initial.endpoint_view()
            if layout_group(initial) != layout or _layout_bin(initial) != layout_bin:
                raise ValueError(f"Cannot reconstruct the recorded layout: {slot}")
            commands, goals = [], []
            for forward_row, inverse_row in zip(decoded[::2], decoded[1::2], strict=True):
                prompt = json.loads(forward_row["prompt"][1]["content"].rsplit("\n", 1)[-1])
                command = CueAction.model_validate(prompt["action"])
                commands.append(command)
                replay = simulator.replay(initial, command)
                goals.append(replay.goal)
                answer = forward_row["completion"][0]["content"].split("</think>\n")[1]
                if (
                    format_forward_answer(forward_truth(initial, replay, simulator.config, table))
                    != answer
                ):
                    raise ValueError(f"Forward completion does not match replay: {slot}")
                witness = parse_goal_answer(
                    inverse_row["completion"][0]["content"], allow_json_fallback=False
                )
                if not score_endpoint(
                    simulator.replay(initial, witness.cue_action()).goal, replay.goal, radius
                ).success:
                    raise ValueError(f"Serialized inverse completion fails to reach goal: {slot}")
            altered = [
                key
                for key, value in commands[0].model_dump().items()
                if commands[1].model_dump()[key] != value
            ]
            if (
                altered != [meta["intervention"]]
                or endpoint_loss(*goals, ball_radius_m=radius, table_diagonal_m=diagonal) < 2
            ):
                raise ValueError(f"Pair is not a valid single-coordinate intervention: {slot}")
    if (
        len(seen_slots) != n
        or dict(row_counts) | {"total": len(seen_rows)} != manifest["sft_examples"]
        or manifest["png_count"] != 5 * n
    ):
        raise ValueError("Audited counts do not match the manifest")
    return {
        "status": "verified",
        "manifest_sha256": file_sha256(manifest_path),
        "independent_layout_groups": n,
        "unique_sft_inputs": len(seen_rows),
        "rows_by_task": dict(row_counts),
        "png_hashes_verified": 5 * n,
        "physics_pairs_independently_replayed": replay_pairs,
        "exclusion_overlap": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--suite", type=Path, default=Path("data/research/inverser-vlm-v1"))
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--formal", type=Path, default=Path("data/research/formal-test-v1"))
    parser.add_argument("--replay-pairs", type=int, default=256)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = verify_corpus(args.corpus, args.suite, args.pilot, args.formal, args.replay_pairs)
    if args.receipt:
        write_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2))
