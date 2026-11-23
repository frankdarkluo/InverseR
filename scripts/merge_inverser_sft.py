"""Merge frozen v1 train and v2 SFT without changing either source or the v1 benchmark."""

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import batched
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

from inverser.benchmark.vlm_suite import (
    SUITE_DIR,
    format_goal_answer,
    parse_goal_answer,
    verify_vlm_suite,
)
from inverser.capo.capo_trainer import sft_examples
from inverser.capo.sft_corpus import SCHEMA, _layout_bin, _write_rows
from inverser.metrics import score_endpoint
from inverser.physics import SimulationFailure, SnookerSimulator
from inverser.runtime import check_storage, file_sha256, fingerprint, write_json
from inverser.schemas import CueAction, Layout, PhysicsConfig
from inverser.study_calls import require_data_path


def _replay_legacy(lines: tuple[str, ...]) -> dict[str, str]:
    sim = SnookerSimulator(PhysicsConfig())
    radius = sim.config.ball_radius_m
    rejected = {}
    for line in lines:
        shot = json.loads(line)
        initial = Layout.model_validate(shot["initial"])
        target = Layout.model_validate(shot["goal"])
        rendered = format_goal_answer(CueAction.model_validate(shot["action"]))
        action = parse_goal_answer(rendered, allow_json_fallback=False).cue_action()
        if score_endpoint(initial, target, radius).success:
            rejected[shot["group_id"]] = "static_goal"
            continue
        try:
            replay = sim.replay(initial, action)
        except SimulationFailure:
            rejected[shot["group_id"]] = "simulation_failure"
            continue
        if not score_endpoint(replay.goal, target, radius).success:
            rejected[shot["group_id"]] = "serialized_witness_miss"
    return rejected


def _v1_pairs(train: Path):
    rows = sft_examples(train)
    hashes = json.loads((train / "manifest.json").read_text())["artifact_sha256"]
    with (train / "private/labels.jsonl").open() as labels:
        for pair, records in zip(batched(labels, 2), batched(rows, 4), strict=True):
            anchor, twin = map(json.loads, pair)
            layout = anchor["layout_sha256"]
            if (
                len(records) != 4
                or anchor["twin_role"] != "anchor"
                or twin["twin_role"] != "twin"
                or twin["layout_sha256"] != layout
            ):
                raise ValueError(f"Unpaired v1 SFT group: {layout}")
            images = {}
            for record in records:
                paths = []
                for image in record["image_paths"]:
                    rel = Path(image).relative_to(train / "public")
                    name = f"images/legacy/{rel.relative_to('images')}"
                    paths.append(name)
                    images[name] = hashes[f"public/{rel}"]
                record["image_paths"] = paths
            meta = {
                "source": SUITE_DIR,
                "source_group_id": anchor["group_id"],
                "layout_sha256": layout,
                "layout_bin": _layout_bin(Layout.model_validate(anchor["initial"])),
                "family": anchor["family"],
                "balls": len(anchor["initial"]["balls"]),
                "intervention": anchor["twin_intervention"],
                "images": images,
            }
            yield meta, records


def merge(suite: Path, v2: Path, output: Path, pilot: Path, formal: Path, workers: int = 8) -> dict:
    require_data_path(output)
    if output.exists():
        raise FileExistsError(output)
    suite, v2 = suite.resolve(), v2.resolve()
    m2 = json.loads((v2 / "manifest.json").read_text())
    if suite.name != SUITE_DIR or (m2["schema_version"], m2["split"]) != (SCHEMA, "train"):
        raise ValueError("Expected the frozen v1 suite and v2 training corpus")
    for role, digest in m2["exclusions"]["v1_manifests"].items():
        if file_sha256(suite / role / "manifest.json") != digest:
            raise ValueError(f"Wrong v1 source: {role}")
    for name in ("sft.jsonl", "groups.jsonl"):
        if file_sha256(v2 / name) != m2["artifact_sha256"][name]:
            raise ValueError(f"Wrong v2 source: {name}")

    check_storage((v2 / "sft.jsonl").stat().st_size + 300_000_000)
    with TemporaryDirectory(prefix=".inverser-merge-", dir=output.parent) as temp:
        stage = Path(temp) / output.name
        pool = stage / "sft_pool"
        pool.mkdir(parents=True, mode=0o700)
        frozen = stage / SUITE_DIR
        copytree(suite, frozen, copy_function=os.link)
        v1_audit = verify_vlm_suite(stage, pilot, formal)
        with (
            (frozen / "train/private/labels.jsonl").open() as labels,
            ProcessPoolExecutor(max_workers=workers) as executor,
        ):
            quality_failures = {
                group: reason
                for batch in executor.map(_replay_legacy, batched(labels, 256))
                for group, reason in batch.items()
            }
        layouts, bins, initials, model_inputs = set(), set(), set(), set()
        for role in ("valid", "test"):
            split = frozen / role
            layouts.update(json.loads((split / "groups.json").read_text()))
            with (split / "private/labels.jsonl").open() as labels:
                bins.update(
                    _layout_bin(Layout.model_validate(json.loads(line)["initial"]))
                    for line in labels
                )
            hashes = json.loads((split / "manifest.json").read_text())["artifact_sha256"]
            initials.update(h for path, h in hashes.items() if path.endswith("-init.png"))
        kept: Counter[str] = Counter()
        rejected: Counter[str] = Counter()

        with (
            (pool / "groups.jsonl").open("w") as groups,
            (pool / "sft.jsonl").open("w") as rows,
            (v2 / "groups.jsonl").open() as v2_groups,
            (v2 / "sft.jsonl").open() as v2_rows,
        ):
            for source, pairs in (
                (
                    SCHEMA,
                    (
                        (json.loads(g), tuple(map(json.loads, r)))
                        for g, r in zip(v2_groups, batched(v2_rows, 4), strict=True)
                    ),
                ),
                (SUITE_DIR, _v1_pairs(frozen / "train")),
            ):
                for meta, records in pairs:
                    images = meta["images"]
                    (init,) = [h for p, h in images.items() if p.endswith("-init.png")]
                    layout, layout_bin = meta["layout_sha256"], meta["layout_bin"]
                    if (
                        len(records) != 4
                        or len(images) != 5
                        or set(images) != {p for row in records for p in row["image_paths"]}
                    ):
                        raise ValueError(f"Incomplete group: {layout}")
                    keys = {
                        fingerprint([row["prompt"], [images[p] for p in row["image_paths"]]])
                        for row in records
                    }
                    clashes = (
                        ("layout", layout in layouts),
                        ("layout_bin", layout_bin in bins),
                        ("initial_image", init in initials),
                        ("model_input", len(keys) != 4 or not keys.isdisjoint(model_inputs)),
                    )
                    reason = next((name for name, duplicate in clashes if duplicate), None)
                    if reason:
                        if source == SCHEMA:
                            raise ValueError(f"Duplicate or held-out v2 group: {layout} ({reason})")
                        rejected[reason] += 1
                        continue
                    if source == SUITE_DIR and meta["source_group_id"] in quality_failures:
                        rejected[quality_failures[meta["source_group_id"]]] += 1
                        continue
                    layouts.add(layout)
                    bins.add(layout_bin)
                    initials.add(init)
                    model_inputs.update(keys)
                    for name, digest in images.items():
                        image = pool / name
                        if not image.resolve().is_relative_to(pool.resolve()):
                            raise ValueError(f"Unsafe image path: {name}")
                        original = (
                            v2 / name
                            if source == SCHEMA
                            else frozen
                            / "train/public/images"
                            / Path(name).relative_to("images/legacy")
                        )
                        image.parent.mkdir(parents=True, exist_ok=True)
                        os.link(original, image)
                        if file_sha256(image) != digest:
                            raise ValueError(f"Corrupt image: {name}")
                    _write_rows(rows, iter(records))
                    _write_rows(groups, iter([meta | {"source": source}]))
                    kept[source] += 1

        if kept[SCHEMA] != m2["independent_layout_groups"] or (
            kept[SUITE_DIR] + sum(rejected.values()) != v1_audit["splits"]["train"]["layout_groups"]
        ):
            raise ValueError("Merge did not account for every source layout")
        n = sum(kept.values())
        manifest = {
            "schema_version": "inverser-sft-v3",
            "split": "pool",
            "purpose": "merged_source_pool_not_a_final_experiment_split",
            "independent_layout_groups": n,
            "sft_examples": {"forward": 2 * n, "inverse": 2 * n, "total": len(model_inputs)},
            "png_paths_verified": 5 * n,
            "kept_groups_by_source": dict(kept),
            "v1_groups_dropped_by_first_matching_reason": dict(rejected),
            "v1_serialized_witness_shots_checked": v1_audit["splits"]["train"]["scenes"],
            "v1_quality_failures_by_group": quality_failures,
            "source_v2_manifest_sha256": file_sha256(v2 / "manifest.json"),
            "frozen_v1": v1_audit["splits"],
            "artifact_sha256": {
                name: file_sha256(pool / name) for name in ("sft.jsonl", "groups.jsonl")
            },
        }
        write_json(pool / "manifest.json", manifest)
        stage.rename(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("suite", "v2", "output", "pilot", "formal"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    print(json.dumps(merge(**vars(parser.parse_args())), indent=2))
