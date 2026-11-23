"""Build the InverseR SFT / RL / valid / test splits.

Inputs:  an existing InverseR directory (sft/rl/valid/test; every label carries its initial layout,
         actions and source), the pilot / formal-test exclusion lists and, optionally, an older
         suite whose valid/test layouts are excluded and whose train layouts may only go to SFT.
         Groups whose source is the legacy train split always stay in SFT.
Outputs: <output>/{sft,rl,valid,test}/: public/scenes.jsonl, public/images/, private/labels.jsonl,
         private/physics.json, groups.json, manifest.json. sft and rl carry manifest split "train";
         scene ids are prefixed sft- / rl- to stay unique.
Default (re-split): free_roll is 1/8 of layout groups and each multi-ball family 7/24 (2/3/4 balls
equal); multi-ball groups are drawn in proportion to their anchor/twin contact type, so 60-70% of
scenes need an object ball to move; T5 allows 3 attempts; test gets newly generated unknown-friction
episodes. With --panels the memberships stay fixed and only the pre-registered prefixes change.
Every group is replayed from its labels and standard groups must reproduce the source PNGs.
"""

import argparse
import io
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from inverser.benchmark import vlm_suite
from inverser.benchmark.vlm_suite import (
    FAMILIES,
    SUITE_DIR,
    TIMES,
    WITHHELD,
    _boundary_map,
    _export_scene,
    _legacy_groups,
    _scratch,
    _terminal_shift,
    layout_group,
    requires_contact,
    scene_naming,
    verify_vlm_suite,
)
from inverser.capo.sft_corpus import _contact_count, _layout_bin
from inverser.metrics import score_endpoint
from inverser.physics import SimulationFailure, SnookerSimulator
from inverser.pilot import sample_case
from inverser.runtime import file_sha256, write_json
from inverser.schemas import CueAction, Layout, PhysicsConfig
from inverser.study import public_physics
from inverser.study_calls import require_data_path

SPLIT = {"sft": "train", "rl": "train", "valid": "valid", "test": "test"}
MULTI_PER_STRATUM = {"valid": 930, "test": 930, "rl": 1490}  # sft keeps every remaining group
PANEL_PER_STRATUM = {"valid": 120, "test": 240}  # stratified prefix of each file
OOD_PER_STRATUM = {"free_roll": 77, "multi": 60}  # per friction regime, test only
T5_ATTEMPTS = 3
CONTACT_SHARE = (0.60, 0.70)  # scenes whose goal needs an object ball to move
SUFFIXES = ("-init.png", "-anchor-goal.png", "-anchor-fwd.png", "-twin-goal.png", "-twin-fwd.png")
LEGACY = "inverser-vlm-v1"  # source.dataset of groups taken from the legacy train split
_SIMS: dict[str, SnookerSimulator]
_VIS: dict


def _free_roll(multi_per_stratum: int) -> int:
    # Three multi-ball families x three ball counts; free_roll is then 1/8 of all groups.
    return round(9 * multi_per_stratum / 7)


def _simulators() -> dict[str, SnookerSimulator]:
    # Same three regimes as vlm_suite.prepare_vlm_suite, where they are defined inline.
    config = PhysicsConfig()
    sims = {"standard": SnookerSimulator(config)}
    for name, scale, restitution in (("slow_friction", 0.5, 0.65), ("fast_friction", 2, 0.97)):
        sims[name] = SnookerSimulator(
            PhysicsConfig.model_validate(
                config.model_dump()
                | {
                    "sliding_friction": config.sliding_friction * scale,
                    "rolling_friction": config.rolling_friction * scale,
                    "cushion_restitution": restitution,
                }
            )
        )
    return sims


def _init() -> None:
    global _SIMS, _VIS
    _SIMS = _simulators()
    _VIS = {
        name: public_physics(sim, withheld_parameters=WITHHELD if name != "standard" else ())
        for name, sim in _SIMS.items()
    }


def _needs_contact(label: dict) -> bool:
    return requires_contact(
        Layout.model_validate(label["initial"]),
        Layout.model_validate(label["goal"]),
        PhysicsConfig().ball_radius_m,
    )


def _metas(root: Path) -> list[dict]:
    """Layout groups of one split in file order, with labels and (standard) source PNG hashes."""
    hashes = json.loads((root / "manifest.json").read_text())["artifact_sha256"]
    metas: list[dict] = []
    with (
        (root / "public/scenes.jsonl").open() as public,
        (root / "private/labels.jsonl").open() as private,
    ):
        for scene_line, label_line in zip(public, private, strict=True):
            scene, label = json.loads(scene_line), json.loads(label_line)
            if label["twin_role"] == "anchor":
                metas.append({"labels": [], "images": {}, "regime": label["regime"]})
            meta = metas[-1]
            meta["labels"].append(label)
            if label["regime"] == "standard":
                goal = scene["goals"][0]
                for name in (goal["initial_image"], goal["goal_image"], scene["forward"]["image"]):
                    meta["images"][name] = hashes[f"public/{name}"]
    for meta in metas:
        anchor = meta["labels"][0]
        contacts = sum(map(_needs_contact, meta["labels"]))
        meta |= {
            "family": anchor["family"],
            "balls": len(anchor["initial"]["balls"]),
            "source": anchor["source"],
            "layout_sha256": anchor["layout_sha256"],
            "layout_bin": _layout_bin(Layout.model_validate(anchor["initial"])),
            "tier": (
                "roll"
                if anchor["family"] == "free_roll"
                else f"contact_{contacts}_of_{len(meta['labels'])}"
            ),
        }
        if meta["regime"] != "standard":
            del meta["images"]  # unknown-friction PNGs are re-rendered, not compared
    return metas


def _export(task: tuple) -> tuple[str, str, list[bool], str | None]:
    out, prefix, regime, ordinal, meta = task
    sim, vis = _SIMS[regime], _VIS[regime]
    labels = meta["labels"]
    initial = Layout.model_validate(labels[0]["initial"])
    field = labels[0]["twin_intervention"]
    shots = [
        (action, sim.replay(initial, action))
        for action in (CueAction.model_validate(label["action"]) for label in labels)
    ]
    effect = _terminal_shift(shots[0][1].goal, shots[1][1].goal, sim.config) if field else None
    public, private = io.StringIO(), io.StringIO()
    written = {}
    for role, (action, replay) in zip(("anchor", "twin"), shots):
        _export_scene(
            out,
            prefix,
            public,
            private,
            layout_group(initial),
            ordinal,
            meta["family"],
            regime,
            role,
            initial,
            action,
            replay,
            sim,
            vis,
            effect,
            field,
            t5_attempts=T5_ATTEMPTS,
        )
        _, _, init_name, goal_name, fwd_name = scene_naming(
            prefix, regime, meta["family"], len(initial.balls), ordinal, role
        )
        written |= {"-init.png": init_name, f"-{role}-goal.png": goal_name}
        if fwd_name:
            written[f"-{role}-fwd.png"] = fwd_name
    if "images" in meta:
        expected = {s: h for p, h in meta["images"].items() for s in SUFFIXES if p.endswith(s)}
        actual = {s: file_sha256(out / "public" / name) for s, name in written.items()}
        if actual != expected:
            raise RuntimeError(f"Re-rendered PNGs differ from the source: {meta['source']}")
    lines = [json.loads(line) for line in private.getvalue().split("\n") if line]
    rows = "".join(json.dumps(label | {"source": meta["source"]}) + "\n" for label in lines)
    return public.getvalue(), rows, [_needs_contact(label) for label in lines], field


def _take(by_tier: dict[str, list], n: int) -> list:
    """Remove n items, split across tiers in proportion to what is left (largest remainder)."""
    total = sum(map(len, by_tier.values()))
    if n > total:
        raise ValueError(f"Need {n} groups but only {total} remain")
    quota = {tier: n * len(items) // total for tier, items in by_tier.items()}
    leftover = sorted(by_tier, key=lambda tier: -(n * len(by_tier[tier]) % total))
    for tier in leftover[: n - sum(quota.values())]:
        quota[tier] += 1
    picked = []
    for tier, k in quota.items():
        picked += by_tier[tier][:k]
        del by_tier[tier][:k]
    return picked


def _by_tier(items: list) -> dict[str, list]:
    tiers: dict[str, list] = {}
    for item in items:
        tiers.setdefault(item["tier"], []).append(item)
    return dict(sorted(tiers.items()))


def _order(parts: list[tuple[list, list]]) -> list:
    """Panels first, then the rest; each block interleaves strata in proportion to their size."""
    keyed = [
        ((block, (i + 0.5) / len(items)), item)
        for panel, rest in parts
        for block, items in ((0, panel), (1, rest))
        for i, item in enumerate(items)
    ]
    return [item for _, item in sorted(keyed, key=lambda pair: pair[0])]


def _plan(groups: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    strata: dict[tuple[str, int], dict[str, list]] = {}
    for group in groups:
        strata.setdefault((group["family"], group["balls"]), {}).setdefault(
            group["tier"], []
        ).append(group)
    for tiers in strata.values():
        for items in tiers.values():
            items.sort(key=lambda g: json.dumps(g["source"], sort_keys=True))
            rng.shuffle(items)
    parts: dict[str, list] = {name: [] for name in SPLIT}
    remaining = {}
    for (family, balls), tiers in sorted(strata.items()):
        # Legacy-train groups were trained on, so they may only go to SFT.
        fresh = {
            t: [g for g in items if g["source"]["dataset"] != LEGACY] for t, items in tiers.items()
        }
        for name, multi in MULTI_PER_STRATUM.items():
            picked = _take(fresh, multi if balls > 1 else _free_roll(multi))
            panel_n = PANEL_PER_STRATUM.get(name, 0)
            panel = _take(_by_tier(picked), panel_n if balls > 1 else _free_roll(panel_n))
            for group in panel:
                group["panel"] = True
            in_panel = {id(group) for group in panel}
            parts[name].append((panel, [g for g in picked if id(g) not in in_panel]))
        trained = [g for items in tiers.values() for g in items if g["source"]["dataset"] == LEGACY]
        remaining[family, balls] = (trained, fresh)
    # SFT keeps all legacy-train groups and fills each stratum to one common size, by contact type.
    multi = min(len(t) + sum(map(len, f.values())) for (_, b), (t, f) in remaining.items() if b > 1)
    for (family, balls), (trained, fresh) in sorted(remaining.items()):
        size = multi if balls > 1 else _free_roll(multi)
        parts["sft"].append(([], trained + _take(fresh, size - len(trained))))
    plan = {}
    for name, stratum_parts in parts.items():
        for panel, rest in stratum_parts:
            rng.shuffle(panel)
            rng.shuffle(rest)
        plan[name] = _order(stratum_parts)
    return plan


def _ood(seed: int, seen: set[str]) -> list[dict]:
    """Layout hashes and 5 mm cells in `seen` are excluded; new episodes are added to it."""
    sims = _simulators()
    radius = sims["standard"].config.ball_radius_m
    entries: list[dict] = []
    for regime in ("slow_friction", "fast_friction"):
        for family in FAMILIES:
            for balls in (1,) if family == "free_roll" else (2, 3, 4):
                n = OOD_PER_STRATUM["free_roll" if balls == 1 else "multi"]
                # A quarter of multi-ball anchors avoid contact, keeping ~2/3 contact goals overall.
                tiers = (
                    ["roll"] * n
                    if balls == 1
                    else (
                        ["no_contact"] * (n // 4)
                        + ["multi_contact"] * (n // 10 if balls > 2 else 0)
                    )
                )
                tiers += ["contact"] * (n - len(tiers))
                for tier in tiers:
                    slot = len(entries)
                    for attempt in range(100_000):
                        initial, action, _ = sample_case(
                            seed,
                            slot,
                            attempt,
                            sims[regime].config,
                            family_override=family,
                            ball_count=balls,
                        )
                        initial = initial.endpoint_view()
                        layout, cell = layout_group(initial), _layout_bin(initial)
                        if layout in seen or cell in seen:
                            continue
                        try:
                            replay = sims[regime].replay(initial, action)
                        except SimulationFailure:
                            continue
                        contacts = _contact_count(replay)
                        if not (
                            _scratch(replay)
                            or score_endpoint(replay.goal, initial, radius).success
                            or (tier == "no_contact" and contacts)
                            or (tier == "contact" and not contacts)
                            or (tier == "multi_contact" and contacts < 2)
                        ):
                            break
                    else:
                        raise RuntimeError(f"No OOD layout for {regime}/{family}/{balls}/{tier}")
                    seen |= {layout, cell}
                    entries.append(
                        {
                            "regime": regime,
                            "family": family,
                            "balls": balls,
                            "labels": [
                                {
                                    "initial": initial.model_dump(mode="json"),
                                    "action": action.model_dump(mode="json"),
                                    "twin_intervention": None,
                                }
                            ],
                            "source": {
                                "dataset": "inverser-ood",
                                "seed": seed,
                                "slot": slot,
                                "candidate": attempt,
                                "tier": tier,
                            },
                        }
                    )
    return entries


def _reordered_test(test: Path, standard: int, ood: int) -> list[dict]:
    """Keep test membership; the first `standard` groups and a stratified `ood` slice lead."""
    metas = _metas(test)
    std = [m for m in metas if m["regime"] == "standard"]
    strata: dict[tuple, list] = {}
    for meta in metas:
        if meta["regime"] != "standard":
            key = (meta["regime"], meta["family"], meta["balls"], meta["tier"])
            strata.setdefault(key, []).append(meta)
    episodes = _order([([], items) for _, items in sorted(strata.items())])
    for meta in std[:standard] + episodes[:ood]:
        meta["panel"] = True
    return std[:standard] + episodes[:ood] + std[standard:] + episodes[ood:]


def _export_splits(stage: Path, plan: dict, provenance: dict, seed: int, workers: int) -> None:
    """Write each planned split; panel_prefix counts the leading panel groups."""
    regimes = ("standard", "slow_friction", "fast_friction")
    tasks = []
    for name, metas in plan.items():
        (stage / name / "public").mkdir(parents=True)
        (stage / name / "private").mkdir()
        ordinals: Counter[tuple] = Counter()
        for meta in metas:
            regime = meta.get("regime", "standard")
            ordinals[regime, meta["family"], meta["balls"]] += 1
            ordinal = ordinals[regime, meta["family"], meta["balls"]]
            tasks.append((name, regime, meta, (stage / name, name, regime, ordinal, meta)))
    stats = {name: Counter() for name in plan}
    layouts: dict[str, set] = {name: set() for name in plan}
    in_panel = dict.fromkeys(plan, True)
    streams = {
        name: tuple(
            (stage / name / f).open("w") for f in ("public/scenes.jsonl", "private/labels.jsonl")
        )
        for name in plan
    }
    with ProcessPoolExecutor(workers, initializer=_init) as executor:
        results = executor.map(_export, [task[-1] for task in tasks], chunksize=32)
        for (name, regime, meta, _), (public, private, needs, field) in zip(tasks, results):
            streams[name][0].write(public)
            streams[name][1].write(private)
            group = json.loads(private.split("\n", 1)[0])["layout_sha256"]
            if group in layouts[name]:
                raise ValueError(f"Duplicate layout inside {name}: {group}")
            layouts[name].add(group)
            s = stats[name]
            in_panel[name] = in_panel[name] and bool(meta.get("panel"))
            if in_panel[name]:
                s["panel/layout_groups"] += 1
                s["panel/scene_rows"] += len(needs)
            s[f"coverage/{'standard' if regime == 'standard' else 'ood'}/{meta['family']}"] += 1
            s[f"balls/{regime}/{meta['family']}/{meta['balls']}"] += len(needs)
            s[f"scenes/{regime}"] += len(needs)
            s[f"contact/{regime}"] += sum(needs)
            s[f"source/{meta['source']['dataset']}"] += 1
            s[f"intervention/{field}" if field else f"ood/{regime}"] += 1
        for stream in (f for pair in streams.values() for f in pair):
            stream.close()
        if len(set.union(*layouts.values())) != sum(map(len, layouts.values())):
            raise ValueError("A layout group appears in more than one split")
        sims = _simulators()
        table = public_physics(sims["standard"]).table
        for name in plan:
            out, s = stage / name, stats[name]
            for regime in regimes:
                n = s[f"scenes/{regime}"]
                if n and not CONTACT_SHARE[0] <= s[f"contact/{regime}"] / n <= CONTACT_SHARE[1]:
                    raise ValueError(f"{name}/{regime}: contact share outside {CONTACT_SHARE}")
            write_json(out / "groups.json", sorted(layouts[name]))
            write_json(out / "private/physics.json", {k: sim.metadata() for k, sim in sims.items()})
            files = sorted(path for path in out.rglob("*") if path.is_file())
            hashes = dict(
                zip(
                    (str(path.relative_to(out)) for path in files),
                    executor.map(file_sha256, files, chunksize=256),
                )
            )

            def pick(prefix: str, s: Counter = s) -> dict:
                return {k.split("/", 1)[1]: v for k, v in sorted(s.items()) if k.startswith(prefix)}

            ood_rows = s["scenes/slow_friction"] + s["scenes/fast_friction"]
            write_json(
                out / "manifest.json",
                {
                    "schema_version": "inverser-vlm-v1",
                    "dataset": "inverser",
                    "pool": name,
                    "split": SPLIT[name],
                    "family_counts": pick("coverage/"),
                    "ball_count_by_regime_family": pick("balls/"),
                    "standard_scene_count": s["scenes/standard"],
                    "ood_episode_count": ood_rows,
                    "ood_regime_counts": pick("ood/"),
                    "independent_layout_groups": len(layouts[name]),
                    "panel_prefix": pick("panel/") or None,
                    "contact_required_scenes": {
                        regime: [s[f"contact/{regime}"], s[f"scenes/{regime}"]]
                        for regime in regimes
                        if s[f"scenes/{regime}"]
                    },
                    "sources": pick("source/"),
                    "twin_interventions": pick("intervention/"),
                    "witness_replays_verified": s["scenes/standard"] + ood_rows,
                    "seed": seed,
                    "format": {
                        "times_s": TIMES,
                        "tolerance_radii": 1.0,
                        "minimum_twin_shift_radii": 2.0,
                        "ray_seconds": 0.16,
                        "t5_attempts": T5_ATTEMPTS,
                        "boundary_map": _boundary_map(table),
                        "ood_withheld_fields": WITHHELD,
                    },
                    "legacy_exclusions": provenance,
                    "artifact_sha256": hashes,
                },
            )


def _verify(roots: dict[str, Path], pilot: Path, formal: Path) -> dict:
    """Run verify_vlm_suite twice, once with sft and once with rl standing in for train."""
    manifests = {
        name: json.loads((root / "manifest.json").read_text()) for name, root in roots.items()
    }
    counts = {
        name: {f: m["family_counts"].get(f"standard/{f}", 0) for f in FAMILIES}
        for name, m in manifests.items()
    }
    ood = {f: manifests["test"]["family_counts"].get(f"ood/{f}", 0) for f in FAMILIES}
    receipts = {}
    for name in ("sft", "rl"):
        with TemporaryDirectory(prefix=".verify-", dir=roots["test"].parent) as temp:
            suite = Path(temp) / SUITE_DIR
            suite.mkdir()
            for split, target in (("train", name), ("valid", "valid"), ("test", "test")):
                (suite / split).symlink_to(roots[target].resolve())

            def naming(split: str, *args: object, _name: str = name) -> tuple:
                return scene_naming(_name if split == "train" else split, *args)

            with mock.patch.object(vlm_suite, "scene_naming", naming):
                receipts[name] = verify_vlm_suite(
                    Path(temp),
                    pilot,
                    formal,
                    counts=(counts[name], counts["valid"], counts["test"]),
                    ood_per_family=ood,
                    t5_attempts=T5_ATTEMPTS,
                )
    return receipts


def _keep(source: Path, valid: int, test: int, ood: int) -> dict[str, list[dict]]:
    """Same memberships and standard order; only the pre-registered valid/test prefixes change."""
    plan = {
        name: [m for m in _metas(source / name) if m["regime"] == "standard"]
        for name in ("sft", "rl", "valid")
    }
    for meta in plan["valid"][:valid]:
        meta["panel"] = True
    plan["test"] = _reordered_test(source / "test", test, ood)
    return plan


def build(
    source: Path,
    legacy: Path | None,
    pilot: Path,
    formal: Path,
    output: Path,
    seed: int,
    workers: int,
    panels: tuple[int, int, int] | None = None,
) -> dict:
    require_data_path(output)
    if output.exists():
        raise FileExistsError(output)
    source = source.resolve()
    excluded, provenance = _legacy_groups(pilot, formal)
    if panels:
        seed = json.loads((source / "test/manifest.json").read_text())["seed"]
        plan = _keep(source, *panels)
    else:
        plan = _resplit(source, legacy, excluded, random.Random(seed), seed)
    with TemporaryDirectory(prefix=".inverser-build-", dir=output.parent) as temp:
        stage = Path(temp) / output.name
        _export_splits(stage, plan, provenance, seed, workers)
        receipts = _verify({name: stage / name for name in SPLIT}, pilot, formal)
        stage.rename(output)
    return {"layout_groups": {name: len(metas) for name, metas in plan.items()}, "verify": receipts}


def _resplit(
    source: Path, legacy: Path | None, excluded: set, rng: random.Random, seed: int
) -> dict[str, list[dict]]:
    groups = [m for split in SPLIT for m in _metas(source / split) if m["regime"] == "standard"]
    # Legacy valid/test and the pilot/formal layouts are off-limits everywhere; legacy-train
    # layouts and their near-duplicates only in SFT.
    held_out, train_keys = set(excluded), set()
    for split in ("train", "valid", "test") if legacy else ():
        keys = set(json.loads((legacy / split / "groups.json").read_text()))
        with (legacy / split / "private/labels.jsonl").open() as stream:
            keys |= {
                _layout_bin(Layout.model_validate(json.loads(line)["initial"])) for line in stream
            }
        if split == "train":
            train_keys = keys
        else:
            held_out |= keys
    plan = _plan(groups, rng)
    for name, metas in plan.items():
        for meta in metas:
            keys = {meta["layout_sha256"], meta["layout_bin"]}
            fresh = meta["source"]["dataset"] != LEGACY
            if keys & held_out or ((name != "sft" or fresh) and keys & train_keys):
                raise ValueError(
                    f"{name} group overlaps legacy or excluded layouts: {meta['source']}"
                )
    seen = held_out | train_keys | {g["layout_sha256"] for g in groups}
    seen |= {g["layout_bin"] for g in groups}
    episodes = _ood(seed, seen)
    for meta in episodes:
        meta["panel"] = True
    panel = sum(bool(meta.get("panel")) for meta in plan["test"])
    plan["test"] = plan["test"][:panel] + episodes + plan["test"][panel:]
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "pilot", "formal", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument(
        "--panels",
        type=int,
        nargs=3,
        metavar=("VALID", "TEST", "OOD"),
        help="keep memberships; leading valid groups, test groups and OOD episodes of each panel",
    )
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))
