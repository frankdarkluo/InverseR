"""Rewrite the <think> traces of the InverseR SFT split as faithful intermediate-state chains.

Input:  an SFT split such as data/research/inverser/sft (train manifest, scenes, labels, images).
Output: <output>/sft.jsonl and manifest.json (schema inverser-sft-v2, read by train-capo-sft) and
        <output>/images/ hard-linked from the source split. Prompts, images and <answer> blocks are
        exactly those of capo_trainer.sft_examples; only the text inside <think> changes.
Every number in a new trace comes from the Pooltool replay of the witness shot or from an exact
Pooltool closed form, so the reasoning lands on the unchanged answer:
  launch    v_ball = 2 v_stick / (1 + m/M + 2.5 (side^2 + vertical^2))
  transfer  v_object = (1 + e_b)/2 * U * cos(cut), cut measured from the line of centres
  rolling   v^2 falls by 2 mu_r g per metre; a centre-hit ball rolls d = K v^2 from a stun start
"""

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from inverser.benchmark.vlm_suite import _boundary_map
from inverser.capo.capo_trainer import sft_examples
from inverser.physics import SnookerSimulator, sample_event_motion
from inverser.runtime import file_sha256
from inverser.schemas import CueAction, Layout, PhysicsConfig
from inverser.study import public_physics

CFG = PhysicsConfig()
R, W, L = CFG.ball_radius_m, CFG.table_width_m, CFG.table_length_m
G = CFG.gravity_mps2
MASS_RATIO = CFG.ball_mass_kg / 0.567  # Pooltool default cue mass M (physics.json cue_specs)
TRANSFER = (1 + CFG.ball_restitution) / 2
ROLL_LOSS = 2 * CFG.rolling_friction * G
SLIDE_LOSS = 2 * CFG.sliding_friction * G
K_TEXT = f"k = |1-q^2|/{SLIDE_LOSS:.2f} + q^2/{ROLL_LOSS:.3f}, q = 5/7*(1+b)"
EVENT_WORDS = (
    "ball_ball",
    "_cushion",
    "sliding_rolling",
    "stationary",
    "stick_ball",
    "ball_pocket",
)
ROLLING, POCKETED = 3, 4  # Pooltool motion states
SIM = SIDE = None


def _init() -> None:
    global SIM, SIDE
    SIM = SnookerSimulator(CFG)
    SIDE = _boundary_map(public_physics(SIM).table)


def factor(side: float, vertical: float) -> float:
    return 2 / (1 + MASS_RATIO + 2.5 * (side**2 + vertical**2))


def reach_k(vertical: float) -> float:
    """Open-table roll-out d = k v^2: slide until rolling at q v, q = 5/7 (1 + b), then roll."""
    q = 5 / 7 * (1 + vertical)
    return abs(1 - q * q) / SLIDE_LOSS + q * q / ROLL_LOSS


def heading(v) -> float:
    return math.degrees(math.atan2(v[1], v[0])) % 360


def turn(a: float, b: float) -> float:
    """Signed smallest angle a - b in degrees."""
    return (a - b + 180) % 360 - 180


def speed(v) -> float:
    return math.hypot(v[0], v[1])


def xy(p) -> str:
    return f"({p[0]:.3f}, {p[1]:.3f})"


def tip(a: CueAction) -> str:
    words = []
    if abs(a.tip_vertical) >= 0.005:
        kind = "follow" if a.tip_vertical > 0 else "draw"
        words.append(f"{kind} (tip_vertical {a.tip_vertical:+.2f})")
    if abs(a.tip_side) >= 0.005:
        kind = "right" if a.tip_side > 0 else "left"
        words.append(f"{kind} English (tip_side {a.tip_side:+.2f})")
    return " + ".join(words) or "centre-ball hit"


def centre(a: CueAction) -> bool:
    return a.tip_side == 0 and a.tip_vertical == 0


def mirror_heading(start, end, rails) -> float:
    """Aim at the goal reflected across each cushion (ball-centre lines), last cushion first."""
    x, y = end
    for rail in reversed(rails):
        if rail in ("LEFT", "RIGHT"):
            x = 2 * (R if rail == "LEFT" else W - R) - x
        else:
            y = 2 * (R if rail == "BOTTOM" else L - R) - y
    return heading((x - start[0], y - start[1]))


class Shot:
    """Per-ball states around each simulator event of one witness replay."""

    def __init__(self, label: dict):
        self.action = CueAction.model_validate(label["action"])
        initial = Layout.model_validate(label["initial"])
        self.replay = SIM.replay(initial, self.action)
        # Replays keep the tracking ids; answer slots follow endpoint_view, which renumbers reds
        # by position, so map each tracked ball to its slot through its (unique) start position.
        slot_at = {b.xy_m: i for i, b in enumerate(initial.endpoint_view().balls)}
        self.slot = {b.ball_id: slot_at[b.xy_m] for b in initial.balls}
        self.start = {
            b.ball_id: b.xy_m for b in sorted(initial.balls, key=lambda b: self.slot[b.ball_id])
        }
        self.goal = {b.ball_id: b.xy_m for b in Layout.model_validate(label["goal"]).balls}
        self.hist: dict[str, list] = {}
        for m in self.replay.motion:
            self.hist.setdefault(m.ball_id, []).append(m)
        for b in self.replay.goal.balls:  # the replay must be the labelled witness
            g = self.goal[b.ball_id]
            assert (b.xy_m is None) == (g is None) and (g is None or math.dist(b.xy_m, g) < 1e-6)
        self.events = [
            (
                "hit"
                if e.event_type == "ball_ball"
                else "pot"
                if e.event_type == "ball_pocket"
                else "rail",
                e,
            )
            for e in self.replay.events
            if e.event_type
            in ("ball_ball", "ball_pocket", "ball_linear_cushion", "ball_circular_cushion")
        ]
        self.v0 = speed(self.replay.cue_velocity)
        self.h0 = heading(self.replay.cue_velocity)
        self.f = factor(self.action.tip_side, self.action.tip_vertical)
        self.k = reach_k(self.action.tip_vertical)
        self.reach = self.k * self.v0**2
        assert abs(self.v0 / self.action.cue_speed_mps / self.f - 1) < 1e-6, "launch factor"
        self.squirt = turn(self.h0, self.action.azimuth_deg)

    def name(self, ball_id: str) -> str:
        return "cue" if self.slot[ball_id] == 0 else f"ball {self.slot[ball_id]}"

    def around(self, ball_id: str, t: float):
        """State just before t (evolved) and the recorded state just after t."""
        post = [m for m in self.hist[ball_id] if abs(m.time_s - t) < 1e-9][-1]
        pre = sample_event_motion(tuple(self.hist[ball_id]), CFG, (max(0.0, t - 1e-7),))[0]
        return pre, post

    def contact(self, t: float, i: str, j: str):
        """Striker (faster ball before impact) first: its incoming state, both outgoing states,
        line of centres striker->struck and cut angle."""
        pre_i, post_i = self.around(i, t)
        pre_j, post_j = self.around(j, t)
        if speed(pre_j.velocity_mps) > speed(pre_i.velocity_mps):
            i, j, pre_i, post_i, post_j = j, i, pre_j, post_j, post_i
        line = heading(
            (post_j.position_m[0] - pre_i.position_m[0], post_j.position_m[1] - pre_i.position_m[1])
        )
        return i, j, pre_i, post_i, post_j, line, abs(turn(heading(pre_i.velocity_mps), line))

    def rails(self, ball_id: str, after: float = -1.0) -> list[str]:
        return [
            SIDE[e.agents[1]]
            for k, e in self.events
            if k == "rail" and e.agents[0] == ball_id and e.time_s > after
        ]

    def launch_line(self) -> tuple[str, str | None]:
        """Text for each ball ahead of the cue, and the ball the cue meets first on this line."""
        rows = []
        ux, uy = math.cos(math.radians(self.h0)), math.sin(math.radians(self.h0))
        cx, cy = self.start["cue"]
        for ball_id, p in self.start.items():
            along = (p[0] - cx) * ux + (p[1] - cy) * uy
            if ball_id == "cue" or along <= 0:
                continue
            off = abs((p[1] - cy) * ux - (p[0] - cx) * uy) / R
            if off < 2:
                touch = along - R * math.sqrt(4 - off * off)
                where = "within" if touch <= self.reach else "beyond"
                rows.append(
                    (
                        along,
                        off,
                        touch <= self.reach,
                        ball_id,
                        f"contact at {touch:.2f} m, {where} reach",
                    )
                )
            else:
                rows.append((along, off, False, ball_id, "clear"))
        rows.sort()
        met = next((b for _, _, hit, b, _ in rows if hit), None)
        text = ", ".join(
            f"{self.name(b)} {off:.2f}r off-line ({note})" for _, off, _, b, note in rows
        )
        return text or "no ball ahead on the launch line", met

    def rest(self) -> str:
        parts, still = [], []
        for ball_id in self.start:
            last = self.hist[ball_id][-1]
            if last.motion_state == POCKETED:
                parts.append(f"{self.name(ball_id)} pocketed")
            elif math.dist(last.position_m[:2], self.start[ball_id]) > 1e-6:
                parts.append(
                    f"{self.name(ball_id)} stops {xy(last.position_m)} at {last.time_s:.1f} s"
                )
            else:
                still.append(self.name(ball_id))
        return "Rest: " + "; ".join(parts) + (f"; untouched: {', '.join(still)}." if still else ".")

    def launch(self) -> str:
        a = self.action
        note = (
            f" (English squirts it {self.squirt:+.2f} deg off the cue azimuth {a.azimuth_deg:.2f})"
            if abs(self.squirt) >= 0.005
            else ""
        )
        return (
            f"stick {a.cue_speed_mps:.2f} m/s at {a.azimuth_deg:.2f} deg, {tip(a)}: "
            f"factor 2/(1+{MASS_RATIO:.3f}+2.5*s^2) = {self.f:.4f}, so the cue ball leaves at "
            f"{self.v0:.3f} m/s heading {self.h0:.2f} deg{note}; on an open table it rolls out "
            f"k*v^2 = {self.k:.3f}*{self.v0:.3f}^2 = {self.reach:.2f} m "
            f"({K_TEXT}, b = {a.tip_vertical:+.2f})"
        )


def forward_trace(s: Shot) -> str:
    lines = [
        "Read: " + "; ".join(f"{s.name(b)} {xy(p)}" for b, p in s.start.items()) + ".",
        f"Strike: {s.launch()}.",
    ]
    if len(s.start) > 1:
        lines.append(
            f"Launch line check (hit if < 2r off-line and within reach): {s.launch_line()[0]}."
        )
    for n, (kind, e) in enumerate(s.events):
        t = e.time_s
        if kind == "pot":
            lines.append(f"t={t:.2f} s: {s.name(e.agents[0])} drops into a pocket.")
        elif n >= 8:  # keep long chains short; every contact and cushion is still listed
            i, j = s.contact(t, *e.agents)[:2] if kind == "hit" else e.agents
            other = s.name(j) if kind == "hit" else f"{SIDE[j]} cushion"
            lines.append(f"t={t:.2f} s: {s.name(i)} hits {other}.")
        elif kind == "hit":
            i, j, pre, post_i, post_j, _, cut = s.contact(t, *e.agents)
            u = speed(pre.velocity_mps)
            lines.append(
                f"t={t:.2f} s: {s.name(i)} ({u:.2f} m/s) hits {s.name(j)}, cut {cut:.1f} deg -> "
                f"{s.name(j)} {TRANSFER:.3f}*{u:.2f}*cos{cut:.1f} = "
                f"{speed(post_j.velocity_mps):.2f} m/s heading "
                f"{heading(post_j.velocity_mps):.0f} deg; "
                f"{s.name(i)} {speed(post_i.velocity_mps):.2f} m/s heading "
                f"{heading(post_i.velocity_mps):.0f} deg."
            )
        else:
            _, post = s.around(e.agents[0], t)
            lines.append(
                f"t={t:.2f} s: {s.name(e.agents[0])} hits {SIDE[e.agents[1]]} cushion -> "
                f"{speed(post.velocity_mps):.2f} m/s heading {heading(post.velocity_mps):.0f} deg."
            )
    lines.append(s.rest())
    return "\n".join(f"{n}. {line}" for n, line in enumerate(lines, start=1))


def inverse_trace(s: Shot) -> str:
    a = s.action
    moved, stay = [], []
    for b, p in s.start.items():
        g = s.goal[b]
        if g is None:
            moved.append(f"{s.name(b)} {xy(p)} -> pocket")
        elif math.dist(p, g) > R:
            moved.append(f"{s.name(b)} {xy(p)} -> {xy(g)}")
        else:
            stay.append(s.name(b))
    lines = ["Goal change: " + "; ".join(moved) + (f"; stay: {', '.join(stay)}." if stay else ".")]
    hits = [e for k, e in s.events if k == "hit"]
    stick = f"{s.v0:.3f}/{s.f:.4f} = {s.v0 / s.f:.3f} m/s"
    aim = (
        f"; English squirts the cue ball {s.squirt:+.2f} deg, so the cue azimuth is "
        f"{s.h0:.2f}-({s.squirt:+.2f}) = {s.h0 - s.squirt:.2f} deg"
        if abs(s.squirt) >= 0.005
        else ""
    )
    if hits and "cue" in hits[0].agents:
        t = hits[0].time_s
        target = [b for b in hits[0].agents if b != "cue"][0]
        _, _, pre, post_c, post_o, line, cut = s.contact(t, "cue", target)
        h_o, v_o = heading(post_o.velocity_mps), speed(post_o.velocity_mps)
        rails, p, g = s.rails(target), s.start[target], s.goal[target]
        route = (
            "into a pocket"
            if g is None
            else f"via {', '.join(rails)} cushion"
            if rails
            else "on a direct line"
        )
        again = any(target in e.agents for e in hits[1:])
        if again:  # its goal mixes two impacts, so a mirror of the goal would mislead
            route += " (it is struck again later, see below)"
        elif rails and g is not None:
            direct = heading((g[0] - p[0], g[1] - p[1]))
            push = heading((p[0] - s.start["cue"][0], p[1] - s.start["cue"][1]))
            if abs(turn(direct, push)) > 90:
                route += (
                    f" (the direct heading {direct:.0f} deg points back toward the cue, "
                    "a cut over 90 deg, so it must bank)"
                )
            route += (
                f"; mirroring its goal across {', '.join(rails)} gives heading "
                f"{mirror_heading(p, g, rails):.0f} deg"
            )
        roll = (
            f" (direct roll-out about 2.36*v^2 = {2.36 * v_o**2:.2f} m)"
            if not rails and g is not None and not again
            else ""
        )
        lines.append(
            f"{s.name(target)} goes {route}: it must leave heading {h_o:.0f} deg "
            f"at about {v_o:.2f} m/s{roll}."
        )
        chain = []
        for e in hits[1:4]:
            i, j, _, _, post_j, _, _ = s.contact(e.time_s, *e.agents)
            chain.append(
                f"Then at t={e.time_s:.2f} s {s.name(i)} hits {s.name(j)}, which leaves heading "
                f"{heading(post_j.velocity_mps):.0f} deg at {speed(post_j.velocity_mps):.2f} m/s."
            )
        ghost = (
            p[0] - 2 * R * math.cos(math.radians(line)),
            p[1] - 2 * R * math.sin(math.radians(line)),
        )
        throw = (
            f"; throw turns {s.name(target)} to {h_o:.0f} deg"
            if abs(turn(h_o, line)) >= 0.5
            else ""
        )
        lines.append(
            f"Contact: line of centres {line:.1f} deg, so the ghost-ball centre is "
            f"{s.name(target)} - 2r along it = {xy(ghost)}{throw}; the cue ball must travel at "
            f"{s.h0:.2f} deg, a {cut:.1f} deg cut{aim}."
        )
        u = speed(pre.velocity_mps)
        kick = [
            SIDE[e.agents[1]]
            for k, e in s.events
            if k == "rail" and e.agents[0] == "cue" and e.time_s < t
        ]
        before = [m.position_m[:2] for m in s.hist["cue"] if m.time_s < t] + [pre.position_m[:2]]
        travel = sum(math.dist(m, n) for m, n in zip(before, before[1:]))
        if kick:
            slide = f" (kicking off the {', '.join(kick)} cushion first)"
        elif centre(a):
            slide = f" (slides, then rolls with v^2 falling {ROLL_LOSS:.3f} per m)"
        else:
            slide = ""
        lines.append(
            f"Pace: to give {s.name(target)} {v_o:.2f} m/s at a {cut:.1f} deg cut the cue must "
            f"arrive at {v_o:.2f}/({TRANSFER:.3f}*cos{cut:.1f}) = {u:.2f} m/s after {travel:.2f} m"
            f"{slide}, so it leaves at {s.v0:.3f} m/s; "
            f"factor 2/(1+{MASS_RATIO:.3f}+2.5*s^2) = {s.f:.4f} -> stick {stick}."
        )
        # The cue keeps the velocity component across the line of centres, on its incoming side.
        tangent = (line + (90 if turn(heading(pre.velocity_mps), line) > 0 else -90)) % 360
        nxt = [m for m in s.hist["cue"] if m.time_s > t + 1e-9]
        # Bend is read at the next roll-in; if a cushion comes first, use the post-impact heading.
        rolled = nxt and nxt[0].motion_state == ROLLING
        bend = heading((nxt[0] if rolled else post_c).velocity_mps)
        if abs(a.tip_vertical) >= 0.005:
            how = f"with {tip(a)} it bends from the tangent {tangent:.0f} deg to {bend:.0f} deg"
        elif pre.motion_state == ROLLING:
            how = (
                f"it was rolling at impact, so natural follow bends it from the tangent "
                f"{tangent:.0f} deg to {bend:.0f} deg"
            )
        else:
            how = (
                f"it was still sliding at impact, so it leaves near the tangent {tangent:.0f} deg "
                f"({bend:.0f} deg)"
            )
        cue_rails = s.rails("cue", after=t)
        lines.append(
            f"Cue after contact: {how}"
            + (f", then {', '.join(cue_rails)} cushion" if cue_rails else "")
            + f" to reach its goal -> {tip(a)}."
        )
        lines += chain
    else:
        rails, g = s.rails("cue"), s.goal["cue"]
        if len(s.start) > 1:
            lines.append(f"Cue alone, nothing to meet on its launch line: {s.launch_line()[0]}.")
        hist = s.hist["cue"]
        path = sum(math.dist(m.position_m[:2], n.position_m[:2]) for m, n in zip(hist, hist[1:]))
        if rails:
            lines.append(
                f"Route via {', '.join(rails)} cushion: mirroring the goal gives heading "
                f"{mirror_heading(s.start['cue'], g, rails):.0f} deg, actual {s.h0:.2f} deg; "
                f"the path is {path:.2f} m, a cushion-free roll of that length would need "
                f"sqrt({path:.2f}/{s.k:.3f}) = {math.sqrt(path / s.k):.2f} m/s, and the "
                f"cushions cost speed, so launch {s.v0:.3f} m/s."
            )
        else:
            assert abs(math.sqrt(path / s.k) - s.v0) < 1e-3, "open-table roll-out law"
            lines.append(
                f"Direct roll {path:.3f} m at {s.h0:.2f} deg with {tip(a)}: "
                f"v = sqrt(d/k) = sqrt({path:.3f}/{s.k:.3f}) = {s.v0:.3f} m/s ({K_TEXT}, "
                f"b = {a.tip_vertical:+.2f})."
            )
        lines.append(
            f"Stick: factor 2/(1+{MASS_RATIO:.3f}+2.5*s^2) = {s.f:.4f} -> {stick}, {tip(a)}{aim}."
        )
    return "\n".join(f"{n}. {line}" for n, line in enumerate(lines, start=1))


def traces(label: dict) -> tuple[str, str]:
    s = Shot(label)
    out = forward_trace(s), inverse_trace(s)
    assert not any(word in text for text in out for word in EVENT_WORDS), "simulator event names"
    # The traces must name the same balls and cushions as the unchanged answer labels.
    truth, hits = label["truth"], [e.agents for k, e in s.events if k == "hit"]
    first = next((s.slot[b] for a in hits if "cue" in a for b in a if b != "cue"), None)
    struck = {s.slot[b] for a in hits for b in a}
    assert first == truth["first_contact"], "first contact"
    assert [i in struck for i in range(1, len(s.slot))] == truth["object_collisions"], "hits"
    assert [SIDE[e.agents[1]] for k, e in s.events if k == "rail"] == truth["cushion_sequence"]
    cue_first = next((k for k, e in s.events if "cue" in e.agents and k != "pot"), None)
    met = s.launch_line()[1]
    if met is None:
        assert cue_first != "hit" or near_boundary(s), "launch line misses a direct hit"
    else:
        assert s.slot[met] == truth["first_contact"] or near_boundary(s), "launch line hit"
    return out


def near_boundary(s: Shot) -> bool:
    """A ball within 0.02r of the 2r line or 1 cm of the reach edge (rounding cases)."""
    cx, cy = s.start["cue"]
    ux, uy = math.cos(math.radians(s.h0)), math.sin(math.radians(s.h0))
    for b, p in s.start.items():
        along = (p[0] - cx) * ux + (p[1] - cy) * uy
        off = abs((p[1] - cy) * ux - (p[0] - cx) * uy) / R
        if b != "cue" and along > 0 and abs(off - 2) < 0.02:
            return True
        if (
            b != "cue"
            and along > 0
            and off < 2
            and abs(along - R * math.sqrt(4 - off * off) - s.reach) < 0.01
        ):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("data/research/inverser/sft"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, help="first N scenes only (smoke runs)")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--keep-think", action="store_true", help="keep the template traces (matched control)"
    )
    args = parser.parse_args()
    split, out = args.split, args.output
    rows = sft_examples(split, args.count)
    with (split / "private/labels.jsonl").open() as f:
        labels = [json.loads(line) for _, line in zip(range(len(rows) // 2), f)]
    assert len(rows) == 2 * len(labels)
    out.mkdir(parents=True)  # never overwrite an existing corpus
    new = None
    if not args.keep_think:
        with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
            new = list(pool.map(traces, labels, chunksize=64))
    public = (split / "public").resolve()
    with (out / "sft.jsonl").open("w") as f:
        for n, row in enumerate(rows):
            if new is not None:
                answer = row["completion"][0]["content"].split("\n</think>\n", 1)[1]
                think = new[n // 2][n % 2]
                row["completion"][0]["content"] = f"<think>\n{think}\n</think>\n{answer}"
            paths = []
            for path in row["image_paths"]:
                rel = Path(path).resolve().relative_to(public)
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                if not (out / rel).exists():
                    os.link(public / rel, out / rel)
                paths.append(str(rel))
            row["image_paths"] = paths
            f.write(json.dumps(row) + "\n")
    source = json.loads((split / "manifest.json").read_text())
    keep = (
        "standard_scene_count",
        "independent_layout_groups",
        "family_counts",
        "witness_replays_verified",
    )
    # schema inverser-sft-v2 makes train-capo-sft read sft.jsonl with corpus-relative images.
    manifest = {
        "dataset": "inverser-sft-template" if args.keep_think else "inverser-sft-think",
        "schema_version": "inverser-sft-v2",
        "split": "train",
        "source": {
            "split": str(split),
            "manifest_sha256": file_sha256(split / "manifest.json"),
            "scenes": len(labels),
        },
        "builder_sha256": file_sha256(Path(__file__)),
        "sft_examples": {"total": len(rows), "forward": len(labels), "inverse": len(labels)},
        "artifact_sha256": {"sft.jsonl": file_sha256(out / "sft.jsonl")},
    } | {key: source[key] for key in keep if key in source and args.count is None}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["sft_examples"]), out)


if __name__ == "__main__":
    main()
