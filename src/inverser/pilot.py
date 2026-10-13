import fcntl
import json
import logging
import math
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Literal

import numpy as np

from inverser.physics import SimulationFailure, SnookerSimulator
from inverser.runtime import (
    PROJECT_ROOT,
    check_storage,
    file_sha256,
    fingerprint,
    write_json,
)
from inverser.schemas import BallPosition, CueAction, Layout, PhysicsConfig, ShotRecord

LOGGER = logging.getLogger(__name__)
MAX_PILOT_TASKS = 5000
MAX_RECORD_BYTES = 1_048_576
MAX_CANDIDATES = 20
Family = Literal["free_roll", "center", "vertical", "side"]


def sample_case(
    seed: int, index: int, candidate: int, config: PhysicsConfig
) -> tuple[Layout, CueAction, Family]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, candidate]))
    family: Family
    if index % 10 == 0:
        family = "free_roll"
    elif index % 3 == 0:
        family = "center"
    elif index % 3 == 1:
        family = "vertical"
    else:
        family = "side"
    count = 1 if family == "free_roll" else 2 + index % 3
    margin = config.ball_radius_m * 4
    positions: list[tuple[float, float]] = []
    for _ in range(count):
        for _attempt in range(1000):
            xy = (
                float(rng.uniform(margin, config.table_width_m - margin)),
                float(rng.uniform(margin, config.table_length_m - margin)),
            )
            if all(math.dist(xy, existing) > 5 * config.ball_radius_m for existing in positions):
                positions.append(xy)
                break
        else:
            raise RuntimeError("Could not sample non-overlapping pilot positions")
    balls = [BallPosition(ball_id="cue", kind="cue", xy_m=positions[0])]
    balls.extend(
        BallPosition(ball_id=f"red_{i}", kind="red", xy_m=xy) for i, xy in enumerate(positions[1:])
    )
    if count == 1:
        angle = float(rng.uniform(0, 360))
    else:
        dx, dy = np.asarray(positions[1]) - positions[0]
        angle = math.degrees(math.atan2(dy, dx))
        spread = math.degrees(math.asin(2 * config.ball_radius_m / math.hypot(dx, dy)))
        angle = float((angle + rng.uniform(-0.8 * spread, 0.8 * spread)) % 360)
    side = float(rng.uniform(-0.35, 0.35)) if family == "side" else 0.0
    vertical = float(rng.uniform(-0.4, 0.4)) if family == "vertical" else 0.0
    action = CueAction(
        azimuth_deg=angle,
        cue_speed_mps=float(rng.uniform(0.35, 1.5)),
        tip_side=side,
        tip_vertical=vertical,
    )
    return Layout(balls=tuple(balls)), action, family


def source_fingerprint() -> str:
    return fingerprint(
        {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in sorted((PROJECT_ROOT / "src" / "inverser").glob("*.py"))
        }
    )


def _export(connection: sqlite3.Connection, output: Path) -> dict[str, str]:
    hashes = {}
    for public, name in ((True, "public.jsonl"), (False, "privileged.jsonl")):
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=output, prefix=f".{name}.", delete=False
            ) as stream:
                temporary = Path(stream.name)
                for (payload,) in connection.execute("SELECT payload FROM records ORDER BY idx"):
                    record = ShotRecord.model_validate_json(payload)
                    value = record.public_input() if public else record
                    stream.write(value.model_dump_json() + "\n")
            temporary.replace(output / name)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        hashes[name] = file_sha256(output / name)
    return hashes


def generate_pilot(output: Path, *, count: int, seed: int) -> dict[str, object]:
    if not 1 <= count <= MAX_PILOT_TASKS:
        raise ValueError("This pre-G1 command permits 1-5000 pilot tasks, not benchmark expansion")
    if seed < 0:
        raise ValueError("Seed must be nonnegative")
    check_storage(count * MAX_RECORD_BYTES * 4)
    output.mkdir(parents=True, exist_ok=True)
    config = PhysicsConfig()
    simulator = SnookerSimulator(config)
    run_config = {
        "schema_version": "pilot-v0",
        "count": count,
        "seed": seed,
        "source_hash": source_fingerprint(),
        "physics": simulator.metadata(),
    }
    manifest: dict[str, object] = {}
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with closing(sqlite3.connect(output / "pilot.sqlite")) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS records (
                    idx INTEGER PRIMARY KEY, group_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY, idx INTEGER NOT NULL, candidate INTEGER NOT NULL,
                    status TEXT NOT NULL, reason TEXT
                );
                """
            )
            existing = connection.execute(
                "SELECT value FROM metadata WHERE key='config_hash'"
            ).fetchone()
            config_hash = fingerprint(run_config)
            if existing and existing[0] != config_hash:
                raise ValueError("Resume refused: source/config changed; use a new pilot directory")
            connection.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('config_hash', ?)", (config_hash,)
            )
            connection.commit()
            try:
                for index in range(count):
                    if connection.execute("SELECT 1 FROM records WHERE idx=?", (index,)).fetchone():
                        continue
                    last_rejected = connection.execute(
                        "SELECT MAX(candidate) FROM attempts WHERE idx=? AND status='rejected'",
                        (index,),
                    ).fetchone()[0]
                    start = 0 if last_rejected is None else last_rejected + 1
                    for candidate in range(start, MAX_CANDIDATES):
                        initial, action, family = sample_case(seed, index, candidate, config)
                        cursor = connection.execute(
                            "INSERT INTO attempts(idx,candidate,status) VALUES (?,?,'started')",
                            (index, candidate),
                        )
                        attempt_id = cursor.lastrowid
                        connection.commit()
                        try:
                            replay = simulator.replay(initial, action)
                            if any(
                                ball.kind == "cue" and ball.status == "pocketed"
                                for ball in replay.goal.balls
                            ):
                                raise SimulationFailure(
                                    "Witness scratches; rejected by pilot policy"
                                )
                        except SimulationFailure as error:
                            LOGGER.warning(
                                "Rejected index=%s candidate=%s: %s", index, candidate, error
                            )
                            connection.execute(
                                "UPDATE attempts SET status='rejected',reason=? WHERE id=?",
                                (str(error), attempt_id),
                            )
                            connection.commit()
                            continue
                        record = ShotRecord(
                            task_id=f"pilot-{seed}-{index:06d}",
                            group_id=fingerprint(
                                {
                                    "initial": initial.endpoint_view().model_dump(mode="json"),
                                    "physics": config.model_dump(mode="json"),
                                }
                            ),
                            family=family,
                            physics=config,
                            initial=initial,
                            goal=replay.goal,
                            witness_action=action,
                            duration_s=replay.duration_s,
                            events=replay.events,
                            event_sampled_motion=replay.motion,
                            cue_initial_velocity_mps=replay.cue_velocity,
                            cue_initial_angular_velocity_radps=replay.cue_omega,
                        )
                        payload = record.model_dump_json()
                        if len(payload.encode()) > MAX_RECORD_BYTES:
                            raise RuntimeError("Record exceeds the reserved pilot storage bound")
                        with connection:
                            connection.execute(
                                "INSERT INTO records VALUES (?,?,?)",
                                (index, record.group_id, payload),
                            )
                            connection.execute(
                                "UPDATE attempts SET status='accepted' WHERE id=?", (attempt_id,)
                            )
                        break
                    else:
                        raise RuntimeError(f"No acceptable candidate for task index {index}")
                    if (index + 1) % 100 == 0:
                        LOGGER.info("Committed %s/%s pilot tasks", index + 1, count)
            finally:
                completed = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
                manifest = {
                    "run_config": run_config,
                    "config_hash": config_hash,
                    "status": "complete" if completed == count else "incomplete",
                    "accepted_tasks": completed,
                    "requested_tasks": count,
                    "real_observed_tasks": 0,
                    "partition": "development",
                    "issued_attempts": connection.execute(
                        "SELECT COUNT(*) FROM attempts"
                    ).fetchone()[0],
                    "attempt_status": dict(
                        connection.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status")
                    ),
                    "rejection_reasons": dict(
                        connection.execute(
                            "SELECT reason,COUNT(*) FROM attempts "
                            "WHERE status='rejected' GROUP BY reason"
                        )
                    ),
                }
                write_json(output / "manifest.json", manifest)
            manifest["exports"] = _export(connection, output)
            write_json(output / "manifest.json", manifest)
    return manifest


def audit_pilot(output: Path, *, replay_count: int = 20) -> dict[str, object]:
    from inverser.metrics import score_endpoint

    if replay_count < 0:
        raise ValueError("Replay count must be nonnegative")
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Cannot audit an incomplete dataset as a completed pilot")
    for name, expected in manifest["exports"].items():
        if file_sha256(output / name) != expected:
            raise ValueError(f"Export checksum mismatch: {name}")
    families: Counter[str] = Counter()
    events: Counter[str] = Counter()
    groups: set[str] = set()
    ids: set[str] = set()
    replayed = 0
    max_error = 0.0
    simulator = SnookerSimulator(PhysicsConfig())
    total = 0
    with (
        (output / "privileged.jsonl").open() as private,
        (output / "public.jsonl").open() as public,
    ):
        for index, (private_line, public_line) in enumerate(zip(private, public, strict=True)):
            record = ShotRecord.model_validate_json(private_line)
            if json.loads(public_line) != record.public_input().model_dump(mode="json"):
                raise ValueError("Public/privileged views disagree or contain unexpected fields")
            if record.task_id in ids or record.group_id in groups:
                raise ValueError("Repeated task or initial-layout group")
            ids.add(record.task_id)
            groups.add(record.group_id)
            families[record.family] += 1
            events.update(event.event_type for event in record.events)
            if index < replay_count:
                replay = simulator.replay(record.initial, record.witness_action)
                score = score_endpoint(replay.goal, record.goal, tolerance_m=1e-8)
                if not score.success or replay.events != record.events:
                    raise ValueError(f"Generating witness failed exact replay: {record.task_id}")
                max_error = max(max_error, score.max_position_error_m or 0.0)
                replayed += 1
            total += 1
    if total != manifest["accepted_tasks"]:
        raise ValueError("Manifest count does not match the exported task count")
    report = {
        "status": "passed",
        "tasks_validated": total,
        "distinct_layout_groups": len(groups),
        "source_kind": "simulated",
        "real_observed_tasks": 0,
        "families": dict(families),
        "event_counts": dict(events),
        "witnesses_replayed": replayed,
        "max_witness_endpoint_error_m": max_error,
        "model_experiments_performed": False,
        "gate": "G1 requires researcher review; this is not the frozen benchmark",
    }
    write_json(output / "audit.json", report)
    return report
