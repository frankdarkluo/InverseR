import csv
import fcntl
import gzip
import json
import logging
import multiprocessing
import os
import sqlite3
import subprocess
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import closing
from pathlib import Path
from typing import Literal, Self

import cv2
import numpy as np
from pydantic import Field, model_validator

from inverser.processing import (
    ShotWindowDetector,
    Snapshot,
    decode_window,
    export_candidate,
    write_image,
)
from inverser.real_data import (
    EpisodeMedia,
    SnookerEpisode,
    SourceProvenance,
    calibration_record,
    export_episode,
    make_episode,
    render_clean_table_pair,
)
from inverser.runtime import PROJECT_ROOT, check_storage, file_sha256, fingerprint, write_json
from inverser.schemas import Record
from inverser.table_detection import locate_table
from inverser.vision import BallAssociator, TableProjection, observe_frame

LOGGER = logging.getLogger(__name__)
SOURCE_FILES = (
    "collection.py",
    "real_data.py",
    "processing.py",
    "vision.py",
    "table_detection.py",
    "runtime.py",
)


class CollectionJob(Record):
    job_id: str
    video_path: str
    recording_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_size_bytes: int = Field(gt=0)
    source_mtime_ns: int
    group_id: str
    original_width: int = Field(gt=0)
    original_height: int = Field(gt=0)
    duration_s: float = Field(gt=0)
    owned_start_s: float = Field(ge=0)
    owned_end_s: float = Field(gt=0)
    scan_start_s: float = Field(ge=0)
    scan_end_s: float = Field(gt=0)
    corner_file: str | None
    output: str
    sample_hz: float = Field(default=5, ge=2, le=15)
    decode_width: int = Field(default=960, ge=256, le=1920)
    max_exports: int = Field(default=100, ge=1, le=200)

    @model_validator(mode="after")
    def check_coverage(self) -> Self:
        if not (
            self.scan_start_s
            <= self.owned_start_s
            < self.owned_end_s
            <= self.scan_end_s
            <= self.duration_s
        ):
            raise ValueError("Invalid scan/context ownership interval")
        return self


def ownership_windows(
    duration_s: float, *, core_seconds: float = 600, context_seconds: float = 90
) -> list[tuple[float, float, float, float]]:
    if duration_s <= 0 or not 60 <= core_seconds <= 720 or not 65 <= context_seconds <= 90:
        raise ValueError("Collection windows need positive duration and bounded core/context")
    windows = []
    start = 0.0
    while start < duration_s:
        end = min(start + core_seconds, duration_s)
        windows.append(
            (start, end, max(0.0, start - context_seconds), min(duration_s, end + context_seconds))
        )
        start = end
    return windows


def owns_candidate(job: CollectionJob, onset_s: float) -> bool:
    return job.owned_start_s <= onset_s < job.owned_end_s


def code_hashes() -> dict[str, str]:
    return {name: file_sha256(Path(__file__).with_name(name)) for name in SOURCE_FILES}


def _source_groups(
    source: Path, inventory: list[dict]
) -> tuple[dict[str, str], list[dict[str, object]]]:
    groups: dict[str, str] = {}
    previews: dict[tuple[float, str], list[str]] = {}
    for item in inventory:
        if item.get("kind") != "video":
            continue
        groups[item["path"]] = item["sha256"]
        preview = source / "reports" / "contact_sheets" / f"{Path(item['path']).stem}.png"
        if preview.is_file():
            key = (round(item["video"]["duration_s"], 2), file_sha256(preview))
            previews.setdefault(key, []).append(item["path"])
    possible: list[dict[str, object]] = []
    for (_, preview_hash), names in previews.items():
        if len(names) > 1:
            group = fingerprint({"sampled_preview": preview_hash, "filenames": sorted(names)})
            for name in names:
                groups[name] = group
            possible.append({"files": names, "group_id": group})
    return groups, possible


def prepare_jobs(
    source: Path, output: Path, *, sample_hz: float = 5, core_seconds: float = 600
) -> tuple[list[CollectionJob], dict[str, object]]:
    source = source.resolve()
    output = output.resolve()
    if not output.is_relative_to((PROJECT_ROOT / "data").resolve()):
        raise ValueError("Collection output must stay under InverseR/data")
    inventory_path = source / "inventory.json"
    inventory = json.loads(inventory_path.read_text())["files"]
    groups, possible_groups = _source_groups(source, inventory)
    jobs = []
    for item in inventory:
        if item.get("kind") != "video":
            continue
        path = source / "files" / item["path"]
        if not path.resolve().is_relative_to(source / "files") or path.is_symlink():
            raise ValueError("Source video escapes its acquired file directory")
        stat = path.stat()
        if not item["checksum_verified"] or stat.st_size != item["bytes"]:
            raise ValueError("Source differs from its acquisition inventory")
        info = item["video"]
        if sample_hz > info["fps"]:
            raise ValueError("Collection sampling exceeds a source frame rate")
        corners = path.with_name(f"{path.stem}_corners.json")
        for index, (start, end, scan_start, scan_end) in enumerate(
            ownership_windows(info["duration_s"], core_seconds=core_seconds)
        ):
            job_id = f"{path.stem}-window-{index:05d}"
            jobs.append(
                CollectionJob(
                    job_id=job_id,
                    video_path=str(path),
                    recording_id=path.stem,
                    source_sha256=item["sha256"],
                    source_size_bytes=stat.st_size,
                    source_mtime_ns=stat.st_mtime_ns,
                    group_id=groups[item["path"]],
                    original_width=info["width"],
                    original_height=info["height"],
                    duration_s=info["duration_s"],
                    owned_start_s=start,
                    owned_end_s=end,
                    scan_start_s=scan_start,
                    scan_end_s=scan_end,
                    corner_file=str(corners) if corners.is_file() else None,
                    output=str(output / "windows" / path.stem / f"{index:05d}"),
                    sample_hz=sample_hz,
                )
            )
    if not jobs:
        raise ValueError("No acquired videos in source inventory")
    jobs.sort(key=lambda job: (job.owned_start_s, job.recording_id))
    manifest = {
        "source": str(source),
        "output": str(output),
        "inventory_sha256": file_sha256(inventory_path),
        "schema_version": "snooker-observation-v1",
        "sample_hz": sample_hz,
        "core_seconds": core_seconds,
        "context_seconds": 90,
        "code_hashes": code_hashes(),
        "videos": len({job.recording_id for job in jobs}),
        "jobs": len(jobs),
        "owned_video_seconds": sum(job.owned_end_s - job.owned_start_s for job in jobs),
        "sampled_preview_duplicate_groups": possible_groups,
        "duplicate_group_note": (
            "Identical sampled contact sheets and duration share a split group; "
            "not proof of identical complete videos. All sources are still scanned."
        ),
        "training_or_formal_evaluation": False,
    }
    return jobs, manifest


def _projection_for_frame(
    frame: np.ndarray,
    job: CollectionJob,
    width: int,
    height: int,
    *,
    prefer_sidecar: bool,
) -> tuple[TableProjection | None, Literal["supplied_corners", "automatic_felt_quad"]]:
    if prefer_sidecar and job.corner_file:
        points = json.loads(Path(job.corner_file).read_text())
        scale_x, scale_y = width / job.original_width, height / job.original_height
        scaled = [[point[0] * scale_x, point[1] * scale_y] for point in points]
        projection = TableProjection(scaled, width, height)
        rectified = projection.warp(frame)
        hsv = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV)
        felt = (hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 100) & (hsv[:, :, 1] >= 40)
        if float(felt[10:-10, 10:-10].mean()) >= 0.70:
            return projection, "supplied_corners"
        LOGGER.info(
            "Supplied corners do not cover a clear table view in %s; trying geometry", job.job_id
        )
    found = locate_table(frame)
    if found is None:
        return None, "automatic_felt_quad"
    return (
        TableProjection(
            [list(point) for point in found.points],
            width,
            height,
            corner_order="ordered_short_edge",
        ),
        "automatic_felt_quad",
    )


def collect_window(
    job: CollectionJob, expected_code: dict[str, str], attempt: int = 1
) -> dict[str, object]:
    if code_hashes() != expected_code:
        raise RuntimeError("Collection source changed; refuse mixed-code results")
    source = Path(job.video_path)
    stat = source.stat()
    if stat.st_size != job.source_size_bytes or stat.st_mtime_ns != job.source_mtime_ns:
        raise ValueError("Source video changed during corpus collection")
    output = Path(job.output) / f"attempt_{attempt:02d}"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "job.json", job.model_dump(mode="json"))
    cv2.setNumThreads(1)
    decode_width = min(job.decode_width, job.original_width)
    decode_height = 2 * round(job.original_height * decode_width / job.original_width / 2)
    projector: TableProjection | None = None
    method: Literal["supplied_corners", "automatic_felt_quad"] = "automatic_felt_quad"
    tracker = BallAssociator()
    detector = ShotWindowDetector(0.02619375, cue_trigger_only=True)
    frames = 0
    valid = 0
    clear = 0
    outside_owner = 0
    export_cap = 0
    view_relocations = 0
    records: list[str] = []
    calibration_counter: Counter[str] = Counter()
    next_location_check = job.scan_start_s
    finished = False
    start = time.monotonic()
    provenance = SourceProvenance(
        collection="hf_snooker_videos",
        recording_id=job.recording_id,
        original_file=job.video_path,
        original_sha256=job.source_sha256,
        group_id=job.group_id,
        media_license="unverified",
        content_duplicate_group=job.group_id if job.group_id != job.source_sha256 else None,
    )
    report: dict[str, object] = {}
    try:
        with gzip.open(
            output / "frame_observations.jsonl.gz", "wt", encoding="utf-8"
        ) as observations:
            for timestamp, frame in decode_window(
                source,
                start_s=job.scan_start_s,
                duration_s=job.scan_end_s - job.scan_start_s,
                sample_hz=job.sample_hz,
                width=job.original_width,
                height=job.original_height,
                scale_width=decode_width,
            ):
                frames += 1
                if projector is None and timestamp >= next_location_check:
                    projector, method = _projection_for_frame(
                        frame,
                        job,
                        decode_width,
                        decode_height,
                        prefer_sidecar=job.corner_file is not None,
                    )
                    next_location_check = timestamp + 10
                    if projector:
                        calibration_counter[method] += 1
                        write_json(
                            output / f"calibration_{sum(calibration_counter.values()):03d}.json",
                            calibration_record(projector, method, "nominal_regulation").model_dump(
                                mode="json"
                            ),
                        )
                if projector is None:
                    observations.write(
                        json.dumps({"time_s": timestamp, "table_detected": False}) + "\n"
                    )
                    continue
                observation, rectified = observe_frame(frame, projector, tracker, timestamp)
                observations.write(json.dumps(observation.json(), allow_nan=False) + "\n")
                valid += observation.view_valid
                clear += observation.visibility_clear
                if not observation.view_valid and timestamp >= next_location_check:
                    projector = None
                    tracker = BallAssociator()
                    detector = ShotWindowDetector(0.02619375, cue_trigger_only=True)
                    next_location_check = timestamp
                    view_relocations += 1
                    continue
                snapshot = Snapshot(observation, frame, rectified)
                candidate = detector.update(snapshot)
                if candidate is None:
                    continue
                onset = candidate.motion_onset_s
                if onset is None:
                    raise ValueError("Corpus candidate lacks its estimated onset")
                if not owns_candidate(job, onset):
                    outside_owner += 1
                    continue
                if len(records) >= job.max_exports:
                    export_cap += 1
                    continue
                index = len(records)
                exported = export_candidate(
                    candidate,
                    source,
                    output,
                    index,
                    job.source_sha256,
                    source_duration_s=job.duration_s,
                )
                directory = Path(exported.observations).parent
                initial_bev, terminal_bev = (
                    directory / "initial_bev.jpg",
                    directory / "terminal_bev.jpg",
                )
                write_image(initial_bev, candidate.initial.rectified)
                write_image(terminal_bev, candidate.goal.rectified)
                media = EpisodeMedia(
                    initial_source=str(directory / "initial_source.jpg"),
                    terminal_source=str(directory / "goal_source.jpg"),
                    initial_bev=str(initial_bev),
                    terminal_bev=str(terminal_bev),
                    clean_table_pair=str(directory / "clean_table_pair.png"),
                    source_clip=exported.clip,
                    source_clip_start_s=max(0, candidate.initial.observation.time_s - 0.5),
                    source_clip_end_s=min(candidate.goal.observation.time_s + 0.5, job.duration_s),
                )
                episode = make_episode(
                    provenance,
                    calibration_record(projector, method, "nominal_regulation"),
                    candidate.trace,
                    onset,
                    job.sample_hz,
                    media,
                )
                export_episode(directory / "structured.json", episode)
                render_clean_table_pair(directory / "clean_table_pair.png", episode)
                records.append(str(directory / "structured.json"))
                if len(records) % 10 == 0:
                    write_json(
                        output / "state.json",
                        {
                            "status": "collecting",
                            "frames": frames,
                            "raw_candidates": len(records),
                            "time_s": timestamp,
                        },
                    )
            finished = True
    finally:
        report = {
            "status": "complete" if finished else "failed",
            "job_id": job.job_id,
            "recording_id": job.recording_id,
            "owned_start_s": job.owned_start_s,
            "owned_end_s": job.owned_end_s,
            "scan_start_s": job.scan_start_s,
            "scan_end_s": job.scan_end_s,
            "sampled_frames": frames,
            "valid_view_frames": valid,
            "visibility_clear_frames": clear,
            "raw_candidate_records": len(records),
            "records": records,
            "context_candidates_not_owned": outside_owner,
            "candidate_export_cap_exclusions": export_cap,
            "calibration_methods": dict(calibration_counter),
            "view_relocations": view_relocations,
            "elapsed_seconds": time.monotonic() - start,
            "accepted_benchmark_tasks": 0,
            "code_hashes": expected_code,
        }
        write_json(output / "report.json", report)
        write_json(
            output / "state.json", {key: value for key, value in report.items() if key != "records"}
        )
    return report


def _publish_collection_state(
    connection: sqlite3.Connection, output: Path, *, runner_active: bool = True
) -> dict[str, object]:
    counts = dict(connection.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status"))
    owned = connection.execute(
        "SELECT COALESCE(SUM(owned_seconds),0) FROM jobs WHERE status='complete'"
    ).fetchone()[0]
    count = connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    frames = connection.execute(
        "SELECT COALESCE(SUM(sampled_frames),0) FROM jobs WHERE status='complete'"
    ).fetchone()[0]
    status = "collecting" if runner_active else "paused"
    if not counts.get("pending") and not counts.get("running"):
        status = "failed" if counts.get("failed") else "complete"
    state = {
        "status": status,
        "runner_active": runner_active,
        "jobs": counts,
        "collected_video_hours": owned / 3600,
        "sampled_frames": frames,
        "raw_candidates": count,
        "accepted_benchmark_tasks": 0,
        "schema": "snooker-observation-v1",
        "model_or_api_calls": 0,
    }
    write_json(output / "collection_state.json", state)
    return state


def _index_episodes(connection: sqlite3.Connection, report: dict[str, object]) -> None:
    records = report["records"]
    if not isinstance(records, list):
        raise ValueError("Worker did not return a structured record list")
    for path in records:
        episode = SnookerEpisode.model_validate_json(Path(path).read_text())
        connection.execute(
            "INSERT OR IGNORE INTO episodes VALUES (?,?,?,?,?,?,?)",
            (
                episode.episode_id,
                episode.source.recording_id,
                episode.source.group_id,
                episode.start_s,
                episode.terminal_s,
                str(path),
                episode.model_dump_json(),
            ),
        )


def export_collection(connection: sqlite3.Connection, output: Path) -> None:
    temporary = output / ".episodes.jsonl.tmp"
    with temporary.open("w") as stream:
        for (payload,) in connection.execute(
            "SELECT payload FROM episodes ORDER BY recording_id,start_s"
        ):
            stream.write(payload + "\n")
    temporary.replace(output / "episodes.jsonl")
    with (output / "episode_index.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["episode_id", "recording_id", "group_id", "start_s", "terminal_s", "structured_record"]
        )
        writer.writerows(
            connection.execute(
                "SELECT id,recording_id,group_id,start_s,end_s,path FROM episodes "
                "ORDER BY recording_id,start_s"
            )
        )


def collect_corpus(
    source: Path,
    output: Path,
    *,
    workers: int = 4,
    sample_hz: float = 5,
    limit_jobs: int | None = None,
) -> dict[str, object]:
    if not 1 <= workers <= 4:
        raise ValueError("Use 1-4 collection workers within the 16-CPU budget")
    if limit_jobs is not None and limit_jobs < 1:
        raise ValueError("A diagnostic job limit must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    jobs, manifest = prepare_jobs(source, output, sample_hz=sample_hz)
    run_code = code_hashes()
    if run_code != manifest["code_hashes"]:
        raise RuntimeError("Code changed while preparing the collection manifest")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    recovery_manifest = output / "recovery_manifest.json"
    if recovery_manifest.exists():
        manifest["recovery"] = json.loads(recovery_manifest.read_text())
    check_storage(workers * 3_000_000_000)
    with (output / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "collection_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Collection source/code/config changed; use a new version")
        write_json(manifest_path, manifest)
        with closing(sqlite3.connect(output / "collection.sqlite")) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempt INTEGER NOT NULL DEFAULT 0,owned_seconds REAL NOT NULL,
                    sampled_frames INTEGER NOT NULL DEFAULT 0,report_path TEXT,error TEXT
                );
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,recording_id TEXT NOT NULL,group_id TEXT NOT NULL,
                    start_s REAL NOT NULL,end_s REAL NOT NULL,
                    path TEXT NOT NULL,payload TEXT NOT NULL
                );
                """
            )
            for job in jobs:
                connection.execute(
                    "INSERT OR IGNORE INTO jobs(id,payload,owned_seconds) VALUES (?,?,?)",
                    (job.job_id, job.model_dump_json(), job.owned_end_s - job.owned_start_s),
                )
            connection.execute("UPDATE jobs SET status='pending' WHERE status='running'")
            connection.commit()
            _publish_collection_state(connection, output)
            rows = list(
                connection.execute(
                    "SELECT id,payload,attempt FROM jobs WHERE status='pending' ORDER BY rowid"
                )
            )
            if limit_jobs is not None:
                rows = rows[:limit_jobs]
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                pending: dict[Future[dict[str, object]], tuple[CollectionJob, int]] = {}
                cursor = iter(rows)

                def submit_next() -> bool:
                    try:
                        job_id, payload, previous_attempt = next(cursor)
                    except StopIteration:
                        return False
                    job = CollectionJob.model_validate_json(payload)
                    attempt = previous_attempt + 1
                    connection.execute(
                        "UPDATE jobs SET status='running',attempt=? WHERE id=?", (attempt, job_id)
                    )
                    connection.commit()
                    future = executor.submit(collect_window, job, run_code, attempt)
                    pending[future] = (job, attempt)
                    _publish_collection_state(connection, output)
                    return True

                for _ in range(workers):
                    submit_next()
                completed = 0
                while pending:
                    done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    for future in done:
                        job, attempt = pending.pop(future)
                        try:
                            report = future.result()
                            if report["status"] != "complete":
                                raise RuntimeError("Worker returned an incomplete result")
                            with connection:
                                _index_episodes(connection, report)
                                connection.execute(
                                    "UPDATE jobs SET status='complete',sampled_frames=?,"
                                    "report_path=?,error=NULL WHERE id=?",
                                    (
                                        report["sampled_frames"],
                                        str(
                                            Path(job.output)
                                            / f"attempt_{attempt:02d}"
                                            / "report.json"
                                        ),
                                        job.job_id,
                                    ),
                                )
                            completed += 1
                            LOGGER.info(
                                "Completed %s: %s raw candidates",
                                job.job_id,
                                report["raw_candidate_records"],
                            )
                        except (
                            OSError,
                            ValueError,
                            RuntimeError,
                            subprocess.CalledProcessError,
                            subprocess.TimeoutExpired,
                        ) as error:
                            LOGGER.error("Collection job failed %s: %s", job.job_id, error)
                            with connection:
                                connection.execute(
                                    "UPDATE jobs SET status='failed',error=? WHERE id=?",
                                    (f"{type(error).__name__}: {error}", job.job_id),
                                )
                        if completed % 10 == 0:
                            check_storage(workers * 3_000_000_000)
                        if completed % 50 == 0:
                            export_collection(connection, output)
                        submit_next()
                    _publish_collection_state(connection, output)
            export_collection(connection, output)
            state = _publish_collection_state(connection, output, runner_active=False)
            failed_count = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status='failed'"
            ).fetchone()[0]
            if failed_count:
                raise RuntimeError(
                    "Some corpus jobs failed; inspect the recorded errors before retrying"
                )
            return state


def recover_failed_corpus(
    source: Path, parent: Path, output: Path, *, workers: int = 4
) -> dict[str, object]:
    """Preserve the frozen parent run and retry only its failed jobs in a new version."""
    parent = parent.resolve()
    output = output.resolve()
    data_root = (PROJECT_ROOT / "data").resolve()
    if not parent.is_relative_to(data_root) or not output.is_relative_to(data_root):
        raise ValueError("Corpus recovery must remain inside InverseR/data")
    if output == parent or output.is_relative_to(parent):
        raise ValueError("Recovery needs a separate directory, not the frozen parent output")
    if not 1 <= workers <= 4:
        raise ValueError("Use 1-4 recovery workers within the approved budget")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".recovery.lock").open("a") as recovery_lock:
        fcntl.flock(recovery_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent_manifest_path = parent / "collection_manifest.json"
        original = json.loads(parent_manifest_path.read_text())
        if Path(original["source"]).resolve() != source.resolve():
            raise ValueError("Recovery source does not match the parent collection")
        inventory_path = source / "inventory.json"
        if file_sha256(inventory_path) != original["inventory_sha256"]:
            raise ValueError("Source inventory changed since the frozen parent run")
        sample_hz = original["sample_hz"]
        receipt_path = output / "recovery_manifest.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt["parent_manifest_sha256"] != file_sha256(parent_manifest_path)
                or receipt["recovery_code_hashes"] != code_hashes()
            ):
                raise ValueError("Recovery provenance changed; use a new recovery version")
        else:
            if any(path.name != ".recovery.lock" for path in output.iterdir()):
                raise ValueError("Refusing to overwrite an existing untracked recovery directory")
            with (parent / ".runner.lock").open("a") as parent_lock:
                fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                database = parent / "collection.sqlite"
                with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as original_db:
                    active = original_db.execute(
                        "SELECT COUNT(*) FROM jobs WHERE status IN ('pending','running')"
                    ).fetchone()[0]
                    if active:
                        raise ValueError("Finish or stop the parent collection before recovery")
                    failures = original_db.execute(
                        "SELECT id,attempt,error FROM jobs WHERE status='failed' ORDER BY id"
                    ).fetchall()
                    if not failures:
                        raise ValueError("The parent collection has no failed jobs to recover")
                    new_jobs, _ = prepare_jobs(source, output, sample_hz=sample_hz)
                    job_by_id = {job.job_id: job for job in new_jobs}
                    check_storage(database.stat().st_size + workers * 3_000_000_000)
                    temporary = output / ".collection.sqlite.tmp"
                    try:
                        with closing(sqlite3.connect(temporary)) as recovery_db:
                            original_db.backup(recovery_db)
                            inherited_jobs = recovery_db.execute(
                                "SELECT COUNT(*) FROM jobs WHERE status='complete'"
                            ).fetchone()[0]
                            inherited_episodes = recovery_db.execute(
                                "SELECT COUNT(*) FROM episodes"
                            ).fetchone()[0]
                            for job_id, _, _ in failures:
                                if job_id not in job_by_id:
                                    raise ValueError(f"Recovery window disappeared: {job_id}")
                                recovery_db.execute(
                                    "UPDATE jobs SET payload=?,status='pending',attempt=0,"
                                    "sampled_frames=0,report_path=NULL,error=NULL WHERE id=?",
                                    (job_by_id[job_id].model_dump_json(), job_id),
                                )
                            recovery_db.commit()
                        temporary.replace(output / "collection.sqlite")
                    finally:
                        if temporary.exists():
                            temporary.unlink()
                    write_json(
                        receipt_path,
                        {
                            "parent_collection": str(parent),
                            "parent_manifest_sha256": file_sha256(parent_manifest_path),
                            "parent_code_hashes": original["code_hashes"],
                            "recovery_code_hashes": code_hashes(),
                            "inherited_complete_jobs": inherited_jobs,
                            "inherited_raw_episodes": inherited_episodes,
                            "failed_jobs": [
                                {"job_id": name, "parent_attempt": attempt, "parent_error": error}
                                for name, attempt, error in failures
                            ],
                            "scope": "Retry failed windows only; do not relabel inherited records",
                        },
                    )
        return collect_corpus(source, output, workers=workers, sample_hz=sample_hz)
