import csv
import fcntl
import json
import logging
import re
import sqlite3
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from contextlib import closing
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from inverser.author_uploads import (
    UploadedAsset,
    _clip_background,
    convert_uploaded_clip,
    verify_uploaded_asset,
)
from inverser.real_data import SnookerEpisode
from inverser.runtime import PROJECT_ROOT, check_storage, file_sha256, fingerprint, write_json
from inverser.schemas import Record

LOGGER = logging.getLogger(__name__)
ANNOTATION_ROLES = {"manual_path_image", "tracking_image", "reconstructed_animation"}
CONVERSION_CODE = (
    "author_sync.py",
    "author_uploads.py",
    "real_data.py",
    "vision.py",
    "table_detection.py",
    "processing.py",
    "runtime.py",
)


class RecordingPlan(Record):
    group_id: str
    aliases: tuple[str, ...]
    selected_original: str
    raw_capture: str | None
    background: str | None
    annotation_paths: tuple[str, ...]
    manual_path_images: tuple[str, ...]
    ready: bool
    deferred_reasons: tuple[str, ...]


def plan_recordings(assets: dict[str, UploadedAsset]) -> tuple[RecordingPlan, ...]:
    originals = [
        asset
        for asset in assets.values()
        if asset.game == "snooker" and asset.role == "original_video"
    ]
    parent = list(range(len(originals)))

    def component(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, original in enumerate(originals):
        raw = assets.get(str(PurePosixPath(original.path).with_suffix(".dat")))
        keys = []
        if original.stable and original.sha256:
            keys.append(("video", original.sha256))
        if raw is not None and raw.stable and raw.sha256:
            keys.append(("raw", raw.sha256))
        for key in keys:
            if key in seen:
                parent[component(index)] = component(seen[key])
            else:
                seen[key] = index
    clusters: dict[int, list[UploadedAsset]] = defaultdict(list)
    for index, original in enumerate(originals):
        clusters[component(index)].append(original)
    plans = []
    for clips in clusters.values():

        def preference(original: UploadedAsset) -> tuple[bool, bool, str]:
            raw = assets.get(str(PurePosixPath(original.path).with_suffix(".dat")))
            ready_capture = original.stable and raw is not None and raw.stable
            folder = str(PurePosixPath(original.path).parent) + "/"
            manual = any(
                asset.stable and asset.role == "manual_path_image" and asset.path.startswith(folder)
                for asset in assets.values()
            )
            return not ready_capture, not manual, original.path

        clips.sort(key=preference)
        selected = clips[0]
        raw = assets.get(str(PurePosixPath(selected.path).with_suffix(".dat")))
        background = _clip_background(Path(), selected, assets)
        manual_images = sorted(
            {
                asset.path
                for clip in clips
                for asset in assets.values()
                if asset.stable
                and asset.role == "manual_path_image"
                and asset.path.startswith(str(PurePosixPath(clip.path).parent) + "/")
            }
        )
        prefix = str(PurePosixPath(selected.path).parent) + "/"
        annotations = sorted(
            set(manual_images)
            | {
                asset.path
                for asset in assets.values()
                if asset.stable and asset.role in ANNOTATION_ROLES and asset.path.startswith(prefix)
            }
        )
        reasons = []
        if not selected.stable:
            reasons.append("original_video_still_uploading")
        if raw is None or not raw.stable:
            reasons.append("complete_stable_raw_capture_not_available")
        if background is None:
            reasons.append("stable_source_background_not_available")
        raw_hashes = {
            asset.sha256
            for clip in clips
            if (asset := assets.get(str(PurePosixPath(clip.path).with_suffix(".dat")))) is not None
            and asset.stable
            and asset.sha256
        }
        if len(raw_hashes) > 1:
            reasons.append("identical_video_aliases_have_conflicting_raw_captures")
        group_id = (
            next(iter(raw_hashes))
            if len(raw_hashes) == 1
            else fingerprint(
                {
                    "stable_video_hashes": sorted(
                        asset.sha256 for asset in clips if asset.stable and asset.sha256
                    ),
                    "paths": sorted(asset.path for asset in clips),
                }
            )
        )
        plans.append(
            RecordingPlan(
                group_id=group_id,
                aliases=tuple(sorted(asset.path for asset in clips)),
                selected_original=selected.path,
                raw_capture=raw.path if raw is not None and raw.stable else None,
                background=background.path if background is not None else None,
                annotation_paths=tuple(annotations),
                manual_path_images=tuple(manual_images),
                ready=not reasons,
                deferred_reasons=tuple(reasons),
            )
        )
    return tuple(sorted(plans, key=lambda plan: (not bool(plan.manual_path_images), plan.group_id)))


def _reported_jaccard(path: Path) -> dict[str, float]:
    scores = {}
    for line in path.read_text().splitlines():
        matched = re.fullmatch(r"(.+):\s*([0-9.eE+-]+)\s*", line.strip())
        if matched is None:
            continue
        label, value = matched.groups()
        label = label.upper().replace(" ", "_")
        for name in (
            "MEDIAN_FLOW",
            "VACHASPATI",
            "BOOSTING",
            "MOLT",
            "LEGG",
            "MOSSE",
            "CSRT",
            "KCF",
            "TLD",
            "MIL",
        ):
            if label.startswith(name):
                scores[name] = float(value)
                break
    return scores


def audit_manual_paths(
    source: Path, assets: dict[str, UploadedAsset], output: Path, hash_cache: Path | None
) -> dict[str, object]:
    rows = []
    skipped = []
    methods: dict[str, list[float]] = defaultdict(list)
    mismatches = 0
    manuals = [
        asset
        for asset in assets.values()
        if asset.game == "snooker" and asset.role == "manual_path_image" and asset.stable
    ]
    for manual in sorted(manuals, key=lambda asset: asset.path):
        manual_path = verify_uploaded_asset(source, manual, hash_cache)
        target = cv2.imread(str(manual_path), cv2.IMREAD_GRAYSCALE)
        if target is None or not np.any(target < 128):
            raise ValueError(f"Manual path image cannot be interpreted: {manual.path}")
        directory = PurePosixPath(manual.path).parent
        coefficients = assets.get(str(directory / "Jaccard_Coefficient.txt"))
        reported = {}
        if coefficients is not None and coefficients.stable:
            reported = _reported_jaccard(verify_uploaded_asset(source, coefficients, hash_cache))
        for predicted in assets.values():
            if (
                not predicted.stable
                or predicted.role != "tracking_image"
                or not predicted.path.startswith(str(directory / "TrackingResults") + "/")
            ):
                continue
            predicted_path = verify_uploaded_asset(source, predicted, hash_cache)
            image = cv2.imread(str(predicted_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Algorithm path image is unreadable: {predicted.path}")
            if image.shape != target.shape:
                skipped.append(
                    {
                        "manual": manual.path,
                        "prediction": predicted.path,
                        "reason": "native_canvas_shape_mismatch_no_resizing_applied",
                    }
                )
                continue
            a, b = target < 128, image < 128
            intersection, union = int(np.count_nonzero(a & b)), int(np.count_nonzero(a | b))
            score = intersection / union
            method = PurePosixPath(predicted.path).stem.removesuffix("_track").upper()
            expected = reported.get(method)
            discrepancy = abs(score - expected) if expected is not None else None
            mismatches += discrepancy is not None and discrepancy > 1e-5
            methods[method].append(score)
            rows.append(
                {
                    "recording": str(directory),
                    "method": method,
                    "manual_image": manual.path,
                    "algorithm_image": predicted.path,
                    "manual_sha256": manual.sha256,
                    "algorithm_sha256": predicted.sha256,
                    "raster_width": target.shape[1],
                    "raster_height": target.shape[0],
                    "path_iou": score,
                    "source_reported_jaccard": expected,
                    "absolute_reported_score_difference": discrepancy,
                    "timed_ball_identity_or_action_accuracy": False,
                }
            )
    report = {
        "manual_images_available": len(manuals),
        "native_raster_comparisons": len(rows),
        "binary_threshold": 128,
        "same_canvas_only": True,
        "resizing_or_alignment_fitted": False,
        "source_score_discrepancies_over_1e_5": mismatches,
        "method_mean_path_iou": {
            name: float(np.mean(values)) for name, values in sorted(methods.items())
        },
        "method_comparison_counts": {name: len(values) for name, values in sorted(methods.items())},
        "skipped": skipped,
        "rows": rows,
        "interpretation": (
            "Reproduced source-algorithm path-image metrics, not our model performance "
            "or time-indexed action GT."
        ),
    }
    write_json(output / "manual_path_audit.json", report)
    with (output / "manual_path_scores.csv").open("w", newline="") as stream:
        fields = [
            "recording",
            "method",
            "path_iou",
            "source_reported_jaccard",
            "absolute_reported_score_difference",
            "manual_image",
            "algorithm_image",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    return {key: value for key, value in report.items() if key != "rows"}


def sync_snooker_uploads(
    inventory: Path, output: Path, *, max_new_clips: int = 20, hash_cache: Path | None = None
) -> dict[str, object]:
    if not 1 <= max_new_clips <= 100:
        raise ValueError("One sync processes at most 1-100 new stable recordings")
    snapshot = json.loads(inventory.read_text())
    if snapshot.get("schema_version") != "author-direct-upload-v1":
        raise ValueError("Expected a stable-file direct-upload snapshot")
    source = Path(snapshot["source"]).resolve()
    output = output.resolve()
    data_root = (PROJECT_ROOT / "data").resolve()
    if (
        not source.is_relative_to(data_root)
        or not output.is_relative_to(data_root)
        or source == output
        or output.is_relative_to(source)
    ):
        raise ValueError("Sync outputs must be separate from uploaded source files")
    assets = {asset.path: asset for asset in map(UploadedAsset.model_validate, snapshot["assets"])}
    plans = plan_recordings(assets)
    code = {name: file_sha256(Path(__file__).with_name(name)) for name in CONVERSION_CODE}
    output.mkdir(parents=True, exist_ok=True)
    check_storage(2_000_000_000)
    cv2.setNumThreads(1)
    with (output / ".sync.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = output / "pipeline_config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != code:
            raise ValueError("Conversion code changed; use a new pipeline output version")
        write_json(config_path, code)
        batch = output / "snapshots" / file_sha256(inventory)
        batch.mkdir(parents=True, exist_ok=True)
        write_json(
            batch / "recording_plan.json",
            {
                "inventory": str(inventory.resolve()),
                "plans": [plan.model_dump(mode="json") for plan in plans],
            },
        )
        reports = []
        converted = reused = failed = pending_limit = 0
        with closing(sqlite3.connect(output / "sync.sqlite")) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS conversions ("
                "input_hash TEXT PRIMARY KEY,group_id TEXT NOT NULL,status TEXT NOT NULL,"
                "attempt INTEGER NOT NULL,record_path TEXT,report TEXT,error TEXT)"
            )
            for plan in plans:
                if not plan.ready:
                    LOGGER.info("Deferred %s: %s", plan.selected_original, plan.deferred_reasons)
                    continue
                input_names = (
                    plan.selected_original,
                    plan.raw_capture,
                    plan.background,
                    *plan.annotation_paths,
                )
                dependencies = {
                    name: assets[name].sha256 for name in input_names if name is not None
                }
                input_hash = fingerprint(
                    {"inputs": dependencies, "group_id": plan.group_id, "code": code}
                )
                existing = connection.execute(
                    "SELECT status,attempt,record_path,report FROM conversions WHERE input_hash=?",
                    (input_hash,),
                ).fetchone()
                if existing and existing[0] == "complete":
                    for name in dependencies:
                        verify_uploaded_asset(source, assets[name], hash_cache)
                    if existing[2] is None:
                        raise ValueError("Complete conversion has no structured record reference")
                    record = SnookerEpisode.model_validate_json(Path(existing[2]).read_text())
                    if record.source.group_id != plan.group_id:
                        raise ValueError("Cached record and duplicate-source group disagree")
                    reused_report = json.loads(existing[3])
                    reused_report["source_aliases"] = list(plan.aliases)
                    reports.append(reused_report)
                    reused += 1
                    continue
                if converted + failed >= max_new_clips:
                    pending_limit += 1
                    continue
                attempt = existing[1] + 1 if existing else 1
                destination = output / "records" / input_hash / f"attempt_{attempt:02d}"
                with connection:
                    connection.execute(
                        "INSERT OR REPLACE INTO conversions VALUES (?,?,?, ?,NULL,NULL,NULL)",
                        (input_hash, plan.group_id, "running", attempt),
                    )
                try:
                    report = convert_uploaded_clip(
                        source,
                        assets[plan.selected_original],
                        assets,
                        destination,
                        hash_cache=hash_cache,
                        annotation_paths=plan.annotation_paths,
                        group_id=plan.group_id,
                    )
                    report["source_aliases"] = list(plan.aliases)
                    record_path = report["structured_record"]
                    if not isinstance(record_path, str):
                        raise ValueError("Conversion has no structured record path")
                    SnookerEpisode.model_validate_json(Path(record_path).read_text())
                    write_json(destination / "report.json", report)
                    with connection:
                        connection.execute(
                            "UPDATE conversions SET status='complete',record_path=?,report=?,"
                            "error=NULL WHERE input_hash=?",
                            (record_path, json.dumps(report, allow_nan=False), input_hash),
                        )
                    reports.append(report)
                    converted += 1
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    ET.ParseError,
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as error:
                    LOGGER.error("Failed snooker recording %s: %s", plan.selected_original, error)
                    with connection:
                        connection.execute(
                            "UPDATE conversions SET status='failed',error=? WHERE input_hash=?",
                            (f"{type(error).__name__}: {error}", input_hash),
                        )
                    failed += 1
            manual = audit_manual_paths(source, assets, batch, hash_cache)
            with (batch / "episodes.jsonl").open("w") as stream:
                for report in reports:
                    record_path = report["structured_record"]
                    if not isinstance(record_path, str):
                        raise ValueError("Invalid experiment-record path")
                    episode = SnookerEpisode.model_validate_json(Path(record_path).read_text())
                    stream.write(episode.model_dump_json() + "\n")
            with (batch / "episode_index.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "episode_id",
                        "source_group",
                        "record_path",
                        "manual_path_images",
                        "source_aliases",
                    ]
                )
                for report in reports:
                    record_path = report["structured_record"]
                    if not isinstance(record_path, str):
                        raise ValueError("Invalid record reference")
                    episode = SnookerEpisode.model_validate_json(Path(record_path).read_text())
                    writer.writerow(
                        [
                            episode.episode_id,
                            episode.source.group_id,
                            record_path,
                            report["original_manual_path_images"],
                            ";".join(report["source_aliases"]),
                        ]
                    )
            summary: dict[str, object] = {
                "status": "synced_snapshot" if not failed else "snapshot_with_conversion_failures",
                "inventory": str(inventory.resolve()),
                "original_snooker_video_files": sum(
                    asset.game == "snooker" and asset.role == "original_video"
                    for asset in assets.values()
                ),
                "content_groups": len(plans),
                "ready_content_groups": sum(plan.ready for plan in plans),
                "deferred_content_groups": sum(not plan.ready for plan in plans),
                "converted_this_run": converted,
                "reused_this_run": reused,
                "failed_this_run": failed,
                "ready_not_processed_due_limit": pending_limit,
                "snapshot_records": len(reports),
                "records_with_manual_paths": sum(
                    bool(report["original_manual_path_images"]) for report in reports
                ),
                "manual_path_audit": manual,
                "experiment_index": str(batch / "episode_index.csv"),
                "episodes_jsonl": str(batch / "episodes.jsonl"),
                "source_files_modified": False,
                "accepted_benchmark_tasks": 0,
                "model_or_api_calls": 0,
                "split_requirement": (
                    "One group per raw recording; aliases/manual copies must not cross splits."
                ),
            }
            write_json(batch / "summary.json", summary)
            write_json(output / "latest.json", summary | {"snapshot_output": str(batch)})
            if failed:
                raise RuntimeError(
                    "Some stable clips failed conversion; inspect sync.sqlite before retrying"
                )
            return summary
