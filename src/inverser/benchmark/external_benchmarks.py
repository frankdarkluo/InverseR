"""Verify pinned external assets without fetching data or making model calls."""

import io
import json
import subprocess
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from PIL import Image

from inverser.archive import inspect_members
from inverser.runtime import PROJECT_ROOT, file_sha256, fingerprint

REPOSITORIES = {"DeepPHY", "PhysBench", "cuetip", "dino_wm", "phyre"}
SNAPSHOTS = {"USC-GVL/PhysBench", "facebook/dinov2-small", "MMMU/MMMU", "xai-org/RealWorldQA"}


def verify_external_suites(root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Asset success is separate from simulator readiness; never synthesize missing labels."""
    root = root.resolve()
    manifest_path = root / "data/external/manifest.json"
    if not manifest_path.is_file():
        return {"status": "incomplete", "errors": ["missing_external_manifest"]}
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    checked = 0
    names = {row["name"] for row in manifest.get("repositories", [])}
    repos = {row["repo_id"] for row in manifest.get("snapshots", [])}
    errors.extend(f"missing_repository:{name}" for name in sorted(REPOSITORIES - names))
    errors.extend(f"missing_snapshot:{repo}" for repo in sorted(SNAPSHOTS - repos))
    for row in manifest.get("repositories", []) + manifest.get("snapshots", []):
        directory = (root / row["path"]).resolve()
        if not directory.is_relative_to(root):
            errors.append("asset_directory_outside_project")
            continue
        if not row.get("files"):
            errors.append(f"empty_file_manifest:{row['path']}")
        for name, checksum in row.get("files", {}).items():
            path = directory / name
            if not path.resolve().is_relative_to(directory) or not path.is_file():
                errors.append(f"missing_or_unsafe_file:{row['path']}/{name}")
            elif file_sha256(path) != checksum:
                errors.append(f"checksum_mismatch:{row['path']}/{name}")
            else:
                checked += 1
        if "commit" in row:
            result = subprocess.run(
                ["git", "-C", str(directory), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode or result.stdout.strip() != row["commit"]:
                errors.append(f"repository_revision_mismatch:{row['path']}")
        elif not row.get("revision"):
            errors.append(f"missing_snapshot_revision:{row['path']}")
    sample: dict[str, object] | None = None
    if not errors:
        try:
            sample = inspect_physbench(root / "data/external/PhysBench")
        except (ValueError, KeyError, OSError, BadZipFile) as error:
            errors.append(f"physbench_sample_failed:{type(error).__name__}")
    pending = ["DeepPHY Pooltool runtime", "DeepPHY I-PHYRE runtime", "DeepPHY PHYRE runtime"]
    iphyre = root / "data/external/verification-iphyre-v1.json"
    runtime_receipts = {}
    if iphyre.is_file():
        receipt = json.loads(iphyre.read_text())
        if (
            receipt["status"] == "passed"
            and receipt["runner_sha256"] == file_sha256(root / "scripts/iphyre_runner.py")
            and receipt["patch_sha256"]
            == file_sha256(root / "data/external/DeepPHY/res/02_iphyre/simulator.py")
        ):
            pending.remove("DeepPHY I-PHYRE runtime")
            runtime_receipts["DeepPHY I-PHYRE runtime"] = file_sha256(iphyre)
    phyre = root / "data/external/verification-phyre-v1.json"
    if phyre.is_file():
        receipt = json.loads(phyre.read_text())
        if (
            receipt["status"] == "passed"
            and receipt["runner_sha256"] == file_sha256(root / "scripts/phyre_runner.py")
            and receipt["config_sha256"]
            == file_sha256(root / "data/external/DeepPHY/conf/env_config_01_phyre.json")
        ):
            pending.remove("DeepPHY PHYRE runtime")
            runtime_receipts["DeepPHY PHYRE runtime"] = file_sha256(phyre)
    pooltool = root / "data/external/verification-pooltool-v1.json"
    if pooltool.is_file():
        receipt = json.loads(pooltool.read_text())
        if (
            receipt["status"] == "passed"
            and receipt["runner_sha256"] == file_sha256(root / "scripts/pooltool_runner.py")
            and receipt["config_sha256"]
            == file_sha256(root / "data/external/DeepPHY/conf/env_config_04_pooltool.json")
        ):
            pending.remove("DeepPHY Pooltool runtime")
            runtime_receipts["DeepPHY Pooltool runtime"] = file_sha256(pooltool)
    return {
        "status": "ready" if not pending and not errors else "incomplete",
        "asset_status": "passed" if not errors else "failed",
        "manifest_sha256": file_sha256(manifest_path),
        "verified_files": checked,
        "errors": errors,
        "physbench": sample,
        "runtime_smokes_pending": pending,
        "runtime_receipts_sha256": runtime_receipts,
        "model_calls": 0,
    }


def inspect_physbench(path: Path) -> dict[str, object]:
    """Load official dynamics labels and verify referenced media, sampling one PNG decode."""
    rows = json.loads((path / "dataset/test.json").read_text())
    labels = json.loads((path / "eval/physbench/test_answer.json").read_text())
    by_id = {row["idx"]: row for row in rows}
    if len(by_id) != len(rows) or len({row["idx"] for row in labels}) != len(labels):
        raise ValueError("PhysBench task IDs must be unique")
    if set(by_id) != {row["idx"] for row in labels}:
        raise ValueError("PhysBench labels must match the pinned task inventory")
    dynamics = [row for row in labels if row["task_type"] == "dynamics"]
    if not dynamics or any(not isinstance(row["answer"], str) for row in labels):
        raise ValueError("PhysBench dynamics or labels are invalid")
    with (
        ZipFile(path / "dataset/image.zip") as images,
        ZipFile(path / "dataset/video.zip") as videos,
    ):
        inspect_members(images.infolist())
        inspect_members(videos.infolist())
        image_names = set(images.namelist())
        media = image_names | set(videos.namelist())
        missing = {name for row in rows for name in row["file_name"] if name not in media}
        if missing:
            raise ValueError(f"PhysBench has {len(missing)} missing media references")
        image_name = next(
            name
            for label in sorted(dynamics, key=lambda row: row["idx"])
            for name in by_id[label["idx"]]["file_name"]
            if name in image_names and name.lower().endswith((".png", ".jpg"))
        )
        with Image.open(io.BytesIO(images.read(image_name))) as image:
            image.load()
            size = image.size
    return {
        "tasks": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "dynamics_tasks": len(dynamics),
        "dynamics_subtypes": dict(Counter(row["sub_type"] for row in dynamics)),
        "raw_answer_counts": dict(Counter(row["answer"] for row in labels)),
        "empty_answer_ids": [row["idx"] for row in labels if not row["answer"].strip()],
        "label_policy": (
            "preserve_raw_labels_pending_official_scorer_review_no_silent_normalization"
        ),
        "dynamics_ids_sha256": fingerprint(sorted(row["idx"] for row in dynamics)),
        "decoded_sample": {"file": image_name, "size": list(size)},
        "source_labels": "official_pinned_repository_not_model_generated",
    }
