import csv
import fcntl
import hashlib
import json
import logging
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Self
from urllib.parse import urlparse

from pydantic import Field, model_validator

from inverser.runtime import PROJECT_ROOT, check_storage, file_sha256, write_json
from inverser.schemas import Record
from inverser.sources import Source, load_sources, probe_video

LOGGER = logging.getLogger(__name__)
SNAPSHOT_REVISION = "dcbcdf46b71c30e76bf51220f697836abd9c60bf"


class RemoteFile(Record):
    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    lfs_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    git_blob_id: str = Field(pattern=r"^[0-9a-f]{40}$")

    @model_validator(mode="after")
    def check_path(self) -> Self:
        path = PurePosixPath(self.path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in self.path
            or str(path) != self.path
            or path.parts[0] == ".cache"
        ):
            raise ValueError("Unsafe repository file path")
        return self


def require_local_download(source: Source) -> str:
    parsed = urlparse(source.url)
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.scheme != "https"
        or parsed.hostname != "huggingface.co"
        or parsed.query
        or len(parts) != 3
        or parts[0] != "datasets"
    ):
        raise ValueError("Expected an explicit Hugging Face dataset repository URL")
    if not source.local_download_requested or source.technical_access != "reachable":
        raise PermissionError(
            "Full local acquisition has not been requested and verified reachable"
        )
    if source.acquire == "denied" or source.process == "denied":
        raise PermissionError("Explicit source restrictions prohibit this acquisition")
    return "/".join(parts[1:])


def verify_file(path: Path, expected: RemoteFile) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != expected.size_bytes:
        raise ValueError(f"Missing, unsafe or wrong-sized file: {expected.path}")
    sha256 = file_sha256(path)
    if expected.lfs_sha256:
        if sha256 != expected.lfs_sha256:
            raise ValueError(f"LFS checksum mismatch: {expected.path}")
    else:
        digest = hashlib.sha1(f"blob {expected.size_bytes}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected.git_blob_id:
            raise ValueError(f"Git blob checksum mismatch: {expected.path}")
    return sha256


def _file_inventory(path: Path, expected: RemoteFile, sha256: str) -> dict[str, object]:
    item: dict[str, object] = {
        "path": expected.path,
        "bytes": expected.size_bytes,
        "sha256": sha256,
        "checksum_verified": True,
    }
    if path.suffix.lower() == ".mp4":
        item["kind"] = "video"
        try:
            item["video"] = probe_video(path).model_dump(mode="json")
            item["probe_status"] = "passed"
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            ValueError,
            KeyError,
        ) as exc:
            LOGGER.error("Video probe failed for %s: %s", expected.path, type(exc).__name__)
            item["probe_status"] = "failed"
            item["probe_error"] = type(exc).__name__
    elif path.suffix.lower() == ".json":
        item["kind"] = "json_sidecar"
        with path.open() as stream:
            content = json.load(stream)
        item["json_root_type"] = type(content).__name__
        item["json_entries"] = len(content) if isinstance(content, (list, dict)) else None
        item["annotation_status"] = "source sidecar; not verified ball trajectories"
    else:
        item["kind"] = "repository_metadata"
    return item


def _write_inventory(output: Path, records: dict[str, dict[str, object]]) -> None:
    rows = [records[name] for name in sorted(records)]
    write_json(output / "inventory.json", {"files": rows, "accepted_shot_tasks": 0})
    fields = [
        "path",
        "kind",
        "bytes",
        "sha256",
        "probe_status",
        "width",
        "height",
        "duration_s",
        "fps",
        "codec",
    ]
    temporary = output / ".inventory.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {field: row.get(field, "") for field in fields}
            video = row.get("video")
            if isinstance(video, dict):
                values.update({field: video.get(field, "") for field in fields[5:]})
            writer.writerow(values)
    temporary.replace(output / "inventory.csv")


def download_hf_source(
    registry: Path,
    output: Path,
    *,
    source_id: str = "hf_spdev39_snooker",
    revision: str = SNAPSHOT_REVISION,
    max_workers: int = 4,
) -> dict[str, object]:
    if not 1 <= max_workers <= 4:
        raise ValueError("At most four concurrent file downloads are approved")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("Download must be pinned to an immutable commit SHA")
    source = next((item for item in load_sources(registry) if item.source_id == source_id), None)
    if source is None:
        raise ValueError(f"Unknown source: {source_id}")
    repo_id = require_local_download(source)
    output = output.resolve()
    if not output.is_relative_to((PROJECT_ROOT / "data").resolve()):
        raise ValueError("Raw downloads must remain inside the project's data directory")

    os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    import httpx
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    info = HfApi(token=False).dataset_info(
        repo_id, revision=revision, files_metadata=True, timeout=30, token=False
    )
    if info.sha != revision or info.private or info.gated:
        raise PermissionError("Expected the pinned public, ungated repository")
    if not info.siblings:
        raise ValueError("Repository has no file manifest")
    remote_files = []
    for entry in info.siblings:
        if entry.size is None or entry.blob_id is None:
            raise ValueError(f"Missing file size/checksum metadata: {entry.rfilename}")
        remote_files.append(
            RemoteFile(
                path=entry.rfilename,
                size_bytes=entry.size,
                lfs_sha256=entry.lfs.sha256 if entry.lfs else None,
                git_blob_id=entry.blob_id,
            )
        )
    files = tuple(remote_files)
    if len({item.path for item in files}) != len(files):
        raise ValueError("Duplicate paths in the repository manifest")
    total_bytes = sum(item.size_bytes for item in files)
    size_by_path = {item.path: item.size_bytes for item in files}
    check_storage(total_bytes)
    manifest = {
        "source_id": source_id,
        "repo_id": repo_id,
        "revision": revision,
        "files": [item.model_dump(mode="json") for item in files],
        "expected_file_count": len(files),
        "expected_bytes": total_bytes,
        "media_license": source.license,
        "media_rights": "unverified" if source.acquire == "unverified" else source.acquire,
        "local_execution_approval": "researcher explicitly requested full raw acquisition",
        "training_approved": False,
        "redistribution_approved": False,
        "authentication_used": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    file_root = output / "files"
    if file_root.is_symlink():
        raise ValueError("Raw-file directory cannot be a symlink")
    file_root.mkdir(exist_ok=True)
    manifest_path = output / "source_manifest.json"
    records: dict[str, dict[str, object]] = {}
    active: set[str] = set()
    failures: dict[str, str] = {}
    guard = threading.Lock()
    started = datetime.now(UTC).isoformat()

    def progress(status: str) -> dict[str, object]:
        state = {
            "status": status,
            "repo_id": repo_id,
            "revision": revision,
            "started_utc": started,
            "updated_utc": datetime.now(UTC).isoformat(),
            "expected_files": len(files),
            "expected_bytes": total_bytes,
            "verified_files": len(records),
            "verified_bytes": sum(size_by_path[name] for name in records),
            "active_files": sorted(active),
            "failed_files": failures,
            "max_download_workers": max_workers,
            "authentication_used": False,
            "accepted_shot_tasks": 0,
            "g1_passed": False,
        }
        write_json(output / "acquisition_state.json", state)
        return state

    def transfer(expected: RemoteFile) -> dict[str, object]:
        with guard:
            active.add(expected.path)
            progress("downloading")
        try:
            target = (file_root / expected.path).resolve()
            if not target.is_relative_to(file_root):
                raise ValueError("Raw-file path escapes the acquisition directory")
            try:
                downloaded = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=expected.path,
                        revision=revision,
                        local_dir=file_root,
                        token=False,
                        etag_timeout=30,
                    )
                )
            except (HfHubHTTPError, httpx.RequestError) as exc:
                LOGGER.error("Download failed for %s: %s", expected.path, type(exc).__name__)
                raise RuntimeError(
                    f"Public download failed for {expected.path}; resume this pinned directory"
                ) from None
            if downloaded.resolve() != (file_root / expected.path).resolve():
                raise ValueError("Download resolved to an unexpected destination")
            sha256 = verify_file(downloaded, expected)
            row = _file_inventory(downloaded, expected, sha256)
            with guard:
                records[expected.path] = row
                LOGGER.info(
                    "Verified %s (%s/%s files, %.2f/%.2f GB)",
                    expected.path,
                    len(records),
                    len(files),
                    sum(size_by_path[name] for name in records) / 1e9,
                    total_bytes / 1e9,
                )
            return row
        except (OSError, ValueError, RuntimeError) as exc:
            with guard:
                failures[expected.path] = type(exc).__name__
            LOGGER.error(
                "Acquisition/verification failed for %s: %s", expected.path, type(exc).__name__
            )
            raise
        finally:
            with guard:
                active.discard(expected.path)
                progress("downloading")

    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Resume refused: repository or acquisition contract changed")
        elif any(file_root.iterdir()):
            raise ValueError("Refusing to overwrite raw files without a source manifest")
        write_json(manifest_path, manifest)
        progress("downloading")
        executor = ThreadPoolExecutor(max_workers=max_workers)
        completed = False
        try:
            futures = [
                executor.submit(transfer, item)
                for item in sorted(files, key=lambda item: item.size_bytes)
            ]
            for future in as_completed(futures):
                future.result()
            check_storage()
            completed = len(records) == len(files)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            _write_inventory(output, records)
            with guard:
                state = progress("complete" if completed else "incomplete")
    return state
