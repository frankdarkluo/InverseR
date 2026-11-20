import fcntl
import hashlib
import logging
import os
import stat
import tempfile
import time
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TypedDict
from zipfile import BadZipFile, ZipFile, ZipInfo, is_zipfile

from inverser.runtime import PROJECT_ROOT, check_storage, file_sha256, write_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArchiveMember:
    path: str
    bytes: int
    compressed_bytes: int
    crc32: int
    directory: bool


class MemberReceipt(TypedDict):
    path: str
    bytes: int
    crc32: int
    sha256: str


def file_signature(path: Path) -> tuple[int, int, int, int]:
    if path.is_symlink():
        raise ValueError("Uploaded archive must not be a symlink")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Uploaded archive must be a regular file")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def inspect_members(infos: list[ZipInfo]) -> tuple[ArchiveMember, ...]:
    if not infos or len(infos) > 100_000:
        raise ValueError("Archive must contain 1-100000 entries")
    members = []
    paths: dict[str, bool] = {}
    for info in infos:
        name = info.filename.rstrip("/") if info.is_dir() else info.filename
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or "\x00" in name
            or str(path) != name
            or ":" in path.parts[0]
            or path.parts[0] in ("_import", ".extracting")
        ):
            raise ValueError(f"Unsafe archive member: {info.filename!r}")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode) or (
            mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)
        ):
            raise ValueError(f"Non-regular archive entry: {info.filename!r}")
        if info.flag_bits & 1:
            raise ValueError(f"Encrypted archive entry is not supported: {info.filename!r}")
        if name.casefold() in paths:
            raise ValueError(f"Duplicate or case-ambiguous archive path: {info.filename!r}")
        paths[name.casefold()] = info.is_dir()
        members.append(
            ArchiveMember(name, info.file_size, info.compress_size, info.CRC, info.is_dir())
        )
    for member in members:
        for parent in PurePosixPath(member.path).parents:
            if str(parent) != "." and paths.get(str(parent).casefold()) is False:
                raise ValueError(f"File/directory conflict in archive: {member.path}")
    return tuple(members)


def _check_unchanged(archive: Path, signature: tuple[int, int, int, int]) -> None:
    if file_signature(archive) != signature:
        raise RuntimeError("Archive changed while importing; stop and wait for upload completion")


def _verify_existing(path: Path, member: ArchiveMember) -> MemberReceipt:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != member.bytes:
        raise ValueError(f"Refusing to overwrite an incompatible existing file: {member.path}")
    digest = hashlib.sha256()
    crc = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
    if crc != member.crc32:
        raise ValueError(f"Existing member CRC does not match archive: {member.path}")
    return {"path": member.path, "bytes": member.bytes, "crc32": crc, "sha256": digest.hexdigest()}


def extract_archive(archive: Path, output: Path) -> dict[str, object]:
    archive = archive.absolute()
    output = output.resolve()
    if not output.is_relative_to((PROJECT_ROOT / "data").resolve()):
        raise ValueError("Extraction must stay under InverseR/data")
    if archive.resolve().is_relative_to(output):
        raise ValueError("Preserved upload must not be inside the extraction directory")
    signature = file_signature(archive)
    metadata = output / "_import"
    files = output / "files"
    if metadata.is_symlink() or files.is_symlink():
        raise ValueError("Extraction directories cannot be symlinks")
    metadata.mkdir(parents=True, exist_ok=True)
    files.mkdir(exist_ok=True)
    receipts: list[MemberReceipt] = []
    with (metadata / ".extract.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with ZipFile(archive) as zipped:
                members = inspect_members(zipped.infolist())
                total = sum(member.bytes for member in members if not member.directory)
                check_storage(total)
                write_json(
                    metadata / "source_manifest.json",
                    {
                        "archive": str(archive),
                        "archive_signature": signature,
                        "uncompressed_bytes": total,
                        "members": [asdict(member) for member in members],
                        "source": (
                            "researcher-provided upload; contents not relabelled as ground truth"
                        ),
                    },
                )
                for member, info in zip(members, zipped.infolist(), strict=True):
                    _check_unchanged(archive, signature)
                    target = files / member.path
                    if not target.resolve().is_relative_to(files):
                        raise ValueError(f"Extraction path escapes source directory: {member.path}")
                    if member.directory:
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if target.exists():
                        receipt = _verify_existing(target, member)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary: Path | None = None
                        try:
                            with (
                                zipped.open(info) as source,
                                tempfile.NamedTemporaryFile(
                                    "wb",
                                    dir=target.parent,
                                    prefix=f".{target.name}.",
                                    delete=False,
                                ) as destination,
                            ):
                                temporary = Path(destination.name)
                                digest = hashlib.sha256()
                                crc = 0
                                received = 0
                                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                                    received += len(chunk)
                                    if received > member.bytes:
                                        raise ValueError(
                                            "Member exceeds its declared uncompressed size"
                                        )
                                    destination.write(chunk)
                                    digest.update(chunk)
                                    crc = zlib.crc32(chunk, crc)
                                destination.flush()
                                os.fsync(destination.fileno())
                            if received != member.bytes or crc != member.crc32:
                                raise ValueError(f"Archive member integrity failure: {member.path}")
                            _check_unchanged(archive, signature)
                            temporary.replace(target)
                            receipt = {
                                "path": member.path,
                                "bytes": received,
                                "crc32": crc,
                                "sha256": digest.hexdigest(),
                            }
                        finally:
                            if temporary is not None and temporary.exists():
                                temporary.unlink()
                    receipts.append(receipt)
                    write_json(
                        metadata / "extraction_state.json",
                        {
                            "status": "extracting",
                            "verified_files": len(receipts),
                            "verified_bytes": sum(row["bytes"] for row in receipts),
                            "archive_preserved": True,
                        },
                    )
                    if len(receipts) % 25 == 0:
                        LOGGER.info("Extracted and CRC-verified %s files", len(receipts))
            _check_unchanged(archive, signature)
            archive_hash = file_sha256(archive)
            _check_unchanged(archive, signature)
            counts = Counter(Path(row["path"]).suffix.lower() for row in receipts)
            report = {
                "status": "complete",
                "archive": str(archive),
                "archive_bytes": signature[2],
                "archive_sha256": archive_hash,
                "archive_preserved": True,
                "extracted_files": len(receipts),
                "extracted_bytes": sum(row["bytes"] for row in receipts),
                "suffix_counts": dict(counts),
                "manual_path_images": [
                    row["path"]
                    for row in receipts
                    if Path(row["path"]).name.lower() == "manual_track.png"
                ],
                "files": receipts,
                "completed_utc": datetime.now(UTC).isoformat(),
                "accepted_shot_tasks": 0,
                "label_warning": (
                    "Source manual-path images are not automatically timed trajectories."
                ),
            }
            write_json(metadata / "inventory.json", report)
            write_json(
                metadata / "extraction_state.json",
                {k: v for k, v in report.items() if k != "files"},
            )
            return report
        except (BadZipFile, OSError, ValueError, RuntimeError):
            write_json(
                metadata / "extraction_state.json",
                {"status": "failed", "verified_files": len(receipts), "archive_preserved": True},
            )
            raise


def wait_for_upload(
    archive: Path,
    output: Path,
    *,
    stable_seconds: float = 120,
    poll_seconds: float = 15,
    max_wait_seconds: float = 21600,
) -> dict[str, object]:
    if min(stable_seconds, poll_seconds, max_wait_seconds) <= 0:
        raise ValueError("Upload-monitor intervals must be positive")
    output = output.resolve()
    if not output.is_relative_to((PROJECT_ROOT / "data").resolve()):
        raise ValueError("Upload-monitor output must stay under InverseR/data")
    if (output / "_import").is_symlink():
        raise ValueError("Upload-monitor metadata cannot be a symlink")
    metadata = output / "_import"
    metadata.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    unchanged_since = started
    previous: tuple[int, int, int, int] | None = None
    with (metadata / ".monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.monotonic() - started < max_wait_seconds:
            now = time.monotonic()
            signature = file_signature(archive) if archive.exists() else None
            if signature != previous:
                previous = signature
                unchanged_since = now
            valid_zip = signature is not None and is_zipfile(archive)
            stable_for = now - unchanged_since
            state = {
                "status": "waiting_for_complete_upload",
                "archive": str(archive),
                "bytes_observed": signature[2] if signature else 0,
                "zip_directory_complete": valid_zip,
                "unchanged_seconds": round(stable_for, 2),
                "required_stable_seconds": stable_seconds,
                "elapsed_seconds": round(now - started, 2),
                "updated_utc": datetime.now(UTC).isoformat(),
            }
            write_json(metadata / "upload_state.json", state)
            if valid_zip and stable_for >= stable_seconds:
                LOGGER.info("Upload stable and ZIP complete; beginning CRC-verified extraction")
                result = extract_archive(archive, output)
                write_json(
                    metadata / "upload_state.json",
                    state | {"status": "extracted", "archive_sha256": result["archive_sha256"]},
                )
                return result
            time.sleep(poll_seconds)
        write_json(
            metadata / "upload_state.json",
            {"status": "timed_out_waiting_for_upload", "archive_preserved": True},
        )
        raise TimeoutError("Upload did not become stable and ZIP-complete within the wait budget")
