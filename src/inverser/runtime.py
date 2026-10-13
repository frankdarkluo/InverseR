import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_LIMIT_BYTES = 1_000_000_000_000
MIN_FREE_BYTES = 200_000_000_000


def configure_physics_runtime() -> None:
    """Isolate Pooltool's hard-coded HOME configuration inside this CLI process."""
    home = PROJECT_ROOT / ".cache" / "pooltool-home"
    if "pooltool" in sys.modules and Path.home() != home:
        raise RuntimeError("Pooltool was imported before its isolated runtime was configured")
    home.mkdir(parents=True, exist_ok=True)
    numba_cache = PROJECT_ROOT / ".cache" / "numba"
    numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(home)
    os.environ["NUMBA_CACHE_DIR"] = str(numba_cache)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def check_storage(reserve_bytes: int = 0) -> None:
    if reserve_bytes < 0:
        raise ValueError("Storage reservation must be nonnegative")
    size = 0
    for root, _, files in os.walk(PROJECT_ROOT, followlinks=False):
        for name in files:
            path = Path(root) / name
            if not path.is_symlink():
                size += path.stat().st_size
    if size + reserve_bytes > STORAGE_LIMIT_BYTES:
        raise RuntimeError("Project storage cap would be exceeded")
    if shutil.disk_usage(PROJECT_ROOT).free - reserve_bytes < MIN_FREE_BYTES:
        raise RuntimeError("Shared data-volume free-space guard would be violated")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
