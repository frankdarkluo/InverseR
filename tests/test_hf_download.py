import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import inverser.hf_download as download
from inverser.sources import load_sources, require_acquisition


@pytest.mark.parametrize("path", ["../clip.mp4", "/clip.mp4", "a\\b.mp4", ".cache/x", "a/./b"])
def test_unsafe_repository_paths_are_rejected(path):
    with pytest.raises(ValidationError):
        download.RemoteFile(path=path, size_bytes=0, git_blob_id="a" * 40)


def test_local_request_does_not_change_media_permission():
    sources = load_sources(Path("configs/sources.json"))
    source = next(item for item in sources if item.source_id == "hf_spdev39_snooker")
    assert download.require_local_download(source) == "spdev39/snookerVideos"
    assert source.acquire == "unverified"
    assert source.access == "blocked"
    with pytest.raises(PermissionError):
        require_acquisition(source, SimpleNamespace())


def test_corrupt_lfs_file_fails_verification(tmp_path):
    path = tmp_path / "sample"
    path.write_bytes(b"abc")
    expected = download.RemoteFile(
        path="sample", size_bytes=3, lfs_sha256="0" * 64, git_blob_id="a" * 40
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        download.verify_file(path, expected)


def test_public_snapshot_roundtrip_resume_and_limits(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(download, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(download, "check_storage", lambda *args: None)
    payloads = {"sample.txt": b"public fixture", "corners.json": b"[[1,2],[3,4],[5,6],[7,8]]"}
    siblings = [
        SimpleNamespace(
            rfilename=name,
            size=len(payload),
            lfs=SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest()),
            blob_id="a" * 40,
        )
        for name, payload in payloads.items()
    ]
    requests = []

    class FakeApi:
        def __init__(self, token):
            assert token is False

        def dataset_info(self, repo, **kwargs):
            assert kwargs["token"] is False
            assert kwargs["revision"] == download.SNAPSHOT_REVISION
            return SimpleNamespace(
                sha=download.SNAPSHOT_REVISION, private=False, gated=False, siblings=siblings
            )

    def fake_download(**kwargs):
        assert kwargs["token"] is False
        requests.append(kwargs["filename"])
        path = kwargs["local_dir"] / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payloads[kwargs["filename"]])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    output = tmp_path / "data" / "source"
    first = download.download_hf_source(Path("configs/sources.json"), output)
    second = download.download_hf_source(Path("configs/sources.json"), output)
    assert first["status"] == second["status"] == "complete"
    assert first["verified_files"] == 2
    assert len(requests) == 4
    manifest = json.loads((output / "source_manifest.json").read_text())
    assert manifest["media_rights"] == "unverified"
    assert manifest["training_approved"] is False
    assert (output / "inventory.csv").is_file()
    with pytest.raises(ValueError, match="four"):
        download.download_hf_source(Path("configs/sources.json"), output, max_workers=8)
    with pytest.raises(ValueError, match="inside"):
        download.download_hf_source(Path("configs/sources.json"), tmp_path / "outside")

    interrupted = tmp_path / "data" / "interrupted"

    def interrupted_download(**kwargs):
        if kwargs["filename"] == "corners.json":
            raise RuntimeError("fixture transfer interruption")
        return fake_download(**kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", interrupted_download)
    with pytest.raises(RuntimeError, match="fixture transfer"):
        download.download_hf_source(Path("configs/sources.json"), interrupted, max_workers=1)
    state = json.loads((interrupted / "acquisition_state.json").read_text())
    assert state["status"] == "incomplete"
    assert state["verified_files"] == 1
    assert state["failed_files"] == {"corners.json": "RuntimeError"}
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    resumed = download.download_hf_source(Path("configs/sources.json"), interrupted, max_workers=1)
    assert resumed["status"] == "complete"
    assert resumed["verified_files"] == 2
