import json
import stat
from zipfile import BadZipFile, ZipFile, ZipInfo

import pytest

import inverser.archive as archive_module


@pytest.mark.parametrize(
    "name", ["../escape", "/escape", "a\\b", "C:/escape", "a/./b", "_import/x"]
)
def test_archive_paths_cannot_escape_or_replace_metadata(name):
    with pytest.raises(ValueError, match="Unsafe"):
        archive_module.inspect_members([ZipInfo(name)])


def test_symlink_and_file_directory_conflict_rejected():
    link = ZipInfo("link")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError, match="Non-regular"):
        archive_module.inspect_members([link])
    parent, child = ZipInfo("folder"), ZipInfo("folder/file")
    parent.CRC = child.CRC = 0
    with pytest.raises(ValueError, match="conflict"):
        archive_module.inspect_members([parent, child])


def test_archive_extract_and_resume_preserves_original(tmp_path, monkeypatch):
    monkeypatch.setattr(archive_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(archive_module, "check_storage", lambda reserve: None)
    upload = tmp_path / "source.zip"
    with ZipFile(upload, "w") as zipped:
        zipped.writestr("snooker/shot_0/notes.txt", b"source fixture")
        zipped.writestr("Snooker_Manual/shot_0/manual_track.png", b"image placeholder")
    original = upload.read_bytes()
    output = tmp_path / "data" / "billiards"
    first = archive_module.extract_archive(upload, output)
    second = archive_module.extract_archive(upload, output)
    assert first["files"] == second["files"]
    assert first["extracted_files"] == 2
    assert upload.read_bytes() == original
    assert first["accepted_shot_tasks"] == 0
    (output / "files" / "snooker/shot_0/notes.txt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="existing"):
        archive_module.extract_archive(upload, output)


def test_incomplete_upload_is_not_extracted(tmp_path, monkeypatch):
    monkeypatch.setattr(archive_module, "PROJECT_ROOT", tmp_path)
    upload = tmp_path / "incomplete.zip"
    upload.write_bytes(b"PK uploading, without central directory")
    output = tmp_path / "data" / "billiards"
    with pytest.raises(TimeoutError):
        archive_module.wait_for_upload(
            upload, output, stable_seconds=0.01, poll_seconds=0.01, max_wait_seconds=0.04
        )
    assert not (output / "files").exists()
    assert upload.exists()


def test_archive_symlink_and_output_escape_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(archive_module, "PROJECT_ROOT", tmp_path)
    target = tmp_path / "real.zip"
    target.write_bytes(b"PK")
    link = tmp_path / "link.zip"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        archive_module.file_signature(link)
    with pytest.raises(ValueError, match="under"):
        archive_module.extract_archive(target, tmp_path / "outside")


def test_complete_upload_is_imported_after_stability(tmp_path, monkeypatch):
    monkeypatch.setattr(archive_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(archive_module, "check_storage", lambda reserve: None)
    upload = tmp_path / "ready.zip"
    with ZipFile(upload, "w") as zipped:
        zipped.writestr("shot/source.txt", b"complete fixture")
    output = tmp_path / "data" / "billiards"
    result = archive_module.wait_for_upload(
        upload, output, stable_seconds=0.02, poll_seconds=0.01, max_wait_seconds=3
    )
    state = json.loads((output / "_import" / "upload_state.json").read_text())
    assert result["status"] == "complete" and state["status"] == "extracted"
    assert upload.is_file()


def test_bad_crc_is_not_published_as_an_extracted_file(tmp_path, monkeypatch):
    monkeypatch.setattr(archive_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(archive_module, "check_storage", lambda reserve: None)
    upload = tmp_path / "corrupt.zip"
    with ZipFile(upload, "w") as zipped:
        zipped.writestr("shot.txt", b"fixture content")
    damaged = bytearray(upload.read_bytes())
    damaged[damaged.index(b"fixture content")] ^= 1
    upload.write_bytes(damaged)
    output = tmp_path / "data" / "billiards"
    with pytest.raises(BadZipFile, match="CRC"):
        archive_module.extract_archive(upload, output)
    assert not (output / "files" / "shot.txt").exists()
    assert upload.exists()
    assert (
        json.loads((output / "_import" / "extraction_state.json").read_text())["status"] == "failed"
    )
