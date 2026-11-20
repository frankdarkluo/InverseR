import io
import json
from zipfile import ZipFile

import pytest
from PIL import Image

from inverser.external_benchmarks import inspect_physbench, verify_external_suites
from inverser.runtime import file_sha256


def test_missing_or_changed_assets_never_pass(tmp_path):
    assert verify_external_suites(tmp_path)["status"] == "incomplete"
    external = tmp_path / "data/external"
    external.mkdir(parents=True)
    artifact = external / "fixture.txt"
    artifact.write_text("original")
    (external / "manifest.json").write_text(
        json.dumps(
            {
                "repositories": [],
                "snapshots": [
                    {
                        "repo_id": "fixture",
                        "path": "data/external",
                        "revision": "fixture-revision",
                        "files": {"fixture.txt": file_sha256(artifact)},
                    }
                ],
            }
        )
    )
    report = verify_external_suites(tmp_path)
    assert report["verified_files"] == 1 and report["asset_status"] == "failed"
    artifact.write_text("changed")
    report = verify_external_suites(tmp_path)
    assert "checksum_mismatch:data/external/fixture.txt" in report["errors"]


def test_physbench_joins_official_labels_and_decodes_without_extracting(tmp_path):
    (tmp_path / "dataset").mkdir()
    (tmp_path / "eval/physbench").mkdir(parents=True)
    tasks = [{"idx": 0, "split": "val", "file_name": ["fixture.png"]}]
    labels = [{"idx": 0, "answer": "A", "task_type": "dynamics", "sub_type": "collision"}]
    (tmp_path / "dataset/test.json").write_text(json.dumps(tasks))
    (tmp_path / "eval/physbench/test_answer.json").write_text(json.dumps(labels))
    stream = io.BytesIO()
    Image.new("RGB", (10, 20)).save(stream, format="PNG")
    with ZipFile(tmp_path / "dataset/image.zip", "w") as archive:
        archive.writestr("fixture.png", stream.getvalue())
    with ZipFile(tmp_path / "dataset/video.zip", "w") as archive:
        archive.writestr("unused.mp4", b"fixture_not_decoded_by_image_sample_check")
    report = inspect_physbench(tmp_path)
    assert report["dynamics_tasks"] == 1 and report["split_counts"] == {"val": 1}
    assert report["decoded_sample"]["size"] == [10, 20]
    assert not (tmp_path / "dataset/fixture.png").exists()
    tasks[0]["file_name"] = ["missing.png"]
    (tmp_path / "dataset/test.json").write_text(json.dumps(tasks))
    with pytest.raises(ValueError, match="missing media"):
        inspect_physbench(tmp_path)
