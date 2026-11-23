import json

import pytest

from inverser.benchmark.vlm_suite import parse_forward_answer, parse_goal_answer
from inverser.capo import sft_corpus
from inverser.capo.capo_trainer import _training_dataset
from inverser.runtime import file_sha256, fingerprint


def test_v2_build_refills_duplicate_pair_and_loads_native_sft_rows(tmp_path, monkeypatch):
    duplicate, _, _ = _first_pair(tmp_path)
    names = duplicate["images"]
    monkeypatch.setattr(sft_corpus, "require_data_path", lambda _: None)
    monkeypatch.setattr(sft_corpus, "check_storage", lambda _: None)
    monkeypatch.setattr(
        sft_corpus,
        "_exclusions",
        lambda *_: ({duplicate["layout_sha256"]}, set(), set(), {}),
    )
    output = tmp_path / "train"
    manifest = sft_corpus.build_corpus(
        output, tmp_path, tmp_path, tmp_path, pairs=4, seed=20261120, workers=1
    )
    assert manifest["sft_examples"] == {"forward": 8, "inverse": 8, "total": 16}
    assert manifest["rejections"]["duplicate_input_or_layout"] >= 1
    meta = [json.loads(line) for line in (output / "groups.jsonl").read_text().splitlines()]
    assert next(m for m in meta if m["slot"] == 0)["generation"] > 0
    assert len({m["layout_sha256"] for m in meta}) == 4
    assert [len(m["images"]) for m in meta] == [len(names)] * 4
    assert all(m["endpoint_shift_radii"] >= 2 for m in meta)
    rows = [json.loads(line) for line in (output / "sft.jsonl").read_text().splitlines()]
    inputs = []
    for i, row in enumerate(rows):
        image_paths = row["image_paths"]
        reply = row["completion"][0]["content"]
        if i % 2:
            assert len(image_paths) == 2 and image_paths[-1].endswith("-goal.png")
            parse_goal_answer(reply, allow_json_fallback=False)
            assert '"action":' not in row["prompt"][1]["content"]
        else:
            assert len(image_paths) == 1 and image_paths[0].endswith("-fwd.png")
            parse_forward_answer(reply, allow_json_fallback=False)
        inputs.append(fingerprint([row["prompt"], [file_sha256(output / p) for p in image_paths]]))
    assert len(set(inputs)) == len(rows)

    pytest.importorskip("datasets")
    assert len(_training_dataset(output, None)) == 16
    assert len(_training_dataset(output, 1)) == 2
    with (output / "sft.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        _training_dataset(output, None)


def _first_pair(tmp_path):
    sft_corpus._init_worker(tmp_path / "fixture", 20261120)
    return sft_corpus._pair(0, 0)
