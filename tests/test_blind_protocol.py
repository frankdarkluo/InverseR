import json

import pytest

from inverser.blind_protocol import freeze_blind_protocol, verify_blind_protocol
from inverser.capo_selection import select_checkpoint
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.study_calls import local_vllm_backends, save_job


def test_checkpoint_selection_rejects_partial_or_wrong_scene_inventory(tmp_path, monkeypatch):
    from inverser import study_calls

    monkeypatch.setattr(study_calls, "PROJECT_ROOT", tmp_path)
    root = tmp_path / "data"
    valid = root / "valid/manifest.json"
    scenes = valid.parent / "public/scenes.jsonl"
    scenes.parent.mkdir(parents=True)
    scenes.write_text('{"scene_id":"x"}\n')
    write_json(
        valid,
        {
            "split": "valid",
            "standard_scene_count": 1,
            "artifact_sha256": {"public/scenes.jsonl": file_sha256(scenes)},
        },
    )
    candidates = {}
    for name in ("partial", "lower", "better"):
        checkpoint = root / name / "checkpoint.json"
        write_json(checkpoint, {"files": {"weights.safetensors": name}})
        run = root / f"evaluation-{name}"
        candidates[checkpoint] = run
        spec = {
            "data": file_sha256(valid),
            "count": 1,
            "constrain_json": True,
            "seed": 11,
            "tasks": ["forward", "T4", "T5"],
            "backend": {"model_revision": "checkpoint-manifest-sha256:" + file_sha256(checkpoint)},
        }
        run_hash = fingerprint(spec)
        write_json(run / "manifest.json", spec | {"run_hash": run_hash})
        tasks = spec["tasks"][:2] if name == "partial" else spec["tasks"]
        write_json(
            run / "summary.json",
            {
                "status": "complete",
                "run_hash": run_hash,
                "request_status": {"complete": len(tasks)},
                "jobs": len(tasks),
            },
        )
        for task in tasks:
            save_job(
                run / f"jobs/{task}.json",
                run_hash,
                {
                    "task": task,
                    "scene_id": "x",
                    "scores": {"S_total": 0.5},
                    "success": name == "better",
                },
            )
    destination = root / "selected.json"
    with pytest.raises(ValueError, match="inventory"):
        select_checkpoint(destination, valid, candidates)
    candidates.pop(root / "partial/checkpoint.json")
    result = select_checkpoint(destination, valid, candidates)
    assert result["checkpoint_manifest_sha256"] == file_sha256(root / "better/checkpoint.json")


def test_blind_freeze_checks_selection_backend_dataset_and_weights(tmp_path, monkeypatch):
    from inverser import study_calls

    monkeypatch.setattr(study_calls, "PROJECT_ROOT", tmp_path)
    root = tmp_path / "data"
    test, valid = (root / f"{name}.json" for name in ("test", "valid"))
    write_json(test, {"split": "test"})
    write_json(valid, {"split": "valid"})
    weights = root / "arm" / "model.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"fixture weights")
    checkpoint = weights.parent / "checkpoint.json"
    write_json(checkpoint, {"files": {weights.name: file_sha256(weights)}})
    valid_eval = root / "selected.json"
    write_json(
        valid_eval,
        {
            "status": "complete",
            "valid_manifest_sha256": file_sha256(valid),
            "checkpoint_manifest_sha256": file_sha256(checkpoint),
        },
    )
    backend = local_vllm_backends(
        "http://127.0.0.1:8000/v1",
        ("fixture",),
        model_revision="checkpoint-manifest-sha256:" + file_sha256(checkpoint),
        reasoning_effort="none",
        max_output_tokens=128,
        request_token_reservation=4096,
        timeout_s=30,
    )[0]
    arms = root / "arms.json"
    write_json(
        arms,
        [
            {
                "arm": "CAPO",
                "seed": 11,
                "backend_sha256": fingerprint(backend.model_dump(mode="json")),
                "checkpoint_manifest": str(checkpoint),
                "checkpoint_manifest_sha256": file_sha256(checkpoint),
                "selection_summary": str(valid_eval),
                "selection_summary_sha256": file_sha256(valid_eval),
            }
        ],
    )
    output = root / "blind.json"
    receipt = freeze_blind_protocol(output, test, valid, arms)
    assert verify_blind_protocol(output, test, backend, "CAPO", 11) == receipt.protocol_sha256
    with pytest.raises(ValueError, match="immutable"):
        freeze_blind_protocol(output, test, valid, arms)
    with pytest.raises(ValueError, match="not frozen"):
        verify_blind_protocol(output, test, backend, "CAPO", 12)
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset changed"):
        verify_blind_protocol(output, test, backend, "CAPO", 11)
    raw = json.loads(output.read_text())
    raw["arms"][0]["seed"] = 12
    write_json(output, raw)
    with pytest.raises(ValueError, match="protocol changed"):
        verify_blind_protocol(output, test, backend, "CAPO", 12)
