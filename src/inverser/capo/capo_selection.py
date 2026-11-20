"""Select checkpoints from completed, comparable valid runs before a test freeze."""

import json
from pathlib import Path
from statistics import mean
from typing import Any

from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.study_calls import load_job, require_data_path


def select_checkpoint(
    output: Path,
    valid_manifest: Path,
    candidates: dict[Path, Path],
) -> dict[str, Any]:
    """Choose one checkpoint for an arm/seed by the fixed mean of forward, T4 and T5."""
    require_data_path(output)
    if output.exists() or not candidates:
        raise ValueError("Selection requires candidates and a new receipt path")
    valid = json.loads(valid_manifest.read_text())
    if valid["split"] != "valid":
        raise ValueError("Checkpoint selection may read only the valid partition")
    valid_hash = file_sha256(valid_manifest)
    size = int(valid["standard_scene_count"])
    scenes = valid_manifest.parent / "public/scenes.jsonl"
    if file_sha256(scenes) != valid["artifact_sha256"]["public/scenes.jsonl"]:
        raise ValueError("Public valid index does not match the manifest")
    with scenes.open() as stream:
        identifiers = {json.loads(line)["scene_id"] for line in stream}
    if len(identifiers) != size:
        raise ValueError("Valid scene count differs from the frozen index")
    expected_keys = {(task, scene) for task in ("forward", "T4", "T5") for scene in identifiers}
    rank: list[dict[str, Any]] = []
    comparison: dict[str, Any] | None = None
    for checkpoint, run in sorted(candidates.items()):
        require_data_path(checkpoint)
        require_data_path(run)
        checkpoint_hash = file_sha256(checkpoint)
        params = json.loads((run / "manifest.json").read_text())
        run_hash = params.pop("run_hash")
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text())
        if (
            fingerprint(params) != run_hash
            or summary.get("run_hash") != run_hash
            or summary.get("status") != "complete"
            or summary.get("request_status") is None
            or set(summary["request_status"]) != {"complete"}
            or params.get("data") != valid_hash
            or params.get("count") != size
            or set(params.get("tasks", ())) != {"forward", "T4", "T5"}
            or params.get("blind_protocol_sha256")
            or not params.get("constrain_json")
        ):
            raise ValueError("Selection requires complete, full-task, constrained valid results")
        backend = params["backend"]
        if backend.get("model_revision") != "checkpoint-manifest-sha256:" + checkpoint_hash:
            raise ValueError("Valid run does not attest the candidate checkpoint")
        config = {
            "backend": {
                k: v
                for k, v in backend.items()
                if k not in ("alias", "model_id", "model_revision", "timeout_s")
            },
            "producer": params.get("producer"),
            "seed": params.get("seed"),
        }
        if comparison is not None and config != comparison:
            raise ValueError("Candidates do not share one inference protocol and seed")
        comparison = config
        scores: dict[str, list[float]] = {"forward": [], "T4": [], "T5": []}
        keys = set()
        for path in sorted((run / "jobs").glob("*.json")):
            row = load_job(path, run_hash)
            if row is None:
                raise ValueError("Valid job is missing")
            task, scene = row["task"], row["scene_id"]
            if row.get("ood") or (task, scene) in keys or task not in scores:
                raise ValueError("Unexpected or duplicate valid task")
            keys.add((task, scene))
            if task == "forward":
                assert isinstance(row["scores"], dict)
                scores["forward"].append(float(row["scores"]["S_total"]))
            else:
                if type(row["success"]) is not bool:
                    raise ValueError("Invalid valid success field")
                scores[str(task)].append(float(row["success"]))
        if any(len(values) != size for values in scores.values()) or keys != expected_keys:
            raise ValueError("Incomplete or incomparable valid task inventory")
        if summary.get("jobs") != len(keys):
            raise ValueError("Valid summary and job inventory disagree")
        metrics = {key: mean(values) for key, values in scores.items()}
        if not all(0 <= v <= 1 for v in metrics.values()):
            raise ValueError("Valid metric is outside [0,1]")
        rank.append(
            {
                "checkpoint_manifest": str(checkpoint),
                "checkpoint_manifest_sha256": checkpoint_hash,
                "valid_run_sha256": file_sha256(run / "manifest.json"),
                "valid_summary_sha256": file_sha256(summary_path),
                "metrics": metrics,
                "selection_score": mean(metrics.values()),
            }
        )
    rank.sort(key=lambda item: (-item["selection_score"], item["checkpoint_manifest_sha256"]))
    result = {
        "status": "complete",
        "valid_manifest_sha256": valid_hash,
        "selection_rule": "equal_mean_of_S_total_T4_T5_with_sha256_tiebreak",
        "checkpoint_manifest_sha256": rank[0]["checkpoint_manifest_sha256"],
        "ranked": rank,
    }
    write_json(output, result)
    return result
