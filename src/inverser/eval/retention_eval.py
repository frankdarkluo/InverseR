"""Local, paired MMMU validation and RealWorldQA exact-answer retention evaluation."""

import ast
import io
import json
import re
import unicodedata
from itertools import islice
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
from pydantic import Field

from inverser.evaluation import BudgetExhausted, BudgetLimits, TrialLedger, UnresolvedRequest
from inverser.public_eval import _panels
from inverser.runtime import PROJECT_ROOT, file_sha256, fingerprint, write_json
from inverser.schemas import Record
from inverser.study_calls import StudyAdapter, StudyBackendConfig, load_job, lock_run, save_job
from inverser.vlm_eval import _png_asset, visual_request

SUITES = {"MMMU": 900, "RealWorldQA": 765}


class RetentionAnswer(Record):
    answer: str = Field(min_length=1, max_length=400)


def exact_answer(predicted: str, expected: str) -> bool:
    def normalized(answer: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", answer).casefold().split())

    reference = [expected]
    if expected.startswith("["):
        try:
            alternatives = ast.literal_eval(expected)
        except (ValueError, SyntaxError):
            pass
        else:
            if isinstance(alternatives, list):
                reference = alternatives
    return any(normalized(predicted) == normalized(str(value)) for value in reference)


def retention_prompt(dataset: str, example: dict[str, Any]) -> tuple[str, tuple[Image.Image, ...]]:
    prompt = str(example["question"])
    decoded = []
    captions = []
    keys = ("image",) if dataset == "RealWorldQA" else tuple(f"image_{i}" for i in range(1, 8))
    for i, key in enumerate(keys, 1):
        blob = example.get(key)
        if blob and blob["bytes"]:
            with Image.open(io.BytesIO(blob["bytes"])) as img:
                # Do not alter source files or persist modified benchmark images.
                picture = img.convert("RGB")
            decoded.append(picture)
            captions.append(f"M{i}")
    if dataset == "MMMU":
        prompt = re.sub(r"<image (\d+)>", r"[M\1]", prompt)
        if example["question_type"] == "multiple-choice":
            choices = ast.literal_eval(example["options"])
            prompt += "\n" + "\n".join(f"{chr(65 + i)}. {text}" for i, text in enumerate(choices))
            prompt += "\nReturn the option letter."
    prompt += '\nReturn only one concise JSON answer, e.g. {"answer":"A"}.'
    if len(decoded) > 2:
        prompt = "Match numbered visual references to the corresponding labeled panels.\n" + prompt
        decoded = [_panels(decoded, captions, 2)]
    elif captions:
        prompt = "Images in order: " + ", ".join(captions) + "\n" + prompt
    return prompt, tuple(decoded)


def run_retention(
    dataset: str,
    output: Path,
    backend: StudyBackendConfig,
    *,
    count: int | None,
    budget_tokens: int,
    execute: bool,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, object]:
    import pyarrow.parquet as pq

    if backend.kind not in ("local_vllm", "mock"):
        raise ValueError("Retention evaluation must keep the original visual assets local")
    if dataset not in SUITES or count is not None and not 0 < count <= SUITES[dataset]:
        raise ValueError("Invalid retention dataset or case limit")
    root = PROJECT_ROOT / "data/external/general_vlm_eval" / dataset
    manifest_path = PROJECT_ROOT / "data/external/manifest.json"
    artifacts = json.loads(manifest_path.read_text())["snapshots"]
    snapshot = next(item for item in artifacts if item["repo_id"].endswith("/" + dataset))
    files = sorted(path for path in root.rglob("*.parquet") if ".cache" not in path.parts)
    if sum(pq.ParquetFile(path).metadata.num_rows for path in files) != SUITES[dataset]:
        raise ValueError("Unexpected pinned Parquet inventory")
    for path in files:
        key = str(path.relative_to(root))
        if file_sha256(path) != snapshot["files"][key]:
            raise ValueError("Retention source checksum mismatch")
    total = count or SUITES[dataset]
    limits = BudgetLimits(
        max_model_requests=total,
        max_physics_evaluations=0,
        max_total_tokens=budget_tokens,
    )
    run_hash = lock_run(
        output,
        {
            "schema_version": "retention-exact-match-v1",
            "dataset": dataset,
            "source_revision": snapshot["revision"],
            "source_manifest_sha256": file_sha256(manifest_path),
            "count": total,
            "score": "NFKC_casefold_space_normalized_exact_match",
            "backend": backend.model_dump(mode="json"),
            "endpoint": fingerprint(backend.base_url),
            "limits": limits.model_dump(mode="json"),
            "producer": file_sha256(Path(__file__)),
        },
        reserve_bytes=20_000_000,
    )
    if not execute:
        return {"status": "planned_no_calls", "request_cap": total, "run_hash": run_hash}
    ledger = TrialLedger(output / "ledger.sqlite", limits)
    completed: list[dict[str, Any]] = []
    stop = None
    try:
        with StudyAdapter(
            backend.model_copy(update={"execution_authorized": True}), transport=transport
        ) as client:
            data = (
                row
                for path in files
                for batch in pq.ParquetFile(path).iter_batches(batch_size=8)
                for row in batch.to_pylist()
            )
            for idx, row in enumerate(islice(data, total)):
                path = output / "jobs" / f"{idx:05d}.json"
                if (saved := load_job(path, run_hash)) is not None:
                    completed.append(saved)
                    continue
                key = row.get("id", f"{dataset}-{idx:05d}")
                prompt, pictures = retention_prompt(dataset, row)
                assets = []
                for image in pictures:
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    assets.append(_png_asset(buffer.getvalue(), key))
                reply = visual_request(
                    client,
                    ledger,
                    prompt,
                    tuple(assets),
                    episode_key=key,
                    seed=0,
                    output_schema=RetentionAnswer.model_json_schema(),
                )
                try:
                    answer = RetentionAnswer.model_validate_json(reply.content).answer
                except ValueError:
                    answer = None
                result = {
                    "idx": key,
                    "dataset": dataset,
                    "answer": answer,
                    "question_type": row.get("question_type", "short_answer"),
                    "correct": answer is not None and exact_answer(answer, row["answer"]),
                    "reply": reply.model_dump(mode="json"),
                }
                save_job(path, run_hash, result)
                completed.append(result)
    except (httpx.HTTPError, BudgetExhausted, UnresolvedRequest, ValueError, SyntaxError) as error:
        stop = type(error).__name__
    finally:
        report: dict[str, object] = {
            "status": "complete" if stop is None and len(completed) == total else "stopped",
            "jobs": len(completed),
            "run_hash": run_hash,
            "stop_error": stop,
            "correct": sum(item["correct"] for item in completed),
            "invalid": sum(item["answer"] is None for item in completed),
            "counts": ledger.counts(),
            "usage": ledger.usage_totals(),
            "request_status": dict(
                ledger.connection.execute("SELECT status, COUNT(*) FROM requests GROUP BY status")
            ),
        }
        write_json(output / "summary.json", report)
        ledger.close()
    return report
