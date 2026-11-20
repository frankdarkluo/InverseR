"""PhysBench image-route evaluation with upstream task IDs and a resumable request ledger."""

import io
import json
import re
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from zipfile import ZipFile

import httpx
from PIL import Image, ImageDraw, ImageOps
from pydantic import Field

from inverser.evaluation import BudgetExhausted, BudgetLimits, TrialLedger, UnresolvedRequest
from inverser.runtime import file_sha256, fingerprint, write_json
from inverser.schemas import Record
from inverser.study_calls import StudyAdapter, StudyBackendConfig, load_job, lock_run, save_job
from inverser.vlm_eval import _png_asset, visual_request

MEDIA = re.compile(r"<(video|image)>")


class PublicAnswer(Record):
    answer: str = Field(pattern=r"^[A-E](?: [A-E]){0,3}$")


def _panels(images: list[Image.Image], captions: list[str], columns: int) -> Image.Image:
    w, h = 352, 240
    result = Image.new("RGB", (columns * w, ((len(images) + columns - 1) // columns) * h), "white")
    draw = ImageDraw.Draw(result)
    for i, (image, caption) in enumerate(zip(images, captions, strict=True)):
        x, y = (i % columns) * w, i // columns * h
        image = ImageOps.contain(image, (w - 8, h - 24))
        result.paste(image, (x + (w - image.width) // 2, y + 20))
        draw.text((x + 5, y + 4), caption, fill="black")
    return result


def _video_frames(video: bytes) -> list[Image.Image]:
    import cv2

    frames = []
    with NamedTemporaryFile(suffix=".mp4") as file:
        file.write(video)
        file.flush()
        cap = cv2.VideoCapture(file.name)
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame in list(range(0, total, max(1, total // 4)))[:4]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, pixels = cap.read()
                if not ok:
                    raise ValueError(f"Unable to decode official video frame {frame}")
                frames.append(Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)))
        finally:
            cap.release()
    if not frames:
        raise ValueError("PhysBench video contains no readable frames")
    return frames


def physbench_prompt(
    task: dict, images: ZipFile, videos: ZipFile
) -> tuple[str, tuple[Image.Image, ...]]:
    names = task["file_name"]
    panels = []
    captions = []
    video_grid = None
    for index, name in enumerate(names, 1):
        if name.lower().endswith(".mp4"):
            if video_grid is not None:
                raise ValueError("Expected at most one PhysBench video")
            frames = _video_frames(videos.read(name))
            video_grid = _panels(frames, [f"M{index} frame {i + 1}" for i in range(len(frames))], 2)
        else:
            with Image.open(io.BytesIO(images.read(name))) as image:
                panels.append(image.convert("RGB"))
            captions.append(f"M{index}")
    # Original task order and option text stay intact. The visual representation is an adaptation.
    markers = iter(range(1, len(names) + 1))
    question = MEDIA.sub(lambda _: f"[M{next(markers, '?')}]", task["question"])
    if "[M?]" in question or next(markers, None) is not None and len(names) > 1:
        raise ValueError("PhysBench visual placeholders do not match referenced file list")
    visuals = (() if video_grid is None else (video_grid,)) + (
        () if not panels else (_panels(panels, captions, min(2, len(panels))),)
    )
    if not visuals:
        raise ValueError("PhysBench task has no visual evidence")
    return (
        "Read the question and match markers M1, M2, ... to the labeled visuals. "
        "Video grids show chronological frames in reading order. Answer with the option "
        'letter in JSON, e.g. {"answer":"A"}. Use space-separated letters only when the '
        "question asks for multiple answers.\n" + question,
        visuals,
    )


def run_physbench(
    source: Path,
    output: Path,
    backend: StudyBackendConfig,
    *,
    split: str = "val",
    collision_only: bool = False,
    count: int | None = None,
    budget_tokens: int,
    execute: bool,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, object]:
    if split not in ("val", "test") or count is not None and count <= 0:
        raise ValueError("Choose val or test and a positive optional case limit")
    dataset = source / "dataset"
    task_path = dataset / "test.json"
    label_path = source / "eval/physbench/test_answer.json"
    tasks = json.loads(task_path.read_text())
    labels = {row["idx"]: row for row in json.loads(label_path.read_text())}
    rows = [
        task
        for task in tasks
        if task["split"] == split
        and labels[task["idx"]]["task_type"] == "dynamics"
        and (not collision_only or labels[task["idx"]]["sub_type"] == "collision")
    ][:count]
    if not rows:
        raise ValueError("No matching official PhysBench tasks")
    limits = BudgetLimits(
        max_model_requests=len(rows),
        max_physics_evaluations=0,
        max_total_tokens=budget_tokens,
    )
    protocol: dict[str, object] = {
        "schema_version": "physbench-panels-v1",
        "split": split,
        "asset_hashes": {
            path.name: file_sha256(path)
            for path in (
                task_path,
                label_path,
                dataset / "image.zip",
                dataset / "video.zip",
            )
        },
        "ids": [row["idx"] for row in rows],
        "backend": backend.model_dump(mode="json"),
        "endpoint_hash": fingerprint(backend.base_url),
        "limits": limits.model_dump(mode="json"),
        "producer": file_sha256(Path(__file__)),
        "evaluation": "official_dynamics_rows_with_image_panel_adapter_not_official_leaderboard",
    }
    run_hash = lock_run(output, protocol, reserve_bytes=20_000_000)
    if not execute:
        return {"status": "planned_no_calls", "run_hash": run_hash, "request_cap": len(rows)}
    ledger = TrialLedger(output / "ledger.sqlite", limits)
    finished: list[dict[str, Any]] = []
    error_code = None
    try:
        with (
            StudyAdapter(
                backend.model_copy(update={"execution_authorized": True}), transport=transport
            ) as adapter,
            ZipFile(dataset / "image.zip") as image_zip,
            ZipFile(dataset / "video.zip") as video_zip,
        ):
            for task in rows:
                index = task["idx"]
                path = output / "jobs" / f"{index:05d}.json"
                if (saved := load_job(path, run_hash)) is not None:
                    finished.append(saved)
                    continue
                prompt, pictures = physbench_prompt(task, image_zip, video_zip)
                assets = []
                for image in pictures:
                    png = io.BytesIO()
                    image.save(png, format="PNG")
                    assets.append(_png_asset(png.getvalue(), str(index)))
                reply = visual_request(
                    adapter,
                    ledger,
                    prompt,
                    tuple(assets),
                    episode_key=f"PhysBench/{index}",
                    seed=0,
                    output_schema=PublicAnswer.model_json_schema(),
                )
                try:
                    answer = PublicAnswer.model_validate_json(reply.content).answer
                except ValueError:
                    answer = None
                label = labels[index]["answer"]
                # Upstream accepts exact or prefix match; blank labels are not exact answers.
                result = {
                    "idx": index,
                    "answer": answer,
                    "label": label,
                    "sub_type": labels[index]["sub_type"],
                    "correct": bool(label.strip() and answer is not None and answer == label),
                    "upstream_prefix_match": answer is not None and answer.startswith(label),
                    "blank_label": not label.strip(),
                    "reply": reply.model_dump(mode="json"),
                }
                save_job(path, run_hash, result)
                finished.append(result)
    except (httpx.HTTPError, BudgetExhausted, UnresolvedRequest, ValueError, KeyError) as error:
        error_code = type(error).__name__
    finally:
        correct = Counter(row["sub_type"] for row in finished if row["correct"])
        totals = Counter(row["sub_type"] for row in finished)
        report: dict[str, object] = {
            "status": "complete" if len(finished) == len(rows) else "stopped",
            "run_hash": run_hash,
            "expected": len(rows),
            "completed": len(finished),
            "error": error_code,
            "correct": sum(correct.values()),
            "invalid": sum(row["answer"] is None for row in finished),
            "blank_labels": sum(row["blank_label"] for row in finished),
            "upstream_prefix_matches": sum(row["upstream_prefix_match"] for row in finished),
            "subtypes": {key: {"correct": correct[key], "total": n} for key, n in totals.items()},
            "usage": ledger.usage_totals(),
            "counts": ledger.counts(),
            "request_status": dict(
                ledger.connection.execute("SELECT status, COUNT(*) FROM requests GROUP BY status")
            ),
        }
        write_json(
            output / "submission.json",
            [{"idx": row["idx"], "answer": row["answer"]} for row in finished],
        )
        write_json(output / "summary.json", report)
        ledger.close()
    return report
