import io
import json
from zipfile import ZipFile

import httpx
from PIL import Image

from inverser import study_calls
from inverser.public_eval import run_physbench
from inverser.study_calls import local_vllm_backends


def test_physbench_keeps_task_ids_scores_and_completed_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(study_calls, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "source"
    (source / "dataset").mkdir(parents=True)
    labels = source / "eval/physbench"
    labels.mkdir(parents=True)
    cases = [
        {"idx": 3, "split": "val", "question": "<image> A or B?", "file_name": ["fixture.png"]},
        {"idx": 8, "split": "test", "question": "<image> C or D?", "file_name": ["fixture.png"]},
    ]
    (source / "dataset/test.json").write_text(json.dumps(cases))
    (labels / "test_answer.json").write_text(
        json.dumps(
            [
                {"idx": 3, "answer": "", "task_type": "dynamics", "sub_type": "collision"},
                {"idx": 8, "answer": "C", "task_type": "dynamics", "sub_type": "collision"},
            ]
        )
    )
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10), "green").save(buffer, format="PNG")
    with ZipFile(source / "dataset/image.zip", "w") as archive:
        archive.writestr("fixture.png", buffer.getvalue())
    with ZipFile(source / "dataset/video.zip", "w"):
        pass
    backend = local_vllm_backends(
        "http://127.0.0.1:8000/v1",
        ("fixture",),
        model_revision="fixture",
        reasoning_effort="none",
        max_output_tokens=128,
        request_token_reservation=4096,
        timeout_s=30,
    )[0]
    sent = []

    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(
            200,
            json={
                "model": "fixture",
                "choices": [{"message": {"content": '{"answer":"A"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    output = tmp_path / "data/research/phys"
    for _ in range(2):
        summary = run_physbench(
            source,
            output,
            backend,
            split="val",
            budget_tokens=4096,
            execute=True,
            transport=httpx.MockTransport(handle),
        )
        assert summary["status"] == "complete" and summary["request_status"] == {"complete": 1}
        assert summary["correct"] == 0 and summary["upstream_prefix_matches"] == 1
        assert summary["blank_labels"] == 1 and summary["completed"] == 1
    assert len(sent) == 1
    assert "[M1] A or B?" in sent[0]["messages"][1]["content"][0]["text"]
    assert "C or D?" not in str(sent[0]["messages"][1]["content"][0])
    assert json.loads((output / "submission.json").read_text()) == [{"idx": 3, "answer": "A"}]
