import io

from PIL import Image

from inverser.retention_eval import exact_answer, retention_prompt


def test_retention_handles_open_alternatives_without_substring_credit():
    assert exact_answer(" florida ", "['Tampa', 'Florida']")
    assert exact_answer("Ａ", "A")
    assert not exact_answer("I predict A", "A")
    assert exact_answer("[not a list]", "[not a list]")


def test_mmmu_panel_keeps_relative_image_order_and_does_not_send_label():
    buffer = io.BytesIO()
    Image.new("RGB", (50, 20), "green").save(buffer, format="PNG")
    example = {
        "question": "Compare <image 1> <image 2> <image 3>.",
        "options": "['red','blue']",
        "answer": "secret",
        "question_type": "multiple-choice",
    }
    for i in range(1, 4):
        example[f"image_{i}"] = {"bytes": buffer.getvalue()}
    prompt, pages = retention_prompt("MMMU", example)
    assert "[M1] [M2] [M3]" in prompt and "A. red\nB. blue" in prompt
    assert "secret" not in prompt
    assert len(pages) == 1 and pages[0].size == (704, 480)
