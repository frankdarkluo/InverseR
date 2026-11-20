import json
from pathlib import Path

import httpx
import pytest
from PIL import ImageChops
from pydantic import ValidationError

from inverser.capo_trainer import sft_examples
from inverser.physics import SnookerSimulator
from inverser.preview import render_vlm_forward_scene, render_vlm_goal_pair
from inverser.schemas import BallPosition, CueAction, Layout, PhysicsConfig
from inverser.study import public_physics
from inverser.study_calls import local_vllm_backends
from inverser.vlm_eval import run_vlm_eval
from inverser.vlm_suite import (
    WITHHELD,
    CoordinateOnly,
    VLMForwardOutput,
    VLMGoalInput,
    VLMGoalOutput,
    format_coordinate_answer,
    format_forward_answer,
    format_goal_answer,
    forward_scores,
    forward_truth,
    layout_group,
    parse_coordinate_answer,
    parse_forward_answer,
    parse_goal_answer,
    prepare_vlm_suite,
    verify_vlm_suite,
)


def test_forward_metrics_and_strict_response_contract():
    values = {
        "object_collisions": (True,),
        "first_contact": 1,
        "cushion_set": ("TOP", "LEFT"),
        "cushion_sequence": ("TOP", "LEFT", "TOP"),
        "frames": (((0.0, 0.0), (0.0, 1.0)),) * 5,
        "resting_state": ((1.0, 0.0), "pocketed"),
    }
    truth = VLMForwardOutput.model_validate(values)
    reordered = VLMForwardOutput.model_validate(values | {"cushion_set": ("LEFT", "TOP")})
    assert forward_scores(reordered, truth, 0.125)["S_total"] == 1
    assert forward_scores(reordered, truth, 0.125)["S_total_3r"] == 1
    predicted = VLMForwardOutput.model_validate(
        values
        | {
            "object_collisions": (False,),
            "first_contact": None,
            "cushion_sequence": ("LEFT", "TOP", "TOP"),
            "frames": (((0.125, 0.0), (0.0, 1.0)),) * 5,
            "resting_state": ((1.126, 0.0), (0.0, 0.0)),
        }
    )
    scores = forward_scores(predicted, truth, 0.125)
    assert scores == {
        "T1": False,
        "T2": False,
        "T3": 10 / 12,
        "T3_all_slots": False,
        "S_total": 0.4 * (10 / 12),
        "first_contact_exact": False,
        "cushion_sequence_exact": False,
        "T1_per_ball": 0.0,
        "T2_cushion_set": True,
        "T3_1r": 10 / 12,
        "T3_2r": 11 / 12,
        "T3_3r": 11 / 12,
        "T3_5r": 11 / 12,
        "S_total_3r": 0.3 + 0.4 * (11 / 12),
    }
    for edit in (
        {"object_collisions": ("true",)},
        {"first_contact": 2},
        {"cushion_set": ("TOP",)},
        {"frames": (((1.0, 0.0),),) * 5},
    ):
        with pytest.raises(ValidationError):
            VLMForwardOutput.model_validate(values | edit)
    with pytest.raises(ValidationError):
        VLMGoalOutput(
            azimuth_deg=0,
            cue_speed_mps=0.3,
            tip_side=0.5,
            tip_vertical=0.5,
        )


def test_answer_tag_serializers_and_strict_parsers_cover_edge_cases():
    multi = VLMForwardOutput(
        object_collisions=(True, False),
        first_contact=1,
        cushion_set=("LEFT", "BOTTOM"),
        cushion_sequence=("LEFT", "BOTTOM", "LEFT"),
        frames=(((0.224, 1.692), (1.050, 2.110), (0.500, 0.600)),) * 5,
        resting_state=((1.020, 0.145), "pocketed", (0.500, 0.600)),
    )
    tagged = (
        "<think>Cue hits object 1 then LEFT, BOTTOM, LEFT.</think>\n"
        f"{format_forward_answer(multi)}"
    )
    parsed = parse_forward_answer(tagged, ball_count=3, allow_json_fallback=False)
    assert parsed == multi

    solo = VLMForwardOutput(
        object_collisions=(),
        first_contact=None,
        cushion_set=(),
        cushion_sequence=(),
        frames=(((0.400, 0.800),),) * 5,
        resting_state=((0.400, 1.250),),
    )
    assert "<collisions>none</collisions>" in format_forward_answer(solo)
    assert "<cushions>none</cushions>" in format_forward_answer(solo)
    assert parse_forward_answer(format_forward_answer(solo), ball_count=1) == solo
    assert forward_scores(solo, solo, 0.0262)["S_total_3r"] == 1.0

    coords = CoordinateOnly(frames=multi.frames, resting_state=multi.resting_state)
    assert (
        parse_coordinate_answer(
            format_coordinate_answer(coords), ball_count=3, allow_json_fallback=False
        )
        == coords
    )

    shot = VLMGoalOutput(azimuth_deg=62.45, cue_speed_mps=1.38, tip_side=0.0, tip_vertical=0.25)
    tagged_shot = f"<think>follow shot</think>\n{format_goal_answer(shot)}"
    assert parse_goal_answer(tagged_shot, allow_json_fallback=False) == shot
    # Historical JSON fallback remains available only when enabled.
    assert parse_goal_answer(shot.model_dump_json()) == shot
    with pytest.raises(ValueError):
        parse_goal_answer(shot.model_dump_json(), allow_json_fallback=False)
    for bad in (
        "<answer><azimuth_deg>10</azimuth_deg></answer>",
        f"{format_goal_answer(shot)}\n{format_goal_answer(shot)}",
        format_forward_answer(multi).replace("0: 1.020", "2: 1.020"),
    ):
        with pytest.raises(ValueError):
            parse_forward_answer(bad, ball_count=3, allow_json_fallback=False)


def test_forward_truth_keeps_numbered_contact_and_bounce_identity():
    physics = PhysicsConfig()
    sim = SnookerSimulator(physics)
    table = public_physics(sim).table
    initial = Layout(
        balls=(
            BallPosition(ball_id="cue", kind="cue", xy_m=(0.45, 0.25)),
            BallPosition(ball_id="test_red", kind="red", xy_m=(0.78, 0.25)),
        )
    )
    action = CueAction(azimuth_deg=0, cue_speed_mps=2.5, tip_side=0, tip_vertical=0)
    replay = sim.replay(initial, action)
    truth = forward_truth(initial, replay, physics, table)
    assert truth.object_collisions == (True,) and truth.first_contact == 1
    assert truth.cushion_sequence[0] == "RIGHT"
    assert truth.cushion_set[0] == "RIGHT"
    assert len(truth.frames) == 5 and len(truth.resting_state) == 2
    assert forward_scores(truth, truth, physics.ball_radius_m)["S_total"] == 1
    images = render_vlm_goal_pair(initial.endpoint_view(), replay.goal.endpoint_view(), table)
    assert all(max(image.size) == 1024 for image in images)
    a = render_vlm_forward_scene(initial.endpoint_view(), table, action)
    faster = action.model_copy(update={"cue_speed_mps": 3.0})
    b = render_vlm_forward_scene(initial.endpoint_view(), table, faster)
    assert a.size == b.size and ImageChops.difference(a, b).getbbox()


def test_prepare_and_verify_twin_splits(tmp_path, monkeypatch, study_fixture):
    from inverser import study_calls, vlm_suite

    _, record, _ = study_fixture
    monkeypatch.setattr(study_calls, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(vlm_suite, "check_storage", lambda _: None)
    pilot = tmp_path / "pilot.jsonl"
    pilot.write_text(record.model_dump_json() + "\n")
    formal = tmp_path / "formal"
    (formal / "private").mkdir(parents=True)
    (formal / "private/selection.json").write_text(
        json.dumps({"formal_test": [{"case_id": "fixture"}]})
    )
    (formal / "private/fixture.json").write_text(json.dumps({"record": record.model_dump()}))
    root = tmp_path / "data/research"
    options = dict(counts=(1, 1, 1), ood_per_family=2)
    receipt = prepare_vlm_suite(root, pilot, formal, **options)
    assert receipt["group_overlap"] == 0
    assert receipt["splits"]["test"]["witness_replays_verified"] == 16
    verified = verify_vlm_suite(root, pilot, formal, **options)
    assert verified["status"] == "verified" and verified["splits"]["train"]["scenes"] == 8
    seen = {layout_group(record.initial)}
    for split in ("train", "valid", "test"):
        here = root / "inverser-vlm-v1" / split
        groups = set(json.loads((here / "groups.json").read_text()))
        assert not groups & seen
        seen.update(groups)
        inputs = _rows(here / "public/scenes.jsonl")
        for scene in inputs:
            for item in scene["goals"]:
                goal = VLMGoalInput.model_validate(item)
                assert "action" not in item and "twin_effect_radii" not in item
                if goal.withheld_parameters:
                    assert goal.withheld_parameters == WITHHELD
                    assert goal.max_attempts == 6 and scene["forward"] is None
                    assert not set(goal.parameters) & set(WITHHELD)
                    assert goal.initial_image.startswith(("images/ood/slow-", "images/ood/fast-"))
                    assert goal.goal_image.endswith("-goal.png")
                else:
                    assert goal.initial_image.startswith(
                        ("images/roll/", "images/center/", "images/vert/", "images/side/")
                    )
    with pytest.raises(ValueError, match="immutable"):
        prepare_vlm_suite(root, pilot, formal, **options)
    train = root / "inverser-vlm-v1/train"
    examples = sft_examples(train, count=1)
    assert [len(row["image_paths"]) for row in examples] == [1, 2]
    fwd_comp = examples[0]["completion"][0]["content"]
    inv_comp = examples[1]["completion"][0]["content"]
    inv_think = inv_comp.split("</think>", 1)[0]
    scene = _rows(train / "private/labels.jsonl")[0]
    assert scene["scene_id"].startswith("train-") and len(scene["layout_sha256"]) == 64
    truth_out = VLMForwardOutput.model_validate(scene["truth"])
    action = CueAction.model_validate(scene["action"])
    assert "Cue direction" not in inv_think and "Simulator event prefix" not in inv_think
    assert f"{action.cue_speed_mps:.3f}" not in inv_think
    parsed_fwd = parse_forward_answer(fwd_comp, allow_json_fallback=False)
    assert parsed_fwd.object_collisions == truth_out.object_collisions
    assert parsed_fwd.cushion_sequence == truth_out.cushion_sequence
    assert forward_scores(parsed_fwd, truth_out, 0.0262)["S_total"] == 1.0
    parsed_inv = parse_goal_answer(inv_comp, allow_json_fallback=False).cue_action()
    assert parsed_inv.azimuth_deg == pytest.approx(action.azimuth_deg, abs=1e-3)
    with pytest.raises(ValueError, match="training"):
        sft_examples(root / "inverser-vlm-v1/valid", count=1)
    error_action = action.model_copy(update={"azimuth_deg": (action.azimuth_deg + 180) % 360})
    response_text = [
        f"<think>trace</think>\n{format_forward_answer(truth_out)}",
        "<answer>invalid</answer>",
        format_goal_answer(error_action),
        format_goal_answer(action),
    ]
    prompts = []
    payloads = []

    def handle(request):
        body = json.loads(request.content)
        payloads.append(body)
        prompts.append(body["messages"][1]["content"])
        return httpx.Response(
            200,
            json={
                "model": "fixture",
                "choices": [{"message": {"content": response_text[len(prompts) - 1]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    backend = local_vllm_backends(
        "http://127.0.0.1:8000/v1",
        ("fixture",),
        model_revision="fixture",
        reasoning_effort="none",
        max_output_tokens=200,
        request_token_reservation=1000,
        timeout_s=30,
    )[0]
    run_options = dict(
        tasks=("forward", "T4", "T5"),
        count=1,
        budget_tokens=8000,
        execute=True,
        transport=httpx.MockTransport(handle),
    )
    run = root / "evaluation"
    evaluated = run_vlm_eval(train, run, backend, **run_options)
    assert evaluated["status"] == "complete" and evaluated["jobs"] == 3
    assert all("response_format" not in body and body["stop"] == ["</answer>"] for body in payloads)
    assert all(body["include_stop_str_in_output"] is True for body in payloads)
    t5_job = json.loads(next((run / "jobs").glob("*-T5.json")).read_text())["payload"]
    assert t5_job["success"] is True and len(t5_job["attempts"]) == 2
    fwd_job = json.loads(next((run / "jobs").glob("*-forward.json")).read_text())["payload"]
    assert fwd_job["valid"] is True
    assert fwd_job["scores"]["S_total_3r"] >= fwd_job["scores"]["S_total"]
    assert evaluated["counts"] == {
        "model_requests": 4,
        "physics_attempts": 2,
        "physics_evaluations": 2,
    }
    assert [len(message) for message in prompts] == [2, 3, 3, 3]
    assert "latest outcome (right)" in prompts[3][0]["text"]
    assert (
        '"action":' not in prompts[1][0]["text"] and "Attempt observations" in prompts[3][0]["text"]
    )
    resumed = run_vlm_eval(train, run, backend, **run_options)
    assert len(prompts) == 4 and resumed["simulator_rollouts_this_process"] == 0
    with pytest.raises(ValueError, match="Blind test"):
        run_vlm_eval(here, root / "blind", backend, **run_options)
    (here / "groups.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_vlm_suite(root, pilot, formal, **options)


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]
