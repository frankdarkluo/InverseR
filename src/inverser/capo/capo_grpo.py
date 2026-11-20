"""TRL GRPO integration for paired forward and simulator-grounded inverse rewards."""

import json
import os
import random
from collections import Counter
from collections.abc import Callable
from math import isfinite
from pathlib import Path
from typing import Any

from PIL import Image

from inverser.capo_env import (
    CAUSAL_PROMPT,
    GoalEpisode,
    forward_reward,
    twin_contrast,
    varied_group,
)
from inverser.capo_trainer import LORA_TARGETS
from inverser.runtime import file_sha256, write_json
from inverser.schemas import Layout, PhysicsConfig
from inverser.study_calls import create_artifact_dir, producer_hashes
from inverser.vlm_suite import (
    FAMILIES,
    HORIZON_TAGS,
    VLMForwardInput,
    VLMForwardOutput,
    VLMGoalInput,
    parse_coordinate_answer,
    parse_forward_answer,
)

GROUP = 8
FAMILY_STAGES = (("free_roll",), ("free_roll", "center"), FAMILIES)
TWIN_WEIGHT = 0.5
T5_SYSTEM_PROMPT = (
    "In the reasoning segment (<think>...</think>, within 1024 tokens), analyze ball "
    "positions, contact geometry, friction, and prior shot residuals, then call try_shot "
    "to test your calibrated shot until the goal is reached."
)


def decode_completion(tokenizer: Any, ids: list[int]) -> tuple[str | None, str, int | None]:
    # Qwen starts <think> in the prompt; decode generation without deleting </think>.
    begin = tokenizer.convert_tokens_to_ids("<think>")
    end = tokenizer.convert_tokens_to_ids("</think>")
    if end not in ids:
        return None, "", None
    boundary = ids.index(end)
    if begin in ids[boundary + 1 :] or end in ids[boundary + 1 :]:
        return None, "", None
    trace = ids[:boundary]
    if trace and trace[0] == begin:
        trace = trace[1:]
    if begin in trace:
        return None, "", None
    return (
        tokenizer.decode(trace, skip_special_tokens=True),
        tokenizer.decode(ids[boundary + 1 :], skip_special_tokens=True).strip(),
        len(trace),
    )


def score_forward_completion(
    text: str,
    truth: VLMForwardOutput,
    radius: float,
    *,
    mode: str,
    reasoning_tokens: int | None,
) -> tuple[float, VLMForwardOutput | None]:
    ball_count = len(truth.resting_state)
    try:
        if mode == "capo":
            prediction = parse_forward_answer(
                text, ball_count=ball_count, allow_json_fallback=False
            )
        else:
            answer = parse_coordinate_answer(
                text, ball_count=ball_count, allow_json_fallback=False
            )
            prediction = VLMForwardOutput.model_validate(
                answer.model_dump()
                | {
                    "object_collisions": (False,) * len(truth.object_collisions),
                    "first_contact": None,
                    "cushion_set": (),
                    "cushion_sequence": (),
                }
            )
    except ValueError:
        return -1.0, None
    parts = forward_reward(prediction, truth, radius, reasoning_tokens=reasoning_tokens)
    return (parts.total if mode == "capo" else parts.coordinate), prediction


def pair_shaped_rewards(
    a: list[tuple[float, VLMForwardOutput | None]],
    b: list[tuple[float, VLMForwardOutput | None]],
    truths: tuple[VLMForwardOutput, VLMForwardOutput],
    radius: float,
) -> tuple[list[float], list[float]]:
    left, right = [], []
    for (ra, pa), (rb, pb) in zip(a, b, strict=True):
        extra = TWIN_WEIGHT * twin_contrast((pa, pb), truths, radius) if pa and pb else 0
        left.append(ra + extra)
        right.append(rb + extra)
    return left, right


def rank_pair_keys(keys: tuple[str, str], rank: int, ranks: int) -> tuple[str, ...]:
    if ranks not in (1, 2) or not 0 <= rank < ranks:
        raise ValueError("CAPO supports one or two trainer ranks")
    return keys if ranks == 1 else (keys[rank],)


def aligned_pair_rewards(
    samples: list[tuple[str, float, VLMForwardOutput | None, VLMForwardOutput, float]],
) -> list[float]:
    if len(samples) != 2 * GROUP:
        raise ValueError("Paired rollout did not return two aligned groups of eight")
    key_a, key_b = samples[0][0], samples[GROUP][0]
    if key_a == key_b or [s[0] for s in samples] != [key_a] * GROUP + [key_b] * GROUP:
        raise ValueError("Paired rollout requires one anchor and one twin in rank order")
    if any(row[4] != samples[0][4] for row in samples):
        raise ValueError("Paired rollout tables disagree")
    a, b = pair_shaped_rewards(
        [(r, p) for _, r, p, _, _ in samples[:GROUP]],
        [(r, p) for _, r, p, _, _ in samples[GROUP:]],
        (samples[0][3], samples[GROUP][3]),
        samples[0][4],
    )
    return a + b


def complete_twins(rows: list[dict]) -> list[tuple[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    for row in rows:
        label = row["label"]
        members = grouped.setdefault(label["group_id"], {})
        role = label["twin_role"]
        if role not in ("anchor", "twin") or role in members:
            raise ValueError("Duplicate or unknown twin role")
        members[role] = label["scene_id"]
    pairs = [(pair["anchor"], pair["twin"]) for pair in grouped.values() if len(pair) == 2]
    if not pairs:
        raise ValueError("Twin training requires at least one complete layout pair")
    return pairs


def curriculum_schedule(
    stage_steps: tuple[int, int, int], mode: str = "capo"
) -> list[tuple[int, str]]:
    if any(steps < 2 for steps in stage_steps[:2]) or stage_steps[2] < 3:
        raise ValueError("Both early stages need two steps and the final one three")
    schedule: list[tuple[int, str]] = []
    for stage, steps in enumerate(stage_steps, 1):
        tasks = ("forward", "T4", "T5") if stage == 3 else ("forward", "T4")
        schedule.extend((stage, tasks[index % len(tasks)]) for index in range(steps))
    if mode == "capo":
        return schedule
    # Keep total optimizer and forward steps fixed across arms. Non-interactive controls
    # spend the T5 allocation on T4; all families are available from the start.
    control_tasks = ["T4" if task == "T5" else task for _, task in schedule]
    if mode == "uncoupled":
        control_tasks = ["forward"] * control_tasks.count("forward") + ["T4"] * control_tasks.count(
            "T4"
        )
    elif mode != "outcome":
        raise ValueError("Unknown CAPO schedule arm")
    return [(3, task) for task in control_tasks]


def resample_informative_pair(
    draw: Callable[[], tuple[dict, list[float]]], stats: Counter[str]
) -> dict:
    for _ in range(8):
        scored, sampled = draw()
        if len(sampled) != 2 * GROUP:
            raise ValueError("DAPO expects exactly two rollout groups")
        if varied_group(sampled[:GROUP]) and varied_group(sampled[GROUP:]):
            stats["accepted_pair_groups"] += 1
            stats["accepted_completions"] += 2 * GROUP
            return scored
        stats["rejected_zero_variance_pair_groups"] += 1
        del scored
    raise RuntimeError("DAPO exhausted eight attempts without a nondegenerate twin pair")


def rl_rows(split: Path, *, stage: int, mode: str, task: str, limit: int | None) -> list[dict]:
    manifest = json.loads((split / "manifest.json").read_text())
    if manifest["split"] != "train":
        raise ValueError("GRPO can only load training data")
    if stage not in (1, 2, 3) or mode not in ("capo", "outcome", "uncoupled"):
        raise ValueError("Unknown CAPO stage or reward arm")
    if task not in ("forward", "T4", "T5") or limit is not None and limit < 1:
        raise ValueError("Use forward, T4 or T5 and a positive optional scene limit")
    if task == "T5" and mode != "capo":
        raise ValueError("The outcome-only control does not receive T5 interactions")
    sources = [split / name for name in ("public/scenes.jsonl", "private/labels.jsonl")]
    for source in sources:
        if file_sha256(source) != manifest["artifact_sha256"][str(source.relative_to(split))]:
            raise ValueError("Training data does not match the frozen manifest")
    rows = []
    public = split / "public"
    with sources[0].open() as scenes, sources[1].open() as labels:
        for scene_line, label_line in zip(scenes, labels, strict=True):
            scene, label = json.loads(scene_line), json.loads(label_line)
            if scene["scene_id"] != label["scene_id"]:
                raise ValueError("Public and private training rows do not match")
            if label["family"] not in FAMILY_STAGES[stage - 1]:
                continue
            forward = VLMForwardInput.model_validate(scene["forward"])
            goal = VLMGoalInput.model_validate(scene["goals"][0])
            names = (forward.image,) if task == "forward" else (goal.initial_image, goal.goal_image)
            for name in names:
                path = (public / name).resolve()
                if (
                    not path.is_relative_to(public.resolve())
                    or file_sha256(path) != (manifest["artifact_sha256"][f"public/{name}"])
                ):
                    raise ValueError("Training image failed its path/hash check")
            system_prompt = CAUSAL_PROMPT
            prompt = (
                forward.prompt()
                if task == "forward"
                else goal.prompt() + "\nThe images in order are INITIAL and GOAL."
            )
            if task == "forward" and mode != "capo":
                slots_hint = " | ".join(
                    f"{slot}: 0.500, 1.000" for slot in range(forward.ball_count)
                )
                coord_template = (
                    "<answer>\n"
                    + "\n".join(f"<{tag}>{slots_hint}</{tag}>" for tag in HORIZON_TAGS)
                    + "\n</answer>"
                )
                info = forward.model_dump(
                    mode="json", exclude={"image", "scene_id", "task_ids"}
                )
                prompt = (
                    f"Track {forward.ball_count} numbered balls (0..{forward.ball_count - 1}, "
                    "0 = cue) from this top-down image after the specified cue action (x right, "
                    "y up in metres). Predict every numbered slot in order at 1s..5s and when "
                    "stationary using 'slot: x, y' or 'slot: pocketed'. Return only the <answer> "
                    f"block:\n{coord_template}\n"
                    + json.dumps(info, separators=(",", ":"), allow_nan=False)
                )
            elif task == "T5":
                system_prompt = T5_SYSTEM_PROMPT
                goal_t5 = VLMGoalInput.model_validate(scene["goals"][-1])
                meta = json.dumps(
                    goal_t5.model_dump(
                        mode="json", exclude={"initial_image", "goal_image", "episode_id"}
                    ),
                    separators=(",", ":"),
                    allow_nan=False,
                )
                prompt = (
                    "Propose shots from the INITIAL image to reach the GOAL image without "
                    "scratching (x right, y up; 0 <= azimuth_deg < 360, 0 < cue_speed_mps <= 3, "
                    "-0.5 <= tip_side, tip_vertical <= 0.5). The images in order are INITIAL and "
                    "GOAL. Use the try_shot function to execute each attempt and inspect its "
                    "outcome before revising. Stop after success or six attempts. A final message "
                    "without a simulated shot does not count as a successful attempt.\n" + meta
                )
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "images": [str(public / name) for name in names],
                    "scene": scene,
                    "label": label,
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("Stage has no training rows")
    return rows


def run_grpo(
    split: Path,
    base: Path,
    adapter: Path,
    output: Path,
    *,
    mode: str,
    task: str,
    stage: int,
    steps: int,
    seed: int,
    count: int | None,
    server_url: str | None,
    stage_steps: tuple[int, int, int] | None = None,
    fsdp: bool = False,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    import torch
    from accelerate import PartialState
    from accelerate.utils import gather_object
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from inverser.physics import SnookerSimulator

    if not torch.cuda.is_available() or steps <= 0:
        raise ValueError("GRPO needs CUDA and a positive step allocation")
    joint = task == "joint"
    ranks = int(os.environ.get("WORLD_SIZE", "1"))
    if (ranks, fsdp) not in ((1, False), (2, True)):
        raise ValueError("Launch one rank directly or two ranks with --fsdp")
    if fsdp and not server_url:
        raise ValueError("Two-rank CAPO requires a separate vLLM rollout server")
    if joint and (stage_steps is None or sum(stage_steps) != steps):
        raise ValueError("Joint CAPO requires three curriculum step budgets summing to --steps")
    if not joint and stage_steps is not None:
        raise ValueError("Stage step budgets apply only to joint CAPO")
    if task in ("T5", "joint") and not server_url:
        raise ValueError("T5 integration requires a verified multimodal vLLM rollout server")
    schedule = curriculum_schedule(stage_steps, mode) if stage_steps is not None else []
    interactive = task == "T5" or joint and mode == "capo"
    if joint:
        forward_rows = rl_rows(split, stage=3, mode=mode, task="forward", limit=count)
        inverse_rows = rl_rows(split, stage=3, mode=mode, task="T4", limit=count)
        loop_rows = (
            rl_rows(split, stage=3, mode=mode, task="T5", limit=count) if interactive else []
        )
        rows = forward_rows + inverse_rows + loop_rows
    else:
        rows = rl_rows(split, stage=stage, mode=mode, task=task, limit=count)
    if output.exists():
        raise ValueError("A new trainer run requires a new output directory")
    resumed_step = 0
    if resume_from is not None:
        from inverser.study_calls import require_data_path

        require_data_path(resume_from)
        if not (fsdp and joint and mode == "capo") or stage_steps is None:
            raise ValueError("Stage recovery is only supported for FSDP joint CAPO")
        previous = json.loads((resume_from.parent / "manifest.json").read_text())
        state = json.loads((resume_from / "trainer_state.json").read_text())
        resumed_step = state["global_step"]
        if (
            previous["curriculum_optimizer_schedule"] != [list(row) for row in schedule]
            or previous["seed"] != seed
            or previous["requested_steps"] != steps
            or previous["trainer_ranks"] != ranks
            or previous["sft_adapter_sha256"] != file_sha256(adapter / "adapter_model.safetensors")
            or previous["train_manifest_sha256"] != file_sha256(split / "manifest.json")
            or resumed_step not in (stage_steps[0], sum(stage_steps[:2]))
        ):
            raise ValueError("Resume requires an exact completed curriculum stage checkpoint")
    # Construct GRPOConfig on both ranks before Accelerate initializes FSDP.
    backbone = AutoModelForImageTextToText.from_pretrained(
        base,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    policy = PeftModel.from_pretrained(backbone, adapter, is_trainable=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    by_id = {row["label"]["scene_id"]: row for row in rows}
    paired = mode == "capo" and task in ("forward", "joint")
    pairs = complete_twins(forward_rows if joint else rows) if paired else []
    if joint:
        prepared = {
            "forward": {row["label"]["scene_id"]: row for row in forward_rows},
            "T4": {row["label"]["scene_id"]: row for row in inverse_rows},
            "T5": {row["label"]["scene_id"]: row for row in loop_rows},
        }
        ids_by_stage = {
            current: [
                key
                for key, row in prepared["forward"].items()
                if row["label"]["family"] in FAMILY_STAGES[current - 1]
            ]
            for current in (1, 2, 3)
        }
        pairs_by_stage = {
            current: [
                pair
                for pair in pairs
                if prepared["forward"][pair[0]]["label"]["family"] in FAMILY_STAGES[current - 1]
            ]
            for current in (1, 2, 3)
        }
        if paired and not all(pairs_by_stage.values()):
            raise ValueError("Every curriculum stage needs at least one complete twin pair")
    rng = random.Random(seed)
    simulated = None
    stats: Counter[str] = Counter()
    latest_rewards: list[float] = []
    loop_rewards: list[float] = []
    active_task = task

    def episode_for(key: str, goal_index: int) -> GoalEpisode:
        nonlocal simulated
        if simulated is None:
            simulated = SnookerSimulator(PhysicsConfig())
        row = by_id[key]
        return GoalEpisode(
            VLMGoalInput.model_validate(row["scene"]["goals"][goal_index]),
            Layout.model_validate(row["label"]["initial"]),
            Layout.model_validate(row["label"]["goal"]),
            simulated,
        )

    class TrainingEpisode:
        def reset(self, scene_id: str, **kwargs: Any) -> None:
            self.episode = episode_for(scene_id, -1) if active_task == "T5" else None
            self.invalid_calls = 0

        def try_shot(
            self,
            azimuth_deg: float,
            cue_speed_mps: float,
            tip_side: float,
            tip_vertical: float,
        ) -> list[dict]:
            """Execute one billiards shot, returning the observed terminal state and residual.

            Args:
                azimuth_deg: Aim direction in degrees, from +x towards +y.
                cue_speed_mps: Positive cue-stick speed no greater than 3 m/s.
                tip_side: Horizontal offset of the tip.
                tip_vertical: Vertical offset of the tip.
            """
            env = self.episode
            if env is None or env.done:
                return [{"type": "text", "text": "Episode terminated; do not make more attempts."}]
            action_json = json.dumps(
                {
                    "azimuth_deg": azimuth_deg,
                    "cue_speed_mps": cue_speed_mps,
                    "tip_side": tip_side,
                    "tip_vertical": tip_vertical,
                }
            )
            attempt = env.step(action_json)
            if attempt.action is None:
                self.invalid_calls += 1
            stats["T5_tool_attempts"] += 1
            feedback, image = env.feedback(attempt)
            blocks: list[dict[str, Any]] = [{"type": "text", "text": feedback}]
            if image is not None:
                # Downsample the full table so up to six feedback frames fit the context.
                image.thumbnail((224, 384), Image.Resampling.LANCZOS)
                blocks.append({"type": "image", "image": image})
            return blocks

        def get_reward(self) -> float | None:
            if self.episode is None:
                return None
            attempts = self.episode.attempts
            if not attempts or all(attempt.action is None for attempt in attempts):
                reward = -1.0
            else:
                self_reward = attempts[-1].reward
                if len(attempts) > 1:
                    self_reward += 0.25 * max(0.0, self_reward - attempts[0].reward)
                reward = self_reward - 0.05 * self.invalid_calls
            if joint:
                # Joint mode records this channel before the callback applies DAPO filtering.
                loop_rewards.append(reward)
            return reward

    def rewards(
        completion_ids: list[list[int]],
        scene_id: list[str],
        log_metric: Callable[[str, float], None] | None = None,
        **other: Any,
    ) -> list[float]:
        nonlocal latest_rewards
        answers = []
        shaped = []
        valid_count = 0
        t4_hits = 0
        t4_losses: list[float] = []
        for ids, key in zip(completion_ids, scene_id, strict=True):
            _, answer, reasoning_tokens = decode_completion(processor.tokenizer, ids)
            row = by_id[key]
            label, scene = row["label"], row["scene"]
            if active_task == "forward":
                forward = VLMForwardInput.model_validate(scene["forward"])
                truth = VLMForwardOutput.model_validate(label["truth"])
                reward, prediction = score_forward_completion(
                    answer,
                    truth,
                    forward.ball_radius_m,
                    mode=mode,
                    reasoning_tokens=reasoning_tokens,
                )
                is_valid = prediction is not None
                valid_count += int(is_valid)
                stats["valid_forward"] += is_valid
                shaped.append((key, reward, prediction, truth, forward.ball_radius_m))
            elif active_task == "T4":
                attempt = episode_for(key, 0).step(answer)
                is_valid = attempt.action is not None
                valid_count += int(is_valid)
                stats["valid_inverse"] += is_valid
                t4_hits += int(attempt.success)
                if attempt.loss_radii is not None:
                    t4_losses.append(float(attempt.loss_radii))
                reward = attempt.reward if is_valid else -1.0
            else:
                return [0.0] * len(completion_ids)
            answers.append(reward)
            stats["completions"] += 1
            stats["reasoning_tokens"] += reasoning_tokens or 0
        if log_metric is not None and completion_ids:
            log_metric("reward/valid_format_rate", valid_count / len(completion_ids))
            if active_task == "T4":
                log_metric("reward/t4_goal_hit_rate", t4_hits / len(completion_ids))
                if t4_losses:
                    log_metric("reward/t4_mean_loss_radii", sum(t4_losses) / len(t4_losses))
        if paired and active_task == "forward":
            scores = aligned_pair_rewards(gather_object(shaped))
            offset = distributed.process_index * len(answers)
            answers = scores[offset : offset + len(answers)]
            stats["paired_rollout_groups"] += 1
        latest_rewards = answers
        return answers

    def transform(batch: dict) -> dict:
        visuals = []
        for names in batch["images"]:
            group = []
            for name in names:
                with Image.open(name) as image:
                    group.append(image.convert("RGB"))
            visuals.append(group)
        return batch | {"images": visuals}

    def unpack(row: dict) -> dict:
        return {
            "prompt": row["prompt"],
            "images": row["images"],
            "scene_id": row["label"]["scene_id"],
        }

    dataset = Dataset.from_list([unpack(row) for row in rows]).with_transform(transform)

    class PairedTrainer(GRPOTrainer):
        def _generate_and_score_completions(self, inputs: list[dict]) -> dict:
            if self.accelerator.num_processes != ranks or len(inputs) != 2 * GROUP // ranks:
                raise ValueError("Expected one local rollout group per rank under FSDP")
            generate = super()._generate_and_score_completions
            nonlocal active_task
            active_stage = stage
            if joint:
                active_stage, active_task = schedule[self.state.global_step]
                self.vllm_generation.max_completion_length = 4096 if active_task == "T5" else 1536
                stats[f"stage{active_stage}_{active_task}_optimizer_batches"] += 1
                if not paired:
                    keys = rng.choices(ids_by_stage[active_stage], k=2)
                    return generate(self._expand(keys, active_task))
                if active_task != "forward":

                    def draw_inverse() -> tuple[dict, list[float]]:
                        keys = rng.choices(ids_by_stage[active_stage], k=2)
                        loop_rewards.clear()
                        scored = generate(self._expand(keys, active_task))
                        sampled = loop_rewards if active_task == "T5" else latest_rewards
                        return scored, gather_object(sampled)

                    return resample_informative_pair(draw_inverse, stats)

            def draw() -> tuple[dict, list[float]]:
                keys = rng.choice(pairs_by_stage[active_stage] if joint else pairs)
                return generate(self._expand(keys, "forward")), gather_object(latest_rewards)

            return resample_informative_pair(draw, stats)

        def _expand(self, keys, kind: str) -> list[dict]:
            ready = []
            for key in rank_pair_keys(tuple(keys), self.accelerator.process_index, ranks):
                row = unpack(prepared[kind][key] if joint else by_id[key])
                row["images"] = transform({"images": [row["images"]]})["images"][0]
                if joint and interactive:
                    row["environment"] = kind
                ready.extend([row] * GROUP)
            return ready

    class NoToolEpisode:
        def reset(self, **kwargs: Any) -> None:
            pass

    class ShardedTrainer(GRPOTrainer):
        def create_model_card(self, *args: Any, **kwargs: Any) -> None:
            return None

        def _save_checkpoint(self, model, trial):
            super()._save_checkpoint(model, trial)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                from inverser.capo_checkpoint import export_fsdp_adapter

                checkpoint = output / f"checkpoint-{self.state.global_step}"
                export_fsdp_adapter(checkpoint, adapter, checkpoint / "full_adapter")
            self.accelerator.wait_for_everyone()

        def save_collective_final(self) -> None:
            from accelerate.utils.fsdp_utils import save_fsdp_model

            final = output / "checkpoint"
            self.save_model(str(final))
            save_fsdp_model(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.model,
                str(final),
                adapter_only=True,
            )
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                from inverser.capo_checkpoint import export_fsdp_adapter

                export_fsdp_adapter(final, adapter, final / "full_adapter")
            self.accelerator.wait_for_everyone()

    class ShardedPairedTrainer(PairedTrainer, ShardedTrainer):
        pass

    class DistributedT5Trainer(ShardedPairedTrainer):
        def _tool_call_loop(self, *args, **kwargs):
            if active_task != "T5":
                return super()._tool_call_loop(*args, **kwargs)
            # TRL's loop is local: ranks can finish at different turns and have different prompt
            # counts. Its stock server helper has equal-count offsets and calls collectives per
            # local turn. Each rank sends its own follow-up prompts without a collective.
            # Only the main rank owns the shared client's NCCL weight-transfer communicator.
            generation = self.vllm_generation
            original = getattr(self, "_generate_single_turn")
            client = getattr(generation, "vllm_client", None) or getattr(
                self, "_local_tool_client", None
            )
            if client is None:
                from trl.generation.vllm_client import VLLMClient

                client = VLLMClient(base_url=server_url)
                self._local_tool_client = client

            def post_tool(prompt_ids, images, fields, num_generations, **other):
                # The initial call already synchronized policy weights. All follow-up prompts
                # are unique, stay on their owner, and use that same policy snapshot.
                if num_generations != 1:
                    raise ValueError("Each post-tool history needs exactly one completion")
                features = client.image_features(images) if images else None
                if features is not None:
                    features = [
                        generation._place_features(feature, ids)
                        for feature, ids in zip(features, prompt_ids, strict=True)
                    ]
                result = client.generate(
                    prompts=prompt_ids,
                    features=features,
                    n=1,
                    temperature=generation.temperature,
                    top_p=generation.top_p,
                    top_k=generation.top_k,
                    min_p=generation.min_p or 0.0,
                    repetition_penalty=generation.repetition_penalty,
                    max_tokens=generation.max_completion_length,
                    logprobs=generation.logprobs,
                    generation_kwargs=generation.generation_kwargs,
                )
                ids = result["completion_ids"]
                logprobs = [[token[0] for token in row] for row in result["logprobs"]]
                if len(ids) != len(prompt_ids):
                    raise RuntimeError("vLLM post-tool generation lost a local prompt")
                return ids, logprobs, None

            setattr(self, "_generate_single_turn", post_tool)
            try:
                return super()._tool_call_loop(*args, **kwargs)
            finally:
                setattr(self, "_generate_single_turn", original)

    class StageCheckpoints(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if (
                mode == "capo"
                and stage_steps
                and state.global_step in (stage_steps[0], sum(stage_steps[:2]))
            ):
                control.should_save = True
            if mode == "uncoupled" and state.global_step == sum(
                task == "forward" for _, task in schedule
            ):
                control.should_save = True
            return control

    local_samples = (2 * GROUP if paired or joint else GROUP) // ranks
    per_device_bs = 2 if interactive else min(4, local_samples)
    config = GRPOConfig(
        output_dir=str(output),
        max_steps=steps,
        seed=seed,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=local_samples // per_device_bs,
        num_generations=GROUP,
        max_completion_length=4096 if interactive else 1536,
        temperature=0.8,
        beta=0.0,
        learning_rate=5e-6,
        gradient_checkpointing=not fsdp,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        scale_rewards="group",
        loss_type="dapo",
        report_to=[],
        remove_unused_columns=False,
        logging_steps=1,
        save_strategy="steps" if joint else "no",
        save_steps=steps + 1,
        disable_tqdm=True,
        use_vllm=bool(server_url),
        max_tool_calling_iterations=6 if interactive else None,
        vllm_mode="server",
        vllm_server_base_url=server_url,
        vllm_importance_sampling_mode="token_truncate",
        generation_kwargs={"thinking_token_budget": 1024} if server_url else None,
        fsdp=fsdp,
        fsdp_config={
            "version": 2,
            "activation_checkpointing": True,
            "transformer_layer_cls_to_wrap": ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"],
        }
        if fsdp
        else None,
    )
    distributed = PartialState()
    if distributed.is_main_process:
        create_artifact_dir(output, 2_000_000_000)
    distributed.wait_for_everyone()
    if fsdp:
        trainer_class = (
            DistributedT5Trainer
            if joint and interactive
            else ShardedPairedTrainer
            if paired or joint
            else ShardedTrainer
        )
    else:
        trainer_class = PairedTrainer if paired or joint else GRPOTrainer
    from inverser.wandb_tracker import (
        extract_dataset_metadata,
        finish_wandb_run,
        format_capo_run_name,
        make_hf_wandb_callback,
        start_wandb_run,
    )

    startup_producers = producer_hashes("capo_grpo.py", "capo_env.py", "capo_checkpoint.py")
    wandb_name = format_capo_run_name(
        mode=mode,
        task=task,
        steps=steps,
        seed=seed,
        stage_steps=stage_steps,
        group_size=GROUP,
        kl_beta=0.0,
        learning_rate=5e-6,
    )
    wandb_run = (
        start_wandb_run(
            name=wandb_name,
            job_type="capo-grpo",
            config={
                "reward": mode,
                "task": task,
                "stage": stage,
                "stage_steps": list(stage_steps) if stage_steps else None,
                "seed": seed,
                "requested_steps": steps,
                "group_size": GROUP,
                "temperature": 0.8,
                "kl_beta": 0.0,
                "learning_rate": 5e-6,
                "loss_type": "dapo",
                "vllm_importance_sampling_mode": "token_truncate",
                "thinking_token_budget": 1024 if server_url else None,
                "trainer_ranks": ranks,
                "trainer_sharding": "FSDP2" if fsdp else None,
                "paired_forward": paired,
                "twin_weight": TWIN_WEIGHT if paired else None,
            }
            | extract_dataset_metadata(split),
            tags=["capo", mode, task],
            group=f"stage1-seed{seed}",
        )
        if distributed.is_main_process and count is None
        else None
    )
    callbacks: list[Any] = [StageCheckpoints()] if joint else []
    wandb_cb = make_hf_wandb_callback(
        wandb_run,
        schedule=schedule,
        csv_path=(output / "metrics.csv") if distributed.is_main_process else None,
    )
    if wandb_cb is not None:
        callbacks.append(wandb_cb)
    trainer: Any = trainer_class(
        model=policy,
        reward_funcs=None if task == "T5" else rewards,
        environment_factory=(
            {"forward": NoToolEpisode, "T4": NoToolEpisode, "T5": TrainingEpisode}
            if joint and interactive
            else TrainingEpisode
            if task == "T5"
            else None
        ),
        args=config,
        processing_class=processor,
        train_dataset=dataset,
        callbacks=callbacks or None,
    )
    manifest = {
        "purpose": (
            "joint_curriculum"
            if joint
            else "paired_forward_dapo_integration_not_full_capo"
            if paired
            else "single_task_grpo_integration_not_full_capo"
        ),
        "reward": mode,
        "task": task,
        "stage": stage,
        "curriculum_optimizer_schedule": schedule,
        "seed": seed,
        "requested_steps": steps,
        "group_size": GROUP,
        "trainer_ranks": ranks,
        "trainer_sharding": "FSDP2" if fsdp else None,
        "resumed_step": resumed_step,
        "resumed_from_collective_sha256": (
            file_sha256(resume_from / "pytorch_model_fsdp.bin") if resume_from else None
        ),
        "paired_forward": paired,
        "matched_control_t5_reallocated_to_T4": joint and not interactive,
        "twin_weight": TWIN_WEIGHT if paired else None,
        "max_pair_resampling_attempts": 8 if paired else None,
        "rollout_engine": "trl_vllm_server" if server_url else "transformers",
        "train_manifest_sha256": file_sha256(split / "manifest.json"),
        "sft_adapter_sha256": file_sha256(adapter / "adapter_model.safetensors"),
        "base_config_sha256": file_sha256(base / "config.json"),
        "producer": startup_producers,
        "lora_target_pattern": LORA_TARGETS,
    }
    if distributed.is_main_process:
        write_json(output / "manifest.json", manifest)
        from shutil import copyfile

        (output / "source").mkdir()
        for filename, digest in startup_producers.items():
            path = Path(__file__).with_name(filename)
            if file_sha256(path) != digest:
                raise RuntimeError("Trainer source changed during startup")
            copyfile(path, output / "source" / filename)
    result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    if fsdp:
        trainer.save_collective_final()
    else:
        trainer.save_model(str(output / "checkpoint"))
    measured = [
        float(item["grad_norm"]) for item in trainer.state.log_history if "grad_norm" in item
    ]
    if len(measured) != steps or not all(isfinite(x) for x in measured):
        raise RuntimeError("Training did not log finite gradients at every optimizer step")
    reduced: Counter[str] = Counter()
    # Local completions and physics calls are sums; acceptance decisions are replicated.
    for row in gather_object([dict(stats)]):
        for key, value in row.items():
            if key.startswith(("paired_", "accepted_", "rejected_", "stage")):
                reduced[key] = max(reduced[key], value)
            else:
                reduced[key] += value
    summary = {
        "status": "complete",
        "steps": result.global_step,
        "gradient_norms": measured,
        "nonzero_gradient_steps": sum(x > 0 for x in measured),
        "reward_receipts": dict(reduced),
        "simulator_rollouts": sum(gather_object([simulated.rollouts if simulated else 0])),
        "history": trainer.state.log_history,
        "adapter_directory": "checkpoint/full_adapter" if fsdp else "checkpoint",
    }
    if distributed.is_main_process:
        write_json(output / "summary.json", summary)
        final_metrics: dict[str, Any] = {
            "summary/steps": result.global_step,
            "summary/nonzero_gradient_steps": sum(x > 0 for x in measured),
            "summary/simulator_rollouts": summary["simulator_rollouts"],
        }
        for key, val in reduced.items():
            final_metrics[f"receipts/{key}"] = val
        finish_wandb_run(
            wandb_run,
            summary_metrics=final_metrics,
            artifact_name=wandb_name,
            artifact_type="capo-grpo-metadata",
            artifact_files=[
                output / "manifest.json",
                output / "summary.json",
                output / "checkpoint/full_adapter/manifest.json",
            ],
        )
        return summary
    return {}
