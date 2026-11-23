"""TRL GRPO integration for paired forward and simulator-grounded inverse rewards."""

import json
import os
import random
from collections import Counter
from collections.abc import Callable
from math import isfinite
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image

from inverser.capo.t5_feedback import feedback_pair, t5_feedback, t5_turn_prompt
from inverser.capo_env import (
    GoalEpisode,
    event_agreement,
    forward_reward,
    inverse_reward,
    twin_contrast,
    varied_group,
)
from inverser.capo_trainer import LORA_TARGETS
from inverser.model_adapters import CAUSAL_PROMPT
from inverser.physics import Replay
from inverser.runtime import file_sha256, write_json
from inverser.schemas import CueAction, Layout, PhysicsConfig
from inverser.study_calls import create_artifact_dir, producer_hashes
from inverser.training_env import goal_reward, tail_reward
from inverser.vlm_suite import (
    FAMILIES,
    VLMForwardInput,
    VLMForwardOutput,
    VLMGoalInput,
    forward_truth,
    normalize_goal_reply,
    parse_forward_answer,
    parse_goal_answer,
    parse_goal_forecast,
)

GROUP = 8
FAMILY_STAGES = (("free_roll",), ("free_roll", "center"), FAMILIES)
TWIN_WEIGHT = 0.5
ALPHA = 0.25
MODES = ("outcome", "additive", "capo")  # capo = FAPO with the multiplicative contact gate
IMPROVEMENT_WEIGHT = 0.25  # T5: credit for beating the first attempt's goal reward


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
    try:
        prediction = parse_forward_answer(
            text, ball_count=len(truth.resting_state), allow_json_fallback=False
        )
    except ValueError:
        return -1.0, None
    # alpha = 1 turns the coordinate gate off: additive E + C.
    alpha = 1.0 if mode == "additive" else ALPHA
    parts = forward_reward(
        prediction, truth, radius, alpha=alpha, reasoning_tokens=reasoning_tokens
    )
    if mode == "outcome":
        return parts.coordinate - parts.overrun_penalty, prediction
    return parts.total, prediction


def score_goal_completion(
    text: str, episode: GoalEpisode, *, mode: str, beta_e: float
) -> tuple[float, dict[str, float] | None]:
    """RL credit of one strict 7-tag T4/T5 reply executed on ``episode`` (earlier attempts
    already replayed) and the per-shot measurements logged each step; None when unusable."""
    try:
        forecast = parse_goal_forecast(text, episode.task.ball_count)
        shot = parse_goal_answer(text, allow_json_fallback=False)
    except ValueError:
        return -1.0, None
    if forecast is None:
        return -1.0, None
    replays: list[Replay] = []

    def capture(action: CueAction) -> Replay:
        replays.append(episode.simulator.replay(episode.initial, action))
        return replays[-1]

    attempt = episode.step(shot.model_dump_json(), replay_action=capture)
    baseline = episode.baseline_radii
    near = goal_reward(attempt.loss_radii, attempt.success, baseline_radii=baseline)
    tail = tail_reward(attempt.loss_radii, baseline)
    event, zero_contact = 0.0, 0.0
    if attempt.outcome is not None and replays:
        contacts = forward_truth(
            episode.initial, replays[-1], episode.simulator.config, episode.table
        )
        event = event_agreement(forecast, contacts)
        zero_contact = float(not any(contacts.object_collisions) and not contacts.cushion_sequence)
    reward = inverse_reward(mode, near, tail, event, alpha=ALPHA, beta_e=beta_e)
    if len(episode.attempts) > 1:
        first = episode.attempts[0]
        before = goal_reward(first.loss_radii, first.success, baseline_radii=baseline)
        before += tail_reward(first.loss_radii, baseline)
        reward += IMPROVEMENT_WEIGHT * max(0.0, near + tail - before)
    return reward, {
        "near": near,
        "tail": tail,
        "event": event,
        "success": float(attempt.success),
        "speed": shot.cue_speed_mps,
        "zero_contact": zero_contact,
    }


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


def rank_pair_keys(keys: tuple[Any, Any], rank: int, ranks: int) -> tuple[Any, ...]:
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


def curriculum_schedule(stage_steps: tuple[int, int, int]) -> list[tuple[int, str]]:
    """One schedule for every reward arm: arms differ only in how E enters the reward."""
    if any(steps < 2 for steps in stage_steps[:2]) or stage_steps[2] < 3:
        raise ValueError("Both early stages need two steps and the final one three")
    schedule: list[tuple[int, str]] = []
    for stage, steps in enumerate(stage_steps, 1):
        tasks = ("forward", "T4", "T5") if stage == 3 else ("forward", "T4")
        schedule.extend((stage, tasks[index % len(tasks)]) for index in range(steps))
    return schedule


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


def rl_rows(split: Path, *, stage: int, task: str, limit: int | None) -> list[dict]:
    manifest = json.loads((split / "manifest.json").read_text())
    if manifest["split"] != "train":
        raise ValueError("GRPO can only load training data")
    if stage not in (1, 2, 3):
        raise ValueError("Unknown CAPO stage")
    if task not in ("forward", "T4", "T5") or limit is not None and limit < 1:
        raise ValueError("Use forward, T4 or T5 and a positive optional scene limit")
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
            goal = VLMGoalInput.model_validate(scene["goals"][-1 if task == "T5" else 0])
            names = (forward.image,) if task == "forward" else (goal.initial_image, goal.goal_image)
            for name in names:
                path = (public / name).resolve()
                if (
                    not path.is_relative_to(public.resolve())
                    or file_sha256(path) != (manifest["artifact_sha256"][f"public/{name}"])
                ):
                    raise ValueError("Training image failed its path/hash check")
            # Every arm sees the evaluation prompt; T5 rows are its first turn.
            prompt = forward.prompt() if task == "forward" else t5_turn_prompt(goal, [], False)
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": CAUSAL_PROMPT},
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
    beta_e: float | None = None,
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
    if mode not in MODES:
        raise ValueError("Unknown reward arm")
    if (mode == "additive") != (beta_e is not None) or beta_e is not None and not (
        isfinite(beta_e) and beta_e >= 0
    ):
        raise ValueError("The additive arm alone needs a calibrated, finite, non-negative beta_e")
    if task == "T5":
        raise ValueError("T5 trains per turn inside --task joint")
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
    if joint and not server_url:
        raise ValueError("T5 history turns require a verified multimodal vLLM rollout server")
    schedule = curriculum_schedule(stage_steps) if stage_steps is not None else []
    if joint:
        forward_rows = rl_rows(split, stage=3, task="forward", limit=count)
        inverse_rows = rl_rows(split, stage=3, task="T4", limit=count)
        loop_rows = rl_rows(split, stage=3, task="T5", limit=count)
        rows = forward_rows + inverse_rows + loop_rows
    else:
        rows = rl_rows(split, stage=stage, task=task, limit=count)
    if output.exists():
        raise ValueError("A new trainer run requires a new output directory")
    resumed_step = 0
    if resume_from is not None:
        from inverser.study_calls import require_data_path

        require_data_path(resume_from)
        if not (fsdp and joint) or stage_steps is None:
            raise ValueError("Stage recovery is only supported for FSDP joint runs")
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
    paired = task in ("forward", "joint")  # anchor/twin sampling and DAPO for every arm
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

    def rewards(
        completion_ids: list[list[int]],
        scene_id: list[str],
        log_metric: Callable[[str, float], None] | None = None,
        t5_prefix: list[list[str]] | None = None,
        **other: Any,
    ) -> list[float]:
        nonlocal latest_rewards
        answers = []
        shaped = []
        valid_count = 0
        events: list[float] = []
        shots: list[dict[str, float]] = []
        prefixes = t5_prefix or [[] for _ in completion_ids]
        for ids, key, prefix in zip(completion_ids, scene_id, prefixes, strict=True):
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
                if prediction is not None:
                    events.append(event_agreement(prediction, truth))
                stats["valid_forward"] += is_valid
                shaped.append((key, reward, prediction, truth, forward.ball_radius_m))
            else:
                episode = episode_for(key, 0 if active_task == "T4" else -1)
                for earlier in prefix:  # T5 history turns replay deterministically
                    episode.step(normalize_goal_reply(earlier))
                reward, measured = score_goal_completion(
                    answer, episode, mode=mode, beta_e=beta_e or 0.0
                )
                is_valid = measured is not None
                if measured is not None:
                    events.append(measured["event"])
                    shots.append(measured)
                stats["valid_inverse"] += is_valid
            valid_count += int(is_valid)
            answers.append(reward)
            stats["completions"] += 1
            stats["reasoning_tokens"] += reasoning_tokens or 0
        if log_metric is not None and completion_ids:
            log_metric("reward/valid_format_rate", valid_count / len(completion_ids))
            if events:
                log_metric("reward/event_agreement", mean(events))
            if shots:
                speeds = [shot["speed"] for shot in shots]
                log_metric("reward/goal_hit_rate", mean(shot["success"] for shot in shots))
                log_metric("shot/near_mean", mean(shot["near"] for shot in shots))
                log_metric("shot/tail_mean", mean(shot["tail"] for shot in shots))
                log_metric("shot/zero_contact_rate", mean(shot["zero_contact"] for shot in shots))
                log_metric("shot/cue_speed_mean", mean(speeds))
                log_metric("shot/cue_speed_median", median(speeds))
            if active_task == "T5":
                log_metric("t5/turn_mean", mean(len(prefix) + 1 for prefix in prefixes))
        if paired and active_task == "forward" and mode != "outcome":
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
                stats[f"stage{active_stage}_{active_task}_optimizer_batches"] += 1
                if active_task != "forward":

                    def draw_inverse() -> tuple[dict, list[float]]:
                        keys = rng.choices(ids_by_stage[active_stage], k=2)
                        # Both ranks draw the same turns before the rank split.
                        turns = [
                            rng.randint(1, by_id[key]["scene"]["goals"][-1]["max_attempts"])
                            if active_task == "T5"
                            else 1
                            for key in keys
                        ]
                        scored = generate(self._expand(keys, active_task, turns))
                        return scored, gather_object(latest_rewards)

                    return resample_informative_pair(draw_inverse, stats)

            def draw() -> tuple[dict, list[float]]:
                keys = rng.choice(pairs_by_stage[active_stage] if joint else pairs)
                return generate(self._expand(keys, "forward")), gather_object(latest_rewards)

            return resample_informative_pair(draw, stats)

        def _expand(self, keys, kind: str, turns=(1, 1)) -> list[dict]:
            if kind == "T5" and self.state.global_step != self._last_loaded_step:
                # Every rank syncs before any rank-local history request. The sync gathers FSDP
                # shards, so starting it while another rank waits on the server deadlocks both.
                self.vllm_generation.sync_weights()
                self._last_loaded_step = self.state.global_step
                # Rank 0 resets the server caches after the last gather; wait for it before any
                # rank sends a history request, or a reset can hit an in-flight request.
                self.accelerator.wait_for_everyone()
            ready = []
            local = rank_pair_keys(tuple(zip(keys, turns)), self.accelerator.process_index, ranks)
            for key, turn in local:
                row = unpack(prepared[kind][key] if joint else by_id[key])
                prefix: list[str] = []
                if kind == "T5":
                    row["prompt"], row["images"], prefix = self._t5_turn(key, turn)
                else:
                    row["images"] = transform({"images": [row["images"]]})["images"][0]
                row["t5_prefix"] = prefix
                ready.extend([row] * GROUP)
            return ready

        def _t5_turn(self, key: str, turn: int) -> tuple[list[dict], list[Image.Image], list[str]]:
            """Play turns 1..turn-1 with the policy exactly as evaluation does and return turn
            ``turn``'s prompt, its two images and the earlier replies. A history turn that
            already reaches the goal is not kept: the group re-samples that turn instead."""
            episode = episode_for(key, -1)
            initial_path, goal_path = prepared["T5"][key]["images"]
            with Image.open(initial_path) as image:
                initial = image.convert("RGB")
            with Image.open(goal_path) as image:
                goal = image.copy()  # feedback_pair gets the PNG as evaluation decodes it
            history: list[str] = []
            replies: list[str] = []
            latest: Image.Image | None = None

            def turn_inputs() -> tuple[list[dict], list[Image.Image]]:
                text = t5_turn_prompt(episode.task, history, with_outcome=latest is not None)
                second = goal.convert("RGB") if latest is None else feedback_pair(goal, latest)
                pictures = [initial, second]
                prompt = [
                    {"role": "system", "content": CAUSAL_PROMPT},
                    {"role": "user", "content": text},
                ]
                return prompt, pictures

            for _ in range(turn - 1):
                reply = self._sample_text(*turn_inputs())
                replays: list[Replay] = []

                def capture(action: CueAction) -> Replay:
                    replays.append(episode.simulator.replay(episode.initial, action))
                    return replays[-1]

                if episode.step(normalize_goal_reply(reply), replay_action=capture).success:
                    break
                replies.append(reply)
                feedback, latest = t5_feedback(episode, replays[-1] if replays else None)
                history.append(feedback)
            stats["T5_history_turns"] += len(replies)
            return *turn_inputs(), replies

        def _sample_text(self, prompt: list[dict], images: list[Image.Image]) -> str:
            """One policy reply for a T5 history turn. Each rank calls the rollout server itself,
            without collectives, after `_expand` synced this step's weights."""
            from trl.data_utils import prepare_multimodal_messages

            generation = self.vllm_generation
            client = getattr(generation, "vllm_client", None) or getattr(
                self, "_local_history_client", None
            )
            if client is None:
                from trl.generation.vllm_client import VLLMClient

                client = VLLMClient(base_url=server_url)
                self._local_history_client = client
            prompt_ids, pictures, _ = self._tokenize_prompts(
                [prepare_multimodal_messages(prompt, images=images)]
            )
            features = client.image_features(pictures) if pictures else None
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
            return decode_completion(processor.tokenizer, result["completion_ids"][0])[1]

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

    class StageCheckpoints(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if stage_steps and state.global_step in (stage_steps[0], sum(stage_steps[:2])):
                control.should_save = True
            return control

    local_samples = (2 * GROUP if paired or joint else GROUP) // ranks
    per_device_bs = min(4, local_samples)
    config = GRPOConfig(
        output_dir=str(output),
        max_steps=steps,
        seed=seed,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=local_samples // per_device_bs,
        num_generations=GROUP,
        max_completion_length=1536,
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
        vllm_mode="server",
        vllm_server_base_url=server_url,
        vllm_importance_sampling_mode="token_truncate",
        generation_kwargs={"thinking_token_budget": 1024} if server_url else None,
        fsdp=fsdp,  # type: ignore[arg-type]
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
        trainer_class: Any = ShardedPairedTrainer if paired or joint else ShardedTrainer
    else:
        trainer_class = PairedTrainer if paired or joint else GRPOTrainer
    from inverser.wandb_tracker import (
        extract_dataset_metadata,
        finish_wandb_run,
        format_capo_run_name,
        make_hf_wandb_callback,
        start_wandb_run,
    )

    startup_producers = producer_hashes(
        "capo_grpo.py", "capo_env.py", "capo_checkpoint.py", "training_env.py", "t5_feedback.py"
    )
    twin_weight = TWIN_WEIGHT if paired and mode != "outcome" else None
    reward_definition = {
        "reward_composition": mode,
        "alpha": ALPHA,
        "beta_e": beta_e,
        "event_terms": ["object_collision_f1", "first_contact", "cushion_set"],
        "goal_reward": "1[succ] + max(0, 1 - L/min(20, L0)) + 0.2 max(0, 1 - L/L0)",
        "t5_training": "per_turn_single_turn",
        "twin_weight": twin_weight,
    }
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
            }
            | reward_definition
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
        reward_funcs=rewards,
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
        "max_pair_resampling_attempts": 8 if paired else None,
        "rollout_engine": "trl_vllm_server" if server_url else "transformers",
        "train_manifest_sha256": file_sha256(split / "manifest.json"),
        "sft_adapter_sha256": file_sha256(adapter / "adapter_model.safetensors"),
        "base_config_sha256": file_sha256(base / "config.json"),
        "producer": startup_producers,
        "lora_target_pattern": LORA_TARGETS,
    } | reward_definition
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
