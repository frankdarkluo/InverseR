"""CAPO supervised arm: verified simulator traces and completion-only multimodal LoRA."""

import json
import math
from itertools import islice
from pathlib import Path
from typing import Any

from PIL import Image

from inverser.model_adapters import CAUSAL_PROMPT as TAG_CAUSAL_PROMPT
from inverser.runtime import file_sha256, write_json
from inverser.schemas import CueAction, Layout
from inverser.study_calls import create_artifact_dir, producer_hashes
from inverser.vlm_suite import (
    VLMForwardInput,
    VLMForwardOutput,
    VLMGoalInput,
    format_forward_answer,
    format_goal_answer,
)

# Top 6 of the 27 ViT blocks see patches before the 2x2 spatial merge (sub-patch ball centres).
LORA_TARGETS = (
    r".*(language_model.*\.(q_proj|k_proj|v_proj|o_proj)|visual\.merger\.linear_fc[12]"
    r"|visual\.blocks\.2[1-6]\.(attn\.(qkv|proj)|mlp\.linear_fc[12]))"
)


def _forward_reasoning_trace(
    initial: Layout, forward: VLMForwardInput, truth: VLMForwardOutput, events: list[dict[str, Any]]
) -> str:
    slots = initial.endpoint_view().balls
    init_desc = ", ".join(
        f"slot {idx} ({ball.kind}) at ({ball.xy_m[0]:.3f}, {ball.xy_m[1]:.3f})m"
        for idx, ball in enumerate(slots)
        if ball.xy_m is not None
    )
    act = forward.action
    contacts = [
        f"{event['event_type']}({'/'.join(event['agents'])})"
        for event in events
        if event["event_type"]
        in {
            "ball_ball",
            "ball_linear_cushion",
            "ball_circular_cushion",
            "ball_pocket",
            "sliding_rolling",
        }
    ][:12]
    chain = " -> ".join(contacts) if contacts else "pure planar deceleration without contact"
    cushions = ", ".join(truth.cushion_sequence) if truth.cushion_sequence else "none"
    first = "none" if truth.first_contact is None else f"object slot {truth.first_contact}"
    return (
        f"1. Visual layout: {init_desc}. "
        f"2. Applied cue command: azimuth={act.azimuth_deg:.2f} deg, speed={act.cue_speed_mps:.2f} "
        f"m/s, tip_side={act.tip_side:.2f}, tip_vertical={act.tip_vertical:.2f}. "
        f"3. Causal contact & spin rollout: {chain}; first cue-object contact={first}; "
        f"ordered cushion bounces={cushions}. "
        "4. Integrate post-contact sliding-to-rolling trajectories at 1s..5s until stationary."
    )


def _inverse_reasoning_trace(
    initial: Layout,
    goal: Layout,
    truth: VLMForwardOutput | None,
    events: list[dict[str, Any]],
    ball_radius_m: float,
) -> str:
    init_balls = initial.endpoint_view().balls
    goal_balls = goal.endpoint_view().balls
    cue_init = init_balls[0].xy_m or (0.0, 0.0)
    cue_goal = goal_balls[0].xy_m
    cue_target_desc = (
        f"({cue_goal[0]:.3f}, {cue_goal[1]:.3f})m" if cue_goal is not None else "on-table safe zone"
    )
    if len(init_balls) == 1 and cue_goal is not None:
        dx, dy = cue_goal[0] - cue_init[0], cue_goal[1] - cue_init[1]
        direct_deg = math.degrees(math.atan2(dy, dx)) % 360.0
        dist_m = math.hypot(dx, dy)
        geom = (
            f"Single-ball planar path from cue ({cue_init[0]:.3f}, {cue_init[1]:.3f})m to goal "
            f"{cue_target_desc} (displacement {dist_m:.3f}m along ~{direct_deg:.1f} deg)."
        )
    else:
        obj_init = [
            f"slot {i} at ({b.xy_m[0]:.3f}, {b.xy_m[1]:.3f})m"
            for i, b in enumerate(init_balls[1:], start=1)
            if b.xy_m is not None
        ]
        obj_goal = [
            f"({b.xy_m[0]:.3f}, {b.xy_m[1]:.3f})m" if b.xy_m is not None else "pocketed"
            for b in goal_balls[1:]
        ]
        first_obj = init_balls[1].xy_m if len(init_balls) > 1 else None
        ghost_note = ""
        if first_obj is not None and goal_balls[1].xy_m is not None:
            ox, oy = goal_balls[1].xy_m[0] - first_obj[0], goal_balls[1].xy_m[1] - first_obj[1]
            span = math.hypot(ox, oy)
            if span > 1e-6:
                gx = first_obj[0] - 2.0 * ball_radius_m * ox / span
                gy = first_obj[1] - 2.0 * ball_radius_m * oy / span
                aim = math.degrees(math.atan2(gy - cue_init[1], gx - cue_init[0])) % 360.0
                ghost_note = (
                    f" Ghost-ball contact center near ({gx:.3f}, {gy:.3f})m gives initial "
                    f"geometric line ~{aim:.1f} deg."
                )
        geom = (
            f"Cue starts at ({cue_init[0]:.3f}, {cue_init[1]:.3f})m targeting {cue_target_desc}; "
            f"objects move from [{', '.join(obj_init)}] to [{', '.join(obj_goal)}].{ghost_note}"
        )
    has_cushion = any("cushion" in ev["event_type"] for ev in events)
    has_pot = any(ev["event_type"] == "ball_pocket" for ev in events)
    cushions = (
        ", ".join(truth.cushion_sequence)
        if truth is not None and truth.cushion_sequence
        else ("cushion rebound required" if has_cushion else "direct line without rail bounce")
    )
    return (
        f"1. Visual state comparison: {geom} "
        f"2. Contact topology: cushion sequence={cushions}; object pot={has_pot}. "
        "3. Friction & 3D spin calibration: account for cue-ball throw, sliding-to-rolling "
        "transition, and post-impact tangent deflection (follow/draw/side offset) to match both "
        "object-ball displacement and cue-ball resting position within 1 ball radius."
    )


def sft_examples(
    split: Path, count: int | None = None, *, check_image_hashes: bool = True
) -> list[dict[str, Any]]:
    if count is not None and count <= 0:
        raise ValueError("SFT sample limit must be positive")
    manifest = json.loads((split / "manifest.json").read_text())
    if manifest["split"] != "train":
        raise ValueError("SFT data must be sourced from the training split only")
    public = split / "public"
    paths = (public / "scenes.jsonl", split / "private/labels.jsonl")
    for path in paths:
        if file_sha256(path) != manifest["artifact_sha256"][str(path.relative_to(split))]:
            raise ValueError(f"SFT dataset hash mismatch: {path.name}")
    rows = []
    with paths[0].open() as scenes, paths[1].open() as labels:
        for scene_line, label_line in islice(zip(scenes, labels, strict=True), count):
            scene, label = json.loads(scene_line), json.loads(label_line)
            if scene["scene_id"] != label["scene_id"]:
                raise ValueError("SFT input and truth ordering disagree")
            forward = VLMForwardInput.model_validate(scene["forward"])
            goal = VLMGoalInput.model_validate(scene["goals"][0])
            truth = VLMForwardOutput.model_validate(label["truth"])
            action = CueAction.model_validate(label["action"])
            initial = Layout.model_validate(label["initial"])
            target = Layout.model_validate(label["goal"])
            forward_trace = _forward_reasoning_trace(initial, forward, truth, label["events"])
            inverse_trace = _inverse_reasoning_trace(
                initial, target, truth, label["events"], goal.ball_radius_m
            )
            for prompt, images, trace, answer in (
                (forward.prompt(), (forward.image,), forward_trace, format_forward_answer(truth)),
                (
                    goal.prompt() + "\nThe images in order are INITIAL and GOAL.",
                    (goal.initial_image, goal.goal_image),
                    inverse_trace,
                    # The shot plus the contacts it produces (the label truth is that rollout).
                    format_goal_answer(action, forecast=truth),
                ),
            ):
                for name in images:
                    image = (public / name).resolve()
                    if not image.is_relative_to(public.resolve()):
                        raise ValueError("SFT image path leaves the public split")
                    if file_sha256(image) != manifest["artifact_sha256"][f"public/{name}"]:
                        raise ValueError(f"SFT image hash mismatch: {name}")
                rows.append(
                    {
                        "prompt": [
                            {"role": "system", "content": TAG_CAUSAL_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "completion": [
                            {
                                "role": "assistant",
                                "content": f"<think>\n{trace}\n</think>\n{answer}",
                            }
                        ],
                        "image_paths": [str(public / name) for name in images],
                    }
                )
    if not rows:
        raise ValueError("SFT requires at least one complete training scene")
    return rows


def _training_dataset(split: Path, count: int | None):
    from datasets import Dataset

    manifest = json.loads((split / "manifest.json").read_text())
    if manifest.get("schema_version") != "inverser-sft-v2":
        return Dataset.from_list(sft_examples(split, count))
    payload = split / "sft.jsonl"
    if manifest["split"] != "train" or (count is not None and count <= 0):
        raise ValueError("SFT requires a train-only corpus and a positive sample limit")
    if file_sha256(payload) != manifest["artifact_sha256"]["sft.jsonl"]:
        raise ValueError("SFT corpus hash mismatch")
    dataset = Dataset.from_json(str(payload), keep_in_memory=False)
    if len(dataset) != manifest["sft_examples"]["total"]:
        raise ValueError("SFT corpus size does not match the manifest")
    return dataset if count is None else dataset.select(range(min(len(dataset), 2 * count)))


def run_sft(
    split: Path,
    model: Path,
    output: Path,
    *,
    count: int | None,
    steps: int,
    seed: int,
    effective_batch_size: int = 32,
    save_every: int | None = None,
    fsdp: bool = False,
) -> dict[str, object]:
    # Keep optional GPU packages and CUDA initialization out of corpus and scoring commands.
    import torch
    from accelerate import PartialState
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    if not torch.cuda.is_available():
        raise RuntimeError("CAPO SFT requires CUDA")
    if steps < 1 or effective_batch_size < 1:
        raise ValueError("SFT needs positive optimizer-step and batch-size budgets")
    distributed = PartialState()
    torch.manual_seed(seed)
    with distributed.main_process_first():
        dataset = _training_dataset(split, count)
    world_size = max(1, distributed.num_processes)
    per_device_bs = 4 if len(dataset) >= 4 * world_size else 1
    grad_accum = max(1, effective_batch_size // (world_size * per_device_bs))
    if distributed.is_main_process:
        create_artifact_dir(output, 2_000_000_000)
    distributed.wait_for_everyone()
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    backbone = AutoModelForImageTextToText.from_pretrained(
        model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    backbone.config.use_cache = False
    # The language model reads text_config; FSDP activation checkpointing would grow its KV cache.
    backbone.config.text_config.use_cache = False
    policy = get_peft_model(
        backbone,
        LoraConfig(r=64, lora_alpha=128, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"),
    )
    names = [name for name, param in policy.named_parameters() if param.requires_grad]
    if not all(
        any(part in name for name in names)
        for part in (".visual.merger.", ".visual.blocks.26.", ".language_model.")
    ):
        raise RuntimeError("LoRA must cover the language model, top ViT blocks and the merger")
    collator = DataCollatorForVisionLanguageModeling(processor, completion_only_loss=True)
    relative_images = (
        json.loads((split / "manifest.json").read_text()).get("schema_version") == "inverser-sft-v2"
    )

    def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
        for example in examples:
            images = []
            for path in example["image_paths"]:
                location = (split / path).resolve() if relative_images else Path(path)
                if relative_images and not location.is_relative_to(split.resolve()):
                    raise ValueError(f"SFT image path leaves the corpus: {path}")
                with Image.open(location) as image:
                    images.append(image.convert("RGB"))
            example["images"] = images
        batch = collator(examples)
        if not (batch["labels"] != -100).any(dim=1).all():
            raise ValueError("Each SFT row must expose completion tokens to the loss")
        return batch

    settings = TrainingArguments(
        output_dir=str(output),
        max_steps=steps,
        seed=seed,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        gradient_checkpointing=not fsdp,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        learning_rate=2e-5,
        logging_steps=1,
        # Intermediate adapters (checkpoint-N) let validation track T4 during long SFT runs.
        save_strategy="steps" if save_every else "no",
        save_steps=save_every or 500,
        save_only_model=not fsdp,  # FSDP writes pytorch_model_fsdp.bin with the trainer state
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=True,
        # Same FSDP2 sharding as train-capo-grpo --fsdp; needed once the base outgrows a GPU (27B).
        fsdp=fsdp,  # type: ignore[arg-type]
        fsdp_config={
            "version": 2,
            "activation_checkpointing": True,
            "transformer_layer_cls_to_wrap": ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"],
        }
        if fsdp
        else None,
    )
    from inverser.wandb_tracker import (
        extract_dataset_metadata,
        finish_wandb_run,
        format_sft_run_name,
        make_hf_wandb_callback,
        start_wandb_run,
    )

    eff_bs = world_size * per_device_bs * grad_accum
    wandb_name = format_sft_run_name(
        model_name=model.name,
        lora_rank=64,
        effective_batch_size=eff_bs,
        learning_rate=2e-5,
        steps=steps,
        seed=seed,
    )
    wandb_run = (
        start_wandb_run(
            name=wandb_name,
            job_type="sft",
            config={
                "model_name": model.name,
                "answer_contract": "answer-tag-v2",
                "seed": seed,
                "steps": steps,
                "world_size": world_size,
                "per_device_train_batch_size": per_device_bs,
                "gradient_accumulation_steps": grad_accum,
                "effective_batch_size": eff_bs,
                "examples": len(dataset),
                "lora_rank": 64,
                "lora_alpha": 128,
                "learning_rate": 2e-5,
                "loss": "completion_only",
            }
            | extract_dataset_metadata(split),
            tags=["sft", "answer-tag-v2"],
            group="sft-warmup",
        )
        if distributed.is_main_process and count is None
        else None
    )
    wandb_cb = make_hf_wandb_callback(
        wandb_run,
        csv_path=(output / "metrics.csv") if distributed.is_main_process else None,
    )
    initial = output / "initial_adapter"
    if fsdp and distributed.is_main_process:
        policy.save_pretrained(str(initial))  # reference key set for export_fsdp_adapter

    class SFTTrainer(Trainer):
        def _save_checkpoint(self, model, trial):
            super()._save_checkpoint(model, trial)
            if fsdp:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    from inverser.capo_checkpoint import export_fsdp_adapter

                    checkpoint = output / f"checkpoint-{self.state.global_step}"
                    export_fsdp_adapter(checkpoint, initial, checkpoint / "full_adapter")
                self.accelerator.wait_for_everyone()

    trainer = SFTTrainer(
        model=policy,
        args=settings,
        train_dataset=dataset,
        data_collator=collate,
        processing_class=processor,
        callbacks=[wandb_cb] if wandb_cb is not None else None,
    )
    if distributed.is_main_process:
        write_json(
            output / "manifest.json",
            {
                "purpose": (
                    "multimodal_sft_optimizer_smoke" if count is not None else "multimodal_sft"
                ),
                "answer_contract": "answer-tag-v2",
                "dataset_manifest_sha256": file_sha256(split / "manifest.json"),
                "model_config_sha256": file_sha256(model / "config.json"),
                "producer": producer_hashes("capo_trainer.py", "capo_env.py", "vlm_suite.py"),
                "seed": seed,
                "steps": steps,
                "world_size": world_size,
                "per_device_train_batch_size": per_device_bs,
                "gradient_accumulation_steps": grad_accum,
                "effective_batch_size": eff_bs,
                "examples": len(dataset),
                "trainable_names": names,
                "lora_rank": 64,
                "lora_alpha": 128,
                "loss": "completion_only",
                "attention": "sdpa",
            },
        )
    distributed.wait_for_everyone()
    result = trainer.train()
    if fsdp:
        from accelerate.utils.fsdp_utils import save_fsdp_model

        final = output / "checkpoint"
        trainer.save_model(str(final))
        plugin = trainer.accelerator.state.fsdp_plugin
        save_fsdp_model(plugin, trainer.accelerator, trainer.model, str(final), adapter_only=True)
        distributed.wait_for_everyone()
        if distributed.is_main_process:
            from inverser.capo_checkpoint import export_fsdp_adapter

            export_fsdp_adapter(final, initial, final / "full_adapter")
    elif distributed.is_main_process:
        trainer.save_model(str(output / "checkpoint"))
    if distributed.is_main_process:
        report: dict[str, object] = {
            "status": "complete",
            "steps": result.global_step,
            "world_size": world_size,
            "effective_batch_size": eff_bs,
            "loss": result.training_loss,
            "metrics": result.metrics,
            "history": trainer.state.log_history,
        }
        write_json(output / "summary.json", report)
        finish_wandb_run(
            wandb_run,
            summary_metrics={
                "summary/train_loss": result.training_loss,
                "summary/steps": result.global_step,
                "summary/train_runtime_s": result.metrics.get("train_runtime"),
                "summary/samples_per_second": result.metrics.get("train_samples_per_second"),
            },
            artifact_name=wandb_name,
            artifact_type="sft-metadata",
            artifact_files=[output / "manifest.json", output / "summary.json"],
        )
        return report
    return {}
