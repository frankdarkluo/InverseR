import json

from inverser.wandb_tracker import (
    configure_wandb_env,
    extract_dataset_metadata,
    format_capo_run_name,
    format_sft_run_name,
    normalize_step_metrics,
    sync_run_to_wandb,
)


def test_run_naming_and_step_metric_normalization():
    sft_name = format_sft_run_name(
        model_name="Qwen3.5-9B",
        lora_rank=64,
        effective_batch_size=32,
        learning_rate=2e-5,
        steps=400,
        seed=20261119,
    )
    assert sft_name == "sft-qwen3.5-9b-r64-bs32-lr2e-05-steps400-seed20261119"

    capo_name = format_capo_run_name(
        mode="capo",
        task="joint",
        steps=200,
        seed=20261119,
        stage_steps=(40, 60, 100),
        group_size=8,
        kl_beta=0.0,
        learning_rate=5e-6,
    )
    assert capo_name == "capo-capo-joint-s40_60_100-g8-kl0-lr5e-06-steps200-seed20261119"

    schedule = [(1, "forward"), (2, "T4"), (3, "T5")]
    normalized = normalize_step_metrics(
        {
            "loss": "0.0046",
            "grad_norm": "2.67",
            "rewards/rewards/mean": "0.1432",
            "rewards/rewards/std": "0.4088",
            "frac_reward_zero_std": "0",
            "entropy": "0.2418",
            "completions/clipped_ratio": "0",
        },
        step=3,
        schedule=schedule,
    )
    assert normalized["reward/mean"] == 0.1432
    assert normalized["reward/std"] == 0.4088
    assert normalized["reward/frac_zero_std"] == 0.0
    assert normalized["curriculum/stage"] == 3
    assert normalized["curriculum/is_T5"] == 1 and normalized["curriculum/is_forward"] == 0


def test_dataset_metadata_and_offline_sync(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    split = tmp_path / "train"
    split.mkdir()
    (split / "manifest.json").write_text(
        json.dumps(
            {
                "split": "train",
                "schema_version": "inverser-vlm-v1",
                "standard_scene_count": 24000,
                "anchor_count": 12000,
                "twin_count": 12000,
                "independent_layout_groups": 12000,
                "ood_episode_count": 0,
                "family_counts": {"standard/free_roll": 3000, "standard/side": 3000},
                "format": {"tolerance_radii": 1.0, "minimum_twin_shift_radii": 2.0},
            }
        )
    )
    meta = extract_dataset_metadata(split)
    assert meta["dataset/standard_scene_count"] == 24000
    assert meta["dataset/family_standard-side"] == 3000

    assert configure_wandb_env(tmp_path) is True
    run_dir = tmp_path / "sft-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "purpose": "multimodal_sft",
                "lora_rank": 64,
                "effective_batch_size": 32,
                "steps": 2,
                "seed": 7,
                "trainable_names": ["a", "b"],
            }
        )
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "loss": 0.25,
                "steps": 2,
                "metrics": {"train_runtime": 12.0, "train_samples_per_second": 5.0},
                "history": [{"step": 1, "loss": 0.3}, {"step": 2, "loss": 0.2}],
            }
        )
    )
    receipt = sync_run_to_wandb(run_dir, split_dir=split, root=tmp_path)
    assert receipt["run_name"].startswith("sft-")
    assert (tmp_path / "artifacts/wandb").is_dir()
