"""Fault-tolerant Weights & Biases monitoring and cloud archival for SFT, CAPO, and VLM evals."""

import csv
import json
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inverser.runtime import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)
DEFAULT_ENTITY = "frankdarkluo"
DEFAULT_PROJECT = "inverser"

_STEP_METRIC_MAP = {
    "loss": "train/loss",
    "policy_loss": "train/policy_loss",
    "grad_norm": "train/grad_norm",
    "learning_rate": "train/learning_rate",
    "epoch": "train/epoch",
    "step_time": "train/step_time_s",
    "num_tokens": "train/num_tokens",
    "reward": "reward/mean",
    "rewards/rewards/mean": "reward/mean",
    "rewards/TrainingEpisode/mean": "reward/t5_episode_mean",
    "reward_std": "reward/std",
    "rewards/rewards/std": "reward/std",
    "rewards/TrainingEpisode/std": "reward/t5_episode_std",
    "frac_reward_zero_std": "reward/frac_zero_std",
    "reward/valid_format_rate": "reward/valid_format_rate",
    "reward/t4_goal_hit_rate": "reward/t4_goal_hit_rate",
    "reward/t4_mean_loss_radii": "reward/t4_mean_loss_radii",
    "entropy": "policy/entropy",
    "kl": "policy/kl",
    "tools/call_frequency": "tools/call_frequency",
    "tools/failure_frequency": "tools/failure_frequency",
    "sampling/importance_sampling_ratio/mean": "sampling/importance_ratio_mean",
    "sampling/importance_sampling_ratio/min": "sampling/importance_ratio_min",
    "sampling/importance_sampling_ratio/max": "sampling/importance_ratio_max",
    "sampling/sampling_logp_difference/mean": "sampling/logp_diff_mean",
    "sampling/sampling_logp_difference/max": "sampling/logp_diff_max",
    "clip_ratio/region_mean": "clip/region_mean",
    "clip_ratio/low_mean": "clip/low_mean",
    "clip_ratio/high_mean": "clip/high_mean",
    "clip_ratio/high_max": "clip/high_max",
    "completions/mean_length": "completions/mean_length",
    "completions/min_length": "completions/min_length",
    "completions/max_length": "completions/max_length",
    "completions/mean_terminated_length": "completions/mean_terminated_length",
    "completions/clipped_ratio": "completions/clipped_ratio",
}

_TRAIN_METRIC_PREFIXES = (
    "train/*",
    "reward/*",
    "policy/*",
    "sampling/*",
    "clip/*",
    "completions/*",
    "tools/*",
    "curriculum/*",
)


def wandb_storage_dir(root: Path = PROJECT_ROOT) -> Path:
    directory = root / "artifacts" / "wandb"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_wandb_env(root: Path = PROJECT_ROOT) -> bool:
    """Load gitignored .env credentials if present; return True when active logging is enabled."""
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return False
    env_file = root / ".env"
    if env_file.is_file():
        for raw_line in env_file.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key.startswith("WANDB_") and value:
                os.environ.setdefault(key, value)
    os.environ.setdefault("WANDB_DIR", str(wandb_storage_dir(root)))
    os.environ.setdefault("WANDB_SILENT", "true")
    mode = os.environ.get("WANDB_MODE", "online").lower()
    return mode == "offline" or bool(os.environ.get("WANDB_API_KEY"))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-")


def format_sft_run_name(
    *,
    model_name: str,
    lora_rank: int,
    effective_batch_size: int,
    learning_rate: float,
    steps: int,
    seed: int,
) -> str:
    return (
        f"sft-{_slug(model_name)}-r{lora_rank}-bs{effective_batch_size}"
        f"-lr{learning_rate:g}-steps{steps}-seed{seed}"
    )


def format_capo_run_name(
    *,
    mode: str,
    task: str,
    steps: int,
    seed: int,
    stage_steps: Sequence[int] | None = None,
    group_size: int = 8,
    kl_beta: float = 0.0,
    learning_rate: float = 5e-6,
) -> str:
    stage_part = (
        f"s{'_'.join(str(x) for x in stage_steps)}" if stage_steps else f"steps{steps}"
    )
    return (
        f"capo-{_slug(mode)}-{_slug(task)}-{stage_part}"
        f"-g{group_size}-kl{kl_beta:g}-lr{learning_rate:g}-steps{steps}-seed{seed}"
    )


def extract_dataset_metadata(split: Path | None) -> dict[str, Any]:
    if split is None or not (split / "manifest.json").is_file():
        return {}
    manifest = json.loads((split / "manifest.json").read_text())
    meta: dict[str, Any] = {
        "dataset/split": manifest.get("split"),
        "dataset/schema_version": manifest.get("schema_version"),
    }
    for key in (
        "standard_scene_count",
        "anchor_count",
        "twin_count",
        "independent_layout_groups",
        "ood_episode_count",
        "witness_replays_verified",
    ):
        if key in manifest:
            meta[f"dataset/{key}"] = manifest[key]
    for fam, count in manifest.get("family_counts", {}).items():
        meta[f"dataset/family_{_slug(fam)}"] = count
    fmt = manifest.get("format", {})
    for key in ("tolerance_radii", "minimum_twin_shift_radii"):
        if key in fmt:
            meta[f"dataset/{key}"] = fmt[key]
    return meta


def normalize_step_metrics(
    entry: dict[str, Any],
    *,
    step: int | None = None,
    schedule: Sequence[Sequence[Any]] | None = None,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for raw_key, target_key in _STEP_METRIC_MAP.items():
        if raw_key not in entry or entry[raw_key] is None:
            continue
        try:
            value = float(entry[raw_key])
        except (TypeError, ValueError):
            continue
        if value == value:  # finite / non-NaN check
            metrics[target_key] = value
    idx = (step - 1) if step is not None else None
    if step is not None:
        metrics["train/global_step"] = step
    if schedule and idx is not None and 0 <= idx < len(schedule):
        stage, task_name = int(schedule[idx][0]), str(schedule[idx][1])
        metrics["curriculum/stage"] = stage
        metrics["curriculum/is_forward"] = int(task_name == "forward")
        metrics["curriculum/is_T4"] = int(task_name == "T4")
        metrics["curriculum/is_T5"] = int(task_name == "T5")
    return metrics


def _configure_step_axes(run: Any, *, step_metric: str, prefixes: Sequence[str]) -> None:
    try:
        run.define_metric(step_metric)
        for prefix in prefixes:
            run.define_metric(prefix, step_metric=step_metric)
    except Exception:
        LOGGER.warning("W&B define_metric failed", exc_info=True)


def append_metrics_csv(csv_path: Path, row: dict[str, float | int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "train/global_step",
        "train/loss",
        "train/loss_ema",
        "train/grad_norm",
        "train/learning_rate",
        "reward/mean",
        "reward/mean_ema",
        "reward/std",
        "reward/frac_zero_std",
        "reward/valid_format_rate",
        "policy/entropy",
        "completions/mean_length",
        "completions/clipped_ratio",
        "curriculum/stage",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def start_wandb_run(
    *,
    name: str,
    job_type: str,
    config: dict[str, Any],
    tags: Sequence[str] = (),
    group: str | None = None,
    root: Path = PROJECT_ROOT,
) -> Any | None:
    if not configure_wandb_env(root):
        return None
    try:
        import wandb

        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY),
            project=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
            name=name,
            job_type=job_type,
            group=group,
            tags=list(tags),
            config=config,
            dir=str(wandb_storage_dir(root)),
            reinit="create_new",
        )
        if job_type in ("sft", "capo-grpo"):
            _configure_step_axes(
                run, step_metric="train/global_step", prefixes=_TRAIN_METRIC_PREFIXES
            )
        return run
    except Exception:
        LOGGER.warning("W&B initialization failed; continuing local run unaffected", exc_info=True)
        return None


def make_hf_wandb_callback(
    run: Any | None,
    *,
    schedule: Sequence[Sequence[Any]] | None = None,
    csv_path: Path | None = None,
) -> Any | None:
    if run is None and csv_path is None:
        return None
    active_run = run
    from transformers import TrainerCallback

    class _WandbStepCallback(TrainerCallback):
        def __init__(self) -> None:
            self.loss_ema: float | None = None
            self.reward_ema: float | None = None

        def on_log(
            self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any
        ) -> None:
            if not logs or not state.is_world_process_zero:
                return
            step = int(state.global_step)
            payload = normalize_step_metrics(logs, step=step, schedule=schedule)
            if "train/loss" in payload:
                val = float(payload["train/loss"])
                self.loss_ema = val if self.loss_ema is None else 0.9 * self.loss_ema + 0.1 * val
                payload["train/loss_ema"] = self.loss_ema
            if "reward/mean" in payload:
                rval = float(payload["reward/mean"])
                self.reward_ema = (
                    rval if self.reward_ema is None else 0.9 * self.reward_ema + 0.1 * rval
                )
                payload["reward/mean_ema"] = self.reward_ema
            if csv_path is not None and payload:
                try:
                    append_metrics_csv(csv_path, payload)
                except Exception:
                    LOGGER.warning(
                        "Local metrics.csv append failed at step %d", step, exc_info=True
                    )
            if active_run is not None and len(payload) > 1:
                try:
                    active_run.log(payload, step=step)
                except Exception:
                    LOGGER.warning("W&B step log failed at step %d", step, exc_info=True)

    return _WandbStepCallback()


def finish_wandb_run(
    run: Any | None,
    *,
    summary_metrics: dict[str, Any] | None = None,
    artifact_name: str | None = None,
    artifact_type: str = "experiment-results",
    artifact_files: Sequence[Path] = (),
) -> str | None:
    if run is None:
        return None
    url: str | None = getattr(run, "url", None)
    try:
        # Store static run-end totals in config (not unlogged summary scalars, which create
        # empty `_step` line-chart boxes in the W&B Charts UI).
        if summary_metrics:
            clean_meta = {
                key: value
                for key, value in summary_metrics.items()
                if isinstance(value, (int, float, str, bool))
            }
            if clean_meta:
                run.config.update({"run_totals": clean_meta}, allow_val_change=True)
        existing = [p for p in artifact_files if p.is_file()]
        if artifact_name and existing:
            import wandb

            artifact = wandb.Artifact(name=_slug(artifact_name), type=artifact_type)
            for file_path in existing:
                artifact.add_file(str(file_path), name=file_path.name)
            run.log_artifact(artifact)
    except Exception:
        LOGGER.warning("W&B config/artifact upload encountered an error", exc_info=True)
    finally:
        try:
            run.finish()
        except Exception:
            LOGGER.warning("W&B finish encountered an error", exc_info=True)
    return url


def _flatten_eval_analysis(analysis: dict[str, Any]) -> dict[str, float | int]:
    flat: dict[str, float | int] = {}
    for arm, content in analysis.get("arms", {}).items():
        fwd_all = content.get("forward", {}).get("all", {})
        if "valid" in fwd_all and "n" in fwd_all and fwd_all["n"]:
            flat[f"eval/{arm}/forward_valid_rate"] = fwd_all["valid"] / fwd_all["n"]
        for metric, val in fwd_all.get("mean", {}).items():
            if isinstance(val, (int, float)):
                flat[f"eval/{arm}/{metric}"] = float(val)
        if isinstance(content.get("T4", {}).get("rate"), (int, float)):
            flat[f"eval/{arm}/T4_rate"] = float(content["T4"]["rate"])
        t5_6 = content.get("T5_by_attempt", {}).get("6", {}).get("rate")
        if isinstance(t5_6, (int, float)):
            flat[f"eval/{arm}/T5_by_6_rate"] = float(t5_6)
    for pair_name, metrics in analysis.get("paired", {}).items():
        for metric, stats in metrics.items():
            diff = stats.get("mean_difference_a_minus_b")
            if isinstance(diff, (int, float)):
                flat[f"paired/{pair_name}/{metric}_diff"] = float(diff)
    return flat


def _log_analysis_charts(run: Any, run_dir: Path, analysis: dict[str, Any]) -> None:
    import wandb

    _configure_step_axes(run, step_metric="tolerance/radius_r", prefixes=("tolerance/*",))
    arms = analysis.get("arms", {})
    for step_idx, radius in enumerate((1, 2, 3, 5), start=1):
        point: dict[str, float | int] = {"tolerance/radius_r": radius}
        for arm, content in arms.items():
            mean_scores = content.get("forward", {}).get("all", {}).get("mean", {})
            val = mean_scores.get(f"T3_{radius}r")
            if isinstance(val, (int, float)):
                point[f"tolerance/{arm}_T3_accuracy"] = float(val)
        if len(point) > 1:
            run.log(point, step=step_idx)

    bar_metrics = ("T1", "T1_per_ball", "T2_cushion_set", "T3_1r", "T3_3r", "S_total", "S_total_3r")
    for metric in bar_metrics:
        rows = []
        for arm, content in sorted(arms.items()):
            val = content.get("forward", {}).get("all", {}).get("mean", {}).get(metric)
            if isinstance(val, (int, float)):
                rows.append([arm, round(100.0 * float(val), 2)])
        if rows:
            tbl = wandb.Table(data=rows, columns=["arm", "score_pct"])
            run.log(
                {
                    f"charts/{metric}_by_arm": wandb.plot.bar(
                        tbl, "arm", "score_pct", title=f"{metric} (%) by Arm"
                    )
                }
            )

    for pair_name, metrics in analysis.get("paired", {}).items():
        gain_rows = []
        ci_rows = []
        for metric, stats in sorted(metrics.items()):
            diff = stats.get("mean_difference_a_minus_b")
            ci = stats.get("confidence_interval") or [None, None]
            if isinstance(diff, (int, float)):
                # Note: a_minus_b < 0 means arm B beat arm A; plot B - A gain in percentage points.
                b_minus_a_pct = round(-100.0 * float(diff), 2)
                gain_rows.append([metric, b_minus_a_pct])
                ci_rows.append(
                    [
                        metric,
                        b_minus_a_pct,
                        round(-100.0 * float(ci[1]), 2) if ci[1] is not None else None,
                        round(-100.0 * float(ci[0]), 2) if ci[0] is not None else None,
                    ]
                )
        if gain_rows:
            gain_tbl = wandb.Table(data=gain_rows, columns=["metric", "gain_pct_pts"])
            run.log(
                {
                    f"charts/gain_{pair_name}": wandb.plot.bar(
                        gain_tbl,
                        "metric",
                        "gain_pct_pts",
                        title=f"Paired Gain (B - A, % pts): {pair_name}",
                    ),
                    f"tables/paired_ci_{pair_name}": wandb.Table(
                        data=ci_rows,
                        columns=["metric", "gain_B_minus_A_pct", "ci95_low_pct", "ci95_high_pct"],
                    ),
                }
            )

    if (run_dir / "table.csv").is_file():
        with (run_dir / "table.csv").open() as stream:
            csv_rows = list(csv.reader(stream))
        if len(csv_rows) > 1:
            cols: list[str | int] = list(csv_rows[0])
            run.log({"tables/summary_table": wandb.Table(columns=cols, data=csv_rows[1:])})


def _log_vlm_eval_job_curves(run: Any, run_dir: Path) -> None:
    jobs_dir = run_dir / "jobs"
    if not jobs_dir.is_dir():
        return
    _configure_step_axes(run, step_metric="eval/scene_index", prefixes=("eval/*",))
    grouped: dict[str, dict[str, Any]] = {}
    for path in sorted(jobs_dir.glob("*.json")):
        payload = json.loads(path.read_text()).get("payload", {})
        sid = payload.get("scene_id")
        task = payload.get("task")
        if sid and task:
            grouped.setdefault(str(sid), {})[str(task)] = payload

    n_fwd = n_t4 = n_t5 = 0
    sum_t1 = sum_t2 = sum_t3_1r = sum_t3_3r = sum_s = sum_s3r = 0.0
    t4_valid = 0
    t5_within_5r = 0
    t5_best_sum = 0.0
    t5_valid_scenes = 0

    for idx, (_, tasks) in enumerate(sorted(grouped.items()), start=1):
        point: dict[str, float | int] = {"eval/scene_index": idx}
        if "forward" in tasks:
            n_fwd += 1
            sc = tasks["forward"].get("scores", {})
            sum_t1 += float(sc.get("T1", 0.0))
            sum_t2 += float(sc.get("T2_cushion_set", sc.get("T2", 0.0)))
            sum_t3_1r += float(sc.get("T3_1r", sc.get("T3", 0.0)))
            sum_t3_3r += float(sc.get("T3_3r", 0.0))
            sum_s += float(sc.get("S_total", 0.0))
            sum_s3r += float(sc.get("S_total_3r", 0.0))
            point["eval/running_T1"] = sum_t1 / n_fwd
            point["eval/running_T2_set"] = sum_t2 / n_fwd
            point["eval/running_T3_1r"] = sum_t3_1r / n_fwd
            point["eval/running_T3_3r"] = sum_t3_3r / n_fwd
            point["eval/running_S_total"] = sum_s / n_fwd
            point["eval/running_S_total_3r"] = sum_s3r / n_fwd
        if "T4" in tasks:
            n_t4 += 1
            if tasks["T4"].get("loss_radii") is not None:
                t4_valid += 1
            point["eval/running_T4_valid_rate"] = t4_valid / n_t4
        if "T5" in tasks:
            n_t5 += 1
            losses = [
                float(a["loss_radii"])
                for a in tasks["T5"].get("attempts", [])
                if a.get("loss_radii") is not None
            ]
            if losses:
                best = min(losses)
                t5_valid_scenes += 1
                t5_best_sum += best
                if best <= 5.0:
                    t5_within_5r += 1
                point["eval/running_T5_mean_best_loss_r"] = t5_best_sum / t5_valid_scenes
            point["eval/running_T5_within_5r_rate"] = t5_within_5r / n_t5
        run.log(point, step=idx)


def sync_run_to_wandb(
    run_dir: Path,
    *,
    split_dir: Path | None = None,
    run_name: str | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Upload a completed SFT, CAPO/GRPO, VLM eval, or paired analysis directory to W&B."""
    run_dir = run_dir.resolve()
    ds_meta = extract_dataset_metadata(split_dir)

    if (run_dir / "analysis.json").is_file():
        analysis = json.loads((run_dir / "analysis.json").read_text())
        name = run_name or f"capo-eval-{_slug(run_dir.name)}"
        config = {
            "analysis_dir": run_dir.name,
            "sources": list(analysis.get("sources", {})),
        } | ds_meta
        run = start_wandb_run(
            name=name,
            job_type="eval-analysis",
            config=config,
            tags=["eval", "paired-analysis", "capo"],
            root=root,
        )
        summary = _flatten_eval_analysis(analysis)
        if run is not None:
            _log_analysis_charts(run, run_dir, analysis)
        files = [
            p
            for p in sorted(run_dir.iterdir())
            if p.is_file() and p.suffix in (".json", ".csv", ".svg")
        ]
        url = finish_wandb_run(
            run,
            summary_metrics=summary,
            artifact_name=name,
            artifact_type="capo-analysis",
            artifact_files=files,
        )
        return {"status": "synced" if url else "skipped", "run_name": name, "url": url}

    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    purpose = manifest.get("purpose", "")

    if purpose.startswith("multimodal_sft"):
        lr = 2e-5
        name = run_name or format_sft_run_name(
            model_name="qwen3.5-9b",
            lora_rank=int(manifest.get("lora_rank", 64)),
            effective_batch_size=int(manifest.get("effective_batch_size", 32)),
            learning_rate=lr,
            steps=int(manifest["steps"]),
            seed=int(manifest["seed"]),
        )
        config = (
            {k: v for k, v in manifest.items() if k != "trainable_names"}
            | {"trainable_tensor_count": len(manifest.get("trainable_names", []))}
            | ds_meta
        )
        run = start_wandb_run(
            name=name,
            job_type="sft",
            config=config,
            tags=["sft", str(manifest.get("answer_contract", "tag"))],
            group="sft-warmup",
            root=root,
        )
        step_counter = 0
        loss_ema: float | None = None
        for item in summary.get("history", []):
            if "loss" not in item:
                continue
            step_counter = int(item.get("step", step_counter + 1))
            payload = normalize_step_metrics(item, step=step_counter)
            if "train/loss" in payload:
                v = float(payload["train/loss"])
                loss_ema = v if loss_ema is None else 0.9 * loss_ema + 0.1 * v
                payload["train/loss_ema"] = loss_ema
            if run is not None:
                run.log(payload, step=step_counter)
        url = finish_wandb_run(
            run,
            summary_metrics={
                "train_loss": summary.get("loss"),
                "steps": summary.get("steps"),
                "train_runtime_s": summary.get("metrics", {}).get("train_runtime"),
                "samples_per_second": summary.get("metrics", {}).get("train_samples_per_second"),
            },
            artifact_name=name,
            artifact_type="sft-metadata",
            artifact_files=[run_dir / "manifest.json", run_dir / "summary.json"],
        )
        return {"status": "synced" if url else "skipped", "run_name": name, "url": url}

    if "reward" in manifest and "requested_steps" in manifest:
        schedule = manifest.get("curriculum_optimizer_schedule") or []
        stage_counts = [sum(1 for s, _ in schedule if s == stage_id) for stage_id in (1, 2, 3)]
        stage_steps = stage_counts if all(x > 0 for x in stage_counts) else None
        name = run_name or format_capo_run_name(
            mode=str(manifest["reward"]),
            task=str(manifest["task"]),
            steps=int(manifest["requested_steps"]),
            seed=int(manifest["seed"]),
            stage_steps=stage_steps,
            group_size=int(manifest.get("group_size", 8)),
            kl_beta=0.0,
            learning_rate=5e-6,
        )
        config = (
            {k: v for k, v in manifest.items() if k != "curriculum_optimizer_schedule"}
            | {"kl_beta": 0.0, "learning_rate": 5e-6, "stage_steps": stage_steps}
            | ds_meta
        )
        run = start_wandb_run(
            name=name,
            job_type="capo-grpo",
            config=config,
            tags=["capo", str(manifest["reward"]), str(manifest["task"])],
            group=f"stage1-seed{manifest['seed']}",
            root=root,
        )
        step_counter = 0
        loss_ema = None
        reward_ema: float | None = None
        for item in summary.get("history", []):
            if "grad_norm" not in item and "loss" not in item:
                continue
            step_counter = int(item.get("step", step_counter + 1))
            payload = normalize_step_metrics(item, step=step_counter, schedule=schedule)
            if "train/loss" in payload:
                lv = float(payload["train/loss"])
                loss_ema = lv if loss_ema is None else 0.9 * loss_ema + 0.1 * lv
                payload["train/loss_ema"] = loss_ema
            if "reward/mean" in payload:
                rv = float(payload["reward/mean"])
                reward_ema = rv if reward_ema is None else 0.9 * reward_ema + 0.1 * rv
                payload["reward/mean_ema"] = reward_ema
            if run is not None:
                run.log(payload, step=step_counter)
        final_summary: dict[str, Any] = {
            "steps": summary.get("steps"),
            "nonzero_gradient_steps": summary.get("nonzero_gradient_steps"),
            "simulator_rollouts": summary.get("simulator_rollouts"),
        }
        for k, v in summary.get("reward_receipts", {}).items():
            final_summary[f"receipt_{k}"] = v
        url = finish_wandb_run(
            run,
            summary_metrics=final_summary,
            artifact_name=name,
            artifact_type="capo-grpo-metadata",
            artifact_files=[
                run_dir / "manifest.json",
                run_dir / "summary.json",
                run_dir / "metrics.csv",
                run_dir / "checkpoint/full_adapter/manifest.json",
            ],
        )
        return {"status": "synced" if url else "skipped", "run_name": name, "url": url}

    name = run_name or f"capo-eval-{_slug(run_dir.name)}"
    usage = summary.get("usage", {})
    by_reply = summary.get("usage_by_reply", {})
    run = start_wandb_run(
        name=name,
        job_type="vlm-eval",
        config=manifest | ds_meta,
        tags=["vlm-eval", str(manifest.get("backend", {}).get("alias", "vlm"))],
        root=root,
    )
    if run is not None:
        _log_vlm_eval_job_curves(run, run_dir)
    url = finish_wandb_run(
        run,
        summary_metrics={
            "jobs": summary.get("jobs"),
            "model_requests": summary.get("counts", {}).get("model_requests"),
            "physics_attempts": summary.get("counts", {}).get("physics_attempts"),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "finish_length": by_reply.get("finish_reasons", {}).get("length", 0),
            "finish_stop": by_reply.get("finish_reasons", {}).get("stop", 0),
        },
        artifact_name=name,
        artifact_type="vlm-eval-metadata",
        artifact_files=[run_dir / "manifest.json", run_dir / "summary.json"],
    )
    return {"status": "synced" if url else "skipped", "run_name": name, "url": url}
