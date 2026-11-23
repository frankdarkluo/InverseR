import argparse
import json
import logging
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="InverseR benchmark, CAPO training, and eval CLI")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate-pilot", help="Generate at most 5000 pre-G1 tasks")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--count", type=int, default=2000)
    generate.add_argument("--seed", type=int, default=20261010)
    audit = commands.add_parser("audit-pilot", help="Validate schemas, exports and witness replay")
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--replay-count", type=int, default=20)
    preview = commands.add_parser("preview", help="Render public endpoint-only task pairs")
    preview.add_argument("--input", type=Path, required=True)
    preview.add_argument("--output", type=Path, required=True)
    study_dry = commands.add_parser(
        "study-dry-run", help="Validate paired study plumbing with a mock, not model inference"
    )
    study_dry.add_argument("--input", type=Path, required=True)
    study_dry.add_argument("--output", type=Path, required=True)
    training_set = commands.add_parser(
        "prepare-training-set",
        help="Fresh post-training layouts disjoint from development and formal-test groups (CPU)",
    )
    training_set.add_argument("--output", type=Path, required=True)
    training_set.add_argument("--seed", type=int, default=20261103)
    training_set.add_argument("--per-family", type=int, default=1000)
    training_set.add_argument("--max-index", type=int, default=200_000)
    training_set.add_argument("--exclude-records", type=Path, nargs="*", default=[])
    training_set.add_argument("--exclude-formal-panel", type=Path, nargs="*", default=[])
    verify_external = commands.add_parser(
        "verify-external-suites", help="Verify pinned external assets without network/model calls"
    )
    verify_external.add_argument("--output", type=Path)
    vlm_suite = commands.add_parser(
        "prepare-vlm-suite", help="Generate and independently replay all three InverseR-VLM splits"
    )
    vlm_suite.add_argument("--output-root", type=Path, default=Path("data/research"))
    vlm_suite.add_argument(
        "--exclude-pilot", type=Path, default=Path("data/pilot-v0/privileged.jsonl")
    )
    vlm_suite.add_argument(
        "--exclude-formal", type=Path, default=Path("data/research/formal-test-v1")
    )
    vlm_suite.add_argument("--verify-only", action="store_true")
    vlm_eval = commands.add_parser(
        "run-vlm-eval", help="Ledgered local VLM evaluation on train or valid, never blind test"
    )
    vlm_eval.add_argument("--split", type=Path, required=True)
    vlm_eval.add_argument("--output", type=Path, required=True)
    vlm_eval_backend = vlm_eval.add_mutually_exclusive_group(required=True)
    vlm_eval_backend.add_argument("--local-vllm-url")
    vlm_eval_backend.add_argument(
        "--proxy",
        action="store_true",
        help="OpenAI-compatible INVERSE_PROXY_BASE_URL with configs/proxy_model_aliases.json",
    )
    vlm_eval.add_argument("--model", required=True)
    vlm_eval.add_argument("--revision", required=True)
    vlm_eval.add_argument("--tasks", choices=["forward", "T4", "T5"], nargs="+", required=True)
    vlm_eval.add_argument("--thinking", action="store_true")
    vlm_eval.add_argument("--thinking-token-budget", type=int)
    vlm_eval.add_argument("--max-output-tokens", type=int)
    vlm_eval.add_argument("--workers", type=int, default=1)
    vlm_eval.add_argument("--count", type=int)
    vlm_eval.add_argument("--max-total-tokens", type=int, required=True)
    vlm_eval.add_argument("--execute", action="store_true")
    vlm_eval.add_argument("--blind-protocol", type=Path)
    vlm_eval.add_argument("--arm")
    vlm_eval.add_argument("--seed", type=int, default=0)
    vlm_eval.add_argument(
        "--temperature", type=float, default=0.0, help="> 0 samples (calibrate-fapo-beta)"
    )
    freeze_blind = commands.add_parser(
        "freeze-vlm-blind-protocol", help="Pin verified dataset, checkpoint and selection hashes"
    )
    freeze_blind.add_argument("--test-manifest", type=Path, required=True)
    freeze_blind.add_argument("--valid-manifest", type=Path, required=True)
    freeze_blind.add_argument("--arms", type=Path, required=True)
    freeze_blind.add_argument("--output", type=Path, required=True)
    calibrate = commands.add_parser(
        "calibrate-fapo-beta", help="Pre-register the additive arm's beta_E from sampled T4 runs"
    )
    calibrate.add_argument("--split", type=Path, required=True)
    calibrate.add_argument("--run", type=Path, action="append", required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    selection = commands.add_parser(
        "select-capo-checkpoint", help="Pick a checkpoint from complete and matched valid runs"
    )
    selection.add_argument("--valid-manifest", type=Path, required=True)
    selection.add_argument(
        "--candidate",
        nargs=2,
        action="append",
        required=True,
        metavar=("CHECKPOINT_MANIFEST", "RUN"),
    )
    selection.add_argument("--output", type=Path, required=True)
    export_adapter = commands.add_parser(
        "export-capo-fsdp-adapter", help="Export all verified LoRA tensors from a collective save"
    )
    export_adapter.add_argument("--checkpoint", type=Path, required=True)
    export_adapter.add_argument("--initial-adapter", type=Path, required=True)
    export_adapter.add_argument("--output", type=Path, required=True)
    merge_command = commands.add_parser(
        "merge-capo-adapter", help="Merge a complete local LoRA adapter into a hashed bf16 model"
    )
    merge_command.add_argument("--model-path", type=Path, required=True)
    merge_command.add_argument("--adapter", type=Path, required=True)
    merge_command.add_argument("--output", type=Path, required=True)
    merge_command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    agentic = commands.add_parser(
        "run-agentic-baseline", help="Run a matched-budget CueTip or DeepPHY-WM adaptation"
    )
    agentic.add_argument("--split", type=Path, required=True)
    agentic.add_argument("--output", type=Path, required=True)
    agentic.add_argument("--local-vllm-url", required=True)
    agentic.add_argument("--model", required=True)
    agentic.add_argument("--revision", required=True)
    agentic.add_argument(
        "--method", choices=("cuetip", "wm_reflection", "self_verify"), required=True
    )
    agentic.add_argument("--thinking", action="store_true")
    agentic.add_argument("--thinking-token-budget", type=int)
    agentic.add_argument("--max-output-tokens", type=int)
    agentic.add_argument("--workers", type=int, default=1)
    agentic.add_argument("--count", type=int)
    agentic.add_argument("--max-total-tokens", type=int, required=True)
    agentic.add_argument("--execute", action="store_true")
    capo_sft = commands.add_parser(
        "train-capo-sft", help="Run the multimodal SFT baseline or two-step smoke on CUDA"
    )
    capo_sft.add_argument("--split", type=Path, required=True)
    capo_sft.add_argument("--output", type=Path, required=True)
    capo_sft.add_argument("--model-path", type=Path, required=True)
    capo_sft.add_argument("--steps", type=int, required=True)
    capo_sft.add_argument("--count", type=int)
    capo_sft.add_argument("--seed", type=int, default=42)
    capo_sft.add_argument("--save-every", type=int, help="also save the adapter every N steps")
    grpo = commands.add_parser(
        "train-capo-grpo", help="Multimodal GRPO arms or FSDP joint curriculum on CUDA"
    )
    grpo.add_argument("--split", type=Path, required=True)
    grpo.add_argument("--model-path", type=Path, required=True)
    grpo.add_argument("--adapter", type=Path, required=True)
    grpo.add_argument("--output", type=Path, required=True)
    grpo.add_argument("--mode", choices=("outcome", "additive", "capo"), required=True)
    grpo.add_argument("--task", choices=("forward", "T4", "joint"), required=True)
    grpo.add_argument("--beta-e", type=float, help="Additive arm only: calibrate-fapo-beta value")
    grpo.add_argument("--stage", type=int, choices=(1, 2, 3), default=3)
    grpo.add_argument("--steps", type=int, required=True)
    grpo.add_argument("--stage-steps", nargs=3, type=int)
    grpo.add_argument("--seed", type=int, default=42)
    grpo.add_argument("--count", type=int)
    grpo.add_argument("--vllm-server-url")
    grpo.add_argument("--fsdp", action="store_true", help="Use two torchrun FSDP2 trainer ranks")
    grpo.add_argument("--resume-from", type=Path, help="Recover an exact completed FSDP stage")
    wm = commands.add_parser(
        "train-learned-wm", help="Fit a GPU world-model baseline and score CEM on valid"
    )
    wm.add_argument("--method", choices=("dino", "state"), required=True)
    wm.add_argument("--train", type=Path, default=Path("data/research/inverser/sft"))
    wm.add_argument("--valid", type=Path, default=Path("data/research/inverser/valid"))
    wm.add_argument("--dino-path", type=Path, default=Path("models/dinov2-small"))
    wm.add_argument("--output", type=Path, required=True)
    wm.add_argument("--epochs", type=int, required=True)
    wm.add_argument("--seed", type=int, default=42)
    wm.add_argument("--count", type=int)
    wm.add_argument("--valid-count", type=int)
    oracle = commands.add_parser(
        "run-oracle-cem", help="CPU upper bound: ground-truth state + Pooltool cost under CEM"
    )
    oracle.add_argument("--valid", type=Path, default=Path("data/research/inverser/valid"))
    oracle.add_argument("--output", type=Path, required=True)
    oracle.add_argument("--seed", type=int, default=42)
    oracle.add_argument("--workers", type=int, default=32)
    oracle.add_argument("--count", type=int, help="Scene prefix, e.g. 2468 for the v3 valid panel")
    oracle.add_argument(
        "--refine-evals", type=int, default=1500, help="Nelder-Mead polish; 0 = matched-budget CEM"
    )
    public_eval = commands.add_parser(
        "run-public-eval", help="Ledgered local PhysBench or DeepPHY puzzle evaluation"
    )
    public_eval.add_argument(
        "--suite",
        choices=("physbench", "iphyre", "phyre", "pooltool", "MMMU", "RealWorldQA"),
        default="physbench",
    )
    public_eval.add_argument("--source", type=Path, default=Path("data/external/PhysBench"))
    public_eval.add_argument("--output", type=Path, required=True)
    public_eval.add_argument("--local-vllm-url", required=True)
    public_eval.add_argument("--model", required=True)
    public_eval.add_argument("--revision", required=True)
    public_eval.add_argument("--split", choices=["val", "test"], default="val")
    public_eval.add_argument("--collision-only", action="store_true")
    public_eval.add_argument("--thinking", action="store_true")
    public_eval.add_argument("--count", type=int)
    public_eval.add_argument("--max-total-tokens", type=int, required=True)
    public_eval.add_argument("--execute", action="store_true")
    capo_analysis = commands.add_parser(
        "analyze-capo-runs", help="Rescore completed CAPO evaluation envelopes without any calls"
    )
    capo_analysis.add_argument(
        "--run", nargs=2, action="append", required=True, metavar=("NAME", "DIRECTORY")
    )
    capo_analysis.add_argument("--output", type=Path, required=True)
    wandb_sync = commands.add_parser(
        "sync-wandb", help="Upload completed SFT, CAPO/GRPO, or evaluation artifacts to W&B"
    )
    wandb_sync.add_argument("--run-dir", type=Path, required=True)
    wandb_sync.add_argument("--split", type=Path)
    wandb_sync.add_argument("--name")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command in ("generate-pilot", "audit-pilot"):
        from inverser.runtime import configure_physics_runtime

        configure_physics_runtime()
    if args.command == "generate-pilot":
        from inverser.pilot import generate_pilot

        result = generate_pilot(args.output, count=args.count, seed=args.seed)
    elif args.command == "audit-pilot":
        from inverser.pilot import audit_pilot

        result = audit_pilot(args.output, replay_count=args.replay_count)
    elif args.command == "study-dry-run":
        from inverser.evaluation import study_dry_run

        result = study_dry_run(args.input, args.output)
    elif args.command == "prepare-training-set":
        from inverser.training_env import TrainingSetConfig, prepare_training_set

        result = prepare_training_set(
            args.output,
            TrainingSetConfig(seed=args.seed, per_family=args.per_family, max_index=args.max_index),
            exclude_records=tuple(args.exclude_records),
            exclude_panels=tuple(args.exclude_formal_panel),
        )
    elif args.command == "analyze-capo-runs":
        from inverser.paper_analysis import analyze_runs

        runs = {name: Path(path) for name, path in args.run}
        if len(runs) != len(args.run):
            parser.error("analysis run names must be distinct")
        result = analyze_runs(runs, args.output)
    elif args.command == "sync-wandb":
        from inverser.wandb_tracker import sync_run_to_wandb

        result = sync_run_to_wandb(args.run_dir, split_dir=args.split, run_name=args.name)
    elif args.command == "run-public-eval":
        from inverser.public_eval import run_physbench
        from inverser.study_calls import local_vllm_backends

        backend = local_vllm_backends(
            args.local_vllm_url,
            (args.model,),
            model_revision=args.revision,
            reasoning_effort="high" if args.thinking else "none",
            max_output_tokens=2048,
            request_token_reservation=16384,
            timeout_s=180,
            enable_thinking=args.thinking,
            thinking_token_budget=1024 if args.thinking else None,
        )[0]
        if args.suite in ("MMMU", "RealWorldQA"):
            from inverser.retention_eval import run_retention

            if args.collision_only or args.split != "val":
                parser.error("--split and --collision-only only apply to PhysBench")
            result = run_retention(
                args.suite,
                args.output,
                backend,
                count=args.count,
                budget_tokens=args.max_total_tokens,
                execute=args.execute,
            )
        elif args.suite != "physbench":
            from inverser.iphyre_eval import run_iphyre

            if args.collision_only or args.split != "val":
                parser.error("--split and --collision-only only apply to PhysBench")
            result = run_iphyre(
                args.output,
                backend,
                count=args.count,
                budget_tokens=args.max_total_tokens,
                execute=args.execute,
                suite=args.suite,
            )
        else:
            result = run_physbench(
                args.source,
                args.output,
                backend,
                split=args.split,
                collision_only=args.collision_only,
                count=args.count,
                budget_tokens=args.max_total_tokens,
                execute=args.execute,
            )
    elif args.command == "train-learned-wm":
        from inverser.learned_wm import train_world_model

        result = train_world_model(
            args.method,
            args.train,
            args.valid,
            args.output,
            args.dino_path,
            epochs=args.epochs,
            seed=args.seed,
            limit=args.count,
            valid_limit=args.valid_count,
        )
    elif args.command == "run-oracle-cem":
        from inverser.learned_wm import run_oracle_cem

        result = run_oracle_cem(
            args.valid,
            args.output,
            seed=args.seed,
            workers=args.workers,
            refine_evals=args.refine_evals,
            count=args.count,
        )
    elif args.command == "train-capo-grpo":
        from inverser.capo_grpo import run_grpo

        result = run_grpo(
            args.split,
            args.model_path,
            args.adapter,
            args.output,
            mode=args.mode,
            task=args.task,
            stage=args.stage,
            steps=args.steps,
            seed=args.seed,
            count=args.count,
            server_url=args.vllm_server_url,
            stage_steps=tuple(args.stage_steps) if args.stage_steps else None,
            fsdp=args.fsdp,
            resume_from=args.resume_from,
            beta_e=args.beta_e,
        )
    elif args.command == "train-capo-sft":
        from inverser.capo_trainer import run_sft

        result = run_sft(
            args.split,
            args.model_path,
            args.output,
            count=args.count,
            steps=args.steps,
            seed=args.seed,
            save_every=args.save_every,
        )
    elif args.command == "run-agentic-baseline":
        from inverser.agentic_baselines import run_agentic_baseline
        from inverser.study_calls import local_vllm_backends

        backend = local_vllm_backends(
            args.local_vllm_url,
            (args.model,),
            model_revision=args.revision,
            reasoning_effort="high" if args.thinking else "none",
            max_output_tokens=args.max_output_tokens or (2048 if args.thinking else 1024),
            request_token_reservation=16384,
            timeout_s=150,
            enable_thinking=args.thinking,
            temperature=0.65 if args.method == "self_verify" else 0,
            thinking_token_budget=(
                args.thinking_token_budget
                if args.thinking_token_budget is not None
                else (1024 if args.thinking else None)
            ),
        )[0]
        result = run_agentic_baseline(
            args.split,
            args.output,
            backend,
            method=args.method,
            count=args.count,
            budget_tokens=args.max_total_tokens,
            execute=args.execute,
            workers=args.workers,
        )
    elif args.command == "merge-capo-adapter":
        from inverser.capo_checkpoint import merge_adapter

        result = merge_adapter(args.model_path, args.adapter, args.output, device=args.device)
    elif args.command == "export-capo-fsdp-adapter":
        from inverser.capo_checkpoint import export_fsdp_adapter

        result = export_fsdp_adapter(args.checkpoint, args.initial_adapter, args.output)
    elif args.command == "calibrate-fapo-beta":
        from inverser.capo_selection import calibrate_beta_e

        result = calibrate_beta_e(args.output, args.split, args.run)
    elif args.command == "select-capo-checkpoint":
        from inverser.capo_selection import select_checkpoint

        candidates = {Path(checkpoint): Path(run) for checkpoint, run in args.candidate}
        if len(candidates) != len(args.candidate):
            parser.error("checkpoint candidates must be distinct")
        result = select_checkpoint(args.output, args.valid_manifest, candidates)
    elif args.command == "freeze-vlm-blind-protocol":
        from inverser.blind_protocol import freeze_blind_protocol

        result = freeze_blind_protocol(
            args.output,
            args.test_manifest,
            args.valid_manifest,
            args.arms,
        ).model_dump(mode="json")
    elif args.command == "run-vlm-eval":
        from inverser.study_calls import local_vllm_backends, openai_proxy_backends
        from inverser.vlm_eval import run_vlm_eval

        if args.proxy:
            backend = openai_proxy_backends(
                Path("configs/proxy_model_aliases.json"),
                (args.model,),
                model_revision=args.revision,
                reasoning_effort="high" if args.thinking else "none",
                max_output_tokens=args.max_output_tokens or 8192,
                request_token_reservation=16384,
                timeout_s=300,
            )[0]
        else:
            backend = local_vllm_backends(
                args.local_vllm_url,
                (args.model,),
                model_revision=args.revision,
                reasoning_effort="high" if args.thinking else "none",
                max_output_tokens=args.max_output_tokens or (2048 if args.thinking else 1024),
                request_token_reservation=16384,
                timeout_s=180,
                temperature=args.temperature,
                enable_thinking=args.thinking,
                thinking_token_budget=(
                    args.thinking_token_budget
                    if args.thinking_token_budget is not None
                    else (1024 if args.thinking else None)
                ),
            )[0]
        result = run_vlm_eval(
            args.split,
            args.output,
            backend,
            tasks=tuple(args.tasks),
            count=args.count,
            budget_tokens=args.max_total_tokens,
            execute=args.execute,
            blind_protocol=args.blind_protocol,
            arm=args.arm,
            seed=args.seed,
            workers=args.workers,
        )
    elif args.command == "prepare-vlm-suite":
        from inverser.vlm_suite import prepare_vlm_suite, verify_vlm_suite

        operation = verify_vlm_suite if args.verify_only else prepare_vlm_suite
        result = operation(args.output_root, args.exclude_pilot, args.exclude_formal)
    elif args.command == "verify-external-suites":
        from inverser.external_benchmarks import verify_external_suites
        from inverser.runtime import write_json

        result = verify_external_suites()
        if args.output:
            write_json(args.output, result)
    else:
        from inverser.preview import render_preview

        render_preview(args.input, args.output)
        result = {"preview": str(args.output)}
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.command == "verify-external-suites" and result.get("status") != "ready":
        return 1
    if (
        args.command in ("run-vlm-eval", "run-public-eval", "run-agentic-baseline")
        and result.get("status") == "stopped"
    ):
        return 1
    return 0
