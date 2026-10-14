import argparse
import json
import logging
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="InverseR data-quality pilot tools")
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
    acquire = commands.add_parser(
        "acquire-source-pilot", help="Acquire only rights-qualified samples"
    )
    acquire.add_argument("--registry", type=Path, default=Path("configs/sources.json"))
    acquire.add_argument("--output", type=Path, required=True)
    sources = commands.add_parser("source-catalog", help="Show verified source states and blockers")
    sources.add_argument("--registry", type=Path, default=Path("configs/sources.json"))
    hf_download = commands.add_parser(
        "download-hf-source", help="Acquire an explicitly requested public repository snapshot"
    )
    hf_download.add_argument("--registry", type=Path, default=Path("configs/sources.json"))
    hf_download.add_argument("--output", type=Path, required=True)
    hf_download.add_argument("--max-workers", type=int, default=4)
    archive = commands.add_parser(
        "wait-import-archive", help="Wait for a stable complete ZIP, then safely extract and verify"
    )
    archive.add_argument("--archive", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)
    archive.add_argument("--stable-seconds", type=float, default=120)
    archive.add_argument("--max-wait-seconds", type=float, default=21600)
    processing = commands.add_parser(
        "process-video-pilot",
        help="Scan bounded calibrated windows and export Silver shot candidates",
    )
    processing.add_argument("--source", type=Path, required=True)
    processing.add_argument("--output", type=Path, required=True)
    processing.add_argument(
        "--videos", nargs="+", default=["match12.mp4", "match13.mp4", "match29.mp4"]
    )
    processing.add_argument("--duration", type=float, default=600)
    processing.add_argument("--start-fraction", type=float, default=0.05)
    processing.add_argument("--sample-hz", type=float, default=5)
    processing.add_argument("--max-candidates", type=int, default=12)
    processing_audit = commands.add_parser(
        "audit-video-pilot", help="Validate candidate exports and keep unresolved quality flags"
    )
    processing_audit.add_argument("--output", type=Path, required=True)
    collection = commands.add_parser(
        "collect-video-corpus",
        help="Collect full recordings into resumable BEV observation episodes",
    )
    collection.add_argument("--source", type=Path, required=True)
    collection.add_argument("--output", type=Path, required=True)
    collection.add_argument("--workers", type=int, default=4)
    collection.add_argument("--sample-hz", type=float, default=5)
    collection.add_argument("--limit-jobs", type=int)
    recovery = commands.add_parser(
        "recover-video-corpus", help="Retry failed windows without modifying the frozen parent run"
    )
    recovery.add_argument("--source", type=Path, required=True)
    recovery.add_argument("--parent", type=Path, required=True)
    recovery.add_argument("--output", type=Path, required=True)
    recovery.add_argument("--workers", type=int, default=4)
    author = commands.add_parser(
        "process-author-dataset", help="Process uploaded source clips into the common BEV schema"
    )
    author.add_argument("--source", type=Path, required=True)
    author.add_argument("--output", type=Path, required=True)
    author.add_argument("--wait-ready-seconds", type=float, default=0)
    screen = commands.add_parser(
        "screen-real-corpus", help="Automatically triage complete raw episodes without claiming GT"
    )
    screen.add_argument("--source", type=Path, required=True)
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--workers", type=int, default=4)
    uploaded_inventory = commands.add_parser(
        "inventory-author-uploads", help="Snapshot stable direct uploads without waiting for a ZIP"
    )
    uploaded_inventory.add_argument("--source", type=Path, required=True)
    uploaded_inventory.add_argument("--output", type=Path, required=True)
    uploaded_inventory.add_argument("--stable-seconds", type=float, default=120)
    uploaded_inventory.add_argument("--observe-seconds", type=float, default=2)
    uploaded_inventory.add_argument("--games", nargs="+", choices=["snooker", "blackball", "carom"])
    uploaded_inventory.add_argument("--hash-cache", type=Path)
    uploaded_inventory.add_argument("--essential-only", action="store_true")
    uploaded_pilot = commands.add_parser(
        "convert-author-upload-pilot",
        help="Convert stable source captures with explicit game labels",
    )
    uploaded_pilot.add_argument("--inventory", type=Path, required=True)
    uploaded_pilot.add_argument("--output", type=Path, required=True)
    uploaded_pilot.add_argument("--game", choices=["snooker", "blackball"], required=True)
    uploaded_pilot.add_argument("--limit", type=int, default=5)
    author_sync = commands.add_parser(
        "sync-snooker-uploads",
        help="Incrementally convert unique stable snooker captures and audit native manual paths",
    )
    author_sync.add_argument("--inventory", type=Path, required=True)
    author_sync.add_argument("--output", type=Path, required=True)
    author_sync.add_argument("--max-new-clips", type=int, default=20)
    author_sync.add_argument("--hash-cache", type=Path)
    catalog = commands.add_parser(
        "build-research-catalog",
        help="Index real and simulated sources with honest label boundaries",
    )
    catalog.add_argument("--author", type=Path, required=True)
    catalog.add_argument("--author-inventory", type=Path, required=True)
    catalog.add_argument("--matches", type=Path, required=True)
    catalog.add_argument("--match-screening", type=Path, required=True)
    catalog.add_argument("--synthetic", type=Path, required=True)
    catalog.add_argument("--output", type=Path, required=True)
    study_dry = commands.add_parser(
        "study-dry-run", help="Validate paired study plumbing with a mock, not model inference"
    )
    study_dry.add_argument("--input", type=Path, required=True)
    study_dry.add_argument("--output", type=Path, required=True)
    baseline_dry = commands.add_parser(
        "baseline-dry-run",
        help="Validate classical search and forecast metrics on one development fixture",
    )
    baseline_dry.add_argument("--input", type=Path, required=True)
    baseline_dry.add_argument("--output", type=Path, required=True)
    baseline_dry.add_argument("--objective-budget", type=int, default=8)
    baseline_dry.add_argument("--candidates", type=int, default=4)
    baseline_dry.add_argument("--seed", type=int, default=20261013)
    baseline_dry.add_argument("--limit-jobs", type=int)
    canvas_audit = commands.add_parser(
        "audit-author-canvases",
        help="Audit at most 16 native source canvases using two sampled frames per recording",
    )
    canvas_audit.add_argument("--episodes", type=Path, required=True)
    canvas_audit.add_argument("--inventory", type=Path, required=True)
    canvas_audit.add_argument("--output", type=Path, required=True)
    canvas_audit.add_argument("--limit-recordings", type=int, default=16)
    point_audit = commands.add_parser(
        "audit-native-point-compatibility",
        help="Check sampled detections against held-out native paths without refitting the mapping",
    )
    point_audit.add_argument("--mapping", type=Path, required=True)
    point_audit.add_argument("--episodes", type=Path, required=True)
    point_audit.add_argument("--output", type=Path, required=True)
    semantic_qc = commands.add_parser(
        "run-author-semantic-qc",
        help="Compare five fixed CPU-only pipeline profiles on at most 16 author recordings",
    )
    semantic_qc.add_argument("--mapping", type=Path, required=True)
    semantic_qc.add_argument("--episodes", type=Path, required=True)
    semantic_qc.add_argument("--output", type=Path, required=True)
    semantic_qc.add_argument("--workers", type=int, default=4)
    semantic_qc.add_argument("--max-new-output-gb", type=float, default=20)
    semantic_qc.add_argument("--limit-recordings", type=int)
    semantic_qc.add_argument(
        "--profile-suite",
        choices=["initial_ablation", "cue_gap_no_background"],
        default="initial_ablation",
    )
    analytics_audit = commands.add_parser(
        "audit-billiards-analytics",
        help="Audit acquired nine-ball layouts/trajectories without assuming cue or spin GT",
    )
    analytics_audit.add_argument("--dataset", type=Path, required=True)
    analytics_audit.add_argument("--trajectories", type=Path)
    analytics_audit.add_argument("--output", type=Path, required=True)
    real_forecast = commands.add_parser(
        "prepare-real-world-model-pilot",
        help="Export source-annotated factual-prefix forecasts, not real action interventions",
    )
    real_forecast.add_argument("--trajectories", type=Path, required=True)
    real_forecast.add_argument("--output", type=Path, required=True)
    diagnostic_panel = commands.add_parser(
        "prepare-diagnostic-panel",
        help="Lock a small 64+64 development panel without model inference or formal-test claims",
    )
    diagnostic_panel.add_argument("--synthetic", type=Path, required=True)
    diagnostic_panel.add_argument("--real-public", type=Path, required=True)
    diagnostic_panel.add_argument("--real-private", type=Path, required=True)
    diagnostic_panel.add_argument("--output", type=Path, required=True)
    diagnostic_panel.add_argument("--seed", type=int, default=20261014)
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
    elif args.command == "acquire-source-pilot":
        from inverser.sources import acquire_source_pilot

        result = acquire_source_pilot(args.registry, args.output)
    elif args.command == "source-catalog":
        from inverser.sources import load_sources

        result = {
            "sources": [source.model_dump(mode="json") for source in load_sources(args.registry)]
        }
    elif args.command == "download-hf-source":
        from inverser.hf_download import download_hf_source

        result = download_hf_source(args.registry, args.output, max_workers=args.max_workers)
    elif args.command == "wait-import-archive":
        from inverser.archive import wait_for_upload

        result = wait_for_upload(
            args.archive,
            args.output,
            stable_seconds=args.stable_seconds,
            max_wait_seconds=args.max_wait_seconds,
        )
    elif args.command == "process-video-pilot":
        from inverser.processing import process_video_pilot

        result = process_video_pilot(
            args.source,
            args.output,
            args.videos,
            duration_s=args.duration,
            start_fraction=args.start_fraction,
            sample_hz=args.sample_hz,
            max_candidates=args.max_candidates,
        )
    elif args.command == "audit-video-pilot":
        from inverser.processing_audit import audit_processing, render_processing_overview

        result = audit_processing(args.output)
        if result["candidate_clips"]:
            render_processing_overview(args.output)
    elif args.command == "collect-video-corpus":
        from inverser.collection import collect_corpus

        result = collect_corpus(
            args.source,
            args.output,
            workers=args.workers,
            sample_hz=args.sample_hz,
            limit_jobs=args.limit_jobs,
        )
    elif args.command == "recover-video-corpus":
        from inverser.collection import recover_failed_corpus

        result = recover_failed_corpus(args.source, args.parent, args.output, workers=args.workers)
    elif args.command == "process-author-dataset":
        from inverser.author_data import process_author_dataset

        result = process_author_dataset(
            args.source, args.output, wait_ready_seconds=args.wait_ready_seconds
        )
    elif args.command == "screen-real-corpus":
        from inverser.quality_screening import screen_corpus, screening_gallery

        result = screen_corpus(args.source, args.output, workers=args.workers)
        screening_gallery(args.output)
    elif args.command == "inventory-author-uploads":
        from inverser.author_uploads import snapshot_uploads

        result = snapshot_uploads(
            args.source,
            args.output,
            stable_seconds=args.stable_seconds,
            observe_seconds=args.observe_seconds,
            games=tuple(args.games) if args.games else None,
            hash_cache=args.hash_cache,
            essential_only=args.essential_only,
        )
    elif args.command == "convert-author-upload-pilot":
        from inverser.author_uploads import convert_upload_pilot

        result = convert_upload_pilot(args.inventory, args.output, game=args.game, limit=args.limit)
    elif args.command == "sync-snooker-uploads":
        from inverser.author_sync import sync_snooker_uploads

        result = sync_snooker_uploads(
            args.inventory,
            args.output,
            max_new_clips=args.max_new_clips,
            hash_cache=args.hash_cache,
        )
    elif args.command == "build-research-catalog":
        from inverser.corpus_catalog import build_corpus_catalog

        result = build_corpus_catalog(
            args.author,
            args.author_inventory,
            args.matches,
            args.match_screening,
            args.synthetic,
            args.output,
        )
    elif args.command == "study-dry-run":
        from inverser.evaluation import study_dry_run

        result = study_dry_run(args.input, args.output)
    elif args.command == "baseline-dry-run":
        from inverser.study_runner import EngineeringRunConfig, baseline_dry_run

        result = baseline_dry_run(
            args.input,
            args.output,
            EngineeringRunConfig(
                seed=args.seed,
                objective_budget=args.objective_budget,
                candidate_count=args.candidates,
            ),
            limit_jobs=args.limit_jobs,
        )
    elif args.command == "audit-author-canvases":
        import cv2

        from inverser.canvas_audit import audit_author_canvases

        cv2.setNumThreads(1)
        report = audit_author_canvases(
            args.episodes,
            args.inventory,
            args.output,
            limit_recordings=args.limit_recordings,
        )
        result = report.model_dump(mode="json", exclude={"recordings"})
    elif args.command == "audit-native-point-compatibility":
        import cv2

        from inverser.canvas_audit import audit_registered_points

        cv2.setNumThreads(1)
        point_report = audit_registered_points(args.mapping, args.episodes, args.output)
        result = point_report.model_dump(mode="json", exclude={"rows"})
    elif args.command == "run-author-semantic-qc":
        from inverser.semantic_qc import SemanticQCConfig, run_semantic_qc

        result = run_semantic_qc(
            args.mapping,
            args.episodes,
            args.output,
            SemanticQCConfig(
                workers=args.workers,
                max_new_output_bytes=int(args.max_new_output_gb * 1_000_000_000),
                profile_suite=args.profile_suite,
            ),
            limit_recordings=args.limit_recordings,
        )
    elif args.command == "audit-billiards-analytics":
        from inverser.analytics_data import audit_analytics_data

        result = audit_analytics_data(args.dataset, args.output, args.trajectories)
    elif args.command == "prepare-real-world-model-pilot":
        from inverser.world_model_data import build_factual_forecast_pilot

        result = build_factual_forecast_pilot(args.trajectories, args.output)
    elif args.command == "prepare-diagnostic-panel":
        from inverser.diagnostic_panel import DiagnosticPanelConfig, prepare_diagnostic_panel

        result = prepare_diagnostic_panel(
            args.synthetic,
            args.real_public,
            args.real_private,
            args.output,
            DiagnosticPanelConfig(seed=args.seed),
        )
    else:
        from inverser.preview import render_preview

        render_preview(args.input, args.output)
        result = {"preview": str(args.output)}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
