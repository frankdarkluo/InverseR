# InverseR: evidence-driven physical prediction and goal-conditioned planning

## Status and purpose

Status: initial setup and raw-data engineering were approved and executed.
The current empirical-study supplement was approved for staged preparation.
It does not authorize model jobs, serving installation or data expansion before
their per-route resources and research gates are confirmed.
G1-G4 have not been passed; formal model experiments remain unexecuted.

The active next-stage plan is [Section 13](#13-active-empirical-study-plan).
It supersedes conflicting earlier resource and million-scale assumptions:

- The researcher confirmed empirical-first, structured-state/BEV paired inputs,
  and inference-only diagnostics before any new training.
- APIs may receive self-generated synthetic states/images only; real recordings
  and manual annotations remain local.
- GPU, CPU, storage and API allocations need reconfirmation. Historical caps
  below do not authorize current consumption.
- Preparation, compatibility checks, development diagnosis and formal evaluation
  are separate stages with explicit approvals, not one automatic full sweep.

### Initial execution checkpoint (before full HF acquisition)

- Native Claude ARS v3.23.0 installed and enabled; installed skill hashes match the pinned source and native hooks passed harmless smoke checks.
- Five namespaced Copilot entries are installed and validated on disk. Actual live invocation returned "not found"; session/window refresh is required before claiming activation.
- Isolated CPU project environment, source registry, rights-gated sample acquisition, typed task/physics/metric contracts, resumable synthetic pilot and endpoint previews are implemented.
- Ruff and Mypy passed; 23 tests passed.
- Generated 2,000 synthetic development tasks from 2,039 attempts, with 39 scratches explicitly rejected. All 2,000 generating witnesses passed replay, with zero endpoint discrepancy in the pinned environment.
- Downloaded and decoded three licensed Commons clips. Sampled-frame triage did not establish eligible complete endpoint pairs; accepted real tasks remain zero.
- A reachable approximately 150 GB snooker-video collection lacks verified media rights; the author-published 100-shot snooker source remains access-blocked. No rights-unclear corpus was downloaded.
- Full real calibration/tracking, million-task expansion, model inference/training and API comparisons have not been performed. G1 has not passed.

### Subsequent raw-acquisition instruction

The researcher explicitly requested the entire public `spdev39/snookerVideos`
repository locally under InverseR/data. This changes the raw-acquisition limit
for that source only. Revision `dcbcdf46b71c30e76bf51220f697836abd9c60bf`
contains 54 files (32 videos, 21 corner JSON sidecars and repository metadata),
totalling 149,838,665,026 bytes. The four-worker anonymous download completed at
2026-10-10 13:27:34 UTC. All 54 source-file checksums and all 32 video stream
probes passed. Final inventories and 32 contact sheets were persisted.

The first completed video, `match13.mp4`, is 1280x720/30fps and approximately
2h18m. Sampled-frame triage shows a promising near-complete table view, but does
not establish shot labels. All videos total 228.3802 hours. The 21 corner
sidecars have four points inside their associated frame bounds; eleven videos
have no sidecar. No exact duplicate video hashes were found, but overlapping
footage has not been ruled out. Media rights and shot-label validation remain
unresolved; training, redistribution and G1 remain unapproved.

### Local processing checkpoint

The researcher moved the snapshot to `data/snookerVideos` and authorised local
quality processing. The first three-video pilot scanned ten minutes per video
at five frames per second: 9,000 frames and 32 provisional exported candidates.
Candidate clips, endpoint images and observed-only trajectories passed schema,
timestamp and media checks. Twenty-two have nontrivial cue endpoint displacement;
all remain Silver/unverified with unresolved full-inventory/identity flags.
No candidate was promoted to benchmark ground truth.

Ruff, Mypy, dependency consistency and 53 tests passed. The failed early
plane-warp-first detector output was retained; source-image sphere detection,
circular hue handling and observed-motion segmentation replaced that approach.

`data/OneDrive_1_2026-10-10.zip` is uploading. The attached
`billiard-upload-import` monitor waits for a complete ZIP directory and two
minutes of stability, then safely extracts with CRC/SHA receipts while preserving
the original. Last observed upload was 5.99 GB with no complete directory.
No author-dataset payload or shot count is assumed before completion.

### Whole-source collection checkpoint

The researcher authorised all-video slicing and requested common overhead/BEV
structure for the HF matches and uploaded author dataset. The scale decision is
to finish source collection first, then choose model experiments based on
qualification rate.

The paper's core methods and Section 4.1 were read against the author README
and example. It uses a small experimental table, 51 mm snooker balls and
640x480/20fps RGB capture; manual PNG paths and generated X3D worlds are not
treated as measured cue-action or angular-velocity labels.

`snooker-observation-v1` and both source adapters are implemented. Seventy-one
tests, Ruff and Mypy passed, including common-record, native sphere geometry,
archive-source filtering, ownership and resume tests.

`full-snooker-corpus-v1` is an attached four-worker collection command covering
32 recordings/228.38h through 1,387 ownership windows at 5 sampled fps.
State and SQLite records are at `data/structured/snooker-corpus-v1/`.
It is running, not complete; do not restart or claim final sample counts.

`billiard-structured-v1` is a separate attached consumer waiting for the existing
`billiard-upload-import` to finish. It does not re-extract the ZIP. After import
it selects original snooker AVI clips, deduplicates exact hashes and emits the
same common schema with the paper's source-specific geometry and preserved
annotation provenance. Its state is under `data/structured/billiard-v1/`.

At launch the upload was approximately 11.2GB and structurally incomplete.
No benchmark gate, formal experiment or model/API operation is authorised by
merely collecting raw source candidates.

### Archive verification and failed-window recovery checkpoint

The importer and author consumer exhausted their wait budgets without extracting.
At 2026-10-11 17:14 UTC the upload remained 21,407,354,487 bytes, unchanged for
approximately 25 hours, with a local ZIP header but no ZIP/ZIP64 end directory.
The researcher has not checked whether the original local archive opens.
Preserve the file and wait for a verified complete upload; do not force extraction
or repeatedly reset a six-hour wait.

The original full collector completed 1,375/1,387 windows and indexed 7,084 raw
episodes. Twelve failed on a 4x4 component whose subpixel core contains no
pixel centers. The exact real frames and a new regression reproduced that error.
The fix counts/logs the component as unresolved instead of aborting or inventing
a color. Seventy-three tests, Ruff and Mypy passed.

The `snooker-failed-window-recovery` command completed successfully after retrying
only those twelve windows into `data/structured/snooker-corpus-v1-recovery`.
Its manifest records parent/recovery code hashes; the original successful
outputs and ledger are unchanged.

All 1,387 windows now cover 228.38 source hours, with 7,114 raw episodes.
Every consolidated record passed schema validation; JSONL/database/state counts
agree and all referenced endpoint/BEV/clip assets exist. Twenty-seven recordings
produced candidates, while all 32 were scanned. These are raw, unverified
observations, not independent accepted shot GT. The completed collection audit
is persisted; no collector is still running and no model experiments have run.

### Complete raw-corpus screening checkpoint

The researcher requested autonomous quality screening and an evidence-based
judgement about further scale. All 7,114 records were screened in a separate
versioned output, with source payload/hash, media, pixel/metric, trajectory,
terminal-rest, color/inventory, camera and probable-duplicate checks.

First-pass screening was preserved. Assistant temporal review of all 17
first-pass priorities identified a duplicate hidden by differing BEV
projections. Version 2 uses native endpoint hashes with the same source/time
constraints. Its final counts are 16 automatic review priorities, 6,004
requiring reprocessing, 887 outside the conservative protocol and 207 probable
duplicates; zero structural/media-invalid records. Raw records are unchanged.

One priority visibly misses a black ball and is separately held for re-detection.
The remaining 15 are unverified review priorities, not accepted single-shot GT.
There are 369 motion-supported repair priorities across 19 recordings; with
the held inventory case, the targeted repair queue contains 370 records.
All prior outputs, reasons and source groups are retained.

Seventy-nine tests, Ruff and Mypy passed. The screen used CPU only; an NVML
driver/library mismatch was observed and no GPU operation was attempted.
No model/API experiment, human-Gold claim or benchmark-freeze decision occurred.
Final outputs are under `data/quality/snooker-screen-v2`, with separate
assistant triage and validation records. The next useful scaling step is
targeted extraction repair, not duplicating frames or lowering thresholds.

### Direct author-folder upload checkpoint

The researcher is uploading folders directly to `data/Billiard-dataset`; this
path no longer depends on the incomplete ZIP/import ledger. An immutable
2026-10-13 02:51:16 UTC snapshot contains 3,157 files/38.42 GB, 79 original
blackball AVI recordings (77 stable), 79 raw BGR captures and one background.
No snooker original or manual-path PNG was observed in that snapshot.
Recent/changing inputs are deferred; all uploads remain read-only.

Direct-upload inventory and bounded conversion commands are implemented.
`BlackballEpisode` shares observation/calibration/trajectory/media fields with
the snooker record, but has a distinct game/schema and cannot pass the main
snooker validator. No blackball recording is added to snooker benchmark counts.

Five stable blackball captures were converted to
`data/structured/author-blackball-pilot-v2`. They contain 1,182 complete raw
frames, source/BEV endpoints, observed-only tracks and explicit unverified
quality flags. Fifteen algorithm-animation exports contain 177 track sequences;
these do not represent 177 independent shots or measured GT.

Real-format checks found a one-frame count difference in all five pilots.
BlackBall_0's extra final AVI frame is black. The converter prefers whole raw
BGR captures when available. Its MOLT reconstruction uses 264 normalised keys
and an 11.1645-second animation cycle versus the 13.2-second AVI duration;
native scene units and animation clocks remain separate until aligned.

Ninety-three tests, Ruff, Mypy, record/media checks and frozen-code hashes pass.
There were no model/API calls, main benchmark additions, automatic expansion
of later uploads or new cue/spin ground-truth claims. The next use is controlled
perception/tracking diagnosis and conditional weak-supervision/transfer studies.
Original snooker captures/background/manual references remain the priority
upload for the main task.

### Optimised incremental snooker/manual ingestion checkpoint

The researcher started uploading both snooker directories and requested
preprocessing/experiment-code optimisation. The latest frozen essential-input
snapshot contains 65 original snooker video files, 63 stable, 16 manual path
images and two backgrounds. Exact video/raw-content grouping yields 57
recording identities, 55 ready and two deferred upload groups.

The optional checksum cache validates path/size/mtime/ctime/device/inode.
The first scoped scan hashed 1,352 files/24.25GB; the subsequent scan reused
all 1,352 and read only 39 new/changed files/0.66GB. Large comparison renders
and static scene assets are excluded from the essential-input pass.

The new finite sync command groups ordinary/manual aliases by content,
prefers a stable manual representation, defers missing/raw-upload conflicts,
and fingerprints source/background/annotation/code dependencies. Completed
conversions are reused; later annotations produce a new per-record version.
Old input snapshots and outputs remain. Content groups and aliases are
included in experiment indexes, with no automatic frozen train/test split.

All 55 ready recording groups converted successfully into existing strict
snooker records, processing 21,265 complete raw frames. Sixteen records have
original manual-path references. First five conversions were reused, fifty
added; an identical full rerun reused all 55 and converted zero. All records,
media and source/code fingerprints passed their declared checks.

The manual rasters share native canvases with provided algorithm path rasters.
All 160 comparisons (16 manual cases x ten source algorithms) reproduce the
provided Jaccard values within 1e-5, with no resizing/fitted alignment. These
are source-algorithm spatial path metrics, not our method performance or
time-indexed/action/spin GT.

One hundred tests, Ruff and Mypy passed. No GPU/model/API operation or benchmark
acceptance occurred. All 55 still have incomplete-inventory uncertainty and
other detector/identity flags. The usable next experiment is independently
checked extraction development on the manual reference set, followed by
source-grouped endpoint/evidence and transfer tests after qualification.

Commands and results are documented in `docs/pipelines/incremental_snooker_pipeline.md`;
latest index/validation are linked by
`data/structured/author-snooker-incremental-v1/latest.json`.
No ongoing uploader watcher was installed; new arrivals need another immutable
inventory plus finite sync, not silent insertion into an old experiment.

### Completed author upload and world-model research checkpoint

The latest complete direct-folder upload was independently audited: 100
content-distinct original captures per game, 354 video copies, and 54 manual
copies linked to originals. The 7,394 essential assets are stable/checksummed.
Content uniqueness is not proof of statistical independence across a table,
camera or session. Preserve source families in all later splits.

All 100 main snooker captures were converted: 37,876 complete raw frames and
16 native manual spatial-path references. This sync converted 45 new captures
and reused 55; no failures/deferred captures remain. All 100 still have
automatic-detector, inventory and identity uncertainty. No measured real action
or independently accepted main physical-GT task exists.

The completed snapshot hash is
`551139d568f7263b67345a51fefcb4ef1fcba82106c4359f9abb65230e52564e`.
The source manifests and receipts are retained in
`data/quality/author-completed-upload-snapshot-v1/` and
`data/structured/author-snooker-incremental-v1/`.
The incomplete ZIP was preserved and was not needed for direct-folder conversion.

`data/research/physics-world-model-catalog-v1/` now joins 9,414 typed references:
7,214 main real observations (100 author plus 7,114 weak match candidates),
200 auxiliary captures, and 2,000 simulator-truth development shots. Auxiliary
captures are references, not a claim that all were converted. All references,
IDs and source hashes passed validation. The complete code suite passed 103
tests, Ruff and Mypy. No evaluation-model inference, training, API experiment
or physical execution ran. Research assistance and sampled-frame review are
not formal model-performance experiments.

Three independent public-primary-source research perspectives completed:
learned world-model interfaces, physical-evaluation validity, and closest
inverse planning/system identification. Fifteen method/benchmark works and
the author dataset paper were verified. A narrow follow-up confirmed the
official PhysBench ICLR 2025 record. Formal/preprint title differences for
World Models and DreamerV3 were preserved.

The Chinese working survey and empirical proposal is persistent at
`docs/research/world_model_research.md`. It answers the frozen survey RQs and proposes
paper RQs aligned with the existing ambiguity/attribution, structured-search
and real-transfer research axes. Real post-impact observations may train
autonomous forecasting, but do not provide measured cue-to-motion labels.
Separate native action-conditioned prediction, learned transition dynamics,
calibrated mechanistic models, value surrogates and LLM tool control.

The researcher selected the empirical-first prediction/planning-consistency
framing after reading the proposal: diagnose capabilities, then decide whether
a learned world model or agent method is warranted. This confirms the working
research direction, not data acceptance or any formal-evaluation approval.

Next proposed scope: independently establish native manual-raster alignment,
repair the 100 fixed-camera main captures and 370 targeted match candidates,
then prepare small controlled action-intervention and shared-candidate views.
Do not silently pass G1/G2/G3/G4, expand to a million, select a new learned
method or launch formal model evaluation. At the venue decision, choose CVPR
or continue toward IJCAI; their review/deadline windows overlap.

Build a reproducible snooker inverse-action benchmark and an experiment harness. Given initial and desired terminal table states, the main task is to find any feasible single shot. Historical reconstruction is a separate diagnostic. Establish the benchmark and capability boundaries before deciding whether the paper should be method-led or benchmark-led.

The assistant executes approved engineering and research-support work; the researcher owns research decisions and verifies scientific claims. Acceptance at a venue is not a deliverable that can be guaranteed.

## 1. Confirmed decisions

| Item | Confirmed decision |
| --- | --- |
| Skills installation | Personal/global installation in both channels: native Claude Code plugin and Copilot-compatible personal skills. Preserve existing customizations. |
| Main domain | Snooker only. Other cue sports may supply auxiliary pretraining, not main benchmark counts or main test results. |
| Main task | Goal-conditioned, physically feasible single-shot synthesis. Do not require matching the historical human action. |
| Initial scope | Planar, low-elevation shots. Include direction, speed and spin-related controls; expand contact complexity in documented tiers. Defer jumps, masse, complex break shots, full-match planning and comprehensive tournament-rule reasoning. |
| Scale | Historical long-term target: 1,000,000 independent main-domain physical tasks. The current proposed empirical study does not require this before its pilot or evidence decision; expansion remains separately gated. |
| Real data | Actively discover public research datasets and match-video resources. Confirm source provenance and permitted uses before acquisition. |
| Human annotation | No current manual review capacity. New automatic labels are Silver/weak labels. Published manual annotations retain their original provenance; do not claim new human-verified Gold. |
| Local model | Start with the researcher-provided read-only `Qwen3.6-27B` checkpoint under a configurable model root. |
| Closed models | Later compare requested GPT-5.5 and GPT-6.1 Sol deployments if the existing API actually exposes them and supports the required image inputs. Never silently substitute a model. |
| API cap | Pending reconfirmation for the current stage. Require a positive per-model request/token cap before paid calls; probes, retries and loops are included. The earlier 100-request allowance is historical, not renewed approval. |
| GPU allocation | Pending explicit permitted UUIDs and allocation. Historical ceiling was two idle A100 80GB devices. Do not load weights, stop or preempt jobs based on apparent idleness alone. |
| CPU/network allocation | Pending reconfirmation for actual experiment jobs. Historical limits were 16 CPU workers and four downloads; this plan requires no new media download. |
| Storage | Pending a confirmed experiment/storage reservation. Historical project ceiling was 1 TB. Reuse existing shared weights read-only, never duplicate them by default. |
| Autonomy | Automatic execution within a stage. Wait for approval at data quality, benchmark freeze, method selection and formal evaluation. |
| Paper positioning | Choose the main contribution after diagnostic evidence, rather than precommitting to a positive method result. |
| Publication | No automatic data release, upload, email outreach, manuscript submission or simultaneous submission. |

### Initial planning facts (historical; current facts are in Section 13)

- The initial project snapshot contained the proposal PDF, an empty `main.py`, and no established implementation or test suite. Current implementation status is recorded in Section 13 and the repository README.
- The 45-page `docs/Inverse Reasoning in Billiards.pdf` and the preceding discussion are research inputs, not verified experimental evidence.
- The existing personal skill collections already contain `deep-research` and several writing skills. Upstream's same-named skill must not replace them.
- `claude` and `copilot` CLIs are installed. The requested native Claude plugin was not found by the read-only installation check.
- Python 3.12, FFmpeg and PDF text extraction are available. No project dependency installation has been performed for this plan.
- All four visible A100 devices were busy during inspection. Actual permitted GPU UUIDs must be confirmed at execution time; apparent idleness alone is not permission.
- The data volume had approximately 2.6 TB available; the root filesystem had approximately 13 GB available. These are observations, not reserved capacity.
- The requested Qwen model has a vision encoder according to its model card, uses `Qwen3_5ForConditionalGeneration`, and all 15 index-referenced weight shards exist. This does not yet establish runtime compatibility or numerical correctness.
- A separate existing inference client was inspected structurally, read-only, without printing credentials or making API/authentication requests. It contains image/text message construction, bearer-authenticated requests and response validation.
- Unrelated editor attachments and external projects are outside the task.

## 2. Research questions and falsifiable hypotheses

### RQ1: evidence, ambiguity and failure attribution

What can initial and terminal observations determine, and how do perception, contact-path ambiguity and missing intermediate evidence affect feasible-shot search?

Experiments:

- Matched exact-coordinate, noisy-coordinate, synthetic-image and real-observation tracks.
- Endpoint-only inputs versus explicitly privileged first-contact, sparse-event and sparse-trajectory diagnostics.
- No-spin, longitudinal-spin and side-spin action spaces, with matched search budgets.
- Small analytical cases and bounded high-budget searches that expose alternative feasible controls and contact paths.
- No-tool physical-consistency checks using valid coordinate transformations and controlled changes to actions or physics.

Hypothesis: some failures are information limitations or path-selection failures, rather than continuous-parameter precision failures.

Falsifier: improvements are explained by perception alone or by extra numerical search; path or intermediate-evidence interventions provide little additional benefit.

Do not equate failure to find an alternative with uniqueness. Do not score a feasible alternative as wrong merely because it differs from the recorded action.

### RQ2: structured search versus ordinary numerical optimization

Can a learned controller use contact-path hypotheses and simulator feedback to allocate search effort better than strong classical search, direct model proposals or fixed event-then-optimize pipelines?

Experiments:

- Geometry-initialized multi-start optimization.
- A small coordinate/set model proposing actions, followed by the same optimizer.
- Direct Qwen action generation.
- SimLM-style feedback and CueTip-style event-conditioned optimization adapted to quantitative endpoint goals.
- A learned path-switching/refinement/stopping controller, only if pilot evidence supports this mechanism.
- Matched bottom-level optimizers, bounds, observations, training budgets and total simulator accounting.
- Ablations of event feedback, branch switching, learned budget allocation and uncertainty checks.

Hypothesis: contact-mode changes make local refinement insufficient, and feedback-conditioned branch selection can improve selected-shot success at fixed resources.

Falsifier: a geometry-based initializer or small model matches the controller, or simulator-query savings disappear in wall-clock/token cost.

Using a simulator or proposing event sequences is not a novelty claim: SimLM, CueTip and earlier robotic billiards already cover closely related ingredients.

### RQ3: real-data value, scale and transfer

How do data scale, data provenance, label quality and real-data inclusion affect held-out snooker performance and physical consistency?

Experiments:

- Nested training subsets targeting 10k, 100k and 1M tasks, conditional on achieved validated counts.
- Separate data-exposure experiments from fixed-token/fixed-update compute comparisons. Report how many distinct examples the model actually saw.
- Synthetic-only versus synthetic plus real weak data, with actual available real counts and quality strata disclosed.
- Real-layout-conditioned simulation as a separate category, never counted as observed real dynamics.
- Optional pool auxiliary pretraining as a controlled transfer experiment with game-specific physics.
- Held-out layout families, contact compositions, camera conditions and physical parameter settings.
- Real-video consistency using observations not supplied to the inverse model; use separate support shots for physical calibration.

Hypothesis: targeted real layouts and reliable observed dynamics improve transfer more than undifferentiated scale.

Falsifier: benefits vanish after matching data/compute, depend on shared-source leakage, or deteriorate as weak-label quantity increases.

The million-task target must yield a scale/coverage investigation, not merely a larger row count. Negative and null outcomes remain reportable.

## 3. Workstream A: install and verify ARS in both personal channels

### Upstream provenance

- Repository: `Imbad0202/academic-research-skills`.
- Inspected manifest version: `3.23.0`.
- Pin inspected commit: `e614b322297ba5b072a106942177e218cbe2d48f`.
- Upstream license: CC BY-NC 4.0. Preserve notices and attribution, mark any compatibility adaptations, and do not silently relicense vendored material.
- Upstream explicitly states that it handles research/writing/review and experiment provenance, but does not run experiments. InverseR needs its own execution harness.

### Installation procedure after approval

1. Fetch a version-pinned personal source snapshot outside InverseR, for example under `$XDG_DATA_HOME/academic-research-skills/<commit>/`.
2. Inspect the pinned manifests, skill assets, symlinks, hook entry points and invoked dependencies before enabling them. Do not pipe a remote installer into a shell.
3. Keep the five skill directories plus required `shared/`, `scripts/`, references, commands and metadata intact. Avoid copying only `SKILL.md` while breaking resource paths.
4. Register the pinned local marketplace with Claude Code at user scope and install the namespaced plugin using the installed CLI's supported commands.
5. Validate the plugin with `claude plugin validate`; verify user-scope installation using filtered `claude plugin list --json` output.
6. Expose Copilot personal entry points under independent names:
   - `ars-deep-research`
   - `ars-academic-paper`
   - `ars-academic-paper-reviewer`
   - `ars-academic-pipeline`
   - `ars-sr-screener`
7. Use thin compatibility entry points pointing to the pinned personal snapshot. Preserve upstream attribution and an explicit original-to-adapted name map. Translate tool/routing references to supported Copilot workflows without claiming Claude-specific hooks work in Copilot.
8. Keep ARS entry points explicitly invocable initially so they do not replace existing broad-trigger research/writing skills. Do not edit existing skill contents.
9. Verify each channel independently: discovery, name resolution, reference paths, a harmless public-topic invocation, and reported limitations. If the running Copilot session needs a refresh, ask for it; installation on disk alone is not proof of invocation.

### Runtime boundaries

- Keep automatic cross-provider manuscript/code transmission and optional external audit transports disabled.
- Public bibliographic lookups are allowed; do not send unpublished research records, source code, credentials or private manuscript content to third-party services.
- Native Claude hooks are inspected and tested in their native channel. They are not a substitute for experiment isolation or credential protection.
- No unattended plugin upgrades during a research run. Record source commit, version, file hashes and adapter changes.
- Native command defaults and Copilot behavior can differ. Document unavailable capabilities instead of reporting a fake successful equivalent.
- Installing a skill does not authorize a large agent team, a billed model call, or removal of the researcher's stage confirmations.
- Existing global settings are not broadly rewritten. Any necessary installation changes are minimal, reversible and limited to this plugin/adapter.

Acceptance: both requested channels are demonstrably usable; existing skills still resolve; no unrelated configuration or credentials have changed.

## 4. Workstream B: bootstrap a small, reproducible project

Use a dedicated Python environment on the data volume. Add dependency manifests before installation, pin versions after compatibility checks, and avoid changing shared environments.

Proposed code surfaces:

```text
pyproject.toml
src/inverser/
  schemas.py
  cli.py
  data/          # source registry, acquisition, extraction, quality, splits
  physics/       # Pooltool adapter, action/state conversion, replay
  agents/        # classical, small learned, local VLM, API, controller
  training/      # dataset views, LoRA runs, controller supervision
  evaluation/    # budgets, metrics, matched protocols, statistics
  runner/        # finite job execution, resource accounting, resumption
configs/
tests/
README.md
docs/benchmark/benchmark_protocol.md
docs/data/data_card.md
```

Keep modules small and extract helpers only when actually shared. Do not introduce a distributed orchestration framework or dashboard before a CLI and resumable local ledger are sufficient.

Generated data belongs under project `data/`; experiments under a separate artifact directory. Store operational ledgers in SQLite and machine-readable manifests/results in JSON/Parquet. Do not place large data, checkpoints or secrets in version control.

If still absent, initialize local version control when bootstrapping. Do not automatically commit, push or publish. Record a source-tree hash when a commit ID is unavailable.

## 5. Workstream C: real-resource discovery and weak-label construction

### Source discovery priorities

1. Published snooker datasets and author/institutional repositories with actual shot assets.
2. Public match-video collections with explicit research/download permissions and documented rights.
3. Repository or archive mirrors whose relationship to the original source can be established.
4. Pool datasets for auxiliary pretraining only.

Search official repositories, scholarly dataset references, institutional archives, Zenodo/Figshare metadata and public media indexes. Use search output only to discover candidates; verify its claims against primary pages and sample assets.

Record each candidate in a source manifest with:

- Original URL, author/rights holder, source publication and retrieval evidence.
- Actual content type: full video, shot clip, time-stamped trajectory, static image or match statistics.
- Count units, modality, game, camera coverage and declared annotations.
- Separate permissions for acquisition, research processing and redistribution.
- Access state, required authorization, download size and checksums when available.
- Eligibility decision and specific rejection/hold reason.

Public viewing is not proof of reuse permission. Do not bypass access controls, rate limits, paywalls or download restrictions. Draft permission requests if useful, but do not send them without approval.

### Verified starting points and caveats

- Rodriguez-Lozano `Billiard-dataset`: author manifest advertises 100 snooker shots and 16 manual snooker examples within a 300-shot, approximately 190 GB multi-game collection, with CC BY-SA 4.0. The earlier download check returned 403. Actual compressed-video access and manual-label format must be verified; manual trajectory images are not automatically numerical time series.
- `Billiards Sports Analytics: Datasets and Tasks`: accessible example trajectories and strike attributes, author-declared CC0 data, but primarily nine-ball. Use as auxiliary ingestion/pretraining evidence, not main snooker test data. Original broadcast rights and complete raw-video availability are separate questions.
- Figshare `Snooker datasets`, record 13604480: primary metadata describes match results, tournaments and rankings. Exclude from shot-dynamics construction.
- Tracking/BEV demonstration repositories are engineering references, not evidence of licensed large video corpora.
- A search-generated claim about Zenodo record 8180864 could not be verified and returned 404. Exclude it.

Do not promise a large real-shot count from these starting points. The discovery/eligibility ledger determines the real contribution to scale.

### Acquisition and extraction pipeline

1. Acquire only eligible sources, in bounded batches with resumable downloads and checksums.
2. Deduplicate exact files, re-encodes, replay clips and repeated broadcasts before splitting.
3. Detect stable table-view segments, camera changes and candidate single-shot windows.
4. Identify the initial stationary layout and post-shot resting layout before referee respotting or another external intervention.
5. Calibrate the table and account for lens distortion and ball-center height where the view requires it.
6. Detect/track balls with confidence, uncertainty and visibility states. Distinguish pocketed, occluded and unobserved balls.
7. Extract observable events and time-stamped trajectories. Never assume the first tracked moving frame is the instant after cue impact.
8. Validate completeness and quality automatically, retaining explicit exclusions and failure reasons.
9. Produce state/image pairs and privileged interior observations in separate dataset views.

Use independent extraction configurations and cross-checks to flag disagreements. Agreement is a quality proxy, not a replacement for measured accuracy.

### Label provenance and claim limits

| Label tier | Permitted interpretation |
| --- | --- |
| Original source manual annotation | Use only the specific manually annotated quantities, with original provenance and limitations. |
| Silver observation | Automatically estimated positions, tracks, timing, events and calibration uncertainty. Not human-verified truth. |
| Fitted latent hypothesis | Inferred speed, angular velocity, cue impact or physical parameters. Conditional on the assumed model and available observations. |
| Simulated truth | Exact generating controls/parameters for the pinned simulator, not measured real physics. |

Ordinary match video does not generally determine cue force, tip offsets or spin uniquely. More optimization cannot recover information that was not observed.

Real weak labels may support perception, observed-motion learning or model-consistency objectives. Do not automatically turn inverse-fitted real controls into hard action-SFT targets.

### Data-quality pilot and Gate G1

- Start with a bounded representative source/clip pilot, targeting 50 candidate snooker shot windows if eligible assets permit, rather than first downloading whole collections.
- Validate parsers and physical conventions with a synthetic pilot of approximately 2,000-5,000 tasks.
- Measure qualified-shot yield, missing states, calibration consistency, cross-extractor disagreement, source bias, source-label coverage, storage per accepted task and simulator real-fit limitations.
- Use published manual observations for independent checks where their actual format allows; otherwise explicitly report the absence of independent real-label validation.
- No minimum real count is invented if sources are inaccessible or ineligible.

Pause for G1: the user accepts observed data quality and limitations, authorizes a revised collection strategy, or declines expansion. Do not silently replace the real-first plan with a synthetic-only result.

## 6. Workstream D: physical task and million-task construction

### Canonical contract

Represent one task with:

- `task_id`, `group_id`, source identity, provenance tier, game and split.
- Initial/goal ball states in metric table coordinates, with visibility and pocket status.
- Ball classes and observable identity information; exchangeable red balls are not assigned hidden track identities in endpoint-only inputs.
- Explicit goal tolerances and any supplied constraints such as no-scratch or ball-on.
- Input images/rendering references and camera/calibration metadata.
- Action convention, feasible bounds, table/ball/cue configuration and simulator version.
- Privileged generating action, events and trajectory in a separate view inaccessible to the main inference loader.
- Confidence/uncertainty for estimated observations and fit status for latent hypotheses.

Main actions use cue azimuth, cue impact speed and two ball-contact offsets, with fixed documented low elevation. Store the resulting cue-ball linear/angular velocities separately.

Pooltool `V0` is cue speed, not ball speed; `a,b` are normalized contact offsets, not angular velocity. Set every relevant default explicitly, including center-hit offsets. Pin snooker geometry, ball radius and collision models instead of trusting generic defaults.

Preserve all observed balls as obstacles even when only a few participate in motion. Score all constrained positions/statuses without diluting target error through many stationary balls.

Full tournament rules are not inferable from two layouts. Only enforce rule constraints actually included in the task contract.

### Feasibility-preserving generation

1. Sample valid snooker layouts and bounded cue actions.
2. Run the pinned forward simulator.
3. Save terminal states, contact events and generating witness actions.
4. Use initial plus terminal states as the inverse task; hide the witness from inference.
5. Stratify by moving-ball count, contact composition, cushion interactions, longitudinal/side spin, sensitivity and target precision.
6. Include simple analytical/sanity cases as well as difficult cases; do not filter all simple shots away.
7. Use real layouts as conditional seeds where permitted, with the resulting dynamics clearly marked simulated.

Independent random target layouts are not the default: most arbitrary full-table goals may be unreachable from a single shot.

Not every observed real goal is reproducible by the simulator. Record out-of-model real shots and their frequency instead of inventing a fitted success label.

### Counts, storage and splits

- Target one million unique primary snooker tasks; other-game auxiliary samples are accounted separately.
- Suggested split targets are 970k train, 10k development and 20k test, adjusted only to preserve indivisible source/layout groups. Publish achieved counts.
- Multi-view renders, trajectory frames, crops and transformed copies do not inflate independent task counts.
- Keep near-duplicate perturbations and shared base layouts in the same split. Split match/table/camera provenance where coverage allows.
- Predefine held-out contact-family, camera and physics regimes before generating corresponding test banks.
- Generate in deterministic, resumable shards with manifests and checksums.
- Store coordinates/metadata/events compactly. Render synthetic images on demand; retain dense trajectories for the required diagnostic/training subsets rather than rendering a million videos.
- Maintain a minimum shared-disk free-space guard in addition to the 1 TB project cap. Count pending downloads/writes against the reservation ledger.
- Delete only explicitly identified disposable artifacts generated by this project, and retain provenance needed for reproducibility.

### Automatic validation

- Every released shard must pass schema, bounds, identity/status and split-leakage checks.
- Replay generating actions with the same configuration; compare terminal states/events within a documented numerical tolerance.
- Validate unit conversion, cue-to-ball mapping, no-overlap initialization, stopping conditions, collision/pocket events and serialization.
- Test analytical rolling/collision cases and physical sanity within the chosen solver's documented assumptions.
- Failed simulation, nonsettling motion or invalid action is an explicit failure, never a fallback success.
- Alternate-action searches annotate discovered feasible modes, not mathematical uniqueness or exhaustive feasible-set coverage.
- A finite fitted candidate distribution is not called a calibrated Bayesian posterior without a defined prior, likelihood and calibration evaluation.

### Gate G2: benchmark freeze

Present source counts, actual scale, provenance composition, rejection statistics, representative examples, data/split hashes, task contract, tolerances and limitations.

Freeze data version, hidden-test protocol and metric definitions before running the substantive RQ program. Later changes require versioned invalidation and re-evaluation, not moving thresholds to improve results.

## 7. Workstream E: a verifiable evaluation and execution harness

### Baselines and model adapters

- Random/low-discrepancy proposals as sanity checks.
- Geometry-initialized multi-start CMA-ES or differential evolution, tuned fairly on development data.
- A small coordinate/set-based action proposer plus the same optimizer.
- Qwen3.6-27B direct generation, fixed feedback loop and event-conditioned optimization.
- Explicitly labeled SimLM/CueTip task adaptations; do not imply unchanged reproduction when goals differ.
- Closed-model adapters only after mocked protocol tests and an explicitly
  confirmed per-model request cap; no historical allowance is automatically renewed.

Validate local processor/tokenizer/runtime compatibility before loading the model. Start with bounded image resolution/context, a single tiny inference check and measured memory usage on allocated GPUs. For training prefer LoRA/QLoRA or another compatible parameter-efficient route; full 27B fine-tuning is not assumed feasible.

Reuse the referenced API protocol and authentication pattern without modifying the other project, copying hardcoded credentials, dumping endpoint secrets or importing the other project's operational state. Secrets remain runtime-only. Validate deployed model identifiers, response model/version, image support and usage fields.

A small request allowance supports smoke/diagnosis, not an automatically complete
multi-budget leaderboard. Select a deterministic common subset before invocation
and confirm the formal-evaluation allocation separately.

### Budget and output contract

- Count every actual physics rollout inside optimization, reranking, repairs and uncertainty ensembles.
- Separate simulator rollouts, rejected action attempts, model requests/tokens, preprocessing and offline training cost.
- Provide the same input information and permitted tools to comparable baselines.
- Suggested online budgets: 0, 8, 32 and 128 rollouts; calibrate runtime feasibility on development data before freezing.
- The agent returns its final chosen action, not an unscored collection from which the evaluator secretly selects the best.
- Auxiliary candidate-set diversity is reported separately.
- Bound retries, token output, job size and resource use. Missing configuration, exhausted quota and unsupported models fail explicitly.
- Hidden final scoring rollouts are not exposed to the agent; distinguish evaluator cost from agent-accessible search cost.

### Metrics

Main:

- Selected-shot success under explicit endpoint/status/constraint tolerances.
- Success-versus-rollout and success-versus-wall-clock curves.
- Normalized target error, invalid-action rate and no-solution-within-budget rate.

Diagnostics:

- Discovered same-goal path diversity and sensitivity.
- Event/trajectory agreement only for the separately defined reconstruction task or explicitly constrained goals.
- Per-source, per-label-tier, per-complexity and per-physics breakdowns.
- Paired comparisons and confidence intervals clustered by original layout/match rather than correlated frames.
- Multiple seeds for stochastic training/search; disclose limited API repetitions.

Real-data boundaries:

- Endpoint agreement in a simulator is model-conditioned feasibility, not physical-table execution.
- Held-out interior real observations provide reconstruction consistency, not proof of the actual spin/control.
- Estimate shared physical parameters using separate support shots. Do not tune them against hidden query trajectories.
- A sample with no independent observed labels cannot be used to assert measured real action accuracy.

### Leakage controls

- Main inputs expose only permitted endpoint observations and task constraints.
- Remove or separately flag cue-direction arrows, visible aiming cues, replays, captions or metadata that reveal the generating action.
- Do not leak historical red-ball IDs that are unobservable in the endpoints.
- Unknown visibility is not automatically encoded as pocketed.
- Prompt tuning, model selection and controller training use development data only.
- Automated research loops have no access to hidden test metrics during optimization.

## 8. Workstream F: diagnostic experiments and conditional method development

Run RQ1 diagnostics and initial RQ2/RQ3 development comparisons after G2. Use a stratified development panel before expanding expensive sweeps.

Do not search until a desired positive result appears. Pre-register experimental comparisons, retain unsuccessful runs, record each hypothesis and code/data change, and separate exploratory from confirmatory analyses.

### Gate G3: choose the research contribution

Present:

- Which failure sources are supported by measured evidence.
- Whether Qwen proposals or event information add value beyond geometry/small models.
- Whether scale/real-data composition changes performance under appropriate controls.
- Strongest baseline, operational cost and limitations.
- A recommendation: method-led, benchmark-led, or revise the scope.

Wait for the researcher's decision.

### Conditional method: learned contact-path search controller

Only pursue if the diagnostic supports it.

State: endpoint observations, candidate contact paths, actual simulated event traces, residual errors, uncertainty summaries and remaining budget.

Decisions: choose/switch a path, refine its continuous controls, request bounded verification or stop.

Execution: a shared numerical optimizer performs continuous refinement; the model need not invent high-precision floating-point calculations in prose.

Training:

- Generate offline search traces with both useful and unsuccessful branches.
- Learn structured proposals and feedback-conditioned search decisions.
- Begin with supervised/parameter-efficient training.
- Add preference or reward-based training only after the simpler mechanism works and the user accepts the extension.
- Privileged trajectories may supply training supervision, but cannot remove irreducible ambiguity of endpoint-only inputs.

Ablate the controller against fixed budgets, heuristic switching, the same optimizer without language and a small learned policy. Any apparent efficiency gain must survive full cost accounting.

If the evidence favors a benchmark paper, prioritize provenance/quality and informative capability boundaries rather than adding an unnecessary controller.

### Gate G4: formal evaluation approval

Freeze selected checkpoints, source/data hashes, comparison matrix, prompts, seeds, budgets, tolerances and allowed claims.

Present projected request counts and storage reservations from measured pilot logs, not unbounded sweeps. Obtain any additional API allocation separately.

Only then execute hidden-test evaluation and produce immutable result bundles.

## 9. Autoresearch runner and persistence

Each finite job records:

- Run ID, hypothesis/RQ, config and source/data hashes.
- Model identity, seed, split and parent run.
- Declared resource/request limits and observed consumption.
- Start/finish status, checkpoints, metrics, failures and artifact paths.

Use a resumable SQLite queue and append-only result/log records. Retries are bounded and recorded. Test a forced interruption and resumption without duplicate task counts or repeated billed calls.

The runner may propose the next development experiment within approved scope. It may not alter frozen tests, redefine the RQ, expand budgets, hide failures or fabricate missing results.

Long processes remain attached to the session unless the user explicitly requests persistence beyond session exit. Verify any started process and endpoint locally; do not imply work is running when only a plan or configuration exists.

Pause on resource allocation conflicts, data-permission ambiguity, changed evaluation semantics or exhausted quotas.

## 10. Evidence package, writing and submission decision

Generate tables and figures directly from immutable experiment outputs. Link every empirical claim to run IDs, data versions and a stated scope.

Use ARS for literature verification, evidence-to-claim checks, drafting, review and revision. Its review is assistance, not an acceptance predictor or proof that experiments occurred.

Write the methods, data card, failure cases, real-label limitations and reproducibility instructions before making broad physical-understanding claims. Simulator-assisted success alone does not establish native model understanding.

Use the official chosen venue's LaTeX template and instructions; do not inherit ARS's generic APA formatting. Check author responsibility, AI-disclosure and data/code-release requirements. Do not manufacture bibliographic fields or author attestations.

Verified venue constraints, not execution-time estimates:

- CVPR 2027: registration November 10, 2026; paper submission November 16, 2026, AoE.
- CVPR's stated review period overlaps IJCAI's submission window.
- IJCAI 2027 homepage currently lists abstract January 4, 2027 and full paper January 11, 2027, AoE. Recheck the detailed main-track CFP and policies when published.
- Decide before submission whether to submit to CVPR or continue toward IJCAI. Do not assume that waiting for a CVPR decision permits submitting the same paper to IJCAI's earlier deadline.

Evidence gate for a method-led submission: a defensible mechanism, fair strong baselines, reproducible gains/limits, essential ablations and appropriately scoped real evidence.

Evidence gate for a benchmark-led submission: a justified evaluation gap, auditable construction, meaningful independent coverage, honest label tiers, reproducible evaluation and nontrivial supported insights.

If the evidence does not meet either gate, report that and continue research rather than inventing a submission-ready conclusion.

## 11. Todos and dependencies

Tracking statuses live in the session SQL database; this table is the human-readable dependency plan.

| ID | Work | Prerequisites |
| --- | --- | --- |
| ars-dual-install | Install and verify both personal ARS channels | Plan approval |
| project-bootstrap | Establish isolated project environment, manifests and tests | Plan approval |
| source-catalog | Verify and catalog public snooker resources and permissions | Plan approval |
| task-contract | Implement schemas, action/goal conventions and metric tests | project-bootstrap |
| real-ingestion | Implement eligible-source acquisition and weak-label ingestion | source-catalog, task-contract |
| physics-adapter | Implement pinned snooker physics and replay validation | task-contract |
| quality-pilot | Measure real/synthetic quality, yield and failure modes | real-ingestion, physics-adapter |
| gate-data-quality | Obtain G1 data-quality decision | quality-pilot |
| benchmark-build | Build a validated empirical panel; expand toward 1M only if justified and approved | gate-data-quality, study-labels-renderer, study-resource-confirmation |
| model-adapters | Runtime-validate configured local/API deployments | project-bootstrap, task-contract, study-model-client, study-resource-confirmation |
| evaluation-harness | Implement baselines, accounting and resumable jobs; mocked clients precede runtime | physics-adapter, study-budget-ledger, study-model-client |
| gate-benchmark-freeze | Obtain G2 benchmark/protocol freeze | benchmark-build, evaluation-harness |
| diagnostic-rqs | Run development experiments for RQ1-RQ3 | gate-benchmark-freeze |
| gate-method-selection | Obtain G3 contribution/method decision | diagnostic-rqs |
| selected-research-path | Implement and validate the selected method or benchmark analyses | gate-method-selection |
| gate-formal-evaluation | Obtain G4 checkpoint/protocol/budget approval | selected-research-path |
| formal-evaluation | Execute fixed hidden-test comparisons and audit results | gate-formal-evaluation |
| evidence-and-paper | Produce reproducible evidence package and reviewed venue-specific draft | formal-evaluation, ars-dual-install |

Initial implementation authorization covers the setup and quality-pilot prerequisites through G1. Approval of this plan is not approval to bypass G1-G4.

## 12. Acceptance checks

- Both skills channels have an observed discovery/invocation check; existing skills remain intact.
- Dependency manifests, isolated environments and targeted tests cover the actual implementation.
- The physics/metric harness rejects wrong shapes, invalid actions, nonsettling episodes and leaked hidden labels.
- Simulator budget tests count optimizer-internal and ensemble rollouts correctly.
- Download and API quotas persist across retries and resumption.
- Split tests reject repeated matches, shared layout families and transformed duplicates across partitions.
- Real-source permissions and label tiers are inspectable for every accepted sample.
- Million-scale claims use achieved independent snooker counts, not auxiliary pool data or frame counts.
- No private credentials, source files or manuscripts are transmitted to external comparison services.
- Real action accuracy and real execution claims remain absent without independent measurements/execution.
- All result tables can be regenerated from stored runs, including failures and null results.

## 13. Active empirical-study plan

### 13.1 已确认范围与未确认资源

本轮目标是制定可逐阶段执行的方案，围绕“物理预测是否帮助可执行的 goal-conditioned planning”形成 empirical study。主任务仍是 snooker 单杆、初始台面与目标终态到任意可行击球，不要求复现历史人类动作。

研究者已确认：

| 决策 | 当前范围 |
| --- | --- |
| 输入 | 结构化状态与 BEV 图像配对；真实视频为独立观察验证 |
| 第一轮 | 已有模型 inference 与经典搜索，不训练新的 world model，不微调大模型 |
| API 数据 | 仅自生成合成状态与 BEV；真实录制和人工标签不外发 |
| 资源 | 当前额度未确定；本方案不假定 GPU、API 或存储已获分配 |
| 方法 | 先诊断，后决定 companion method；不预设必须提出新算法 |
| 规模 | 百万为长期扩建选项，不是小型实证或首篇文章的先决条件 |

排除首轮：full-match 策略、完整比赛规则推理、跳球/masse、真实机械击球、新模型训练、自动媒体发布，以及在冻结 test 上调 prompt。

所有执行任务在启动前需要有限、明确的资源配置。代码和 mock-test 准备不需要付费 API 或 GPU；实际数据生成、server 与模型任务等待对应 CPU/存储/GPU/API 额度。没有 quota 时标记 blocked，不把 None 当作无限额。

资源确认按 route 记录：可以先确认 CPU/storage 并将 API/GPU 禁用，
不要求所有资源同时具备。Detector/renderer/client 的代码准备可先推进；
100+370 的实际重处理等待 CPU/storage 额度，模型路线等待自己的额度。

### 13.2 研究参考与可借鉴的证据逻辑

主参考是 [WorldArena](https://arxiv.org/abs/2602.08971)，2026 年 arXiv preprint：把预测的感知质量与功能效用分开测量。辅参考是 [CLEVRER](https://arxiv.org/abs/1910.01442)，ICLR 2020：事实、未来与反事实事件诊断；以及 [Kang 等的 ICML 2025 工作](https://proceedings.mlr.press/v267/kang25g.html)：受控 ID/OOD/组合泛化。

详细比较、主要 PDF 的 section/table 定位及 bibliographic 状态在 [参考说明](empirical_study_references.md)。

ICLR 2027 的公开检索遇到 API 403 与 forum/PDF browser verification，未能同时核实新投稿的身份、状态和方法。没有绕过或去匿名化；不把未核实的索引结果写成已发表或已接收论文。

借鉴方法，不复制结果或文章表达：

- WorldArena：prediction quality 与 achieved utility 分离；本项目补同案例、同候选、同 planner 的归因。
- CLEVRER：事实与干预分别计分，报告同一对案例的联合正确性。
- Kang 等：分开物理参数分布、组合变化与保持物理不变的视觉变化。
- SimLM/CueTip：作为强最近邻 baseline，不把五个 cue 控制或 simulator loop 当新贡献。

### 13.3 当前代码能做什么、缺什么

以下是只读核查得到的事实，不是待实现组件已完成：

| 现有表面 | 可复用部分 | 必须补齐的实际缺口 |
| --- | --- | --- |
| [schemas.py](../../src/inverser/schemas.py) | Layout、CueAction、PhysicsConfig、开发真值/公开视图 | TaskInput 总包含 goal 与实际 PhysicsConfig，不能直接用于盲测正向或未知物理；需要新 study view |
| [physics.py](../../src/inverser/physics.py) | 固定 solver、显式 cue 参数、结束/finite/cap 拒绝、rollout counter | Replay 是 event-sampled history；需要物理正确的 regular-time labels 与语义事件映射 |
| [metrics.py](../../src/inverser/metrics.py) | scratch、inventory/status、红球最小瓶颈最大误差 | 补预测、事件、干预、候选排序、regret、cost 与 grouped statistics，不替换主成功定义 |
| [preview.py](../../src/inverser/preview.py) | PIL 渲染与球型配色 | 现有图是低分辨率 montage，x/y scale 分开；需要等比例、物理几何一致的正式 paired renderer |
| [real_data.py](../../src/inverser/real_data.py) | 观察、缺失、来源与置信度边界 | 真实 action/omega 为 None；legacy acceptance 为 Literal False，不在原记录中悄悄改成 GT |
| [author_sync.py](../../src/inverser/author_sync.py) | 去重、增量复用与 native raster 审计 | 视频到 manual canvas 的独立映射与我们自己的路径评分仍需验证 |
| [corpus_catalog.py](../../src/inverser/corpus_catalog.py) | 分角色引用和 label guard | 目录不是任务验收；实验 loader 要按 capability 而不是总行数取样 |
| [cli.py](../../src/inverser/cli.py) | CLI 路由、当前处理命令 | 没有模型 client、baseline runner 或正式 study 命令 |

当前 CueAction 的准确自由度：

- `0 <= azimuth_deg < 360`；
- `0 < cue_speed_mps <= 3`，指 cue-stick speed；
- `tip_side`、`tip_vertical` 各在 `[-0.5,0.5]`，同时满足半径 0.5 的接触 disk；
- `elevation_deg == 0`，不是首轮还可以优化的第五维；
- tip offset 不是实测 angular velocity。初始 cue-ball 的 v/omega 可作为单独 privileged 诊断，不混入主输入。

保留 pilot 现有约束与旧文件。不能仅增大 `MAX_PILOT_TASKS` 或改旧 partition 来绕过 G1、制造盲测 test。

### 13.4 三个论文 RQ 与可证伪假设

| RQ | 主比较 | 指标 | 可推翻的解释 |
| --- | --- | --- | --- |
| RQ1：原生预测与干预一致性 | 同状态、动作/参数干预；state vs BEV；合法 nuisance；必要时加 oracle-kinematic 诊断 | moving/static 分列的位置误差、接触/落袋事件、顺序、干预响应、联合正确性 | 模型忽略动作；改进只来自额外 v/omega；外观/输入顺序改变结果超过物理干预 |
| RQ2：预测如何变成规划效用，工具增益在哪 | blind candidate forecasting、直接选择、共同 planner；native/self-reflect/genuine/shuffled feedback；几何数值强对照 | bank 排序与 regret、最终提交 success、optimism gap、calls/tokens/rollouts 与成本 | 更准预测不改善选择；收益由更多计算或真值工具解释；简单 selector/几何优化已同样有效 |
| RQ3：误观测与未知物理怎样泛化 | oracle/noisy 坐标、图像；known/nominal-shift/support-adapted physics；支持/查询分离；单独真实观察验证 | held-layout/held-parameter 差值、校准后的查询效用、真实空间兼容性及 failure categories | 支持拟合不能转到新目标；数据/反馈泄漏；宽泛假设集合覆盖高但不 sharp |

所有 hypotheses 都可能得到 null/negative result。暂不写 Finding，不用“LLM 完全不懂物理”概括被测条件的失败。

### 13.5 数据与划分：小型证据核心先行

已有来源仍是：

- 主作者：100 个内容去重 snooker 录制、37,876 raw frame、16 个源人工空间路径参考。
- 比赛：7,114 个弱 raw 候选；优先修 370 条队列，不先重新全库扫描。
- 辅助：blackball/carom 各 100 个来源录制，不并入主 snooker；当前不声称全部转换。
- 2,000 个模拟开发 shot；可做开发 fixtures，不能重新称为 hidden test。

当前真实实测 cue/spin 为零，完整物理任务验收为零；人工 raster 是空间 union，不提供 event order、temporal ADE 或实际动作。

数据产品：

1. 开发 fixtures：复用现有开发来源，验证合同、渲染、GT sampler 和成本记录。
2. 小型 controlled panel：建议设计上限 64 个独特布局 family，覆盖 free roll、ball contact、cushion 与复合/敏感场景；实际运行 N 按确认 quota 缩小。
3. 每布局一个可达 goal 和动作干预对；same-family 的 twins、views 与多个候选不是更多独立布局。
4. Core 先保持当前 1-4 球的可复现工程范围；彩球、密集完整 inventory 和更多接触层次作为明确分层扩展，不能说 pilot 已覆盖完整专业 snooker。
5. 正式 evaluation 使用新、未用于开发的布局与 seed，分组锁定；实际大小由 pilot 的方差、目标统计精度与资源决定，不预承诺固定大规模。

所有普通原件/Manual、near duplicate/replay、同布局扰动、calibration support 都保持 family 分组。作者同一桌/相机的条件不称多桌泛化。真实子集若无法独立对齐或确认用途，保留 blocked/weak 并呈现，不补造 GT。

### 13.6 任务与输出合同

#### A. Blind forward prediction

输入仅 initial、指定 action、公共 geometry/已声明 physics 和预测采样时刻；没有 goal、future frame、生成 family 或 witness 标签。

cue-space 条件还应向所有方法声明相同的 cue specs、impact convention
和实际 table/pocket geometry。当前 PhysicsConfig 不单独包含这些完整规格；
不能把未提供的引擎特定 cue-to-motion 映射当成模型动力学失败。

输出为指定球初始 slot 的未来位置/status 与语义事件。正规时刻标签由物理演化产生；跨碰撞不能简单线性插值。接触时刻的 state 约定、事件 tie-group 和容差预先声明。

Forward 跟踪初始可见 slot，内部映射到真值球；inverse 的红球目标仍按可交换集合评分。不要把不同端点的红球 ID 当成历史身份。

若 cue 参数到 v/omega 的差异是瓶颈，可提供 post-impact oracle-kinematic 对照；它是额外信息的诊断上界，不当成原生输入改进。

#### B. Goal-conditioned action

输入 initial + desired terminal、public physics 条件、no-scratch 与 endpoint tolerance。两模态的区别仅在 geometry 表达：state 给坐标，BEV 给图像/坐标轴/尺寸，不在 alt text、文件名或 metadata 中塞坐标。

模型返回单一最终动作，沿用上述 CueAction 语义；numeric validation 与规范化规则统一、显式、可测试。主 evaluator 不从未计费候选中替模型挑最佳动作。

主成功默认沿用 `tolerance_radii=1.0`；`0.25r/0.5r` 为预注册敏感性曲线，不看成绩后改主阈值。有效不同路线、不同 spin 不因偏离 witness 被判错。

#### C. Fixed candidate bank

共同 bank 的候选在所有方法间完全一致，顺序随机化。建议核心 K=8，扩展 K=16；真实取值在 quota/protocol freeze 时确定。

bank 从声明的、独立于 goal 的动作 prior 采样，再选合法 witness 产生
goal（或用 bank 外 witness）。不要通过按 goal 定制的候选模式给
blind forecaster 暗示未来。No-spin action-space 对照使用各空间都有合法
witness 的共同目标；有 spin witness 不证明无 spin 不可达。

- 先对每个候选盲测预测，不让 forecaster 看 goal。
- 同一模型直接看到 goal 选择候选，以及由相同预测经确定性 selector 选择，分别测量。
- GT bank oracle 是有限候选上界，不是全动作空间全局最优。
- bank 可以没有可行候选，但 goal 本身可有 bank 外 witness；“none in this pool”不等于不可达。
- 各候选的预测调用和长输出成本全部计入；batched 预测与单候选预测是不同协议，不混成免费一个 call。

#### D. Real factual observations

分开 prefix-only factual path prediction 与 endpoint-conditioned reconstruction。后者看到 goal，不叫 blind forward。真实 cue 动作、逐时刻人工轨迹与 spin accuracy 不评分。

人工图仅做已核查 canvas 上的 IoU/对称空间距离。报告 static/prefix-only baseline；不能从 path union 声称未来-only ADE 或 event chronology。多假设报告数量/面积，不能靠涂满整桌取高 coverage。

### 13.7 逐步实验矩阵，不做全笛卡尔积

| 编号 | 实验 | 起步范围 | 扩展条件 |
| --- | --- | --- | --- |
| E0 | schema/GT/renderer/mock-client/budget sanity | 无模型调用；开发 fixtures | 必须先通过 |
| E1 | no-tool forward + direct goal | state/BEV 同案例；known physics；原生提交 | 部署和 G1/G2 条件确认 |
| E2 | action/physics intervention 与 nuisance | E1 的配对布局；盲测预测、联合一致性 | 不以 nuisance action 参数差惩罚合法多解 |
| E3 | prediction-to-ranking-to-success | 共同 bank；blind forecast 与 direct selection；同 selector/planner | 配对子集、K 与 API 额度足够 |
| E4 | feedback attribution 与自由提案 | native/self-reflection/genuine/shuffled；强几何/数值基线 | 相同 turns/证据/rollout budgets，分别计成本 |
| E5 | observation/physics shift | 同 panel 的 noisy geometry、nominal shift、独立 support adaptation | 真值状态/参数不被 tool 暗中提供 |
| E6 | held-out real factual compatibility | 16 个源 snooker manual refs 可用部分；单独来源分组 | native mapping 与标签能力先验收 |

E3/E4 是扩展诊断，不要求每个模型在起步时跑完。E5 中经典参数拟合不是新神经模型训练；是否运行在已有支持/资源条件下另定。新 GNN/dynamics、LoRA/RL 不在这轮默认范围。

### 13.8 Baseline 与模型选择

必要 non-LLM baseline：

- random/low-discrepancy proposals；
- geometry-initialized、预算受控的 multi-start 数值搜索；
- 共同 bank 中 best-revealed candidate 的简单 selector；
- 若适用，free-roll 的声明假设下解析 sanity，不称 native LLM。

LLM 系统：

- direct action 与 blind prediction；
- self-refinement without new physical evidence；
- SimLM-style numerical feedback；
- CueTip-style event-then-optimize 的任务适配，不声称原论文复现。

先利用已有 SciPy 和 geometry 共用 bottom optimizer，开发阶段选择/固定具体方法；计算 DE/annealing 等真实 function evaluations，不用 optimizer iteration 冒充 query 数。需要新直接依赖时先改 manifest，再安装。没有新神经训练作为必选 baseline。

本地实际核查的候选：

| 路径 | config 中的 architecture | 使用意图 |
| --- | --- | --- |
| `Qwen3.6-27B` | Qwen3_5ForConditionalGeneration | 原指定主本地模型，资源确认后验证 |
| `Qwen2.5-VL-7B-Instruct` | Qwen2_5_VLForConditionalGeneration | 小模型/较早代际对照，不作为纯 size ablation |
| `Qwen3-VL-32B-Instruct` | Qwen3VLForConditionalGeneration | 可选代际/架构对照 |
| `GLM-4.6V` | Glm4vMoeForConditionalGeneration | 可选跨家族，先验证 checkpoint 与容量 |
| `Qwen3.5-122B-A10B` | Qwen3_5MoeForConditionalGeneration | 大 MoE 扩展，不把 active parameters 当 resident memory |
| `Intern-S2-Preview` | InternS2PreviewForConditionalGeneration | 可选特殊架构，native/config/tokenizer 兼容性先验证 |

官方 vLLM main registry 可找到六个 architecture；这不证明已安装、具体 checkpoint 已完整、GPU 可容纳、稳定 release 兼容或产生正确图像推理。不能直接把六个目录称为六个可跑模型。

规划时所选的编辑器解释器并非 project `.venv`。两个已检查环境均为 Python 3.12.3，metadata 中没有 torch/vllm/transformers；这不代表机器其他环境没有。project `.venv` 有 Pooltool 0.6.0/httpx 0.28.1，另一个开发环境有 OpenAI 3.15.0/httpx 0.28.1。未加载模型或初始化 CUDA。

默认模型矩阵是“主本地 + 资源允许的对照 + 实际可用 API”，不是必须跑所有目录。闭源原请求的部署需核实真实 model ID/image support；不给不存在的部署换别名。

### 13.9 API 与 vLLM 两条执行路径

采用一个 typed OpenAI-compatible HTTP client，配置 backend alias、model ID、capabilities 和 runtime-only authentication。provider 差异显式记录；不复制另一项目的密钥、endpoint secret 或运行状态。

四种资源情形：

| 可用资源 | 可以推进 | 不应声称 |
| --- | --- | --- |
| 尚无确认模型资源 | code、mock tests、数据审计与合同准备 | 已开始 inference 或部署 |
| 只有确认 API quota | 合成 state/BEV 与 CPU 参考；闭源配对小 panel | 已完成本地对照 |
| 只有确认 GPU | 只读现有 weights 的 vLLM 本地路线 | 已完成 API leaderboard |
| 两者都有 | 相同 common panel 的比较；更大本地子集单列 | 把不同样本量直接混成公平排行 |

vLLM 路线：

1. 优先复用用户指定兼容环境；否则批准后建立独立 inference environment，不覆盖 physics `.venv`、tactic 或系统 driver。
2. 固定 vLLM/torch/transformers、model/config/tokenizer revision、dtype、quantization、TP 与 chat template；核查 indexed shards。
3. 显式限制 context/image count/输出与并发；读取 checkpoint 的 generation_config 不作为隐藏 sampling 默认。
4. 在确认 GPU UUID 与端口后启动 session-attached server，只绑定 loopback；不自动 detach，不停止他人任务。
5. 验证 health、`/v1/models`、实际返回身份、结构化状态、单图和 initial/goal 双图语义，再扩大 batch。
6. 记录实际 processor resize、vision tokens（若可得）、prompt/response hash、usage 与失败；未提供信息记 unknown，不填零。
7. 不自动启用 trust_remote_code，不静默换 Transformers backend、量化 checkpoint 或模型。兼容性失败明确暂停。

首轮科学主比较使用相同普通 JSON 指令与 parser；provider-specific grammar 为独立工程消融。若使用 structured decoding，按 pinned 版本采用官方 `structured_outputs`/`response_format`，不沿用已移除的 guided_* 字段，也不声称 grammar 保证物理正确。

API 路线：

- 只有 synthetic payload；没有真实源图、manual annotation、源码或凭证外发。
- 每 model 的 probe/请求/重试/loops 都受明确 quota；无 quota 不调用。
- 核实 vision、model ID、usage、reasoning/sampling 可控性；未提供的内部信息不作假设。
- 无自动模型替换或“图片不支持就改为文字”。
- 丢失响应但可能已计费的请求标 unknown-response/billing-uncertain；无 provider idempotency 时不承诺 exactly-once，不自动重发掩盖风险。

### 13.10 成本、统计与信息公平

计账至少分开：

- 请求尝试、成功回复、未知回复、token/usage、quota 消耗；
- agent-visible 物理证据、实际 physics evaluations、cache hits；
- nominal/fitted model simulator 与隐藏 reference；
- offline GT/render/参数拟合、online inference/优化、事后 judge；
- 时间与硬件为实际测量指标，不给项目执行时长预测。

Core native panel 若每布局有两个 forward 动作与一个 goal，两个输入模态，需要 `6*N` 个模型请求，不含 probes/repairs。Bank blind forecasts 另需按 `N_bank*K*views` 计费，feedback 的每个 turn 继续加算。提前由 quota 推导可完成的共同 N，不能用“100 次请求”当成“100 个多轮 episode”。

物理预算建议核心 B=0/8/32，128 为可选扩展。同配置比较相同可见信息和 bottom optimizer；population evaluations、ensemble、参数校准都计数。

在 noisy/image 条件下，planning tool 默认使用已披露的 estimated state 与 nominal/fitted parameters。若它使用隐藏 true state/parameters，是 oracle-feedback 上界，不能当普通 perception/unknown-physics 改进。

噪声可能造成 overlap、越界或不完整的 estimated state。不得静默 clamp、
丢球或补齐；合法性检查与任何声明的 state repair 单独记录、计成本并消融。
无法构造有效估计时给明确失败原因，而不是让 tool 偷用 true state。

归因先做相同 bank 和相同证据，再做自由提案；不能只匹配 query 总数却给某方法更高质量信息。Shuffled feedback 是明确配置的合成干预，不是引擎错误被偷偷忽略。

统计：

- independent layout/source family 是单位，views/twins 不增加独立 N；
- 预注册 paired differences、layout/source-clustered CI；search 的 stochastic seeds 独立记录；
- 小 API panel 报 pilot 与不确定性，不承诺可辨别所有小效应；
- compare common panel，扩展本地面板单列；不按结果挑选给 API 的简单案例；
- performance 一律不排除格式/非法动作失败后只报好样本；backend unsupported/未执行 case 与已提交模型失败分别报告；
- bank success-regret 与候选 cost-regret 明确定义，cost 的状态/距离 penalty 先固定并测试；不是全局最优证明；
- 预测质量与 utility 的相关性不是因果结论，配合信息替换、同 planner 与配对干预解释。

### 13.11 实现表面与最小改动策略

批准后按需求逐批实现，不提前创建一个大框架：

| 表面 | 拟议改动 | 验收点 |
| --- | --- | --- |
| 新 `src/inverser/study.py` | case/view/capability/partition 合同、paired bank 与 manifest | forward 无 goal；hidden-physics 无真实系数；旧 pilot 只用于 dev |
| [preview.py](../../src/inverser/preview.py) | 抽取 reusable metric renderer，保留已有 preview 行为 | 相同物理 scale、轴/尺寸/球/pocket geometry；无 witness/label 泄漏 |
| [physics.py](../../src/inverser/physics.py) | 最小扩展物理采样/语义事件，保留 replay 与 rejection 语义 | 碰撞前后、重复事件时间、finite/cap 检验；旧 witness 回归 |
| 新 `src/inverser/model_adapters.py` | 一个 typed transport 与显式 provider capability | mock response/errors/usage/model mismatch；secret-free logs |
| 新 `src/inverser/evaluation.py` | shared baselines、budget ledger、job resume、最终提交与统计 | optimizer 内部调用都计账；格式失败/unknown response 明确；无隐藏 best-of-K |
| [metrics.py](../../src/inverser/metrics.py) | 增加诊断，保留 score_endpoint | 红球交换、status/scratch、moving/static 与 rank/regret 的反例测试 |
| [vision.py](../../src/inverser/vision.py)及[author_sync.py](../../src/inverser/author_sync.py) | 定向重处理与固定 canvas 映射验证 | 原始数据不改；无假时间/旋转/落袋标签 |
| [cli.py](../../src/inverser/cli.py)、manifest/tests | 逐阶段 prepare/dry-run/run/report 路由及显式可选依赖 | 这些命令是拟议的，不表示现在已可调用 |

新增依赖先写 manifest；server 环境与 physics 隔离。不因为 transitive package 已存在就不声明直接依赖。不修改本项目以外的环境或仓库。

### 13.12 顺序、并行准备与研究者 gate

用阶段与验收条件推进，不给 calendar 或执行时长估计：

| 阶段 | 交付物 | 前置条件/停止点 |
| --- | --- | --- |
| P0：计划与资源 | 当前计划、参考来源、模型/数据边界；实际 quota 配置 | 当前计划审批；实际 resource 另确认 |
| P1：代码与数据质量准备 | study views/metric renderer、mock client/ledger；100 author +370 targeted repair 的独立 QC | 两条准备线可独立推进；不运行模型；G1 未自动通过 |
| P2：backend smoke | 已分配设备/额度下的身份、单位、JSON、图像支持与失败日志 | 资源确认，smoke 算额度；不是模型性能 Finding |
| P3：开发 panel/protocol | validated panel、独立 formal seeds、模板/指标/bounds/budgets 与版本 hashes | G1 接受实际质量/局限后，G2 冻结；不是必须先生成百万 |
| P4：inference-only 诊断 | E1/E2，再按 quota 加 E3/E4；E5/E6 条件满足才加 | 显式 development 探索，不使用 hidden-test 反馈调参 |
| P5：贡献选择 | 失败归因、null/positive evidence、是否需要小动力学/新 agent | G3：empirical-only 或机制方案；训练另审批 |
| P6：正式评价与文章 | 固定 checkpoint、prompt、测试、统计和可复现图表 | G4 与正式 quota；不保证中稿、不自动发布/投稿 |

若真实路径映射未能独立验证，提交 blocked 原因与可用标签；研究者决定是否接受合成开发诊断加暂缓真实 arm，不默默宣称已完成 real-first benchmark。

### 13.13 文章的结果驱动结构

工作题名仍可用 “InverseR: Measuring the prediction-planning gap in multimodal physical reasoning”；gap 需要由测量支持，不作为预设结果。

主文结构：

1. Introduction：一个可达目标的多解案例；不把历史路径当唯一正确；提出三 RQ。
2. Task and measurement design：四个自由控制、state/BEV pair、known/unknown 信息、native/tool 边界。
3. Construction and quality：来源、capability tiers、分组、GT sampler/renderer 与 rejected cases。
4. Empirical results：按 RQ 排列预测/事件、rank/goal utility、feedback、shift/real evidence；没有结果不写 Findings。
5. Discussion：信息不足、模型利用、预算、真实动作未测以及适用范围。

计划图表：

- Figure 1：同一 goal 的已发现可行路线与“预测正确/规划成功”不能互相替代的示意。
- Table 1：最近邻任务、动作/spin、可见信息、执行与真实标签能力。
- Figure 2：真实观察与受控仿真的分层构建及质量 gate。
- Main table：model × input × native/tool 的提交 success、误差、invalid 与成本。
- Diagnostic figures：candidate ranking/regret、prediction vs utility、feedback/shift 的配对变化。
- Real table：source-held spatial agreement、样本选择与 mapping 误差，不写真实 cue accuracy。

贡献以实际证据定稿：明确且可执行的测量、可审计的数据层、可复现的能力边界。新方法不是必须，只有证据与研究者决策支持才加入。

### 13.14 当前新增执行 todos

SQL 是状态与依赖的查询来源；下面不是勾选清单。

| ID | 内容 | 实际前置 |
| --- | --- | --- |
| gate-study-plan | 确认当前计划与可实现阶段 | empirical-execution-plan |
| study-resource-confirmation | 确认 GPU/API/CPU/storage 具体有限额度 | 可独立确认，不假定列表顺序 |
| study-view-contract | public/private/forward/unknown-physics 合同 | gate-study-plan、task-contract、physics-adapter |
| observation-core-repair | 100 固定相机与370队列、native mapping 与 QC | gate-study-plan、audit-final-author-data、full-video-collection、study-resource-confirmation（CPU/storage route） |
| study-labels-renderer | 物理时刻/事件与等比例 paired renderer | study-view-contract、physics-adapter |
| study-model-client | typed client/mock capability/usage/error tests | study-view-contract、project-bootstrap |
| study-budget-ledger | quota/calls/cache/resume 与最终提交审计 | study-view-contract、physics-adapter |

后续现有 `quality-pilot`、`benchmark-build`、`model-adapters`、`evaluation-harness` 与 G1-G4 的描述/依赖按上述阶段更新。原始未解决媒体许可/真实完整 GT 不被标为已解决；先完成已具备原始来源的 QC，不让过时的“还没有源文件”描述替代真实阻塞原因。

## Primary references used to ground this plan

- ARS pinned upstream: https://github.com/Imbad0202/academic-research-skills/tree/e614b322297ba5b072a106942177e218cbe2d48f
- ARS setup: https://github.com/Imbad0202/academic-research-skills/blob/e614b322297ba5b072a106942177e218cbe2d48f/docs/SETUP.md
- ARS channel limitations: https://github.com/Imbad0202/academic-research-skills/blob/e614b322297ba5b072a106942177e218cbe2d48f/docs/CONTROL_AVAILABILITY.md
- ARS data flows: https://github.com/Imbad0202/academic-research-skills/blob/e614b322297ba5b072a106942177e218cbe2d48f/docs/DATA_FLOWS.md
- VS Code skills: https://code.visualstudio.com/docs/agent-customization/agent-skills
- SimLM: https://arxiv.org/html/2312.14215v2
- CueTip: https://arxiv.org/html/2501.18291v2
- Robotic billiards endpoint control: https://doi.org/10.1109/TMECH.2015.2461547
- BilliardPhys-Bench: https://arxiv.org/html/2605.30900v1
- Real billiards source: https://github.com/FJ-Rodriguez-Lozano/Billiard-dataset
- Sports analytics: https://arxiv.org/html/2407.19686v1
- Sports analytics supplement: https://zhengwang125.github.io/paper/SDM_Supplementary.pdf
- Pooltool: https://pooltool.readthedocs.io/
- CVPR CFP: https://cvpr.thecvf.com/Conferences/2027/CallForPapers
- IJCAI homepage: https://2027.ijcai.org/
