# Local real-video quality pilot

This stage processes source media locally. It does not train a model, call an API
or turn automatic observations into measured action ground truth.

## Source and pilot scope

The verified downloaded snapshot is at `data/snookerVideos/files/`. The
source inventory and checksums are preserved in `data/snookerVideos/`.

The first quality run uses ten-minute windows from `match12.mp4`, `match13.mp4`
and `match29.mp4`, starting at 5% of each recording's duration. These were chosen
for existing corner sidecars and differing resolutions, not for presumed success.
At 5 sampled frames per second this covers 9,000 sampled frames, not the whole
228-hour corpus.

The first full pilot completed with 9,000 sampled frames and 32 candidate clips.
All exported observation streams and source/configuration hashes were checked.
The quality audit prioritised 22 nontrivial cue-endpoint-change candidates; none
was promoted to independently verified single-shot ground truth. All 32 retain
unresolved endpoint/identity flags. See the generated `quality_review.json`.

```bash
.venv/bin/python -m pip install -e '.[processing]'
.venv/bin/python -m inverser process-video-pilot \
  --source data/snookerVideos --output data/processed/snooker-quality-v0 \
  --videos match12.mp4 match13.mp4 match29.mp4 \
  --duration 600 --sample-hz 5 --max-candidates 12
.venv/bin/python -m inverser audit-video-pilot \
  --output data/processed/snooker-quality-v0
```

The CLI limits one pilot to three videos, 900 seconds per video and 50 exported
candidates per video. The initial run exports at most twelve per video.
Use a new output version when source/configuration changes. A complete unchanged
per-video run can be reused; an incomplete run is reported and not silently
appended to.

## Implemented processing

1. Validate the source's manifest, byte size and corner sidecar.
2. Build a planar homography using the supplied corner order and nominal metric
   table dimensions. Check corner geometry and frame bounds.
3. Detect candidate balls in the native source image with perspective-dependent
   radius, color and shape evidence. Do not assume the silhouette of a sphere
   remains circular after a table-plane warp.
4. Project detected centers to table coordinates. No lens or ball-center-height
   correction is claimed.
5. Associate observed detections using class-compatible, speed-bounded matching.
   Lost detections are not interpolated into fake observations or labelled
   pocketed. IDs are local estimated tracking identities.
6. Propose stationary-motion-stationary windows using cue and matched-motion
   evidence. Short visibility gaps are retained as quality warnings, not hidden.
7. Save source-relative clips, endpoint images and actual sampled track points.
8. Audit timestamp/trajectory consistency, clip decoding and unresolved quality
   flags. Prioritise nontrivial cue endpoint changes for subsequent review.

The state-machine proposals are not verified cue strikes. A small endpoint
change may be a real return shot, a player's preparation, a detector error or
an unrelated intervention. Color errors, touching red clusters, cue occlusion
and weak camera calibration remain blockers to complete table inventories.

## Outputs

```text
data/processed/snooker-quality-v0/
  summary.json
  candidate_index.csv
  quality_review.json
  candidate_quality.csv
  pilot_overview.jpg
  match13/
    config.json
    processing_state.json
    report.json
    calibration_preview.jpg
    frame_observations.jsonl
    candidates/candidate_0000/
      observations.json
      clip.mp4
      clip_reference.json
      initial_source.jpg
      goal_source.jpg
      endpoint_pair.jpg
```

Video folders mirror the same structure. Configuration records include code,
source and sidecar hashes plus calibration conventions.

Every candidate is `silver_unverified`, development-only and explicitly
`benchmark_accepted: false`. Full ball inventory and exactly one cue strike
are not verified. Shape/color scores are heuristics, not calibrated posterior
probabilities. Timestamps use the sample grid, not observed cue-impact instants.
No cue force, tip-contact offset or angular velocity is assigned.

The audit's nontrivial-motion count is an inspection priority, not an accuracy
or accepted-benchmark count. Earlier development failures remain in their own
smoke-run directories rather than being overwritten.

## Upload-safe OneDrive archive import

The researcher-provided `data/OneDrive_1_2026-10-10.zip` is still an upload until
both its size/mtime are stable and its ZIP directory is complete.

```bash
.venv/bin/python -m inverser wait-import-archive \
  --archive data/OneDrive_1_2026-10-10.zip \
  --output data/Billiard-dataset \
  --stable-seconds 120 --max-wait-seconds 21600
```

The attached monitor waits at most six hours, writes its status and imports only
after two minutes of stability and a valid ZIP directory. Do not start a second
monitor while the existing process holds its lock.

Extraction preserves the ZIP and original member filenames. It rejects unsafe
paths, symlinks, encrypted entries and duplicate/file-directory conflicts.
Files are published only after size/CRC verification; SHA-256 receipts are
recorded. Existing extracted members must match their CRC and size before reuse.
The project storage and shared-volume free-space guard applies to the declared
uncompressed size.

```text
data/Billiard-dataset/
  _import/
    upload_state.json
    source_manifest.json
    extraction_state.json
    inventory.json
  files/                    # Original archive hierarchy
```

Until the monitor finishes, no dataset contents or shot counts are assumed.
After completion, inspect actual game folders and manual annotation formats.
An original `manual_track.png` is a source path image, not automatically a
timestamped numerical trajectory or measured spin label.

## Validation

Tests cover native-image sphere detection, red hue wrap, metric axes,
non-invented missing tracks, moving-object stop conditions, non-finite
observations, upload completeness, archive path safety, preservation/resume
and CRC failures. Run the existing Ruff, Mypy and Pytest checks in the isolated
project environment.

Current validation: 73 Pytest cases, Ruff, Mypy and dependency consistency checks
passed. This does not substitute for independent real-label accuracy measurements.

## Complete-recording collection

The researcher subsequently authorised collection from the complete video
corpus. This uses a separate versioned output rather than removing pilot
limits or overwriting the quality panel.

```bash
.venv/bin/python -m inverser collect-video-corpus \
  --source data/snookerVideos --output data/structured/snooker-corpus-v1 \
  --workers 4 --sample-hz 5
```

The plan covers all 32 recordings (228.38 source hours) using 1,387
ten-minute ownership intervals and 90-second context on each side where
available. A shot is exported only by its onset's owning interval.
The decoder uses at most 960-pixel width for the raw pass; native source
files remain untouched. This is coarse acquisition, not high-speed spin
measurement.

Supplied corner sidecars are used when they cover a cloth-like table region.
Otherwise a convex felt quadrilateral is estimated. Clipped quadrilaterals are
rejected, and perspective trapezoids are not treated as horizontal overhead
rectangles. Re-estimation and missing views are recorded, not filled with
invented coordinates.

Each worker uses bounded FFmpeg/OpenCV resources. Four workers, the single
archive importer and its single source-processing consumer stay within the
approved CPU allocation. No GPU or external model is used.

```text
data/structured/snooker-corpus-v1/
  collection_manifest.json
  collection_state.json
  collection.sqlite
  episodes.jsonl
  episode_index.csv
  windows/<recording>/<interval>/attempt_01/
    job.json
    report.json
    frame_observations.jsonl.gz
    candidates/<candidate>/
      structured.json
      endpoints.json
      clean_table_pair.png
      clip.mp4
      ...
```

The SQLite ledger and state file are live. Consolidated exports checkpoint
periodically and at completion. Interrupted jobs get a new attempt directory;
completed jobs are not rerun. Errors, source changes, missing views and export
caps remain visible. The corpus remains raw candidates until quality review.

### Failed-window recovery

The first full collection completed 1,375 windows and left twelve failed:
a 4x4 image component with a 1.166-pixel nominal radius had no pixel center
inside its subpixel color core. The original detector incorrectly aborted the
whole window. The corrected detector flags and logs this under-resolved component
instead of assigning it a fabricated color or suppressing the quality problem.

The original manifest, database, successful episodes and failed outputs are
preserved. Retry only the failed windows into a separate version:

```bash
.venv/bin/python -m inverser recover-video-corpus \
  --source data/snookerVideos \
  --parent data/structured/snooker-corpus-v1 \
  --output data/structured/snooker-corpus-v1-recovery --workers 4
```

The recovery ledger inherits completed records by reference without rerunning
them. Its recovery manifest explicitly distinguishes old and new code hashes.
The new consolidated index is a composite of inherited and recovered results,
not a claim that every frame was rerun with the patched code. Unknown labels
and benchmark nonacceptance remain unchanged.

Recovery completed: all 1,387 windows are complete and 7,114 raw episode records
are indexed. The full record validator, export-count check and existence checks
for all referenced endpoint/BEV/clip assets passed. No model experiment or
independent physical-label validation is implied.

## Author-archive processing after import

```bash
.venv/bin/python -m inverser process-author-dataset \
  --source data/Billiard-dataset --output data/structured/billiard-v1 \
  --wait-ready-seconds 21600
```

This consumer waits for the existing safe importer to finish; it does not
start another ZIP extraction. It selects original snooker `BGR.avi` clips,
excludes tracking/reconstruction videos and deduplicates exact original-file
hashes. Background images can seed table localisation.

The original monitor and consumer both exhausted their six-hour waits without
extraction. The current archive is 21,407,354,487 bytes, with neither ZIP nor
ZIP64 footer present, and was unchanged for approximately 25 hours at the latest
check. Do not restart another long wait or force partial extraction merely
because the file is large. Check the original ZIP and provide a complete archive
before restarting these commands.

Both adapters follow the [common BEV schema](../data/real_data_schema.md). In particular,
the author paper's small experimental-table dimensions and 51 mm balls stay
distinct from regulation dimensions. Original path PNGs and X3D results are
references, not automatically measured action labels.

## Direct-folder uploads without a ZIP

Use the [direct-upload adapter](author_direct_uploads.md) when source folders
are arriving directly under `data/Billiard-dataset`. It takes an immutable
stable-file snapshot, classifies original versus generated assets, and converts
a bounded per-game pilot without waiting for `_import/inventory.json`.

Blackball records are auxiliary and use a separate strict schema. Raw BGR
frames and original AVI clocks are explicitly distinguished; reconstructed X3D
positions keep source-native coordinates and animation times until alignment
is verified. No existing snooker output is relabelled or overwritten.
