# Incremental snooker uploads and reproducible experiment inputs

The source now contains both `snooker` and `Snooker_Manual`. They are handled as
two representations of recordings, not automatically as independent examples.
Raw data, existing blackball pilots and previous match-video outputs remain
untouched.

## Implemented adjustments

### 1. Check new inputs, not the entire uploaded dataset every time

```bash
.venv/bin/python -m inverser inventory-author-uploads \
  --source data/Billiard-dataset \
  --output data/quality/snooker-upload-snapshot-v3 \
  --games snooker --essential-only \
  --hash-cache .cache/author-upload-hashes.sqlite \
  --stable-seconds 120 --observe-seconds 5
```

The game filter covers both snooker folders. The essential-input filter keeps
original AVI/BGR captures, background images, manual paths, algorithm path
images, animation tracks and source text. Large tracking/reconstruction movies
and static 3D scene assets are not rehashed merely to ingest original shots.

An SQLite hash cache reuses SHA-256 only when path, size, mtime, ctime, device
and inode match. A same-size file replacement with a restored mtime invalidates
the cache. This is an operational optimisation, not a claim that every byte was
reread on each scan. Changed or recently uploaded files remain deferred.

The original immutable snapshot workflow still works without a cache. Each new
scan writes its own input manifest; nothing is silently added to an old snapshot.

### 2. Deduplicate recordings and retain manual annotation associations

Recordings are grouped by identical complete video or raw-capture hashes.
Directory indices are not sufficient evidence of equivalence.

For example, the supplied `Snooker_0/BGR.avi` and
`Snooker_Manual_0/BGR.avi` share their complete video hash and raw capture.
They constitute one recording with multiple source paths. The planner prefers
a stable manual-labelled representation, retaining every alias and its
original manual path image.

Conflicting raw captures attached to the same video hash are deferred for
inspection, not silently merged. A changing/missing raw capture or background
also defers conversion instead of falling back to a possibly incomplete AVI.

Every group has one content identity for experiment indexing. An original clip
and its manual copy must not appear in different training/test partitions.

### 3. Reuse complete conversions and version newly supplied labels

```bash
.venv/bin/python -m inverser sync-snooker-uploads \
  --inventory data/quality/snooker-upload-snapshot-v3/inventory.json \
  --output data/structured/author-snooker-incremental-v1 \
  --hash-cache .cache/author-upload-hashes.sqlite \
  --max-new-clips 100
```

The synchronizer processes only stable unique recordings. It fingerprints the
source video/raw capture, background, linked annotations and conversion code.
Previously completed identical inputs are reused after source checks.
Failed/interrupted inputs get a new attempt directory; the old attempt remains.

A later manual annotation changes that recording's input version. It creates a
new conversion version and snapshot index, not a second independent shot or a
modification of the previous result. A pipeline-code change requires a new
pipeline output directory.

One invocation is a finite batch, not an unbounded watcher. Newly arriving files
are picked up by the next inventory/sync invocation. There is no background
training, automatic publication or model API use.

```text
data/structured/author-snooker-incremental-v1/
  pipeline_config.json
  sync.sqlite
  latest.json
  records/<input-fingerprint>/attempt_01/
    structured.json
    endpoints.json
    observations.jsonl.gz
    report.json
    source/BEV images and algorithm_tracks/
  snapshots/<inventory-hash>/
    recording_plan.json
    episodes.jsonl
    episode_index.csv
    manual_path_audit.json
    manual_path_scores.csv
    summary.json
```

The experiment index contains one current record per source group, linked
manual-path counts and source aliases. It does not assign a frozen train/test
split or declare the records ground truth.

## What the newly available manual labels can measure

The source manual paths are aggregate raster images. In the inspected case they
have a 571x300 canvas, not the 640x480 source-video frame geometry.

The audit compares manual and provided algorithm rasters only on equal native
canvases, at a fixed grayscale threshold of 128. It performs no fitted alignment,
resizing or temporal interpolation. Mismatched canvases are reported explicitly.

The current 16 manual images support 160 comparisons across ten provided
algorithms. The reproduced Jaccard values match the author's stored values to
within `1e-5`. This verifies the path-image metric interpretation, not our
pipeline's detection accuracy or an inverse model's performance.

These labels do **not** directly supply:

- frame-wise identity/time ground truth in the original video;
- measured cue impact speed, tip offsets, force or spin;
- a verified metric-coordinate mapping into the current BEV projection.

The recovered algorithm X3D paths remain separate, with their scene units and
animation clocks flagged as unverified. Missing or occluded balls are not
automatically marked pocketed.

## Measured ingestion optimisation

The first essential snooker snapshot hashed 1,352 stable files, reading
24,247,318,337 bytes. A subsequent snapshot reused those 1,352 hashes and read
only 656,074,578 bytes from 39 newly arrived/changed files. More files arrived
between the two scans, so these are observed read counts, not a same-input
wall-clock speed benchmark.

At the subsequent snapshot there were 65 original video files, 63 stable, and
16 manual-path images. Exact-content grouping and readiness determine the
distinct converted count; do not report 65 as independent ground-truth shots.

That snapshot grouped into 57 recording identities, with 55 ready for
conversion and two deferred upload groups. All 55 were converted and validated:
21,265 complete raw frames, 16 records linked to original manual paths, and no
conversion failures. Five previous conversions were reused and fifty new ones
processed. Re-running the identical snapshot produced **zero new conversions
and 55 reuses**. The first snapshot and earlier conversion versions remain.

The [latest snapshot pointer](../../data/structured/author-snooker-incremental-v1/latest.json)
locates the current episode index and native manual-path audit. Each snapshot
index is one view of the currently known recording groups, not a declaration
that ongoing uploads are complete.

All 55 still carry `complete_inventory_unknown`, and some have unresolved
components or identity changes. The pipeline has verified conversion and
provenance, not complete table labels or inverse-action accuracy.

## Recommended experiment adjustments

| Experiment | Appropriate input and control |
| --- | --- |
| Extraction development | Keep manual-path imagery as an independent path reference; first verify the video-to-raster mapping before measuring our own path IoU |
| Evidence/ambiguity RQ | Use separate endpoint-only and privileged-trajectory views; do not leak source manual paths or algorithm outputs into the default inverse input |
| Search RQ | Use the same source-specific geometry and uncertainty for all optimizers/models; do not silently substitute regulation dimensions for the experimental table |
| Real transfer | Separate the small fixed-camera author domain from broadcast videos; report each domain before pooling |
| Physical calibration | Fit shared table parameters on support recordings and evaluate different query recordings; no per-test trajectory tuning |
| Scale effects | Count unique physical recordings and distinct simulated tasks separately; source aliases, frames and methods do not increase sample size |

The manual subset has already been used for format and source-metric checks.
Any later tuning against it must be disclosed; it is not automatically an
untouched final test set. Split by source/content group, and consider a
table/camera-domain holdout for transfer claims.

The next core bottleneck remains detector/track completeness. Schema-valid
records with linked manual files are not automatically accepted physical
benchmark tasks. The existing data-quality, benchmark-freeze and formal
evaluation approvals remain in effect.

## Validation

Regression tests cover cached versus fresh hashing, same-size replacement,
game filtering, unfinished raw companions, conflicting aliases, ordinary/manual
deduplication, late annotation arrivals, immutable old indices, reuse of complete
conversions, endpoint-view isolation and native manual-canvas mismatches.
Existing Ruff, Mypy and Pytest validation are preserved.

Validation at that snapshot: 100 tests, Ruff and Mypy passed. All 55 record/media/source
hash checks and 160 source Jaccard comparisons passed their declared checks.
No training, API comparison or physical execution was performed.

## Completed-upload follow-up

The final author snapshot contains 116 snooker video copies, grouped by video
and raw-capture hashes into exactly 100 independent recordings. All are stable,
ready and converted; 16 retain manual-path references. The full conversion uses
37,876 complete raw frames, reuses 55 prior conversions and adds 45.

Use `latest.json` for the final snapshot instead of treating the earlier
65-file snapshot as the final data size. Dataset-label quality remains a
separate acceptance question. The [cross-source catalog](../data/research_corpus_catalog.md)
links these records to match-video candidates and simulator truth.
