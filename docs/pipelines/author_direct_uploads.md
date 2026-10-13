# Directly uploaded Billiard-dataset assets

The researcher is uploading original source folders to `data/Billiard-dataset`.
This ingestion path does not require a ZIP or the old `_import` ledger.
Uploads remain read-only; inventories and converted data are written elsewhere.

## Observed contents, not advertised totals

The immutable inventory at `data/quality/author-upload-inventory-v1/` was
completed at 2026-10-13 02:51:16 UTC:

- 79 original `BGR.avi` recordings, all in **blackball**.
- 77 original recordings met the age, unchanged-stat and checksum checks.
- 79 raw `BGR.dat` captures and one empty-table background image.
- 237 reconstructed animations, 237 reconstruction videos and 157 tracking
  comparison videos.
- No snooker original capture and no `manual_track.png` was observed.
- 3,157 total files, approximately 38.42 GB at snapshot time; 3,076 files were
  unchanged and hashed.

These numbers are a dated upload snapshot. Files arriving afterwards are not
silently added to it. Generated videos, algorithms and animation frames do not
increase independent-shot counts.

Blackball is auxiliary data under the approved snooker-first project scope.
It is not relabelled as snooker, and it does not increment main benchmark size.
Snooker uploads can use the same observation fields once that source arrives.

## Asset roles and uses

| Asset | Conversion | Intended use and limit |
| --- | --- | --- |
| `BGR.avi` | Probe resolution/clock, source playback and endpoint inspection | Independent source recording, not a cue-control label |
| `BGR.dat` | Read complete 8-bit BGR frames using checked geometry and declared capture rate | Avoid compression artifacts and bad rendered terminal frames; preserve timing uncertainty |
| `background.png` | Estimate cloth ROI and a planar table transform | Detection/calibration diagnostics; not per-recording lens or metric calibration |
| `manual_track.png` | Keep original path image and provenance | When supplied, path-level independent checks; not automatically timestamped trajectories |
| `TrackingResults/*_track.png` | Keep per-algorithm output reference | Compare extraction failure patterns, not measured GT |
| `AnimationsResults/animation_*.x3d` | Parse `PositionInterpolator`, `TimeSensor` and clock routes into separate numeric files | Candidate weak supervision after coordinate/time verification |
| Tracking/reconstruction AVI | Preserve as derived evidence, not duplicate original capture | Diagnostic rendering only |

The source publication is the supplied
[Applied Intelligence paper](../papers/s10489-023-04542-3.pdf). Its small
experimental table and 57 mm blackball ball diameter are recorded separately
from 51 mm experimental snooker balls and regulation snooker geometry.
The 1.854x0.935 m values are the paper's baize-plus-cushion dimensions, not a
new measurement of playing bounds.

## Actual pilot results

Five stable captures were converted under
`data/structured/author-blackball-pilot-v2/`:

- Five `blackball-observation-v1` records with source and BEV endpoint images,
  observed trajectories, quality flags and null cue/spin ground truth.
- 1,182 original raw frames were read, not manufactured through interpolation.
- Fifteen separate algorithm-animation exports: LEGG, MOLT and VACHASPATI for
  each of the five recordings.
- All five conversions completed. No accepted snooker benchmark task, model
  inference, training or external API operation was added.

### Two source-format traps verified in real files

1. `BlackBall_0/BGR.dat` contains 263 complete raw frames. The AVI has 264
   decoded frames, and its last frame is black. The converter uses the complete
   raw capture when available instead of treating that black frame as a final
   physical table state. The same one-frame count difference was observed in
   the other four pilot clips, but their terminal AVI image content has not all
   been independently reviewed.
2. `BlackBall_0`'s MOLT animation has 16 position tracks and 264 normalised
   time keys per track, with `cycleInterval=11.1645`. The AVI duration is
   13.2 seconds. LEGG and VACHASPATI also differ in sample count and animation
   timing. Animation time is not assumed to be capture time, and scene
   coordinates are not silently converted to physical metres.

The weak-animation exports retain source-native xyz coordinates, normalised
keys, cycle intervals and explicit unverified time/units flags. They are not
merged into independently observed ball trajectories.

The existing detector still misses touching-ball clusters. In the first pilot
clip the initial rack is visibly dense but only the cue is detected; this
becomes an unresolved-component/full-inventory flag, not a one-ball true
layout. No pilot record is eligible as a complete inverse-action test yet.

## Compatible structure, separate game semantics

The existing `snooker-observation-v1` contract remains unchanged for main
snooker records. `blackball-observation-v1` shares the observed-state,
calibration, trajectory, media and quality fields, but has an explicit
blackball game tag and cannot pass the snooker record validator.

Endpoint-only JSON removes interior track identities and algorithm trajectories.
Source video and raw-frame clocks remain distinguishable. All automatic labels
are unverified and `benchmark_accepted` remains false.

## Reproduce and continue uploads

```bash
.venv/bin/python -m inverser inventory-author-uploads \
  --source data/Billiard-dataset \
  --output data/quality/author-upload-inventory-v2 \
  --stable-seconds 120 --observe-seconds 5

.venv/bin/python -m inverser convert-author-upload-pilot \
  --inventory data/quality/author-upload-inventory-v2/inventory.json \
  --output data/structured/author-blackball-pilot-v3 \
  --game blackball --limit 5
```

Each inventory is immutable. Recent or changing files are deferred, including
files changing during hashing. Stability is an ingestion safeguard, not proof
that every expected dataset asset has arrived.

The conversion pilot permits 1-20 recordings, uses only stable original inputs,
checks their hashes again, rejects partial raw frames, and records explicit
failures. It neither changes global environments nor executes X3D scripts.
No newly arriving source is automatically mixed into an old run.

## How this supports the current research

1. **Perception and tracking diagnostics first.** Controlled overhead clips
   isolate touching-ball, blur, dark-ball and identity errors that currently
   limit useful extraction from match footage. They are a development domain,
   not the held-out snooker test set.
2. **Weak trajectory supervision, conditionally.** Algorithm-produced numeric
   paths may support proposals or a teacher after verifying image-space mapping
   and time alignment. Agreement between three algorithms is not independent
   ground truth.
3. **Table-specific physical fitting.** Verified observed motion can later fit
   shared cloth/cushion parameters on support shots. Do not tune the same
   parameters against hidden test trajectories or report inferred spin as measured.
4. **A controlled transfer experiment.** Compare snooker-only training with
   blackball auxiliary training at matched data/compute, then evaluate on
   independent snooker recordings. Different ball/table geometry must stay explicit.
5. **Source-layout-conditioned simulation.** Verified real layouts can seed
   new simulated controls/goals in the synthetic stratum. They do not become
   additional observed real shots.

The next useful uploaded subset is the original snooker captures plus their
background and manual-path files if available. Large tracking comparison
renders are less valuable for main benchmark scale than those original assets.

## Subsequent snooker arrival

Both snooker folders have now arrived. A later essential-input snapshot on
2026-10-13 contained 65 original video files (63 stable) and all 16 manual
path images. These are dated observations during an ongoing upload, not final
dataset counts.

Use the [incremental snooker pipeline](incremental_snooker_pipeline.md) rather
than repeatedly converting the first five files or rereading all auxiliary
movie assets. It caches unchanged hashes, links exact-content ordinary/manual
aliases, converts each unique ready recording once per input version, and
reproduces native manual-path metrics without inventing timestamps or controls.
