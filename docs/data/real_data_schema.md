# Common BEV-first real-data contract

The main snooker source adapters emit `snooker-observation-v1`, validated by
`SnookerEpisode`. Original match videos and author-recorded clips are not
combined by copying labels or assuming identical physics.

The direct-upload adapter emits `blackball-observation-v1` for auxiliary
blackball captures. It shares the observation fields but has a distinct game
tag and schema, rejected by the main snooker validator. See
[direct-upload provenance and conversion](../pipelines/author_direct_uploads.md).

## Primary-source facts that affect the contract

The supplied [Applied Intelligence paper](../papers/s10489-023-04542-3.pdf),
Section 4.1, reports a 640x480 RGB camera, 20 fps, focal length 525 pixels and
a camera height of 1.853 m. The experimental table is 2.115x1.205x0.78 m, with
a reported 1.854x0.935 m baize-plus-cushion region. Its snooker balls have a
51 mm diameter. This is not a regulation-size professional snooker table.

The paper's Section 3 describes initial baize/ball detection, local trackers
weighted by histogram and distance evidence, then X3D animation from recovered
positions. The z movement into pockets is a tracking/reconstruction convention,
not measured angular velocity or a calibrated cue-control label.

The [author README](https://github.com/FJ-Rodriguez-Lozano/Billiard-dataset/blob/main/README.md)
documents original `BGR.avi`, raw `BGR.dat`, manual-path PNGs, tracking outputs
and reconstructed X3D files. `BGR.dat` is headerless 640x480 BGR data in the
provided reader example. The adapter currently reads actual AVI stream
properties; it does not assign frame times from raw-byte counts without checking
the sensor/time convention.

The paper says manually selected trajectories were exported as PNGs. Those
files are retained as original annotations. Their presence does not establish
time-indexed numerical ground truth, cue force or spin.

## Record fields

| Field | Meaning |
| --- | --- |
| `source` | Collection, original file/hash, recording identity, split group and source licence status |
| `calibration` | Original/scaled image geometry, homography, table/ball dimensions, their basis, and uncorrected effects |
| `initial`, `terminal` | Actually detected balls at the sampled boundary times; no implicit missing or pocketed balls |
| `trajectories` | Observed positions only, with local track IDs and gaps; no interpolated samples disguised as observations |
| `estimated_events` | Unverified cue-motion onset or ball-proximity evidence, not verified physical collision sequences |
| `source_annotations` | Original manual path images, tracking outputs or reconstructed files, with provenance and hashes |
| `media` | Source endpoint images, rectified BEV images, a clean estimated table view and a source clip reference |
| `quality` | Unresolved identity, occlusion, inventory, calibration and motion flags; acceptance remains false |
| `cue_control_ground_truth`, `angular_velocity_ground_truth` | Null for these unmeasured real sources |

Metric coordinates use the table's short axis for x and long axis for y.
The image-facing orientation is explicit but not claimed to identify the
baulk end without separate evidence. Real table dimensions are recorded
per source, not silently replaced by simulator defaults.

The HF adapter uses nominal regulation dimensions. The author adapter uses
the paper's baize-plus-cushion dimensions and labels that basis explicitly.
Neither is claimed as a per-recording measured playing-area calibration.
Lens distortion, ball-center height and metric error remain uncorrected or
unmeasured in these raw adapters.

## Files per raw episode

```text
structured.json          # Full common record with privileged observations
endpoints.json           # Boundary-only observations, no interior identities/events
initial_source.jpg
terminal_source.jpg      # Author adapter; match adapter retains goal_source.jpg
initial_bev.jpg
terminal_bev.jpg
clean_table_pair.png     # Estimated locations, not a photograph or verified layout
```

Source clips retain their original format when already segmented in the author
dataset. Match-derived clips are compressed copies with explicit source
time ranges; originals remain untouched.

`endpoints.json` removes track IDs and does not expose interior trajectories,
estimated onset, event order or control labels. Red-ball identities are not
made observable by secretly importing full-trajectory tracking.

## Scale, grouping and uncertainty

Raw collection uses disjoint ownership intervals with overlapping context.
A candidate is owned by the interval containing its estimated motion onset,
so context frames do not become additional samples or duplicate owners.
Recording/hash grouping is retained for later train/test splits.

An identical set of sampled contact sheets was found for `8.mp4` and
`match19.mp4`. They share a conservative split group, but both are scanned:
sampled-image equality is not a proof that the full recordings are identical.
Author source clips with identical complete-file SHA-256 hashes are deduplicated,
preferring the version with original manual annotations.

The adapters collect raw or Silver-unverified observations. They do not promote
them to Gold because of detector agreement or a large count. Correct
serialization, reproducible hashes and decodable videos do not establish
correct physical labels.

After full collection, qualification rate and failure strata determine the
experiment panel. No model experiment threshold is triggered by frame count
alone, and raw collection does not bypass benchmark freeze or formal-evaluation
approval.

## Validation

Tests cover both adapters against the same record validator, horizontal
experimental-table geometry, metric conventions, original annotation types,
unknown motion onset, lack of invented actions, endpoint-view isolation,
ownership boundaries and interrupted-job resumption.
