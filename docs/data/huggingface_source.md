# Public snooker-video snapshot

Source: [spdev39/snookerVideos](https://huggingface.co/datasets/spdev39/snookerVideos).

Pinned revision: `dcbcdf46b71c30e76bf51220f697836abd9c60bf`.

The researcher explicitly requested full local raw acquisition on 2026-10-10.
The source is public and ungated; no token is used or saved.

## Local organization

```text
data/snookerVideos/
  source_manifest.json       # Pinned remote inventory and source-use status
  acquisition_state.json     # Live transfer, verification and failure counts
  files/                     # Original filenames, unchanged
    *.mp4
    *_corners.json
    .gitattributes
    .cache/huggingface/       # SDK resume bookkeeping and incomplete downloads
  inventory.json             # Completed-file hashes and media/sidecar properties
  inventory.csv              # Flat inventory for inspection
  reports/
    corner_index.json        # Sidecar-to-video filename associations
    summary.json             # Completed acquisition and coverage summary
    video_index.csv          # Per-video properties and corner/preview links
    contact_sheets/          # Four sampled frames for each of the 32 videos
    contact_sheets.json      # Preview inventory and sampling limitations
    match13_probe.json       # First completed video's stream properties
    match13_contact.png      # Four sampled frames for local visual triage
```

The manifest contains 54 source files: 32 videos, 21 corner-coordinate JSON
sidecars and one repository metadata file. Total source size is 149,838,665,026
bytes; videos account for 149,838,661,502 bytes.

**Acquisition completed on 2026-10-10 at 13:27:34 UTC.** All 54 files passed
size and LFS SHA-256 or Git-blob verification. No files are missing or failed,
and all 32 video stream probes passed. These are source-file counts, not
accepted shot counts.

The videos total **228.3802 hours**. Resolution coverage is 14 at 1280x720,
17 at 1920x1080 and one at 1920x1072. Sixteen videos are 30 fps and sixteen
are 60 fps; all use H.264. No exact duplicate video hashes were found within
this snapshot; that does not establish absence of overlapping or re-encoded
footage.

See [summary.json](../../data/snookerVideos/reports/summary.json) and
[video_index.csv](../../data/snookerVideos/reports/video_index.csv) for actual
counts and per-file properties.

## Download and resume

```bash
.venv/bin/python -m pip install -e '.[data]'
.venv/bin/python -m inverser download-hf-source \
  --output data/snookerVideos --max-workers 4
```

Re-running the same pinned command reuses Hugging Face's partial-download/cache
bookkeeping and re-verifies local files. Changed snapshot contracts require a
separate directory. Never delete incomplete files merely to restart a transfer.

At most four files transfer concurrently. The downloader enforces the project
storage/free-space preflight, validates file paths and sizes, checks LFS SHA-256
or ordinary Git blob hashes, and records explicit failures. It uses the official
Hugging Face client with authentication disabled. It does not modify global login
state or send private research content.

## Initial inspection

`match13.mp4` has already completed download and local decoding:

- H.264, 1280 by 720, 30 fps.
- Duration: 8,275.323311 seconds, approximately 2 hours 18 minutes.
- Four sampled frames show a near-complete table view and scoreboard, making it
  more promising than the cropped short Commons clips.

This is assistant visual triage, not a full-frame audit, human annotation or
verification of complete shot endpoints.

All 21 sidecars have four numeric coordinate pairs. Their filenames associate
with 21 downloaded videos, and their coordinates lie inside those videos'
frame bounds. Eleven videos have no matching sidecar. Coordinate order,
reference frame, scaling and consistency across camera changes remain unverified.
These are table-corner annotations, not ball trajectories or measured cue spin.

Contact sheets were generated for all 32 videos using four timestamps per video.
They are inspection aids, not full decoding validation or automatically accepted
shot annotations.

## Source-use boundaries

The source's media provenance and licence remain unverified. Local execution
approval is recorded separately from rights-holder permission. Original videos
are not cleared for redistribution, and no model training, public release or G1
approval is implied by acquiring them.

`Source.access` retains the source-use workflow gate. The separate
`technical_access` field records that public downloads are reachable, and
`local_download_requested` records this explicit acquisition request.

The next stage is shot-window, calibration and observation-quality validation
under the project's approval process. Do not count 32 videos, their frames or
corner files as independent benchmark shots.
