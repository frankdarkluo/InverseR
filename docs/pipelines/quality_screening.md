# Automatic screening of the complete raw corpus

This stage triages the 7,114 provisional episodes after complete video
collection. It preserves the raw videos, original records and collection
ledger. A higher-priority episode is not automatically a complete physical
shot, a calibrated posterior or human-verified ground truth.

## Reproduce

```bash
.venv/bin/python -m inverser screen-real-corpus \
  --source data/structured/snooker-corpus-v1-recovery \
  --output data/quality/snooker-screen-v2 --workers 4
```

The screening input must be a complete, inactive collection. The output is a
separate version. Input database, collection manifest, screening source, record
schema, OpenCV version and policy are recorded in `run_config.json`. Changed
policies or code require a new screening directory. Raw candidate exports are
not rewritten or deleted.

## Checks and their evidence limits

1. Validate every full structured record and compare the frozen SQLite payload
   with its original on-disk record.
2. Verify endpoint positions against their pixel/metric conversion and declared
   trajectories. Check table bounds, ball count and the calibration's own corner
   mapping. Internal calibration consistency is not real-world metric accuracy.
3. Decode endpoint source and BEV images; probe every candidate video and compare
   its duration with the declared source interval.
4. Check observed cue identity/continuity, excursion, speed, motion bursts and a
   stationary terminal segment. A separate check requires retained object balls
   to have observed terminal rest.
5. Detect duplicate unique ball colours, unresolved endpoint components,
   unreliable shape evidence and high endpoint identity churn.
6. Estimate endpoint camera consistency from ORB features outside the table and
   scoreboard. Insufficient features or poor inliers are an unresolved result,
   not a zero-motion default.
7. Flag probable cross-recording duplicates only inside a shared source group
   with matching source times and similar hashes of both original endpoint images.
   This is a conservative warning, not exhaustive video deduplication.

Missing observations are never treated as measured pots. No missing ball
coordinates, action parameters, spin or trajectory samples are invented.

## Fixed first-pass policy

These are heuristic triage thresholds, not calibrated physical error bounds.
They were fixed before running the full screening output and are stored with
that run.

| Criterion for priority review | First-pass threshold |
| --- | --- |
| Episode duration | 2 to 45 seconds |
| Visible sampled-frame fraction | At least 0.90 |
| Minimum endpoint shape score | At least 0.55 |
| Endpoint identity-set retention | At least 0.80 |
| Cue track | One local identity, observed continuously between endpoints |
| Resolved cue motion | Excursion at least two nominal radii and peak observed speed at least 0.20 m/s |
| Separated motion bursts | Exactly one resolvable burst; otherwise inspect for multiple shots/interventions |
| Observed terminal rest | At least 0.80 seconds for cue and retained object balls |
| Cue/object rest step | At most half a nominal radius per sampled step |
| Endpoint inventory | No unresolved components or duplicated uniquely coloured balls |
| Rectified cloth coverage | At least 0.70 |
| Camera evidence | At least 12 feature matches, at least 0.50 transform inliers |
| Endpoint camera shift | At most 1% of source-image diagonal |

Return shots are not rejected merely because the cue ends near its initial
position: the screening uses trajectory excursion rather than endpoint shift
alone. Low-speed, short or weakly observed legitimate shots can still fall
outside these conservative criteria. Their source records remain available
for a future challenge split or re-extraction.

A cue-only rest check is insufficient when other balls still move. Conversely,
even retained-object rest cannot verify a complete inventory when the detector
misses entire balls. That limitation remains explicit.

## Output categories

| Category | Interpretation |
| --- | --- |
| `priority_review` | Passes the automatic observability checks; inspect before physical evaluation |
| `needs_reprocessing` | Potentially useful motion with detection, identity, camera or boundary defects |
| `not_suitable` | Too short or no resolved motion above the first-pass noise threshold |
| `probable_duplicate` | Same-time endpoint evidence matches another candidate in a shared recording group |
| `invalid` | Structural, media or stored-coordinate inconsistency |

No category means Gold. `priority_episodes.jsonl` retains the original unverified
record fields, including `benchmark_accepted: false`.

Outputs include the complete decision JSONL, a CSV for each category, an all-row
CSV, priority record references, per-source counts and explicit reasons.

## Assistant spot-check

The runner creates a reproducible stratified sample: up to eight episodes,
uniformly selected without replacement within each automatic category, using
seed `20261011`. Source endpoint galleries and a sample manifest are saved.

Assistant visual review can catch obvious errors and identify which checks are
missing. It is not human annotation, does not independently measure trajectory
accuracy and does not certify every unsampled record. If a sampled priority
episode is doubtful, record that doubt instead of relabelling it as physical GT.

## Is 7,114 enough, and how can it grow?

The raw count cannot answer this question by itself. Report post-screening
quality strata, independent source groups, retained task complexity and
coverage before selecting an experimental panel.

Use two scale axes separately:

- A reliable held-out evaluation set with source diversity and a clear
  uncertainty/label policy.
- A larger training or exploratory collection, which can retain appropriately
  labelled weak data.

The entire existing source was already scanned. Expanding it by using more
frames or overlapping crops does not create new independent shots.
The most useful expansion routes are:

1. Recover genuine shots currently missed by colour, touching-ball, cue
   visibility, camera calibration or initial/terminal boundary errors.
2. Add genuinely independent, appropriately sourced recordings or the complete
   author archive once its ZIP integrity and metadata are verified.
3. Generate distinct simulated controls and goals in a separate synthetic
   provenance stratum, including real-layout-conditioned simulation if used.

Do not lower filtering thresholds until a desired count appears. Changes need
a new version and a documented trade-off. No positive model insight is claimed
from dataset quantity or automatic triage alone.

## Versioned duplicate correction

The first screen is preserved under `data/quality/snooker-screen-v1`.
It used BEV endpoint hashes. Temporal review of its priority list found the same
source-time shot in `8.mp4` and `match19.mp4` still appearing twice.
Their original endpoint image hashes matched exactly, while their BEV hashes
differed by 22 and 20 bits because the projection configurations differed.

Version 2 uses native endpoint image hashes for this comparison, with the same
source-group/time constraints. It does not lower observability thresholds to
increase counts. Both run versions and the original raw data are preserved.

## Validation

Tests cover pixel/metric consistency, observed trajectory continuity, real
motion versus jitter, return shots, object rest, multiple motion bursts,
camera shifts, unresolved camera evidence, chained duplicate grouping and
read-only repeatable screening. Existing package lint, static type checks and
tests remain required.

## Completed first full screen

All 7,114 records were screened. Final automatic categories in version 2:

| Category | Records |
| --- | ---: |
| Priority review | 16 |
| Needs reprocessing | 6,004 |
| Outside this conservative protocol | 887 |
| Probable duplicate | 207 |
| Invalid structural/media record | 0 |

The counts sum to the original 7,114. All category CSVs, record hashes and the
unchanged input database were validated. Seventy-nine tests, Ruff and Mypy
passed. No GPU, model inference or API call was used.

Temporal sheets were inspected for every one of the first-pass 17 priority
episodes, plus eight sampled endpoint pairs in each category. This exposed
the duplicate fixed in version 2 and a concrete inventory failure: in one
episode a black ball is visible, but the detector reports a minimum inventory
of one ball. That episode is separately held for re-detection.

The [assistant triage](../../data/quality/snooker-screen-v2/assistant_triage.json)
retains **15 unverified review priorities**, not 15 accepted ground-truth tasks.
The raw records and automatic decisions remain intact.

Within the reprocessing category, **369 records from 19 recordings** have
continuous observed cue motion, one heuristic motion burst, terminal cue rest
and at least 90% clear observations. Their main remaining defects are endpoint
components, identity churn, color assignment or weak shapes. Including the
visually identified inventory failure yields a **370-record targeted repair
queue**, rather than blindly increasing downloads.

Key outputs:

- [Final screening summary](../../data/quality/snooker-screen-v2/summary.json).
- [All decisions](../../data/quality/snooker-screen-v2/all.csv).
- [Unverified shortlist after assistant triage](../../data/quality/snooker-screen-v2/priority_after_assistant_triage.csv).
- [Targeted reprocessing queue](../../data/quality/snooker-screen-v2/assistant_reprocessing_queue.csv).
- [Validation receipt](../../data/quality/snooker-screen-v2/validation.json).

All source footage was scanned, and the acquisition reports show zero export
cap exclusions. Five recordings produced no candidates because calibration,
visibility or ball detection did not qualify their frames. Recovering useful
shots from these recordings is a source of real expansion; increasing sample
frame rate alone would not create new independent shots.

The current limiting factor is extraction quality and independent label
verification, not a need to manufacture a larger raw count. This conclusion
concerns the data pipeline, not the inverse-reasoning ability of any model.
