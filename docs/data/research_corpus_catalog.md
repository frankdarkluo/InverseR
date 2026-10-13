# Research corpus: sources, units and label capabilities

The local research catalog combines the completed author upload, collected
match-video candidates and the existing simulator pilot without copying their
payloads or merging incompatible label types.

## Actual source counts

The completed author upload has 100 distinct original video/raw-capture hashes
per game: snooker, blackball and carom. It contains 354 video file copies because
the manual subsets repeat 16, 22 and 16 recordings respectively. Those copies
are not additional physical shots.

All 7,394 essential author assets passed the stable-file snapshot and hash
checks. Essential assets occupy 103.70 GB; the full source tree, including
derived movies and scene assets, is approximately 200.35 GB.

All 100 unique snooker recordings have been converted: 37,876 complete raw
frames, 16 linked manual-path references and no conversion failures.
They remain automatically estimated observations, not accepted physical GT.

The match source contains 32 videos/228.38 hours and 7,114 collected raw
candidates. Automatic screening assigns 16 review priorities, 6,004 records
needing reprocessing, 887 outside its conservative protocol and 207 probable
duplicates. One priority has a separate assistant hold for a missed black ball.
The catalog retains these decisions rather than presenting every row as usable.

## Catalog strata

| Role | Index references | Available labels | Main limitation |
| --- | ---: | --- | --- |
| Main real observations | 7,214 | 100 author recordings plus 7,114 weak match observations; observed endpoints/tracks and quality fields | Neither the total nor the automatic shortlist is independent, complete physical GT |
| Auxiliary captures | 200 | 100 blackball and 100 carom originals, with original/raw references | Separate game geometry/semantics; not main snooker benchmark counts |
| Simulator pilot | 2,000 | Generating cue controls, physics parameters, events and simulated motion | Pinned simulator truth, development-only, not measured real actions |

There are **9,414 typed references**, not 9,414 Gold real shots or one million
tasks. All main real controls/spins remain unmeasured. The author manual images
are native spatial path unions, not timestamped per-ball action labels.

The auxiliary rows are source references. They do not imply that all 200
auxiliary recordings have been converted or used in training.

## Reproduce

```bash
.venv/bin/python -m inverser build-research-catalog \
  --author data/structured/author-snooker-incremental-v1 \
  --author-inventory data/quality/author-completed-upload-snapshot-v1/inventory.json \
  --matches data/structured/snooker-corpus-v1-recovery \
  --match-screening data/quality/snooker-screen-v2 \
  --synthetic data/pilot-v0 \
  --output data/research/physics-world-model-catalog-v2
```

Use a new catalog directory for changed sources. The existing version is at
`data/research/physics-world-model-catalog-v1/`:

```text
summary.json
validation.json
index.csv
real_main_observation.jsonl
auxiliary_capture.jsonl
synthetic_shot.jsonl
```

Each row states its source/game, source group, original/private payload reference,
optional endpoint-only input reference, line position for JSONL payloads,
source manual-path references, automatic quality status and assistant hold.
The exact-action flag is permitted only in the simulator stratum.

## Experiment use

- Select tasks by required label capabilities before sampling or counting them.
- For physical control supervision, use declared simulator controls; inferred
  real controls need a separate pseudo-label protocol.
- For real observation consistency, use withheld interior observations and keep
  complete-inventory/calibration uncertainty explicit.
- For a manual-path metric, compare the documented native raster representation;
  do not infer time, identity or spin from it.
- Keep source groups together. Avoid duplicate copies, overlapping broadcasts,
  shared layout variants or shared calibration shots crossing partitions.
- Auxiliary pretraining and cross-game tests must be separate from main
  snooker success metrics.
- Catalog creation does not freeze the benchmark, approve model training,
  resolve the broadcast source's reuse rights or establish real execution.

## Validation

All 9,414 rows passed the catalog model, distinct-index-ID checks, file-reference
checks and source-manifest hash checks. The source payloads remain unmodified.
One hundred and three tests, Ruff and Mypy passed. No model/API experiment ran.

## Research design

The [world-model research report](../research/world_model_research.md) connects these label
capabilities to verified prior work and prospective empirical studies. It covers
action-conditioned prediction, tool attribution, support/query physical
adaptation, real spatial-path compatibility, leakage controls and cost accounting.
It does not accept the real observations as physical GT or approve benchmark
freezing, million-task expansion, training or formal evaluation.
