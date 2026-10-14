# Inference-study preparation, not model evaluation

The approved [research plan](../research/plan.md) separates preparation,
resource confirmation, model compatibility, development diagnosis and formal
evaluation. This page documents the implemented preparation surfaces only.

## Safe task views

`study-forward-v1` contains the initial state, requested cue action, explicit
table/cue geometry and visible physical parameters, and sampling times. It
does not include a goal, generating family, source group or hidden witness.

`study-goal-v1` contains initial and target states, the same public physics,
the declared radius-normalized tolerance and no-scratch constraint. Reds in
the target are exchangeable endpoint-local objects, not historical track IDs.

The existing `pilot-v0` records remain development-only. They are not modified
or renamed into a blind test set. The study's private record supplies reference
labels; only typed public views enter the model-call serializer.

Known physics includes the actual cue specifications and pocket/cushion
geometry. Withheld coefficients are omitted rather than filled with nominal
defaults. Cue speed is stick speed; tip offsets are normalized contact positions,
not angular velocity. Elevation is explicitly zero.

## Paired inputs and labels

- Structured requests contain ball coordinates.
- BEV requests carry initial/goal PNGs as needed, plus inventory and public
  physical specifications. Ball coordinates are removed from the text input.
- A forward request cannot attach a goal image.
- The new renderer uses one physical pixel scale for both axes and the
  configured pocket/cushion geometry. The earlier montage remains a preview.
- Initial red slots are ordered by x then y; no future tracking identity is
  supplied.
- Regular-time labels use Pooltool's piecewise physical evolution from the
  last event, including right-limit states at duplicate collision timestamps.
  They are not linearly interpolated across collisions.
- Semantic contact labels exclude solver transitions. Unsupported event
  types are explicit errors, not silently invented contacts.

## Mock transport and durable accounting

The typed HTTP adapter supports configured API and loopback vLLM routes, but
actual execution needs resource authorization, a declared model revision and
runtime credentials where applicable. It does not launch a server or load
weights. Mock mode requires an explicit mock transport.

Each run ledger records requests before dispatch. Completed identical requests
are cached across restart. Failed, reserved or unknown-response requests are
not automatically resubmitted, because the original may have been billed.
Returned model identity, response shape, sampling times, ball slots and usage
are checked. Missing usage stays unknown.

Request, physics-evaluation and physics-attempt limits are distinct. Physical
reservations are atomic across connections; optimizer/internal or failed
replays cannot bypass accounting. A zero allocation disables that route.

For actual model calls, a finite total-token allocation is required. The
configured per-request reservation covers the permitted response and the
operator's input allowance. Unknown response/usage retains its reservation.
If provider-reported consumption exceeds the reservation, the overrun is
persisted and the run stops for review: reservations are admission control,
not a guarantee about a provider's hidden billing.

Keep one ledger per configured experiment/backend budget. Any explicit cap
change requires a new ledger rather than silently resetting consumption.
Endpoint/revision and input/prompt hashes participate in request identity;
credentials are never part of persisted configuration.

## Run the engineering fixture

From the repository root:

```bash
.venv/bin/python -m pip install -e '.[inference]'
.venv/bin/python -m inverser study-dry-run \
  --input data/pilot-v0/privileged.jsonl \
  --output .cache/study-engineering-dry-run-v1
```

Use a new output directory. The command processes one existing development
record, creates both public input views and BEV images, exercises mock request
cache/resumption, and evaluates the fixed fixture action with the reference.

The report explicitly states:

```json
{
  "status": "engineering_mock_dry_run",
  "actual_model_inference": false,
  "paid_api_requests": 0,
  "fixture_action_source": "privileged_witness_for_engineering_only",
  "model_performance_claim": false
}
```

A perfect fixture replay is an engineering check, not Qwen/API accuracy.
No actual model, API request, GPU job, training or benchmark expansion occurs.

## Real-observation mapping preparation

`validate_spatial_mapping` accepts independently sourced calibration landmarks
and held-out source-group landmarks. It rejects overlapping groups and
singular/projectively invalid mappings, and reports held-out pixel error.

Passing this check establishes only the declared spatial mapping. Its result
keeps measured cue action, timed manual trajectory, metric-world accuracy and
physical-benchmark acceptance false. No transform is fitted to a hidden query
manual path to inflate IoU. The actual author/manual mapping and 100+370
reprocessing remain separate resource/quality-gated jobs.

## Classical search and diagnostic plumbing

The next preparation checkpoint adds:

- Random and scrambled Sobol actions with uniform direction/speed and a
  radius-normalized contact-disk prior.
- Geometry-seeded, bounded multi-start Powell search using the existing SciPy
  optimizer. The geometry supplies an initializer, not a recovered historical
  action. It requires the declared known-physics, fully observed state.
- A scalar search loss that preserves maximum-error red bottleneck matching.
  Status mismatches cost `table_diagonal / ball_radius + 1`; a scratch adds
  another such penalty. The primary success test remains `score_endpoint`,
  with the declared tolerance and constraints, not equality to a witness.
- Separate moving/static forward errors and sample counts, initial-slot
  identity for forward trajectories/resting predictions, timed contact
  precision/recall/F1 and order with simultaneous-event ties.
- Paired intervention-response error and joint factual/intervened correctness.
  Correct factual predictions alone do not satisfy intervention consistency.
- A goal-independent fixed action bank, rank correlation, selected-action
  regret within that bank, and explicit failed-reference/abstention outcomes.
- Equal source/layout-group paired bootstrap summaries. One independent group
  produces no confidence interval; action variants do not create more groups.

Zero search budget yields no selected action. All objective calls, including
cached calls and failed simulations, appear in the attempt trace. The ledger
separately bounds actual forward evaluations and attempts. Search returns its
selected action; the evaluator does not silently choose a better one.

Known simulation errors remain failures, including after cache reuse. An
unfinished reserved replay is not automatically re-executed. Completed replays
can be recovered across a crash before the job artifact is written. Search,
hidden judging and offline reference generation have distinct persistent cost
roles and cache identities; oracle evidence cannot be reclassified as search.
Legacy costs remain unclassified rather than having a role invented.

Forward event scores declare both-empty event sets correct, but retain event
counts and individual strata. Constant or insufficient ranks have undefined
correlation, not a fabricated zero. A bank with failed reference simulations
cannot certify that no feasible action exists. Even an entirely evaluated
infeasible bank says nothing definitive about the full action space.

### Recoverable engineering run

```bash
.venv/bin/python -m inverser baseline-dry-run \
  --input data/pilot-v0/privileged.jsonl \
  --output .cache/baseline-engineering-v1 \
  --objective-budget 8 --candidates 4 --seed 20261013 \
  --limit-jobs 1

.venv/bin/python -m inverser baseline-dry-run \
  --input data/pilot-v0/privileged.jsonl \
  --output .cache/baseline-engineering-v1 \
  --objective-budget 8 --candidates 4 --seed 20261013
```

This command always uses just one existing development layout. It is hard
limited to 16 objective calls per baseline and eight bank actions. It does not
generate a new benchmark panel, invoke a model, process videos or allocate GPUs.
The bank forecasts use an explicitly named private-reference mock fixture;
perfect fixture predictions are not model performance.

The first invocation reports `partial`. The second resumes remaining jobs.
Repeating a completed run consumes no new model requests or physics evaluations.
Inputs, code, dependencies and resolved engine metadata are frozen by the
manifest. Completed job payloads are checksummed; changed configuration, inputs,
code or corrupted artifacts are explicit errors, not silent reruns.

Artifacts include the manifest, SQLite ledger, three classical-search jobs,
the fixed-bank diagnostic job, the goal-free public forward prompts and a
validation receipt. `physics_usage_by_role` separates agent search, hidden
judging and offline reference. Mock request counts are not paid API requests.

### Observed engineering validation

The full suite passes 157 tests; Ruff and Mypy pass for 30 source files. Tests
cover an interrupted job after physics computation, replay-cache recovery,
immutable artifact checks, atomic limits, role separation, hidden-physics
rejection, red identity, contact ties and clustered statistics.

A persistent CLI fixture was executed partially, resumed to completion and
repeated. It recorded 17 search evaluations, three hidden-judge evaluations,
four offline bank references and eight mock forecast requests. The completed
repeat added zero jobs and left all four artifact hashes and counts unchanged.

These are engineering receipts for one development layout, not an empirical
leaderboard, an independent test set or evidence of model physical competence.
Actual resource allocations, real-data G1 validation, protocol freeze and model
experiments remain separate gates.
