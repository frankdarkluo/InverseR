# Pilot protocol, not a frozen benchmark

The current implementation is a pre-G1 engineering pilot. It permits at most
5,000 development tasks per run and does not implement million-task expansion or
formal model evaluation yet.

## Physical contract

- Table: 1.778 by 3.569 metres. Pocket/jaw geometry comes from the named snooker
  preset in Pooltool 0.6.0, not a claim of calibrated professional-table fidelity.
- Ball radius/mass/friction and cue/resolver parameters are explicit in each run
  manifest. The cue uses the pinned simulator's generic cue specifications;
  these are not measured properties of a real snooker cue.
- Controls: azimuth in degrees, cue-stick impact speed in m/s and radius-normalized
  side/vertical contact offsets. Elevation is zero for this pilot.
- The contact-offset disk is conservatively bounded at radius 0.5. It is not a
  complete model of human cue feasibility.
- The cue ball's resulting linear/angular velocities are saved separately from
  the cue controls.
- Initial states are stationary and fully observed. Pocketed terminal balls have
  no table-plane position; unknown observations are a distinct state.

Pooltool's event cap can force its internal balls stationary. InverseR explicitly
rejects capped/nonsettling runs instead of accepting that state as a valid goal.
Default user physics files are not used: the CLI configures a process-local HOME
under the project cache before importing Pooltool and constructs the pinned
resolver explicitly.

## Sampling and provenance

The pilot generates cue-only sanity cases and one-to-three-red-ball layouts.
It retains center, longitudinal-offset and side-offset cases; generating a
spinning shot does not prove that spin is necessary for the same goal.
Initial layouts are sampled independently, not taken from real matches.
Scratches and invalid/nonsettling physics are rejected with recorded reasons.

Every accepted record is `simulated_truth`, is in the development partition and
contains a hidden generating witness. There are no observed real tasks or
human-verified labels in these synthetic pilot files.

## Public and privileged views

- `public.jsonl`: initial/goal observations, geometry/physics convention, task
  constraints and tolerance. No generating controls or intermediate trajectory.
- `privileged.jsonl`: full generation records for controlled training/auditing.
- `pilot.sqlite`: resumable generation records and issued attempt statuses.
- `manifest.json`: source/config fingerprint, counts, rejections and export hashes.

Red IDs in the public view are assigned separately per endpoint. Red balls are
exchangeable; labels must not expose historical tracking identities.

The main endpoint metric uses minimum-bottleneck matching within each ball
class/status and a maximum position error, not a sum-optimal assignment or a
whole-table average diluted by stationary balls. Pocket counts/status and
no-scratch are separate success conditions.

## Verification and limits

The audit checks every exported schema, ID/group uniqueness, public/private view
consistency and file hash. A requested subset of generating witnesses is rerun;
the report states the exact number, rather than implying every task was replayed.

Tests cover analytical center-hit motion, action validity, identity invariance,
physical limits, idempotent resumption and interrupted generation.

None of these checks establish real-video label accuracy, full 22-ball coverage,
real execution, model physical understanding or a learned method advantage.

## Reproduce

```bash
.venv/bin/python -m inverser generate-pilot --output data/pilot-v0 --count 2000
.venv/bin/python -m inverser audit-pilot --output data/pilot-v0 --replay-count 100
.venv/bin/python -m inverser preview --input data/pilot-v0/public.jsonl --output artifacts/pilot-v0.png
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/mypy
```

Use the same source and configuration to resume the same output directory.
Changed source/configurations require a new versioned directory; they never
silently append incompatible records to the old pilot.
