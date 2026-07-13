# Uniform FASRC Reporter Progress Design

## Goal

Every live job on `/app/fasrc` must show a progress bar. When the workload has
a meaningful numeric counter, the bar must be determinate and must receive that
counter through the structured `Reporter` event stream. While a job is queued,
initializing, or running an uncounted stage, the same shared monitor shows an
indeterminate bar instead of leaving an empty gap.

The immediate missing producer is the Lens Isolation mock-sky generation job.
`lens_isolation_train.py` already emits Reporter step events, but
`lens_isolation_generate.py` currently prints split summaries without emitting
numeric progress. The shared React monitor then has no step event to render.

## Scope

This change will:

1. expose exact per-example progress from Lens Isolation split generation;
2. adapt that progress to cumulative `Reporter.set_step(...)` events in the
   Lens Isolation generation CLI;
3. ensure known Lens Isolation training totals are published immediately,
   before the first evaluation callback;
4. render one shared determinate-or-indeterminate progress treatment for live
   FASRC jobs; and
5. add focused regression coverage for the producer, event fold, and frontend.

It will not parse `tqdm`, stdout, stderr, raw log files, SLURM elapsed time, or
partially written artifacts to estimate progress. It will not introduce a
second progress API or couple the reusable records module to the WebUI.

## Progress contract

`Reporter` remains the only machine-readable progress channel for FASRC jobs.
The workload owns the counter because it is the only layer that knows what one
unit of completed work means.

The UI applies this precedence:

1. if the latest Reporter status contains `step.total > 0`, show a determinate
   bar using `step.current / step.total` and its label;
2. otherwise, if the SLURM state is `PENDING` or `RUNNING`, show an animated
   indeterminate bar with a short state-appropriate label; and
3. do not show a live fallback bar for terminal jobs.

Numeric events always take precedence over the fallback. The UI never replaces
or guesses a Reporter counter.

## Generation producer

`generate_split(...)` will accept an optional progress callback without
importing `Reporter`. The callback receives the split-local completed count and
total.

For a split that must be generated, it reports:

```text
0 / split_count
1 / split_count
...
split_count / split_count
```

An example counts as complete only after its aligned scene, lens, optional
dirty record, and manifest row have been prepared by the ordered writer loop.
Worker futures do not report completion directly because a simulated example
is not yet part of the ordered atomic output until that loop consumes it.

For a validated split reused from disk, the callback reports
`split_count / split_count` once. A zero-sized split reports `0 / 0` without
creating a fictitious numeric denominator.

`lens_isolation_generate.py` owns the Reporter adapter. It computes one grand
total across train, validate, and test and maps split-local callbacks to a
cumulative counter:

```text
global_current = completed_prior_splits + split_current
global_total   = ntrain + nvalid + ntest
```

Before each split it publishes a stage name and an initial cumulative step.
The label identifies the split and local count. Reused splits therefore advance
the same bar immediately instead of appearing stalled. The CLI also runs the
standard resource sampler so the existing monitor can show CPU utilization.

## Training producer

Lens Isolation training already routes evaluation-time metrics and cumulative
member/step progress through `Reporter`. It will additionally emit the known
`0 / (members * steps)` counter before expensive model and dataset setup. This
makes the determinate bar available from the start rather than only after the
first `evaluate_every` boundary.

Existing evaluation callbacks remain the update cadence. This avoids writing a
high-frequency event for every gradient step while preserving an exact numeric
counter whenever an update is published.

## Shared frontend behavior

`JobStatusBody` will receive the current SLURM state as well as the folded
Reporter status. Both consumers—the inline submitted-step monitor and the
`Current Submission` card—will pass that state through the same component.

The existing `ProgressBar` already supports `value={null}` as an indeterminate
animation. No new visual component or page-specific CSS is needed. Reporter
stage text, resource gauges, warnings, and errors keep their existing layout.

This keeps Lens Isolation, synthetic sky generation, training, and every other
FASRC workload visually uniform while allowing progressively richer Reporter
instrumentation per job.

## Failure and transition behavior

- Missing or temporarily unreadable event files produce the live indeterminate
  fallback rather than an empty monitor.
- A transient SSH poll failure keeps the last known job shell; it does not
  manufacture numeric progress.
- Invalid or zero Reporter totals do not produce division by zero and fall back
  to the live indeterminate state.
- Reporter write failures retain existing behavior and are not hidden by log
  scraping.
- Terminal state badges, warnings, and errors remain visible; the generic live
  fallback stops once the job is terminal.

## Testing and verification

Focused tests will establish:

1. sequential and threaded `generate_split(...)` calls emit ordered exact
   progress ending at the split total;
2. reused splits emit their completed numeric state without regeneration;
3. Lens Isolation CLI Reporter events fold to the expected cumulative total;
4. training publishes its known initial total through Reporter;
5. the shared frontend selects determinate Reporter progress over the fallback
   and uses indeterminate progress only for live jobs without a numeric step;
6. existing Lens Isolation and job-status tests remain green; and
7. the React production build succeeds.

Repo-specific verification will use the focused Conda test environment with
plugin autoload and JIT disabled where required, followed by Ruff on changed
Python files, the frontend build, and `git diff --check`.

## Acceptance criteria

- Starting Lens Isolation mock-sky generation immediately shows a bar on
  `/app/fasrc`.
- Once generation begins, the bar shows the exact cumulative completed/total
  count across train, validation, and test.
- Reused splits advance the numeric bar correctly.
- Lens Isolation training shows its numeric total before the first validation.
- Any other queued or running FASRC job without numeric Reporter progress shows
  the same indeterminate bar treatment.
- No numeric value is derived from logs or inferred by the frontend.
