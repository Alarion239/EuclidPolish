# Resource advisor: past-run usage dashboard + automatic resource recommendations

Date: 2026-09-30 · Status: approved (user, 2026-09-30)

## Problem

The console submits FASRC jobs with hand-typed CPUs / memory / time limit, but
nothing shows how much the previous runs of the same kind actually used. The
data already exists: the local job ledger `~/.euclid_polish/fasrc_job_log.csv`
(`JobRecord`, `euclid_polish/observability/job_log.py`, read through
`fasrc_jobs.JOBLOG.list_all()` / `history_for_step()`) holds, per job, the
requested resources (`partition, req_cpus, req_gpus, req_memory,
req_time_limit`), the task params (`params_json`) and the post-mortem
accounting from sacct + jobstats (`state, elapsed_seconds, cpu_seconds,
cpu_efficiency, max_rss_mb, alloc_cpus, alloc_gpus, alloc_memory_mb,
gpu_util_mean, gpu_util_peak, gpu_mem_peak_mb, jobstats_cpu_util,
jobstats_cpu_memory_used_mb, jobstats_cpu_memory_alloc_mb, jobstats_gpu_util,
jobstats_gpu_memory_used_mb, jobstats_gpu_memory_total_mb`, …). 474 jobs as of
today; `synthetic_generate` (112) and `ensemble_train` (58) dominate. Array jobs
are one ledger row whose allocation fields are **per task** and whose elapsed
is the max over tasks (`web/sacct.py::parse_sacct_output`) — per task is the
unit we recommend in.

## What we build

1. **A pure recommender** `euclid_polish/observability/resource_advisor.py`
   (no Flask, no I/O; takes ledger rows `list[dict[str, str]]`).
2. **Routes** `euclid_polish/web/routes/resources.py` (a new `ROUTE_MODULES`
   group), all local/offline (the ledger is local; never `@requires_fasrc`,
   never starts a job).
3. **Runs › Resources tab** (`/runs/resources`): the dashboard.
4. **A shared "Recommended from N past runs" callout** with an **Apply**
   button, in the generic FASRC `StepCard` (covers Synthetic › generate and
   every step card) and in Models › Train's Resources card. Nothing changes
   unless Apply is clicked.

## Recommender

### Row normalisation

`normalize_row(row) -> RunUsage` (dataclass or dict) with numbers parsed
defensively (blank / garbage → `None`, never raises):

- `req_memory_mb` from `req_memory` (`"32G"`, `"17GB"`, `"4000M"`, `"1T"`,
  plain number = MB, SLURM semantics) — fall back to `alloc_memory_mb`.
- `req_time_s` from `req_time_limit` (`MM`, `MM:SS`, `H:MM:SS`, `D-HH`,
  `D-HH:MM`, `D-HH:MM:SS`) — SLURM formats.
- `cpus = alloc_cpus or req_cpus`, `gpus = alloc_gpus or req_gpus`.
- `elapsed_s`, `cpu_efficiency` (0–1), `cores_used = cpu_efficiency × cpus`.
- `peak_mem_mb = max(max_rss_mb, jobstats_cpu_memory_used_mb)` (either may be
  blank; jobstats is node-level and sometimes the larger).
- `gpu_util` = `jobstats_gpu_util` else `gpu_util_mean` (percent).
- `gpu_mem_used_mb` = `jobstats_gpu_memory_used_mb` else `gpu_mem_peak_mb`
  — displayed only, never used to recommend: TensorFlow preallocates the whole
  card, so ~98 % is meaningless (say so in the UI).
- `mem_ratio = peak_mem_mb / req_memory_mb`, `time_ratio = elapsed_s / req_time_s`.
- `params` = parsed `params_json` (bad JSON → `{}`).
- `units`, `units_label`, `key`, `key_label` from the step profile (below).

### Step profiles (workload scaling)

A profile maps task params → `(units, units_label)` (the amount of work, used to
scale time) and a categorical similarity `key` (+ human `key_label`), plus
`memory_scales_with_cpus` and `gpu` flags.

- `synthetic_generate` — units = images actually generated:
  `regenerate_splits` (comma list of train/validate/test) → sum of
  `n_train`/`n_valid`/`n_test` for those splits; `force` truthy → all three;
  neither → `None` ("cache-first resume": unknown work). key = (sorted splits
  or "resume"/"all", `image_size`, onthefly_train truthy). Memory scales with
  CPUs (multiprocessing workers: ≈1.5 GB/CPU measured: 15 GB @ 10 CPUs,
  29–31 GB @ 20).
- `ensemble_train` — units = training steps per member (per array task):
  `steps` for add/fork; `extra_steps` for `mode=continue` +
  `continue_basis=extra`; `None` for `continue_basis=target` (unknown).
  key = (kind: `new` for add/fork / `continue`, `batch_size`, the set of
  `num_res_blocks` in `member_spec`, multi-knee = any spec has `asinh_knees`,
  `hr_crop_size`). GPU step; memory does not scale with CPUs.
- `train`, `lensfinder_train`, `lens_isolation_train` — units = `steps` if
  present; GPU.
- every other step — units `None`, key `()`.

Params arrive from the ledger as strings or native JSON values; the member spec
may be a JSON string or a list. Parse tolerantly.

### Which runs count

Per step: rows with `elapsed_s > 0` and state in
`COMPLETED` (full evidence), `OUT_OF_MEMORY` (memory lower bound: it needed more
than `req_memory_mb`), `TIMEOUT` (time lower bound: it needed more than
`elapsed_s`). `FAILED` and `CANCELLED` are excluded from recommendations
(errors are not resource problems) but counted in summaries. RUNNING / PENDING
/ blank ignored.

Match levels, tried in order; the first level with **≥ 3** usable runs wins,
else the best non-empty level is used with low confidence:

1. `exact` — same key **and** same CPU count (only meaningful when the step's
   memory scales with CPUs; otherwise skip to 2),
2. `similar` — same key,
3. `step` — every usable run of the step.

Within the level take the **20 most recent** (by `submitted_at`). The basis
reports the level, its human label, the jobids used and n.

### The recommendation

Inputs: `step_id`, all ledger rows, the planned task params, the current form
resources (`n_cpus, n_gpus, memory, time_limit`; any may be missing), and the
step's `needs_gpu`, `fixed_cpus`, `fixed_gpus`, default resources.

- **CPUs** (skip when `fixed_cpus` set):
  - CPU steps: if median `cpu_efficiency` < 0.35 → `ceil(p75(cores_used) × 1.3)`
    (≥ 1), with the note "fewer CPUs may lengthen the run"; otherwise keep the
    current value.
  - GPU steps: never reduce. If median `gpu_util` < 60 % **and** median
    `cpu_efficiency` > 0.75 → `ceil(current × 1.5)` with the note "GPU waits on
    the input pipeline (starved): add CPUs"; else keep.
- **GPUs**: keep (fixed per step).
- **Memory**: `p90(peak_mem_mb)` over COMPLETED + TIMEOUT runs × 1.2; if
  `memory_scales_with_cpus`: `p90(peak_mem_mb / cpus) × recommended_cpus × 1.2`.
  OOM floor: over OOM runs in the level, `max(req_memory_mb × 1.25)` (per-CPU
  scaled likewise). Round **up** to a multiple of 4 GB, minimum 4 GB. Format
  `"36G"`.
- **Time**: with units known for the plan and for ≥ 1 counted run:
  `rate = elapsed_s / units` per run; `p90(rate) × planned_units × 1.2`, at
  least `+ 5 min` over `p90(rate) × planned_units`. Without units:
  `p90(elapsed_s) × 1.2` (+ 5 min floor). TIMEOUT runs are included (they are
  lower bounds) and, if any TIMEOUT run in the level would still time out
  under the recommendation (its scaled elapsed ≥ recommended), bump to 1.5 ×
  that scaled elapsed and warn. Round **up** to 15 min, minimum 15 min, cap
  at 3 days (warn when capped). Format `H:MM:SS`, or `D-HH:MM:SS` ≥ 24 h.
- **Confidence**: `high` = ≥ 5 runs at `exact`/`similar` and (units known or
  step has no units); `medium` = ≥ 3; else `low`.
- **Changes**: one entry per field whose recommendation differs from the
  current form value (compare parsed quantities, not strings), with a
  one-sentence reason quoting the evidence, e.g. "p90 peak 15.4 GB of 17 GB
  over 8 runs; 1 OOM at 15 GB".
- **Notes / warnings**: GPU-memory preallocation caveat on GPU steps; "units
  unknown for this plan (cache-first resume / continue to target): time from
  whole-run elapsed" when relevant; low-n warning.
- No usable runs → `available: false` and the current values echoed.

Percentiles: linear interpolation (numpy-style) over the non-None values; with
one value it is that value.

### Summaries (dashboard)

Per step, over all its rows: `runs`, counts by state (`completed, oom,
timeout, failed, cancelled, running`), `success_rate` (completed / finished),
`last_submitted_at`, `needs_gpu`, medians of `cpu_efficiency`, `gpu_util`,
`mem_ratio`, `time_ratio`, `p90 peak_mem_mb`, and waste totals over finished
runs: `cpu_hours_alloc = Σ cpus × elapsed`, `cpu_hours_used = Σ cores_used ×
elapsed`, `gpu_hours_alloc = Σ gpus × elapsed`, `gpu_hours_used = Σ gpus ×
elapsed × gpu_util/100`, `mem_gb_hours_alloc/used` likewise. Array jobs count
one task's allocation (per-task honest unit; say so in a hint).

## API

All `GET`s are read-only and offline.

- `GET /api/fasrc/resources` →
  `{ok, steps: [StepSummary…]}` — ordered `ensemble_train`,
  `synthetic_generate` first, then by `last_submitted_at` desc. StepSummary:
  `{step_id, label, needs_gpu, runs, states: {completed, oom, timeout, failed,
  cancelled, running}, success_rate, last_submitted_at, cpu_efficiency,
  gpu_util, mem_ratio, time_ratio, peak_mem_p90_mb, cpu_hours_alloc,
  cpu_hours_used, gpu_hours_alloc, gpu_hours_used, mem_gb_hours_alloc,
  mem_gb_hours_used}` (medians; `null` when unknown). `label` = the step
  registry's human title when the step still exists, else the step_id.
- `GET /api/fasrc/resources/<step_id>` → `{ok, step_id, summary, runs:
  [RunUsage…] (newest first, ≤ 200), recommendation}` where `recommendation`
  is computed for the step's **latest counted run's params** and the step's
  default resources (so the dashboard can show "for the next run like the
  last one"). Unknown step with no rows → 404 `{ok: false, error}`; a
  historical step no longer in the registry but with rows still answers.
  RunUsage JSON: `{jobid, submitted_at, state, partition, cpus, gpus,
  req_memory, req_memory_mb, req_time_limit, req_time_s, elapsed_s,
  cpu_efficiency, cores_used, peak_mem_mb, mem_ratio, time_ratio, gpu_util,
  gpu_mem_used_mb, units, units_label, key_label, label}`.
- `POST /api/fasrc/resources/<step_id>/recommend` (JSON body
  `{params: {…}, resources: {n_cpus, n_gpus, memory, time_limit}}`, also
  accepts form fields: resource keys are split out, the rest are params) →
  Recommendation:
  `{ok, step_id, available, confidence, resources: {n_cpus, n_gpus, memory,
  time_limit}, current: {…}, changes: [{field, current, recommended,
  reason}], basis: {level, level_label, n_runs, jobids, units, units_label,
  rate_s_per_unit}, notes: [str], warnings: [str]}` — `resources.n_cpus`
  and `n_gpus` are strings like the forms use; `memory` `"36G"`;
  `time_limit` SLURM text. It is a POST only because the params are large;
  it mutates nothing (the mutation guard must allow it like the other
  read-ish POSTs — check `register_mutation_guard`).

The ledger is re-read per request; cache the parsed rows keyed on the CSV's
`(mtime_ns, size)` so the dashboard does not re-parse 2.8 MB every call.

## Frontend

Follow `web/frontend/src/FOUNDATION.md` (GET via `useResource`, POST via
`apiPost`, colours only from theme tokens, the ui kit, `Plot` for charts,
DataTable for tables). Load the `dataviz` skill before writing chart code.

- **Registration**: `workspaces/runs/index.tsx` (`resources` tab), `app/nav.ts`
  (label "Resources", description "What past runs asked for vs used, and what
  to ask for next", keywords cpu/memory/time/gpu/efficiency/sacct),
  `spa_routes.json` `runs.tabs` (+ whatever mirrors it: `app/manifest.ts`,
  redirect-case tests, Python spa route tests), and FOUNDATION §11.3/§11.13
  notes.
- **`workspaces/runs/tabs/Resources.tsx`**: step picker (URL state `?step=`,
  default `ensemble_train`), a step list with small per-step stats (runs,
  success, OOM/timeouts, CPU eff, GPU util); for the picked step: stat tiles
  (runs · success rate · OOM · timeouts · median CPU efficiency · median GPU
  util · median memory used/requested · median time used/limit · wasted
  CPU-h / GPU-h), charts (per run over time: peak memory vs requested, elapsed
  vs time limit — OOM/TIMEOUT runs marked), a DataTable of recent runs with
  requested-vs-used bars and state badges (row → open the job inspector like
  History does), and the recommendation card ("For the next run like the last
  one", with a link to where that step is submitted: Models › Train for
  `ensemble_train`, Synthetic status/generate for `synthetic_generate`, else
  Runs › Steps).
- **`ResourceAdvice` component** (shared: `workspaces/shared/` or `src/fasrc`
  neighbourhood, per FOUNDATION §11.6): props `{stepId, params, resources,
  onApply(res)}`; debounced POST to `/recommend`; renders a compact Callout:
  "Recommended from N past runs (level label) · confidence", the change list
  (current → recommended, reason), notes/warnings collapsed, an **Apply**
  button (disabled when no changes) and an "Open usage" link to
  `/runs/resources?step=`. Loading/empty/error states are quiet (no big
  errors in the form).
- **StepCard**: render `ResourceAdvice` under the resources grid; Apply sets the
  editable resource fields (respect `fixed_cpus` / `fixed_gpus`; partition is
  fixed).
- **Models › Train**: render it in the Resources card with the built params;
  Apply → `ownRes({n_cpus, memory, time_limit})` (per model = per array task).
  Keep the existing "as job X" hint behaviour.
- Pure helpers (formatting, level labels, change mapping) in a `model.ts` with
  vitest tests; a component test for `ResourceAdvice` (renders changes, Apply
  calls back with the recommended resources, no-history state) and a smoke
  test for the tab.

## Testing

- `tests/test_resource_advisor.py`: parsing (memory/time formats, blanks),
  per-profile units/keys (splits, force, resume, continue/extra/target,
  member_spec as string/list), level fallback, OOM floor, TIMEOUT bump,
  CPU rules (CPU-step reduce, GPU-step starved increase, never reduce GPU
  CPUs), rounding/formatting, confidence, no-history.
- `tests/test_resource_routes.py`: the three endpoints on a temp ledger
  (monkeypatched `fasrc_jobs.JOBLOG`), 404, form + JSON bodies, no job spawned.
- Frontend: vitest as above; `npm run typecheck`, `npm run lint`,
  `npm test`, `npm run build` (the build output `static/dist/` is committed).
- Manual: the dashboard against the real ledger in the browser pane.

## Out of scope

Changing step defaults automatically; live (running-job) utilisation; queue
wait-time prediction; per-node speed modelling.
