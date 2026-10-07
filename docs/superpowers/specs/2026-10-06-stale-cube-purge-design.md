# Automatic stale-cube purge — design

2026-10-06. Approved in conversation.

## Problem

The console caches model outputs as `.npy` cubes and keeps them after they can
no longer be used. On 2026-10-06 the project held 66 GB, 40 GB of it cubes no
reader would accept:

- the four member-cube buckets under `data/vis/ensemble/starfull/`
  (`cubes`, `cubes_validate`, `cubes_blackout`, `cubes_validate_blackout`),
  made by checkpoints older than the members' current ones;
- the real-field cubes under `data/euclid_inference/real_fields/<field>/cubes`,
  made by members 145–168, none of them active.

The writers already discard such cubes, but only when they next run on that
bucket (Evaluate, a combiner/gate fit, a compare, a real-field refresh). Until
then they sit on disk, sometimes for weeks.

## Goal

Delete cached cubes as soon as they become unreusable, without the user
clicking anything, and without ever deleting a cube a reader could still use.

"Unreusable" means: the next writer of that cache would throw it away. The
purge decides this with the cache's own rules (checkpoint fingerprints,
records fingerprints, manifest field lists), so it can never disagree with
the writers.

## When it runs

Four jobs make cubes stale. Each, on success, records that a purge is due:

| Job | Why cubes go stale |
|---|---|
| `job_archive_member` | the archived member's cubes |
| `job_ensemble_pull` (not dry-run, ≥1 member changed) | changed members' cubes |
| FASRC ensemble mirror (`/api/fasrc/mirror/trigger`) | any member may have changed |
| records sync (`_job_sky_sync`) | buckets made from the old records |

The request is a small JSON file (`<data>/_stale_purge_pending.json`, reasons +
timestamp), so it survives a console restart.

The job registry calls a finish hook after every job. When a purge is pending
and no job is running, the hook starts one job, kind `storage-purge-stale`,
labelled "Storage: delete stale cubes". Its log lists what it deleted and the
bytes freed. Waiting for an idle registry means it never deletes cubes that a
running Evaluate, fit or compare is reading. Opening a page never starts it.

The purge job's own completion does not trigger the hook, so a failing purge
retries after the next other job, not in a loop. The pending file is cleared
only if no new request arrived while the purge ran.

## What it deletes

### 1. Ensemble member-cube buckets

For both regimes (starfull, starless) and all four buckets, given the active
members' labels and current checkpoint fingerprints
(`ensemble_registry.regime_labels`, `ensemble.member_fingerprints`) and the
current records fingerprint of the bucket's records:

1. **No manifest but files present**: delete the directory (no manifest says
   what made the files; every writer would wipe it).
2. **Records changed**: the bucket's recorded records fingerprint (manifest
   `records_fp`; blackout buckets: `identity.source`) differs from the current
   one → delete the directory. Skipped when the current fingerprint is
   unknown (records not synced locally). Current fingerprints are computed
   exactly as the writers do: test = `_eval_records_fingerprint(rdir,
   eval_subset(rdir), starless)`, validate and validate-blackout = the
   `"validate"` subset, blackout = the test fingerprint.
3. **Positional bucket**: migrate first with `migrate_positional_bucket`, adopting
   the fingerprints the writer would adopt (test bucket:
   `_proven_test_fingerprints`; others: none).
4. **Members**: a member whose label is not active, or whose recorded
   fingerprint differs from its current one, loses its cubes and its entry in
   `member_labels` / `member_fps`.
5. **Aggregates**: when step 4 removed any member, every per-field file that is
   not a member cube or `lr_` (mean `sr_`, `std_`, `pcaN_`, combiner outputs)
   is deleted, `has_combiner*` flags are cleared and `pca_amps`/`pca_var`
   emptied. The writers recompute them (Evaluate rewrites every field's
   aggregates; the validate fill recomputes a field without `sr_`).
6. **Orphan fields**: every per-field file of a field not in the manifest's
   `indices` is deleted, including blackout `lr_` files left from an earlier
   stamping.

The manifest is rewritten atomically in the same pass, so every label it lists
has current cubes. Readers and the next writer therefore see a consistent
bucket: the next fill infers only the members the purge removed.

The archive shortcut keeps working without inference: the next Evaluate no
longer finds the archived member in the bucket, so it takes the normal path,
which reads every remaining member from cache and recomputes the aggregates.
`_reconcile_combiner_on_archives` still runs.

### 2. Real-field cubes

Real-field caches are positional (`member<i>_<tile:03d>.npy`, labels in the
manifest). Changes:

- `cache_real_field` records `member_fps` in the manifest and reuses (hard-links)
  only members whose label AND fingerprint match. Today it matches by label
  only, so a continued member's old cubes are reused.
- `refresh_real_field_combiners` treats a fingerprint mismatch like a label
  mismatch ("member cache is stale").
- The purge, per field: a member that is inactive, has no recorded
  fingerprint, or whose fingerprint changed is dropped. The kept members are
  renumbered with the same hard-link remap `cache_real_field` uses; every
  membership-dependent file (members, `sr_`, `std_`, `pcaN_`, combiner outputs)
  is deleted; the manifest records the kept members (`member_labels`,
  `member_fps`, `run_members`), empties `combiner_kinds`, `combiner_state`,
  `pca_amps`, `pca_var`. `lr_` tiles, `raw/`, `original_stack.fits` and
  `diagnostics.json` stay (Sky tiles' field source needs the first three). The
  viewer then shows LR only until the next refresh, which re-infers the dropped
  members.

### 3. Experiments member-SR cache

`data/euclid_inference/experiments/cache/<source>/<tile>/<slug>.npy` + `.json`:
an entry whose recorded `member_fingerprint` is not the member's current
fingerprint (an archived member has none) is deleted with its sidecar.
Entries whose label is not an ensemble member label are left alone.

## Not touched

Eval-gallery and NEXUS SR files (overwritten in place), study fields (frozen,
own LRU), gate variant folders, tracking zips, the records, the FASRC fetch
cache (own LRU).

## Related fix

`build_blackout_fields` keeps an existing `lr_` file even when the field is new
to the bucket, then serves that old file next to freshly inferred member cubes
on the next run. It will overwrite `lr_` whenever the field was not in the
manifest's `indices`.

## Concurrency

- In-process: the purge starts only when the registry is idle. A job started
  during the (seconds-long) purge can race it; deletions on both sides tolerate
  an already-missing file (`sync_bucket_members`, `migrate_positional_bucket`
  and the purge use a quiet remove/replace).
- Two consoles on one data dir: each has its own registry, so a purge in one
  can run during a job in the other. It only deletes what the cache rules call
  stale, which a job started before the triggering change was already racing
  (archive deletes the checkpoint, pull replaces it). Accepted.

## Layout

- `euclid_polish/eval/ensemble_cube_cache.py`: `purge_stale_bucket(cubes_dir, *,
  name, current, records_fp, adopt) -> PurgeResult` — pure, file-system only.
- `euclid_polish/web/helpers/real_field.py`: `purge_stale_real_fields(current)`
  plus the fingerprint changes above.
- `euclid_polish/web/helpers/experiments.py`: `purge_stale_member_sr_cache(current_fn)`.
- `euclid_polish/web/helpers/stale_purge.py` (new): `request_stale_purge(reason)`,
  `purge_stale_caches(progress)` (computes labels, fingerprints, records
  fingerprints; calls the three purges; returns a report), `job_stale_purge`,
  `maybe_start_stale_purge(registry)`.
- `euclid_polish/web/jobs.py`: `JobRegistry.add_finish_hook(fn)`; `_runner`
  calls hooks after the job's status is final.
- Triggers wired in `ensemble_viz.job_archive_member`, `job_ensemble_pull`,
  the mirror trigger route and `views._job_sky_sync`; the hook is registered in
  app setup.

## Testing

- `purge_stale_bucket`: departed member, continued member, positional bucket
  (unproven → all members dropped; proven test fingerprints → kept), records
  change wipes, records unknown keeps, no-manifest wipe, aggregates dropped only
  on a membership change, orphan fields (incl. blackout `lr_`), missing files
  tolerated, bytes reported.
- Integration with the existing fake ensemble (`tests/_ensemble_cube_cache_fixtures.py`):
  Evaluate → continue one member → purge → Evaluate runs only that member and
  the mean equals the member mean; Evaluate → archive → purge → Evaluate runs
  no member.
- Real field: fingerprints recorded; a continued member is not reused; purge
  renumbers kept members and the viewer meta lists them.
- Experiments cache: stale entries removed, current and non-member entries kept.
- Trigger: request writes the pending file; the hook starts the job only when
  idle; the purge job's own finish does not retrigger; a request during the
  purge survives it.
- Blackout: a stale `lr_` for a field new to the bucket is overwritten.
