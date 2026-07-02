# Ensemble-Only Model Architecture

**Date:** 2026-07-01
**Status:** Approved

## Problem

The codebase mixes two model representations: a legacy single model
(`ckpt/wdsr`, plus a stale VIS-only variant `ckpt/wdsr-vis`) and the
ensemble (`ckpt/ensemble/member_NN/`). Eval, inference, and the WebUI
branch between them at runtime (`load_eval_ensemble_or_single()`,
`hasattr(model, "member_arrays")` checks, `vis_only` toggles,
`_ckpt_dir_for_kind()`). A single model is just an ensemble of size 1,
so the split buys nothing and costs duplicated paths and staleness
bugs.

Additionally there is no way to retire an ensemble member: discovery
is a bare directory glob, so every trained member participates in the
ensemble mean forever, and per-member caches (cube cache, eval
products) have no invalidation story when membership changes.

## Decision

Ensemble-only. `EnsembleModel` is the single public model struct;
`Model` is demoted to the internal per-member implementation. The
legacy single-model checkpoints are zipped and archived into the
tracking store. The ensemble page gains per-member archiving backed by
a registry file, and caches keyed by membership are invalidated lazily
on read.

Chosen approach: **registry file as source of truth** (over pure
filesystem-glob discovery, and over integrating with the
provenance/UUID branch, which stays off main).

## Design

### 1. Ensemble-only model struct

- `EnsembleModel` (`euclid_polish/ensemble.py`) is the only model type
  routes, eval runners, and inference helpers construct.
- `Model` (`euclid_polish/model.py`) remains as the internal
  per-member loader/trainer used by `EnsembleModel`; no route or eval
  code imports it for inference.
- `load_eval_ensemble_or_single()` → `load_eval_ensemble()` in
  `euclid_polish/eval/ensemble_infer.py`. Always returns an
  `EnsembleModel` built from the registry's active members; raises a
  clear error if no active members exist.
- All `hasattr(model, "member_arrays")` branches are removed.
  Disagreement artifacts (std.fits, pca{0,1,2}.fits, member cubes) are
  gated on `model.n_members > 1`, not on model type. A 1-member
  ensemble therefore evaluates exactly like the old single model —
  no all-zero std cubes, and `can_reuse_eval_object()` does not demand
  disagreement files for it.

### 2. Registry

- File: `<ensemble_dir>/registry.json`.
- Shape:

  ```json
  {
    "active": ["member_00", "member_01"],
    "archived": [
      {"name": "member_02", "archived_at": "...", "zip": "...", "commit": "..."}
    ]
  }
  ```

- New module `euclid_polish/ensemble_registry.py` owns load, save,
  bootstrap, and the archive transition.
- Bootstrap rule: on load, any `member_*` directory on disk with a
  checkpoint that is not mentioned in the registry (neither active nor
  archived) is auto-appended to `active` and the file is saved. A
  missing registry file is equivalent to an empty one plus bootstrap.
  This keeps FASRC-synced members working with zero manual steps.
- Discovery everywhere (`ensemble_available()`, `ensemble_status()`,
  `EnsembleModel` loading) = registry active list, verified against
  disk (an active entry whose directory vanished is dropped with a
  warning, not an error).

### 3. Zip archive into tracking

- New `TrackingStore.archive_model_zip(src_dir, name, note=None)` in
  `euclid_polish/tracking/store.py`: zips the full directory tree into
  `tracking/current/models/<name>.zip` with a `meta.json` sidecar
  (source path, git commit, timestamp, file manifest with sizes).
- Caller (archive route / migration script) deletes the source
  directory after a successful zip.
- Campaign save/archive and the holylabs rsync mirror are unchanged —
  zips ride along like any other file under `models/`.

### 4. One-shot migration of the legacy single model

- `scripts/migrate_single_model.py`: for each of `ckpt/wdsr` and
  `ckpt/wdsr-vis` that exists, `archive_model_zip()` it into the
  current campaign, append a campaign log entry, then delete the local
  directory. Idempotent (skips what is already gone).
- Config cleanup: the ensemble dir becomes the primary anchor.
  `DEFAULT_CHECKPOINT_DIR` survives only as the path seed from which
  `ensemble_dir()` is derived (env override `EUCLID_POLISH_CKPT_DIR`
  keeps working); nothing loads a model from it directly.
- Removed: `vis_only` toggle on /inference, `-vis` directory scanning
  in `_checkpoints_status()`, `_ckpt_dir_for_kind()`'s single-model
  resolution. The /inference model choice reduces to the ensemble
  (member tracks stay an internal detail).
- FASRC-side `ckpt/wdsr` is NOT touched automatically; the migration
  logs a reminder in the campaign log.

### 5. Ensemble page as model manager

- Member list rendered from the registry: per-member row with name,
  seed, tracks (psnr / +loss_best), on-disk size, and an **Archive**
  button.
- POST `/ensemble/archive-member` (member name in form): zip to
  tracking → registry transition (active → archived tombstone) →
  delete member dir → eager prune of cheap per-member caches (cube
  cache entries and PNGs that name the member).
- Archived members shown in a collapsed history section from the
  tombstones (name, date, zip location).
- `/training` page content (TFRecord status, train-job submission
  controls incl. n_members/steps/LR-schedule knobs) moves onto
  `/ensemble`; `/training` becomes a redirect to `/ensemble`.

### 6. Lazy cache invalidation

- Membership fingerprint = sorted active member labels (e.g.
  `["00·psnr", "00·loss", ...]`), as already recorded in
  `viz_index.json` and `eval_summary.json`.
- `load_cached_member_stack()` (`eval/ensemble_cube_cache.py`):
  if the manifest's `member_labels` contain a label whose member is
  not active, delete the stale cube `.npy` files and the manifest,
  return a cache miss. Invoked lazily on any read.
- `eval_summary.json` reads on the ensemble page get the same check
  (marked stale in the UI rather than deleted silently — it is a
  small summary, and showing "stale: membership changed" is more
  useful than a blank).
- Eval-object reuse (`can_reuse_eval_object()` /
  `grouped_runner.py`): ensemble products (SR from mean, std/pca)
  written by an ensemble whose membership no longer matches are
  regenerated. Implementation: eval objects record the membership
  fingerprint in their existing per-object metadata; mismatch → not
  reusable.
- In-memory: `EnsembleModel` construction reads the registry, so
  archived members are simply never loaded.

### 7. Testing

- Registry: bootstrap from bare directories, archive transition,
  vanished-directory tolerance, active-label computation.
- Zip archive: round-trip (archive → extract → checkpoint file set
  identical), meta.json contents.
- Cache invalidation: manifest with archived label → files deleted,
  miss returned; matching manifest → hit.
- Loader: ensemble-of-1 loads and predicts; zero members raises the
  clear error.
- Existing eval tests migrated off `load_eval_ensemble_or_single`.

## Out of scope

- FASRC remote checkpoint cleanup (logged, manual).
- Un-archiving a member from a zip (the zip + tombstone make it
  possible by hand; no UI).
- Provenance/UUID integration (separate branch).
- Renumbering members after archiving — names are stable forever;
  `MEMBER_DIR_FMT` numbering just continues past gaps.
