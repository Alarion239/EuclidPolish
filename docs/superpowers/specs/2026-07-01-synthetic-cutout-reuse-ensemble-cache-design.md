# Reuse cached test-field ensemble cubes for synthetic cutouts

Date: 2026-07-01
Status: approved (pending spec review)

## Goal

Cut the cost of the synthetic evaluation subgroups (`syn-lens`, `syn-gal`) by **reusing the
ensemble page's already-cached per-test-field cubes** instead of re-running the CNN ensemble on
those fields. Synthetic cutouts are source-centered stamps cropped from test fields; the ensemble
page already runs the ensemble on those fields and caches per-field member cubes. Cropping the
cached members to each source stamp reproduces the exact same cutout with **zero CNN inference**.

## Decisions (from brainstorming)

- Applies to **both** `syn-lens` and `syn-gal` (same fields, same mechanism).
- **Fall back to inference** when a source's field is not in the ensemble cube cache (beyond the
  200-field cap, ensemble eval not yet run, or subset mismatch). Non-destructive; correctness and
  coverage preserved.
- Reuse the cached **member** cubes and recompute per-stamp PCA (Approach A) — **bit-identical** to
  the current synthetic output. (Rejected Approach B: cropping field-level `sr`/`std`/`pca` cubes
  directly — cheaper but approximate, since it would use field-level PCA modes rather than the
  stamp's own PCA.)

## Current state (verified)

- `euclid_polish/eval/synthetic_runner.py::run_synthetic_eval`: for each source it reads the field's
  LR/HR records, runs `sr_from_model(model, lr_cube)` on the **full field**, then `crop_stamp(...,
  cx=x_pix, cy=y_pix, m=EVAL_HR_SIZE)` per band to the source stamp; writes `SR.fits`/`HR.fits`/
  `original_stack.fits`; and (P2-T6) crops the member stack the same way and calls
  `write_disagreement_cubes(obj_dir, mem_st)` (per-stamp PCA). It already selects the subset via
  `eval_subset(rdir)` → `test` when `dirty_test` exists, else `validate`, and reads
  `sources_{sub}.csv`. Fields are keyed by 0-based record index (`Image.index` == CSV `field_index`).
- `euclid_polish/web/helpers/ensemble_viz.py::job_ensemble_evaluate`: evaluates the same subset
  (`eval_subset`), and for every evaluated field (cap `ENSEMBLE_VIZ_FIELDS_MAX = 200`) caches
  `sr_{rec:05d}.npy`, `std_…`, `pca0..2_…`, and **`member{i}_{rec:05d}.npy`** into
  `data/vis/ensemble/cubes/`, plus `viz_index.json` = `{subset, indices, pca_n, pca_amps,
  member_labels}`. The cube dir is wiped and rewritten on each ensemble eval.
- Geometry matches: `x_pix/y_pix` are HR-grid pixels; cached cubes are full-field HR-grid `(H,W,C)`;
  `crop_stamp(cube, cx=x_pix, cy=y_pix, m=EVAL_HR_SIZE)` extracts the identical stamp. `crop_stamp`
  is `plane[y0:y0+m, x0:x0+m]` with `x0=round(cx)-m//2`.

## Design

### Component 1 — `euclid_polish/eval/ensemble_cube_cache.py` (new)

```
load_cached_member_stack(field_index: int, *, subset: str,
                         cubes_dir: str | None = None) -> np.ndarray | None
```

- `cubes_dir` defaults to `os.path.join(Config.VIS_DIR, "ensemble", "cubes")` — computed directly
  from `Config` so the `eval` module does not import the `web` layer (avoids an eval→web
  dependency). This must match `viewer_data._ensemble_cubes_dir()`; add a comment cross-referencing
  it so the two stay in sync.
- Reads `viz_index.json`. Returns `None` unless: the manifest loads, its `subset` equals the
  `subset` argument, and `field_index` is in `indices`.
- Determines member count from `len(member_labels)` (fallback: count `member*_{field_index:05d}.npy`
  files). Loads `member{i}_{field_index:05d}.npy` for `i` in range via `np.load(path,
  mmap_mode="r")` (only the cropped rows page in downstream) and stacks to `(M, H, W, C)`.
- Any error (missing manifest/file, JSON/OS error, zero members) → `None`. Never raises.
- Pure w.r.t. inputs besides disk reads; unit-testable with fake cubes + manifest.

### Component 2 — `synthetic_runner` hook (minimal)

At the per-field branch (`if idx != cur_idx:`), before calling `sr_from_model`:

1. `cached = load_cached_member_stack(idx, subset=sub)`.
2. If `cached is not None`: set `members_full = cached`, `sr_arr = members_full.mean(axis=0)`, and
   **skip** `sr_from_model`. Log `field {idx}: reused ensemble cache`.
3. Else: current path — `_, sr_data, members_full = sr_from_model(model, lr_cube)`;
   `sr_arr = np.asarray(sr_data, np.float32)`. Log `field {idx}: inference`.

Everything downstream is unchanged: crop `sr_arr` → `sr_cube_st`, crop `members_full` → `mem_st`,
`write_disagreement_cubes(obj_dir, mem_st)`, and the LR/HR/SR stamp writes + flux metrics. LR/HR
stamps remain cheap crops of the records.

Note: reuse keys purely on cache presence + subset match, so a cached field's synthetic products
come from the cached ensemble cubes even if the grouped run's selected model differs — this is the
intended cost-saving (the cache is the source of truth for the ensemble disagreement products).

## Isolation / boundaries

- The cache reader is a single self-contained function with a clear `-> ndarray | None` contract;
  the synthetic runner only branches on hit/miss. No change to `sr_from_model`,
  `write_disagreement_cubes`, `enforce_object_sizes`, or the viewer.
- Failure is contained: a bad/absent cache degrades to the existing inference path.

## Testing

- `load_cached_member_stack`: (a) hit returns `(M,H,W,C)` equal to the written members; (b) miss
  returns `None` for — uncached `field_index`, subset mismatch, missing `viz_index.json`, a missing
  `member{i}` file.
- Integration (synthetic path, no real model/GPU): with a fake cached field, assert the emitted
  `std`/`pca`/SR stamp cubes equal cropping the cached members through `write_disagreement_cubes`,
  **and** that `sr_from_model` is not called for the cached field (monkeypatch it to raise). Assert
  the uncached field still routes through `sr_from_model`.
- Full suite stays green.

## Cost / dependency

- Savings: synthetic cutouts whose field is cached skip CNN inference entirely (the expensive part);
  only the small per-stamp SVD + record crops remain. With the ensemble having processed the test
  fields, essentially all synthetic inference is eliminated (bounded by the 200-field cap).
- Workflow dependency (document in the runner log + a code comment): the ensemble page wipes and
  rewrites its cube cache each eval, so the intended order is **run the ensemble page test eval,
  then the grouped evaluation**. Uncached fields fall back to inference automatically.

## Out of scope

- Real objects (A/B/C lenses, real-gal) — archive cutouts at (ra,dec), not test-field crops.
- Changing the ensemble page's caching, the cap, or its cache lifetime.
- Sharing a single field-cube cache bidirectionally between the two pages beyond this read path.
