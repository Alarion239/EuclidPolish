# Unified grouped evaluation — design

## Context

Today the `/evaluation` page runs one catalog (a single lens grade) and scores
Zoobot morphology on it. We want a **unified, grouped** analysis that compares
the model's behaviour across four cohorts in one run, and includes the
synthetic validation set (which has HR ground truth) as a group called
`synthetic`. This lets us see whether SR's effect on morphology / photometry
differs between lens grades and the controlled synthetic case where we *can*
measure recovery against truth.

## Three-step flow (one run dir)

1. **Prepare** (N per group, cutout size) → one run dir `eval_results/<run>/` with
   one sub-dir per object and a single `manifest.csv` whose **`grade` column is
   the group** ∈ {`A`, `B`, `C`, `synthetic`}:
   - **A / B / C** — N real lens cutouts each, fetched at the chosen cutout size
     and run through SR. **LR + SR** only (real Euclid has no HR). Reuses the
     catalog-eval loop.
   - **synthetic** — a **seeded random N** of the cached validation records
     (`dirty/hr/clean_validate.tfrecord`, resolved via `_sky_records_local_dir`),
     run through SR. **LR + SR + HR** (HR = `hr_validate` VIS). Only this group
     gets the toward-HR morphology metric and SR-vs-HR quality.
   - Runs **locally, in-process** (TF is in the WebUI env) as a background job
     with a progress bar + log, using the local checkpoint (`./ckpt/wdsr`).
2. **Run Zoobot** once over the whole run (existing local job): before/after for
   all; +HR for the synthetic subset.
3. **Plots, color-coded by group:**
   - **Morphology summary** (existing panel) — Pearson histogram + PCA shift map
     now **colored by group**, joined to `grade` from the manifest.
   - **New "Transformation" panel:**
     - *SR-vs-HR recovery* (synthetic only): paired PSNR of LR-vs-HR vs SR-vs-HR
       (does SR move the image toward truth).
     - *Flux & basic stats by group*: box/violin of flux ratio Σ SR/LR per group.
     - *Example triptych strip*: one example per group — LR | SR for lenses,
       LR | SR | HR for synthetic.

## Components

- `euclid_polish/eval/catalog_runner.py` — extract the per-object loop into a
  reusable `_eval_rows(...)` that yields manifest dicts (tagged with `group`);
  `run_catalog_eval` keeps its single-grade behaviour.
- `euclid_polish/eval/synthetic_runner.py` (new) — `run_synthetic_eval(...)`:
  load N validation triptychs → reconstruct → write `original_stack.fits` /
  `SR.fits` / `HR.fits`; compute `flux_ratio_sr_over_lr`, `psnr_lr_hr`,
  `psnr_sr_hr`.
- `euclid_polish/eval/grouped_runner.py` (new) — `run_grouped_analysis(...)`:
  load model once, run A/B/C (catalog loop) + synthetic, aggregate into one
  `manifest.csv` (columns: `id, ra, dec, grade, ok, error, out_subdir,
  lr_total_e, sr_total_e, flux_ratio_sr_over_lr, psnr_lr_hr, psnr_sr_hr`).
- `euclid_polish/eval/zoobot_morph.py` — `render_morphology_summary` colours by
  group (reads `grade` from the manifest); new
  `render_transformation_summary(run_dir, out_png)`.
- `routes/evaluation.py` — `/api/evaluation/run-grouped` (spawn the local job);
  `/api/evaluation/transformation?run=` (render+serve the transform PNG,
  cached, `?fresh`).
- `evaluation.html` — a "Prepare grouped analysis" form (N, cutout size, run
  name) with the shared progress/log panel; a Transformation panel beside the
  Morphology panel.

## Notes / constraints

- Real lenses have no HR by construction — PSNR columns are blank for A/B/C; the
  SR-vs-HR panel shows only `synthetic`.
- `cutout_size` applies to the lens fetch; synthetic uses the validation
  records' native size.
- Everything is local/CPU; no FASRC. Auto-fetches the lens catalog if missing.
- Verification: a tiny run (N=2) produces 8 object dirs (2×A/B/C + 2 synthetic)
  + one manifest; Zoobot scores them; both summary PNGs render and color by
  group. Unit tests stub the model/fetch so no network/weights are needed.
