# Ensemble → Evaluation integration: power-spectrum parity, mean-model default, disagreement movies

Date: 2026-07-01
Status: approved (pending spec review)

## Goal

Two related pieces of work:

1. **Power-spectrum parity.** Restyle the *ensemble page's* power-spectrum plot to match
   the *evaluation page's* look (θ log x-axis, LR-Nyquist line, PSF-FWHM guide, horizontal
   1.0 reference), showing — on the **test set** — each member's performance line, the
   ensemble mean's line, and the cross-correlation.
2. **Wire the ensemble into evaluation.** Make the **ensemble mean the default evaluation
   model**, and surface the ensemble **disagreement movie** in the evaluation viewer for
   **all evaluated objects** (real lens A/B/C, real-gal, syn-lens, syn-gal).

The evaluation page's own sky-field power-spectrum plot (`render_power_spectrum_summary`)
is **unchanged**. Only the ensemble page's plot is restyled.

## Decisions (from brainstorming)

- Power-spectrum plot: **VIS band, T(k) + r(k) panels**, eval-page aesthetics.
- Evaluation model: **ensemble mean is the default** (with graceful fallback — see below).
- Movie coverage: **all evaluated objects**.
- "Each model's performance line" = **transfer function T(k) = √(P_SR/P_HR)**; "cross-correlation"
  = **r(k) = P_HR×SR / √(P_HR·P_SR)**.
- HR-free coherence `ρ(k)`: **off by default** in the new plot (faithful to eval page); may be
  retained later as an optional dashed companion on the r panel.

## Current state (reference)

- Eval-page PS: `euclid_polish/eval/power_spectrum.py::render_power_spectrum_summary` — 2×2
  (linear/asinh × T/r), 4 bands, ±1σ, sky validation fields. Helpers: `tukey_window_2d`,
  `cross_power_2d`, `k_magnitude_2d`, `log_k_edges`, `bin_powers`, `ratios_from_powers`,
  `BandStat`, band colors, θ conversion, PSF-FWHM guides.
- Ensemble-page PS: `render_ensemble_power_spectrum` — 2-panel (P(k) + coherence), VIS only,
  test cutouts. Data via `EnsembleSpectrumAccumulator.curves()` already includes
  `k, P_hr, P_sr, P_disagree, r, T, coherent_frac, rho, P_members (M×nbins), r_members (M×nbins)`.
- Ensemble mean model: `euclid_polish/ensemble.py::EnsembleModel` — `predict()→(mean,std)`,
  `member_arrays()→(M,H,W,C)`, `upsample()` returns the mean `Image` (drop-in for `Model.upsample`).
  Factory `load_ensemble(base_dir=<ckpt>/ensemble, include_loss_best=True)`.
- Eval model wiring: `euclid_polish/eval/grouped_runner.py::run_grouped_analysis(..., model=None,
  checkpoint=None)` — loads a single `Model` from `Config.DEFAULT_CHECKPOINT_DIR` via
  `catalog_runner.load_eval_model`. Accepts an existing `model=` instance already.
- Eval viewer: `cutout_viewer.js` already has the `morph` tier (`startMorph`, driven by
  `mean + Σ aᵢ·sin(2π fᵢ t)·componentᵢ`, sliders `morphAmp`/`morphSpeed`). Eval viewer meta
  (`viewer_data.py::_eval_meta`/`_eval_cube`) exposes only LR/SR/HR today. Ensemble viewer
  (`_ensemble_meta`/`_ensemble_cube`) already exposes `morph` + `pca_n` + `pca_amps` + std/pca cubes.
- Eval groups: `synthetic_runner._SUBGROUPS = (("syn-lens","lens"),("syn-gal","galaxy"))`;
  real groups keyed by manifest `grade` ∈ {A,B,C,gal}. Both syn groups are source-centered.

## Design

### §1 — Ensemble power-spectrum plot (VIS, T + r), eval-page style

New renderer (replacing `render_ensemble_power_spectrum`) producing a two-panel VIS figure,
reusing the eval-page plot helpers for a consistent look:

- **Left — T(k):** per-member faint lines `T_member = √(P_members / P_hr)`; ensemble **mean bold**
  `T = √(P_sr / P_hr)`. Horizontal `T=1` reference.
- **Right — r(k):** per-member `r_members` faint; ensemble **mean `r` bold**. Horizontal `r=1`
  reference.
- Shared: θ = 1/(2k) log x-axis, LR-Nyquist vertical line, VIS PSF-FWHM guide, VIS band color,
  n_fields annotation. Test-set source (unchanged from current ensemble eval).

All inputs already come from `EnsembleSpectrumAccumulator.curves()`. No new spectral math; only
the derived per-member `T` (from `P_members` and `P_hr`). The JSON sidecar
(`ensemble_power_spectrum.json`) is retained as-is.

`ρ(k)` remains computed in `curves()` but is not plotted by default.

### §2 — Ensemble mean as default evaluation model + per-object disagreement cubes

Because obtaining the mean requires running all M members, the same pass yields the
disagreement/PCA cubes — one code path serves both asks.

- **Model loading:** `run_grouped_analysis` (and the `/api/evaluation/run-grouped` route) default
  to `load_ensemble()`. **Graceful fallback:** if the ensemble dir is missing/empty (no members),
  fall back to the single `Config.DEFAULT_CHECKPOINT_DIR` model and log the fallback; movies are
  then unavailable for that run.
- **Per-object outputs (ensemble path):** for each evaluated object compute
  `preds = member_arrays(lr)`, `mean = preds.mean(0)`, `std = preds.std(0)`,
  `(_, comps, amps) = pca_field(preds, ENSEMBLE_PCA_COMPONENTS)`. Write:
  - `SR.fits` ← ensemble **mean** (canonical SR; feeds all metrics + existing PNG renderer).
  - `std.npy`, `pca0.npy`, `pca1.npy`, `pca2.npy` in the object's out dir.
  - `pca_n` and per-object `pca_amps` written to a per-object sidecar `disagreement.json` in the
    object's out dir (avoids churning the grouped-manifest schema; `_eval_meta` reads these back).
- **Coverage:** all groups (A/B/C, gal, syn-lens, syn-gal), since it is per-object. Movies need
  only the members (no HR required).
- **Metrics:** unchanged in definition; SR is now the ensemble mean.

### §3 — Movie tier in the evaluation viewer

- `viewer_data.py::_eval_meta`: advertise the `morph` tier + `pca_n` + per-object `pca_amps`
  whenever the cached cubes exist for the evaluated objects.
- `viewer_data.py::_eval_cube`: serve `std` / `pca0..2` tiers from each object's out dir
  (alongside the existing LR/SR/HR).
- **No JS changes** — `cutout_viewer.js` plays the movie identically for lenses and galaxies;
  group membership only filters the gallery.

## Isolation / boundaries

- The new PS renderer is a pure function `(out_png, curves, n_fields) → out_png`, testable from
  synthetic `curves()` dicts; it shares low-level helpers with the eval-page renderer but does not
  alter it.
- The ensemble-eval per-object cube emission is confined to the ensemble branch of the grouped
  runner; the single-model path is untouched (fallback preserves old behavior exactly).
- Viewer changes are additive: new tiers appear only when cubes exist.

## Testing

- New PS renderer: produces a PNG from a synthetic `curves()` dict; per-member `T` derivation
  (`√(P_members/P_hr)`) is correct; handles `M<2` (no per-member lines) and empty/NaN bins.
- Grouped runner: uses ensemble by default; **falls back to single model** when the ensemble dir
  is absent/empty (no crash, logged).
- Ensemble path emits `std`/`pca0..2` cubes + `pca_amps` for a small synthetic object set.
- `_eval_meta` advertises `morph` only when cubes are present; `_eval_cube` serves `std`/`pca*`.
- Existing eval-page sky PS test and ensemble PS JSON test remain green.

## Cost / risks

- Evaluation becomes ~M× slower (all members per object); surfaced in the UI/log.
- Extra disk: 4 cubes/object (`std` + 3 PCA) across all evaluated objects.
- Fallback ensures eval still works with no trained ensemble.

## Out of scope

- Changing the eval-page sky-field PS plot.
- A UI dropdown to pick between single/ensemble (ensemble is simply the default; single is the
  automatic fallback). Can be added later if desired.
- Incommensurate-frequency / member-spline morph trajectory experiments (separate track).
