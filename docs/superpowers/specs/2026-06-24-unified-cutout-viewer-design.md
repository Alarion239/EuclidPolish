# Unified cutout viewer — design

**Date:** 2026-06-24
**Status:** approved (Layout A), implementing

## Problem

Every page that shows sky-field cutouts (LR / HR / SR, per-band VIS/NISP)
renders its own way: `/sky`, `/cutouts`, `/evaluation`, `/inference`,
`/roundtrip`, `/tng`, `/poster` each ship server-rendered matplotlib/PIL
PNGs with different (or no) controls. Every asinh/clip slider tick is a
matplotlib round-trip; raw pixels never reach the browser.

## Goal

One reusable, client-side cutout viewer embedded across the nav pages,
giving: **(1)** one image at a time, **(2)** fast next/prev with prefetch,
**(3)** color control — a single band (VIS/Y_E/J_E/H_E) **or** Lupton RGB
**or** temperature (blackbody-T) color, **(4)** live asinh-knee + brightness.

## Decisions (locked)

- **Render client-side.** Ship the raw N-band float cube to the browser;
  render in a `<canvas>` with plain JS recompute (no WebGL — stamps are
  53–256 px, sub-ms per frame). Controls become instant; no round-trips.
- **Scope:** convert `/sky`, `/cutouts`, `/evaluation` in one pass.
- **Packaging:** one embedded ES module (`static/cutout_viewer.js`), no
  build step. Matches the existing `static/*.js` convention.
- **All three color modes client-side**, including temperature.
- **Layout A** (toolbar on top), matching the existing `/sky`,
  `/evaluation` look: chips + sliders above a dark image frame, nav below.

## Architecture

### Backend — data endpoints

`euclid_polish/web/helpers/viewer_data.py` — a small **collection registry**.
Each collection exposes:

- `meta(params) -> dict`: `count`, `tiers` (`[{key,label}]`), `default_tier`,
  `band_names`, and (evaluation) a per-index `objects` list with each
  object's available tiers + label/grade.
- `cube(index, tier, params) -> (ndarray (H,W,C) float32, info)` where
  `info` = `{label, asinh, pixscale}`.

Three collections:

| collection   | params    | tiers                                  | source |
|--------------|-----------|----------------------------------------|--------|
| `sky`        | `subset`  | `dirty`→LR, `clean`→HR, `hr`→HR-target | TFRecords via `read_multiband_skyimages` (FASRC-synced cache) |
| `cutouts`    | —         | `real`→Euclid                          | per-band FITS, i-th valid-in-all-4 star, stacked to (H,W,4) |
| `evaluation` | —         | `LR`/`SR`/`HR` (per-object)            | `original_stack.fits` (4,53,53) / `SR.fits` (4,106,106) / `HR.fits` (syn only) under `EVAL_RESULTS_DIR` |

Band order everywhere: `Config.LR_INPUT_BAND_NAMES = (VIS, Y_E, J_E, H_E)`.

`euclid_polish/web/routes/viewer.py` (registered in `app.py`):

- `GET /viewer/meta/<collection>?<params>` → JSON: collection meta **plus**
  the global color constants the JS needs (per band `t_total_s`,
  `zeropoint_ab`, `solar_ab_mag`, `pivot_um`, `asinh_scale_e`; RGB scheme
  `vis_nisp = [H_E, J_E, VIS]`; `default_asinh = STRETCH_SCALE_E`).
- `GET /viewer/cube/<collection>/<index>?tier=…&<params>` → binary
  `Float32` body (H·W·C, C-order) with headers `X-Cube-Shape: H,W,C`,
  `X-Cube-Bands`, `X-Cube-Label`, `X-Cube-Asinh`, `X-Cube-Pixscale`.

~180 KB/cube at 106²×4 — fine raw; prefetched.

### Frontend — `static/cutout_viewer.js`

`mountCutoutViewer(rootEl, {collection, params})`:

- Fetches meta, builds toolbar (tier chips when >1 tier; color chips =
  bands + `Lupton` + `Temp`; `asinh knee` + `brightness` sliders), a dark
  square canvas frame (nearest-neighbour scaling so LR stays crisp), and a
  nav strip (prev / `i / N` / next, ← → keys, ⎵ play to auto-advance).
- Cube cache keyed `tier:index`; on settle, prefetch `index±1` (current
  tier) and the other tiers of `index` → instant flips.
- **Two-stage render** (keeps sliders buttery):
  1. *Prepare* (on cube/color change): per-pixel **color** + linear
     **intensity I**. band→gray(I=plane); Lupton→calibrated R/G/B channels
     + I=mean; Temp→per-pixel Planckian-locus hue (max=1) + I=mean
     calibrated intensity. Temp's per-pixel blackbody-T fit (96-T grid) runs
     here, ~10–30 ms, cached.
  2. *Transfer* (on slider change, cheap): one asinh map
     `t = arcsinh(I/K) / arcsinh(W/K)`, clipped [0,1], with **K = asinh
     knee**, **W = white point** (the "brightness" slider, inverted; higher
     brightness → lower W → brighter). gray→`255·t` (white-on-black);
     Lupton→rescale channels by `t/I` (hue-preserving); Temp→`srgb(hue·t)`.

K, W are in VIS-equivalent electrons (converted to calibrated units for
Lupton/Temp via `ab_flux_norm(VIS)·solar_balance(VIS)`), so **a fixed slider
setting means the same physical stretch across LR/SR/HR** — flipping tier at
identical stretch is the headline feature.

### Rendering parity

Color math is ported verbatim from `euclid_polish/visualization/color.py`
(`_ab_flux_norm`, `_solar_balance`, `_planckian_xy`, `_xy_to_linear_srgb`,
`_srgb_gamma_encode`, `fit_color_temperature`). The Temp mode with default
`W = 30·K` is byte-equivalent to `eye_rgb`. Lupton/gray adopt an explicit
black/knee/white asinh (absolute, not per-image percentile) — same family as
before, deliberately swapped for cross-tier comparability. A Node parity
check (`scripts/check_viewer_parity.mjs`) compares JS vs Python on sample
pixels.

### Page integration

- `/sky`, `/cutouts`: replace the `toolbar` + `viz-area` blocks with a
  viewer mount; keep the sync forms / action cards / inspect links.
- `/evaluation`: add the viewer as the primary LR/SR/HR cutout browser at
  the top of the results; existing thumbnails jump the viewer to that object.

## Out of scope (this pass)

`/inference`, `/roundtrip`, `/tng`, `/poster` (pre-rendered job artifacts);
HST experimental lanes. The component is built to extend to them later.
