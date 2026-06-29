# FastEPSFBuilder — Design Spec

**Date:** 2026-06-29
**Status:** Approved, implementing
**Scope:** Speed up empirical PSF (ePSF) extraction with a numerically-identical
drop-in for photutils' `EPSFBuilder`.

## Problem

PSF extraction builds one ePSF per spatial cluster per band via
photutils `EPSFBuilder` (`euclid_polish/psf/psf_extractor.py:build_epsf_from_stars`).
Profiling one production-config build (100 stars, `oversampling=2`,
`maxiters=10`, 511×511 ePSF grid) shows **28.3 s**, of which **~71 % is
`scipy.interpolate.RectBivariateSpline.__call__`**.

Root cause: photutils' `_LegacyEPSFModel.evaluate` evaluates the ePSF model
with `interpolator.ev(xi, yi)` — FITPACK's *scattered* point-by-point path —
over each star's ~65 k sample points. But those points are a **separable
grid** (the cutout index grid shifted by a scalar centroid offset). Evaluated
the right way they cost far less:

| Method (511² grid, 65 k points, per star) | ms/call | speedup |
|---|---|---|
| `RectBivariateSpline.ev()` — *photutils path* | 15.94 | 1× |
| `RectBivariateSpline.__call__(grid=True)` | 0.38 | **42×** |
| `map_coordinates` order 3 (bicubic) | 4.53 | 3.5× |
| Julia `Interpolations.jl` cubic BSpline | 0.47 | 34× |

The gap is an **API misuse inside photutils, not a language ceiling**: scipy
used correctly (`grid=True`) matches/beats Julia. A Julia or Numba rewrite
buys nothing here and costs a two-language codebase around a correctness-
critical algorithm — rejected.

The two methods that hit the slow `.ev()` path:
- `EPSFBuilder._resample_residual` (epsf.py:437) — ~57 % of build time.
- `EPSFBuilder._recenter_epsf` (epsf.py:576) — ~15 % of build time.

## Goal

One ePSF build ~28 s → ~10–14 s (**~2–3×**), output numerically identical to
stock photutils (matches to floating-point round-off, ~1e-12). This is a
one-time, cached prep step — pure throughput, zero behavioral change.

## Design (Approach B: subclass the builder)

New module `euclid_polish/psf/fast_epsf.py`:

- **`_separable_eval(epsf, x, y, x_0, y_0)`** — mirrors
  `_LegacyEPSFModel.evaluate`, but when `(x, y)` reconstruct to a full
  separable grid it extracts the 1-D axes and calls
  `epsf.interpolator(xi_axis, yi_axis)` (FITPACK grid path; respects the
  `data.T` axis order used when the interpolator is built), then reshapes to
  match `.ev()`'s output ordering. Returns `None` when the inputs are not a
  clean grid, so callers fall back to stock behaviour.

- **`FastEPSFBuilder(EPSFBuilder)`** — overrides exactly two methods:
  - `_resample_residual(star, epsf)` *(Phase 1)* — fast path when the star is
    fully unmasked (its `_xidx_centered`/`_yidx_centered` then form a complete
    grid); otherwise delegate to `super()`.
  - `_recenter_epsf(...)` *(Phase 2)* — the eval grid is `np.indices(...)`,
    always separable, so always fast path.

  Everything else — sigma-clipping, stacking, convergence, smoothing, the
  `_LegacyEPSFModel` plumbing — is inherited unchanged.

### Wiring

`psf_extractor.build_epsf_from_stars` selects `FastEPSFBuilder` vs stock
`EPSFBuilder` via `Config.PSF_FAST_EPSF_BUILDER` (default **True**), with an
escape hatch to revert instantly if production output ever looks off.

## Correctness gate

A golden-output test asserts `FastEPSFBuilder` produces the same ePSF as stock
`EPSFBuilder`:
- Cases: fully-unmasked stars (fast path) **and** masked/NaN stars (fallback).
- `np.allclose(fast.data, stock.data, rtol=1e-10, atol=1e-12)`.
- Also assert the `_separable_eval` helper matches `epsf.evaluate` directly on
  a known grid.

## Validation

Benchmark on one real cluster of actual star cutouts to confirm ~2–3× and to
measure the fraction of real stars taking the fast path (NaN-masked stars fall
back).

## Phasing

- **Phase 1:** `_resample_residual` only — ~2×, smallest surface. Ship + validate.
- **Phase 2:** add `_recenter_epsf` — ~3×, gated on Phase 1 + golden test holding.

## Out of scope

Deepcopy/glue overhead (~10 %), `map_coordinates` alternatives, any change to
star selection / clustering / background cleaning.

## Risks

1. FITPACK grid vs scattered may differ at ~1e-12 — covered by the golden-test
   tolerance.
2. Heavily-masked stars get no speedup — measured in validation; not a
   correctness issue.
