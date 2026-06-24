# Lens-finder /evaluation preprocessing parity with training

**Date:** 2026-06-24
**Status:** Approved (design)
**Components:** `scripts/lensfinder_score_eval.py`, `euclid_polish/lensfinder/stamps.py`,
`euclid_polish/eval/synthetic_runner.py`, tests

## Problem / motivation

Training stamps recently switched to **106 px source-centered crops** rendered as
**4-band Lupton-asinh RGB** (`lensfinder_build_stamps.py` + `lensfinder/stamps.py`,
commits `d09af9f`/`3497d85`). The `/evaluation` scoring path
(`lensfinder_score_eval.py`) was left behind: it feeds the trained heads a *different*
input distribution than they were trained on, which silently degrades P(lens) scores
and the LR-vs-SR discovery-gain comparison.

Concretely, for each eval object the scorer calls
`zoobot_morph.render_vis_png(SR.fits | original_stack.fits | HR.fits)`, which:

- **does not crop** — it renders the whole eval frame (SR 128 px / LR 64 px), *larger*
  than the training stamp (SR/HR 106 / LR 53), so the source fills a different fraction
  of the frame; and
- renders **VIS-only grayscale**, percentile-normalized — not the 4-band Lupton-asinh
  RGB the heads trained on.

The downstream `get_galaxy_transform(minimal_view_config())` (resize→424 + Zoobot
normalization) is *already identical* between train and eval; only the crop + render
diverge.

## Goal

Preprocess eval cutouts **identically to training**, so every head sees the input
distribution it was trained on.

## Non-goals

- Re-running the trainer or changing the heads.
- The `/inference` synthetic `HR.fits` writer (`jobs_impl.py`) — separate flow,
  different directory.
- The morphology Zoobot script (`zoobot_morphology.py`) — separate feature; reads the
  VIS plane via `load_vis_plane`, unaffected by a 4-band HR.
- A `/evaluation` UI knob for stamp size, or a shared Config constant for it.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| HR (eval `HR.fits` is VIS-only 2-D; heads trained on 4-band) | **Persist 4-band HR** in `synthetic_runner` so HR also gets a true 4-band Lupton render. Existing `eval_results` need regen. |
| `--stamp-m` wiring | **CLI default 106**, matching `build_stamps`. No UI field, no Config constant. |
| Crop center | **Geometric center** of each eval FITS — eval cutouts are source-centered by construction (synthetic & real catalog cutouts both center the target). |
| Crop-time vs regen | **Crop at score time** (non-destructive; works on existing LR/SR). |

## Current vs target preprocessing

| Stage | Training (`build_stamps`) | Eval today | Eval target |
|---|---|---|---|
| Crop | source-centered, SR/HR 106, LR 53 | none (full 128/64 frame) | center-crop SR/HR→106, LR→53 |
| Render | 4-band Lupton RGB → 424 px | VIS-only grayscale → 424 px | **4-band Lupton RGB → 424 px** (same params) |
| Transform | `get_galaxy_transform(minimal_view_config)` | same | same (unchanged) |

Render params copied from training: B=VIS, G=mean(Y_E,J_E), R=H_E; `stretch =
Config.STRETCH_SCALE_E` (100), `Q = 8`, `scale_* = 1.0`, `size = 424`.

## Changes

### 1. `euclid_polish/eval/synthetic_runner.py` — 4-band HR

Today HR is collapsed to VIS: `hr_vis = hr_raw[..., 0]`, `hr_st = crop_stamp(hr_vis, …)`,
written 2-D. Change to crop **all** HR bands (stack like SR) into `hr_cube_st` (m, m, 4)
and write `np.moveaxis(hr_cube_st, -1, 0)` with the same `BANDS` header SR/LR use. Keep
the VIS plane `hr_cube_st[..., 0]` for the existing PSNR metrics (`psnr_lr_hr`,
`psnr_sr_hr`). The HR records are already 4-band; only the discard changes.

### 2. `euclid_polish/lensfinder/stamps.py` — shared eval-render helper

Add two pure functions (numpy/astropy/PIL only — no torch; unit-testable in the main env):

- `load_fits_cube(path) -> np.ndarray` — read a FITS primary HDU and return **band-last**
  `(H, W, C)`. Handles `(C, H, W)` (moveaxis) and 2-D `(H, W)` → `(H, W, 1)`.
- `render_eval_stamp(fits_path, out_png, *, crop_m, scale_r=1.0, scale_g=1.0,
  scale_b=1.0, stretch=Config.STRETCH_SCALE_E, Q=8.0, size=424) -> str` —
  `load_fits_cube` → center-crop every band to `crop_m` via `crop_stamp` at the
  geometric center → `render_stamp_rgb(...)`. This *reuses* `render_stamp_rgb`, so the
  eval render is byte-for-byte the training render. Defensive fallback: if the cube has
  `< 4` bands (e.g. a legacy VIS-only HR.fits not yet regenerated), replicate the single
  band to four so the render still produces a sensible grayscale composite rather than
  crashing.

### 3. `scripts/lensfinder_score_eval.py` — use the training render

- Replace `zm.render_vis_png(src, png, asinh_scale=asinh, size=png_size)` with
  `lf_stamps.render_eval_stamp(src, png, crop_m=(m // 2 if recon == "lr" else m),
  stretch=asinh, Q=args.lupton_q, size=args.png_size, scale_*=args.rgb_scale_*)`.
- Add args mirroring `build_stamps`: `--stamp-m` (default 106), `--lupton-q`
  (default 8.0), `--rgb-scale-r/g/b` (default 1.0). `m` is evened up like elsewhere.
- Keep the existing `get_galaxy_transform(minimal_view_config())` transform untouched.

### 4. Tests

- `tests/test_lensfinder_stamps.py` — `load_fits_cube` shape handling ((C,H,W), 2-D);
  `render_eval_stamp` crops to the requested size, writes a `size×size` RGB PNG, and the
  `< 4`-band fallback path renders without error.
- `tests/test_lensfinder_eval.py` (or the synthetic-runner test) — assert `HR.fits` is
  written 4-band `(4, m, m)` and that PSNR fields are still finite.
- Update any existing assertion that `HR.fits` is 2-D.

## Operational note

Existing `data/eval_results/*/HR.fits` are VIS-only; after this change, regenerate the
synthetic eval for HR to score on a true 4-band render. LR/SR scoring becomes correct
immediately on existing runs (those FITS are already 4-band). The `< 4`-band fallback
keeps a not-yet-regenerated run from crashing in the meantime.

## Risk / impact

- Backward-compatible for all VIS-only `HR.fits` consumers — they read band 0 via
  `load_vis_plane` / `[..., 0]`, which already handles 3-D.
- P(lens) scores on existing eval runs will *change* (LR/SR now 4-band-cropped) — this is
  the intended correction, not a regression.
