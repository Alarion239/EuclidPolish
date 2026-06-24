# 4-band color lens-finder stamps, split GPU/CPU pipeline

**Date:** 2026-06-23
**Status:** Approved (design)
**Components:** `scripts/lensfinder_sr_infer.py` (new), `scripts/lensfinder_build_stamps.py`,
`euclid_polish/lensfinder/stamps.py`, `euclid_polish/web/fasrc_pipeline.py`

## Problem / motivation

The lens-finder stamp build currently (a) runs SR inference and crop/render in one
fused GPU step, and (b) collapses every stamp to the **VIS band only**, rendered as
grayscale-replicated-to-RGB. Two consequences:

- The GPU sits idle during the substantial CPU crop/render/IO portion (the SR pass is
  sequential batch-1 inference interleaved with per-field stamp work).
- Zoobot was pretrained on real *color* composites (Galaxy Zoo DECaLS, Lupton asinh
  grz→RGB). Feeding it single-band gray throws away the deflector-vs-arc color signal
  and sits off Zoobot's training distribution. The fields are already 4-band
  (VIS+Y_E+J_E+H_E) through generation, forward-modelling, and SR output — only the
  stamp cut discards the NISP bands.

## Goals

1. **Split** the fused step into a GPU SR-inference step and a CPU crop/render step,
   so the GPU does only inference and the CPU-bound work runs on the cheaper `shared`
   partition.
2. **4-band color stamps:** keep all four bands and render a Lupton-asinh RGB composite
   matched to Zoobot's GZ-DECaLS scheme, applied identically to LR/SR/HR.
3. **Sizing:** cut LR at 53 px / SR+HR at 106 px (same sky FOV), upscale each to 424
   for the encoder (LR ×8, HR ×4).

## Non-goals

- Real-data deployment of the classifier (this is a synthetic LR-vs-SR-vs-HR eval).
- A grayscale variant or a true 4-channel encoder (a 3-channel Lupton composite is the
  chosen path; the encoder stem stays 3-channel/pretrained).
- Per-source "fill the frame" variable crops (fixed 53/106 crop chosen instead).

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Deployment scope | Synthetic-only eval. |
| GPU vs CPU | Split: SR inference on GPU, crop/render on `shared` CPU. |
| SR persistence | Full **4-band** `sr_{subset}.tfrecord`, raw e⁻ (matches `hr_`/`dirty_`). |
| Channels | **4-band Lupton-asinh RGB**, all heads (B=VIS, G=(Y_E+J_E)/2, R=H_E). |
| Stamp size | Cut LR **53** / SR+HR **106** px; upscale to **424** for the encoder. |
| Comparison invariant | Same crop box + channel scheme + recipe across LR/SR/HR. |

## Background: current flow

`scripts/lensfinder_build_stamps.py` loops fields: `reconstruct(model, lr_cube)` →
`cut_triplet` (VIS band 0 only) → `recon_planes` (LR nearest-upsampled to the HR grid)
→ `render_stamp_png` (asinh→uint8 grayscale → RGB, resized to `--png-size` 424) →
`catalog.csv`. The `LensfinderBuildStampsStep` runs on `shared`/CPU (recent change).
Records (`clean_`/`hr_`/`dirty_`) are already 4-band; the SR model outputs 4 bands.

## Design

### Step A — `lensfinder_sr_infer` (GPU)

New script `scripts/lensfinder_sr_infer.py` + `LensfinderSRInferStep`:

- Args: `--records-dir`, `--subset`, `--checkpoint`, `--num-res-blocks`, `--force`.
- Load the SR model once (`load_model_from_checkpoint`). Stream each field from
  `dirty_{subset}` via `read_multiband_skyimages`, run `reconstruct`, write the 4-band
  SR cube to `sr_{subset}.tfrecord` via `open_multiband_writer` (index preserved).
- **Resumable:** before processing a subset, if `--force` is not set and
  `sr_{subset}.tfrecord` already has one record per input field, skip it (record-count
  check, robust to a truncated mid-write file — `tf.errors.DataLossError` → not
  complete). Mirrors the generation resume pattern.
- FASRC step: `partition="gpu", n_gpus=1, n_cpus=8, memory="48G",
  time_limit="6:00:00"`, `needs_gpu=True`, `conda_env=None` (main TF env), job_name
  `lensfinder-sr-infer`.

`sr_{subset}` SR fields are raw e⁻, 4-band, on the HR grid (same shape as `hr_`).

### Step B — `lensfinder_build_stamps` (shared CPU, revised)

- **Drop** the model load + `reconstruct`. Read `sr_{subset}` alongside `dirty_`/`hr_`;
  `common = LR ∩ SR ∩ HR ∩ sources` field indices.
- Step loses `--checkpoint`/`--num-res-blocks` (now on the infer step); `--stamp-m`
  default becomes **106**; gains Lupton-scale args (see render). Stays `shared`/CPU.

### Sizing — `cut_triplet` + render

- `stamp_m = 106` (even). LR crop = `m // 2 = 53`. Same sky FOV (LR is half-res).
- `cut_triplet` keeps **all 4 bands**:
  `{lr:(53,53,4), sr:(106,106,4), hr:(106,106,4)}` (VIS=band 0, then Y_E,J_E,H_E).
- The old `lr_upsample_to_grid` / `recon_planes` common-canvas step is removed; each
  recon's stamp is composited then resized **straight to 424** in the renderer
  (LR ×8, HR ×4). Edge filter uses `m = 106` (LR must clear `53/2` on its own grid).

### Channels — `render_stamp_rgb` (Lupton asinh)

New function replacing `render_stamp_png` on this path:

```python
def render_stamp_rgb(stamp4, out_png, *, scales, stretch, Q, size=424,
                     desaturate=False):
    """4-band (H,W,4) e⁻ stamp → Lupton-asinh RGB PNG, resized to `size`.

    Band→channel: B=VIS(0), G=mean(Y_E(1), J_E(2)), R=H_E(3). Per-band `scales`
    are the GZ-DECaLS analog of (125, 71.43, 52.63), tuned to electron units.
    `astropy.visualization.make_lupton_rgb(R, G, B, stretch=…, Q=…)` does the
    asinh composite. Identical recipe for LR/SR/HR. Optional low-flux
    desaturation mirrors the GZ speckle fix.
    """
```

- Defaults: `scales=(R,G,B)` tuned constants, `stretch`/`Q` defaults; exposed as
  build-stamps args (`--rgb-scale-r/g/b`, `--lupton-stretch`, `--lupton-q`) so they can
  be calibrated without code edits. Resize to 424 via PIL bilinear.
- `catalog.csv` schema is unchanged (one row per source×recon; `file_loc` → PNG).

### Invariants

Same crop box, same band→RGB mapping, same Lupton parameters, same 424 output for
LR/SR/HR — the only difference between heads is the reconstruction. Encoder input is a
uniform 424×424 RGB for all three heads (no per-head tensor-size confound).

## Wiring

- `fasrc_pipeline.REGISTRY`: add `LensfinderSRInferStep`; revise
  `LensfinderBuildStampsStep.build_command` (drop checkpoint/blocks, add stamp/Lupton
  args; `--stamp-m` default 106).
- `tests/test_fasrc_pipeline.py`: the `expected` step-id→job_name map and GPU-step set
  are already stale (missing the lensfinder steps, 3 failing tests on HEAD). Update them
  to include `lensfinder_generate`, `lensfinder_sr_infer`, `lensfinder_build_stamps`,
  `lensfinder_train` with correct job_names/GPU flags.

## Testing

- **`cut_triplet`** returns 4-band stamps shaped (53,53,4)/(106,106,4)/(106,106,4) and
  all three share the same `(cx, cy)` crop center.
- **`render_stamp_rgb`** on a synthetic 4-band array writes a 3-channel PNG of size
  424×424; a uniform-flux stamp renders without NaNs; band ordering is B=VIS/R=H.
- **SR-infer**: with a tiny model (or stubbed `reconstruct`) over `TinyCosmos`-style
  fields, writes `sr_{subset}` as a readable 4-band record with one example per field;
  re-running resume-skips the complete subset; `--force` regenerates; a truncated
  `sr_` reads as incomplete.
- **`LensfinderSRInferStep`/`LensfinderBuildStampsStep`**: registered with correct
  partition/GPU/env; `build_command` emits expected flags; build-stamps no longer emits
  `--checkpoint`.
- **`test_fasrc_pipeline`** registry/job-name/GPU tests pass with the updated map.

## Risks / notes

- **Disk:** `sr_{subset}` adds ~53 GB for 800 4-band 2040² fields (the cost of
  decoupling). Acceptable; lives in the dedicated lensfinder records dir.
- **Lupton scale calibration:** the per-band scales need one tuning pass on real stamps
  so typical galaxies show good color range (as GZ did "by eye"); defaults are a
  starting point, exposed as args.
- **Supersession:** regenerates existing `catalog.csv`/PNGs (now color, 53/106→424).
- **ConvNeXt scale:** resolved — uniform 424 input, so no 64-px collapse.
