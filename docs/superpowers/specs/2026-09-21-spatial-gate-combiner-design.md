# Spatial gate combiner — design

Date: 2026-09-21 · Status: implemented (see Results)

## Why the RBF combiner stalled

Measured on the cached STARFULL cubes (20 members, 100 validate → 100 test
fields, VIS asinh PSNR averaged per field):

| method | PSNR | vs best member |
|---|---|---|
| best single member (170, L2, knee 10) | 58.78 dB | — |
| ensemble mean | 57.78 dB | −1.00 |
| RBF combiner (`raw_incremental_minmeanmax_rbf`) | 58.81 dB | +0.02 |
| best global linear stack (ceiling for any per-pixel blend) | 58.88 dB | +0.10 |
| oracle per-pixel member choice (uses truth) | 63.07 dB | +4.29 |

* The RBF gate lives in an 80-D space (20 members × 4 bands) where Gaussian
  kernels are almost never active; its held-out L1 did not move across 128
  kernels, so it collapsed to a fixed global blend of two members.
* It trained on 4-band asinh L1 but is scored on VIS asinh PSNR (MSE); its
  own held-out PSNR fell during fitting.
* It saw only the member values at one pixel, so it could not recognise a
  star halo, ringing, or a blacked-out LR region — all spatial patterns.
* 93% of squared error sits on 2.7% of pixels (mid/bright/core sources), and
  ~2/3 of that error is shared by all 20 members (they read the same noisy LR;
  members match the 0.066″ blurred target exactly — error vs target blur peaks
  sharply at the training FWHM). A convex combiner can only act on the
  member-specific third, plus member-specific failures.

## Design (approved: strictly convex)

A small fully convolutional network outputs, per pixel and per band, softmax
weights over the members; the output is the weighted mean in per-band asinh
space. It is convex by construction: never outside the members' range.

* Inputs: all member SRs (asinh, 80 channels); optionally the LR input
  (asinh) plus a dilated mask of zeroed LR pixels (blackouts).
* Architecture: 1×1 conv → space-to-depth → [+LR, mask] → 1×1 → four residual
  dilated 3×3 convs (1, 2, 4, 8) at LR resolution → half-pixel bilinear 2×
  upsample → concat full-res features → 1×1 → logits. ~48k parameters,
  ~62 HR-px receptive field, ~4.5 GMAC per 510² field.
* Loss: band-weighted asinh MSE (each band normalised by the best member's
  error, so 1.0 = best member). Starts from each band's best member (0.9
  weight) and keeps the checkpoint with the lowest held-out loss.
* Data: crops read on demand from the memory-mapped validate cubes (half
  uniform, half centred where the best member's error is), dihedral
  augmentation; 85 fields train / 15 held out. Blackout augmentation: MER-style
  blackouts stamped (via `apply_saturation_masking`) on sources above 3%
  (VIS) / 10% (NISP) of the well in up to 40 training fields, members re-run
  once, cached in `cubes_validate_blackout/`.
* Inference is pure NumPy (`euclid_polish/eval/spatial_gate.py`), training
  is TensorFlow (`spatial_gate_fit.py`); a parity test pins them together.

## Integration

* Registered as combiner kind `spatial_gate` (artifact
  `spatial_gate_combiner/`, cube prefix `comb_spatial_gate`), first in
  `ACTIVE_COMBINER_KINDS` so single-combiner consumers (NEXUS, poster) prefer
  it. Every `apply_field` takes `lr=`; the RBF ignores it.
* Evaluate caches each field's LR (`lr_{rec}.npy`) next to its member cubes;
  older buckets fall back to the `dirty_*` records.
* Web: `/ensemble/combiner/fit` with `model_kind=spatial_gate` fits locally;
  `/ensemble/combiner.json?model_kind=spatial_gate` returns fit history and
  per-band member usage (all / source pixels / by brightness); a
  "Combiner · spatial gate" card on /ensemble; its own viewer tier and
  evaluation series.
* Pruning: a gate can read a subset of members (`active_members`; CLI
  `--members 170,180,…`, card "members" field, route `members=`). It stays
  keyed to the whole ensemble, reports the rest as pruned
  (`needed_member_indices`, `surviving_members`), survives archiving of a
  member it does not read, and `EnsembleModel.member_arrays(lr, indices=)`
  runs only the members it needs.
* CLI: `scripts/fit_spatial_gate.py fit|compare`.

## Results

`scripts/fit_spatial_gate.py compare` on the 100 cached STARFULL test fields
(VIS/Y/J/H asinh PSNR averaged per field) plus 40 test fields with stamped
blackouts. Bin/halo/hole columns are VIS squared error relative to the best
single member (170); halo = 3–15 px around peaks above asinh 4.

| natural test fields | VIS | Y | J | H | sky | mid | core | halo |
|---|---|---|---|---|---|---|---|---|
| ensemble mean | 57.730 | 66.850 | 63.586 | 62.969 | 34.3 | 1.19 | 0.97 | 1.55 |
| best member 170 | 58.784 | 68.201 | 65.110 | 64.185 | 1 | 1 | 1 | 1 |
| RBF combiner | 58.807 | 68.253 | 65.144 | 64.250 | 1.01 | 1.00 | 0.97 | 0.98 |
| gate v1 (pooled loss) | 58.809 | 68.487 | 65.187 | 64.349 | 2.79 | 1.12 | 0.95 | 1.24 |
| gate v2 + LR input | 59.037 | 69.000 | 65.576 | 64.697 | 1.24 | 1.02 | 0.58 | 1.24 |
| gate v2, members only | 59.017 | 68.975 | 65.568 | 64.704 | 1.09 | 1.05 | 0.60 | 1.07 |

| blackout test fields | VIS | J | holes |
|---|---|---|---|
| best member 170 | 55.611 | 58.401 | 1 |
| RBF combiner | 55.609 | 58.519 | 1.045 |
| gate v2 + LR input | 56.143 | 60.380 | 0.923 |
| gate v2, members only | 56.110 | 60.260 | 0.954 |

Lessons:

* v1 chose its reference member and band weights on fields that included
  the blackout copies (so it anchored on 171, not 170) and minimised pooled
  MSE, which a few bright-star fields dominate; it tied the RBF on VIS and
  added sky noise. v2 picks the reference on natural fields and normalises
  each field's loss by that field's best-member error (the per-field PSNR
  the ensemble is scored by): +0.25 dB VIS, +0.5–0.8 dB NISP over the best
  member.
* The LR input + blackout mask buys nothing on PSNR and ~3% in holes, but
  costs sky/halo error, so the default gate reads members only
  (`--lr-input` opts in).
* Halo error improves in 70% of fields (median ratio 0.95); the pooled
  increase comes from three fields with stars 13–15 asinh above sky, where
  most members reproduce an LR artefact the best member suppresses. Only 100
  validate fields exist locally, so such stars are rare in training; more
  validate fields are the direct fix.
* The gate is interpretable: knee-10 L2 members (170, 180) carry sky/faint/
  mid pixels, knee 30–100 (171, 181, 184) bright sources, knee 1000 (187)
  star cores; no L1 member is used. That motivates the pruned gate below.
* Speed: 0.37–0.40 s per 510² field in NumPy, vs ~12 s for the 20 members
  (3%); the RBF takes 0.6 s.

Pruned gate (members 170, 171, 180, 181, 184, 187, members only):

| | natural VIS | Y | J | H | sky | halo | core | blackout J | holes | members run |
|---|---|---|---|---|---|---|---|---|---|---|
| full gate | 59.017 | 68.975 | 65.568 | 64.704 | 1.09 | 1.07 | 0.60 | 60.26 | 0.954 | 20 |
| pruned gate | 59.019 | 68.954 | 65.499 | 64.658 | 1.06 | 1.03 | 0.68 | 59.74 | 1.004 | 6 |

Pruning keeps VIS, trims NISP by ≤0.07 dB and cuts member inference 3.3×,
but the six were chosen from natural-field usage and lose the blackout gain.

Decision: the full members-only gate is installed as the STARFULL
`spatial_gate_combiner` (web re-score: 59.015 dB VIS, +0.233 dB over member
170, +1.287 dB over the ensemble mean; RBF +0.024 dB). The pruned gate is
the option when inference cost matters more than blackout handling.

Follow-ups: more validate fields (the rare very-bright-star failures);
choose a pruned set that also covers blackout usage; make the preferred
fitted kind (not always the RBF) the viewer's main SR tier and the summary
headline.

## Knee-independent loss and mixing space (2026-09-23)

A single scoring knee favours models trained near it, so the gate is now also
judged by the knee-integrated PSNR (mean PSNR over log knee, 0.1–10⁴ e⁻; see
`euclid_polish/eval/knee_psnr.py`). Two options were added to the fitter:

* `loss_knees=ALL_KNEE_LOSS`: the loss is the relative MSE at 11 log-spaced
  knees from 0.1 to 10⁴ e⁻ (each relative to the reference member at that
  knee), averaged; the stand-in for the integrated PSNR.
* `mix_space="linear"`: the convex weighted mean is taken in electrons
  instead of band-knee asinh. Mixing in asinh space is an arithmetic mean
  below the knee and a geometric mean above it, so the knee leaks into the
  output even with a knee-free loss; the linear mean is knee-free, conserves
  flux, and stays convex. The gate still sees its inputs in asinh space
  (input compression only). Old artifacts load as `mix_space="asinh"`.

26 members, 100 STARFULL test fields, 3000 steps:

| gate | integrated VIS | Y | J | H | at 100 e⁻ VIS | Y | J | H |
|---|---|---|---|---|---|---|---|---|
| best member (191) | 54.50 | 63.01 | 60.48 | 59.91 | 58.54 | 67.79 | 64.79 | 63.92 |
| ensemble mean | 53.56 | 62.50 | 59.72 | 59.11 | 57.95 | 67.29 | 63.96 | 63.26 |
| knee-100 loss, asinh mix (installed) | 55.52 | 64.82 | 61.87 | 61.10 | 59.06 | 68.98 | 65.57 | 64.67 |
| all-knee loss, asinh mix | 55.66 | 64.91 | 62.01 | 61.31 | 59.05 | 68.98 | 65.52 | 64.70 |
| all-knee loss, linear mix | 55.65 | 64.94 | 62.06 | 61.30 | 59.05 | 68.96 | 65.56 | 64.70 |

* The all-knee loss adds +0.13–0.21 dB integrated, all of it below ~10 e⁻
  (+0.8–1.1 dB at the 0.1 e⁻ knee), at no cost at 100 e⁻.
* Linear and asinh mixing tie within ±0.05 dB integrated; linear is ahead in
  Y and at J's bright end, asinh at VIS's bright end. Linear is preferred
  for being knee-free and flux-conserving at equal accuracy.
* Held-out gains after step 2000 were ≤0.03 dB, so the default is now 2000
  steps. The fitter hands every improved checkpoint to a `checkpoint`
  callback; `scripts/fit_spatial_gate.py` saves it, so a fit can be stopped
  (SIGTERM or Ctrl-C) and keeps its best gate.

Decision (2026-09-23): the all-knee loss with linear mixing is the default of
`fit_spatial_gate` (and so of the web fit job and the script), and the
`spatial_gate_linear` fit is installed as the STARFULL `spatial_gate_combiner`
(web re-score 59.049 dB VIS, +0.267 dB over the best member, +1.100 dB over
the ensemble mean). The knee-100 asinh gate it replaced is kept at
`spatial_gate_knee100`.
