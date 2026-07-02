# Plateau guard v2: cool from the best + continue from PSNR track

**Date:** 2026-07-02
**Status:** Approved (in-conversation)

## Problem

Job 27202402 (continue member_09) showed the failure mode live: the warm
restart re-heated the LR to ~4.5e-4, the model fell from its 43.97 dB
baseline into the ~43.5 dB skip-only basin (PSNR(raw) frozen at 76.970 for
20k steps), and the plateau guard then *entrenched* it — reduce-LR-in-place
polishes the degenerate solution instead of escaping it. Two aggravators:

1. `Trainer.restore()` resumes from the max-step track across (psnr root,
   loss_best). After a degenerate stretch, loss_best is the higher-step
   track and holds the collapsed weights — the next continue resumes from
   the WRONG model.
2. The guard has one response (reduce LR in place) for two different
   plateaus: a *converged* plateau (score ≈ run best — reduce in place is
   correct) and a *degenerate* plateau (score well below run best — reducing
   in place cements the collapse).

## Decision

- **Continue resumes from the PSNR-best track.** `Trainer` gains
  `resume_track: "latest" | "psnr"`; `restore()` honors it. Threaded
  `EnsembleModel.train_members` → `Model.train` → `Trainer`; continue-mode
  specs set `"psnr"`. (Fresh/fork members have no checkpoint — unaffected.
  Crash-resume of add-mode runs keeps `"latest"` behavior.) Consistent with
  `checkpoint_step()` (root track) used for the continue target, and with
  eval, which is PSNR-best-only since 2026-07-02.
- **Two-regime plateau response ("cool from the best").** When the guard
  fires AND save-best mode is on AND the current score sits ≥
  `plateau_rollback_min_gap` (Config default 0.2, score units ≈ dB) below
  the run's best (`ckpt.psnr`) AND a best-PSNR checkpoint exists: restore
  that checkpoint (weights + optimizer + step rewind, same mechanics as the
  gradient-spike rollback: eval-window stat resets, pbar rewind, plateau
  state reset), THEN apply the LR reduction. Otherwise: reduce in place as
  today. Both paths share the `_lr_scale` bookkeeping and `min_lr` floor.
- Knob plumbed as `--plateau-rollback-min-gap` on `train_ensemble.py` and in
  `EnsembleTrainStep`'s passthrough list; no /config UI field for now.

## Testing

- `Trainer.restore(track="psnr")` picks the root track even when loss_best
  has a higher step (tiny model, real checkpoints).
- Pure regime decision (`_plateau_wants_rollback`) unit-tested: gap above /
  below threshold, no checkpoint, save-every mode.
- `train_members` passes `resume_track="psnr"` exactly for continue specs.

## Out of scope

Loss-function changes (skip-only basin attractiveness) — analysis delivered
separately; any change there needs its own approval.
