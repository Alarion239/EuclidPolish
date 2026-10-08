# Learned output knee for single-image multi-knee members — design

2026-10-08. Agreed in conversation (option C: keep the asinh output, learn its knee).

## Problem

A single-image multi-knee member ("option 2", members 195/196) reads the LR
stretched at six knees and outputs ONE 4-band image `y` in `asinh(x / k)` space;
products are `x = k·sinh(y)`. The output knee `k` (10 e⁻ for every band today)
is a hand-picked number with no principled value: the bands' noise and
brightness scales differ by ~10× (pixel noise ≈ 20 e⁻ VIS, ≈ 2 e⁻ NISP), and
`k` decides where the network's output switches from linear to logarithmic.

## Goal

Learn `k`, one value per band, during training, so no output knee has to be
chosen. Everything else about option 2 stays as it is, so a member trained
this way differs from member 196 only in the learned knee.

Not in scope: a linear input channel (rejected), learnable input knees (they
are a fixed basis 0.1–10⁴ e⁻ the entry conv already weights), learnable loss
knees (they define the objective), a linear-output head.

## Design

### Output head

`wdsr(…, learned_output_knee=k0)` appends one layer after the main + skip sum:

```
x_b = k_b · sinh(clip(y_b, −20, 20))        b = VIS, Y_E, J_E, H_E
ln k_b = ln k_min + (ln k_max − ln k_min) · sigmoid(s · φ_b)
```

- `φ` is a trainable `(4,)` weight named `output_knee_logit`, initialised so
  every `k_b = k0` (the member's `output_knee`, 10 e⁻).
- Bounds `k_min = 0.1`, `k_max = 10⁴` e⁻ (the knee-integrated PSNR range). The
  sigmoid keeps `k` inside them smoothly (no dead gradient at a hard clip); a
  small `k` is member 204's failure mode, where output errors are amplified
  exponentially.
- `s = 0.1` slows the knee relative to the network weights: under Adam, `ln k`
  moves at most ≈ 1.4×10⁻⁴ per step at the peak LR, so it cannot thrash early
  but can still cross the whole range over 100k steps if the data push it.
- The model's output is **electrons**. The clip at ±20 is the same guard every
  member's inference applies today.

### Loss, validation, save-best (unchanged objective)

- Loss: member 196's — L2 (RMSE) per band × knee at the six training knees,
  combined by geometric mean (`knee_loss = balanced`). The electron output is
  stretched at the six knees (`asinh_stretch_multi_knee`) and compared with
  the six-knee stretched target the pipeline already builds. With `k` fixed at
  10 e⁻ this is exactly 196's `knee_expanded_loss`.
- Validation (`_evaluate_multi_knee`): the electron output is stretched at the
  six knees, then scored as for option 2 (per image, per band × knee, against
  each knee's stretched peak); save-best on the mean, as today.
- The learned `k_b` are printed with the per-knee PSNRs at every validation
  window. The CSV schema does not change (a new column would rotate every
  member's log on resume).

### Persistence and loading

- The knee lives in the checkpoint (it is a model weight).
- `origin.json` records `asinh_knees`, `knee_loss` and
  `learned_output_knee: {"init_e": 10, "min_e": 0.1, "max_e": 10000}`, and
  **omits `output_knee`**: code that does not know this member type would
  otherwise apply `10·sinh` to an output that is already in electrons and be
  silently wrong. Without `output_knee` such code fails loudly instead.
- `load_model_from_checkpoint` detects `output_knee_logit` in the checkpoint
  and builds the layer (the checkpoint is the source of truth, as for depth
  and channel counts). `infer_checkpoint_num_res_blocks` excludes that layer
  from its count; channel-count introspection is unaffected (the per-band
  skip kernels are still `(5, 5, 6, 4)`).
- Continue and fork inherit the learned head from the checkpoint and
  `origin.json`, like the knee list.

### Inference

`reconstruct(…, knees=…, output_electrons=True)` stretches the input at every
knee, runs the model and returns its output unchanged (no `sinh`).
`Model._knee_kw` passes it for this member type. The ensemble, the gate and
every evaluator already work in electrons.

### Knobs

- `MemberTrainSpec.learn_output_knee: bool` (needs `asinh_knees` and
  `output_knee`, which becomes the initial knee).
- `train_ensemble.py --learn-output-knee` and member-spec key
  `learn_output_knee`; refused without a multi-knee single-image recipe.
- `EnsembleTrainStep` TaskParam `learn_output_knee`, emitted by
  `build_command`, so Models › Train can submit it.

## The run

One member, `--mode add --count 1`, depth 32, **100k steps** (one cosine
schedule), recipe = job 48107719's option-2 spec plus
`"learn_output_knee": true`: L2, bootstrap 0.7, knees 0.1/1/10/100/1000/10⁴,
balanced, output knee 10 (initial), ICNR, on-the-fly forward, batch 4,
256² × 8 crops, PSF bag 64, warp p = 1 / α ≤ 5 / σ = 3, saturation 0.5, target
FWHM 0.066″, LR 5×10⁻⁴ → 2×10⁻⁵ with 2k warmup. Resources as that job: gpu
partition, 16 CPU, 32 GB, 1 GPU, 3 h (195/196 ran 70k steps in 88–105 min, so
100k ≈ 2.1–2.5 h).

FASRC's checkout (d87ead9) is pulled to the new `main` first. The job is
submitted through the `ensemble_train` step code so the job DB, ledger and
campaign record it, and only when no other FASRC job is queued (the two
running consoles date from 2026-09-27 and would rebuild a queued command
without the new flag).

Comparison afterwards: knee-integrated PSNR on the 100 test fields against
member 196 (fixed knee 10, 70k + 30k steps) and the production gate, plus the
learned knees per band.

## Testing

- The head: initial `k_b = k0`; output `= k·sinh(clip(y))`; `k` stays inside
  the bounds for extreme `φ`; gradients reach `φ`.
- With `k` at 10 e⁻ the new loss and validation metrics equal option 2's on
  the same `y`.
- Checkpoint round trip restores a non-default `k`; depth and channel
  introspection stay correct with the extra layer.
- `reconstruct` returns the model output unchanged for this member type.
- `origin.json` has `learned_output_knee` and no `output_knee`; continue/fork
  keep the head.
- CLI and member-spec validation; `build_command` emits the flag.
