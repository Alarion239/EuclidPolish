# Lens-Isolation Super-Resolution Ensemble Design

## Goal

Add an experimental, pluggable EuclidPolish pipeline that forks strong existing
super-resolution members and fine-tunes them to reconstruct only complete
gravitational-lens systems. A target retains both the foreground lensing galaxy
and the lensed background source (arcs, rings, or multiple images), while stars
and every unrelated/plain galaxy are removed.

The experiment must have its own generated records, checkpoints, manifests,
evaluation products, FASRC jobs, and WebUI surface. Existing records and models
are read-only inputs: this feature must not migrate, overwrite, invalidate, or
require regeneration of any production artifact.

## Scope and non-goals

This pass includes:

1. paired-layer synthetic dataset generation;
2. a separate resumable FASRC generation step;
3. experiment-specific on-the-fly training and loss;
4. one-to-one forks from explicitly selected existing ensemble members;
5. an isolated experimental ensemble and inference API;
6. classifier and reconstruction evaluation;
7. dedicated FASRC training/evaluation steps and sync support; and
8. classic and React WebUI surfaces for operating the experiment.

It does not change the WDSR architecture, production starless/starfull
semantics, existing TFRecord schemas, the production ensemble registry, current
checkpoint directories, or existing FASRC step behavior. It does not claim that
the experimental output is scientifically trustworthy until the acceptance
metrics below have been measured on held-out data.

## Additive architecture

All experiment logic lives below:

```text
euclid_polish/experiments/lens_isolation/
```

with standalone scripts:

```text
scripts/lens_isolation_generate.py
scripts/lens_isolation_train.py
scripts/lens_isolation_evaluate.py
scripts/lens_isolation_infer.py
```

The only changes outside that namespace are additive integration hooks:

- register three new FASRC step classes;
- register one new Web route;
- mount a Lens Isolation navigation entry/page and step-card fields; and
- add focused tests.

Default local/remote experiment paths are beneath:

```text
data/experiments/lens_isolation/
  records/
  ensemble/
  evaluation/
```

Path validation rejects production record/checkpoint directories and rejects an
output member path equal to or nested inside its source checkpoint. Existing
source members are opened read-only and fingerprinted before and after a fork.

## Physical target definition

Generation retains separate floating-point HR layers:

```text
B = ordinary/unrelated galaxy layer
L = complete gravitational-lens-system layer
S = B + L
```

`L` includes all light produced by the physically generated lens system:

- the foreground deflector/lensing galaxy; and
- the lensed background source.

For a negative example, `L` is exactly zero. An unrelated galaxy remains in
`B` even if it overlaps the lens in projection, so it appears in the observed
input but not the target. This separation is possible because layers are kept
during generation; the experiment never attempts to unmix existing composite
records.

Stars are generated separately and never enter `L`. Thus the supervised pair
is:

```text
input  = dirty_LR(S + stars)
target = clean_HR(L)
```

The dirty observation uses the existing `ObservationSimulator` on the complete
field: sampled empirical per-band PSFs, convolution, block rebinning, Poisson
sky/signal noise, read noise, detector artifacts, and saturation. Forward
modelling occurs before cropping so out-of-crop PSF wings remain physical.

## Dataset construction and balance

The generator composes two independent uses of the existing public sky
simulator:

1. generate a galaxy-rich background with no lenses and no deposited stars;
2. for a positive, generate a lens-only layer with no unrelated galaxies or
   stars, retrying boundedly until a valid, fully crop-safe system is rendered;
3. add the layers to form `S`; or use a zero `L` for a negative; and
4. generate the fixed dirty observation for validate/test.

Each split is exactly balanced by construction (alternating then deterministically
shuffled): 50% lens positives and 50% hard negatives. A positive training crop
is centred on the lens-flux centroid with bounded jitter. A negative crop is
centred on a bright ordinary galaxy, also with jitter. This prevents random
empty sky and class imbalance from making the all-zero output an attractive
solution.

The train split stores clean component pairs only; each visit redraws stars,
PSF, noise, artifacts, and crop jitter:

```text
scene_train.tfrecord
lens_train.tfrecord
manifest_train.csv
```

Validate and test are fixed for repeatable checkpoint selection and reporting:

```text
scene_{validate,test}.tfrecord
lens_{validate,test}.tfrecord
dirty_{validate,test}.tfrecord
manifest_{validate,test}.csv
```

Every aligned manifest row records index, split, binary label, scene seed,
forward seed, lens centre, Einstein radius, lens/deflector/source rendering
metadata, ordinary-galaxy count, star count, and schema version. A dataset-level
`dataset.json` records configuration, counts, balance, master seed, generation
commit, and file fingerprints. Paired counts, indices, shapes, and channel
counts are validated before the dataset is accepted.

## Separate FASRC generation step

`lens_isolation_generate` is a dedicated CPU/shared SLURM step analogous to the
current synthetic generator but with no shared output path or resume state. It:

- invokes `scripts/lens_isolation_generate.py`;
- uses its own `records/` directory;
- supports train/validate/test counts, seed, worker count, and image size;
- generates process-local shards and atomically concatenates aligned pairs;
- resumes only shards whose paired records and manifest are complete;
- supports an explicit override/force flag;
- emits structured Reporter progress and resource metrics; and
- fails closed when a lens-positive field cannot be rendered within the bounded
  retry budget.

Changing or resubmitting this step never inspects or deletes production
`records_v2` data.

## Fine-tuning and ensemble formation

`lens_isolation_train` is a separate GPU FASRC step. The user supplies one or
more existing source member names (and optionally a source base directory).
Each experimental member normally forks a different source member one-to-one;
reusing one source is allowed explicitly.

Forking uses the existing model loader to preserve the complete architecture and
weights, but creates a virgin experiment member directory with:

- step zero;
- a fresh optimizer and low warmup/cosine schedule;
- a distinct seed and live PSF bag;
- frequent held-out evaluation; and
- an `origin.json` identifying the lens-isolation schema, source path and
  fingerprint, dataset fingerprint, seed, commit, and loss configuration.

No source member is continued or registered in the production ensemble. The
experimental loader discovers members only beneath its own base directory.

The on-the-fly pipeline zips `scene_train` and `lens_train`, asserts record
alignment, adds fresh stars to the full scene, applies the existing full-field
observation simulator, chooses target-aware crops, then applies the same aligned
dihedral and asinh transformations used by current starless training.

## Collapse-resistant loss and checkpoint selection

Plain image-wide L1 is unsafe because lens light occupies relatively few pixels
and negatives have an all-zero target. The experiment therefore defines a
sample-balanced lens-isolation loss without changing production loss code.

For each positive sample, target brightness supplies a smooth weight map:

```text
w = 1 + lens_weight * sqrt(target / max(target))
```

The weighted absolute reconstruction error is normalized per sample. Its unit
background weight still penalizes unrelated output everywhere, while the
additional lens weight prevents the lens from being numerically overwhelmed by
empty pixels. A positive-only normalized flux-retention term penalizes erasing
the lens system. Each negative contributes its own normalized zero-target error.
Positive and negative per-sample losses are averaged equally, independent of
pixel count.

An experiment-specific `Trainer` subclass reuses the existing training loop,
optimizer, rollback, logging, and checkpoint machinery but overrides validation
aggregation. It keeps two tracks:

- root checkpoint: best held-out detection AUC; and
- `loss_best/`: best balanced lens-isolation validation loss.

Metrics always expose both, preventing an all-zero model from being selected by
background-dominated PSNR and preventing an AUC-only model from hiding poor lens
reconstruction. The training step defaults to a conservative peak learning rate
and supports early manual termination through the normal SLURM controls.

## Experimental inference and classifier score

The isolated ensemble loader reads only experiment members and returns:

- per-member lens-only HR reconstructions;
- ensemble mean;
- per-pixel disagreement; and
- scalar detection scores derived from positive ensemble-mean flux after
  raw-electron inverse stretch (whole-frame plus a configured central aperture
  for candidate-centred stamps).

`lens_isolation_infer.py` accepts an LR FITS cube and writes mean/disagreement
FITS plus JSON metadata. It never replaces normal inference output.

## Evaluation

`lens_isolation_evaluate` is a separate GPU FASRC step that runs every selected
experimental member over the fixed test records and writes under `evaluation/`:

- `predictions.csv` (labels, scores, Einstein radii, flux metrics);
- `metrics.json`;
- ROC and TPR-versus-Einstein-radius plots;
- positive reconstruction and negative residual galleries; and
- member/ensemble disagreement summaries.

Required metrics are:

- ROC AUC;
- TPR at fixed low FPR thresholds;
- TPR versus Einstein radius, including the smallest supported bin;
- positive lens-flux retention;
- positive target PSNR/MAE;
- residual output flux on hard negatives;
- ensemble disagreement on positives and negatives; and
- comparison with every source model and an all-zero baseline.

The zero baseline is expected to have zero recall and perfect suppression; it
must never be presented as a useful classifier merely because it has low global
pixel error.

## WebUI

A dedicated “Lens isolation” page is added to both classic and React navigation.
It explains that this is an experimental selective-reconstruction classifier
and mounts three independent cards:

1. **Generate lens-isolation pairs (CPU)** — dataset counts, seed, workers,
   force, and resource controls;
2. **Fork/train lens-isolation ensemble (GPU)** — source base/members, steps,
   seeds, learning rate/loss weights, and resources; and
3. **Evaluate lens-isolation ensemble (GPU)** — member selection, test limit,
   FPR targets, and resources.

A fourth local sync/status section pulls only the experiment dataset summary,
member metadata, evaluation JSON/plots, and optionally selected FITS examples.
Large training records and checkpoints require explicit opt-in. The generic
FASRC submit/history APIs remain unchanged; the cards are ordinary registered
steps with experiment-specific command builders.

## Error handling and compatibility

- Existing data/model paths are rejected as experiment outputs.
- Existing experiment output is not overwritten without `--force`.
- Source checkpoints must exist and targets must be virgin for a new fork.
- Source fingerprints are checked after forking to prove no mutation.
- Paired TFRecords and manifests are written via temporary shards and atomic
  replacement; incomplete shards are not considered resumable.
- Generation aborts on exhausted lens-render retries rather than silently
  converting intended positives to negatives.
- Training aborts on record-count, index, shape, schema, or dataset-fingerprint
  mismatch.
- Evaluation reports missing members/checkpoints as explicit errors.
- Production commands, routes, records, registries, and inference retain their
  existing defaults and behavior.

## Focused testing and verification

All behavior is developed with red-green-refactor in new focused test modules.
Tests cover:

1. lens layer includes deflector and lensed source but excludes plain galaxies;
2. negative target is exactly zero;
3. dirty input is forward-modelled from the full scene plus stars;
4. exact split balance and deterministic replay;
5. positive lens-centred and hard-negative galaxy-centred crops;
6. paired record/manifest integrity and atomic resume rules;
7. path guards protecting production artifacts;
8. sample-balanced loss penalizes all-zero positive predictions and false
   positive negative predictions;
9. source checkpoint remains unchanged after a fork;
10. separate experimental member discovery and ensemble mean/disagreement;
11. AUC, fixed-FPR, Einstein-radius, flux-retention, and zero-baseline metrics;
12. FASRC commands point only to experiment scripts/paths; and
13. classic/React WebUI registration and submission fields.

Verification runs only these focused tests plus compilation and Ruff. The full
suite is not run, honoring the user's standing instruction.

## Acceptance criteria

Implementation is complete when:

- a focused small dataset can be generated end-to-end in an isolated directory;
- a dry-run FASRC generation command is correct and the WebUI card is mounted;
- a source member can be forked without source mutation;
- the experimental training pipeline consumes paired scene/lens records with
  live forward modelling;
- the ensemble can infer mean, disagreement, and detection scores;
- the evaluator produces all required machine-readable metrics/baselines;
- production paths and registry membership remain unchanged; and
- focused tests, compilation, and Ruff pass.

Scientific promotion is explicitly separate: a trained run must subsequently
meet agreed recall, low-FPR, small-Einstein-radius, lens-flux, and negative
residual thresholds before anyone treats it as more than an experiment.
