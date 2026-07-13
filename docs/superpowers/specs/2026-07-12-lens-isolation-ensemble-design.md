# Lens-Isolation Training-Pair Design

## Goal

Train an experimental EuclidPolish ensemble to reconstruct complete
gravitational-lens systems while suppressing unrelated galaxies and stars.
Each target retains both the foreground deflector and the lensed background
source. The experiment changes only the data supplied to normal training; it
does not introduce a new training regime.

The original synthetic pipeline, production records, and production models are
strictly out of scope for mutation. Lens-isolation records, fine-tuned members,
and evaluation artifacts are disposable experiment outputs beneath
`data/experiments/lens_isolation/`.

## Governing constraint

Generation and training must reuse the existing production behavior:

- the existing `SkySimulator` supplies every galaxy and lens draw;
- the existing `ObservationSimulator` supplies PSF convolution, rebinning,
  noise, detector artifacts, and saturation;
- the existing record-mode `Model.train` path supplies random crops,
  augmentation, normalization, optimization, checkpointing, and logging; and
- the existing WDSR architecture and source-member weights are unchanged.

No production generator, forward model, cropper, model, or trainer code is
modified for this experiment. Experiment code may adapt their outputs, but may
not copy or independently reimplement their scientific algorithms.

## Artifact boundary

The experiment continues to own only:

```text
data/experiments/lens_isolation/
  records/
  ensemble/
  evaluation/
```

The production record and ensemble directories are read-only inputs. Path
guards reject an experiment output that equals, contains, or is contained by a
production artifact root. Source checkpoints are fingerprinted before and
after forking so accidental mutation is detected.

The existing lens-isolation dataset schema and models describe a rejected
balanced/source-centered experiment and are not compatible with this design.
They may be deleted and regenerated in the same experiment directories. A
schema marker prevents old experiment records or checkpoints from being
silently reused. Production artifacts are never part of cleanup.

## Field population

Lens-isolation fields use the same pure-TNG population as current production
synthetic generation:

```text
sersic_density_arcmin2 = 0
tng_density_arcmin2    = 60
tng_redshift_mode      = true
lens_density_arcmin2   = 20
```

The lens-density increase from the production default of `16.5 arcmin^-2` to
`20 arcmin^-2` applies only to this experiment. All other generator parameters
come from the production configuration used by the normal generation step.
The obsolete `tng_fraction` concept is not part of the experiment interface.

Every field is sampled normally. There are no positive/negative field labels,
forced one-lens fields, forced zero-lens fields, bounded retries for crop-safe
lenses, or lens-aware acceptance rules. A Poisson draw may produce any number
of lenses, including zero.

## Exact clean/dirty pair construction

For a generated field, define:

```text
G = all ordinary TNG galaxies
L = all complete lens systems
S = G + L
```

`L` contains every pixel deposited by each normal lens render: both the
foreground deflector and the lensed source. Ordinary TNG galaxies never enter
`L`, even when projected near a lens. Stars never enter `L`.

An experiment-owned capture adapter invokes the existing `SkySimulator`
field-generation path. When that path asks the simulator to add a lens, the
adapter lets the existing lens method render the system exactly once into a
temporary floating-point layer. It adds those same rendered pixels to both the
normal scene canvas and the accumulated lens-only canvas. The adapter neither
draws new parameters nor calls the lens renderer a second time.

This produces two aligned HR arrays from one RNG stream and one physical draw:

```text
scene_HR  = G + L
target_HR = L
```

The adapter lives entirely in the lens-isolation namespace. Production
`SkySimulator` code and behavior remain unchanged. Capture state is scoped to
one field call and is cleared on both success and failure so worker reuse cannot
leak lens pixels between examples.

The existing observation path then processes the complete scene. Stars are
drawn and deposited through the same production fixed-record mechanism before
the full-field observation simulation, so out-of-crop PSF wings remain
physical:

```text
input_LR  = observe(scene_HR + stars)
target_HR = L
```

The observation realization is never applied to the target. The target remains
the clean, native-HR lens-system layer, exactly as the normal training target
remains a clean native-HR scene.

## Generated records

The separate `lens_isolation_generate` CPU/FASRC step writes position-aligned
records for train, validate, and test:

```text
dirty_{train,validate,test}.tfrecord
lens_{train,validate,test}.tfrecord
sources_{train,validate,test}.csv
dataset.json
```

`dirty_*` has the same LR shape, units, channel order, and serialization as
normal dirty records. `lens_*` has the same HR shape, units, channel order, and
serialization as normal clean records. Consequently the existing record-mode
training parser can consume the pair without a special dataset contract.

The source catalog records the simulator metadata for reproducibility and
analysis, but neither training nor crop selection reads source positions.
`dataset.json` records the schema, exact generator and observation
configuration, master seed, split counts, source commit, and record
fingerprints.

Generation reuses the normal process-shard, deterministic-seed, ordered-merge,
atomic-replacement, and Reporter-progress conventions. A split is reusable only
when both aligned records, its source catalog, count, schema, and configuration
fingerprint agree. A changed scientific configuration requires explicit
regeneration rather than mixing old and new shards.

## Training behavior

`lens_isolation_train` forks explicitly selected production members into
experiment-owned member directories, then invokes the existing normal
record-mode `Model.train` interface with:

```text
lr_path = dirty_train.tfrecord
hr_path = lens_train.tfrecord
forward_onthefly = false
```

This is the normal fixed-record training path. It supplies the same:

- uniformly random, block-aligned `96 x 96` HR / `48 x 48` LR crops;
- random dihedral augmentation;
- optional LR noise augmentation and member bootstrap behavior;
- asinh transform and per-member knee;
- model architecture and initialized weights;
- reconstruction loss selected through the normal training interface;
- optimizer, learning-rate schedule, validation, rollback, save-best behavior,
  and training logs.

Only the HR record paired with each dirty LR record differs. There is no
lens-specific cropper, source centering, crop jitter, positive/negative sampler,
custom lens loss, custom optimizer, or custom training loop.

The production on-the-fly path is not used because it accepts one clean scene
as both forward-model source and target. The separately generated dirty/lens
records provide the required different target while keeping all training logic
inside the existing record-mode implementation.

At `20 arcmin^-2`, a random `96 x 96` HR crop covers `0.0064 arcmin^2` and has
about a 12% Poisson probability of containing a lens. With batch size 16 this
is about 1.9 lens-containing crops on average. Batches with no lens are an
accepted consequence of unbiased normal cropping. Ordinary TNG galaxies in
those crops are local zero-target controls because the CNN is local.

## Validation and evaluation

Training validation uses the existing fixed-record validation path over the
aligned `dirty_validate` and `lens_validate` records. Checkpoint selection is
therefore governed by the same loss/metric behavior as normal training.

The separate evaluation step reads the held-out test pair and uses uniformly
random, block-aligned cutouts governed by the same geometry as training. It
does not center on lenses, galaxies, catalog coordinates, flux centroids, or
bright pixels, and it does not resample until a desired class appears.

After cutouts are fixed, reporting may group them by the observed target
content:

- lens-containing random cutouts report reconstruction error and retained
  target flux;
- zero-target random cutouts report residual predicted flux; and
- an optional crop-level ROC/AUC treats nonzero target flux as the label and
  predicted positive flux as the score.

This grouping is analysis only; it cannot influence sampling or training. The
report also includes ungrouped aggregate loss and an all-zero-output baseline
so sparse targets cannot make a numerically small error look scientifically
successful.

## FASRC and WebUI behavior

The existing dedicated experiment surface remains additive:

1. `lens_isolation_generate` creates the paired experiment records on CPU;
2. `lens_isolation_train` forks selected source members and runs normal
   record-mode training on GPU; and
3. `lens_isolation_evaluate` evaluates held-out random cutouts.

The classic and React Lens Isolation pages expose these three steps and report
only experiment-owned paths and status. Generation defaults show pure-TNG and
`20 arcmin^-2` lens settings truthfully. Training controls use normal training
terminology and do not expose the removed lens-specific loss or centering
knobs.

Reporter progress and resource events remain the machine-readable job-status
channel. Cluster scripts keep the direct-execution import bootstrap so they run
outside the repository root.

## Failure behavior

- Missing TNG data fails before record writing with a concrete path and setup
  message; generation never falls back to Sérsic galaxies.
- Missing required empirical PSFs follows the normal generator's configured
  fallback/fail-closed policy and is recorded in dataset metadata.
- A failed lens render follows normal field-generation behavior; it is not
  converted into a synthetic label or retried for crop placement.
- A partial shard is not accepted as complete. Completed aligned shards remain
  resumable according to the normal generation rules.
- Record counts, shapes, channels, schemas, and configuration fingerprints are
  validated before training.
- Existing incompatible experiment artifacts produce a clear reset/regenerate
  instruction. Cleanup is constrained to `data/experiments/lens_isolation/`.
- Missing or non-virgin experiment member targets fail before source weights
  are loaded. Production checkpoints remain unchanged.

## Documentation pass

The top-level README receives a factual whole-file consistency pass. It will:

- describe pure TNG as the current default synthetic population;
- remove obsolete `tng_fraction` instructions and COSMOS-default claims;
- align generation, record-mode, and on-the-fly descriptions with current
  code;
- document the lens-isolation experiment as an additive three-step workflow;
  and
- remove or correct stale command lines, filenames, paths, and UI labels found
  during the review.

The documentation pass must describe implemented behavior only. It may not
change production defaults to make old prose true.

## Focused testing and verification

Implementation follows red-green-refactor with focused tests for:

1. one normal lens draw contributes identical pixels to the complete scene and
   lens-only target;
2. ordinary TNG galaxies and stars are absent from the target;
3. zero-, one-, and multi-lens Poisson outcomes are accepted without labels or
   source-aware retries;
4. the experiment uses pure-TNG configuration and `20 arcmin^-2` lens density
   without changing production defaults;
5. dirty/lens records are aligned and accepted by the normal record parser;
6. training dispatches to normal `Model.train` record mode and does not use the
   custom centered forward, loss, or trainer;
7. random crop coordinates follow normal block-aligned crop semantics and never
   consult source metadata or target flux;
8. incompatible experiment artifacts are rejected or reset without touching a
   production path;
9. source model fingerprints are unchanged after an experiment fork;
10. FASRC commands and classic/React controls expose the corrected workflow;
11. cluster scripts remain importable when invoked outside the repository; and
12. README examples and paths match the implemented commands.

Local verification runs only focused Lens Isolation, normal crop/record
contract, FASRC, WebUI, runtime-contract, Ruff, compile, and frontend-build
checks. The full test suite is not run locally. CI may run the complete suite.

## Acceptance criteria

Implementation is complete when:

- an experiment field is drawn once and yields a complete dirty scene plus an
  exactly aligned lens-only HR target;
- generation uses only TNG ordinary galaxies and an experiment-only lens
  density of `20 arcmin^-2`;
- generated train/validate/test pairs use the normal record shapes and can be
  consumed by unchanged normal record-mode training;
- every training crop is selected by the normal random crop logic;
- experiment training differs from normal training only in the target record
  supplied;
- no production source, record, model, default, route behavior, or artifact is
  modified or deleted;
- the separate FASRC/WebUI workflow operates end to end;
- the README accurately describes the current repository; and
- all focused local checks pass, with full-suite coverage delegated to CI.
