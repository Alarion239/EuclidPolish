# Lens-Isolation Super-Resolution Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an additive experiment that generates full-scene/lens-only pairs, forks existing SR members, trains a selective-reconstruction ensemble, evaluates it as both a reconstructor and classifier, and operates through dedicated WebUI/FASRC steps.

**Architecture:** New modules under `euclid_polish/experiments/lens_isolation/` own paths, layered generation, records, live forward modelling, loss, training, ensemble inference, evaluation, and FASRC command builders. Existing production modules are reused but not behaviorally changed; only small additive registration/navigation hooks expose the experiment.

**Tech Stack:** Python 3.12, NumPy, TensorFlow/Keras, Astropy/FITS, Flask, React/TypeScript, pytest, Ruff, SLURM/FASRC.

## Global Constraints

- Existing production TFRecords and checkpoints are read-only and must never be regenerated, migrated, overwritten, or registered as experiment outputs.
- Experiment artifacts default below `data/experiments/lens_isolation/{records,ensemble,evaluation}`.
- A positive target contains the foreground lensing galaxy and lensed background source; plain galaxies and stars are absent.
- Each split is exactly 50% lens-positive and 50% galaxy-rich hard-negative.
- Train uses live full-field PSF/noise/stars; validate/test use fixed dirty observations.
- Source members are forked into virgin experiment directories with fresh optimizers and step zero.
- Existing FASRC steps, routes, and UI behavior retain their defaults.
- Run only focused tests, compilation, Ruff, and the frontend build; never run the full pytest suite.

**Spec:** `docs/superpowers/specs/2026-07-12-lens-isolation-ensemble-design.md`

---

### Task 1: Experiment paths, schemas, and production guards

**Files:**
- Create: `euclid_polish/experiments/__init__.py`
- Create: `euclid_polish/experiments/lens_isolation/__init__.py`
- Create: `euclid_polish/experiments/lens_isolation/config.py`
- Test: `tests/test_lens_isolation_config.py`

**Interfaces:**
- Produces: `ExperimentPaths`, `DatasetConfig`, `TrainConfig`, `assert_safe_output(path, *, source=None)`.

- [ ] Write failing tests proving defaults stay under `data/experiments/lens_isolation`, production `Config.RECORDS_DIR_V2` and `default_ensemble_dir()` are rejected, source/target nesting is rejected, and a temporary experiment root is accepted.
- [ ] Run `pytest --noconftest tests/test_lens_isolation_config.py -q`; expect import failure.
- [ ] Implement frozen dataclasses with validation (`positive_fraction == 0.5`, even split counts, positive sizes/steps) and realpath/commonpath guards.
- [ ] Rerun the focused file; expect all tests green.

### Task 2: Physically layered example generation

**Files:**
- Create: `euclid_polish/experiments/lens_isolation/generation.py`
- Test: `tests/test_lens_isolation_generation.py`

**Interfaces:**
- Consumes: injected background/lens/star simulators and observation simulator with existing `simulate_field`/`process` contracts.
- Produces: `GeneratedExample(scene: Image, lens: Image, dirty: Image | None, row: dict)` and `LensIsolationGenerator.generate_example(rng, *, label, fixed_dirty)`.

- [ ] Write fakes whose layers have distinguishable pixel values and failing tests showing positive `scene = background + lens`, positive target equals the complete lens layer, negative target is exactly zero, stars affect only dirty input, unrelated galaxies never enter the target, and exhausted lens retries raise `LensRenderError`.
- [ ] Run the focused test and confirm missing implementation failures.
- [ ] Implement public-simulator composition, crop-safe lens metadata checks, bounded retries, immutable layer addition, and manifest-row creation.
- [ ] Rerun until green.

### Task 3: Atomic paired records, exact balance, and deterministic replay

**Files:**
- Create: `euclid_polish/experiments/lens_isolation/records.py`
- Create: `scripts/lens_isolation_generate.py`
- Test: `tests/test_lens_isolation_records.py`
- Test: `tests/test_lens_isolation_generate_cli.py`

**Interfaces:**
- Produces: `generate_split(...)`, `validate_split(...)`, `concat_shards(...)`, `dataset_fingerprint(...)`, and CLI `main(argv=None)`.

- [ ] Add failing tests with tiny injected examples proving exact alternating/shuffled balance, aligned indices/counts, train omission of dirty records, fixed validate/test dirty records, atomic temporary-file replacement, deterministic label/seed replay, incomplete shard rejection, and `--force` overwrite behavior.
- [ ] Add CLI parse/dry-run tests proving the default output is isolated and worker/shard commands never mention production records.
- [ ] Run both focused files; confirm failures.
- [ ] Implement TFRecord/CSV shard writers, validation, SHA-256 dataset manifest, atomic concatenation, bounded process-worker orchestration, structured Reporter progress, and CLI simulator construction using current sky/PSF configuration.
- [ ] Rerun both focused files until green.

### Task 4: Target-aware live forward crops and collapse-resistant loss

**Files:**
- Create: `euclid_polish/experiments/lens_isolation/forward.py`
- Create: `euclid_polish/experiments/lens_isolation/loss.py`
- Create: `euclid_polish/experiments/lens_isolation/datasets.py`
- Test: `tests/test_lens_isolation_forward.py`
- Test: `tests/test_lens_isolation_loss.py`

**Interfaces:**
- Produces: `LensIsolationForward.crops(scene, lens)`, `LensIsolationLoss`, `build_live_dataset(...)`, and `build_fixed_dataset(...)`.

- [ ] Add coordinated fake-forward tests proving the observation model receives full scene plus freshly injected stars before cropping, positive crops include the lens-flux centroid with bounded jitter, negatives centre on the brightest ordinary galaxy, offsets are scale-aligned, and outputs have `(K,c/2,c/2,4)` / `(K,c,c,4)` shapes.
- [ ] Add loss tests proving an all-zero prediction on a positive has nonzero loss/gradient, a nonzero prediction on a negative is penalized, perfect predictions are zero, per-sample weighting makes one positive and one negative contribute equally, and positive flux loss improves as retained flux approaches target.
- [ ] Run both files and confirm failures.
- [ ] Implement thread-safe child RNGs, star injection, full-field observation, centre selection/clamping, paired TF dataset parsing, aligned dihedral/asinh transforms, and serializable TensorFlow loss.
- [ ] Rerun both files until green.

### Task 5: Safe source forks and experiment training

**Files:**
- Create: `euclid_polish/experiments/lens_isolation/training.py`
- Create: `scripts/lens_isolation_train.py`
- Test: `tests/test_lens_isolation_training.py`
- Test: `tests/test_lens_isolation_train_cli.py`

**Interfaces:**
- Produces: `checkpoint_fingerprint(path)`, `LensIsolationTrainer`, `fork_member(...)`, `train_ensemble(...)`, and training CLI `main`.

- [ ] Write failing lightweight tests using injected model/trainer factories proving each target starts virgin at step zero, distinct sources map one-to-one, origin metadata captures source/dataset fingerprints and seeds, duplicate sources require explicit allowance, and source fingerprints remain byte-identical.
- [ ] Add subclass-validation tests proving best-score uses AUC while balanced validation loss feeds `loss_best/`.
- [ ] Add CLI dry-run tests for source member parsing, isolated paths, conservative LR defaults, and invalid/missing sources.
- [ ] Run the focused files and confirm failures.
- [ ] Implement direct use of existing `Model(init_weights_from=...)`, `WarmupCosineDecay`, PSF bags, `LensIsolationTrainer(Trainer)`, live/fixed datasets, dual checkpoint tracks, origin manifests, Reporter callbacks, and source postcondition checks.
- [ ] Rerun until green.

### Task 6: Isolated ensemble inference and FITS command

**Files:**
- Create: `euclid_polish/experiments/lens_isolation/ensemble.py`
- Create: `scripts/lens_isolation_infer.py`
- Test: `tests/test_lens_isolation_ensemble.py`
- Test: `tests/test_lens_isolation_infer_cli.py`

**Interfaces:**
- Produces: `LensIsolationEnsemble.members`, `.predict(lr)`, `detection_scores(mean, aperture=None)`, and FITS CLI.

- [ ] Add fake-model tests proving discovery is limited to the experiment base, production registry/tombstones are ignored, mean/std are correct, positive flux is clipped before scoring, whole-frame and central-aperture scores differ correctly, and empty ensembles fail loudly.
- [ ] Add FITS CLI tests proving mean/disagreement FITS and JSON metadata are written without touching standard inference outputs.
- [ ] Run focused tests and confirm failures.
- [ ] Implement explicit `member_*` discovery, existing `Model` loading, stacked raw-electron inference, mean/std/scoring, and atomic FITS/JSON outputs.
- [ ] Rerun until green.

### Task 7: Reconstruction/classifier evaluation and baselines

**Files:**
- Create: `euclid_polish/experiments/lens_isolation/metrics.py`
- Create: `euclid_polish/experiments/lens_isolation/evaluation.py`
- Create: `scripts/lens_isolation_evaluate.py`
- Test: `tests/test_lens_isolation_metrics.py`
- Test: `tests/test_lens_isolation_evaluation.py`

**Interfaces:**
- Produces: `evaluate_predictions(...)`, `evaluate_records(...)`, `write_report(...)`, and evaluation CLI.

- [ ] Add pure-array tests for ROC AUC, TPR at requested FPRs, theta-E bins, positive flux retention, positive PSNR/MAE, hard-negative residual flux, disagreement summaries, and all-zero baseline zero recall.
- [ ] Add fake-ensemble integration tests proving member, ensemble, source-model, and zero baselines appear in `predictions.csv`/`metrics.json`, plus expected plot/gallery files.
- [ ] Run focused files and confirm failures.
- [ ] Keep the small ROC/AUC primitives local, stream aligned fixed test records/manifests, compute raw metrics, and render reports with Matplotlib/Astropy only.
- [ ] Rerun until green.

### Task 8: Dedicated FASRC step registration

**Files:**
- Create: `euclid_polish/experiments/lens_isolation/fasrc_steps.py`
- Modify: `euclid_polish/web/fasrc_pipeline.py` (additive import/registry entries only)
- Test: `tests/test_lens_isolation_fasrc.py`

**Interfaces:**
- Produces registered IDs `lens_isolation_generate`, `lens_isolation_train`, and `lens_isolation_evaluate`.

- [ ] Write failing registry/command tests proving generation uses CPU/shared resources, training/evaluation use GPU resources, commands invoke only experiment scripts, records/checkpoints/evaluation paths are isolated, generation exposes workers/force/counts, and no existing step command changes.
- [ ] Run focused test and confirm missing IDs.
- [ ] Implement a factory that receives existing base/resource classes (avoiding circular imports), add its returned classes to `STEP_CLASSES`, and preserve all existing IDs/commands.
- [ ] Rerun focused tests until green.

### Task 9: Classic WebUI page, status, and safe sync

**Files:**
- Create: `euclid_polish/web/routes/lens_isolation.py`
- Create: `euclid_polish/web/templates/lens_isolation.html`
- Modify: `euclid_polish/web/app.py` (route registration only)
- Modify: `euclid_polish/web/templates/base.html` (one navigation item)
- Modify: `euclid_polish/web/static/fasrc_step_card.js` (three field groups/artifact mapping)
- Test: `tests/test_lens_isolation_web.py`

**Interfaces:**
- Produces: `/lens-isolation`, `/api/lens-isolation/status`, and `/api/lens-isolation/sync`.

- [ ] Add Flask tests proving the page mounts all three cards, status reads only experiment artifacts, default sync excludes records/checkpoints, explicit flags include them, remote roots map correctly, and existing route endpoints still resolve.
- [ ] Run the focused test and confirm failures.
- [ ] Implement route/template, local summary readers, separate `rsync_pull` calls with opt-ins, navigation entry, and task fields for counts/source members/loss/eval settings.
- [ ] Rerun until green.

### Task 10: React WebUI page and built assets

**Files:**
- Create: `euclid_polish/web/frontend/src/pages/LensIsolation.tsx`
- Modify: `euclid_polish/web/frontend/src/App.tsx` or current route registry file
- Modify: frontend navigation definitions
- Regenerate: `euclid_polish/web/static/dist/`
- Test: `tests/test_lens_isolation_web.py`

**Interfaces:**
- Produces: `/app/lens-isolation` with the same three registered step cards and status/sync controls.

- [ ] Read the frontend-design skill before UI edits and follow existing card primitives rather than creating a parallel visual system.
- [ ] Extend the Web test with source assertions for route/navigation/card IDs; confirm failure.
- [ ] Implement the page with existing `Page`, `Card`, `StepById`, status, and sync components.
- [ ] Run the frontend type/build command and regenerate tracked dist assets.
- [ ] Rerun the focused Web test until green.

### Task 11: Focused end-to-end verification and compatibility audit

**Files:**
- Review every experiment file and additive integration hook.

**Interfaces:**
- Proves all acceptance criteria without executing a production-scale run.

- [ ] Generate a tiny injected/fake dataset end-to-end in a temporary directory and validate its aligned pair/manifest/dataset fingerprints.
- [ ] Run every `tests/test_lens_isolation_*.py` file individually with `--noconftest` where safe; never invoke the full suite.
- [ ] Run selected existing compatibility tests for FASRC registry, Web route integrity, model fork initialization, forward-on-the-fly behavior, and lens metrics.
- [ ] Run `python -m compileall -q euclid_polish scripts tests`.
- [ ] Run `ruff check .`.
- [ ] Run the frontend build and verify the generated shell references the new page bundle.
- [ ] Audit production paths before/after, existing `STEP_CLASSES` commands, source checkpoint fingerprints, and git diff for any non-additive behavior change.
- [ ] Update experiment README/docstrings with exact generate → train → evaluate → infer commands.
- [ ] Commit only after all focused checks pass; do not push without explicit instruction.
