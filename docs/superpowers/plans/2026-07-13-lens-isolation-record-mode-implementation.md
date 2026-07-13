# Lens-Isolation Record-Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the rejected balanced/source-centred lens-isolation training regime with an additive, unbiased normal-record experiment whose target is the complete lens-system layer.

**Architecture:** The experiment temporarily wraps one existing `SkySimulator` field draw to capture each normal lens render into an aligned clean target while preserving the normal scene and observation path. It persists normal-shape `dirty_*` / `lens_*` pairs and invokes the existing `Model.train` fixed-record path; the experiment retains only output-path guards, source-fork provenance, evaluation reporting, and UI/FASRC wiring.

**Tech Stack:** Python 3.12, TensorFlow TFRecord IO, existing `SkySimulator` / `ObservationSimulator` / `Model`, Flask, React/TypeScript, pytest, Ruff.

## Global Constraints

- Production generation, observation, cropper, model, trainer, routes, records, and ensemble outputs remain unmodified and read-only.
- Experiment artifacts live only under `data/experiments/lens_isolation/{records,ensemble,evaluation}`; reject path overlap with production artifacts.
- Use pure TNG ordinary galaxies (`sersic_density_arcmin2=0`, `tng_density_arcmin2=60`, `tng_redshift_mode=true`) and an experiment-only lens density of `20 arcmin^-2`.
- Never balance labels, force lens counts, retry for crop placement, center crops, read source coordinates for training, or add a custom lens loss/trainer.
- Train via `Model.train(lr_path=dirty_train, hr_path=lens_train, forward_onthefly=False)` and retain all normal crop, augmentation, normalization, optimisation, validation, checkpointing, and logging behavior.
- Use red-green-refactor. Local verification is focused: relevant pytest modules, normal record/crop contracts, FASRC/WebUI checks, Ruff, compileall, `npm run build`, and `git diff --check`; do not run the full suite locally.

---

### Task 1: Experiment schema, configuration, and regression guardrails

**Files:**
- Modify: `euclid_polish/experiments/lens_isolation/config.py`
- Modify: `tests/test_lens_isolation_config.py`
- Modify: `tests/test_lens_isolation_generate_cli.py`

**Interfaces:**
- Produces `SCHEMA_VERSION = 2`, `DatasetConfig`, `TrainConfig`, `ExperimentPaths`, `assert_safe_output`, and serialisable scientific configuration/fingerprint helpers.
- Consumed by generation, records, train CLI, FASRC command building, status routes, and tests.

- [ ] **Step 1: Write failing tests for unbalanced normal split counts, pure-TNG defaults, lens density, and stale schema rejection.**

```python
def test_dataset_config_accepts_ordinary_counts_and_uses_pure_tng_lens_defaults():
    cfg = DatasetConfig(n_train=3, n_validate=2, n_test=1)
    assert cfg.sersic_density_arcmin2 == 0.0
    assert cfg.tng_density_arcmin2 == 60.0
    assert cfg.tng_redshift_mode is True
    assert cfg.lens_density_arcmin2 == 20.0

def test_dataset_config_fingerprint_changes_with_scientific_configuration():
    assert DatasetConfig().fingerprint() != DatasetConfig(lens_density_arcmin2=21).fingerprint()
```

- [ ] **Step 2: Run the focused tests and confirm they fail because the old balanced schema is still present.**

Run: `python -m pytest tests/test_lens_isolation_config.py tests/test_lens_isolation_generate_cli.py -q`

Expected: FAIL on the removed `positive_fraction`/even-count contract and missing configuration fields.

- [ ] **Step 3: Implement the schema-v2 normal-population configuration and retain the production-overlap guards.**

```python
@dataclass(frozen=True)
class DatasetConfig:
    n_train: int = 6400
    n_validate: int = 100
    n_test: int = 100
    image_size: int = 510
    seed: int = -1
    sersic_density_arcmin2: float = 0.0
    tng_density_arcmin2: float = 60.0
    tng_redshift_mode: bool = True
    lens_density_arcmin2: float = 20.0

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Re-run the tests and commit the schema boundary.**

Run: `python -m pytest tests/test_lens_isolation_config.py tests/test_lens_isolation_generate_cli.py -q`

Expected: PASS.

### Task 2: Capture normal lens draws and create aligned dirty/lens examples

**Files:**
- Modify: `euclid_polish/experiments/lens_isolation/generation.py`
- Modify: `tests/test_lens_isolation_generation.py`

**Interfaces:**
- Produces `LensCaptureAdapter.generate_example(rng) -> GeneratedExample` with `dirty: Image`, `lens: Image`, and a source-metadata row for one normal field.
- Consumes `SkySimulator.simulate_field`, its existing `_add_lens` method, and `ObservationSimulator.process` without changing production implementations.
- Consumed by record generation.

- [ ] **Step 1: Write failing tests for capture identity, ordinary-galaxy/star exclusion, and zero/one/multiple normal lens outcomes.**

```python
def test_capture_adds_each_single_render_delta_to_scene_and_target(fake_normal_sky):
    example = LensCaptureAdapter(fake_normal_sky, observation).generate_example(np.random.default_rng(1))
    np.testing.assert_array_equal(example.lens.data, fake_normal_sky.rendered_lens_delta)
    np.testing.assert_array_equal(example.dirty_source.data, fake_normal_sky.scene_with_galaxies_and_lenses)

@pytest.mark.parametrize("n_lenses", [0, 1, 2])
def test_normal_poisson_outcomes_are_not_relabelled_or_retried(n_lenses):
    sky = FakeNormalSky(n_lenses=n_lenses)
    example = LensCaptureAdapter(sky, FakeObservation()).generate_example(np.random.default_rng(1))
    assert example.sources["n_lenses"] == n_lenses
    assert sky.lens_calls == n_lenses
```

- [ ] **Step 2: Run the focused generation test and confirm it fails under the label/retry implementation.**

Run: `python -m pytest tests/test_lens_isolation_generation.py -q`

Expected: FAIL because `generate_example` requires `label` and calls crop-safe retry logic.

- [ ] **Step 3: Replace independent lens generation with a scoped capture adapter around one normal `simulate_field` call.**

```python
class LensCaptureAdapter:
    def generate_example(self, rng: np.random.Generator) -> GeneratedExample:
        with self._capture_lens_deltas() as target_canvas:
            scene_hr, metadata = self.sky.simulate_field(rng)
        dirty_lr, _ = self.observation.process(self._with_fixed_stars(scene_hr, metadata), rng)
        return GeneratedExample(dirty=dirty_lr, lens=self._image_like(scene_hr, target_canvas), sources=metadata)
```

The context manager must restore the original method in `finally`, accumulate a before/after canvas delta exactly once for every normal `_add_lens` call, and never modify `SkySimulator` source code.

- [ ] **Step 4: Re-run the generation tests and commit the single-draw capture implementation.**

Run: `python -m pytest tests/test_lens_isolation_generation.py -q`

Expected: PASS.

### Task 3: Normal-shape paired records, resumable sidecars, and metadata

**Files:**
- Modify: `euclid_polish/experiments/lens_isolation/records.py`
- Modify: `scripts/lens_isolation_generate.py`
- Modify: `tests/test_lens_isolation_records.py`
- Modify: `tests/test_lens_isolation_generate_cli.py`

**Interfaces:**
- Produces `dirty_{train,validate,test}.tfrecord`, `lens_{train,validate,test}.tfrecord`, `sources_{train,validate,test}.csv`, and `dataset.json`.
- Exposes `validate_split(records_dir, subset, expected_count, config_fingerprint) -> bool` and `dataset_fingerprint(records_dir) -> str`.
- Consumed by normal record-mode `Model.train`, evaluation, status, and FASRC jobs.

- [ ] **Step 1: Write failing record tests for all-split dirty/lens pairing, normal parser compatibility, source sidecars, config-fingerprint reuse, and atomic incomplete-shard rejection.**

```python
def test_every_split_writes_aligned_dirty_lens_records_and_sources(tmp_path):
    summary = generate_split(TinyGenerator(), str(tmp_path), "train", count=3, seed=7)
    assert summary.count == 3
    assert os.path.isfile(tfrecord_path(str(tmp_path), "dirty_train"))
    assert os.path.isfile(tfrecord_path(str(tmp_path), "lens_train"))
    assert os.path.isfile(os.path.join(tmp_path, "sources_train.csv"))

def test_normal_model_record_parser_reads_lens_isolation_pair(tmp_path):
    model = Model(str(tmp_path / "model"), seed=1)
    dataset = model._build_training_pipeline(dirty_train, lens_train, batch_size=1, augment=False)
    assert next(iter(dataset))[0].shape[-1] == Config.NUM_LR_CHANNELS
```

- [ ] **Step 2: Run the record tests and confirm they fail because train currently writes `scene_train` and no source CSV.**

Run: `python -m pytest tests/test_lens_isolation_records.py tests/test_lens_isolation_generate_cli.py -q`

Expected: FAIL on absent `dirty_train`, source sidecars, and schema/config validation.

- [ ] **Step 3: Persist only normal-format dirty/lens pairs for all subsets and atomically publish source sidecars plus `dataset.json`.**

```python
final_records = {
    "dirty": tfrecord_path(records_dir, f"dirty_{subset}"),
    "lens": tfrecord_path(records_dir, f"lens_{subset}"),
}
writers["dirty"].write(example.dirty.to_tfrecord(index=index))
writers["lens"].write(example.lens.to_tfrecord(index=index))
```

Use deterministic per-field seeds, ordered merging, schema/config fingerprints, Reporter stage/worker events, and a clear regenerate instruction when published artifacts are incompatible.

- [ ] **Step 4: Re-run record/CLI tests and commit the paired-record contract.**

Run: `python -m pytest tests/test_lens_isolation_records.py tests/test_lens_isolation_generate_cli.py -q`

Expected: PASS.

### Task 4: Standard record-mode training and random-cutout evaluation

**Files:**
- Modify: `euclid_polish/experiments/lens_isolation/training.py`
- Modify: `euclid_polish/experiments/lens_isolation/evaluation.py`
- Modify: `scripts/lens_isolation_train.py`
- Modify: `scripts/lens_isolation_evaluate.py`
- Modify: `tests/test_lens_isolation_training.py`
- Modify: `tests/test_lens_isolation_evaluation.py`
- Delete: `euclid_polish/experiments/lens_isolation/datasets.py`
- Delete: `euclid_polish/experiments/lens_isolation/forward.py`
- Delete: `euclid_polish/experiments/lens_isolation/loss.py`
- Delete: `euclid_polish/experiments/lens_isolation/metrics.py`

**Interfaces:**
- Retains `checkpoint_fingerprint` and `fork_member` provenance checks.
- Produces `train_member(model, dirty_train, lens_train, config, reporter) -> None`, which delegates directly to `Model.train`.
- Produces `evaluate_records` returning metrics and rows using deterministic normal block-aligned random cutouts; source metadata is reporting-only.

- [ ] **Step 1: Write failing dispatch/evaluation tests.**

```python
def test_training_calls_normal_model_train_in_fixed_record_mode(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(Model, "train", lambda self, **kwargs: calls.append(kwargs))
    train_member(model, dirty_train, lens_train, TrainConfig(sources=("member_00",)))
    assert calls[0]["lr_path"] == dirty_train
    assert calls[0]["hr_path"] == lens_train
    assert calls[0]["forward_onthefly"] is False

def test_evaluation_cut_coordinates_are_block_aligned_and_ignore_source_metadata():
    coords = sample_random_crop_coordinates(np.random.default_rng(7), field_size=510, crop_size=96, scale=2)
    assert coords[0] % 2 == 0 and coords[1] % 2 == 0
```

- [ ] **Step 2: Run the focused tests and confirm they fail under the custom trainer and source-centred evaluator.**

Run: `python -m pytest tests/test_lens_isolation_training.py tests/test_lens_isolation_evaluation.py -q`

Expected: FAIL on use of `LensIsolationTrainer`, `LensIsolationLoss`, or manifest labels/coordinates.

- [ ] **Step 3: Delegate to `Model.train` and report random-crop aggregate, target-present, zero-target, optional ROC/AUC, and zero-baseline metrics.**

```python
model.train(
    lr_path=dirty_train,
    hr_path=lens_train,
    forward_onthefly=False,
    steps=config.steps,
    batch_size=config.batch_size,
    evaluate_every=config.evaluate_every,
    step_callback=reporter_step,
    eval_callback=reporter_metric,
)
```

Remove the experiment-specific live forward, custom loss, and trainer so no code path can select the old regime.

- [ ] **Step 4: Re-run training/evaluation tests and commit normal-mode dispatch.**

Run: `python -m pytest tests/test_lens_isolation_training.py tests/test_lens_isolation_evaluation.py -q`

Expected: PASS.

### Task 5: FASRC/UI terminology, artifact status, and truthful documentation

**Files:**
- Modify: `euclid_polish/experiments/lens_isolation/fasrc_steps.py`
- Modify: `euclid_polish/web/routes/lens_isolation.py`
- Modify: `euclid_polish/web/templates/lens_isolation.html`
- Modify: `euclid_polish/web/frontend/src/pages/LensIsolation.tsx`
- Modify: `euclid_polish/experiments/lens_isolation/README.md`
- Modify: `README.md`
- Modify: `tests/test_lens_isolation_fasrc.py`
- Modify: `tests/test_lens_isolation_web.py`

**Interfaces:**
- FASRC `lens_isolation_generate`, `lens_isolation_train`, and `lens_isolation_evaluate` expose only current record-mode controls and Reporter events.
- Classic/React pages describe unbiased pure-TNG normal-field generation and standard record-mode training.

- [ ] **Step 1: Write failing FASRC/web tests that reject old balanced, centering, custom-loss, and live-forward language/arguments.**

```python
def test_train_command_uses_normal_training_controls_only():
    command = REGISTRY.get("lens_isolation_train").build_command({"sources": "member_01"})
    assert "--lens-weight" not in command
    assert "--crops-per-field" not in command

def test_pages_state_pure_tng_and_random_normal_crops(client):
    body = client.get("/lens-isolation").get_data(as_text=True)
    assert "Pure TNG" in body
    assert "random, block-aligned crops" in body
```

- [ ] **Step 2: Run the UI/FASRC tests and confirm they fail because the old controls/labels remain.**

Run: `python -m pytest tests/test_lens_isolation_fasrc.py tests/test_lens_isolation_web.py -q`

Expected: FAIL on old flags and 50/50/source-centred claims.

- [ ] **Step 3: Update the three cards, status payloads, commands, and docs to describe the implemented normal workflow only.**

```tsx
<CardHead
  title="1 · Generate paired normal fields"
  sub="Pure TNG ordinary galaxies · 20 lenses / arcmin² · CPU"
/>
<CardHead
  title="2 · Fork and train"
  sub="Normal fixed-record training on dirty LR → lens-only HR pairs · GPU"
/>
```

Remove stale `tng_fraction` / COSMOS-default claims from the top-level README, preserve the existing visual language, and keep only experiment-owned artifact paths in the status/sync UI.

- [ ] **Step 4: Run the focused web tests, build the frontend, then commit the surface/documentation pass.**

Run: `python -m pytest tests/test_lens_isolation_fasrc.py tests/test_lens_isolation_web.py -q && npm run build`

Expected: PASS and a successful production frontend build.

### Task 6: Focused verification and integration

**Files:**
- Verify: all files changed by Tasks 1–5

- [ ] **Step 1: Run focused Lens Isolation plus normal record/crop, FASRC, and web contract tests using the project Conda environment and cache/JIT overrides.**

Run: `env EUCLID_POLISH_DISABLE_AUTO_SSH=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_DISABLE_JIT=1 NUMBA_CACHE_DIR=/private/tmp/euclid_numba_cache XDG_CACHE_HOME=/private/tmp/euclid_xdg_cache MPLCONFIGDIR=/private/tmp/euclid_mpl_cache python -m pytest tests/test_lens_isolation_*.py <selected-normal-record-and-crop-tests> -q`

Expected: PASS with no Lens Isolation regression failures.

- [ ] **Step 2: Run static and build verification.**

Run: `ruff check euclid_polish/experiments/lens_isolation scripts/lens_isolation_*.py euclid_polish/web/routes/lens_isolation.py && python -m compileall euclid_polish/experiments/lens_isolation scripts/lens_isolation_*.py euclid_polish/web/routes/lens_isolation.py && npm run build && git diff --check`

Expected: all commands exit 0.

- [ ] **Step 3: Review the diff against the design acceptance criteria, commit, push, and fast-forward merge the verified branch.**

Run: `git diff --check && git status --short && git log --oneline -1`

Expected: only intended experiment, UI, test, documentation, and built frontend changes are present.
