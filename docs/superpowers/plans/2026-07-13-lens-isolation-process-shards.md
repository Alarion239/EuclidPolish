# Lens-isolation Process-shard Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Make lens-isolation generation use one independent simulator/capture adapter per allocated CPU, with deterministic paired dirty/lens/source shards merged into the existing final artifacts.

**Architecture:** Keep the physical capture adapter and final record schema. Replace the shared thread pool with the normal Sky pattern: process-local simulator initialization, deterministic index-range shards, paired writers per shard, and ordered parent-side merge. FASRC derives workers from allocated CPUs, removing the duplicate Lens-isolation worker input.

**Tech Stack:** Python 3.12, ProcessPoolExecutor, TensorFlow TFRecord I/O, NumPy RNG, FASRC React UI, pytest.

## Global Constraints

- Preserve pure-TNG normal fields and lens-only HR targets.
- Keep final dirty_{split}.tfrecord, lens_{split}.tfrecord, sources_{split}.csv, split_{split}.json, and dataset.json paths unchanged.
- Every shard writes position-aligned dirty/lens records plus source rows for one contiguous global field-index range.
- Derive shard seeds reproducibly from the split seed and shard id.
- Derive FASRC workers from n_cpus. Retain --workers only for direct CLI use.
- Verify with EuclidPolishEnv and the focused test/cache environment.
- Do not run pytest locally: the user deferred test execution to CI after local TensorFlow test processes exhausted available RAM.

---

### Task 1: Write and merge deterministic paired shards

**Files:**
- Modify: euclid_polish/experiments/lens_isolation/records.py
- Test: tests/test_lens_isolation_records.py

**Interfaces:**
- Produces: ShardSpec(subset, shard_id, start, count, seed).
- Produces: make_shards(subset, count, workers, seed) -> list[ShardSpec].
- Produces: write_shard(generator, records_dir, shard) -> ShardSummary.
- Produces: merge_shards(records_dir, subset, shards, config_fingerprint) -> SplitSummary.
- Preserves: generate_split(generator, records_dir, subset, count, seed, config_fingerprint, force, workers) as the serial compatibility entry point, implemented through one shard and the same merge path.

- [ ] **Step 1: Write the failing paired-shard order test.**

~~~python
def test_process_shards_merge_pairs_and_sources_in_global_index_order(tmp_path):
    shards = make_shards("train", count=5, workers=2, seed=17)
    for shard in reversed(shards):
        write_shard(IndexedTinyGenerator(), str(tmp_path), shard)
    summary = merge_shards(str(tmp_path), "train", shards, config_fingerprint="cfg")

    assert summary.count == 5
    assert _record_values(tfrecord_path(str(tmp_path), "dirty_train")) == [100, 101, 102, 103, 104]
    assert _record_values(tfrecord_path(str(tmp_path), "lens_train")) == [0, 1, 2, 3, 4]
    assert _source_field_indices(tmp_path / "sources_train.csv") == [0, 1, 2, 3, 4]
~~~

- [ ] **Step 2: Submit the new regression to CI before relying on it.**

~~~bash
source /Users/alarion239/miniforge3/etc/profile.d/conda.sh
conda activate EuclidPolishEnv
env EUCLID_POLISH_DISABLE_AUTO_SSH=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_DISABLE_JIT=1 NUMBA_CACHE_DIR=/private/tmp/euclid_numba_cache XDG_CACHE_HOME=/private/tmp/euclid_xdg_cache MPLCONFIGDIR=/private/tmp/euclid_mpl_cache python -m pytest tests/test_lens_isolation_records.py::test_process_shards_merge_pairs_and_sources_in_global_index_order -q
~~~

Expected CI result before implementation: FAIL because the new functions do not exist.

- [ ] **Step 3: Implement paired shard storage.**

~~~python
@dataclass(frozen=True)
class ShardSpec:
    subset: str
    shard_id: int
    start: int
    count: int
    seed: tuple[int, int]

def make_shards(subset: str, *, count: int, workers: int, seed: int) -> list[ShardSpec]:
    # Use at least one contiguous shard per worker and at most 256 fields/shard.

def write_shard(generator, records_dir: str, shard: ShardSpec) -> ShardSummary:
    # Write dirty/lens/source .partNNNN files with global field indices.

def merge_shards(records_dir: str, subset: str, shards: Sequence[ShardSpec], *, config_fingerprint: str) -> SplitSummary:
    # Concatenate TFRecord parts in shard_id order, concatenate sources with one
    # header, publish metadata only after every final artifact is complete.
~~~

Use temporary final files plus os.replace. Reuse concat_source_csvs for sidecars. Remove parts only after a successful merge; serial generation delegates to these helpers.

- [ ] **Step 4: Let CI run all record tests after implementation.**

~~~bash
source /Users/alarion239/miniforge3/etc/profile.d/conda.sh
conda activate EuclidPolishEnv
env EUCLID_POLISH_DISABLE_AUTO_SSH=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_DISABLE_JIT=1 NUMBA_CACHE_DIR=/private/tmp/euclid_numba_cache XDG_CACHE_HOME=/private/tmp/euclid_xdg_cache MPLCONFIGDIR=/private/tmp/euclid_mpl_cache python -m pytest tests/test_lens_isolation_records.py -q
~~~

Expected CI result: PASS, including current split validation and atomicity coverage.

### Task 2: Fan out lens capture through process-local workers

**Files:**
- Modify: scripts/lens_isolation_generate.py
- Modify: euclid_polish/experiments/lens_isolation/fasrc_steps.py
- Modify: euclid_polish/web/frontend/src/pages/LensIsolation.tsx
- Test: tests/test_lens_isolation_generate_cli.py
- Test: tests/test_lens_isolation_fasrc.py
- Test: tests/test_lens_isolation_web.py

**Interfaces:**
- Produces: _init_lens_worker(runtime) that builds one SkySimulator, ObservationSimulator, and LensCaptureAdapter in each child.
- Produces: _generate_lens_shard(shard: ShardSpec) that writes one process-local shard.
- Consumes: make_shards, write_shard, and merge_shards from records.py.
- Produces: the FASRC command `scripts/lens_isolation_generate.py --workers <n_cpus>`.

- [ ] **Step 1: Write failing command and UI contract tests.**

~~~python
def test_lens_isolation_fasrc_generation_uses_allocated_cpu_count():
    command = LensIsolationGenerateStep().build_command({"n_cpus": 32, "ntrain": 5})
    assert command[command.index("--workers") + 1] == "32"

def test_lens_isolation_ui_has_no_second_worker_input():
    page = Path("euclid_polish/web/frontend/src/pages/LensIsolation.tsx").read_text()
    assert 'label="workers"' not in page
    assert "setWorkers" not in page
~~~

- [ ] **Step 2: Submit the new tests to CI and confirm current behavior fails.**

~~~bash
source /Users/alarion239/miniforge3/etc/profile.d/conda.sh
conda activate EuclidPolishEnv
env EUCLID_POLISH_DISABLE_AUTO_SSH=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_DISABLE_JIT=1 NUMBA_CACHE_DIR=/private/tmp/euclid_numba_cache XDG_CACHE_HOME=/private/tmp/euclid_xdg_cache MPLCONFIGDIR=/private/tmp/euclid_mpl_cache python -m pytest tests/test_lens_isolation_fasrc.py tests/test_lens_isolation_web.py -q
~~~

Expected CI result before implementation: FAIL because command construction reads a separate workers value and the React page renders it.

- [ ] **Step 3: Implement normal-style process fan-out.**

~~~python
@dataclass(frozen=True)
class LensWorkerRuntime:
    records_dir: str
    image_size: int
    psf_dir: str
    tng_dir: str
    lens_density_arcmin2: float

_WORKER_GENERATOR = None
_WORKER_RECORDS_DIR = ""

def _init_lens_worker(runtime: LensWorkerRuntime) -> None:
    global _WORKER_GENERATOR, _WORKER_RECORDS_DIR
    sky = SkySimulator(None, SkySimulatorConfig(
        image_size=runtime.image_size,
        pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        sersic_density_arcmin2=0.0,
        tng_density_arcmin2=60.0,
        tng_redshift_mode=True,
        tng_galaxy_dir=runtime.tng_dir,
        lens_density_arcmin2=runtime.lens_density_arcmin2,
    ))
    observation = ObservationSimulator(
        psf_sets_by_band=load_all_band_psf_sets(
            psf_dir=runtime.psf_dir,
            target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        ),
        config=ObservationSimulatorConfig(),
    )
    _WORKER_GENERATOR = LensCaptureAdapter(sky, observation)
    _WORKER_RECORDS_DIR = runtime.records_dir

def _generate_lens_shard(shard: ShardSpec) -> ShardSummary:
    return write_shard(_WORKER_GENERATOR, _WORKER_RECORDS_DIR, shard)

with ProcessPoolExecutor(max_workers=workers, initializer=_init_lens_worker, initargs=(runtime,)) as pool:
    summaries = list(pool.map(_generate_lens_shard, shards))
merge_shards(out_dir, subset, summaries, config_fingerprint=config_fingerprint)
~~~

No thread pool or shared capture adapter remains. LensCaptureAdapter mutation is now local to each process. In LensIsolationGenerateStep, derive workers from params["n_cpus"] with the step default as fallback. Remove React workers state and NumberField; FASRC resource controls are the only CPU selector.

- [ ] **Step 4: Let CI run command/UI regressions; run only static checks locally.**

~~~bash
source /Users/alarion239/miniforge3/etc/profile.d/conda.sh
conda activate EuclidPolishEnv
ruff check euclid_polish/experiments/lens_isolation scripts/lens_isolation_generate.py tests/test_lens_isolation_*.py
npm run build --prefix euclid_polish/web/frontend
~~~

Expected: local static commands exit 0; CI confirms the regression tests pass and the React page has no workers field.

### Task 3: Verify and ship the corrected experiment path

**Files:**
- Modify only if verification discovers a defect in the files above.
- Test: tests/test_types_and_tfrecord.py
- Test: tests/test_lens_isolation_*.py

- [ ] **Step 1: Let CI run the scoped record and experiment contract suite.**

~~~bash
source /Users/alarion239/miniforge3/etc/profile.d/conda.sh
conda activate EuclidPolishEnv
env EUCLID_POLISH_DISABLE_AUTO_SSH=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_DISABLE_JIT=1 NUMBA_CACHE_DIR=/private/tmp/euclid_numba_cache XDG_CACHE_HOME=/private/tmp/euclid_xdg_cache MPLCONFIGDIR=/private/tmp/euclid_mpl_cache python -m pytest tests/test_types_and_tfrecord.py tests/test_lens_isolation_*.py -q
~~~

Expected CI result: PASS. Do not run pytest in this local runtime.

- [ ] **Step 2: Run final source and bundle verification.**

~~~bash
source /Users/alarion239/miniforge3/etc/profile.d/conda.sh
conda activate EuclidPolishEnv
ruff check euclid_polish/experiments/lens_isolation scripts/lens_isolation_generate.py tests/test_lens_isolation_*.py
python -m compileall euclid_polish/experiments/lens_isolation scripts/lens_isolation_generate.py
npm run build --prefix euclid_polish/web/frontend
git diff --check
~~~

Expected: all commands exit 0.

- [ ] **Step 3: Commit the correction.**

~~~bash
git add docs/superpowers/plans/2026-07-13-lens-isolation-process-shards.md euclid_polish/experiments/lens_isolation/records.py scripts/lens_isolation_generate.py euclid_polish/experiments/lens_isolation/fasrc_steps.py euclid_polish/web/frontend/src/pages/LensIsolation.tsx tests/test_lens_isolation_records.py tests/test_lens_isolation_generate_cli.py tests/test_lens_isolation_fasrc.py tests/test_lens_isolation_web.py euclid_polish/web/static/dist
git commit -m "fix: shard lens isolation generation by process"
~~~
