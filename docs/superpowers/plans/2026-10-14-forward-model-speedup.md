# Forward-Model Speed-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the on-the-fly forward model's CPU cost per field roughly in half with no physics change, so member training is GPU-bound at ≤16 CPUs per GPU (spec: `docs/superpowers/specs/2026-10-14-forward-model-speedup-design.md`).

**Architecture:**
- **`euclid_polish/psf/fast_ops.py` (new)** holds two numerical kernels: an exact minimal-length circular FFT `same` convolution, and a numba bilinear warp with scipy's `reflect` boundary.
- **`PSF.convolved_with` and `PSF.elastic_warp`** call them.
- **`PSF.elastic_displacement`** smooths in float32.
- **`ObservationSimulator._process_one_band`** reuses the scene's realised PSF for the star plane.
- **Measurement:** before/after profiles and an FFT-library micro-benchmark run on an Ice Lake `gpu_test` node; a `gpu`-partition benchmark decides the 16-CPU default.

**Tech Stack:** numpy, scipy.fft, numba 0.65 (already in the FASRC environment via lenstronomy; declared explicitly), pytest, SLURM on FASRC.

**Conventions:**
- Python: `$PY` = `~/miniforge3/envs/EuclidPolishEnv/bin/python`. Tests: `KMP_DUPLICATE_LIB_OK=TRUE TF_CPP_MIN_LOG_LEVEL=3 $PY -m pytest -q -p no:cacheprovider`.
- Imports go at module scope.
- Commit only the files a task names; the working tree has unrelated user files (`paper_figures/build_figures.py`, collage files, `output/nexus_comparisons/`).
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- After green tests, commit on main and push.

---

### Task 1: Baseline profile on an Ice Lake node (before any code change)

**Files:** scratchpad `profile_forward_model.py` (exists; already copied to FASRC `logs/profiling/`), new sbatch `profile_forward_icelake.sbatch`.

- [ ] **Step 1:** Write the sbatch. Same body as `profile_forward.sbatch` with these differences:
  - `#SBATCH --partition=gpu_test`, `--gres=gpu:3`, `--cpus-per-task=16`, `--mem=64G`, `--time=1:00:00`.
  - Profile arguments: threads `1,2,4,8,16`, processes `1,8,15`.
  - An output directory tag: `ARG1` = `before` or `after`.
- [ ] **Step 2:** `scp` it to `logs/profiling/` and `sbatch` it with the `before` tag. FASRC HEAD is a2ad74e, whose code equals the profiled 265270d.

### Task 2: Minimal circular FFT `same` convolution

**Files:**
- Create: `euclid_polish/psf/fast_ops.py`
- Modify: `euclid_polish/psf/core.py` (`convolved_with`, import line)
- Test: `tests/test_forward_speedup.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
from scipy.signal import fftconvolve

from euclid_polish.psf.fast_ops import circular_same_length, same_convolve


@pytest.mark.parametrize("img_shape,ker_shape", [
    ((64, 64), (127, 127)),     # kernel wider than image (the training case, scaled)
    ((64, 48), (101, 77)),      # non-square, odd
    ((31, 40), (64, 64)),       # even kernel larger than image
    ((64, 64), (10, 10)),       # even kernel smaller than image
    ((50, 50), (7, 7)),         # small odd kernel
])
def test_same_convolve_matches_fftconvolve(img_shape, ker_shape):
    rng = np.random.default_rng(0)
    img = rng.normal(10, 3, img_shape).astype(np.float32)
    img[5, 7] += 1e4
    ker = np.abs(rng.normal(0, 1, ker_shape)).astype(np.float32)
    ker /= ker.sum()
    ref = fftconvolve(img, ker, mode="same")
    out = same_convolve(img, ker)
    assert out.shape == img.shape
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-5 * np.abs(ref).max())


def test_training_geometry_uses_1024():
    # 510² field, 1023² kernel → one 1024-point period suffices (spec §4).
    assert circular_same_length(510, 1023) == 1024
    # the standard full length is never exceeded
    assert circular_same_length(50, 7) <= 50 + 7 - 1 + 8
```

- [ ] **Step 2:** Run `$T tests/test_forward_speedup.py -k same_convolve`. Expected: FAIL (module missing).

- [ ] **Step 3: Implement `fast_ops.py` (convolution part)**

```python
"""Fast numerical kernels for the PSF forward model.

``same_convolve``: ``fftconvolve(mode="same")`` computed on the shortest
circular FFT that reproduces it exactly. A kernel wider than the image needs
only one period of ≥ its own length, not the full linear length. For the
training geometry (510² field, 1023² kernel) that is 1024² instead of 1536².
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sp_fft


def circular_same_length(n_image: int, n_kernel: int) -> int:
    """Shortest fast FFT length M for which a circular convolution of an
    ``n_image`` signal with an ``n_kernel`` kernel equals the linear one on
    the ``same`` window. The window reads kernel indices j - m in
    [c - (n_image - 1), c + n_image - 1], c = (n_kernel - 1) // 2. Exact when
    the kernel fits (M ≥ n_kernel), the window's last index fits
    (M ≥ c + n_image), and negative indices wrap onto zero padding
    (M ≥ n_kernel + n_image - 1 - c)."""
    c = (n_kernel - 1) // 2
    need = max(n_kernel, c + n_image, n_kernel + n_image - 1 - c)
    return int(sp_fft.next_fast_len(need, real=True))


def same_convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """``scipy.signal.fftconvolve(image, kernel, mode="same")`` for 2-D
    arrays, in their common floating dtype (at least float32)."""
    ny, nx = image.shape
    ky, kx = kernel.shape
    my, mx = circular_same_length(ny, ky), circular_same_length(nx, kx)
    dtype = np.result_type(image.dtype, kernel.dtype, np.float32)
    spec = (sp_fft.rfft2(np.asarray(image, dtype), s=(my, mx))
            * sp_fft.rfft2(np.asarray(kernel, dtype), s=(my, mx)))
    full = sp_fft.irfft2(spec, s=(my, mx))
    cy, cx = (ky - 1) // 2, (kx - 1) // 2
    return full[cy:cy + ny, cx:cx + nx]
```

- [ ] **Step 4: Wire into `PSF.convolved_with`** (`core.py`). Replace the `fftconvolve(...)` call:

```python
        k = self.with_unit_sum().data
        return same_convolve(
            np.asarray(image, dtype=np.float32), k,
        ).astype(np.float32)
```

  Add `from euclid_polish.psf.fast_ops import same_convolve` at module scope. Drop `fftconvolve` from the imports if nothing else uses it (`grep -n fftconvolve euclid_polish/psf/core.py`).

- [ ] **Step 5:** Run `$T tests/test_forward_speedup.py tests/test_psf_class.py tests/test_psf_set.py tests/test_observation_simulator.py tests/test_image.py`. Expected: PASS.

- [ ] **Step 6:** Commit `euclid_polish/psf/fast_ops.py euclid_polish/psf/core.py tests/test_forward_speedup.py`, message "Convolve PSFs on the shortest exact circular FFT".

### Task 3: numba float32 bilinear warp

**Files:** Modify `euclid_polish/psf/fast_ops.py`, `euclid_polish/psf/core.py` (`elastic_warp`); test `tests/test_forward_speedup.py`.

- [ ] **Step 1: Failing tests**

```python
from scipy.ndimage import map_coordinates

from euclid_polish.psf import PSF
from euclid_polish.psf.fast_ops import bilinear_warp_reflect


def _reference_warp(data, dy, dx):
    yy, xx = np.indices(data.shape, dtype=np.float64)
    return map_coordinates(np.asarray(data, np.float64), (yy + dy, xx + dx),
                           order=1, mode="reflect")


@pytest.mark.parametrize("shape,alpha", [((41, 41), 5.0), ((33, 50), 20.0), ((64, 64), 3.0)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bilinear_warp_matches_map_coordinates(shape, alpha, dtype):
    rng = np.random.default_rng(1)
    data = rng.random(shape).astype(dtype)
    dy, dx = PSF.elastic_displacement(shape, alpha, 3.0, seed=4)
    ref = _reference_warp(data, dy, dx)
    out = bilinear_warp_reflect(data, dy, dx)
    assert out.dtype == dtype
    tol = 1e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(out, ref, rtol=0, atol=tol * ref.max())


def test_bilinear_warp_reflects_far_beyond_the_edge():
    data = np.arange(12, dtype=np.float64).reshape(3, 4)
    dy = np.full((3, 4), -7.25)
    dx = np.full((3, 4), 9.5)
    np.testing.assert_allclose(bilinear_warp_reflect(data, dy, dx),
                               _reference_warp(data, dy, dx), atol=1e-12)


def test_elastic_warp_equals_scipy_path_after_normalisation():
    g = np.exp(-((np.indices((51, 51)) - 25) ** 2).sum(0) / 30.0).astype(np.float32)
    psf = PSF(data=g / g.sum(), pixel_scale=0.05)
    dy, dx = PSF.elastic_displacement((51, 51), 5.0, 3.0, seed=9)
    ref = np.clip(_reference_warp(psf.data, dy, dx), 0, None)
    ref /= ref.sum()
    out = psf.elastic_warp(5.0, 3.0, seed=9, displacement=(dy, dx)).data
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-6 * ref.max())
```

- [ ] **Step 2:** Run the new tests. Expected: FAIL (`bilinear_warp_reflect` missing).

- [ ] **Step 3: Implement** in `fast_ops.py` (add `import math` and `from numba import njit` at top):

```python
@njit(cache=False, nogil=True)
def _reflect_index(i, n):
    """scipy.ndimage 'reflect' (half-sample symmetric) index: d c b a | a b c d | d c b a."""
    if n == 1:
        return 0
    period = 2 * n
    i = i % period
    if i < 0:
        i += period
    if i >= n:
        i = period - 1 - i
    return i


@njit(cache=False, nogil=True)
def _bilinear_reflect_kernel(data, dy, dx, out):
    h, w = data.shape
    for i in range(h):
        for j in range(w):
            y = i + float(dy[i, j])
            x = j + float(dx[i, j])
            y0 = math.floor(y)
            x0 = math.floor(x)
            fy = y - y0
            fx = x - x0
            ya = _reflect_index(int(y0), h)
            yb = _reflect_index(int(y0) + 1, h)
            xa = _reflect_index(int(x0), w)
            xb = _reflect_index(int(x0) + 1, w)
            out[i, j] = ((1.0 - fy) * ((1.0 - fx) * data[ya, xa] + fx * data[ya, xb])
                         + fy * ((1.0 - fx) * data[yb, xa] + fx * data[yb, xb]))


def bilinear_warp_reflect(data: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """``map_coordinates(data, (yy + dy, xx + dx), order=1, mode="reflect")``
    in ``data``'s dtype; GIL-free."""
    data = np.ascontiguousarray(data)
    out = np.empty_like(data)
    _bilinear_reflect_kernel(data, np.ascontiguousarray(dy), np.ascontiguousarray(dx), out)
    return out
```

- [ ] **Step 4: Wire into `elastic_warp`** (`core.py`). Keep validation and the scipy path for `order != 1`. For `order == 1`:

```python
        if int(order) == 1:
            out = bilinear_warp_reflect(np.asarray(self.data), dy, dx)
            if clip_negative:
                np.maximum(out, 0.0, out=out)
            total = float(out.sum(dtype=np.float64))
            if total > 0.0:
                out /= out.dtype.type(total)
            return replace(self, data=out.astype(self.data.dtype, copy=False),
                           fwhm_arcsec=None)
```

  The existing float64 copy and `np.indices`/`map_coordinates` path follows for other orders.

- [ ] **Step 5:** Run Task 2's command. Expected: PASS, including `test_psf_set.py::test_elastic_warp_*`.

- [ ] **Step 6:** Commit (fast_ops, core, tests), message "Warp PSF kernels with a GIL-free bilinear kernel in their own dtype".

### Task 4: float32 displacement smoothing

**Files:** Modify `euclid_polish/psf/core.py` (`elastic_displacement`); test `tests/test_forward_speedup.py`.

- [ ] **Step 1: Failing test**

```python
def test_displacement_is_float32_and_keeps_the_random_stream():
    dy, dx = PSF.elastic_displacement((40, 30), 5.0, 3.0, seed=11)
    assert dy.dtype == np.float32 and dx.dtype == np.float32
    rng = np.random.default_rng(11)
    from scipy.ndimage import gaussian_filter
    ref_dy = gaussian_filter(rng.uniform(-1.0, 1.0, size=(40, 30)), 3.0, mode="reflect") * 5.0
    np.testing.assert_allclose(dy, ref_dy, rtol=0, atol=1e-5)
```

- [ ] **Step 2:** Run it. Expected: FAIL (dtype float64).
- [ ] **Step 3: Implement.** In `elastic_displacement`, draw exactly as now, then smooth in float32:

```python
        rng = np.random.default_rng(seed)
        scale = np.float32(alpha)
        dy = gaussian_filter(
            rng.uniform(-1.0, 1.0, size=validated_shape).astype(np.float32),
            sigma, mode="reflect",
        ) * scale
        dx = gaussian_filter(
            rng.uniform(-1.0, 1.0, size=validated_shape).astype(np.float32),
            sigma, mode="reflect",
        ) * scale
        return dy, dx
```

- [ ] **Step 4:** Run Task 2's command. Expected: PASS.
- [ ] **Step 5:** Commit, message "Smooth PSF warp displacements in float32".

### Task 5: Star plane reuses the scene's realised PSF

**Files:** Modify `euclid_polish/sky/observation/observation_simulator.py` (`_process_one_band`); test `tests/test_forward_speedup.py`.

- [ ] **Step 1: Failing test.** Builds a 4-band simulator with small Gaussian sets, warps on, and a star plane. It counts `PSFSet.apply_sample` calls per `process()`: expect one per band.

```python
from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.psf.psf_set import PSFSet
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator, ObservationSimulatorConfig,
)


def _gauss_psf(n, s):
    yy, xx = np.indices((n, n)) - (n - 1) / 2
    g = np.exp(-(yy ** 2 + xx ** 2) / (2 * s * s)).astype(np.float32)
    return PSF(data=g / g.sum(), pixel_scale=Config.DEFAULT_PIXEL_SCALE)


def test_star_plane_reuses_the_scene_psf(monkeypatch):
    sets = {b.name: PSFSet.from_psfs([_gauss_psf(31, 2.0), _gauss_psf(31, 3.0)])
            for b in Config.BANDS}
    calls = []
    original = PSFSet.apply_sample

    def counted(self, *a, **k):
        calls.append(1)
        return original(self, *a, **k)

    monkeypatch.setattr(PSFSet, "apply_sample", counted)
    sim = ObservationSimulator(psf_sets_by_band=sets, config=ObservationSimulatorConfig(
        add_noise=False, add_saturation=False, randomize_psf=True, psf_unrotated_prob=1.0,
        psf_warp_prob=1.0, psf_warp_alpha_max=5.0, psf_warp_sigma=3.0))
    field = np.full((96, 96, 4), 5.0, np.float32)
    stars = np.zeros_like(field)
    stars[40, 50, :] = 1e4
    hr = Image(data=field, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
               band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True)
    sim.process(hr, np.random.default_rng(3), star_hr_4ch=stars)
    assert len(calls) == len(Config.LR_INPUT_BAND_NAMES)
```

- [ ] **Step 2:** Run it. Expected: FAIL (8 calls).
- [ ] **Step 3: Implement** in `_process_one_band`:

```python
        if star_hr_channel is not None and np.any(star_hr_channel):
            # The star plane shares the scene's sample: reuse its realised
            # (warped) kernel instead of warping the same kernel again.
            star_psf = (psf if star_psf_spec is psf_spec else self._psf_for_sample(
                band, star_psf_spec, warp_displacements,
            ))
```

- [ ] **Step 4:** Run Task 2's command plus `tests/test_forward_onthefly.py`. Expected: PASS.
- [ ] **Step 5:** Commit, message "Reuse the scene's warped PSF for the star plane".

### Task 6: Declare numba; full suite; push; update FASRC

- [ ] **Step 1:** In `environment.yml`, add `  - numba            # GIL-free PSF warp kernel (euclid_polish/psf/fast_ops.py)` after `scipy`.
- [ ] **Step 2:** Full suite: `$T tests`. Expected: all pass. Lint: `$PY -m ruff check euclid_polish/psf tests/test_forward_speedup.py euclid_polish/sky/observation/observation_simulator.py`.
- [ ] **Step 3:** Commit `environment.yml`, push main, then POST `/api/fasrc/git-pull` (Flask test client with `remote.connect_from_config()`). The response's `env_update_needed` will be true because `environment.yml` changed. numba is already installed there, so no conda run is needed; check `import numba` in the FASRC env.

### Task 7: After profile on Ice Lake

- [ ] **Step 1:** `sbatch` Task 1's script with the `after` tag. Copy the updated profiler first if it changed.
- [ ] **Step 2:** Compare before/after: per-field ms, stage table, threads=16 fields/s.

### Task 8: FFT-library micro-benchmark on Ice Lake

**Files:** scratchpad `bench_fft.py`, `bench_fft.sbatch`.

- [ ] **Step 1:** `bench_fft.py` times single-thread float32 `rfft2(s=(1024,1024))` + `irfft2` of a 510² image, padded, for:
  - scipy (pocketfft);
  - `mkl_fft._scipy_fft` as a `scipy.fft.set_backend` backend;
  - `pyfftw.interfaces.scipy_fft` (planner effort `FFTW_MEASURE`, cache on).

  It reports the median of 50 calls each and checks results agree to 1e-4 relative. It also times `same_convolve` end to end per library.
- [ ] **Step 2:** The sbatch (gpu_test, 1 slice, 7 CPUs, 16G, 30 min) runs `pip install --target $TMPDIR/fftlibs mkl_fft pyfftw` and puts that on `PYTHONPATH`, leaving the shared environment untouched. Submit and read.
- [ ] **Step 3:** Decision rule (spec §5): recommend a library only if it brings a ≥1.3× faster whole field on top of Tasks 2–5. Report; no code change without the user.

### Task 9: GPU-bound check at 16 CPUs (gpu partition)

- [ ] **Step 1:** sbatch on `gpu` (1 GPU, 16 CPUs, 48G, 1 h) running:

```bash
python scripts/benchmark_train_step.py --depths 16 --batches 4 --variants xla+bf16 \
    --crops-per-field 16,8 --star-prior-file logs/pipeline/ensemble-train-20261014-110428.star-population.c18fc1ef2c26.json \
    --out logs/profiling/gpubound-$SLURM_JOB_ID.json
```

- [ ] **Step 2:** GPU-bound if `live` ≥ 0.9 × `gpu` steps/s at K=16. Record K=8 too.

### Task 10: 16-CPU default (only if Task 9 passes)

**Files:** Modify `euclid_polish/web/fasrc_pipeline.py:1593-1596`; test `tests/test_fasrc_pipeline.py` (any test pinning 24).

- [ ] **Step 1:** `grep -n "24" tests/test_fasrc_pipeline.py tests/test_*resource*.py` for tests that pin the default; update them to 16 first (failing).
- [ ] **Step 2:** Set `n_cpus=16` and rewrite the comment with the measured numbers.
- [ ] **Step 3:** Run the fasrc tests, then the full suite; commit; push; FASRC git-pull.

### Task 11: Record

- [ ] Memory note `project_forward_model_speedup.md` + MEMORY.md index line; final report to the user with before/after numbers and the FFT-library recommendation.
