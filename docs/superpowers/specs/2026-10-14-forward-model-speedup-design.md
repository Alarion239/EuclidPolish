# Forward-model speed-up: GPU-bound member training at ≤16 CPUs

**Date:** 2026-10-14 · **Status:** proposed

## Goal

Member training should be limited by the GPU, not by the CPU-side forward model, with at most 16 CPUs per GPU. The design cuts the forward model's CPU cost per field without changing its physics: the elastic warp of the wings stays, and outputs are identical up to float32 rounding. On-the-fly LR generation stays exactly as it is.

## Evidence (profile job 52984745, Sapphire Rapids `test` node; real 1023² rotation pool, 218 recipe)

| Stage | ms per field | Share |
|---|---|---|
| PSF elastic warp: `scipy.ndimage.map_coordinates(order=1)` on 1023² kernels, **6.5 per field** | ~350 | 53 % |
| Warp displacement field: two `gaussian_filter`s on 1023² float64 noise | 56 | 9 % |
| Field ⊗ PSF: `fftconvolve(mode="same")`, 4 bands, 1536² FFTs | 185 | 28 % |
| Noise, rebin, target blur, stars, saturation | ~55 | 8 % |
| **Total (K-independent)** | **653** | |

Threads scale nearly as well as processes up to 16 (19.8 vs 21.0 fields/s), so at 16 CPUs the cost per field is the bottleneck, not the GIL. A process pool is out of scope. K=16 needs ≈22 fields/s to feed an A100 (XLA+bf16 step ≈11.5 ms); K=8 needs ≈44.

The GPU training nodes (`gpu`, `gpu_test`) are Intel Ice Lake, which is slower per core than the profiled Sapphire Rapids node. All final numbers are measured on Ice Lake.

## Changes

1. **Reuse the scene's realised PSF for the star plane.** `ObservationSimulator._process_one_band` calls `_psf_for_sample` a second time for the star plane with `star_psf_spec`. `process()` always sets that to the scene's `psf_spec`, so the identical 1023² warp is recomputed (2.5 extra warps per field). When the two specs are the same object, reuse the already-realised `psf`. The output is bit-identical.

2. **Fast float32 elastic warp.** `PSF.elastic_warp` with `order=1`, the only order the forward model uses, runs a numba `@njit(nogil=True)` bilinear interpolation in float32. It uses the same arithmetic and the same `mode="reflect"` boundary rule as `map_coordinates` (half-sample symmetric: `d c b a | a b c d | d c b a`). Clipping of negatives and renormalisation to unit sum are unchanged. Other orders keep the scipy path. The result matches the old float64 path to float32 rounding (the old result was already cast to float32). numba is already in the FASRC environment (through `lenstronomy`) and gets declared explicitly in `environment.yml`.

3. **Float32 displacement smoothing.** `PSF.elastic_displacement` draws the same `rng.uniform(-1, 1)` stream as today, so record generation and shard-resume replay keep their random-stream consumption. It casts the draw to float32 before the two `gaussian_filter` calls. The geometry differs only at the 1e-7 px level.

4. **Minimal circular FFT for `same` convolution (exact).** `PSF.convolved_with` today computes the full linear convolution (Nx + Nh − 1 = 1532, padded to 1536²) and keeps the centre. For a `same` output, a circular FFT of length M per axis is exact when the kernel fits (M ≥ Nh) and no kernel index it reads wraps onto a non-zero tap. The indices read are j − m ∈ [c − (Nx − 1), c + Nx − 1] with c = (Nh − 1)//2. That holds for M ≥ max(Nh, c + Nx, Nh + Nx − 1 − c). Here it is max(1023, 1021, 1021), so M = 1024 suffices: 2.25× fewer points at a power-of-two size. A helper picks M = the smallest fast length satisfying the condition, falling back to the standard length when the kernel is smaller than the image. It uses `scipy.fft.rfft2/irfft2` in float32. The results equal `fftconvolve(mode="same")` up to float32 rounding, for any image/kernel shape.

5. **FFT library (measure, then decide).** On an Ice Lake node, benchmark single-threaded float32 1024² `rfft2`/`irfft2` for scipy (pocketfft) against `mkl_fft` and pyFFTW. Adopt one through `scipy.fft.set_backend` only if it makes the whole field ≥1.3× faster on top of changes 1–4. Adopting one adds a dependency and needs a FASRC environment update, which the user decides separately with the numbers in hand.

## Expected result

About 653 → ~250–300 ms per field on Sapphire Rapids, and somewhat more on Ice Lake: warps ~350 → ~40, displacement 56 → ~25, FFT 185 → ~70. At 16 CPUs that is roughly 35–40 fields/s, so K=16 is GPU-bound with margin and K=8 is close.

## Verification

- **Unit tests** (`tests/test_forward_speedup.py`):
  - The warp kernel against `map_coordinates(order=1, mode="reflect")`, in float64, on several shapes, displacements and edges. Tolerance ≤ 1e-6 × the kernel max.
  - The minimal-FFT convolution against `scipy.signal.fftconvolve(mode="same")` for odd and even kernels, kernels larger and smaller than the image, and non-square shapes. Tolerance ≤ 1e-5 relative to the image's peak.
  - Star-plane reuse: one warp per band, and identical `process()` output with and without the reuse, on a seeded scene.
- **Full test suite** green; `NUMBA_DISABLE_JIT=1` in the test prefix still works, because the tests use small kernels.
- **Before/after profile** on an Ice Lake node with the existing profiler (`gpu_test`, 1 slice): per-field cost, stage breakdown, thread scaling to 16.
- **GPU-bound check** with `scripts/benchmark_train_step.py`: `gpu` against `live` at 16 CPUs, K=16, on the `gpu` partition. GPU-bound means `live` is within 10 % of `gpu`.

## CPU allocation

After the check, the `ensemble_train` default becomes 16 CPUs (from 24). The gpu_test booking follows automatically (ceil(16/7) = 3 slices).

## Out of scope

- Warping only the PSF core: declined, the wings keep their warp.
- Worker processes: not needed at ≤16 CPUs.
- Changing the K default: per-run choice, unchanged.
- GPU-side forward model.
