"""Micro-benchmarks for the hot paths of the pipeline.

These are NOT correctness tests — they measure wall time on small,
representative inputs so we can detect performance regressions later.
Each benchmark:

  1. warms up once (JIT, numpy/scipy import caches, FFT plans),
  2. measures ``N_REPEATS`` calls,
  3. prints best / mean ms-per-call to stdout, and
  4. asserts a *generous* upper bound so the test still fails if something
     regresses by an order of magnitude.

Run just the benchmarks with verbose output:

    pytest tests/test_benchmarks.py -v -s

The thresholds are set for a modern laptop (Apple Silicon-class). If you
run on a substantially slower machine, expect some assertions to fail —
the asserts are a regression gate, not a portability gate.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

from euclid_polish.config import Config

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

N_WARMUP = 1
N_REPEATS = 5


@contextmanager
def _timer() -> Iterator[dict]:
    """Yield a dict that gets ``ms`` populated on exit."""
    result: dict = {}
    t0 = time.perf_counter()
    yield result
    result["ms"] = (time.perf_counter() - t0) * 1000.0


def _bench(name: str, callable_, n_warmup: int = N_WARMUP, n_repeats: int = N_REPEATS):
    """Time ``callable_`` and print summary; return min/mean ms."""
    for _ in range(n_warmup):
        callable_()
    durations = []
    for _ in range(n_repeats):
        with _timer() as t:
            callable_()
        durations.append(t["ms"])
    best = min(durations)
    mean = sum(durations) / len(durations)
    print(f"\n  [bench] {name:55s}  best={best:7.2f} ms  mean={mean:7.2f} ms  "
          f"(over {n_repeats} runs)")
    return best, mean


# ---------------------------------------------------------------------------
# Sersic rasterisation
# ---------------------------------------------------------------------------

def test_bench_sersic_disk_small():
    """n=1 (disk) Sersic, r_e=0.3", 128² stamp — typical disk galaxy.

    Baseline ~0.4 ms (auto-csub picks csub=1 because n≤1.2).
    """
    from euclid_polish.sky.generation.profiles import draw_sersic
    def call():
        draw_sersic((128, 128), flux=1e5, n=1.0, r_e=0.30, q=0.6,
                    theta_rad=0.5, x0=64, y0=64, pixel_scale=0.05)
    best, _ = _bench("draw_sersic n=1 128² csub_auto", call)
    assert best < 10.0, f"draw_sersic n=1 too slow: {best:.1f} ms"


def test_bench_sersic_bulge_small():
    """n=4 (bulge) Sersic, r_e=0.10", 128² — typical bulge.

    Baseline ~130 ms. csub=31 (r_e_pix=2, n=4, compact-high-n branch) and
    a stamp of side ≈ 18·r_e_pix ≈ 72 → 72×31 = 2232 sub-pixel grid.
    This is the most expensive single-source case in normal use.
    """
    from euclid_polish.sky.generation.profiles import draw_sersic
    def call():
        draw_sersic((128, 128), flux=1e5, n=4.0, r_e=0.10, q=0.8,
                    theta_rad=0.3, x0=64, y0=64, pixel_scale=0.05)
    best, _ = _bench("draw_sersic n=4 r_e=0.1\" 128² csub_auto", call)
    assert best < 400.0, f"draw_sersic n=4 too slow: {best:.1f} ms"


def test_bench_bulge_disk_typical_galaxy():
    """B+D rendering at 252² — the default HR field size.

    Baseline ~285 ms (bulge ≈ 280 ms, disk ≈ 5 ms; the bulge dominates).
    """
    from euclid_polish.sky.generation.profiles import draw_bulge_disk
    def call():
        draw_bulge_disk((252, 252),
                        flux_bulge=3e5, r_e_bulge=0.15, q_bulge=0.8,
                        flux_disk=7e5,  r_e_disk=0.40, q_disk=0.5,
                        theta_rad=0.3, x0=126, y0=126, pixel_scale=0.05)
    best, _ = _bench("draw_bulge_disk 252² csub_auto", call)
    assert best < 600.0, f"draw_bulge_disk too slow: {best:.1f} ms"


# ---------------------------------------------------------------------------
# Bilinear upsample
# ---------------------------------------------------------------------------

def test_bench_bilinear_upsample_3x():
    """NISP native (84²) → VIS LR (252²) via bilinear interpolation.

    One call is made per NISP band in the forward model.
    """
    from euclid_polish.sky.observation.resample import bilinear_upsample
    rng = np.random.default_rng(0)
    img = rng.normal(size=(84, 84)).astype(np.float32)
    def call():
        bilinear_upsample(img, factor=3)
    best, _ = _bench("bilinear_upsample 84² → 252²", call)
    assert best < 20.0, f"bilinear_upsample too slow: {best:.1f} ms"


def test_bench_cubic_upsample_3x():
    """Cubic-spline alternative for the same case.

    This remains available as an experimental alternative to the Q1-matched
    bilinear default.
    """
    from euclid_polish.sky.observation.resample import cubic_upsample
    rng = np.random.default_rng(0)
    img = rng.normal(size=(84, 84)).astype(np.float32)
    def call():
        cubic_upsample(img, factor=3)
    best, _ = _bench("cubic_upsample 84² → 252²", call)
    assert best < 20.0, f"cubic_upsample too slow: {best:.1f} ms"


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def test_bench_forward_one_field_small():
    """End-to-end forward (4-band PSF + noise + resample) on a 96² HR field.

    Baseline ~1 ms. Tiny because all PSFs are small Gaussians and the
    FFT-convolve on 96×96 is fast.
    """
    from euclid_polish.image import Image
    from euclid_polish.sky.observation.observation_simulator import (
        ObservationSimulator,
        ObservationSimulatorConfig,
    )

    rng = np.random.default_rng(0)
    hr = Image(
        data=rng.uniform(0, 1e3, size=(96, 96, 4)).astype(np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES,
        is_clean=True,
    )
    fwd = ObservationSimulator(config=ObservationSimulatorConfig(add_noise=False))
    noise_rng = np.random.default_rng(1)

    def call():
        fwd.process(hr, rng=noise_rng)

    best, _ = _bench("ObservationSimulator.process 96² noise=False", call)
    assert best < 50.0, f"forward(96²) too slow: {best:.1f} ms"


def test_bench_forward_one_field_realistic():
    """End-to-end forward at the default 252² HR field, noise ON.

    Baseline ~4 ms. The dominant cost is the four 252² FFT convolutions;
    Poisson + Gaussian sampling adds a constant ~1 ms.
    """
    from euclid_polish.image import Image
    from euclid_polish.sky.observation.observation_simulator import (
        ObservationSimulator,
        ObservationSimulatorConfig,
    )

    rng = np.random.default_rng(0)
    hr = Image(
        data=rng.uniform(0, 1e3, size=(252, 252, 4)).astype(np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES,
        is_clean=True,
    )
    fwd = ObservationSimulator(config=ObservationSimulatorConfig(add_noise=True))
    noise_rng = np.random.default_rng(1)

    def call():
        fwd.process(hr, rng=noise_rng)

    best, _ = _bench("ObservationSimulator.process 252² noise=True", call, n_repeats=3)
    assert best < 100.0, f"forward(252²) too slow: {best:.1f} ms"


# ---------------------------------------------------------------------------
# Catalog sampling
# ---------------------------------------------------------------------------

def test_bench_catalog_sample():
    """Drawing 100 random galaxies from the (tiny test) catalog."""
    from tests._tiny_catalog import TinyCosmosCatalog
    cat = TinyCosmosCatalog(n_galaxies=2_000, seed=0)
    rng = np.random.default_rng(0)
    def call():
        for _ in range(100):
            cat.sample_galaxy(rng)
    best, _ = _bench("TinyCosmosCatalog.sample_galaxy ×100", call)
    assert best < 50.0, f"catalog sampling too slow: {best:.1f} ms"


# ---------------------------------------------------------------------------
# Lens render
# ---------------------------------------------------------------------------

def test_bench_lens_render_multiband():
    """One current TNG-backed lens system on a 128² four-band canvas."""
    from pathlib import Path

    from euclid_polish.image import Image, Role
    from euclid_polish.sky.generation.lens_population import (
        render_lens_to_multiband_canvas,
        sample_lens_geometry,
    )
    from euclid_polish.tng.types import (
        NominalRadiusGeometry,
        RenderedTNG,
        TNGRenderTrace,
        TNGRotation,
        TNGView,
    )

    params = sample_lens_geometry(np.random.default_rng(0), 250.0)
    assert params is not None
    native_re_px = 20.0
    scale_factor = 0.5
    trace = TNGRenderTrace(
        view=TNGView(Path("unused"), "benchmark", 1, native_re_px),
        rotation=TNGRotation(),
        geometry=NominalRadiusGeometry(
            target_re_arcsec=(
                native_re_px * Config.DEFAULT_PIXEL_SCALE * scale_factor
            ),
            scale_factor=scale_factor,
            radius_rendering="benchmark_shrink_only",
            radius_renderer_fingerprint="benchmark-fixture",
        ),
    )

    def rendered_stamp(side: int) -> RenderedTNG:
        return RenderedTNG(
            image=Image(
                data=np.ones(
                    (side, side, Config.NUM_LR_CHANNELS),
                    dtype=np.float32,
                ),
                pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                band_names=tuple(Config.LR_INPUT_BAND_NAMES),
                is_clean=True,
                role=Role.CLEAN,
            ),
            trace=trace,
        )

    lens_stamp = rendered_stamp(48)
    source_stamp = rendered_stamp(32)

    def call():
        canvas = np.zeros((128, 128, Config.NUM_LR_CHANNELS), dtype=np.float32)
        render_lens_to_multiband_canvas(
            canvas,
            params=params,
            lens_light_stamp=lens_stamp,
            source_stamp=source_stamp,
        )

    best, _ = _bench("render_lens_to_multiband_canvas 128²", call)
    assert best < 600.0, f"lens render too slow: {best:.1f} ms"


# ---------------------------------------------------------------------------
# Data loader (asinh stretch caching)
# ---------------------------------------------------------------------------

def test_bench_asinh_stretch_lr_cached():
    """Per-band asinh stretch on a 64² 4-channel batch — should be cache-fast."""
    import tensorflow as tf

    from euclid_polish.training.augmentation import asinh_stretch_lr

    rng = np.random.default_rng(0)
    x = tf.constant(rng.normal(size=(4, 64, 64, 4), scale=500.0).astype(np.float32))

    def call():
        _ = asinh_stretch_lr(x)

    best, _ = _bench("asinh_stretch_lr 4×64²×4ch", call, n_repeats=20)
    # Baseline ~0.2 ms thanks to the lru_cached stretch-scale tensors.
    assert best < 10.0, f"asinh stretch too slow: {best:.1f} ms"


# ---------------------------------------------------------------------------
# TFRecord I/O
# ---------------------------------------------------------------------------

def test_bench_compute_sersic_stamp_n4_compact():
    """``compute_sersic_stamp`` alone for the worst (compact bulge) case.

    Baseline ~5 ms with the adaptive core+wings sampling.
    """
    from euclid_polish.sky.generation.profiles import compute_sersic_stamp
    def call():
        compute_sersic_stamp((256, 256), n=4.0, r_e=0.10, q=0.8, theta_rad=0.3,
                             x0=128, y0=128, pixel_scale=0.05)
    best, _ = _bench("compute_sersic_stamp n=4 r_e=0.1\" 256²", call)
    assert best < 100.0


def test_bench_add_sersic_to_bands_vs_loop():
    """4-band broadcast vs naive per-band loop — quantifies the speedup."""
    from euclid_polish.sky.generation.profiles import add_sersic_to_bands, draw_sersic
    H = W = 252
    flux_per_band = np.array([1e5, 5e4, 3e4, 2e4], dtype=np.float32)
    canvas_bcast = np.zeros((H, W, 4), dtype=np.float32)
    canvas_loop  = np.zeros((H, W, 4), dtype=np.float32)
    common = {"n": 4.0, "r_e": 0.20, "q": 0.8, "theta_rad": 0.3,
                  "x0": W / 2, "y0": H / 2, "pixel_scale": 0.05}

    def call_bcast():
        canvas_bcast.fill(0)
        add_sersic_to_bands(canvas_bcast, flux_per_band=flux_per_band, **common)

    def call_loop():
        canvas_loop.fill(0)
        for k in range(4):
            draw_sersic((H, W), flux=float(flux_per_band[k]),
                        out=canvas_loop[..., k], add=True, **common)

    best_b, _ = _bench("add_sersic_to_bands 4-band broadcast 252²", call_bcast)
    best_l, _ = _bench("draw_sersic × 4 bands (loop) 252²", call_loop)
    # Broadcast should be at least ~3× faster than the loop on identical
    # geometry; assert weakly to allow CI noise.
    assert best_b < best_l * 0.5, \
        f"broadcast={best_b:.1f} ms not meaningfully faster than loop={best_l:.1f} ms"


def test_bench_evaluate_sersic_at_coords():
    """Arbitrary-coord evaluator on a 252² grid — used by the lens renderer."""
    from euclid_polish.sky.generation.profiles import evaluate_sersic_at_coords
    H = W = 252
    yy, xx = np.indices((H, W), dtype=np.float64)
    x_arcsec = (xx - W / 2) * 0.05
    y_arcsec = (yy - H / 2) * 0.05

    def call():
        evaluate_sersic_at_coords(
            x_arcsec, y_arcsec, flux=1e5, n=4.0, r_e_arcsec=0.20,
            q=0.7, theta_rad=0.3, pixel_scale=0.05,
        )
    best, _ = _bench("evaluate_sersic_at_coords 252²", call)
    assert best < 200.0


def test_bench_forward_512_production():
    """Forward model at the production HR size (512²).

    Baseline ~10-15 ms / field. FFT convolutions on 512² are the cost.
    """
    from euclid_polish.image import Image
    from euclid_polish.sky.observation.observation_simulator import (
        ObservationSimulator,
        ObservationSimulatorConfig,
    )
    rng = np.random.default_rng(0)
    hr = Image(
        data=rng.uniform(0, 1e3, size=(510, 510, 4)).astype(np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES,
        is_clean=True,
    )
    fwd = ObservationSimulator(config=ObservationSimulatorConfig(add_noise=True))
    noise_rng = np.random.default_rng(1)

    def call():
        fwd.process(hr, rng=noise_rng)

    best, _ = _bench("ObservationSimulator.process 512² noise=True", call, n_repeats=3)
    assert best < 300.0


def test_bench_tfrecord_roundtrip(tmp_path):
    """Write and read back 16 multi-band records (96² each)."""
    import numpy as np

    from euclid_polish.image import Image
    from euclid_polish.image.tfio import (
        read_images,
        write_images,
    )

    rng = np.random.default_rng(0)
    imgs = [
        Image(
            data=rng.normal(size=(96, 96, 4)).astype(np.float32),
            pixel_scale_arcsec=0.10,
            band_names=Config.LR_INPUT_BAND_NAMES,
            is_clean=False,
        )
        for _ in range(16)
    ]

    def write_call():
        write_images(imgs, "bench", records_dir=str(tmp_path))

    def read_call():
        read_images(str(tmp_path / "bench.tfrecord"), num_images=16)

    write_call()  # warmup write
    _bench("write 16× 96²×4ch", write_call)
    _bench("read  16× 96²×4ch", read_call)
