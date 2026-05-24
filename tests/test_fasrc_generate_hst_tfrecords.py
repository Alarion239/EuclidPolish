"""Tests for the parallel step-4 worker (HST → Euclid TFRecord generation).

The script does heavy work per galaxy (FITS cutout + spline resample +
forward-model with the differential kernel) inside a ProcessPoolExecutor
pool. These tests cover the worker function in isolation against
synthetic inputs — no FASRC, no real HLSP tiles, no real catalog —
plus a small end-to-end pool run to make sure the multiprocess wiring
actually works.

Tests are scoped to ``_process_one_galaxy`` and friends rather than
the whole ``main()`` because main() pulls in the COSMOS2025 catalog
loader (slow + opens MB of data files); the per-galaxy contract is
where the real logic + parallelism bugs live anyway.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
# Putting scripts/ on sys.path lets ProcessPoolExecutor's spawned
# workers re-import the script by its real filename, so
# ``_init_worker`` and ``_process_one_galaxy`` are pickle-resolvable
# across process boundaries.
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))


# ---------------------------------------------------------------------------
# Synthetic fixtures — minimal HLSP-style FITS tile + minimal diff kernel
# ---------------------------------------------------------------------------

def _make_synthetic_tile(
    tmp_path, *, side_pix: int = 400, scale_arcsec: float = 0.03,
    ra_centre: float = 150.1, dec_centre: float = 2.3,
    blob_sigma_pix: float = 6.0, blob_flux: float = 100.0,
) -> str:
    """Write a tiny HLSP-style FITS tile with a TAN WCS + a centred blob.

    Just enough for ``astropy.nddata.Cutout2D`` to extract from when
    given the same RA/Dec we used as the tile centre. Returns the path.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    yy, xx = np.mgrid[:side_pix, :side_pix]
    cy = cx = side_pix // 2
    data = blob_flux * np.exp(
        -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * blob_sigma_pix ** 2)
    )
    data = data.astype(np.float32)

    w = WCS(naxis=2)
    w.wcs.crpix = [side_pix / 2 + 0.5, side_pix / 2 + 0.5]
    # RA increases to the left in TAN convention (CDELT1 < 0).
    w.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    w.wcs.crval = [ra_centre, dec_centre]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    hdu = fits.PrimaryHDU(data, header=w.to_header())
    path = tmp_path / "tile.fits"
    hdu.writeto(str(path), overwrite=True)
    return str(path)


def _make_synthetic_kernel(tmp_path, *, side: int = 63) -> str:
    """Build a unit-flux Gaussian as a stand-in for the differential
    kernel — just so the worker's fftconvolve has something to run."""
    from euclid_polish.sky.differential_kernel import DifferentialKernel
    sigma = 2.0
    y, x = np.mgrid[:side, :side]
    cy = cx = (side - 1) / 2.0
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
    g = (g / g.sum()).astype(np.float32)
    dk = DifferentialKernel(
        data=g, pixel_scale_arcsec=0.05,
        euclid_band="VIS", hst_filter="F814W",
        regularisation=1e-3,
    )
    path = tmp_path / "kernel.fits"
    dk.save(str(path))
    return str(path)


def _load_script():
    """Import the script under its real name.

    Using a stable name (``fasrc_generate_hst_tfrecords``) — not a
    dynamic one — is critical for the pool integration tests: the
    pickled reference to ``_process_one_galaxy`` carries the module
    name, and the spawned worker subprocess has to be able to import
    that module by the same name. A unique-per-call name would unpickle
    in the worker to ``ModuleNotFoundError``.

    Module state is reset between tests by the ``reset_worker_globals``
    fixture below so the shared instance doesn't leak ``_WORKER_KERNEL``.
    """
    import fasrc_generate_hst_tfrecords as mod
    return mod


@pytest.fixture(autouse=True)
def reset_worker_globals():
    """Wipe the script's worker-init globals before every test so the
    shared module instance behaves like a fresh import."""
    import fasrc_generate_hst_tfrecords as mod
    mod._WORKER_KERNEL = None
    mod._WORKER_IMAGE_SIZE = 0
    yield
    mod._WORKER_KERNEL = None
    mod._WORKER_IMAGE_SIZE = 0


# ---------------------------------------------------------------------------
# CLI defaults — SLURM env honoured
# ---------------------------------------------------------------------------

class TestDefaultWorkerCount:
    """The ``--n-workers`` default reads SLURM_CPUS_PER_TASK so a SLURM
    job picks up exactly the cores it was allocated. Off-cluster we
    fall back to cpu_count()."""

    def test_honours_slurm_cpus_per_task(self, monkeypatch):
        mod = _load_script()
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "12")
        assert mod._default_n_workers() == 12

    def test_falls_back_to_cpu_count_off_cluster(self, monkeypatch):
        mod = _load_script()
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        n = mod._default_n_workers()
        assert n >= 1
        assert n == max(1, os.cpu_count() or 1)

    def test_ignores_garbage_slurm_value(self, monkeypatch):
        """An empty or non-numeric env var must not crash the default
        resolution — fall through to cpu_count() like the env wasn't set."""
        mod = _load_script()
        for bad in ("", "  ", "lots", "-3", "1.5"):
            monkeypatch.setenv("SLURM_CPUS_PER_TASK", bad)
            n = mod._default_n_workers()
            assert n >= 1


# ---------------------------------------------------------------------------
# Worker initialiser
# ---------------------------------------------------------------------------

class TestInitWorker:
    """The pool calls ``_init_worker`` once per worker process to load
    the kernel into a module global. Per-task pickling of the kernel
    would otherwise be ~1 MB × 6400 tasks of pointless serialisation."""

    def test_populates_module_globals(self, tmp_path):
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        assert mod._WORKER_KERNEL is None      # fresh module
        assert mod._WORKER_IMAGE_SIZE == 0
        mod._init_worker(krn, image_size=128)
        assert mod._WORKER_KERNEL is not None
        assert mod._WORKER_KERNEL.shape == (63, 63)
        assert mod._WORKER_IMAGE_SIZE == 128

    def test_is_idempotent(self, tmp_path):
        """Calling twice — e.g. during local debugging — must not blow
        up or corrupt state. The second call just overwrites the
        globals with whatever was passed."""
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        mod._init_worker(krn, image_size=64)
        mod._init_worker(krn, image_size=128)
        assert mod._WORKER_IMAGE_SIZE == 128

    def test_process_without_init_raises(self):
        """If ``initializer=`` was forgotten on the pool, the worker
        function fails loudly with a clear message instead of silently
        crashing on a None deref in the FFT."""
        mod = _load_script()
        task = (0, 150.1, 2.3, (1.0, 1.0, 1.0, 1.0),
                "/nope.fits", 64, 42)
        with pytest.raises(RuntimeError, match="_init_worker"):
            mod._process_one_galaxy(task)


# ---------------------------------------------------------------------------
# _process_one_galaxy — happy path + failure modes
# ---------------------------------------------------------------------------

class TestProcessOneGalaxyHappyPath:

    def test_returns_hr_lr_cubes_with_correct_shape(self, tmp_path):
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(tmp_path, side_pix=400)
        mod._init_worker(krn, image_size=64)

        task = (
            7,                           # catalog_idx
            150.1, 2.3,                  # ra, dec (= tile centre)
            (100.0, 80.0, 60.0, 50.0),   # per-band fluxes (electrons)
            tile,
            200,                         # hlsp_side_pix
            42,                          # seed
        )
        result = mod._process_one_galaxy(task)
        assert result is not None
        catalog_idx, hr_cube, lr_cube = result
        assert catalog_idx == 7
        # HR is target image_size × NUM_LR_CHANNELS.
        assert hr_cube.shape == (64, 64, 4)
        # LR is HR // 2 (×2 rebin in _make_pair).
        assert lr_cube.shape == (32, 32, 4)
        assert hr_cube.dtype == np.float32
        assert lr_cube.dtype == np.float32

    def test_output_arrays_are_finite(self, tmp_path):
        """No NaN/inf leakage from the noise model or bg-subtract."""
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(tmp_path, side_pix=400)
        mod._init_worker(krn, image_size=64)
        task = (0, 150.1, 2.3, (100.0, 80.0, 60.0, 50.0), tile, 200, 1)
        catalog_idx, hr, lr = mod._process_one_galaxy(task)
        assert np.isfinite(hr).all()
        assert np.isfinite(lr).all()

    def test_hr_has_signal_at_centre(self, tmp_path):
        """Sanity: the synthetic tile has a centred Gaussian blob;
        the HR cutout should carry most of its energy near the centre."""
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(tmp_path, side_pix=400)
        mod._init_worker(krn, image_size=64)
        task = (0, 150.1, 2.3, (100.0, 80.0, 60.0, 50.0), tile, 200, 1)
        _, hr, _ = mod._process_one_galaxy(task)
        # Sum over bands → 2D map of total flux.
        flat = hr.sum(axis=-1)
        cy, cx = np.unravel_index(int(np.argmax(flat)), flat.shape)
        # Argmax should be near the centre (within ±3 px of (32, 32)).
        assert abs(cy - 32) <= 3
        assert abs(cx - 32) <= 3


class TestProcessOneGalaxyFailureModes:

    def test_off_tile_returns_none(self, tmp_path):
        """An RA/Dec far from the tile centre — Cutout2D's mode='strict'
        refuses; we should return None, not crash."""
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(
            tmp_path, side_pix=200, ra_centre=150.0, dec_centre=2.0,
        )
        mod._init_worker(krn, image_size=64)
        # 10 deg off — way outside the tile footprint.
        task = (0, 160.0, 12.0, (1.0, 1.0, 1.0, 1.0), tile, 200, 42)
        assert mod._process_one_galaxy(task) is None

    def test_missing_file_returns_none(self, tmp_path):
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        mod._init_worker(krn, image_size=64)
        task = (0, 150.1, 2.3, (1.0, 1.0, 1.0, 1.0),
                str(tmp_path / "does-not-exist.fits"), 200, 42)
        assert mod._process_one_galaxy(task) is None

    def test_cutout_smaller_than_image_size_returns_none(self, tmp_path):
        """If the requested ``hlsp_side_pix`` resamples down to fewer
        HR pixels than ``image_size``, we can't make the HR cube and
        should bail rather than zero-pad and quietly produce noise."""
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(tmp_path, side_pix=200)
        # Demand a huge HR side that the tiny HLSP cutout can't fill.
        mod._init_worker(krn, image_size=2000)
        task = (0, 150.1, 2.3, (1.0, 1.0, 1.0, 1.0), tile, 50, 42)
        assert mod._process_one_galaxy(task) is None

    def test_zero_flux_template_returns_none(self, tmp_path):
        """``_broadcast_hst_to_4bands`` refuses to scale a flat HST
        stamp (sum ≤ 0). The worker should propagate the None."""
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        # Blob with zero flux → after bg-subtract, hr_clean is 0 every-
        # where and _broadcast_hst_to_4bands returns None.
        tile = _make_synthetic_tile(
            tmp_path, side_pix=400, blob_flux=0.0,
        )
        mod._init_worker(krn, image_size=64)
        task = (0, 150.1, 2.3, (100.0, 80.0, 60.0, 50.0), tile, 200, 42)
        assert mod._process_one_galaxy(task) is None


# ---------------------------------------------------------------------------
# RNG determinism — the trickiest parallel-correctness invariant
# ---------------------------------------------------------------------------

class TestRngDeterminism:
    """Each task carries its own seed so noise patterns are
    reproducible *and* independent across galaxies. The HR cube is
    deterministic regardless (no RNG); only the LR cube depends on
    the seed via the Poisson + read-noise draw."""

    def test_same_seed_same_lr(self, tmp_path):
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(tmp_path, side_pix=400)
        mod._init_worker(krn, image_size=64)
        task_a = (0, 150.1, 2.3, (100.0, 80.0, 60.0, 50.0), tile, 200, 12345)
        task_b = (0, 150.1, 2.3, (100.0, 80.0, 60.0, 50.0), tile, 200, 12345)
        _, hr_a, lr_a = mod._process_one_galaxy(task_a)
        _, hr_b, lr_b = mod._process_one_galaxy(task_b)
        np.testing.assert_array_equal(hr_a, hr_b)
        np.testing.assert_array_equal(lr_a, lr_b)

    def test_different_seed_different_lr(self, tmp_path):
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(tmp_path, side_pix=400)
        mod._init_worker(krn, image_size=64)
        t1 = (0, 150.1, 2.3, (100.0, 80.0, 60.0, 50.0), tile, 200, 12345)
        t2 = (0, 150.1, 2.3, (100.0, 80.0, 60.0, 50.0), tile, 200, 99999)
        _, hr1, lr1 = mod._process_one_galaxy(t1)
        _, hr2, lr2 = mod._process_one_galaxy(t2)
        # HR is deterministic (no noise) — should be identical.
        np.testing.assert_array_equal(hr1, hr2)
        # LR depends on the noise draw — should differ.
        assert not np.array_equal(lr1, lr2)


# ---------------------------------------------------------------------------
# Pool integration — end-to-end small parallel run
# ---------------------------------------------------------------------------

class TestPhotometryChain:
    """End-to-end photometric consistency: catalog electron count
    flows through unit-flux template → 4-band broadcast → convolve
    → sum-rebin → noise, and the LR total electrons match the input
    within the noise budget. If anyone ever rescales per-band flux
    or breaks the unit-flux normalisation, these tests catch it
    before users have to wonder why "noise dominates everything"."""

    def test_hr_cube_preserves_catalog_flux_per_band(self, tmp_path):
        """The unit-flux template scaled by per-band catalog flux
        must sum to exactly that flux in each band. This is what
        ``_broadcast_hst_to_4bands`` is supposed to do — the test
        catches any silent rescaling regression."""
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(
            tmp_path, side_pix=400, blob_flux=1000.0,
        )
        mod._init_worker(krn, image_size=64)

        flux_per_band = (50_000.0, 10_000.0, 15_000.0, 12_000.0)
        task = (0, 150.1, 2.3, flux_per_band, tile, 200, 42)
        _, hr_cube, _ = mod._process_one_galaxy(task)

        for k, expected in enumerate(flux_per_band):
            hr_sum = float(hr_cube[..., k].sum())
            assert hr_sum == pytest.approx(expected, rel=1e-3), (
                f"band {k}: HR sum {hr_sum:.1f} e ≠ catalog flux "
                f"{expected:.1f} e — broadcast lost or added flux"
            )

    def test_lr_cube_total_flux_matches_within_noise(self, tmp_path):
        """LR (dirty) total flux ≈ catalog flux per band, within the
        Poisson + read noise budget on the integrated sum.

        ``apply_band_noise`` is zero-mean by construction: it adds
        ``Poisson(signal+sky+dark) - (sky+dark)`` + Gaussian(0, σ_read),
        so ``E[sum(LR)] = sum(signal)``. The standard deviation on
        the integrated sum is ``sqrt(N_pix) × σ_per_pixel``. We allow
        5σ tolerance to be robust across random seeds + a small (<1%)
        edge-flux-leakage allowance from the convolution.
        """
        from euclid_polish.config import Config

        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        # Bright + well-centred so convolution edge losses are minimal.
        tile = _make_synthetic_tile(
            tmp_path, side_pix=600, blob_flux=1000.0, blob_sigma_pix=8.0,
        )
        mod._init_worker(krn, image_size=128)

        flux_per_band = (200_000.0, 80_000.0, 100_000.0, 90_000.0)
        task = (0, 150.1, 2.3, flux_per_band, tile, 300, 42)
        _, _, lr_cube = mod._process_one_galaxy(task)

        n_pix = lr_cube.shape[0] * lr_cube.shape[1]
        for k, expected in enumerate(flux_per_band):
            band = Config.get_band(Config.LR_INPUT_BAND_NAMES[k])
            sky_e  = (band.sky_e_per_s_per_arcsec2
                      * band.pixel_scale_lr_arcsec ** 2
                      * band.t_total_s)
            dark_e = band.dark_e_per_s_per_pix * band.t_total_s
            sigma_per_px = np.sqrt(
                sky_e + dark_e + band.n_exposures * band.read_noise_e ** 2
                + expected / n_pix         # Poisson term on signal
            )
            sigma_on_sum = np.sqrt(n_pix) * sigma_per_px

            lr_sum = float(lr_cube[..., k].sum())
            tol = 5 * sigma_on_sum + 0.02 * expected
            assert abs(lr_sum - expected) < tol, (
                f"band {Config.LR_INPUT_BAND_NAMES[k]}: LR sum "
                f"{lr_sum:.0f} e vs expected {expected:.0f} e, "
                f"diff {lr_sum - expected:+.0f} > {tol:.0f} tol"
            )

    def test_brighter_catalog_flux_gives_brighter_lr(self, tmp_path):
        """Monotonicity: a 10x brighter catalog magnitude must give
        a 10x brighter LR per-pixel peak (within noise). Pins the
        photometric *sign* of the chain — a unit flip or absolute-
        value bug somewhere would break this even when totals match."""
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(
            tmp_path, side_pix=400, blob_flux=1000.0,
        )
        mod._init_worker(krn, image_size=64)
        seed = 12345

        dim   = (5_000.0,  5_000.0,  5_000.0,  5_000.0)
        bright = (50_000.0, 50_000.0, 50_000.0, 50_000.0)

        _, _, lr_dim    = mod._process_one_galaxy(
            (0, 150.1, 2.3, dim,    tile, 200, seed))
        _, _, lr_bright = mod._process_one_galaxy(
            (0, 150.1, 2.3, bright, tile, 200, seed))

        # Same seed → same noise realisation. Difference is purely the
        # 10x signal scaling. Compare *peaks* (most-significant pixel)
        # because noise dominates the LR sum at the dim end.
        for k in range(4):
            peak_dim    = float(lr_dim[..., k].max())
            peak_bright = float(lr_bright[..., k].max())
            ratio = peak_bright / peak_dim if peak_dim > 0 else 0
            # Ratio should be near 10 modulo noise contributions; allow
            # a wide band because the dim peak is shot-noise-influenced.
            assert 5 < ratio < 20, (
                f"band {k}: peak ratio {ratio:.2f} for 10x flux "
                "scaling is wildly off — sign or unit issue?"
            )


class TestPoolIntegration:
    """ProcessPoolExecutor with the real initialiser + worker. Catches
    pickling regressions and any subtle "module globals don't survive
    fork/spawn" surprises."""

    def test_pool_processes_all_tasks(self, tmp_path):
        from concurrent.futures import ProcessPoolExecutor

        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(tmp_path, side_pix=400)

        n_tasks = 6
        tasks = [
            (i, 150.1, 2.3, (100.0, 80.0, 60.0, 50.0), tile, 200, 1000 + i)
            for i in range(n_tasks)
        ]

        with ProcessPoolExecutor(
            max_workers=2,
            initializer=mod._init_worker,
            initargs=(krn, 64),
        ) as pool:
            results = list(pool.map(mod._process_one_galaxy, tasks))

        assert len(results) == n_tasks
        assert all(r is not None for r in results), (
            "all tasks should succeed against the centred tile"
        )
        # Catalog indices preserved through serialisation.
        recovered_ids = sorted(r[0] for r in results)
        assert recovered_ids == list(range(n_tasks))

    def test_pool_mixes_success_and_skip(self, tmp_path):
        """Some tasks should succeed, others (off-tile) should come
        back as None — the consumer-side logic in main() relies on
        being able to tell them apart."""
        from concurrent.futures import ProcessPoolExecutor

        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(
            tmp_path, side_pix=200,
            ra_centre=150.0, dec_centre=2.0,
        )

        ok_tasks = [
            (i, 150.0, 2.0, (100.0, 80.0, 60.0, 50.0), tile, 100, i)
            for i in range(3)
        ]
        bad_tasks = [
            (i + 100, 160.0, 12.0, (1.0, 1.0, 1.0, 1.0), tile, 100, i)
            for i in range(2)
        ]
        with ProcessPoolExecutor(
            max_workers=2,
            initializer=mod._init_worker,
            initargs=(krn, 32),
        ) as pool:
            results = list(pool.map(
                mod._process_one_galaxy, ok_tasks + bad_tasks,
            ))
        successes = [r for r in results if r is not None]
        failures  = [r for r in results if r is None]
        assert len(successes) == 3
        assert len(failures)  == 2
