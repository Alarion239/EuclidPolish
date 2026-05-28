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


# Default per-band ratios for tests — VIS=1.0 by construction, NISP
# bands take order-of-magnitude typical values. Real value at runtime
# comes from the catalog; for tests any plausible vector works.
DEFAULT_TEST_RATIOS = np.array([1.0, 0.20, 0.30, 0.25], dtype=np.float32)

# Loose noise budget for tests so the bright-stamp filter doesn't kick
# in on the synthetic Gaussian tiles. ``max_rel_noise = 1e9`` is the
# "disable filter" knob in the script's argument range; we use a
# huge but finite value so the threshold computation still runs.
DISABLE_FILTER_MAX_REL_NOISE = 1.0e9


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
    mod._WORKER_TYPICAL_RATIOS = None
    mod._WORKER_SIGMA_LR = 0.0
    mod._WORKER_MAX_RELATIVE_NOISE = 0.0
    yield
    mod._WORKER_KERNEL = None
    mod._WORKER_IMAGE_SIZE = 0
    mod._WORKER_TYPICAL_RATIOS = None
    mod._WORKER_SIGMA_LR = 0.0
    mod._WORKER_MAX_RELATIVE_NOISE = 0.0


def _init_worker_with_kernel(mod, kernel_path: str, image_size: int,
                              *, sigma_lr: float = 100.0,
                              max_rel: float = DISABLE_FILTER_MAX_REL_NOISE,
                              star_fwhm: float = 2.5,
                              star_threshold_sigma: float = 0.0):
    """Wrapper around ``_init_worker`` for the common test setup.

    ``sigma_lr`` defaults to a moderate value (~ Euclid VIS budget),
    ``max_rel`` defaults to "essentially infinite" so the bright-stamp
    filter does not reject synthetic Gaussian blobs, and
    ``star_threshold_sigma`` defaults to 0 (star detection disabled) so
    existing tests don't trip it unless they opt in.
    """
    mod._init_worker(
        kernel_path,
        image_size,
        DEFAULT_TEST_RATIOS,
        sigma_lr,
        max_rel,
        star_fwhm,
        star_threshold_sigma,
    )


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
    the kernel + typical per-band ratios into module globals. Per-task
    pickling of these would otherwise be ~1 MB × 6400 tasks of pointless
    serialisation."""

    def test_populates_module_globals(self, tmp_path):
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        assert mod._WORKER_KERNEL is None      # fresh module
        assert mod._WORKER_IMAGE_SIZE == 0
        assert mod._WORKER_TYPICAL_RATIOS is None
        _init_worker_with_kernel(mod, krn, image_size=128)
        assert mod._WORKER_KERNEL is not None
        assert mod._WORKER_KERNEL.shape == (63, 63)
        assert mod._WORKER_IMAGE_SIZE == 128
        np.testing.assert_array_equal(
            mod._WORKER_TYPICAL_RATIOS, DEFAULT_TEST_RATIOS,
        )

    def test_is_idempotent(self, tmp_path):
        """Calling twice — e.g. during local debugging — must not blow
        up or corrupt state. The second call just overwrites the
        globals with whatever was passed."""
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        _init_worker_with_kernel(mod, krn, image_size=64)
        _init_worker_with_kernel(mod, krn, image_size=128)
        assert mod._WORKER_IMAGE_SIZE == 128

    def test_process_without_init_raises(self):
        """If ``initializer=`` was forgotten on the pool, the worker
        function fails loudly with a clear message instead of silently
        crashing on a None deref in the FFT."""
        mod = _load_script()
        task = (0, 150.1, 2.3, "/nope.fits", 64, 42)
        with pytest.raises(RuntimeError, match="_init_worker"):
            mod._process_one_galaxy(task)


# ---------------------------------------------------------------------------
# _is_stamp_too_bright — bright-pixel rejection filter
# ---------------------------------------------------------------------------

class TestIsStampTooBright:
    """The analytic-A forward operator rings around very bright pixels:
    ``A(ε)`` peak is ``|A|_peak · √S_max``. The filter rejects stamps
    where that would exceed ``max_relative_noise × σ_LR``. Tests cover
    threshold crossings, the zero/negative pixel edge case, and the
    scaling behaviour of the ``max_relative_noise`` knob."""

    def _make_kernel_and_sigma(self, *, a_peak: float = 0.5,
                                sigma_lr: float = 10.0) -> tuple:
        """Build a tiny diff kernel with a known peak amplitude.

        Returns ``(kernel, sigma_lr)`` — kernel is a 5×5 array with the
        centre pixel set to ``a_peak`` (and zeros elsewhere).
        """
        k = np.zeros((5, 5), dtype=np.float32)
        k[2, 2] = a_peak
        return k, sigma_lr

    def _stamp_with_peak(self, s_max: float) -> np.ndarray:
        """``(H, W, 4)`` HR cube with channel 0 (VIS) peak at S_max."""
        stamp = np.zeros((8, 8, 4), dtype=np.float32)
        stamp[4, 4, 0] = float(s_max)
        return stamp

    def test_faint_stamp_passes(self):
        """A pixel where √S_max · |A|_peak is well below σ_LR must pass."""
        mod = _load_script()
        kernel, sigma_lr = self._make_kernel_and_sigma(
            a_peak=0.5, sigma_lr=10.0,
        )
        # S_max=4 → √4 · 0.5 = 1.0 e⁻ artefact, 5 × 10 = 50 budget. Pass.
        stamp = self._stamp_with_peak(4.0)
        reject, diag = mod._is_stamp_too_bright(
            stamp,
            diff_kernel=kernel,
            sigma_lr_band=sigma_lr,
            max_relative_noise=5.0,
        )
        assert reject is False
        assert diag["S_max"] == pytest.approx(4.0)
        assert diag["artifact_sigma"] < diag["threshold"]

    def test_bright_stamp_rejects(self):
        """A pixel where √S_max · |A|_peak exceeds k · σ_LR must reject."""
        mod = _load_script()
        kernel, sigma_lr = self._make_kernel_and_sigma(
            a_peak=0.5, sigma_lr=10.0,
        )
        # S_max=1e6 → √1e6 · 0.5 = 500 e⁻ artefact, 5 × 10 = 50 budget.
        # 500 > 50 → reject.
        stamp = self._stamp_with_peak(1.0e6)
        reject, diag = mod._is_stamp_too_bright(
            stamp,
            diff_kernel=kernel,
            sigma_lr_band=sigma_lr,
            max_relative_noise=5.0,
        )
        assert reject is True
        assert diag["artifact_sigma"] > diag["threshold"]

    def test_threshold_scales_with_max_relative_noise(self):
        """Same stamp, tighter ``max_relative_noise`` → rejection;
        looser → pass. Confirms the knob actually controls the cut."""
        mod = _load_script()
        kernel, sigma_lr = self._make_kernel_and_sigma(
            a_peak=0.5, sigma_lr=10.0,
        )
        # √100 · 0.5 = 5; threshold = k · 10. So:
        #   k = 1 → 5 < 10 (pass);  k = 0.1 → 5 > 1 (reject).
        stamp = self._stamp_with_peak(100.0)
        loose, _ = mod._is_stamp_too_bright(
            stamp, diff_kernel=kernel, sigma_lr_band=sigma_lr,
            max_relative_noise=1.0,
        )
        tight, _ = mod._is_stamp_too_bright(
            stamp, diff_kernel=kernel, sigma_lr_band=sigma_lr,
            max_relative_noise=0.1,
        )
        assert loose is False
        assert tight is True

    def test_zero_or_negative_pixel_passes(self):
        """A stamp with peak ≤ 0 has no shot-noise contribution and must
        not be rejected — the ``max(S_max, 0)`` clamp inside the sqrt
        keeps the math finite."""
        mod = _load_script()
        kernel, sigma_lr = self._make_kernel_and_sigma(
            a_peak=0.5, sigma_lr=10.0,
        )
        stamp_zero = self._stamp_with_peak(0.0)
        reject, diag = mod._is_stamp_too_bright(
            stamp_zero,
            diff_kernel=kernel,
            sigma_lr_band=sigma_lr,
            max_relative_noise=5.0,
        )
        assert reject is False
        assert diag["artifact_sigma"] == pytest.approx(0.0)
        # All-negative cube (e.g. sky-subtracted residual) should also pass.
        stamp_neg = np.full((4, 4, 4), -3.0, dtype=np.float32)
        reject_neg, _ = mod._is_stamp_too_bright(
            stamp_neg,
            diff_kernel=kernel,
            sigma_lr_band=sigma_lr,
            max_relative_noise=5.0,
        )
        assert reject_neg is False


# ---------------------------------------------------------------------------
# _has_bright_point_source — star detection / rejection
# ---------------------------------------------------------------------------

class TestHasBrightPointSource:
    """A bright point source (star) must be detected; a smooth extended
    blob (galaxy) must not — that's what keeps galaxy cutouts while
    dropping the A(ε)-ringing-prone stars."""

    @staticmethod
    def _gauss(side, cx, cy, fwhm, amp):
        sigma = fwhm / 2.3548
        y, x = np.mgrid[:side, :side]
        return (amp * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2)
                               / (2 * sigma ** 2)))).astype(np.float32)

    def test_point_source_detected(self):
        mod = _load_script()
        rng = np.random.default_rng(0)
        img = rng.normal(0.0, 1.0, size=(96, 96)).astype(np.float32)
        # Sharp, bright PSF-like star (FWHM ~2.5 px) well above the noise.
        img += self._gauss(96, 48, 48, fwhm=2.5, amp=400.0)
        has_star, diag = mod._has_bright_point_source(
            img, fwhm_px=2.5, threshold_sigma=20.0,
        )
        assert has_star is True
        assert diag["n_sources"] >= 1

    def test_extended_blob_not_flagged(self):
        mod = _load_script()
        rng = np.random.default_rng(1)
        img = rng.normal(0.0, 1.0, size=(96, 96)).astype(np.float32)
        # Broad, low-surface-brightness galaxy-like blob (FWHM ~20 px):
        # bright in total but not PSF-sharp, so DAOStarFinder's
        # sharpness cut rejects it.
        img += self._gauss(96, 48, 48, fwhm=20.0, amp=40.0)
        has_star, _ = mod._has_bright_point_source(
            img, fwhm_px=2.5, threshold_sigma=20.0,
        )
        assert has_star is False

    def test_disabled_when_threshold_zero(self):
        mod = _load_script()
        img = self._gauss(64, 32, 32, fwhm=2.5, amp=1000.0)
        has_star, _ = mod._has_bright_point_source(
            img, fwhm_px=2.5, threshold_sigma=0.0,
        )
        assert has_star is False


# ---------------------------------------------------------------------------
# _process_one_galaxy — happy path + failure modes
# ---------------------------------------------------------------------------

class TestProcessOneGalaxyHappyPath:

    def test_returns_hr_lr_cubes_with_correct_shape(self, tmp_path):
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(tmp_path, side_pix=400)
        _init_worker_with_kernel(mod, krn, image_size=64)

        task = (
            7,             # catalog_idx (only used to label the result)
            150.1, 2.3,    # ra, dec (= tile centre)
            tile,
            200,           # hlsp_side_pix
            42,            # seed
        )
        result, reason = mod._process_one_galaxy(task)
        assert result is not None
        assert reason is None
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
        _init_worker_with_kernel(mod, krn, image_size=64)
        task = (0, 150.1, 2.3, tile, 200, 1)
        result, _ = mod._process_one_galaxy(task)
        _, hr, lr = result
        assert np.isfinite(hr).all()
        assert np.isfinite(lr).all()

    def test_hr_has_signal_at_centre(self, tmp_path):
        """Sanity: the synthetic tile has a centred Gaussian blob;
        the HR cutout should carry most of its energy near the centre."""
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(tmp_path, side_pix=400)
        _init_worker_with_kernel(mod, krn, image_size=64)
        task = (0, 150.1, 2.3, tile, 200, 1)
        result, _ = mod._process_one_galaxy(task)
        _, hr, _ = result
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
        _init_worker_with_kernel(mod, krn, image_size=64)
        # 10 deg off — way outside the tile footprint.
        task = (0, 160.0, 12.0, tile, 200, 42)
        result, reason = mod._process_one_galaxy(task)
        assert result is None
        assert reason == "cutout-failed"

    def test_missing_file_returns_none(self, tmp_path):
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        _init_worker_with_kernel(mod, krn, image_size=64)
        task = (0, 150.1, 2.3,
                str(tmp_path / "does-not-exist.fits"), 200, 42)
        result, reason = mod._process_one_galaxy(task)
        assert result is None
        assert reason == "cutout-failed"

    def test_cutout_smaller_than_image_size_returns_none(self, tmp_path):
        """If the requested ``hlsp_side_pix`` resamples down to fewer
        HR pixels than ``image_size``, we can't make the HR cube and
        should bail rather than zero-pad and quietly produce noise."""
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(tmp_path, side_pix=200)
        # Demand a huge HR side that the tiny HLSP cutout can't fill.
        _init_worker_with_kernel(mod, krn, image_size=2000)
        task = (0, 150.1, 2.3, tile, 50, 42)
        result, reason = mod._process_one_galaxy(task)
        assert result is None
        assert reason == "cutout-too-small"

    def test_all_nan_cutout_returns_none(self, tmp_path):
        """A degenerate cutout where every pixel is NaN can't be turned
        into a usable HR cube — propagate the None instead of writing a
        TFRecord full of NaNs that'd poison training."""
        from astropy.io import fits
        from astropy.wcs import WCS

        # Build a NaN-filled HLSP-shaped tile.
        side = 400
        w = WCS(naxis=2)
        w.wcs.crpix = [side / 2 + 0.5, side / 2 + 0.5]
        w.wcs.cdelt = [-0.03 / 3600.0, 0.03 / 3600.0]
        w.wcs.crval = [150.1, 2.3]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        data = np.full((side, side), np.nan, dtype=np.float32)
        path = str(tmp_path / "nan_tile.fits")
        fits.PrimaryHDU(data, header=w.to_header()).writeto(path, overwrite=True)

        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        _init_worker_with_kernel(mod, krn, image_size=64)
        task = (0, 150.1, 2.3, path, 200, 42)
        result, reason = mod._process_one_galaxy(task)
        assert result is None
        assert reason == "hst-cube-allnan"

    def test_bright_stamp_rejected(self, tmp_path):
        """A stamp whose HR cube would produce A(ε) ringing larger than
        ``max_relative_noise × σ_LR`` must be rejected before forward
        modelling — the whole point of the new filter."""
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        # Big bright blob so the resulting HR cube peak is large in
        # Euclid-electron units.
        tile = _make_synthetic_tile(
            tmp_path, side_pix=400, blob_flux=1.0e6,
        )
        # Pin sigma_lr small + max_rel small → easy rejection. Star
        # detection off (threshold 0) so this isolates the bright-pixel
        # filter.
        mod._init_worker(
            krn, 64, DEFAULT_TEST_RATIOS,
            1.0,   # sigma_lr_band (tight budget)
            0.01,  # max_relative_noise (very strict)
            2.5,   # star_fwhm_px
            0.0,   # star_threshold_sigma (disabled)
        )
        task = (0, 150.1, 2.3, tile, 200, 42)
        result, reason = mod._process_one_galaxy(task)
        assert result is None
        assert reason == "rejected-bright"


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
        _init_worker_with_kernel(mod, krn, image_size=64)
        task_a = (0, 150.1, 2.3, tile, 200, 12345)
        task_b = (0, 150.1, 2.3, tile, 200, 12345)
        res_a, _ = mod._process_one_galaxy(task_a)
        res_b, _ = mod._process_one_galaxy(task_b)
        _, hr_a, lr_a = res_a
        _, hr_b, lr_b = res_b
        np.testing.assert_array_equal(hr_a, hr_b)
        np.testing.assert_array_equal(lr_a, lr_b)

    def test_different_seed_different_lr(self, tmp_path):
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(tmp_path, side_pix=400)
        _init_worker_with_kernel(mod, krn, image_size=64)
        t1 = (0, 150.1, 2.3, tile, 200, 12345)
        t2 = (0, 150.1, 2.3, tile, 200, 99999)
        r1, _ = mod._process_one_galaxy(t1)
        r2, _ = mod._process_one_galaxy(t2)
        _, hr1, lr1 = r1
        _, hr2, lr2 = r2
        # HR is deterministic (no noise) — should be identical.
        np.testing.assert_array_equal(hr1, hr2)
        # LR depends on the noise draw — should differ.
        assert not np.array_equal(lr1, lr2)


# ---------------------------------------------------------------------------
# Pool integration — end-to-end small parallel run
# ---------------------------------------------------------------------------

class TestHstNativePhotometry:
    """The HST→Euclid chain preserves HST's native photometry — every
    source in the cutout keeps its real Euclid magnitude, no
    catalog-flux normalisation.
    """

    def test_resample_preserves_total_flux(self, tmp_path):
        """``_resample_hlsp_to_hr`` integrates over the larger HR pixel
        instead of just interpolating surface brightness."""
        mod = _load_script()
        side = 100
        hlsp = np.ones((side, side), dtype=np.float32)
        hr = mod._resample_hlsp_to_hr(hlsp, hlsp_scale=0.03, hr_scale=0.05)

        # Per-pixel value ≈ 2.78 in the interior.
        b = 5
        assert np.allclose(hr[b:-b, b:-b], (0.05 / 0.03) ** 2, atol=1e-3)

        input_total  = float(hlsp.sum())
        output_total = float(hr.sum())
        assert output_total == pytest.approx(input_total, rel=0.05)

    def test_hst_to_euclid_hr_cube_vis_scaling(self, tmp_path):
        """A constant HST rate maps to Euclid VIS electrons via the
        documented ``rate_ratio × t_total`` formula."""
        from euclid_polish.config import Config

        mod = _load_script()
        rate_per_hr_pix = np.full((8, 8), 1.0, dtype=np.float32)
        ratios = np.array([1.0, 0.1, 0.2, 0.15], dtype=np.float32)
        cube = mod._hst_to_euclid_hr_cube(rate_per_hr_pix, ratios)

        vis = Config.BAND_VIS
        rate_ratio = 10.0 ** (
            -0.4 * (Config.HST_ACS_F814W_AB_ZP_E_PER_S
                    - vis.zeropoint_ab_e_per_s)
        )
        expected_vis = 1.0 * rate_ratio * vis.t_total_s
        assert cube[..., 0] == pytest.approx(expected_vis, rel=1e-5)

        for k in (1, 2, 3):
            assert cube[..., k] == pytest.approx(
                expected_vis * ratios[k], rel=1e-5,
            )

    def test_hst_to_euclid_hr_cube_handles_signed_residual(self):
        """Symmetric residual noise (positive + negative pixels) must
        pass through cleanly — Negative pixels are allowed in the HR
        cube; the forward model's Poisson step clips them at zero
        internally before the draw.
        """
        mod = _load_script()
        rate = np.array([[1.0, -0.2], [-0.3, 0.5]], dtype=np.float32)
        cube = mod._hst_to_euclid_hr_cube(
            rate, np.array([1.0, 0.2, 0.3, 0.25], dtype=np.float32),
        )
        # Negative input pixel ⇒ negative output pixel (no clipping).
        assert cube[0, 1, 0] < 0
        assert cube[1, 0, 0] < 0

    def test_hst_to_euclid_hr_cube_rejects_bad_ratios(self):
        mod = _load_script()
        rate = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="typical_band_ratios"):
            mod._hst_to_euclid_hr_cube(rate, np.array([1.0, 0.2]))

    def test_hst_to_euclid_hr_cube_returns_none_on_all_nan(self):
        mod = _load_script()
        rate = np.full((4, 4), np.nan, dtype=np.float32)
        out = mod._hst_to_euclid_hr_cube(
            rate, np.array([1.0, 0.2, 0.3, 0.25], dtype=np.float32),
        )
        assert out is None

    def test_nisp_channels_use_typical_ratios(self, tmp_path):
        """NISP[k] = VIS × typical_band_ratios[k] per pixel."""
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(tmp_path, side_pix=400)
        ratios = np.array([1.0, 0.15, 0.25, 0.20], dtype=np.float32)
        mod._init_worker(
            krn, 64, ratios, 100.0, DISABLE_FILTER_MAX_REL_NOISE,
            2.5, 0.0,   # star_fwhm_px, star_threshold_sigma (disabled)
        )
        task = (0, 150.1, 2.3, tile, 200, 1)
        result, _ = mod._process_one_galaxy(task)
        _, hr, _ = result

        vis = hr[..., 0]
        for k in (1, 2, 3):
            band_hr = hr[..., k]
            mask = np.abs(vis) > 0.1 * np.abs(vis).max()
            np.testing.assert_allclose(
                band_hr[mask] / vis[mask],
                ratios[k],
                rtol=1e-5,
                err_msg=f"NISP[{k}] / VIS != typical_band_ratios[{k}]",
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
            (i, 150.1, 2.3, tile, 200, 1000 + i)
            for i in range(n_tasks)
        ]

        with ProcessPoolExecutor(
            max_workers=2,
            initializer=mod._init_worker,
            initargs=(krn, 64, DEFAULT_TEST_RATIOS,
                      100.0, DISABLE_FILTER_MAX_REL_NOISE, 2.5, 0.0),
        ) as pool:
            results = list(pool.map(mod._process_one_galaxy, tasks))

        assert len(results) == n_tasks
        successes = [r for r in results if r[0] is not None]
        assert len(successes) == n_tasks, (
            "all tasks should succeed against the centred tile"
        )
        # Catalog indices preserved through serialisation.
        recovered_ids = sorted(r[0][0] for r in results)
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
            (i, 150.0, 2.0, tile, 100, i)
            for i in range(3)
        ]
        bad_tasks = [
            (i + 100, 160.0, 12.0, tile, 100, i)
            for i in range(2)
        ]
        with ProcessPoolExecutor(
            max_workers=2,
            initializer=mod._init_worker,
            initargs=(krn, 32, DEFAULT_TEST_RATIOS,
                      100.0, DISABLE_FILTER_MAX_REL_NOISE, 2.5, 0.0),
        ) as pool:
            results = list(pool.map(
                mod._process_one_galaxy, ok_tasks + bad_tasks,
            ))
        successes = [r for r in results if r[0] is not None]
        failures  = [r for r in results if r[0] is None]
        assert len(successes) == 3
        assert len(failures)  == 2


# ---------------------------------------------------------------------------
# Per-pixel σ_LR computation (used to anchor the bright-stamp threshold)
# ---------------------------------------------------------------------------

class TestEuclidVisSigmaLrPerPixel:
    """The threshold for the bright-stamp filter is anchored to the
    Euclid VIS per-pixel LR noise budget. Verify the helper agrees
    with the closed-form expression and that VIS BandConfig values
    drive it (so a future config change is visible in the test)."""

    def test_matches_closed_form(self):
        from euclid_polish.config import Config
        mod = _load_script()
        sigma = mod._euclid_vis_sigma_lr_per_pixel()
        band = Config.BAND_VIS
        t_total = band.t_total_s
        pixel_area = band.pixel_scale_lr_arcsec ** 2
        sky = band.sky_e_per_s_per_arcsec2 * pixel_area * t_total
        dark = band.dark_e_per_s_per_pix * t_total
        read = band.n_exposures * band.read_noise_e ** 2
        expected = float(np.sqrt(sky + dark + read))
        assert sigma == pytest.approx(expected, rel=1e-6)
        assert sigma > 0
