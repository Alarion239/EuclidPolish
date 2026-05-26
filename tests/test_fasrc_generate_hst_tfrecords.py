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
    mod._WORKER_TRANSITION_MODEL = None
    mod._WORKER_IMAGE_SIZE = 0
    mod._WORKER_TYPICAL_RATIOS = None
    yield
    mod._WORKER_KERNEL = None
    mod._WORKER_TRANSITION_MODEL = None
    mod._WORKER_IMAGE_SIZE = 0
    mod._WORKER_TYPICAL_RATIOS = None


# Default per-band ratios for tests — VIS=1.0 by construction, NISP
# bands take order-of-magnitude typical values. Real value at runtime
# comes from the catalog; for tests any plausible vector works.
DEFAULT_TEST_RATIOS = np.array([1.0, 0.20, 0.30, 0.25], dtype=np.float32)


def _init_worker_with_kernel(mod, kernel_path: str, image_size: int):
    """Wrapper around ``_init_worker`` that uses the analytic-kernel
    fallback path (no transition model). Tests that don't care about
    the CNN path go through this helper to keep the call signature
    short.
    """
    mod._init_worker(
        kernel_path,
        "",        # no transition model
        12,        # transition_channels (ignored when path is "")
        3,         # transition_inner_layers (ignored when path is "")
        image_size,
        DEFAULT_TEST_RATIOS,
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


class TestInitWorkerTwoStage:
    """When the transition model's summary JSON carries a ``denoiser``
    block, ``_init_worker`` must (a) reconstruct the denoiser with the
    architecture in the JSON, (b) load its weights, and (c) populate
    ``_WORKER_DENOISER_MODEL`` so ``_make_pair`` runs the full chain.
    Missing denoiser file → loud RuntimeError, not silent fallback to
    single-stage (that would corrupt the LR shards in subtle ways)."""

    def _make_minimal_transition_pair(self, tmp_path, with_denoiser: bool):
        """Build a tiny transition model + matching summary JSON.

        Returns (transition_weights_path, denoiser_weights_path_or_None,
        kernel_path). Kernel is a small odd Gaussian, transition model
        is built without a baseline (keeps the test fast — the baseline
        load-path is exercised elsewhere).
        """
        from astropy.io import fits
        import json
        from euclid_polish.training.transition_model import (
            HSTDenoiser, HSTtoEuclidTransition,
            save_denoiser_weights, save_model_weights,
        )

        # Tiny analytic kernel (delta) so the script's fallback path
        # is also valid if we need it.
        krn = tmp_path / "diff_kernel.fits"
        delta = np.zeros((9, 9), dtype=np.float32); delta[4, 4] = 1.0
        fits.PrimaryHDU(delta).writeto(str(krn), overwrite=True)

        # Tiny transition model (no baseline so the test stays fast).
        trans_path = tmp_path / "transition.weights.h5"
        m_trans = HSTtoEuclidTransition(
            channels=4, n_inner_layers=1, kernel_size=3,
        )
        save_model_weights(m_trans, str(trans_path))

        # Matching summary JSON.
        summary = {
            "channels":       4,
            "n_inner_layers": 1,
            "kernel_size":    3,
            "analytic_baseline_kernel": "",
            "baseline_crop":  0,
        }

        denoiser_path = None
        if with_denoiser:
            denoiser_path = tmp_path / "hst_denoiser.weights.h5"
            m_den = HSTDenoiser(channels=4, n_inner_layers=1, kernel_size=3)
            save_denoiser_weights(m_den, str(denoiser_path))
            summary["denoiser"] = {
                "weights_path":   str(denoiser_path),
                "channels":       4,
                "n_inner_layers": 1,
                "kernel_size":    3,
            }

        summary_path = (
            str(trans_path)[:-len(".weights.h5")] + "_summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f)
        return str(trans_path), denoiser_path, str(krn)

    def test_loads_denoiser_when_summary_has_block(self, tmp_path):
        mod = _load_script()
        # Make sure stale state from a prior test isn't leaking in.
        mod._WORKER_DENOISER_MODEL = None
        trans_path, den_path, krn = self._make_minimal_transition_pair(
            tmp_path, with_denoiser=True,
        )
        mod._init_worker(
            krn, trans_path, 12, 3, 64, DEFAULT_TEST_RATIOS,
        )
        assert mod._WORKER_TRANSITION_MODEL is not None
        assert mod._WORKER_DENOISER_MODEL is not None, (
            "summary contained a 'denoiser' block but worker didn't "
            "load it — Phase-2 chain is broken"
        )

    def test_skips_denoiser_when_summary_lacks_block(self, tmp_path):
        mod = _load_script()
        mod._WORKER_DENOISER_MODEL = None
        trans_path, _, krn = self._make_minimal_transition_pair(
            tmp_path, with_denoiser=False,
        )
        mod._init_worker(
            krn, trans_path, 12, 3, 64, DEFAULT_TEST_RATIOS,
        )
        assert mod._WORKER_TRANSITION_MODEL is not None
        assert mod._WORKER_DENOISER_MODEL is None, (
            "summary had no 'denoiser' block — single-stage path "
            "should leave _WORKER_DENOISER_MODEL = None"
        )

    def test_missing_denoiser_file_raises(self, tmp_path):
        """Summary records a denoiser path that doesn't exist locally
        (the FASRC-style absolute path won't resolve when running
        elsewhere). The worker must fail loudly rather than silently
        falling back to single-stage and writing wrong LR shards."""
        mod = _load_script()
        mod._WORKER_DENOISER_MODEL = None
        trans_path, _, krn = self._make_minimal_transition_pair(
            tmp_path, with_denoiser=False,
        )
        # Doctor the summary to point at a non-existent denoiser file.
        import json
        summary_path = (
            trans_path[:-len(".weights.h5")] + "_summary.json"
        )
        with open(summary_path) as f:
            s = json.load(f)
        s["denoiser"] = {
            "weights_path":   "/n/nope/hst_denoiser.weights.h5",
            "channels":       4,
            "n_inner_layers": 1,
            "kernel_size":    3,
        }
        with open(summary_path, "w") as f:
            json.dump(s, f)
        with pytest.raises(RuntimeError, match="denoiser"):
            mod._init_worker(
                krn, trans_path, 12, 3, 64, DEFAULT_TEST_RATIOS,
            )


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
        _init_worker_with_kernel(mod, krn, image_size=64)
        task = (0, 150.1, 2.3, tile, 200, 1)
        catalog_idx, hr, lr = mod._process_one_galaxy(task)
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
        _init_worker_with_kernel(mod, krn, image_size=64)
        # 10 deg off — way outside the tile footprint.
        task = (0, 160.0, 12.0, tile, 200, 42)
        assert mod._process_one_galaxy(task) is None

    def test_missing_file_returns_none(self, tmp_path):
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        _init_worker_with_kernel(mod, krn, image_size=64)
        task = (0, 150.1, 2.3,
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
        _init_worker_with_kernel(mod, krn, image_size=2000)
        task = (0, 150.1, 2.3, tile, 50, 42)
        assert mod._process_one_galaxy(task) is None

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
        _init_worker_with_kernel(mod, krn, image_size=64)
        task_a = (0, 150.1, 2.3, tile, 200, 12345)
        task_b = (0, 150.1, 2.3, tile, 200, 12345)
        _, hr_a, lr_a = mod._process_one_galaxy(task_a)
        _, hr_b, lr_b = mod._process_one_galaxy(task_b)
        np.testing.assert_array_equal(hr_a, hr_b)
        np.testing.assert_array_equal(lr_a, lr_b)

    def test_different_seed_different_lr(self, tmp_path):
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(tmp_path, side_pix=400)
        _init_worker_with_kernel(mod, krn, image_size=64)
        t1 = (0, 150.1, 2.3, tile, 200, 12345)
        t2 = (0, 150.1, 2.3, tile, 200, 99999)
        _, hr1, lr1 = mod._process_one_galaxy(t1)
        _, hr2, lr2 = mod._process_one_galaxy(t2)
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

    These tests pin the three properties that broke under the old
    chain (``_broadcast_hst_to_4bands``):

      1. The HLSP→HR resample preserves *total flux*, not just per-pixel
         surface brightness (the area-correction multiplier).
      2. A single source's pixel value scales correctly into Euclid VIS
         electrons via the ZP and stack-time formula.
      3. **Multi-source brightness ratios are preserved** — a cutout
         with two galaxies of brightness ratio ``r`` produces an HR/LR
         with the same ratio, regardless of how much sky surrounds them.
         The old chain allocated one catalog row's electron budget
         across the whole cube; that collapsed any multi-source signal.
    """

    def test_resample_preserves_total_flux(self, tmp_path):
        """``_resample_hlsp_to_hr`` integrates over the larger HR pixel
        instead of just interpolating surface brightness.

        Without the area-correction multiplier, zooming down 0.03″ →
        0.05″ would drop ~36 % of the input flux on the floor (scipy's
        cubic-spline zoom returns the *interpolated value*, not the
        integral over the new pixel area). The new chain fixes that.
        """
        mod = _load_script()
        # Constant input — every pixel = 1.0 e⁻/s. With area correction
        # the output per-pixel value should be (HR_area / HLSP_area) =
        # (0.05/0.03)² ≈ 2.78. Total flux integrated over the (smaller)
        # output grid then equals total flux of the (larger) input grid.
        side = 100
        hlsp = np.ones((side, side), dtype=np.float32)
        hr = mod._resample_hlsp_to_hr(hlsp, hlsp_scale=0.03, hr_scale=0.05)

        # Per-pixel value ≈ 2.78 in the interior.
        b = 5
        assert np.allclose(hr[b:-b, b:-b], (0.05 / 0.03) ** 2, atol=1e-3)

        # Total flux ratio: the output has (100 * 0.6)² ≈ 60² pixels,
        # each holding 2.78 ≈ (1/0.6)². So total ≈ 60² × 2.78 ≈ 10000.
        # Input total = 100² = 10000.
        # (Match within edge-effect loss from the zoom near the borders.)
        input_total  = float(hlsp.sum())
        output_total = float(hr.sum())
        assert output_total == pytest.approx(input_total, rel=0.05)

    def test_hst_to_euclid_hr_cube_vis_scaling(self, tmp_path):
        """A constant HST rate maps to Euclid VIS electrons via the
        documented ``rate_ratio × t_total`` formula.

        Pins the literal photometric constant so a future config tweak
        (different ZP, different stack time) is caught immediately.
        """
        from euclid_polish.config import Config

        mod = _load_script()
        # Per-HR-pixel rate (e⁻/s after area correction).
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

        # NISP bands = VIS × ratio (per pixel).
        for k in (1, 2, 3):
            assert cube[..., k] == pytest.approx(
                expected_vis * ratios[k], rel=1e-5,
            )

    def test_hst_to_euclid_hr_cube_handles_signed_residual(self):
        """Symmetric residual noise (positive + negative pixels) must
        pass through cleanly — the new chain explicitly drops the
        old ``clip(..., 0)`` so noise averages to zero over the cube.
        Negative pixels are allowed in the HR cube; the forward model's
        Poisson step clips them at zero internally before the draw.
        """
        mod = _load_script()
        # Mostly-positive with some negative residual.
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

    def test_multi_source_brightness_ratio_preserved(self, tmp_path):
        """The whole point of switching to HST-native photometry.

        A cutout with TWO sources of known brightness ratio ``r`` (here
        4:1) must produce an HR cube where the per-source peaks still
        sit in that ratio — regardless of how much sky surrounds them.

        The OLD chain (``_broadcast_hst_to_4bands``) normalised the
        whole cube to unit total flux then multiplied by *one* catalog
        row's electron budget, collapsing any multi-source signal: each
        source ended up with roughly its FRACTION of catalog flux, with
        absolute scale dictated by sky residual rather than its real
        HST photometry. This test would have failed on the old chain.
        """
        from astropy.io import fits
        from astropy.wcs import WCS

        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)

        # Build a synthetic HLSP-style tile with TWO Gaussian sources of
        # known peak ratio 4:1.
        side = 400
        sigma = 6.0
        yy, xx = np.mgrid[:side, :side]
        # Source A: peak 400 at (150, 150). Source B: peak 100 at (250, 250).
        ga = 400.0 * np.exp(
            -((xx - 150) ** 2 + (yy - 150) ** 2) / (2.0 * sigma ** 2)
        )
        gb = 100.0 * np.exp(
            -((xx - 250) ** 2 + (yy - 250) ** 2) / (2.0 * sigma ** 2)
        )
        data = (ga + gb).astype(np.float32)

        w = WCS(naxis=2)
        w.wcs.crpix = [side / 2 + 0.5, side / 2 + 0.5]
        w.wcs.cdelt = [-0.03 / 3600.0, 0.03 / 3600.0]
        w.wcs.crval = [150.1, 2.3]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        tile = str(tmp_path / "two_source.fits")
        fits.PrimaryHDU(data, header=w.to_header()).writeto(tile, overwrite=True)

        _init_worker_with_kernel(mod, krn, image_size=128)
        task = (0, 150.1, 2.3, tile, 300, 42)
        result = mod._process_one_galaxy(task)
        assert result is not None
        _, hr, _ = result

        # VIS channel; sources sit at HR positions roughly (60, 60) and
        # (100, 100) after the 0.03→0.05 resample + central crop. Find
        # them by argmax in two halves of the image.
        vis = hr[..., 0]
        upper_left  = vis[:64, :64]
        lower_right = vis[64:, 64:]
        peak_a = float(upper_left.max())
        peak_b = float(lower_right.max())
        ratio  = peak_a / peak_b if peak_b > 0 else 0
        # Expect 4.0 to within a few % — small slack for the
        # interpolation rounding + symmetric residual noise. Old
        # chain would have produced a ratio determined by the cube
        # SUM (a tiny number), not the per-source peaks.
        assert 3.5 < ratio < 4.5, (
            f"multi-source brightness ratio collapsed: peaks "
            f"A={peak_a:.1f}, B={peak_b:.1f}, ratio={ratio:.2f} "
            "(expected ~4.0)"
        )

    def test_nisp_channels_use_typical_ratios(self, tmp_path):
        """NISP[k] = VIS × typical_band_ratios[k] per pixel; verify
        the global colour scale is the only difference between VIS
        and each NISP channel in the HR cube."""
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        tile = _make_synthetic_tile(tmp_path, side_pix=400)
        ratios = np.array([1.0, 0.15, 0.25, 0.20], dtype=np.float32)
        mod._init_worker(
            krn, "", 12, 3, 64, ratios,
        )
        task = (0, 150.1, 2.3, tile, 200, 1)
        _, hr, _ = mod._process_one_galaxy(task)

        vis = hr[..., 0]
        for k in (1, 2, 3):
            band_hr = hr[..., k]
            # Per-pixel ratio should equal the global scale exactly,
            # everywhere VIS is non-zero (mostly the centre).
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
            initargs=(krn, "", 12, 3, 64, DEFAULT_TEST_RATIOS),
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
            initargs=(krn, "", 12, 3, 32, DEFAULT_TEST_RATIOS),
        ) as pool:
            results = list(pool.map(
                mod._process_one_galaxy, ok_tasks + bad_tasks,
            ))
        successes = [r for r in results if r is not None]
        failures  = [r for r in results if r is None]
        assert len(successes) == 3
        assert len(failures)  == 2


# ---------------------------------------------------------------------------
# Transition-model selection (CNN ↔ analytic kernel fallback)
# ---------------------------------------------------------------------------

class TestTransitionModelSelection:
    """``_init_worker`` picks the trained CNN over the analytic kernel
    when both are available, and falls back when the CNN file is
    missing. We exercise both branches without needing a fully-trained
    CNN — a freshly-instantiated, never-saved-to-disk model is enough
    to verify the dispatching logic.
    """

    def _save_untrained_cnn(self, tmp_path) -> str:
        """Write a fresh transition-model weights file to disk.

        At init the model is ~identity (residual init); good enough for
        a pipeline-wiring test, since we only assert which branch ran,
        not the photometric correctness of the output.
        """
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, save_model_weights,
        )
        model = HSTtoEuclidTransition()
        path = str(tmp_path / "transition_model.weights.h5")
        save_model_weights(model, path)
        return path

    def test_init_worker_loads_cnn_when_available(self, tmp_path):
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        cnn = self._save_untrained_cnn(tmp_path)
        mod._init_worker(krn, cnn, 12, 3, 64, DEFAULT_TEST_RATIOS)
        # CNN branch: transition model populated, kernel left as None.
        assert mod._WORKER_TRANSITION_MODEL is not None
        assert mod._WORKER_KERNEL is None

    def test_init_worker_falls_back_to_kernel_when_cnn_missing(self, tmp_path):
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        missing_cnn = str(tmp_path / "does_not_exist.weights.h5")
        mod._init_worker(krn, missing_cnn, 12, 3, 64, DEFAULT_TEST_RATIOS)
        # Kernel branch: analytic kernel populated, CNN left as None.
        assert mod._WORKER_KERNEL is not None
        assert mod._WORKER_TRANSITION_MODEL is None

    def test_init_worker_falls_back_when_cnn_path_blank(self, tmp_path):
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        mod._init_worker(krn, "", 12, 3, 64, DEFAULT_TEST_RATIOS)
        assert mod._WORKER_KERNEL is not None
        assert mod._WORKER_TRANSITION_MODEL is None

    def test_process_galaxy_with_cnn_branch(self, tmp_path):
        """End-to-end through the CNN branch: just-built A_θ ≈ identity,
        so the LR cube should be finite + correctly-shaped (we don't
        assert on numeric closeness to the kernel branch — that's a
        training-quality test, not a wiring test)."""
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        cnn = self._save_untrained_cnn(tmp_path)
        tile = _make_synthetic_tile(tmp_path, side_pix=400)
        mod._init_worker(krn, cnn, 12, 3, 64, DEFAULT_TEST_RATIOS)

        task = (0, 150.1, 2.3, tile, 200, 42)
        result = mod._process_one_galaxy(task)
        assert result is not None
        catalog_idx, hr_cube, lr_cube = result
        assert hr_cube.shape == (64, 64, 4)
        assert lr_cube.shape == (32, 32, 4)
        assert np.isfinite(hr_cube).all()
        assert np.isfinite(lr_cube).all()

    def _save_cnn_with_baseline(self, tmp_path, kernel_fits_path):
        """Write a transition-model weights file from a model that was
        constructed with the analytic baseline kernel pinned. Also
        emits the summary JSON the worker reads to reconstruct the
        same model shape.
        """
        import json as _json
        from euclid_polish.training.transition_model import (
            HSTtoEuclidTransition, save_model_weights,
        )
        # Load the same FITS the worker will see at load time.
        from astropy.io import fits as _fits
        with _fits.open(kernel_fits_path) as hdul:
            baseline_kernel = np.asarray(hdul[0].data, dtype=np.float32)
        model = HSTtoEuclidTransition(
            channels=4, n_inner_layers=1, kernel_size=3,
            baseline_kernel=baseline_kernel,
        )
        weights_path = str(tmp_path / "transition_model.weights.h5")
        save_model_weights(model, weights_path)
        # Drop the summary JSON the worker reads.
        summary_path = str(tmp_path / "transition_model_summary.json")
        with open(summary_path, "w") as f:
            _json.dump({
                "channels":       4,
                "n_inner_layers": 1,
                "kernel_size":    3,
                "analytic_baseline_kernel": kernel_fits_path,
                "baseline_crop":  0,
            }, f)
        return weights_path

    def test_init_worker_reconstructs_model_with_baseline(self, tmp_path):
        """REGRESSION — when the trained model was saved with an
        analytic baseline pinned as a non-trainable layer, the worker
        MUST reconstruct it with the same baseline before loading the
        weights. Reading channels/n_inner_layers from the CLI alone
        gives a model with 0 top-level variables, while the weights
        file has 1 (the baseline kernel) — Keras then raises
        ``expected 0 variables, but received 1`` and the whole
        ProcessPoolExecutor dies.

        Verify that the worker reads the summary JSON, loads the
        baseline FITS, and instantiates the model with both — so
        the weights file loads without error.
        """
        mod = _load_script()
        krn = _make_synthetic_kernel(tmp_path)
        cnn = self._save_cnn_with_baseline(tmp_path, krn)
        # Call with CLI defaults that DON'T match the saved
        # hyperparameters — the worker should ignore the CLI values
        # in favor of the summary's.
        mod._init_worker(krn, cnn, 12, 3, 64, DEFAULT_TEST_RATIOS)
        assert mod._WORKER_TRANSITION_MODEL is not None
        # The reconstructed model must have a baseline kernel.
        assert mod._WORKER_TRANSITION_MODEL._has_baseline is True
        assert mod._WORKER_KERNEL is None

    def test_init_worker_fails_loudly_when_baseline_missing(self, tmp_path):
        """If the summary says a baseline was used but the FITS isn't
        findable, the worker must raise a clear error rather than
        silently constructing the wrong-shape model (which would then
        explode inside ``load_weights`` with the cryptic
        ``expected 0 variables, received 1``)."""
        import json as _json
        import pytest as _pt
        # Save a model WITH baseline using a real kernel...
        krn = _make_synthetic_kernel(tmp_path)
        weights_path = self._save_cnn_with_baseline(tmp_path, krn)
        # ...then doctor the summary to point at a nonexistent FITS.
        summary_path = str(tmp_path / "transition_model_summary.json")
        with open(summary_path) as f:
            s = _json.load(f)
        s["analytic_baseline_kernel"] = "/n/nowhere/does_not_exist.fits"
        with open(summary_path, "w") as f:
            _json.dump(s, f)
        # Also remove the matching local kernel so the candidate-search
        # in _init_worker has nothing to fall back to.
        missing_krn = str(tmp_path / "missing_kernel.fits")
        mod = _load_script()
        with _pt.raises(RuntimeError, match="baseline.*kernel"):
            mod._init_worker(missing_krn, weights_path, 12, 3, 64,
                             DEFAULT_TEST_RATIOS)
