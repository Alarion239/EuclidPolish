"""Tests for the transition-model pair-generation script.

Covers the standalone helpers (PSF resampling, odd-square cropping,
clean-scene streaming) but not the full ``main()`` — running main() end
to end requires real synthetic clean records and Euclid VIS PSF FITS,
both of which live outside the test working tree.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))


def _load_script():
    import fasrc_generate_transition_pairs as mod
    return mod


# ---------------------------------------------------------------------------
# _resample_to_hr_grid
# ---------------------------------------------------------------------------

class TestResampleToHrGrid:

    def test_noop_when_scales_match(self):
        mod = _load_script()
        # Build a centred Gaussian at the HR scale already.
        side = 31
        sigma = 2.0
        y, x = np.mgrid[:side, :side]
        cy = cx = (side - 1) / 2.0
        g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
        g = (g / g.sum()).astype(np.float32)
        from euclid_polish.config import Config
        out = mod._resample_to_hr_grid(g, src_scale=Config.DEFAULT_PIXEL_SCALE)
        np.testing.assert_allclose(out, g, atol=1e-6)

    def test_renormalises_to_unit_flux(self):
        mod = _load_script()
        side = 41
        sigma = 3.0
        y, x = np.mgrid[:side, :side]
        cy = cx = (side - 1) / 2.0
        g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
        g = (g / g.sum()).astype(np.float32)
        # 0.03 → 0.05 = HLSP → HR. The zoom + renormalise should keep
        # sum=1 to within tiny float roundoff.
        out = mod._resample_to_hr_grid(g, src_scale=0.03)
        assert abs(out.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# _crop_to_odd_square
# ---------------------------------------------------------------------------

class TestCropToOddSquare:

    def test_returns_odd_square(self):
        mod = _load_script()
        arr = np.zeros((100, 100), dtype=np.float32)
        arr[50, 50] = 1.0
        out = mod._crop_to_odd_square(arr, side=21)
        assert out.shape == (21, 21)

    def test_pads_when_smaller(self):
        mod = _load_script()
        arr = np.zeros((15, 15), dtype=np.float32)
        arr[7, 7] = 1.0
        out = mod._crop_to_odd_square(arr, side=21)
        assert out.shape == (21, 21)
        # The single hot pixel survived; check it's roughly centred.
        hot = np.argwhere(out > 0.5)
        assert hot.shape[0] == 1
        cy, cx = hot[0]
        assert abs(cy - 10) <= 1
        assert abs(cx - 10) <= 1

    def test_even_side_rejected(self):
        mod = _load_script()
        with pytest.raises(ValueError, match="odd"):
            mod._crop_to_odd_square(np.zeros((10, 10)), side=8)


# ---------------------------------------------------------------------------
# _convolve_pair — produces (HST-blurred, Euclid-blurred) at matching size
# ---------------------------------------------------------------------------

class TestConvolvePair:

    def test_input_at_hr_target_at_lr_default_rebin(self):
        """Default rebin_factor=2: input keeps scene's shape; target
        is sum-rebinned to half the side per axis."""
        mod = _load_script()
        scene = np.zeros((64, 64), dtype=np.float32)
        scene[32, 32] = 100.0
        psf = np.ones((9, 9), dtype=np.float32) / 81.0
        inp, tgt = mod._convolve_pair(scene, psf, psf)
        assert inp.shape == (64, 64)
        assert tgt.shape == (32, 32)    # rebin_factor=2

    def test_shapes_match_with_rebin_one(self):
        """rebin_factor=1 keeps target at HR (legacy mode)."""
        mod = _load_script()
        scene = np.zeros((64, 64), dtype=np.float32)
        scene[32, 32] = 100.0
        psf = np.ones((9, 9), dtype=np.float32) / 81.0
        inp, tgt = mod._convolve_pair(scene, psf, psf, rebin_factor=1)
        assert inp.shape == scene.shape
        assert tgt.shape == scene.shape

    def test_flux_conserved_through_rebin(self):
        """Both convolutions preserve total flux (unit-flux PSFs), and
        sum-rebin is photometric, so input.sum() ≈ target.sum() ≈
        scene.sum()."""
        mod = _load_script()
        scene = np.zeros((64, 64), dtype=np.float32)
        scene[16, 16] = 50.0
        scene[48, 48] = 70.0
        psf_hst = np.ones((5, 5), dtype=np.float32) / 25.0
        psf_eu  = np.zeros((9, 9), dtype=np.float32)
        psf_eu[4, 4] = 1.0
        psf_eu /= psf_eu.sum()
        inp, tgt = mod._convolve_pair(scene, psf_hst, psf_eu)
        np.testing.assert_allclose(inp.sum(), scene.sum(), rtol=1e-4)
        np.testing.assert_allclose(tgt.sum(), scene.sum(), rtol=1e-4)

    def test_target_matches_explicit_sum_rebin(self):
        """``target == sum_rebin(scene ⊛ PSF_Euclid, rebin)`` — the
        helper must not rearrange terms."""
        from euclid_polish.training.transition_augmentations import (
            sum_rebin_2d,
        )
        from scipy import signal as scipy_signal
        mod = _load_script()
        scene = np.zeros((64, 64), dtype=np.float32)
        scene[24:28, 24:28] = 10.0    # extended source
        psf_hst = np.ones((5, 5), dtype=np.float32) / 25.0
        psf_eu  = np.zeros((9, 9), dtype=np.float32)
        psf_eu[4, 4] = 1.0
        psf_eu /= psf_eu.sum()
        inp, tgt = mod._convolve_pair(scene, psf_hst, psf_eu, rebin_factor=2)
        expected_tgt = sum_rebin_2d(
            scipy_signal.fftconvolve(
                scene, psf_eu, mode="same",
            ).astype(np.float32),
            2,
        )
        np.testing.assert_allclose(tgt, expected_tgt, atol=1e-5)


# ---------------------------------------------------------------------------
# _stream_clean_vis_scenes — handshake with the synthetic TFRecord schema
# ---------------------------------------------------------------------------

class TestStreamCleanVisScenes:
    """Verifies the stream helper reads the right schema and centre-
    crops. We synthesise a minimal multiband TFRecord on the fly so the
    test doesn't depend on the synthetic pipeline being run first."""

    def _write_minimal_clean_record(self, tmp_path, name, n_scenes, side):
        """Write a tiny 4-channel clean TFRecord with N scenes."""
        from euclid_polish.config import Config
        from euclid_polish.sky.tfrecord import open_multiband_writer
        from euclid_polish.sky.types import MultiBandSkyImage
        rng = np.random.default_rng(0)
        with open_multiband_writer(name, records_dir=str(tmp_path)) as w:
            for i in range(n_scenes):
                data = rng.uniform(0, 100, size=(side, side, 4)).astype(np.float32)
                img = MultiBandSkyImage(
                    data=data,
                    pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                    band_names=Config.LR_INPUT_BAND_NAMES,
                    is_clean=True,
                )
                w.write(img, index=i)

    def test_yields_correct_shape_and_count(self, tmp_path):
        mod = _load_script()
        self._write_minimal_clean_record(
            tmp_path, "clean_validate", n_scenes=5, side=64,
        )
        out = list(mod._stream_clean_vis_scenes(
            str(tmp_path), "validate", n_max=10, crop=32,
        ))
        assert len(out) == 5     # source had only 5 scenes
        for _, scene in out:
            assert scene.shape == (32, 32)
            assert scene.dtype == np.float32

    def test_caps_at_n_max(self, tmp_path):
        mod = _load_script()
        self._write_minimal_clean_record(
            tmp_path, "clean_train", n_scenes=20, side=64,
        )
        out = list(mod._stream_clean_vis_scenes(
            str(tmp_path), "train", n_max=7, crop=32,
        ))
        assert len(out) == 7

    def test_skips_scenes_smaller_than_crop(self, tmp_path):
        mod = _load_script()
        self._write_minimal_clean_record(
            tmp_path, "clean_train", n_scenes=3, side=32,
        )
        # crop > source side → all scenes get skipped.
        out = list(mod._stream_clean_vis_scenes(
            str(tmp_path), "train", n_max=10, crop=64,
        ))
        assert out == []

    def test_missing_file_raises(self, tmp_path):
        mod = _load_script()
        with pytest.raises(FileNotFoundError):
            list(mod._stream_clean_vis_scenes(
                str(tmp_path), "validate", n_max=1, crop=32,
            ))


# ---------------------------------------------------------------------------
# main() pre-flight regression — refuses to write empty shards
# ---------------------------------------------------------------------------

class TestPreflightFailsLoudly:
    """REGRESSION — the pair-gen script used to silently write 0 pairs
    when ``--crop-size`` was larger than the synthetic scene side. The
    empty 0-byte TFRecord files then crashed the trainer downstream
    with an opaque ``input_train.tfrecord is empty`` error.

    main() now performs a one-record pre-flight probe on
    clean_train.tfrecord BEFORE opening any output writers. If the
    crop is too big, it prints a clear diagnostic and exits with code
    2 — no empty files left behind, no SLURM time wasted on the
    downstream training job that would crash on them.
    """

    def _write_minimal_clean_record(self, tmp_path, name, n_scenes, side):
        from euclid_polish.config import Config
        from euclid_polish.sky.tfrecord import open_multiband_writer
        from euclid_polish.sky.types import MultiBandSkyImage
        rng = np.random.default_rng(0)
        with open_multiband_writer(name, records_dir=str(tmp_path)) as w:
            for i in range(n_scenes):
                data = rng.uniform(0, 100, size=(side, side, 4)).astype(np.float32)
                img = MultiBandSkyImage(
                    data=data,
                    pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                    band_names=Config.LR_INPUT_BAND_NAMES,
                    is_clean=True,
                )
                w.write(img, index=i)

    def test_main_succeeds_end_to_end(self, tmp_path, monkeypatch):
        """REGRESSION — the success path of main() must complete
        without UnboundLocalError or any other Python-level bug.

        We just hit a real-world ``UnboundLocalError: cannot access
        local variable 'Config'`` because a stray
        ``from euclid_polish.config import Config`` inside an
        error-only branch made ``Config`` function-local everywhere
        in main() via Python's static-scope rule. The pre-flight and
        all other tests in this class exercise the FAILURE paths;
        none actually ran main() to the successful completion of
        the writer loop. This test does — with tiny inputs so it
        finishes in seconds — and would have caught that bug before
        the failing job hit SLURM."""
        mod = _load_script()
        # Source clean shards: scenes large enough to satisfy a small crop.
        synth_dir = tmp_path / "synth"
        synth_dir.mkdir()
        self._write_minimal_clean_record(
            synth_dir, "clean_train", n_scenes=3, side=64,
        )
        self._write_minimal_clean_record(
            synth_dir, "clean_validate", n_scenes=1, side=64,
        )

        # Fake HST + Euclid PSF FITS so the PSF-loading branch passes.
        import os as _os
        from astropy.io import fits
        hst_psf_path = tmp_path / "F814W.fits"
        y, x = np.mgrid[:31, :31]
        cy = cx = 15
        g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 3.0 ** 2))
        g = (g / g.sum()).astype(np.float32)
        hdu = fits.PrimaryHDU(g)
        hdu.header["PIXSCALE"] = 0.03
        hdu.header["FILTER"]   = "F814W"
        hdu.writeto(str(hst_psf_path), overwrite=True)
        from euclid_polish.config import Config
        _os.makedirs(Config.EUCLID_PSF_DIR, exist_ok=True)
        euclid_psf_path = _os.path.join(
            Config.EUCLID_PSF_DIR, "euclid_psf_VIS.fits",
        )
        g2 = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 5.0 ** 2))
        g2 = (g2 / g2.sum()).astype(np.float32)
        hdu2 = fits.PrimaryHDU(g2)
        hdu2.header["PXSCALE"] = 0.05
        hdu2.writeto(euclid_psf_path, overwrite=True)

        out_dir = tmp_path / "out"
        argv = [
            "fasrc_generate_transition_pairs.py",
            "--synthetic-records-dir", str(synth_dir),
            "--hst-psf",       str(hst_psf_path),
            "--output-dir",    str(out_dir),
            "--n-train",       "3",
            "--n-valid",       "1",
            "--crop-size",     "32",     # fits inside 64² scenes
            "--rebin-factor",  "2",
        ]
        monkeypatch.setattr("sys.argv", argv)
        rc = mod.main()
        # 0 == success. Any non-zero would mean the failure-branch was
        # taken (UnboundLocalError, missing files, etc.).
        assert rc == 0, f"main() returned {rc} on the happy path"

        # Sanity-check the on-disk artifacts: 4 non-empty TFRecords,
        # JSON summary present.
        assert (out_dir / "input_train.tfrecord").stat().st_size > 0
        assert (out_dir / "target_train.tfrecord").stat().st_size > 0
        assert (out_dir / "input_validate.tfrecord").stat().st_size > 0
        assert (out_dir / "target_validate.tfrecord").stat().st_size > 0
        assert (out_dir / "generation_summary.json").exists()

    def test_main_refuses_oversized_crop(self, tmp_path, monkeypatch):
        """If --crop-size > scene side, main() must return non-zero AND
        leave the output dir empty (no 0-byte shards behind)."""
        mod = _load_script()
        # Seed a synthetic clean_train of 64×64 scenes.
        synth_dir = tmp_path / "synth"
        synth_dir.mkdir()
        self._write_minimal_clean_record(
            synth_dir, "clean_train", n_scenes=3, side=64,
        )
        # Fake HST + Euclid PSF FITS so the PSF-loading branch passes.
        import os as _os
        from astropy.io import fits
        hst_psf_path = tmp_path / "F814W.fits"
        # Small Gaussian as a stand-in.
        y, x = np.mgrid[:31, :31]
        cy = cx = 15
        g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 3.0 ** 2))
        g = (g / g.sum()).astype(np.float32)
        hdu = fits.PrimaryHDU(g)
        hdu.header["PIXSCALE"] = 0.03
        hdu.header["FILTER"]   = "F814W"
        hdu.writeto(str(hst_psf_path), overwrite=True)
        # Stand-in for euclid_psf_VIS.fits at the Config-derived path.
        from euclid_polish.config import Config
        _os.makedirs(Config.EUCLID_PSF_DIR, exist_ok=True)
        euclid_psf_path = _os.path.join(
            Config.EUCLID_PSF_DIR, "euclid_psf_VIS.fits",
        )
        g2 = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 5.0 ** 2))
        g2 = (g2 / g2.sum()).astype(np.float32)
        hdu2 = fits.PrimaryHDU(g2)
        hdu2.header["PXSCALE"] = 0.05
        hdu2.writeto(euclid_psf_path, overwrite=True)

        out_dir = tmp_path / "out"
        # Invoke main with argv: crop=128 (> source 64), rebin=2.
        argv = [
            "fasrc_generate_transition_pairs.py",
            "--synthetic-records-dir", str(synth_dir),
            "--hst-psf",       str(hst_psf_path),
            "--output-dir",    str(out_dir),
            "--n-train",       "3",
            "--n-valid",       "1",
            "--crop-size",     "128",
            "--rebin-factor",  "2",
        ]
        monkeypatch.setattr("sys.argv", argv)
        rc = mod.main()
        assert rc != 0, "main() must return non-zero when crop > scene"
        # No output TFRecord shards left behind.
        if out_dir.exists():
            stragglers = sorted(out_dir.glob("*.tfrecord"))
            assert stragglers == [], (
                f"main() must not leave empty TFRecords on failure; "
                f"found: {stragglers}"
            )
